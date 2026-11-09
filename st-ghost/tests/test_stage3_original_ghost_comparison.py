"""CPU regressions for retry timing, distinct selection and in-task resume."""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_stage3_original_ghost_comparison as runner


class ScriptedVictim:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)

    def projected_probability(self, source, tokens, prompt):
        # Differentiable probability, with scripted values controlling gates.
        return torch.tensor(next(self.probabilities)) + tokens.float().sum() * 0


class FakeDataset:
    def __init__(self, path):
        self.path = path

    def __getitem__(self, image_id):
        return Image.open(self.path).convert("RGB"), str(self.path)


class OriginalGhostComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.source_path = self.output / "source.png"
        Image.new("RGB", (16, 16), "gray").save(self.source_path)
        self.item = {"spec": "dog_bicycle", "image_id": 1,
                     "source_path": str(self.source_path), "source_sha256": runner.digest(self.source_path)}
        self.cfg = {"target_specs": [{"name": "dog_bicycle", "objects": ["dog", "bicycle"]}],
                    "num_steps": 10, "num_generations": 4, "learning_rate": 0.1,
                    "yes_threshold": 0.8, "lambda_clip": 15.0, "lambda_reg": 10.0,
                    "num_images_per_spec": 1}
        self.contrasts = (torch.ones(1, 2), torch.ones(1, 2))

    def runtime(self, probabilities):
        pipe = SimpleNamespace(set_progress_bar_config=lambda **kwargs: None)
        return SimpleNamespace(
            device=torch.device("cpu"), victim=ScriptedVictim(probabilities),
            clip=SimpleNamespace(images=lambda images: torch.tensor([[0.2, 0.3]])),
            projector=SimpleNamespace(model=lambda embedding: embedding),
            load_diffusion=lambda: None, pipe=pipe, diffusion_preprocess=None)

    def candidate(self, valid=False, trigger_yes=False):
        # Held-out Yes by itself must never stop the original trigger protocol.
        return {"generation_index": 0, "target_yes_probability": 0.9 if trigger_yes else 0.1,
                "target_query_yes": int(trigger_yes), "valid_attack_success": int(valid),
                "heldout_target_query_yes": 1, "heldout_valid_attack_success": 1,
                "heldout_target_yes_probability": 0.9, "absence_all": 1,
                "target_answer_unparsed": 0}

    @contextlib.contextmanager
    def runtime_patches(self):
        with patch.object(runner.exp1, "get_vae_features", return_value=torch.zeros(1, 4, 2, 2)), \
             patch.object(torch.cuda, "get_rng_state_all", return_value=[]), \
             patch.object(torch.cuda, "set_rng_state_all"), \
             patch.object(random_seed_owner(), "seed", side_effect=AssertionError("No fixed seed allowed")), \
             patch.object(torch, "manual_seed", side_effect=AssertionError("No fixed seed allowed")), \
             contextlib.redirect_stdout(io.StringIO()):
            yield

    def run_task(self, runtime):
        return runner.run_task(runtime, self.item, self.cfg, self.output,
                               "signature", self.contrasts)

    def test_source_image_loader_preserves_coco_resize_contract(self):
        image = runner.load_source_image(self.source_path)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (336, 336))
        image.close()

    def test_default_config_is_one_multi_object_run_with_inherited_hyperparameters(self):
        cfg = runner.load_config(runner.DEFAULT_CONFIG)
        base = runner.exp1.load_config(Path(cfg["base_config"]))
        self.assertNotIn("seeds", cfg)
        self.assertNotIn("seed", cfg)
        self.assertIsNone(cfg["random_seed"])
        self.assertEqual(cfg["conditions"], ["Multi-O"])
        self.assertEqual(cfg["repeats_per_image"], 1)
        self.assertEqual(cfg["num_images_per_spec"] * len(cfg["target_specs"]), 2000)
        for key in ("num_steps", "learning_rate", "yes_threshold", "lambda_clip", "lambda_reg",
                    "num_generations", "guidance_scale", "diffusion_t", "num_inference_steps", "owl_threshold"):
            self.assertEqual(cfg[key], base[key], key)

    def test_distinct_assignment_reallocates_an_overlap_instead_of_starving_a_group(self):
        pools = {"A": [{"image_id": 1}, {"image_id": 2}], "B": [{"image_id": 1}]}
        owners, assigned = {}, {"A": set(), "B": set()}
        self.assertTrue(runner.augment_assignment("A", pools, owners, assigned))
        self.assertTrue(runner.augment_assignment("B", pools, owners, assigned))
        self.assertEqual(assigned, {"A": {2}, "B": {1}})

    def test_insufficient_distinct_capacity_fails_before_screening(self):
        pools = {"A": {"rows": [{"image_id": 1}]}, "B": {"rows": [{"image_id": 1}]}}
        with self.assertRaises(ValueError):
            runner.check_distinct_capacity(pools, 1)

    def test_failed_candidate_continues_optimization_then_stops_on_valid_success(self):
        runtime = self.runtime([0.7, 0.9, 0.9, 0.92, 0.91, 0.93])
        with self.runtime_patches(), patch.object(runner, "draw_candidate", side_effect=[
                self.candidate(), self.candidate(valid=True, trigger_yes=True)]) as draw:
            payload = self.run_task(runtime)
        self.assertEqual(draw.call_count, 2)
        self.assertEqual(payload["result"]["optimization_updates"], 3)
        self.assertEqual(payload["result"]["stop_reason"], "valid_attack_success")
        self.assertEqual([r["generation_triggered"] for r in payload["traces"]], [0, 1, 1])
        self.assertEqual([call.args[8] for call in draw.call_args_list], [0, 1])
        self.assertEqual([call.args[7] for call in draw.call_args_list], [1, 2])

    def test_four_total_failed_generations_stop_optimization(self):
        runtime = self.runtime([0.9, 0.95] * 10)
        with self.runtime_patches(), patch.object(runner, "draw_candidate", side_effect=[self.candidate()] * 4) as draw:
            payload = self.run_task(runtime)
        self.assertEqual(draw.call_count, 4)
        self.assertEqual(payload["result"]["optimization_updates"], 4)
        self.assertEqual(payload["result"]["num_generations_evaluated"], 4)
        self.assertEqual(payload["result"]["stop_reason"], "generation_budget_exhausted")
        self.assertEqual(payload["result"]["valid_thr_success"], 0)

    def test_post_update_threshold_alone_does_not_trigger_the_reference_qwen_gate(self):
        runtime = self.runtime([0.5, 0.99] * 10)
        with self.runtime_patches(), patch.object(runner, "draw_candidate") as draw:
            payload = self.run_task(runtime)
        draw.assert_not_called()
        self.assertEqual(payload["result"]["embedding_success"], 0)
        self.assertEqual(payload["result"]["post_update_threshold_ever_reached"], 1)
        self.assertEqual(payload["result"]["optimization_updates"], 10)

    def test_exact_threshold_does_not_generate(self):
        self.cfg["yes_threshold"] = float(torch.tensor(0.8))
        runtime = self.runtime([0.8, 0.9] * 10)
        with self.runtime_patches(), patch.object(runner, "draw_candidate") as draw:
            payload = self.run_task(runtime)
        draw.assert_not_called()
        self.assertEqual(payload["result"]["embedding_success"], 0)

    def test_resume_keeps_adamw_progress_and_generation_budget(self):
        runtime = self.runtime([0.9, 0.95] * 10)
        original_save = runner.save_task_state
        saved_states = []

        def save_then_interrupt(*args, **kwargs):
            original_save(*args, **kwargs)
            if len(args[7]) == 2:
                raise KeyboardInterrupt("Simulated interruption after two saved steps")

        def observe_save(*args, **kwargs):
            original_save(*args, **kwargs)
            saved_states.append(torch.load(args[0], weights_only=True))

        with self.runtime_patches(), patch.object(runner, "draw_candidate", side_effect=[self.candidate()] * 4) as draw:
            with patch.object(runner, "save_task_state", side_effect=save_then_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_task(runtime)
            with patch.object(runner, "save_task_state", side_effect=observe_save):
                payload = self.run_task(runtime)
        self.assertEqual(draw.call_count, 4)
        self.assertEqual(payload["result"]["optimization_updates"], 4)
        self.assertEqual(len(payload["generations"]), 4)
        self.assertEqual(float(saved_states[-1]["optimizer"]["state"][0]["step"]), 4)
        self.assertEqual(len(saved_states[-1]["trace"]), 4)

    def test_trigger_and_heldout_outcomes_remain_separate(self):
        diagnostic = {"target_question": "heldout", "target_answer": "Yes",
                      "target_decision": True, "target_yes_probability": 0.95,
                      "target_query_yes": 1, "valid_attack_success": 1, "absence_all": 1}
        runtime = SimpleNamespace(victim=SimpleNamespace(answer=lambda *args: ("No", False, 0.1)))
        with patch.object(runner.exp1, "evaluate_generated", return_value=diagnostic):
            row = runner.evaluate_candidate(runtime, None, None, "trigger")
        self.assertEqual(row["heldout_target_query_yes"], 1)
        self.assertEqual(row["target_query_yes"], 0)
        self.assertEqual(row["valid_attack_success"], 0)
        self.assertEqual(row["target_question"], "trigger")

    def test_reports_count_tasks_without_generations_in_the_success_denominator(self):
        with self.runtime_patches(), patch.object(runner, "draw_candidate") as draw:
            self.run_task(self.runtime([0.3, 0.4] * 10))
        draw.assert_not_called()
        self.item = {**self.item, "image_id": 2}
        success = {**self.candidate(valid=True, trigger_yes=True),
                   "spec": "dog_bicycle", "image_id": 2}
        with self.runtime_patches(), patch.object(runner, "draw_candidate", return_value=success):
            self.run_task(self.runtime([0.9, 0.95]))
        self.cfg["num_images_per_spec"] = 2
        summary = runner.write_reports(self.output, self.cfg, "signature")
        overall = summary["overall"]
        self.assertTrue(summary["evaluation_complete"])
        self.assertEqual(overall["tasks_completed"], 2)
        self.assertEqual(overall["embedding_success_rate"], 0.5)
        self.assertEqual(overall["task_query_yes_rate"], 0.5)
        self.assertEqual(overall["task_valid_success_rate"], 0.5)
        self.assertEqual(overall["images_generated"], 1)
        self.assertEqual(overall["generated_query_yes_rate"], 1.0)
        self.assertAlmostEqual(overall["generated_mean_yes_probability"], 0.9)
        self.assertEqual(len(runner.exp1.read_csv(self.output / "all_steps.csv")), 11)
        self.assertEqual(len(runner.exp1.read_csv(self.output / "all_generations.csv")), 1)

    def test_fp16_overflow_is_detected_before_sending_tokens_to_victim(self):
        runtime = self.runtime([0.9])
        scales = iter([1.0, 1e6])
        runtime.projector.model = lambda embedding: embedding * next(scales)
        with self.runtime_patches(), patch.object(runner, "draw_candidate") as draw:
            payload = self.run_task(runtime)
        draw.assert_not_called()
        self.assertEqual(payload["result"]["stop_reason"], "numerical_failure")
        self.assertEqual(payload["result"]["optimization_updates"], 0)
        self.assertIsNone(payload["result"]["final_embedding_yes_probability"])
        failure = payload["numerical_failure"]
        self.assertEqual(failure["phase"], "after_update")
        self.assertEqual(failure["error"], "Projector output overflows FP16")
        self.assertGreater(failure["details"]["projector_output"]["finite_absmax"], 65504)
        final = torch.load(payload["result"]["final_embedding_path"], weights_only=True)["embedding"]
        torch.testing.assert_close(final, torch.tensor([[0.2, 0.3]]))

    def test_nan_after_update_preserves_completed_steps_candidates_and_optimizer_checkpoint(self):
        runtime = self.runtime([0.9, 0.95, 0.9, float("nan")])
        candidate = {**self.candidate(), "spec": self.item["spec"], "image_id": self.item["image_id"]}
        with self.runtime_patches(), patch.object(runner, "draw_candidate", return_value=candidate) as draw:
            payload = self.run_task(runtime)
        self.assertEqual(draw.call_count, 1)
        self.assertEqual(len(payload["traces"]), 1)
        self.assertEqual(len(payload["generations"]), 1)
        result = payload["result"]
        self.assertEqual(result["stop_reason"], "numerical_failure")
        state = torch.load(result["numerical_checkpoint_path"], weights_only=True)
        final = torch.load(result["final_embedding_path"], weights_only=True)["embedding"]
        torch.testing.assert_close(final, state["embedding"])
        self.assertEqual(float(state["optimizer"]["state"][0]["step"]), 1)
        self.assertTrue(Path(result["numerical_vae_path"]).is_file())
        self.assertFalse((self.output / "in_progress/dog_bicycle__1.pt").exists())
        summary = runner.write_reports(self.output, self.cfg, "signature")
        self.assertEqual(summary["overall"]["tasks_completed"], 1)
        self.assertEqual(summary["overall"]["numerical_failure_tasks"], 1)
        self.assertEqual(summary["overall"]["task_valid_success_rate"], 0.0)

    def test_nan_before_update_and_gradient_failure_handle_an_empty_trace(self):
        for failure_type in ("probability", "gradient"):
            with self.subTest(failure_type=failure_type):
                runtime = self.runtime([float("nan")] if failure_type == "probability" else [0.5])
                if failure_type == "gradient":
                    def invalid_gradient(embedding):
                        embedding.register_hook(lambda grad: torch.full_like(grad, float("nan")))
                        return embedding
                    runtime.projector.model = invalid_gradient
                with self.runtime_patches(), patch.object(runner, "draw_candidate") as draw:
                    payload = self.run_task(runtime)
                draw.assert_not_called()
                self.assertEqual(payload["result"]["stop_reason"], "numerical_failure")
                self.assertEqual(payload["result"]["optimization_updates"], 0)
                expected_phase = "before_update" if failure_type == "probability" else "backward"
                self.assertEqual(payload["numerical_failure"]["phase"], expected_phase)
                # A later task still executes normally in the same process.
                self.item = {**self.item, "image_id": self.item["image_id"] + 1}
        with self.runtime_patches(), patch.object(runner, "draw_candidate", return_value=self.candidate(valid=True)):
            payload = self.run_task(self.runtime([0.9, 0.95]))
        self.assertEqual(payload["result"]["stop_reason"], "valid_attack_success")

    def test_unrelated_runtime_errors_still_propagate(self):
        runtime = self.runtime([])
        runtime.projector.model = lambda embedding: (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
        with self.runtime_patches(), self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            self.run_task(runtime)

    def test_resume_exception_requires_exact_predecessor_and_unchanged_config_dependencies(self):
        cfg = runner.load_config(runner.DEFAULT_CONFIG)
        current_signature, current = runner.experiment_identity(cfg)
        runner_path = str(Path(runner.__file__).resolve())
        archived = {**current, "implementation_sha256": {
            **current["implementation_sha256"], runner_path: runner.NUMERICAL_PATCH_PREDECESSOR}}
        archived_signature = hashlib.sha256(json.dumps(archived, sort_keys=True).encode()).hexdigest()
        runner.json_save(self.output / "input_provenance.json", archived)
        runner.json_save(self.output / "experiment_signature.json", {"sha256": archived_signature})
        self.assertEqual(runner.compatible_identity(self.output, current_signature, current),
                         (archived_signature, archived))
        for change in ("config", "dependency", "predecessor"):
            with self.subTest(change=change):
                altered = {"config": dict(current["config"]),
                           "implementation_sha256": dict(current["implementation_sha256"])}
                if change == "config":
                    altered["config"]["learning_rate"] = 0.2
                elif change == "dependency":
                    altered["implementation_sha256"][str(runner.exp1.ROOT / "st-ghost/run_stage1.py")] = "changed"
                else:
                    wrong = {**archived, "implementation_sha256": {
                        **archived["implementation_sha256"], runner_path: "unknown"}}
                    wrong_signature = hashlib.sha256(json.dumps(wrong, sort_keys=True).encode()).hexdigest()
                    runner.json_save(self.output / "input_provenance.json", wrong)
                    runner.json_save(self.output / "experiment_signature.json", {"sha256": wrong_signature})
                with self.assertRaisesRegex(ValueError, "incompatibly"):
                    runner.compatible_identity(self.output, current_signature, altered)


def random_seed_owner():
    # This module is shared with the runner, so an accidental seed_all call
    # through the imported exp1 helpers also fails the test.
    import random
    return random


if __name__ == "__main__":
    unittest.main()
