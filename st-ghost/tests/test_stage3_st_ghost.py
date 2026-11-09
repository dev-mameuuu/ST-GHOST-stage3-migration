import argparse
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_stage3_st_ghost as experiment

evaluation = experiment.evaluation
four = experiment.four_seeds


class SelectionTests(unittest.TestCase):
    def test_balanced_unique_and_independent_of_input_order_or_attack_outcomes(self):
        names = ["pair_" + str(i) for i in range(5)]
        pool = [{"spec": name, "image_id": group * 400 + i,
                 "source_path": f"/images/{group * 400 + i}.jpg", "attack_success": i % 2}
                for group, name in enumerate(names) for i in range(400)]
        selected = experiment.select_images(pool, names)
        again = experiment.select_images(list(reversed(pool)), list(reversed(names)))
        self.assertEqual(selected, again)
        self.assertEqual(Counter(row["spec"] for row in selected), dict.fromkeys(names, 200))
        self.assertEqual(len({row["image_id"] for row in selected}), 1000)
        changed = [{**row, "attack_success": 1 - row["attack_success"]} for row in pool]
        self.assertEqual([row["image_id"] for row in selected],
                         [row["image_id"] for row in experiment.select_images(changed, names)])
        duplicate = pool.copy()
        duplicate[-1] = duplicate[0]
        with self.assertRaisesRegex(ValueError, "distinct original"):
            experiment.select_images(duplicate, names)

    def test_new_entry_defaults_and_optimizer_keeps_explicit_rate(self):
        with mock.patch.object(sys, "argv", ["stage3", "--validate-only"]):
            args = experiment.parse_args()
        self.assertEqual(args.learning_rate, 0.03)
        self.assertEqual(args.initialization, "clean")
        self.assertEqual(experiment.PLAN.saved_updates, (45,))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args.output_dir = root / "new"
            source = {
                "source": root / "archive", "source_signature": {"sha256": "source"},
                "input_sha256": {}, "specs": {}, "tasks": [],
                "cfg": {"learning_rate": 0.02, "lambda_clean": 1.0, "clean_threshold": 0.85},
            }
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                    torch.cuda, "is_available", side_effect=AssertionError("GPU touched")):
                evaluation.run_experiment(args, source, plan=experiment.PLAN)
            self.assertEqual(source["cfg"]["learning_rate"], 0.03)
            self.assertEqual(source["cfg"]["lambda_ssim"], 1.0)
            self.assertFalse(args.output_dir.exists())
            args.learning_rate = float("nan")
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                evaluation.run_experiment(args, source, plan=experiment.PLAN)


class CleanFourSeedTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "optimization"
        self.source.mkdir()
        original = self.root / "7.png"
        Image.new("RGB", (8, 8), (30, 60, 90)).save(original)
        self.task = {
            "seed": 42, "spec": "cat_chair", "image_id": 7, "source_path": str(original),
            "source_image_sha256": evaluation.sha256(original),
            "embedding_path": "", "embedding_sha256": "", "terminal_step": -1,
            "trigger_prompt": "Do you see both a cat and a chair?", "condition_initialization": "clean",
        }
        self.seed = evaluation.feedback.exp4.generation_seed(42, "cat_chair", 7, 0)
        initial = torch.linspace(-1, 1, 1024).view(1, 1024)
        reference = evaluation.save_clean_reference(
            self.source, self.task, initial, initial, evaluation.CleanReference(0.9, 0.85))
        schedule = {"condition_mode": "staged", "stage_names": list(evaluation.STAGE_NAMES),
                    "active_timesteps": list(range(21, 0, -1)), "stage_mapping": list(range(21))}
        condition_path = evaluation.condition_embedding_path(self.source, self.task, 45)
        saved = evaluation.save_condition_schedule(
            condition_path, initial.unsqueeze(0).repeat(21, 1, 1) + 0.1,
            schedule, self.task, {"optimization_updates": 45, "generation_seed": self.seed},
            initial_embedding=initial)
        generated = evaluation.generated_path(self.source, self.task, "full_step45")
        generated.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (35, 65, 95)).save(generated)
        state = evaluation.image_state(generated)
        row = {**evaluation.descriptor(self.task), "method": "full_step45", "optimization_updates": 45,
               "generation_seed": self.seed, "condition_embedding_path": str(condition_path),
               "condition_embedding_sha256": saved["embedding_sha256"], "generated_path": str(generated),
               "sha256": state["sha256"], "size": state["size"], "ssim_256": 0.81, "clean_cosine": 0.9}
        with (self.source / "generated_images.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        self.write(self.source / "resolved_config.json", {
            "sample_count": 1, "condition_stage_count": 21, "condition_mode": "staged",
            "full_mode_updates": 45, "condition_initialization": "clean",
            "ssim_constraint": evaluation.SSIM_CONSTRAINT,
            "vae_image_contract": evaluation.feedback.VAE_IMAGE_CONTRACT})
        self.write(self.source / "experiment_signature.json", {"sha256": "source"})
        self.write(self.source / "selected_embeddings.json", [evaluation.descriptor(self.task)])
        self.write(self.source / "completion_summary.json", {
            "generation_complete": 1, "victim_evaluations_complete": 1, "evaluation_complete": True})
        self.write(evaluation.generation_checkpoint_path(self.source, self.task), {
            "experiment_signature": "source", "task": evaluation.descriptor(self.task), "clean_reference": reference})
        result = {**row, "image_state": state, **self.answer(False)}
        self.write(evaluation.victim_checkpoint_path(self.source, result), {
            "experiment_signature": "source", "result": result})

    @staticmethod
    def write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def answer(self, yes):
        return {"target_question": self.task["trigger_prompt"], "target_answer": "Yes" if yes else "No",
                "target_decision": yes, "target_query_yes": int(yes),
                "target_answer_unparsed": 0, "target_yes_probability": 0.9 if yes else 0.1}

    def test_clean_schedule_load_any_of_four_and_checkpoint_resume(self):
        with mock.patch.object(evaluation.feedback, "_load_embedding", side_effect=AssertionError("Stage-1 loaded")):
            _, entries, _ = four.load_source(self.source, expected_sample_count=1, expected_initialization="clean")
        self.assertEqual(len(set(entries[0]["generation_seeds"])), 4)
        output = self.root / "four_seeds"
        records = {}
        four.import_originals(output, "signature", entries, records)
        for index in (1, 2, 3):
            row = four.identity(entries[0], index, output)
            path = Path(row["generated_path"])
            Image.new("RGB", (8, 8), (index * 40, 60, 90)).save(path)
            row.update(image_state=evaluation.image_state(path), ssim_256=0.8,
                       clean_cosine=0.9, **self.answer(index == 2))
            four.write_record(output, "signature", row)
            records[four.key(row)] = row
            if index == 2:
                partial = four.save_outputs(output, entries, records, protocol_version=experiment.PROTOCOL_VERSION)
                self.assertFalse(partial["evaluation_complete"])
                self.assertEqual(partial["observed_any_hit_tasks"], 1)
                self.assertIsNone(partial["completed_tasks_any_hit_rate"])
        summary = four.save_outputs(output, entries, records, protocol_version=experiment.PROTOCOL_VERSION)
        self.assertEqual(summary["tasks_expected"], 1)
        self.assertEqual(summary["images_expected"], 4)
        self.assertEqual(summary["new_seed_images_expected"], 3)
        self.assertTrue(summary["evaluation_complete"])
        self.assertEqual(summary["baseline_g0_success_rate"], 0)
        self.assertEqual(summary["completed_tasks_any_hit_rate"], 1)
        self.assertEqual(summary["all_images_query_yes_rate"], 0.25)
        self.assertEqual(summary["protocol_version"], experiment.PROTOCOL_VERSION)
        self.assertEqual(four.load_records(output, "signature", entries), records)
        path = Path(records[(*evaluation.task_key(self.task), 3)]["generated_path"])
        Image.new("RGB", (8, 8), (0, 0, 0)).save(path)
        with self.assertRaisesRegex(ValueError, "Incompatible candidate"):
            four.load_records(output, "signature", entries)

    def test_clean_reference_tampering_is_rejected_without_archive_fallback(self):
        path = evaluation.clean_reference_path(self.source, self.task)
        payload = torch.load(path, weights_only=True)
        payload["clean_embedding"].add_(1)
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "reference provenance"):
            four.load_source(self.source, expected_sample_count=1, expected_initialization="clean")

    def test_prepare_four_candidates_twice_without_gpu_or_new_optimization(self):
        args = argparse.Namespace(source_output=self.source, output_dir=self.root / "four_seeds",
                                  phase="all", validate_only=False, prepare_only=True, max_new_images=None)
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                torch.cuda, "is_available", side_effect=AssertionError("GPU touched")):
            for _ in range(2):
                four.run_experiment(args, expected_sample_count=1, expected_initialization="clean",
                                    protocol_version=experiment.PROTOCOL_VERSION)
        summary = json.loads((args.output_dir / "summary.json").read_text())
        self.assertEqual(summary["images_generated"], 1)
        self.assertEqual(summary["images_evaluated"], 1)
        self.assertFalse(summary["evaluation_complete"])


if __name__ == "__main__":
    unittest.main()
