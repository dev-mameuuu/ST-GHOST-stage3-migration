import contextlib
import csv
import hashlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from diffusers import PNDMScheduler
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_stage2_level_oo_generation_feedback_victim_evaluation as evaluation
from run_stage2_level_oo_atomic_guided_embeddings import _save_embedding_atomic


def fixed_state():
    # A repeated timestep straddles a stage boundary; use execution order.
    return SimpleNamespace(timesteps=torch.tensor([
        421, 401, 381, 361, 361, 321, 301, 281, 261, 241, 221,
        201, 181, 161, 141, 121, 101, 81, 61, 41, 21, 1]), timestep_index=1)


def test_config():
    return dict(lambda_clean=1.0, lambda_victim=2.0, clean_threshold=0.85,
                learning_rate=0.02, weight_decay=0.0, clean_slack=0.02,
                condition_mode="staged")


class VictimFeedbackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = Qwen2_5_VLConfig(
            text_config=dict(
                vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]}),
            vision_config=dict(
                depth=1, hidden_size=32, num_heads=2, intermediate_size=64,
                out_hidden_size=32, fullatt_block_indexes=[0]),
            image_token_id=4, video_token_id=5, vision_start_token_id=6,
            vision_end_token_id=7, bos_token_id=1, pad_token_id=0, eos_token_id=2)
        model = Qwen2_5_VLForConditionalGeneration(config).eval().requires_grad_(False)
        self.victim = SimpleNamespace(
            model=model, processor=Qwen2VLImageProcessorFast(), yes_ids=[9, 10])
        self.config = test_config()
        self.preprocess = mock.patch.object(
            evaluation.stage1, "vllm_standard_preprocessing", side_effect=self.preprocessing)
        self.preprocess.start()
        self.addCleanup(self.preprocess.stop)

    def preprocessing(self, processor, question, image, **kwargs):
        self.assertEqual(question, "joint question")
        self.assertFalse(kwargs["do_rescale"])
        inputs = processor(images=[image], return_tensors="pt", **kwargs)
        count = int(inputs["image_grid_thw"].prod(-1).sum()) // 4
        ids = torch.tensor([[1, 6, *([4] * count), 7, 8]])
        inputs.update(input_ids=ids, attention_mask=torch.ones_like(ids))
        return inputs

    def test_loss_matches_raw_yes_probability_and_has_native_image_gradient(self):
        image = torch.full((1, 3, 56, 56), 0.5, requires_grad=True)
        with torch.no_grad():
            inputs = self.preprocessing(
                self.victim.processor, "joint question", image[0], do_rescale=False)
            logits = self.victim.model(**inputs, use_cache=False).logits[:, -1].float()
            expected = logits.softmax(-1)[0, self.victim.yes_ids].sum().item()
        with mock.patch.object(self.victim.model, "generate", side_effect=AssertionError("decoding used")), \
                mock.patch.object(self.victim.model.visual, "forward", wraps=self.victim.model.visual.forward) as vision:
            terms = evaluation.victim_terms(self.victim, image, "joint question")
            self.assertAlmostEqual(float(terms["feedback_yes_probability"].detach()), expected, places=6)
            self.assertAlmostEqual(float(terms["victim_loss"].detach()), -math.log(expected), places=5)
            terms["victim_loss"].backward()
        self.assertGreaterEqual(vision.call_count, 2)
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.norm()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.victim.model.parameters()))

    def test_detached_preprocessing_is_rejected(self):
        image = torch.full((1, 3, 56, 56), 0.5, requires_grad=True)
        with mock.patch.object(
                evaluation.stage1, "vllm_standard_preprocessing",
                side_effect=lambda processor, question, pixels, **kwargs:
                self.preprocessing(processor, question, pixels.detach(), **kwargs)):
            with self.assertRaisesRegex(RuntimeError, "detached"):
                evaluation.victim_terms(self.victim, image, "joint question")

    def test_yes_log_loss_is_stable_when_softmax_underflows(self):
        image = torch.full((1, 3, 56, 56), 0.5, requires_grad=True)

        def forward(**inputs):
            signal = inputs["pixel_values"].mean()
            logits = torch.stack([signal - 1000, signal * 0]).view(1, 1, 2)
            return SimpleNamespace(logits=logits)

        victim = SimpleNamespace(
            processor=self.victim.processor, yes_ids=[0],
            model=SimpleNamespace(visual=SimpleNamespace(dtype=torch.float32)))

        class FrozenModel:
            visual = SimpleNamespace(dtype=torch.float32)

            def __call__(self, **inputs):
                return forward(**inputs)

        victim.model = FrozenModel()
        terms = evaluation.victim_terms(victim, image, "joint question")
        self.assertTrue(torch.isfinite(terms["victim_loss"]))
        self.assertEqual(float(terms["feedback_yes_probability"].detach()), 0.0)
        terms["victim_loss"].backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.norm()), 0)

    def test_only_residuals_change_and_every_stage_receives_victim_gradient(self):
        z_star = torch.ones(1, 2, requires_grad=True)
        z0 = torch.ones(1, 2, requires_grad=True)
        runtime = SimpleNamespace(victim=self.victim)
        before = {name: parameter.detach().clone()
                  for name, parameter in self.victim.model.named_parameters()}

        def render(_runtime, _state, conditions, _cfg, **kwargs):
            self.assertEqual(kwargs["stage_mapping"], list(range(21)))
            return conditions[:, 0, 0].mean().sigmoid().expand(1, 3, 56, 56)

        with mock.patch.object(evaluation.feedback, "_render", side_effect=render), \
                mock.patch.object(evaluation.feedback, "_clip_image", return_value=z_star.detach()):
            conditions, trace = evaluation.reproduce_full_condition(
                runtime, fixed_state(), z_star, z0, self.config, 1, "joint question",
                evaluation.CleanReference(1.0, 0.85))
        self.assertEqual(conditions.shape, (21, 1, 2))
        self.assertIsNone(z_star.grad)
        self.assertIsNone(z0.grad)
        self.assertEqual(len(trace), 1)
        self.assertAlmostEqual(trace[0]["clean_loss_before_update"], 0.0)
        for index, stage in enumerate(evaluation.STAGE_NAMES):
            self.assertFalse(torch.equal(conditions[index], z_star.detach()))
            self.assertGreater(trace[0][f"{stage}_gradient_norm"], 0)
            self.assertEqual(trace[0][f"{stage}_residual_norm"], 0)
        self.assertNotIn("target_loss_before_update", trace[0])
        self.assertAlmostEqual(trace[0]["total_loss_before_update"],
                               2 * trace[0]["victim_loss_before_update"], places=6)
        for name, parameter in self.victim.model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))

    def test_total_metrics_contain_only_weighted_victim_and_clean_losses(self):
        z_star, z0, z_hat = torch.ones(1, 2), torch.tensor([[1., 0.]]), torch.tensor([[0.6, 0.8]])
        runtime = SimpleNamespace(victim=self.victim)
        reference = evaluation.CleanReference(0.8, 0.78)
        terms = {"victim_loss": torch.tensor(2.0), "feedback_yes_probability": torch.tensor(math.exp(-2))}
        with mock.patch.object(evaluation.feedback, "_clip_image", return_value=z_hat), \
                mock.patch.object(evaluation, "victim_terms", return_value=terms):
            for weight in (0.0, 1.0, 3.0):
                cfg = {**self.config, "lambda_victim": weight, "lambda_clean": 3.0}
                metrics = evaluation.feedback_metrics(
                    runtime, None, z_star, z0, cfg, "joint question", reference)
                self.assertEqual(set(metrics), set(evaluation.METRIC_FIELDS))
                self.assertNotIn("target_loss", metrics)
                self.assertAlmostEqual(metrics["total_loss"], 3 * (0.78 - 0.6) ** 2 + weight * 2, places=6)


class ScheduleTests(unittest.TestCase):
    def test_mapping_uses_active_steps_including_repeated_timesteps(self):
        schedule = evaluation.condition_schedule(fixed_state(), test_config())
        self.assertEqual(schedule["active_timesteps"], [
            401, 381, 361, 361, 321, 301, 281, 261, 241, 221,
            201, 181, 161, 141, 121, 101, 81, 61, 41, 21, 1])
        self.assertEqual(schedule["stage_names"], [f"stage_{index}" for index in range(1, 22)])
        self.assertEqual(schedule["stage_mapping"], list(range(21)))
        evaluation.validate_condition_schedule(schedule)
        shared = evaluation.condition_schedule(fixed_state(), {"condition_mode": "shared"})
        self.assertEqual(shared["stage_names"], ["shared"])
        self.assertEqual(shared["stage_mapping"], [0] * 21)
        with self.assertRaises(ValueError):
            evaluation.stage_mapping(6, "staged")

    def test_adaptive_clean_threshold_remains_a_frozen_hinge(self):
        cfg = test_config()
        for baseline, expected in ((0.92, 0.85), (0.8, 0.78), (0.75, 0.73), (-0.99, -1.0)):
            self.assertAlmostEqual(evaluation.adaptive_clean_threshold(baseline, cfg), expected)
        reference = evaluation.CleanReference(0.8, 0.78)
        z0, target = torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]])
        for cosine, expected_loss in ((0.9, 0.), (0.8, 0.), (0.7, 0.08 ** 2)):
            z_hat = torch.tensor([[cosine, math.sqrt(1 - cosine ** 2)]])
            metrics = evaluation.feature_metrics(z_hat, target, z0, cfg, reference)
            self.assertAlmostEqual(metrics["clean_loss"], expected_loss, places=6)
            self.assertEqual(metrics["effective_clean_threshold"], 0.78)

    def test_gradient_diagnostics_detect_cancellation(self):
        residuals = torch.nn.Parameter(torch.zeros(21, 1, 2))
        residuals.grad = torch.zeros_like(residuals)
        residuals.grad[:2] = torch.tensor([[[1., 0.]], [[-1., 0.]]])
        metrics = evaluation.condition_diagnostics(residuals, evaluation.STAGE_NAMES)
        self.assertEqual(metrics["gradient_cancellation_ratio"], 0)
        self.assertEqual(metrics["gradient_cosine_stage_1_stage_2"], -1)
        self.assertEqual(metrics["gradient_cosine_stage_1_stage_7"], 0)

    def test_cli_defaults_and_invalid_values(self):
        with mock.patch.object(sys, "argv", ["script"]):
            args = evaluation.parse_args()
        self.assertEqual(args.lambda_victim, 2.0)
        self.assertEqual(args.condition_mode, "staged")
        self.assertEqual(args.phase, "all")
        self.assertEqual(args.initialization, "stage1")
        self.assertEqual(args.output_dir, evaluation.OUTPUT_DEFAULT)
        with mock.patch.object(sys, "argv", ["clean-script"]):
            clean_args = evaluation.parse_args(default_initialization="clean")
        self.assertEqual(clean_args.initialization, "clean")
        self.assertEqual(clean_args.output_dir, evaluation.CLEAN_OUTPUT_DEFAULT)
        for flag, values in (("--lambda-victim", ("-1", "nan", "inf")),
                             ("--clean-slack", ("-0.01", "2.1", "nan")),
                             ("--condition-mode", ("bad",)), ("--phase", ("bad",))):
            for value in values:
                with mock.patch.object(sys, "argv", ["script", flag, value]), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    evaluation.parse_args()

    def test_old_experiment_output_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "experiment_signature.json").write_text(json.dumps({
                "protocol_version": "stage2_level_oo_generation_feedback_victim_evaluation_staged_victim_clean_step45_lr003_v1",
                "sha256": "old"}))
            with self.assertRaisesRegex(ValueError, "different experiment"):
                evaluation.ensure_compatible(output, "new")

    def test_legacy_vae_checkpoints_are_rejected_before_any_output_is_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            source = {
                "source": root / "archive", "source_signature": {"sha256": "archive"},
                "input_sha256": {}, "tasks": [], "cfg": test_config(),
            }
            legacy_audit = evaluation.provenance(source, evaluation.FULL_MODE_UPDATES)
            self.assertEqual(legacy_audit.pop("vae_image_contract"),
                             evaluation.feedback.VAE_IMAGE_CONTRACT)
            legacy_signature = hashlib.sha256(
                json.dumps(legacy_audit, sort_keys=True).encode()).hexdigest()
            path = output / "experiment_signature.json"
            path.write_text(json.dumps({"protocol_version": evaluation.PROTOCOL_VERSION,
                                        "sha256": legacy_signature}))
            original = path.read_bytes()
            args = SimpleNamespace(output_dir=output, initialization="stage1", lambda_victim=2.0,
                                   condition_mode="staged", clean_slack=0.02,
                                   validate_only=True, phase="all")
            with self.assertRaisesRegex(ValueError, "different experiment"):
                evaluation.run_experiment(args, source)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(output.iterdir()), [path])


class SSIMConstraintTests(unittest.TestCase):
    @staticmethod
    def config():
        return {**test_config(), "ssim_constraint": evaluation.SSIM_CONSTRAINT,
                "ssim_threshold": 0.8, "lambda_ssim": 1.0}

    def test_torch_ssim_matches_previous_skimage_protocol(self):
        from skimage.metrics import structural_similarity

        torch.manual_seed(21)
        source = torch.rand(1, 3, 768, 768)
        original_shape = source.shape
        reference = evaluation.downsample_for_ssim(source)
        generated = evaluation.downsample_for_ssim(
            (0.7 * source + 0.2 * source.roll(9, -1) + 0.1).clamp(0, 1))
        for image in (reference, generated, torch.full_like(reference, 0.5)):
            expected = structural_similarity(
                reference[0].permute(1, 2, 0).double().numpy(),
                image[0].permute(1, 2, 0).double().numpy(),
                data_range=1.0, channel_axis=2, gaussian_weights=True,
                sigma=1.5, win_size=11, use_sample_covariance=False,
                K1=0.01, K2=0.03)
            self.assertAlmostEqual(float(evaluation.rgb_ssim(image, reference)),
                                   float(expected), delta=5e-5)
        self.assertEqual(source.shape, original_shape)

    def test_hinge_has_pixel_gradients_below_threshold_and_none_above(self):
        reference = torch.full((1, 3, 256, 256), 0.2, requires_grad=True)
        image = torch.full((1, 3, 768, 768), 0.8, requires_grad=True)
        terms = evaluation.ssim_terms(image, reference, self.config())
        self.assertLess(float(terms["ssim_256"].detach()), 0.8)
        self.assertGreater(float(terms["ssim_loss"].detach()), 0)
        terms["ssim_loss"].backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.norm()), 0)
        self.assertIsNone(reference.grad)

        identical = torch.full_like(image, 0.2, requires_grad=True)
        terms = evaluation.ssim_terms(identical, reference, self.config())
        self.assertAlmostEqual(float(terms["ssim_256"].detach()), 1.0, places=5)
        self.assertEqual(float(terms["ssim_loss"].detach()), 0)
        terms["ssim_loss"].backward()
        self.assertEqual(float(identical.grad.norm()), 0)
        self.assertEqual(evaluation.ssim_terms(None, None, test_config()), {})
        with self.assertRaisesRegex(ValueError, "original clean RGB"):
            evaluation.ssim_terms(image, None, self.config())

    def test_ssim_alone_updates_all_21_conditions(self):
        cfg = {**self.config(), "lambda_victim": 0.0}
        initial = torch.ones(1, 2)
        reference = torch.full((1, 3, 256, 256), 0.2)

        def render(runtime, state, conditions, config, **kwargs):
            return conditions.mean().sigmoid().expand(1, 3, 768, 768)

        with mock.patch.object(evaluation.feedback, "_render", side_effect=render), \
                mock.patch.object(evaluation.feedback, "_clip_image", return_value=initial), \
                mock.patch.object(evaluation, "victim_terms", return_value={
                    "victim_loss": torch.tensor(0.), "feedback_yes_probability": torch.tensor(0.5)}):
            conditions, trace = evaluation.reproduce_full_condition(
                SimpleNamespace(victim=object()), fixed_state(), initial, initial,
                cfg, 1, "question", evaluation.CleanReference(1.0, 0.85),
                ssim_reference=reference)
        self.assertEqual(trace[0]["clean_loss_before_update"], 0)
        self.assertAlmostEqual(trace[0]["total_loss_before_update"],
                               trace[0]["ssim_loss_before_update"], places=6)
        for index, stage in enumerate(evaluation.STAGE_NAMES):
            self.assertGreater(trace[0][f"{stage}_gradient_norm"], 0)
            self.assertTrue((conditions[index] < initial).all())

    def test_cli_ssim_defaults_and_invalid_thresholds(self):
        with mock.patch.object(sys, "argv", ["ssim-script"]):
            args = evaluation.parse_args(default_ssim=True)
        self.assertTrue(args.ssim)
        self.assertEqual(args.ssim_threshold, 0.8)
        self.assertEqual(args.output_dir, evaluation.SSIM_OUTPUT_DEFAULT)
        self.assertEqual(args.initialization, "stage1")
        self.assertEqual(args.condition_mode, "staged")
        for value in ("-0.1", "1.1", "nan", "inf"):
            with mock.patch.object(sys, "argv", ["ssim-script", "--ssim-threshold", value]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluation.parse_args(default_ssim=True)


class VAEImageNormalizationTests(unittest.TestCase):
    def test_signed_vae_round_trip_preserves_dark_mid_and_bright_pixels(self):
        # An identity VAE and no-op scheduler isolate the RGB/VAE boundary.
        # The old unsigned encoder input made even black decode to RGB 0.5.
        pixels = torch.tensor([0.0, 0.125, 0.5, 0.875, 1.0]).view(1, 1, 5).repeat(3, 1, 1)
        scaling_factor = 0.25
        encode = mock.Mock(side_effect=lambda value: SimpleNamespace(
            latent_dist=SimpleNamespace(sample=lambda: value.clone())))
        decode = mock.Mock(side_effect=lambda value, **_: (value,))
        scheduler = SimpleNamespace(
            timesteps=torch.tensor([1]), set_timesteps=mock.Mock(),
            add_noise=lambda latent, *_: latent,
            scale_model_input=lambda latent, _: latent,
            step=lambda prediction, timestep, latent, **_: (latent,))

        class UNet(torch.nn.Module):
            dtype = torch.float32

            def forward(self, sample, timestep, **kwargs):
                return (torch.zeros_like(sample),)

        runtime = SimpleNamespace(
            load_diffusion=mock.Mock(), device=torch.device("cpu"),
            diffusion_preprocess=lambda _: pixels,
            pipe=SimpleNamespace(
                unet=UNet(), scheduler=scheduler,
                vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=scaling_factor),
                                    encode=encode, decode=decode),
                noise_image_embeddings=lambda image_embeds, **_: image_embeds,
                prepare_extra_step_kwargs=lambda *_: {}))
        cfg = {"num_inference_steps": 1, "diffusion_t": 0, "guidance_scale": 2.0}
        fork_rng = torch.random.fork_rng
        with mock.patch.object(torch.cuda, "current_device", return_value=0), \
                mock.patch.object(torch.cuda, "manual_seed_all"), \
                mock.patch.object(torch.random, "fork_rng",
                                  side_effect=lambda **_: fork_rng(devices=[])), \
                mock.patch.object(evaluation.feedback, "_prompt_embeddings",
                                  return_value=torch.zeros(2, 1, 1)):
            with evaluation.Image.new("RGB", (5, 1)) as source:
                state = evaluation.feedback._make_state(runtime, source, 42, "", cfg)
        signed_pixels = pixels.unsqueeze(0) * 2 - 1
        torch.testing.assert_close(encode.call_args.args[0], signed_pixels)
        torch.testing.assert_close(state.vae_latent, signed_pixels * scaling_factor)
        self.assertFalse(state.vae_latent.requires_grad)
        state.vae_latent.requires_grad_(True)
        result = evaluation.feedback._render(runtime, state, torch.ones(1, 1024), cfg)
        torch.testing.assert_close(decode.call_args.args[0], signed_pixels)
        torch.testing.assert_close(result, pixels.unsqueeze(0), rtol=0, atol=0)
        result.mean().backward()
        self.assertTrue(torch.isfinite(state.vae_latent.grad).all())
        self.assertTrue((state.vae_latent.grad > 0).all())


class RendererTests(unittest.TestCase):
    def setUp(self):
        # Match the actual unCLIP schedule, whose final executed timestep is 1.
        scheduler = PNDMScheduler(skip_prk_steps=True, steps_offset=1)
        scheduler.set_timesteps(50)

        class UNet(torch.nn.Module):
            dtype = torch.float32

            def forward(self, sample, timestep, class_labels, **kwargs):
                signal = class_labels.sum(-1).view(-1, 1, 1, 1)
                return (0.05 * sample + 0.01 * signal,)

        self.noises = []

        def condition_noise(image_embeds, noise_level, noise):
            self.noises.append(noise.detach().clone())
            return image_embeds + 0.02 * noise

        pipe = SimpleNamespace(
            unet=UNet(), scheduler=scheduler, noise_image_embeddings=condition_noise,
            prepare_extra_step_kwargs=lambda *_: {},
            vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=1.0),
                                decode=lambda latent, **kwargs: (0.01 * latent,)))
        self.runtime = SimpleNamespace(pipe=pipe, device=torch.device("cpu"))
        timesteps = scheduler.timesteps.clone()
        self.state = evaluation.feedback.GenerationState(
            vae_latent=torch.ones(1, 3, 4, 4) * 0.01,
            diffusion_noise=torch.ones(1, 3, 4, 4) * 0.02,
            image_embedding_noise=torch.ones(1, 2),
            prompt_embeds=torch.zeros(2, 1, 1), timestep_index=30,
            timesteps=timesteps, timestep=timesteps[30:31])
        self.cfg = {**test_config(), "num_inference_steps": 50, "guidance_scale": 2.0}
        self.target = torch.tensor([[0.3, 0.7]])

    def test_equal_stage_conditions_match_legacy_render_and_reset_pndm(self):
        with torch.no_grad():
            legacy = evaluation.feedback._render(self.runtime, self.state, self.target, self.cfg)
            conditions = self.target.unsqueeze(0).repeat(21, 1, 1)
            staged = evaluation.render_conditions(self.runtime, self.state, conditions, self.cfg)
            again = evaluation.render_conditions(self.runtime, self.state, conditions, self.cfg)
            shared = evaluation.render_conditions(
                self.runtime, self.state, self.target.unsqueeze(0), {**self.cfg, "condition_mode": "shared"})
        torch.testing.assert_close(staged, legacy, rtol=0, atol=0)
        torch.testing.assert_close(again, legacy, rtol=0, atol=0)
        torch.testing.assert_close(shared, legacy, rtol=0, atol=0)
        for noise in self.noises:
            torch.testing.assert_close(noise, self.state.image_embedding_noise)

    def test_stage_gradients_sum_to_shared_gradient_at_equal_conditions(self):
        target = self.target.clone().requires_grad_(True)
        evaluation.feedback._render(self.runtime, self.state, target, self.cfg).mean().backward()
        conditions = self.target.unsqueeze(0).repeat(21, 1, 1).requires_grad_(True)
        evaluation.render_conditions(self.runtime, self.state, conditions, self.cfg).mean().backward()
        self.assertTrue(torch.isfinite(conditions.grad).all())
        self.assertTrue((conditions.grad.flatten(1).norm(dim=1) > 0).all())
        torch.testing.assert_close(conditions.grad.sum(0), target.grad)

    def test_invalid_stage_mapping_is_rejected(self):
        conditions = self.target.unsqueeze(0).repeat(21, 1, 1)
        with self.assertRaisesRegex(ValueError, "mapping"):
            evaluation.feedback._render(
                self.runtime, self.state, conditions, self.cfg, stage_mapping=[0])


class SnapshotStorageTests(unittest.TestCase):
    def setUp(self):
        self.config = test_config()
        self.z_star = torch.linspace(0.2, 0.8, 1024).view(1, 1024)
        self.runtime = SimpleNamespace(
            device=torch.device("cpu"), victim=object(),
            clip=SimpleNamespace(images=lambda _: self.z_star.clone() * 0.5))

    def make_task(self, root):
        source = root / "source.png"
        evaluation.Image.new("RGB", (4, 4), (128, 128, 128)).save(source)
        archive = _save_embedding_atomic(root / "source.pt", self.z_star, {})
        return {
            "seed": 42, "spec": "cat_chair", "image_id": 1,
            "source_path": str(source), "embedding_path": archive["embedding_path"],
            "embedding_sha256": archive["embedding_sha256"], "terminal_step": 3,
            "trigger_prompt": "joint question", "embedding": self.z_star,
        }

    @staticmethod
    def render_image(conditions):
        # Different stage sensitivities, with gradients reaching every stage.
        weights = torch.arange(1, len(conditions) + 1, dtype=conditions.dtype).view(-1, 1, 1)
        combined = (conditions * weights).sum(0) / weights.sum()
        return combined[:, :3].view(1, 3, 1, 1).sigmoid().expand(1, 3, 4, 4)

    @contextlib.contextmanager
    def mocked_generation(self, fail_on_render=None):
        calls = 0

        def render(runtime, state, conditions, cfg, **kwargs):
            nonlocal calls
            calls += 1
            if calls == fail_on_render:
                raise RuntimeError("interrupted task")
            return self.render_image(conditions)

        def clip_image(runtime, image):
            return image.mean().view(1, 1).expand(1, 1024)

        def victim_terms(victim, image, question):
            probability = image.mean()
            return {"victim_loss": -probability.log(), "feedback_yes_probability": probability}

        with mock.patch.object(evaluation.feedback, "_make_state", side_effect=lambda *_: fixed_state()), \
                mock.patch.object(evaluation.feedback, "_render", side_effect=render) as renderer, \
                mock.patch.object(evaluation.feedback, "_clip_image", side_effect=clip_image), \
                mock.patch.object(evaluation, "victim_terms", side_effect=victim_terms):
            yield renderer

    def test_snapshots_match_completed_updates_and_resume_checks_artifacts(self):
        for mode, count in (("staged", 21), ("shared", 1)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "output"
                task = self.make_task(root)
                cfg = {**self.config, "condition_mode": mode}
                with self.mocked_generation() as renderer:
                    record = evaluation.generate_task(self.runtime, task, cfg, output)
                self.assertEqual(renderer.call_count, 47)
                self.assertEqual(record["full_mode_updates"], 45)
                self.assertEqual([row["updates_completed"] for row in record["optimization_trace"]],
                                 list(range(45)))
                reference = record["clean_reference"]
                archive = torch.load(reference["path"], map_location="cpu", weights_only=True)
                torch.testing.assert_close(archive["clean_embedding"], self.z_star * 0.5)
                torch.testing.assert_close(archive["target_embedding"], self.z_star)
                self.assertNotIn("basis", archive)
                snapshots = record["optimization_snapshots"]
                expected_updates = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
                self.assertEqual([row["optimization_updates"] for row in snapshots], expected_updates)
                self.assertEqual([row["method"] for row in record["images"]], ["baseline_step0", "full_step45"])
                for snapshot in snapshots:
                    step = snapshot["optimization_updates"]
                    path = Path(snapshot["embedding_path"])
                    conditions = evaluation.load_condition_schedule(
                        path, snapshot["embedding_sha256"], record["condition_schedule"],
                        task, step, record["generation_seed"])
                    self.assertEqual(conditions.shape, (count, 1, 1024))
                    self.assertFalse(conditions.requires_grad)
                    if step == 0:
                        torch.testing.assert_close(conditions, self.z_star.unsqueeze(0).repeat(count, 1, 1))
                    else:
                        self.assertFalse(torch.equal(conditions[0], self.z_star))
                    pixels = self.render_image(conditions)[0, :, 0, 0]
                    expected_pixel = tuple((pixels * 255).round().int().tolist())
                    with evaluation.Image.open(snapshot["generated_path"]) as image:
                        self.assertEqual(image.getpixel((0, 0)), expected_pixel)
                    probability = float(self.render_image(conditions).mean())
                    self.assertAlmostEqual(snapshot["feature_metrics"]["victim_loss"],
                                           -math.log(probability), places=6)
                    self.assertAlmostEqual(snapshot["feature_metrics"]["feedback_yes_probability"],
                                           probability, places=6)
                    if 0 < step < 45:
                        trace = record["optimization_trace"][step]
                        for field in ("clean_loss", "victim_loss", "total_loss"):
                            self.assertAlmostEqual(snapshot["feature_metrics"][field],
                                                   trace[f"{field}_before_update"], places=6)
                checkpoint = evaluation.generation_checkpoint_path(output, task)
                evaluation.feedback._save_json_atomic(checkpoint, {"experiment_signature": "test", **record})
                records = evaluation.load_generation_checkpoints(output, "test", [task])
                evaluation.save_outputs(output, records, {})
                with (output / "optimization_snapshots.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([int(row["optimization_updates"]) for row in rows], expected_updates)
                with (output / "generated_images.csv").open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 2)
                with (output / "optimization_trace.csv").open() as handle:
                    trace_rows = list(csv.DictReader(handle))
                self.assertEqual(len(trace_rows), 45)
                self.assertEqual({row["method"] for row in trace_rows}, {"full_step45"})
                completion = json.loads((output / "completion_summary.json").read_text())
                self.assertEqual(completion["saved_optimization_updates"], expected_updates)
                self.assertEqual(completion["optimization_snapshots_expected"], 400)
                evaluated = {
                    (*evaluation.task_key(row), row["method"]): {
                        **row, "target_query_yes": int(row["optimization_updates"] == 45),
                        "target_answer_unparsed": 0,
                    }
                    for row in evaluation.generation_rows(records)
                }
                summary, paired = evaluation.summaries(evaluation.generation_rows(records), evaluated)
                all_methods = {row["method"]: row for row in summary if row["spec"] == "ALL"}
                self.assertEqual(all_methods["full_step45"]["yes_count"], 1)
                self.assertEqual(all_methods["baseline_step0"]["yes_count"], 0)
                self.assertEqual(next(row for row in paired if row["spec"] == "ALL")["no_to_yes"], 1)
                self.assertFalse((output / "task_subspaces").exists())
                Path(snapshots[2]["embedding_path"]).unlink()
                with self.assertRaises(FileNotFoundError):
                    evaluation.load_generation_checkpoints(output, "test", [task])

    def test_intermediate_artifacts_are_durable_before_task_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            task = self.make_task(root)
            with self.mocked_generation(fail_on_render=9), self.assertRaisesRegex(RuntimeError, "interrupted"):
                evaluation.generate_task(self.runtime, task, self.config, output)
            for step in (0, 5):
                manifest = json.loads(evaluation.snapshot_checkpoint_path(output, task, step).read_text())
                snapshot = manifest["snapshot"]
                path = Path(snapshot["embedding_path"])
                payload = torch.load(path, map_location="cpu", weights_only=True)
                evaluation.load_condition_schedule(
                    path, snapshot["embedding_sha256"], payload["condition_schedule"],
                    task, step, manifest["generation_seed"])
                self.assertTrue(Path(snapshot["generated_path"]).is_file())
            self.assertFalse(evaluation.snapshot_checkpoint_path(output, task, 10).exists())
            self.assertFalse(evaluation.generation_checkpoint_path(output, task).exists())

    def test_ssim_snapshots_total_loss_summary_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "ssim-output"
            task = self.make_task(root)
            cfg = SSIMConstraintTests.config()
            runtime = SimpleNamespace(**vars(self.runtime), diffusion_preprocess=lambda _: (
                torch.full((3, 768, 768), 128 / 255.0)))
            with self.mocked_generation():
                record = evaluation.generate_task(runtime, task, cfg, output)
            for snapshot in record["optimization_snapshots"]:
                metrics = snapshot["feature_metrics"]
                self.assertEqual(set(metrics), set(evaluation.METRIC_FIELDS + evaluation.SSIM_METRIC_FIELDS))
                self.assertAlmostEqual(metrics["total_loss"],
                                       2 * metrics["victim_loss"] + metrics["clean_loss"]
                                       + metrics["ssim_loss"], places=5)
                self.assertAlmostEqual(metrics["ssim_loss"],
                                       max(0, 0.8 - metrics["ssim_256"]) ** 2, places=6)
                step = snapshot["optimization_updates"]
                if 0 < step < 45:
                    for field in evaluation.SSIM_METRIC_FIELDS:
                        self.assertEqual(metrics[field], record["optimization_trace"][step][f"{field}_before_update"])
            checkpoint = evaluation.generation_checkpoint_path(output, task)
            evaluation.feedback._save_json_atomic(checkpoint, {"experiment_signature": "ssim-test", **record})
            records = evaluation.load_generation_checkpoints(output, "ssim-test", [task], cfg=cfg)
            evaluation.save_outputs(output, records, {}, tasks=[task])
            summary = json.loads((output / "ssim_summary.json").read_text())
            overall = [row for row in summary if row["spec"] == "ALL"]
            self.assertEqual([row["optimization_updates"] for row in overall], list(evaluation.SAVED_UPDATES))
            self.assertEqual(overall[-1]["ssim_256_mean"], record["optimization_snapshots"][-1]["feature_metrics"]["ssim_256"])
            # Optional-SSIM metrics cannot be silently resumed as the old objective.
            with self.assertRaisesRegex(ValueError, "Missing feature metrics"):
                evaluation.load_generation_checkpoints(output, "ssim-test", [task])

    def test_clean_initialization_drives_generation_snapshots_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "clean-output"
            task = {**self.make_task(root), "condition_initialization": "clean"}
            clean = self.z_star * 0.5
            with self.mocked_generation():
                record = evaluation.generate_task(self.runtime, task, self.config, output)
            reference = torch.load(record["clean_reference"]["path"], weights_only=True)
            torch.testing.assert_close(reference["clean_embedding"], clean, rtol=0, atol=0)
            torch.testing.assert_close(reference["target_embedding"], clean, rtol=0, atol=0)
            snapshots = record["optimization_snapshots"]
            self.assertEqual([row["optimization_updates"] for row in snapshots],
                             [0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
            for snapshot in snapshots:
                payload = torch.load(snapshot["embedding_path"], weights_only=True)
                self.assertEqual(payload["metadata"]["condition_initialization"], "clean")
                torch.testing.assert_close(payload["target_embedding"], clean, rtol=0, atol=0)
                conditions = evaluation.load_condition_schedule(
                    Path(snapshot["embedding_path"]), snapshot["embedding_sha256"],
                    record["condition_schedule"], task, snapshot["optimization_updates"],
                    record["generation_seed"], initial_embedding=clean)
                self.assertEqual(conditions.shape, (21, 1, 1024))
                if snapshot["optimization_updates"] == 0:
                    torch.testing.assert_close(conditions, clean.unsqueeze(0).expand_as(conditions),
                                               rtol=0, atol=0)
                else:
                    self.assertFalse(torch.equal(conditions[0], clean))
            self.assertLess(snapshots[-1]["feature_metrics"]["victim_loss"],
                            snapshots[0]["feature_metrics"]["victim_loss"])
            checkpoint = evaluation.generation_checkpoint_path(output, task)
            evaluation.feedback._save_json_atomic(checkpoint, {"experiment_signature": "clean-test", **record})
            records = evaluation.load_generation_checkpoints(output, "clean-test", [task])
            self.assertEqual(len(records), 1)
            stage1_task = {key: value for key, value in task.items() if key != "condition_initialization"}
            with self.assertRaisesRegex(ValueError, "Incompatible generation checkpoint"):
                evaluation.load_generation_checkpoints(output, "clean-test", [stage1_task])
            with self.assertRaisesRegex(ValueError, "Invalid condition schedule"):
                evaluation.load_condition_schedule(
                    Path(snapshots[0]["embedding_path"]), snapshots[0]["embedding_sha256"],
                    record["condition_schedule"], task, 0, record["generation_seed"],
                    initial_embedding=self.z_star)

            # Changing the archived Stage-1 tensor cannot affect this ablation.
            other_task = {**task, "embedding": self.z_star * 100}
            with self.mocked_generation():
                other = evaluation.generate_task(self.runtime, other_task, self.config, root / "other-output")
            self.assertEqual(record["generation_seed"], other["generation_seed"])
            for first, second in zip(snapshots, other["optimization_snapshots"]):
                self.assertEqual(first["feature_metrics"], second["feature_metrics"])
                self.assertEqual(first["image_state"]["sha256"], second["image_state"]["sha256"])
                first_conditions = torch.load(first["embedding_path"], weights_only=True)["conditions"]
                second_conditions = torch.load(second["embedding_path"], weights_only=True)["conditions"]
                torch.testing.assert_close(first_conditions, second_conditions, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
