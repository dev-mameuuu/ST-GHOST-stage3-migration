"""Stage 2 Level-OO generation feedback experiment.

This runner starts from terminal embeddings archived by
``run_stage2_level_oo_atomic_guided_embeddings.py``.  It keeps the original
GHOST diffusion settings, but makes the image-generation path differentiable:

``condition -> unCLIP UNet/scheduler -> VAE decode -> OpenCLIP image encoder``

Three paired modes are written for every archived task using one fixed
generation state (``generation_index=0``):

``baseline``
    Generate directly from the archived embedding.
``target_recovery``
    Update the condition with the relative squared feature error to the
    archived embedding.
``full``
    Add the clean-image hinge constraint with the default threshold 0.85.

The diffusion noise, source-image VAE latent, prompt embeddings, and image
embedding noise are fixed within each optimization trajectory.  No victim
model or detector is loaded.  The script is a runner only; it is not started
by importing the module.

Validate the archive without loading models::

    .venv-stage1/bin/python st-ghost/run_stage2_level_oo_generation_feedback.py \
        --validate-only
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "st-ghost"
if str(STAGE1) not in sys.path:
    sys.path.insert(0, str(STAGE1))

import run_stage1_exp4 as exp4  # noqa: E402
import run_stage1_exp4_level_oo as oo  # noqa: E402


LEVEL = "Level-OO"
PROTOCOL_VERSION = "stage2_level_oo_generation_feedback_v2"
ARCHIVE_DEFAULT = ROOT / "outputs" / "stage2_level_oo_atomic_guided_embeddings_v1"
OUTPUT_DEFAULT = ROOT / "outputs" / PROTOCOL_VERSION
DEFAULT_MODES = ("baseline", "target_recovery", "full")
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
VAE_IMAGE_CONTRACT = {
    "version": "rgb01_vae_signed_v1",
    "encoder_input": "2 * RGB_[0,1] - 1",
    "latent_scaling": "encode * scaling_factor; decode / scaling_factor",
    "decoder_output": "clamp(decoded / 2 + 0.5, 0, 1)",
}


def _as_int(value: object, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_name(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def _save_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _load_embedding(path: Path, expected_sha256: str = "") -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha256 and _sha256(path) != expected_sha256:
        raise RuntimeError(f"Embedding digest mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "stage2_terminal_embedding_v1":
        raise RuntimeError(f"Unsupported embedding archive format: {path}")
    embedding = payload.get("embedding")
    if not isinstance(embedding, torch.Tensor) or embedding.shape != (1, 1024):
        raise RuntimeError(f"Expected [1, 1024] embedding: {path}")
    if not torch.isfinite(embedding).all().item():
        raise RuntimeError(f"Non-finite embedding: {path}")
    return embedding.detach().to(dtype=torch.float32, device="cpu").contiguous()


def _archive_tasks(archive: Path, only_success: bool = False) -> Tuple[dict, List[dict], List[dict], List[dict]]:
    cfg_path = archive / "resolved_config.json"
    manifest_path = archive / "level_oo_shared_pool_manifest.json"
    specs_path = archive / "target_specs.json"
    for path in (cfg_path, manifest_path, specs_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("protocol_version") != "stage2_level_oo_atomic_guided_embeddings_v1":
        raise RuntimeError(f"Unexpected archive protocol: {cfg.get('protocol_version')!r}")
    manifest = json.loads(manifest_path.read_text())
    specs = json.loads(specs_path.read_text())
    source_by_key = {(str(row.get("spec")), _as_int(row.get("image_id"))): row
                     for row in manifest}
    tasks: List[dict] = []
    checkpoint_count = 0
    for checkpoint in sorted(archive.glob("seed_*/checkpoints/*.json")):
        checkpoint_count += 1
        payload = json.loads(checkpoint.read_text())
        result = payload.get("result", {})
        if not isinstance(result, dict):
            raise RuntimeError(f"Malformed result in {checkpoint}")
        seed = _as_int(result.get("seed"))
        spec = str(result.get("spec", ""))
        image_id = _as_int(result.get("image_id"))
        key = (spec, image_id)
        source = source_by_key.get(key)
        if source is None:
            raise RuntimeError(f"Checkpoint is absent from source manifest: {key}")
        if only_success and not _as_int(result.get("embedding_success")):
            continue
        embedding_path = Path(str(result.get("embedding_path", ""))).resolve()
        if embedding_path.parent.parent.parent != archive.resolve():
            raise RuntimeError(f"Embedding is outside archive: {embedding_path}")
        embedding = _load_embedding(embedding_path, str(result.get("embedding_sha256", "")))
        source_path = Path(str(result.get("source_path", source.get("source_path", "")))).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        tasks.append({
            "seed": seed,
            "spec": spec,
            "image_id": image_id,
            "level": str(result.get("level", LEVEL)),
            "source_path": str(source_path),
            "embedding_path": str(embedding_path),
            "embedding_sha256": str(result.get("embedding_sha256", "")),
            "embedding_success": _as_int(result.get("embedding_success"), 0),
            "terminal_step": _as_int(result.get("terminal_step"), -1),
            "trigger_prompt": str(result.get("trigger_prompt", "")),
            "embedding": embedding,
        })
    if not tasks:
        raise RuntimeError("No usable archived embedding checkpoints were found")
    if checkpoint_count == 0:
        raise RuntimeError(f"No checkpoints under {archive}")
    return cfg, manifest, specs, tasks


@dataclass
class GenerationState:
    vae_latent: torch.Tensor
    diffusion_noise: torch.Tensor
    image_embedding_noise: torch.Tensor
    prompt_embeds: torch.Tensor
    timestep_index: int
    timesteps: torch.Tensor
    timestep: torch.Tensor


def _prompt_embeddings(pipe, prompt: str, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        positive, negative = pipe.encode_prompt(
            prompt=prompt, device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt="low quality, ugly, unrealistic",
            prompt_embeds=None, negative_prompt_embeds=None, lora_scale=None)
        return torch.cat([negative, positive], dim=0).detach().to(dtype=pipe.unet.dtype)


def _make_state(runtime: exp4.Runtime, source: Image.Image, seed: int,
                prompt: str, cfg: Mapping[str, object]) -> GenerationState:
    runtime.load_diffusion()
    pipe = runtime.pipe
    device = runtime.device
    dtype = pipe.unet.dtype
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            source_tensor = runtime.diffusion_preprocess(source).unsqueeze(0).to(device=device, dtype=dtype)
            # diffusion_preprocess returns RGB in [0, 1]; the VAE expects [-1, 1].
            source_tensor = source_tensor * 2 - 1
            vae_latent = pipe.vae.encode(source_tensor).latent_dist.sample()
            vae_latent = vae_latent * pipe.vae.config.scaling_factor
        pipe.scheduler.set_timesteps(int(cfg["num_inference_steps"]), device=device)
        timesteps = pipe.scheduler.timesteps.detach().clone()
        index = min(int(cfg.get("diffusion_t", 0)), len(timesteps) - 1)
        timestep = timesteps[index:index + 1]
        diffusion_noise = torch.randn_like(vae_latent)
        image_embedding_noise = torch.randn((1, 1024), device=device, dtype=dtype)
        prompt_embeds = _prompt_embeddings(pipe, prompt, device)
    return GenerationState(vae_latent.detach(), diffusion_noise.detach(),
                           image_embedding_noise.detach(), prompt_embeds,
                           index, timesteps, timestep)


def _render(runtime: exp4.Runtime, state: GenerationState,
            condition: torch.Tensor, cfg: Mapping[str, object], *,
            stage_mapping: Sequence[int] | None = None) -> torch.Tensor:
    """Render one trajectory, optionally selecting a condition at each step.

    The original [1, 1024] interface is unchanged. Scheduled conditions have
    shape [stages, 1, 1024]; their mapping follows the executed scheduler
    sequence, including repeated PNDM timesteps.
    """
    active_timesteps = state.timesteps[state.timestep_index:]
    if stage_mapping is None:
        if condition.ndim != 2 or condition.shape[0] != 1:
            raise ValueError("A shared generation condition must have shape [1, dim]")
        conditions = condition.unsqueeze(0)
        mapping = [0] * len(active_timesteps)
    else:
        if condition.ndim != 3 or condition.shape[1] != 1:
            raise ValueError("Scheduled conditions must have shape [stages, 1, dim]")
        conditions = condition
        mapping = list(stage_mapping)
        if (len(mapping) != len(active_timesteps)
                or any(type(index) is not int or not 0 <= index < len(conditions)
                       for index in mapping)
                or set(mapping) != set(range(len(conditions)))):
            raise ValueError("Condition mapping must cover every active step and stage")
    pipe = runtime.pipe
    # PNDM retains step history between calls; each render needs a fresh trajectory.
    pipe.scheduler.set_timesteps(int(cfg["num_inference_steps"]), device=runtime.device)
    dtype = pipe.unet.dtype
    stage_embeds = []
    for stage_condition in conditions.to(device=runtime.device, dtype=dtype):
        image_embeds = pipe.noise_image_embeddings(
            image_embeds=stage_condition,
            noise_level=int(cfg.get("noise_level", 0)),
            noise=state.image_embedding_noise)
        stage_embeds.append(torch.cat([torch.zeros_like(image_embeds), image_embeds], dim=0))
    latents = pipe.scheduler.add_noise(
        state.vae_latent, state.diffusion_noise, state.timestep)
    extra = pipe.prepare_extra_step_kwargs(None, None)
    for index, timestep in enumerate(active_timesteps):
        latent_input = torch.cat([latents, latents], dim=0)
        latent_input = pipe.scheduler.scale_model_input(latent_input, timestep)
        unet_kwargs = {
            "encoder_hidden_states": state.prompt_embeds,
            "class_labels": stage_embeds[mapping[index]],
            "cross_attention_kwargs": None,
            "return_dict": False,
        }
        if torch.is_grad_enabled():
            noise_pred = checkpoint(
                pipe.unet, latent_input, timestep, use_reentrant=False,
                **unet_kwargs)[0]
        else:
            noise_pred = pipe.unet(latent_input, timestep, **unet_kwargs)[0]
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + float(cfg["guidance_scale"]) * (noise_text - noise_uncond)
        latents = pipe.scheduler.step(
            noise_pred, timestep, latents, **extra, return_dict=False)[0]
    decoded = pipe.vae.decode(
        latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    # Convert the signed VAE output back to RGB for CLIP, the victim, and saving.
    return (decoded / 2 + 0.5).clamp(0, 1)


def _clip_image(runtime: exp4.Runtime, image: torch.Tensor) -> torch.Tensor:
    model = runtime.clip.model
    dtype = next(model.parameters()).dtype
    pixels = F.interpolate(image.float(), size=(224, 224), mode="bicubic", align_corners=False)
    mean = torch.tensor(CLIP_MEAN, device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    return model.encode_image(((pixels - mean) / std).to(dtype=dtype)).float()


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(f"Non-finite generated image: {path}")
    array = (tensor[0].detach().float().permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).round().astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _feature_metrics(z_hat: torch.Tensor, z_star: torch.Tensor,
                     z0: torch.Tensor, threshold: float,
                     lambda_clean: float) -> Dict[str, float]:
    z_hat = z_hat.float()
    z_star = z_star.float()
    z0 = z0.float()
    target_loss = ((z_hat - z_star).pow(2).sum()
                   / (z_star.pow(2).sum() + 1e-8))
    clean_cos = F.cosine_similarity(z_hat, z0, dim=-1).mean()
    target_cos = F.cosine_similarity(z_hat, z_star, dim=-1).mean()
    clean_loss = torch.relu(torch.as_tensor(threshold, device=z_hat.device) - clean_cos).pow(2)
    total = target_loss + lambda_clean * clean_loss
    return {
        "target_loss": float(target_loss.detach().cpu()),
        "clean_loss": float(clean_loss.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
        "target_cosine": float(target_cos.detach().cpu()),
        "clean_cosine": float(clean_cos.detach().cpu()),
        "feature_mse": float(F.mse_loss(z_hat, z_star).detach().cpu()),
    }


def _optimize(runtime: exp4.Runtime, state: GenerationState,
              z_star: torch.Tensor, z0: torch.Tensor,
              mode: str, cfg: Mapping[str, object]) -> Tuple[torch.Tensor, List[dict]]:
    condition = torch.nn.Parameter(z_star.clone().to(runtime.device))
    steps: List[dict] = []
    if mode == "baseline":
        return condition.detach(), steps
    optimizer = torch.optim.AdamW([condition], lr=float(cfg["learning_rate"]))
    for step in range(int(cfg["outer_steps"])):
        image = _render(runtime, state, condition, cfg)
        z_hat = _clip_image(runtime, image)
        target_loss = ((z_hat - z_star).pow(2).sum()
                       / (z_star.pow(2).sum() + 1e-8))
        clean_cos = F.cosine_similarity(z_hat, z0, dim=-1).mean()
        clean_loss = torch.relu(
            torch.as_tensor(float(cfg["clean_threshold"]), device=clean_cos.device) - clean_cos).pow(2)
        total = target_loss + (float(cfg["lambda_clean"]) * clean_loss if mode == "full" else 0.0)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if condition.grad is None:
            raise RuntimeError(f"Missing condition gradient in {mode}, step {step}")
        if not torch.isfinite(condition.grad).all().item():
            raise RuntimeError(f"Non-finite condition gradient in {mode}, step {step}")
        grad_norm = float(condition.grad.detach().norm().cpu())
        if grad_norm == 0.0:
            raise RuntimeError(f"Zero condition gradient in {mode}, step {step}")
        optimizer.step()
        metrics = _feature_metrics(z_hat.detach(), z_star, z0,
                                   float(cfg["clean_threshold"]), float(cfg["lambda_clean"]))
        metrics.update({"step": step, "mode": mode, "gradient_norm": grad_norm,
                       "update_applied": 1})
        steps.append(metrics)
    return condition.detach(), steps


def _task_run(runtime: exp4.Runtime, task: Mapping[str, object],
              output: Path, cfg: Mapping[str, object], modes: Sequence[str]) -> Tuple[List[dict], List[dict], dict]:
    source = Image.open(str(task["source_path"])).convert("RGB")
    z_star = task["embedding"].to(runtime.device)
    with torch.no_grad():
        z0 = runtime.clip.images([source]).detach().float()
    spec = str(task["spec"])
    seed = int(task["seed"])
    image_id = int(task["image_id"])
    prompt = str(task.get("trigger_prompt", "")).strip()
    if not prompt:
        prompt = oo.oo_optimization_questions(
            exp4.CompositeSpec.from_dict(next(v for v in cfg["target_specs"] if v["name"] == spec)))[0]
    state = _make_state(runtime, source,
                        exp4.generation_seed(seed, spec, image_id, 0),
                        prompt, cfg)
    task_results: List[dict] = []
    generation_rows: List[dict] = []
    step_rows: List[dict] = []
    for mode in modes:
        optimized, traces = _optimize(runtime, state, z_star, z0, mode, cfg)
        step_rows.extend({"seed": seed, "spec": spec, "image_id": image_id, **row}
                         for row in traces)
        final_metrics: List[dict] = []
        with torch.no_grad():
            generated = _render(runtime, state, optimized, cfg)
            z_hat = _clip_image(runtime, generated)
            metrics = _feature_metrics(z_hat, z_star, z0,
                                       float(cfg["clean_threshold"]), float(cfg["lambda_clean"]))
        path = output / "generated" / mode / spec / LEVEL / f"seed{seed}_{image_id}_g0.png"
        _save_image(generated, path)
        row = {"seed": seed, "spec": spec, "image_id": image_id,
               "level": LEVEL, "mode": mode,
               "generation_index": 0,
               "generation_seed": exp4.generation_seed(seed, spec, image_id, 0),
               "generated_path": str(path), "optimization_steps": len(traces),
               **metrics}
        generation_rows.append(row)
        final_metrics.append(metrics)
        aggregate = {key: sum(row[key] for row in final_metrics) / len(final_metrics)
                     for key in final_metrics[0]}
        task_results.append({
            "seed": seed, "spec": spec, "image_id": image_id, "level": LEVEL,
            "mode": mode, "embedding_success": int(task["embedding_success"]),
            "embedding_path": str(task["embedding_path"]),
            "terminal_step": int(task["terminal_step"]),
            "outer_steps": len(traces), "learning_rate": float(cfg["learning_rate"]),
            "lambda_clean": float(cfg["lambda_clean"]),
            "clean_threshold": float(cfg["clean_threshold"]),
            "optimization_generation_index": 0,
            **{f"mean_{key}": value for key, value in aggregate.items()},
        })
        del optimized
        torch.cuda.empty_cache()
    source.close()
    return task_results, generation_rows, {"steps": step_rows}


def _signature(cfg: Mapping[str, object], tasks: Sequence[Mapping[str, object]], modes: Sequence[str]) -> str:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "vae_image_contract": VAE_IMAGE_CONTRACT,
        "archive": cfg["embedding_archive"],
        "modes": list(modes),
        "outer_steps": cfg["outer_steps"],
        "learning_rate": cfg["learning_rate"],
        "lambda_clean": cfg["lambda_clean"],
        "clean_threshold": cfg["clean_threshold"],
        "num_generations": cfg["num_generations"],
        "diffusion_t": cfg["diffusion_t"],
        "num_inference_steps": cfg["num_inference_steps"],
        "guidance_scale": cfg["guidance_scale"],
        "diffusion_compute_dtype": cfg["diffusion_compute_dtype"],
        "tasks": [(int(t["seed"]), str(t["spec"]), int(t["image_id"]), str(t["embedding_sha256"])) for t in tasks],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-archive", type=Path, default=ARCHIVE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--outer-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--lambda-clean", type=float, default=1.0)
    parser.add_argument("--clean-threshold", type=float, default=0.85)
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=list(DEFAULT_MODES))
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N archived tasks")
    parser.add_argument("--only-embedding-success", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.outer_steps < 0 or args.learning_rate <= 0 or args.lambda_clean < 0:
        parser.error("outer-steps must be non-negative; learning-rate positive; lambda-clean non-negative")
    if not 0 <= args.clean_threshold <= 1:
        parser.error("clean-threshold must be in [0, 1]")
    if args.limit < 0:
        parser.error("limit must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    archive = args.embedding_archive.resolve()
    cfg_source, manifest, specs, tasks = _archive_tasks(archive, args.only_embedding_success)
    if args.limit:
        tasks = tasks[:args.limit]
    cfg = dict(cfg_source)
    cfg.update({
        "protocol_version": PROTOCOL_VERSION,
        "vae_image_contract": VAE_IMAGE_CONTRACT,
        "embedding_archive": str(archive),
        "output_dir": str(args.output_dir.resolve()),
        "outer_steps": int(args.outer_steps),
        "learning_rate": float(args.learning_rate),
        "lambda_clean": float(args.lambda_clean),
        "clean_threshold": float(args.clean_threshold),
        "modes": list(args.modes),
        "gradient_generation": True,
        "diffusion_compute_dtype": "float32",
        "num_generations": 1,
        "optimization_generation_index": 0,
        "victim_evaluation": False,
        "detector_evaluation": False,
    })
    signature = _signature(cfg, tasks, args.modes)
    print(json.dumps({"protocol_version": PROTOCOL_VERSION,
                      "archive": str(archive), "tasks": len(tasks),
                      "modes": args.modes, "outer_steps": args.outer_steps,
                      "learning_rate": args.learning_rate,
                      "lambda_clean": args.lambda_clean,
                      "clean_threshold": args.clean_threshold,
                      "signature": signature, "validate_only": args.validate_only},
                     ensure_ascii=False, indent=2), flush=True)
    if args.validate_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the generation feedback experiment")
    output = args.output_dir.resolve()
    signature_path = output / "experiment_signature.json"
    if signature_path.exists():
        previous = json.loads(signature_path.read_text())
        if previous.get("sha256") != signature:
            raise RuntimeError(f"Output directory belongs to another protocol: {output}")
    elif args.no_resume and output.exists() and any(output.iterdir()):
        raise RuntimeError("--no-resume requires an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    _save_json_atomic(signature_path, {"protocol_version": PROTOCOL_VERSION, "sha256": signature})
    _save_json_atomic(output / "resolved_config.json", cfg)
    _save_json_atomic(output / "target_specs.json", specs)
    _save_json_atomic(output / "source_manifest_reference.json", {
        "embedding_archive": str(archive), "source_row_count": len(manifest),
        "pool_reselected": False, "victim_evaluation": False,
    })
    runtime = exp4.Runtime(cfg, attack=False)
    runtime.load_diffusion()
    # fp16 UNet backward produces non-finite condition gradients in this pipeline.
    runtime.pipe.unet.to(dtype=torch.float32)
    runtime.pipe.vae.to(dtype=torch.float32)
    runtime.pipe.unet.requires_grad_(False)
    runtime.pipe.vae.requires_grad_(False)
    runtime.pipe.text_encoder.requires_grad_(False)
    completed = set()
    all_results: List[dict] = []
    all_generations: List[dict] = []
    all_steps: List[dict] = []
    for task in tqdm(tasks, desc="Generation feedback"):
        key = (int(task["seed"]), str(task["spec"]), int(task["image_id"]))
        checkpoint = output / f"seed_{key[0]}" / "checkpoints" / f"{_safe_name(key[1])}__{key[2]}__{LEVEL}.json"
        if not args.no_resume and checkpoint.exists():
            payload = json.loads(checkpoint.read_text())
            if payload.get("experiment_signature") != signature:
                raise RuntimeError(f"Checkpoint signature mismatch: {checkpoint}")
            all_results.extend(payload.get("results", []))
            all_generations.extend(payload.get("generations", []))
            all_steps.extend(payload.get("steps", []))
            completed.add(key)
            continue
        results, generations, traces = _task_run(runtime, task, output, cfg, args.modes)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        _save_json_atomic(checkpoint, {"experiment_signature": signature,
                                       "results": results,
                                       "generations": generations,
                                       "steps": traces["steps"]})
        all_results.extend(results)
        all_generations.extend(generations)
        all_steps.extend(traces["steps"])
        completed.add(key)
        _save_csv_atomic(output / "task_results.csv", all_results)
        _save_csv_atomic(output / "generations.csv", all_generations)
        _save_csv_atomic(output / "optimization_steps.csv", all_steps)
        gc.collect()
        torch.cuda.empty_cache()
    _save_csv_atomic(output / "task_results.csv", all_results)
    _save_csv_atomic(output / "generations.csv", all_generations)
    _save_csv_atomic(output / "optimization_steps.csv", all_steps)
    _save_json_atomic(output / "summary.json", {
        "protocol_version": PROTOCOL_VERSION, "signature": signature,
        "tasks_expected": len(tasks), "tasks_completed": len(completed),
        "task_rows": len(all_results), "generation_rows": len(all_generations),
        "step_rows": len(all_steps), "modes": list(args.modes),
    })


if __name__ == "__main__":
    main()
