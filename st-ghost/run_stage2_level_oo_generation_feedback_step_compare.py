"""Compare generation-feedback losses after 0, 5, 10, 15, 20, 25 updates.

Forty successful terminal embeddings are sampled deterministically: eight per
object pair, spread across archive seeds 42/43/44. Both optimization modes use
the same source image and one fixed generation state (``generation_index=0``)
for every checkpoint. No images, victim answers, or detector outputs are saved
in this experiment.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import run_stage2_level_oo_generation_feedback as feedback


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "stage2_level_oo_generation_feedback_step_compare_v4"
DEFAULT_OUTPUT = ROOT / "outputs" / PROTOCOL
DEFAULT_STEPS = (0, 5, 10, 15, 20, 25)
MODES = ("target_recovery", "full")
SEED_QUOTAS = {42: 3, 43: 3, 44: 2}


def select_tasks(tasks: list[dict], selection_seed: int) -> list[dict]:
    grouped = defaultdict(list)
    for task in tasks:
        grouped[(str(task["spec"]), int(task["seed"]))].append(task)
    rng = random.Random(selection_seed)
    chosen = []
    for spec in sorted({str(task["spec"]) for task in tasks}):
        used_images = set()
        for seed, quota in SEED_QUOTAS.items():
            candidates = sorted(grouped[(spec, seed)], key=lambda row: int(row["image_id"]))
            rng.shuffle(candidates)
            selected = []
            for task in candidates:
                if int(task["image_id"]) not in used_images:
                    selected.append(task)
                    used_images.add(int(task["image_id"]))
                if len(selected) == quota:
                    break
            if len(selected) != quota:
                raise RuntimeError(f"Insufficient distinct images for {spec}, seed {seed}")
            chosen.extend(selected)
    if len(chosen) != 40:
        raise RuntimeError(f"Expected 40 embeddings, selected {len(chosen)}")
    return chosen


def task_descriptor(task: dict) -> dict:
    return {key: task[key] for key in (
        "seed", "spec", "image_id", "source_path", "embedding_path",
        "embedding_sha256", "terminal_step", "trigger_prompt")}


def target_terms(z_hat: torch.Tensor, z_star: torch.Tensor, cfg: dict) -> dict[str, torch.Tensor]:
    """Return the directional primary target and diagnostic legacy error."""
    z_hat = z_hat.float()
    z_star = z_star.float()
    target_cosine = F.cosine_similarity(z_hat, z_star, dim=-1).mean()
    direction_loss = 1.0 - target_cosine
    target_norm = z_star.norm(dim=-1).clamp_min(1e-8)
    norm_loss = ((z_hat.norm(dim=-1) - target_norm) / target_norm).square().mean()
    relative_squared_error = ((z_hat - z_star).square().sum()
                              / (z_star.square().sum() + 1e-8))
    target_loss = (float(cfg["lambda_direction"]) * direction_loss
                   + float(cfg["lambda_norm"]) * norm_loss)
    return {
        "target_loss": target_loss,
        "target_direction_loss": direction_loss,
        "target_norm_loss": norm_loss,
        "target_relative_squared_error": relative_squared_error,
        "target_cosine": target_cosine,
    }


def evaluate(runtime, state, condition, z_star, z0, cfg, task, mode, step):
    with torch.no_grad():
        image = feedback._render(runtime, state, condition, cfg)
        if not torch.isfinite(image).all().item():
            raise RuntimeError(f"Non-finite image: {task['spec']} {task['image_id']} {mode} step {step}")
        z_hat = feedback._clip_image(runtime, image)
        metrics = feedback._feature_metrics(
            z_hat, z_star, z0, cfg["clean_threshold"], cfg["lambda_clean"])
        target = target_terms(z_hat, z_star, cfg)
        metrics.update({name: float(value.detach().cpu()) for name, value in target.items()})
        metrics["total_loss"] = metrics["target_loss"] + (
            float(cfg["lambda_clean"]) * metrics["clean_loss"] if mode == "full" else 0.0)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"Non-finite metrics: {task['spec']} {task['image_id']} {mode} step {step}")
    return [{
        "seed": task["seed"], "spec": task["spec"], "image_id": task["image_id"],
        "mode": mode, "updates_completed": step, "generation_index": 0,
        "generation_seed": feedback.exp4.generation_seed(
            int(task["seed"]), str(task["spec"]), int(task["image_id"]), 0),
        **metrics,
    }]


def run_task(runtime, task, cfg, checkpoint_steps):
    with Image.open(str(task["source_path"])) as handle:
        source = handle.convert("RGB")
    z_star = task["embedding"].to(runtime.device)
    with torch.no_grad():
        z0 = runtime.clip.images([source]).detach().float()
    prompt = str(task["trigger_prompt"]).strip()
    if not prompt:
        spec = next(row for row in cfg["target_specs"] if row["name"] == task["spec"])
        prompt = feedback.oo.oo_optimization_questions(
            feedback.exp4.CompositeSpec.from_dict(spec))[0]
    state = feedback._make_state(
        runtime, source,
        feedback.exp4.generation_seed(int(task["seed"]), str(task["spec"]), int(task["image_id"]), 0),
        prompt, cfg)
    measured = []
    trajectory = []
    for mode in MODES:
        condition = torch.nn.Parameter(z_star.clone())
        optimizer = torch.optim.Adam(
            [condition], lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        for completed in range(checkpoint_steps[-1] + 1):
            if completed in checkpoint_steps:
                measured.extend(evaluate(runtime, state, condition.detach(), z_star, z0,
                                         cfg, task, mode, completed))
            if completed == checkpoint_steps[-1]:
                break
            image = feedback._render(runtime, state, condition, cfg)
            z_hat = feedback._clip_image(runtime, image)
            target = target_terms(z_hat, z_star, cfg)
            target_loss = target["target_loss"]
            clean_cos = F.cosine_similarity(z_hat.float(), z0, dim=-1).mean()
            clean_loss = torch.relu(cfg["clean_threshold"] - clean_cos).square()
            total = target_loss + (cfg["lambda_clean"] * clean_loss if mode == "full" else 0.0)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if condition.grad is None or not torch.isfinite(condition.grad).all().item():
                raise RuntimeError(f"Non-finite condition gradient: {task['spec']} {task['image_id']} {mode} step {completed}")
            grad_norm = float(condition.grad.norm().item())
            if grad_norm == 0:
                raise RuntimeError(f"Zero condition gradient: {task['spec']} {task['image_id']} {mode} step {completed}")
            optimizer.step()
            trajectory.append({
                "seed": task["seed"], "spec": task["spec"], "image_id": task["image_id"],
                "mode": mode, "updates_completed": completed,
                "target_loss_before_update": float(target_loss.detach().item()),
                "target_direction_loss_before_update": float(
                    target["target_direction_loss"].detach().item()),
                "target_norm_loss_before_update": float(target["target_norm_loss"].detach().item()),
                "target_relative_squared_error_before_update": float(
                    target["target_relative_squared_error"].detach().item()),
                "target_cosine_before_update": float(target["target_cosine"].detach().item()),
                "clean_loss_before_update": float(clean_loss.detach().item()),
                "clean_cosine_before_update": float(clean_cos.detach().item()),
                "gradient_norm": grad_norm,
            })
        del condition, optimizer
        torch.cuda.empty_cache()
    return measured, trajectory


def aggregate(rows, group_keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for group, members in sorted(groups.items()):
        output.append({**dict(zip(group_keys, group)), "generation_count": len(members),
                       **{f"mean_{metric}": sum(row[metric] for row in members) / len(members)
                          for metric in (
                              "target_loss", "target_direction_loss", "target_norm_loss",
                              "target_relative_squared_error", "clean_loss", "total_loss",
                              "target_cosine", "clean_cosine")}})
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-archive", type=Path, default=feedback.ARCHIVE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N selected tasks for a smoke test")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if sorted(set(args.steps)) != args.steps or args.steps[0] != 0:
        parser.error("--steps must start at 0 and increase without duplicates")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    return args


def main():
    args = parse_args()
    archive = args.embedding_archive.resolve()
    cfg_source, _, specs, archive_tasks = feedback._archive_tasks(archive, only_success=True)
    selected = select_tasks(archive_tasks, args.selection_seed)
    if args.limit:
        selected = selected[:args.limit]
    cfg = dict(cfg_source)
    cfg.update({"protocol_version": PROTOCOL, "embedding_archive": str(archive),
                "vae_image_contract": feedback.VAE_IMAGE_CONTRACT,
                "output_dir": str(args.output_dir.resolve()), "learning_rate": 0.02,
                "optimizer": "Adam", "weight_decay": 0.0,
                "target_loss": "direction_plus_relative_norm",
                "lambda_direction": 1.0, "lambda_norm": 0.1,
                "lambda_clean": 1.0, "clean_threshold": 0.85,
                "diffusion_compute_dtype": "float32", "checkpoint_steps": args.steps,
                "selection_seed": args.selection_seed, "modes": MODES,
                "num_generations": 1, "optimization_generation_index": 0})
    signature = hashlib.sha256(json.dumps({
        "protocol": PROTOCOL, "selected": [task_descriptor(task) for task in selected],
        "vae_image_contract": feedback.VAE_IMAGE_CONTRACT,
        "checkpoint_steps": args.steps, "learning_rate": cfg["learning_rate"],
        "optimizer": cfg["optimizer"], "weight_decay": cfg["weight_decay"],
        "target_loss": cfg["target_loss"], "lambda_direction": cfg["lambda_direction"],
        "lambda_norm": cfg["lambda_norm"],
        "lambda_clean": cfg["lambda_clean"], "clean_threshold": cfg["clean_threshold"],
        "num_generations": cfg["num_generations"], "num_inference_steps": cfg["num_inference_steps"],
        "diffusion_t": cfg["diffusion_t"], "guidance_scale": cfg["guidance_scale"],
        "diffusion_compute_dtype": cfg["diffusion_compute_dtype"],
    }, sort_keys=True).encode()).hexdigest()
    print(json.dumps({"protocol": PROTOCOL, "selected_embeddings": len(selected),
                      "checkpoint_steps": args.steps, "modes": MODES,
                      "generations_per_checkpoint": cfg["num_generations"],
                      "optimizer": cfg["optimizer"], "learning_rate": cfg["learning_rate"],
                      "weight_decay": cfg["weight_decay"],
                      "target_loss": cfg["target_loss"],
                      "lambda_direction": cfg["lambda_direction"],
                      "lambda_norm": cfg["lambda_norm"],
                      "signature": signature, "validate_only": args.validate_only}, indent=2), flush=True)
    if args.validate_only:
        return
    output = args.output_dir.resolve()
    signature_path = output / "experiment_signature.json"
    if signature_path.exists():
        if json.loads(signature_path.read_text()).get("sha256") != signature:
            raise RuntimeError(f"Output directory belongs to another experiment: {output}")
    output.mkdir(parents=True, exist_ok=True)
    feedback._save_json_atomic(signature_path, {"protocol": PROTOCOL, "sha256": signature})
    feedback._save_json_atomic(output / "resolved_config.json", cfg)
    feedback._save_json_atomic(output / "selected_embeddings.json", [task_descriptor(task) for task in selected])
    feedback._save_json_atomic(output / "target_specs.json", specs)
    runtime = feedback.exp4.Runtime(cfg, attack=False)
    runtime.load_diffusion()
    runtime.pipe.unet.to(dtype=torch.float32).requires_grad_(False)
    runtime.pipe.vae.to(dtype=torch.float32).requires_grad_(False)
    runtime.pipe.text_encoder.requires_grad_(False)
    all_measured = []
    all_trajectory = []
    for task in tqdm(selected, desc="Step comparison"):
        key = f"{feedback._safe_name(task['spec'])}__{task['image_id']}__{feedback.LEVEL}.json"
        checkpoint = output / f"seed_{task['seed']}" / "checkpoints" / key
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved.get("experiment_signature") != signature:
                raise RuntimeError(f"Checkpoint signature mismatch: {checkpoint}")
            measured, trajectory = saved["measurements"], saved["trajectory"]
        else:
            measured, trajectory = run_task(runtime, task, cfg, args.steps)
            feedback._save_json_atomic(checkpoint, {
                "experiment_signature": signature, "task": task_descriptor(task),
                "measurements": measured, "trajectory": trajectory})
        all_measured.extend(measured)
        all_trajectory.extend(trajectory)
        feedback._save_csv_atomic(output / "generation_losses.csv", all_measured)
        feedback._save_csv_atomic(output / "task_step_means.csv", aggregate(
            all_measured, ("seed", "spec", "image_id", "mode", "updates_completed")))
        feedback._save_csv_atomic(output / "group_step_means.csv", aggregate(
            all_measured, ("spec", "mode", "updates_completed")))
        feedback._save_csv_atomic(output / "optimization_trajectory.csv", all_trajectory)
        feedback._save_json_atomic(output / "progress.json", {
            "tasks_expected": len(selected),
            "tasks_completed": len(all_measured) // (len(args.steps) * len(MODES) * cfg["num_generations"]),
            "measurements": len(all_measured), "trajectory_rows": len(all_trajectory),
            "checkpoint_steps": args.steps})
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
