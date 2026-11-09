"""Level-OO joint-update loss diagnostic.

This runner uses the exact source-image pool and optimization settings from
the completed Level-OO experiment.  It follows the original GHOST loop until
the embedding reaches the Yes-probability threshold, then generates images
and stops.  No generated-image query, OWLv2 check, or other detector is run.

For every task it stores the initial and terminal values of three losses:

* ``L_a``: the atomic query for object A;
* ``L_b``: the atomic query for object B;
* ``L_oo``: the fixed canonical two-object query.

Only the sampled two-object attack query is optimized.  The three fixed
diagnostic losses are observations, so their changes reveal whether the
original joint update improves both atomic targets, favors one, or improves
neither.  ``image_generation_completed`` means that the diffusion pipeline
returned and the image was saved; it is deliberately not a semantic attack
success label because this protocol performs no post-generation verification.

Run from the repository root::

    python st-ghost/run_stage2_level_oo_loss_diagnostic.py \
        --source-output outputs/stage1_exp4_level_oo_v1 \
        --output-dir outputs/stage2_level_oo_loss_diagnostic_v1
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "st-ghost"
if str(STAGE1) not in sys.path:
    sys.path.insert(0, str(STAGE1))

import run_stage1_exp4 as exp4  # noqa: E402
import run_stage1_exp4_level_oo as oo  # noqa: E402


LEVEL = oo.LEVEL
PROTOCOL_VERSION = "stage2_level_oo_loss_diagnostic_v1"
SOURCE_OUTPUT_DEFAULT = ROOT / "outputs" / "stage1_exp4_level_oo_v1"
OUTPUT_DEFAULT = ROOT / "outputs" / "stage2_level_oo_loss_diagnostic_v1"
LOSS_NAMES = ("a", "b", "oo")
LOSS_EPSILON = 1e-8
IMPROVEMENT_EPSILON = 1e-7


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _as_float(value: object, default: float = float("nan")) -> float:
    if value is None or value == "" or str(value).lower() in {"nan", "none"}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _loss(probability: torch.Tensor) -> torch.Tensor:
    return -torch.log(probability.clamp_min(LOSS_EPSILON))


def _metric_questions(spec: exp4.CompositeSpec) -> Dict[str, str]:
    atomic = oo.oo_atomic_questions(spec)
    a, b = spec.objects
    return {
        "a": atomic[f"obj:{a}"],
        "b": atomic[f"obj:{b}"],
        "oo": oo.oo_canonical_question(spec),
    }


@torch.no_grad()
def _metric_losses(runtime: exp4.Runtime, source: Image.Image,
                   embedding: torch.Tensor,
                   questions: Mapping[str, str]) -> Dict[str, float]:
    """Evaluate all three fixed diagnostic losses at one embedding state."""
    tokens = runtime.projector.model(embedding).half().squeeze(0)
    values: Dict[str, float] = {}
    for name in LOSS_NAMES:
        probability = runtime.victim.projected_probability(
            source, tokens, questions[name])
        values[name] = float(_loss(probability).detach().cpu())
    return values


def _relative_improvement(initial: float, final: float) -> float:
    if not math.isfinite(initial) or not math.isfinite(final):
        return float("nan")
    return (initial - final) / max(abs(initial), LOSS_EPSILON)


def _directional_pattern(improvement_a: float, improvement_b: float) -> str:
    """Classify the two atomic loss changes for compact group summaries."""
    a_good = improvement_a > IMPROVEMENT_EPSILON
    b_good = improvement_b > IMPROVEMENT_EPSILON
    a_bad = improvement_a < -IMPROVEMENT_EPSILON
    b_bad = improvement_b < -IMPROVEMENT_EPSILON
    if a_good and b_good:
        return "both_improved"
    if a_good and b_bad:
        return "favors_a_sacrifices_b"
    if b_good and a_bad:
        return "favors_b_sacrifices_a"
    if a_good and not b_good:
        return "favors_a"
    if b_good and not a_good:
        return "favors_b"
    if a_bad and b_bad:
        return "both_worsened"
    return "both_weak_or_unchanged"


def validate_source_output(source_output: Path) -> Tuple[dict, List[dict],
                                                            List[exp4.CompositeSpec]]:
    """Load the already selected Level-OO pool without re-filtering images."""
    cfg_path = source_output / "resolved_config.json"
    manifest_path = source_output / "level_oo_shared_pool_manifest.json"
    if not cfg_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "The Level-OO source output must contain resolved_config.json and "
            "level_oo_shared_pool_manifest.json")
    cfg = json.loads(cfg_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if cfg.get("protocol_version") != oo.PROTOCOL_VERSION:
        raise RuntimeError(
            "The source output was not produced by the completed Level-OO "
            f"protocol: {cfg.get('protocol_version')!r}")
    if list(cfg.get("levels", ())) != [LEVEL]:
        raise RuntimeError(
            f"Expected the completed Level-OO source output, got levels="
            f"{cfg.get('levels')!r}")
    specs = [exp4.CompositeSpec.from_dict(value)
             for value in cfg["target_specs"]]
    expected = len(specs) * int(cfg["num_images_per_spec"])
    if len(manifest) != expected:
        raise RuntimeError(
            f"Expected {expected} selected Level-OO rows, got {len(manifest)}")
    required_by_spec = {spec.name: int(cfg["num_images_per_spec"])
                        for spec in specs}
    seen = set()
    for row in manifest:
        spec_name = str(row.get("spec", ""))
        key = (spec_name, _as_int(row.get("image_id"), -1))
        if spec_name not in required_by_spec or key in seen:
            raise RuntimeError(f"Invalid or duplicate source-pool row: {key}")
        seen.add(key)
        if row.get("level") != LEVEL:
            raise RuntimeError(f"Unexpected source level in row {key}")
        source_path = Path(str(row.get("source_path", "")))
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        required_by_spec[spec_name] -= 1
    if any(value != 0 for value in required_by_spec.values()):
        raise RuntimeError(f"Uneven source pool: {required_by_spec}")
    return cfg, manifest, specs


def load_config(source_cfg: Mapping[str, object], source_output: Path,
                output: Path, resume: bool) -> dict:
    cfg = dict(source_cfg)
    cfg["output_dir"] = str(output.resolve())
    cfg["source_output_dir"] = str(source_output.resolve())
    cfg["levels"] = [LEVEL]
    cfg["resume"] = bool(resume)
    cfg["protocol_version"] = PROTOCOL_VERSION
    cfg["generation_evaluation"] = "none"
    cfg["detector_evaluation"] = False
    cfg["diagnostic_loss_names"] = {
        "a": "atomic object A",
        "b": "atomic object B",
        "oo": "canonical joint object query",
    }
    return cfg


def experiment_signature(cfg: Mapping[str, object],
                         manifest: Sequence[Mapping[str, object]]) -> str:
    relevant = {
        "protocol_version": PROTOCOL_VERSION,
        "source_output_dir": cfg["source_output_dir"],
        "levels": [LEVEL],
        "target_specs": cfg["target_specs"],
        "manifest": list(manifest),
        "level_oo_prompts": list(oo.OO_PROMPTS),
        "num_steps": cfg["num_steps"],
        "learning_rate": cfg["learning_rate"],
        "lambda_clip": cfg["lambda_clip"],
        "lambda_reg": cfg["lambda_reg"],
        "yes_threshold": cfg["yes_threshold"],
        "num_generations": cfg["num_generations"],
        "diffusion_t": cfg["diffusion_t"],
        "noise_level": cfg.get("noise_level", 0),
        "num_inference_steps": cfg["num_inference_steps"],
        "guidance_scale": cfg["guidance_scale"],
        "diffusion_prompt_mode": cfg["diffusion_prompt_mode"],
        "detector_evaluation": False,
    }
    payload = json.dumps(relevant, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def generate_images_only(runtime: exp4.Runtime, source: Image.Image,
                         embedding: torch.Tensor, spec: exp4.CompositeSpec,
                         seed: int, image_id: int, trigger_step: int,
                         trigger_prompt: str, cfg: Mapping[str, object],
                         output: Path, metric_values: Mapping[str, float]) \
        -> List[dict]:
    """Run unCLIP generation only; no victim query or detector is called."""
    runtime.load_diffusion()
    rows: List[dict] = []
    count = int(cfg["num_generations"])
    directory = output / "generated" / spec.name / LEVEL
    directory.mkdir(parents=True, exist_ok=True)
    for generation_index in range(count):
        sample_seed = exp4.generation_seed(
            seed, spec.name, image_id, generation_index)
        generated_path = directory / (
            f"seed{seed}_{image_id}_step{trigger_step}_g{generation_index}.png")
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
            vae = exp4.get_vae_features(
                [source], runtime.pipe, runtime.diffusion_preprocess)
            generated = exp4.generate_image(
                runtime.pipe, vae, embedding.half(),
                t=int(cfg["diffusion_t"]),
                num_inference_steps=int(cfg["num_inference_steps"]),
                noise_level=int(cfg.get("noise_level", 0)),
                prompt=trigger_prompt,
                negative_prompt="low quality, ugly, unrealistic",
                guidance_scale=float(cfg["guidance_scale"]),
            )
            del vae
        generated.save(generated_path)
        rows.append({
            "seed": seed,
            "spec": spec.name,
            "image_id": image_id,
            "level": LEVEL,
            "generation_index": generation_index,
            "generation_seed": sample_seed,
            "trigger_step": trigger_step,
            "trigger_prompt": trigger_prompt,
            "diffusion_prompt": trigger_prompt,
            "generated_path": str(generated_path),
            "image_generation_completed": 1,
            **{f"terminal_loss_{name}": float(metric_values[name])
               for name in LOSS_NAMES},
        })
    return rows


def optimize_task(runtime: exp4.Runtime, source: Image.Image,
                  spec: exp4.CompositeSpec, seed: int, image_id: int,
                  cfg: Mapping[str, object], contrast: torch.Tensor,
                  output: Path) -> Tuple[dict, List[dict], List[dict]]:
    with torch.no_grad():
        initial = runtime.clip.images([source]).detach()
    embedding = nn.Parameter(initial.clone())
    optimizer = torch.optim.AdamW(
        [embedding], lr=float(cfg["learning_rate"]))
    questions = _metric_questions(spec)
    initial_losses = _metric_losses(runtime, source, embedding, questions)
    prompts = oo.oo_optimization_questions(spec)
    prompt_rng = random.Random(exp4.stable_seed(
        "stage1-exp4-prompt", seed, spec.name, image_id))
    threshold = float(cfg["yes_threshold"])
    traces: List[dict] = []
    embedding_success = False
    trigger_step, trigger_prompt = -1, ""
    terminal_step, terminal_prompt = -1, ""
    terminal_attack_loss = float("nan")
    terminal_yes_after = float("nan")
    terminal_losses: Dict[str, float] = dict(initial_losses)

    for step in range(int(cfg["num_steps"])):
        prompt = prompts[prompt_rng.randrange(len(prompts))]
        tokens = runtime.projector.model(embedding).half().squeeze(0)
        probability_before = runtime.victim.projected_probability(
            source, tokens, prompt)
        attack_loss = _loss(probability_before)
        clip_loss = exp4.contrast_loss(embedding, contrast)
        regularizer = F.mse_loss(embedding.float(), initial.float())
        total_loss = (attack_loss + float(cfg["lambda_clip"]) * clip_loss
                      + float(cfg["lambda_reg"]) * regularizer)
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            updated_tokens = runtime.projector.model(embedding).half().squeeze(0)
            probability_after = runtime.victim.projected_probability(
                source, updated_tokens, prompt)
            yes_after = float(probability_after.detach().cpu())
        threshold_reached = yes_after >= threshold
        terminal_step = step
        terminal_prompt = prompt
        terminal_attack_loss = float(_loss(probability_after).detach().cpu())
        terminal_yes_after = yes_after
        traces.append({
            "seed": seed,
            "spec": spec.name,
            "image_id": image_id,
            "level": LEVEL,
            "step": step,
            "prompt": prompt,
            "loss": float(total_loss.detach().cpu()),
            "attack_loss": float(attack_loss.detach().cpu()),
            "clip_loss": float(clip_loss.detach().cpu()),
            "reg_loss": float(regularizer.detach().cpu()),
            "yes_before": float(probability_before.detach().cpu()),
            "yes_after": yes_after,
            "threshold_reached": int(threshold_reached),
        })
        if threshold_reached:
            embedding_success = True
            trigger_step, trigger_prompt = step, prompt
            break

    terminal_losses = _metric_losses(runtime, source, embedding, questions)
    del optimizer

    delta = {name: terminal_losses[name] - initial_losses[name]
             for name in LOSS_NAMES}
    improvement = {name: initial_losses[name] - terminal_losses[name]
                   for name in LOSS_NAMES}
    relative = {name: _relative_improvement(
        initial_losses[name], terminal_losses[name]) for name in LOSS_NAMES}
    pattern = _directional_pattern(improvement["a"], improvement["b"])

    generations: List[dict] = []
    if embedding_success:
        generations = generate_images_only(
            runtime, source, embedding.detach(), spec, seed, image_id,
            trigger_step, trigger_prompt, cfg, output, terminal_losses)
    image_generation_completed = int(
        len(generations) == int(cfg["num_generations"]))
    result = {
        "seed": seed,
        "spec": spec.name,
        "image_id": image_id,
        "level": LEVEL,
        "embedding_success": int(embedding_success),
        "termination_reason": "generated_after_threshold" if embedding_success
        else "max_steps_without_threshold",
        "num_optimization_steps": terminal_step + 1,
        "trigger_step": trigger_step,
        "trigger_prompt": trigger_prompt,
        "terminal_step": terminal_step,
        "terminal_prompt": terminal_prompt,
        "terminal_yes_after": terminal_yes_after,
        "terminal_attack_loss": terminal_attack_loss,
        "image_generation_completed": image_generation_completed,
        "num_generations_generated": len(generations),
        "generation_evaluation": "none",
        "detector_evaluation": 0,
        "directional_pattern": pattern,
        "atomic_mean_improvement": (improvement["a"] + improvement["b"]) / 2.0,
        "atomic_min_improvement": min(improvement["a"], improvement["b"]),
        "atomic_improvement_gap": improvement["a"] - improvement["b"],
    }
    for name in LOSS_NAMES:
        result[f"question_{name}"] = questions[name]
        result[f"initial_loss_{name}"] = initial_losses[name]
        result[f"terminal_loss_{name}"] = terminal_losses[name]
        result[f"initial_yes_probability_{name}"] = math.exp(
            -initial_losses[name])
        result[f"terminal_yes_probability_{name}"] = math.exp(
            -terminal_losses[name])
        result[f"delta_loss_{name}"] = delta[name]
        result[f"improvement_{name}"] = improvement[name]
        result[f"relative_improvement_{name}"] = relative[name]
    return result, traces, generations


def _checkpoint_path(seed_dir: Path, spec: str, image_id: int) -> Path:
    return seed_dir / "checkpoints" / f"{_safe_name(spec)}__{image_id}__{LEVEL}.json"


def _load_checkpoints(seed_dir: Path, signature: str) -> Tuple[List[dict],
                                                                 List[dict],
                                                                 List[dict]]:
    results: List[dict] = []
    traces: List[dict] = []
    generations: List[dict] = []
    directory = seed_dir / "checkpoints"
    if not directory.exists():
        return results, traces, generations
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("experiment_signature") != signature:
            raise RuntimeError(f"Checkpoint signature mismatch: {path}")
        results.append(payload["result"])
        traces.extend(payload.get("traces", []))
        generations.extend(payload.get("generations", []))
    return results, traces, generations


def run_seed(runtime: exp4.Runtime, dset, manifest: Sequence[dict],
             specs: Sequence[exp4.CompositeSpec], seed: int,
             cfg: Mapping[str, object], output: Path,
             signature: str) -> Tuple[List[dict], List[dict], List[dict]]:
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("resume", True):
        results, traces, generations = _load_checkpoints(seed_dir, signature)
    else:
        results, traces, generations = [], [], []
    completed = {(str(row.get("spec")), _as_int(row.get("image_id")), LEVEL)
                 for row in results}
    spec_by_name = {spec.name: spec for spec in specs}
    contrasts = {spec.name: oo.oo_contrast_embedding(runtime, spec)
                 for spec in specs}
    jobs = [(spec_by_name[str(item["spec"])], item) for item in manifest]
    with tqdm(total=len(jobs), desc=f"Level-OO loss diagnostic seed {seed}") as progress:
        for spec, item in jobs:
            image_id = _as_int(item["image_id"])
            key = (spec.name, image_id, LEVEL)
            if key in completed:
                progress.update(1)
                continue
            image, source_path = dset[image_id]
            expected_path = Path(str(item["source_path"])).resolve()
            if Path(str(source_path)).resolve() != expected_path:
                raise RuntimeError(
                    f"Source image changed for {spec.name}/{image_id}")
            result, new_traces, new_generations = optimize_task(
                runtime, image, spec, seed, image_id, cfg,
                contrasts[spec.name], seed_dir)
            result["source_path"] = str(source_path)
            result["pool_index"] = _as_int(item.get("pool_index"))
            result["clip_base_score"] = _as_float(item.get("base_score"))
            results.append(result)
            traces.extend(new_traces)
            generations.extend(new_generations)
            exp4.save_json_atomic(_checkpoint_path(seed_dir, spec.name, image_id), {
                "experiment_signature": signature,
                "result": result,
                "traces": new_traces,
                "generations": new_generations,
            })
            completed.add(key)
            if len(results) % int(cfg["checkpoint_every"]) == 0:
                exp4.save_csv_atomic(seed_dir / "steps.csv", traces)
                exp4.save_csv_atomic(seed_dir / "generations.csv", generations)
                exp4.save_csv_atomic(seed_dir / "results.csv", results)
            gc.collect()
            torch.cuda.empty_cache()
            progress.update(1)
    exp4.save_csv_atomic(seed_dir / "steps.csv", traces)
    exp4.save_csv_atomic(seed_dir / "generations.csv", generations)
    exp4.save_csv_atomic(seed_dir / "results.csv", results)
    return results, traces, generations


def _finite_values(rows: Sequence[Mapping[str, object]], key: str) -> np.ndarray:
    values = np.asarray([_as_float(row.get(key)) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _group_summary(rows: Sequence[Mapping[str, object]], group: str,
                   spec_name: str) -> dict:
    result = {"spec": spec_name, "group": group, "n": len(rows)}
    for name in LOSS_NAMES:
        for prefix in ("initial_loss", "terminal_loss", "delta_loss",
                       "improvement", "relative_improvement"):
            values = _finite_values(rows, f"{prefix}_{name}")
            result[f"{prefix}_{name}_mean"] = float(values.mean()) if len(values) else float("nan")
            result[f"{prefix}_{name}_median"] = float(np.median(values)) if len(values) else float("nan")
            result[f"{prefix}_{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    for key in ("embedding_success", "image_generation_completed"):
        values = _finite_values(rows, key)
        result[f"{key}_rate"] = float(values.mean()) if len(values) else float("nan")
    patterns = (
        "both_improved", "favors_a_sacrifices_b", "favors_b_sacrifices_a",
        "favors_a", "favors_b", "both_worsened", "both_weak_or_unchanged",
    )
    for pattern in patterns:
        result[f"pattern_{pattern}_rate"] = (
            sum(row.get("directional_pattern") == pattern for row in rows)
            / len(rows) if rows else float("nan"))
    return result


def build_group_summary(results: Sequence[dict],
                        specs: Sequence[exp4.CompositeSpec]) -> List[dict]:
    rows: List[dict] = []
    for spec_name in [spec.name for spec in specs] + ["ALL"]:
        base = [row for row in results
                if spec_name == "ALL" or row.get("spec") == spec_name]
        groups = {
            "all": base,
            "embedding_success": [row for row in base
                                   if _as_int(row.get("embedding_success"))],
            "embedding_failure": [row for row in base
                                   if not _as_int(row.get("embedding_success"))],
            "embedding_success_generation_completed": [
                row for row in base
                if _as_int(row.get("embedding_success"))
                and _as_int(row.get("image_generation_completed"))],
        }
        for group, group_rows in groups.items():
            rows.append(_group_summary(group_rows, group, spec_name))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path,
                        default=SOURCE_OUTPUT_DEFAULT,
                        help="Completed Level-OO output containing the selected pool")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not load diagnostic checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_output = args.source_output.resolve()
    output = args.output_dir.resolve()
    source_cfg, manifest, specs = validate_source_output(source_output)
    cfg = load_config(source_cfg, source_output, output, not args.no_resume)
    output.mkdir(parents=True, exist_ok=True)
    signature = experiment_signature(cfg, manifest)
    signature_path = output / "experiment_signature.json"
    if signature_path.exists():
        existing = json.loads(signature_path.read_text())
        if existing.get("sha256") != signature:
            raise RuntimeError(
                f"Output directory {output} belongs to another diagnostic run")
    exp4.save_json(output / "resolved_config.json", cfg)
    exp4.save_json(output / "source_manifest_reference.json", {
        "source_output_dir": str(source_output),
        "source_manifest": str(source_output / "level_oo_shared_pool_manifest.json"),
        "source_row_count": len(manifest),
        "pool_reselected": False,
        "post_generation_evaluation": False,
    })
    exp4.save_json(output / "experiment_signature.json", {
        "protocol_version": PROTOCOL_VERSION,
        "sha256": signature,
    })
    exp4.save_json(output / "target_specs.json",
                   [spec.to_dict() for spec in specs])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Level-OO loss diagnostic")
    runtime = exp4.Runtime(cfg, attack=True)
    dset = exp4.load_coco_dataset(cfg["data_path"], split="train")
    all_results: List[dict] = []
    all_traces: List[dict] = []
    all_generations: List[dict] = []
    for seed in cfg["seeds"]:
        exp4.seed_all(seed)
        results, traces, generations = run_seed(
            runtime, dset, manifest, specs, seed, cfg, output, signature)
        all_results.extend(results)
        all_traces.extend(traces)
        all_generations.extend(generations)
    exp4.save_csv(output / "all_results.csv", all_results)
    exp4.save_csv(output / "all_steps.csv", all_traces)
    exp4.save_csv(output / "all_generations.csv", all_generations)
    group_summary = build_group_summary(all_results, specs)
    exp4.save_csv(output / "loss_group_summary.csv", group_summary)
    exp4.save_json(output / "loss_group_summary.json", group_summary)


if __name__ == "__main__":
    main()
