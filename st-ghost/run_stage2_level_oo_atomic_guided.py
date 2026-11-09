"""Stage 2 Level-OO optimization with weaker-atomic-target guidance.

This experiment keeps the completed Level-OO source pool and all of its
optimization/generation settings, but evaluates only ``horse_bus`` and
``cow_motorcycle`` by default.  The Stage 2 ``atomic_guided_all`` runner
extends this experiment to all five pairs and reuses completed tasks.
In addition to the sampled joint-question loss, each
optimization step adds the two atomic object losses.  Their weights are
chosen from the current losses, so the currently weaker atomic target gets
the larger weight.

The objective is L_oo + lambda_atom * (w_a * L_a + w_b * L_b)
+ lambda_clip * L_clip + lambda_reg * L_reg, with detached weights
w_i = L_i / (L_a + L_b), or equal weights if both losses are zero.
lambda_atom defaults to 1.0.  The atomic queries are the same fixed queries
used by the Stage 2 diagnostic; the joint query keeps its original schedule.

The script writes a separate Stage 2 protocol/output and performs no
post-generation detector or semantic evaluation.  It is intentionally a
runner only; invoking it is required to start the experiment.

Run from the repository root::

    .venv-stage1/bin/python st-ghost/run_stage2_level_oo_atomic_guided.py \
        --source-output outputs/stage1_exp4_level_oo_v1 \
        --output-dir outputs/stage2_level_oo_atomic_guided_v1
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

STAGE1 = Path(__file__).resolve().parent
if str(STAGE1) not in sys.path:
    sys.path.insert(0, str(STAGE1))

import run_stage1 as stage1
import run_stage2_level_oo_loss_diagnostic as base


LEVEL = base.LEVEL
PROTOCOL_VERSION = "stage2_level_oo_atomic_guided_v1"
OUTPUT_DEFAULT = base.ROOT / "outputs" / PROTOCOL_VERSION
SOURCE_OUTPUT_DEFAULT = base.SOURCE_OUTPUT_DEFAULT
GUIDED_SPECS = ("horse_bus", "cow_motorcycle")
ALL_GUIDED_SPECS = ("dog_bicycle", "cat_chair", "horse_bus",
                    "cow_motorcycle", "elephant_truck")
DEFAULT_LAMBDA_ATOM = 1.0


class GuidedRuntime(stage1.Runtime):
    """Reuse the original model loaders and diffusion without loading OWLv2."""

    def __init__(self, cfg: dict):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the atomic-guided experiment")
        self.cfg = cfg
        self.device = torch.device("cuda")
        self.clip = stage1.ClipRuntime(cfg, self.device)
        self.victim = stage1.VictimRuntime(cfg, self.device)
        self.projector = stage1.ProjectorRuntime(cfg, self.device)
        self.absence = None
        self.pipe = None
        self.diffusion_preprocess = None


def validate_source_output(source_output: Path, spec_names=GUIDED_SPECS):
    """Validate the complete source run, then retain the requested pairs."""
    if (not spec_names or len(set(spec_names)) != len(spec_names)
            or not set(spec_names).issubset(ALL_GUIDED_SPECS)):
        raise ValueError("Select distinct supported Level-OO pairs")
    cfg, manifest, specs = base.validate_source_output(source_output)
    by_name = {spec.name: spec for spec in specs}
    if not all(name in by_name for name in spec_names):
        raise RuntimeError(
            "The source Level-OO output must contain target specs: "
            + ", ".join(spec_names))
    selected = [by_name[name] for name in spec_names]
    if any(spec.objects != tuple(spec.name.split("_")) for spec in selected):
        raise RuntimeError("Source spec names do not match the target objects")
    selected_names = set(spec_names)
    selected_manifest = [
        row for row in manifest if str(row.get("spec", "")) in selected_names
    ]
    expected = len(selected) * int(cfg["num_images_per_spec"])
    if len(selected_manifest) != expected:
        raise RuntimeError(
            f"Expected {expected} focus-pool rows, got {len(selected_manifest)}")
    selected_cfg = dict(cfg)
    selected_cfg["target_specs"] = [spec.to_dict() for spec in selected]
    return selected_cfg, selected_manifest, selected


def load_config(source_cfg: Mapping[str, object], source_output: Path,
                output: Path, resume: bool,
                lambda_atom: float = DEFAULT_LAMBDA_ATOM,
                spec_names=GUIDED_SPECS) -> dict:
    if not math.isfinite(lambda_atom) or lambda_atom < 0:
        raise ValueError("lambda_atom must be finite and non-negative")
    cfg = base.load_config(source_cfg, source_output, output, resume)
    cfg["target_specs"] = [
        value for value in cfg["target_specs"]
        if str(value.get("name", "")) in spec_names
    ]
    cfg["lambda_atom"] = float(lambda_atom)
    cfg["atomic_guidance_mode"] = "weaker_target_by_current_loss"
    cfg["atomic_weight_rule"] = "L_i / (L_a + L_b); equal weights if both zero"
    cfg["atomic_weights_detached"] = True
    cfg["atomic_question_template"] = base.exp4.ATOMIC_OBJECT
    cfg["reference_protocol_version"] = base.PROTOCOL_VERSION
    cfg["protocol_version"] = PROTOCOL_VERSION
    return cfg


def experiment_signature(cfg: Mapping[str, object],
                         manifest: Sequence[Mapping[str, object]]) -> str:
    payload = json.dumps({
        "config": {key: value for key, value in cfg.items()
                   if key not in ("output_dir", "resume", "checkpoint_every")},
        "manifest": list(manifest),
        "protocol_version": PROTOCOL_VERSION,
        "joint_templates": list(base.oo.OO_PROMPTS),
        "atomic_template": base.exp4.ATOMIC_OBJECT,
    }, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_weights(loss_a: torch.Tensor,
                    loss_b: torch.Tensor) -> Tuple[float, float]:
    """Give the larger weight to the larger current atomic loss."""
    a = float(loss_a.detach().cpu())
    b = float(loss_b.detach().cpu())
    if not (math.isfinite(a) and math.isfinite(b)) or a < 0 or b < 0:
        raise ValueError("Atomic weights require finite, non-negative losses")
    if a + b == 0:
        return 0.5, 0.5
    return a / (a + b), b / (a + b)


def _atomic_loss_gradient(runtime, source, embedding, question, enabled):
    # Release each victim's graph before constructing the next one.  All
    # queries still see exactly the same embedding, before the AdamW step.
    with torch.set_grad_enabled(enabled):
        tokens = runtime.projector.model(embedding).half().squeeze(0)
        probability = runtime.victim.projected_probability(source, tokens, question)
        loss = base._loss(probability)
        gradient = (torch.autograd.grad(loss, embedding)[0]
                    if enabled else torch.zeros_like(embedding))
    return loss.detach(), gradient


def _guided_step(runtime, source, embedding, initial, optimizer, questions,
                 prompt, contrast, cfg):
    """Accumulate g_oo + lambda_atom * (w_a*g_a + w_b*g_b) plus constraints."""
    coefficient = float(cfg["lambda_atom"])
    loss_a, gradient_a = _atomic_loss_gradient(
        runtime, source, embedding, questions["a"], coefficient > 0)
    loss_b, gradient_b = _atomic_loss_gradient(
        runtime, source, embedding, questions["b"], coefficient > 0)
    row = {
        "atomic_loss_a": float(loss_a.cpu()),
        "atomic_loss_b": float(loss_b.cpu()),
        "atomic_yes_before_a": math.exp(-float(loss_a.cpu())),
        "atomic_yes_before_b": math.exp(-float(loss_b.cpu())),
        "atomic_weight_a": float("nan"), "atomic_weight_b": float("nan"),
        "atomic_guidance_loss": float("nan"),
        "weighted_atomic_guidance_loss": float("nan"),
        "weaker_atomic_target": "invalid", "loss": float("nan"),
        "attack_loss": float("nan"), "clip_loss": float("nan"),
        "reg_loss": float("nan"), "yes_before": float("nan"),
        "yes_after": float("nan"), "update_applied": 0,
        "numerical_error": "",
    }
    if not all(torch.isfinite(value).all().item()
               for value in (loss_a, loss_b, gradient_a, gradient_b)):
        row["numerical_error"] = "nonfinite_atomic_loss_or_gradient"
        return row
    weight_a, weight_b = _atomic_weights(loss_a, loss_b)
    atomic_loss = weight_a * row["atomic_loss_a"] + weight_b * row["atomic_loss_b"]
    row.update({
        "atomic_weight_a": weight_a, "atomic_weight_b": weight_b,
        "atomic_guidance_loss": atomic_loss,
        "weighted_atomic_guidance_loss": coefficient * atomic_loss,
        "weaker_atomic_target": ("a" if weight_a > weight_b else
                                 "b" if weight_b > weight_a else "equal"),
    })
    tokens = runtime.projector.model(embedding).half().squeeze(0)
    probability = runtime.victim.projected_probability(source, tokens, prompt)
    attack_loss = base._loss(probability)
    clip_loss = base.exp4.contrast_loss(embedding, contrast)
    regularizer = F.mse_loss(embedding.float(), initial.float())
    joint_objective = (attack_loss + float(cfg["lambda_clip"]) * clip_loss
                       + float(cfg["lambda_reg"]) * regularizer)
    row.update({
        "loss": float(joint_objective.detach().cpu()) + coefficient * atomic_loss,
        "attack_loss": float(attack_loss.detach().cpu()),
        "clip_loss": float(clip_loss.detach().cpu()),
        "reg_loss": float(regularizer.detach().cpu()),
        "yes_before": float(probability.detach().cpu()),
    })
    if not math.isfinite(row["loss"]):
        row["numerical_error"] = "nonfinite_joint_loss"
        return row
    optimizer.zero_grad(set_to_none=True)
    joint_objective.backward()
    # The detached weights implement the stated weighted-gradient update,
    # without differentiating through the loss-based allocation rule.
    with torch.no_grad():
        embedding.grad.add_(
            weight_a * gradient_a + weight_b * gradient_b, alpha=coefficient)
    if not torch.isfinite(embedding.grad).all().item():
        row["numerical_error"] = "nonfinite_combined_gradient"
        return row
    optimizer.step()
    row["update_applied"] = 1
    if not torch.isfinite(embedding).all().item():
        row["numerical_error"] = "nonfinite_updated_embedding"
        return row
    with torch.no_grad():
        tokens = runtime.projector.model(embedding).half().squeeze(0)
        probability = runtime.victim.projected_probability(source, tokens, prompt)
        row["yes_after"] = float(probability.detach().cpu())
    if not math.isfinite(row["yes_after"]):
        row["numerical_error"] = "nonfinite_updated_probability"
    return row


def optimize_task(runtime, source, spec, seed, image_id, cfg, contrast, output,
                  return_embedding=False):
    """Use the original joint threshold, recording all three fixed losses."""
    with torch.no_grad():
        initial = runtime.clip.images([source]).detach()
    embedding = nn.Parameter(initial.clone())
    optimizer = torch.optim.AdamW([embedding], lr=float(cfg["learning_rate"]))
    questions = base._metric_questions(spec)
    initial_losses = base._metric_losses(runtime, source, embedding, questions)
    prompts = base.oo.oo_optimization_questions(spec)
    prompt_rng = random.Random(base.exp4.stable_seed(
        "stage1-exp4-prompt", seed, spec.name, image_id))
    threshold = float(cfg["yes_threshold"])
    traces: List[dict] = []
    embedding_success = False
    trigger_step, trigger_prompt = -1, ""
    terminal_step, terminal_prompt = -1, ""
    terminal_attack_loss = float("nan")
    terminal_yes_after = float("nan")
    numerical_error = ("" if all(math.isfinite(value)
                               for value in initial_losses.values())
                       else "nonfinite_initial_metrics")

    for step in range(int(cfg["num_steps"])):
        if numerical_error:
            break
        prompt = prompts[prompt_rng.randrange(len(prompts))]
        update = _guided_step(runtime, source, embedding, initial, optimizer,
                              questions, prompt, contrast, cfg)
        numerical_error = update["numerical_error"]
        yes_after = update["yes_after"]
        threshold_reached = not numerical_error and yes_after >= threshold
        terminal_step, terminal_prompt = step, prompt
        terminal_attack_loss = -math.log(max(yes_after, base.LOSS_EPSILON))
        terminal_yes_after = yes_after
        traces.append({
            "seed": seed, "spec": spec.name, "image_id": image_id,
            "level": LEVEL, "step": step, "prompt": prompt,
            **update,
            "threshold_reached": int(threshold_reached),
        })
        if threshold_reached:
            embedding_success = True
            trigger_step, trigger_prompt = step, prompt
            break

    terminal_losses = (
        base._metric_losses(runtime, source, embedding, questions)
        if torch.isfinite(embedding).all().item()
        else {name: float("nan") for name in base.LOSS_NAMES})
    metric_losses_finite = all(math.isfinite(value)
                               for value in (*initial_losses.values(),
                                             *terminal_losses.values()))
    if not metric_losses_finite and not numerical_error:
        numerical_error = "nonfinite_terminal_metrics"
    if numerical_error:
        embedding_success = False
    del optimizer
    delta = {name: terminal_losses[name] - initial_losses[name]
             for name in base.LOSS_NAMES}
    improvement = {name: initial_losses[name] - terminal_losses[name]
                   for name in base.LOSS_NAMES}
    relative = {name: base._relative_improvement(
        initial_losses[name], terminal_losses[name]) for name in base.LOSS_NAMES}
    pattern = (base._directional_pattern(improvement["a"], improvement["b"])
               if not numerical_error else "numerical_failure")

    generations: List[dict] = []
    if embedding_success:
        generations = base.generate_images_only(
            runtime, source, embedding.detach(), spec, seed, image_id,
            trigger_step, trigger_prompt, cfg, output, terminal_losses)
    image_generation_completed = int(
        len(generations) == int(cfg["num_generations"]))
    result = {
        "seed": seed, "spec": spec.name, "image_id": image_id,
        "level": LEVEL, "embedding_success": int(embedding_success),
        "termination_reason": ("numerical_failure" if numerical_error else
                               "generated_after_threshold" if embedding_success
                               else "max_steps_without_threshold"),
        "num_optimization_steps": sum(row["update_applied"] for row in traces),
        "num_attempted_steps": len(traces),
        "numerical_error": numerical_error,
        "metric_losses_finite": int(metric_losses_finite),
        "trigger_step": trigger_step, "trigger_prompt": trigger_prompt,
        "terminal_step": terminal_step, "terminal_prompt": terminal_prompt,
        "terminal_yes_after": terminal_yes_after,
        "terminal_attack_loss": terminal_attack_loss,
        "image_generation_completed": image_generation_completed,
        "num_generations_generated": len(generations),
        "generation_evaluation": "none", "detector_evaluation": 0,
        "directional_pattern": pattern,
        "atomic_guidance_mode": cfg["atomic_guidance_mode"],
        "lambda_atom": float(cfg["lambda_atom"]),
        "atomic_mean_improvement": (improvement["a"] + improvement["b"]) / 2.0,
        "atomic_min_improvement": min(improvement["a"], improvement["b"]),
        "atomic_improvement_gap": improvement["a"] - improvement["b"],
    }
    for name in base.LOSS_NAMES:
        result[f"question_{name}"] = questions[name]
        result[f"initial_loss_{name}"] = initial_losses[name]
        result[f"terminal_loss_{name}"] = terminal_losses[name]
        result[f"initial_yes_probability_{name}"] = math.exp(-initial_losses[name])
        result[f"terminal_yes_probability_{name}"] = math.exp(-terminal_losses[name])
        result[f"delta_loss_{name}"] = delta[name]
        result[f"improvement_{name}"] = improvement[name]
        result[f"relative_improvement_{name}"] = relative[name]
    if return_embedding:
        # Keep the terminal state independent of the optimizer and move it to
        # CPU only in the embedding-archiving runner.
        return result, traces, generations, embedding.detach().clone()
    return result, traces, generations


def run_seed(runtime, dset, manifest, specs, seed, cfg, output, signature):
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    results, traces, generations = (base._load_checkpoints(seed_dir, signature)
                                   if cfg.get("resume", True) else ([], [], []))
    completed = {(str(row["spec"]), base._as_int(row["image_id"]))
                 for row in results}
    expected = {(str(row["spec"]), base._as_int(row["image_id"]))
                for row in manifest}
    if len(completed) != len(results) or not completed.issubset(expected):
        raise RuntimeError(f"Unexpected or duplicate checkpoints in {seed_dir}")
    if any(base._as_int(row["seed"]) != seed or row["level"] != LEVEL
           for row in results):
        raise RuntimeError(f"Checkpoint seed/level mismatch in {seed_dir}")
    for row in generations:
        path = Path(row["generated_path"])
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Checkpoint references missing image: {path}")
    spec_by_name = {spec.name: spec for spec in specs}
    contrasts = {}
    with tqdm(total=len(manifest), desc=f"Stage 2 atomic guidance seed {seed}") as progress:
        for item in manifest:
            spec = spec_by_name[str(item["spec"])]
            image_id = base._as_int(item["image_id"])
            key = (spec.name, image_id)
            if key in completed:
                progress.update(1)
                continue
            if spec.name not in contrasts:
                contrasts[spec.name] = base.oo.oo_contrast_embedding(runtime, spec)
            image, source_path = dset[image_id]
            if Path(source_path).resolve() != Path(item["source_path"]).resolve():
                raise RuntimeError(f"Source image changed for {spec.name}/{image_id}")
            result, task_traces, task_generations = optimize_task(
                runtime, image, spec, seed, image_id, cfg,
                contrasts[spec.name], seed_dir)
            result.update({
                "source_path": str(source_path),
                "pool_index": base._as_int(item.get("pool_index")),
                "clip_base_score": base._as_float(item.get("base_score")),
            })
            base.exp4.save_json_atomic(
                base._checkpoint_path(seed_dir, spec.name, image_id), {
                    "experiment_signature": signature,
                    "result": result,
                    "traces": task_traces,
                    "generations": task_generations,
                })
            results.append(result)
            traces.extend(task_traces)
            generations.extend(task_generations)
            completed.add(key)
            if len(results) % int(cfg["checkpoint_every"]) == 0:
                _save_tables(seed_dir, results, traces, generations)
            gc.collect()
            torch.cuda.empty_cache()
            progress.update(1)
    _save_tables(seed_dir, results, traces, generations)
    return results, traces, generations


def _save_tables(output, results, traces, generations, prefix=""):
    for name, rows in (("results", results), ("steps", traces),
                       ("generations", generations)):
        base.exp4.save_csv_atomic(output / f"{prefix}{name}.csv", rows)


def build_group_summary(results, specs):
    """Keep numerical failures in rate denominators, out of loss statistics."""
    valid = [row for row in results if not row.get("numerical_error")]
    summaries = base.build_group_summary(valid, specs)
    for summary in summaries:
        rows = [row for row in results
                if summary["spec"] == "ALL" or row["spec"] == summary["spec"]]
        if summary["group"] == "embedding_success":
            rows = [row for row in rows if row["embedding_success"]]
        elif summary["group"] == "embedding_failure":
            rows = [row for row in rows if not row["embedding_success"]]
        elif summary["group"] == "embedding_success_generation_completed":
            rows = [row for row in rows if row["embedding_success"]
                    and row["image_generation_completed"]]
        summary["n_valid_loss_tasks"] = summary["n"]
        summary["n"] = len(rows)
        summary["n_numerical_failure"] = sum(bool(row.get("numerical_error"))
                                             for row in rows)
        for key in ("embedding_success", "image_generation_completed"):
            summary[f"{key}_rate"] = (sum(row[key] for row in rows) / len(rows)
                                      if rows else float("nan"))
        steps = [row["num_optimization_steps"] for row in rows]
        summary["num_optimization_steps_mean"] = (sum(steps) / len(steps)
                                                   if steps else float("nan"))
    return summaries


def ensure_output_compatibility(output, signature, resume):
    path = output / "experiment_signature.json"
    checkpoints = list(output.glob("seed_*/checkpoints/*.json"))
    if path.exists():
        existing = json.loads(path.read_text())
        if (existing.get("protocol_version") != PROTOCOL_VERSION
                or existing.get("sha256") != signature):
            raise RuntimeError(f"Use a new output directory: incompatible run at {output}")
    elif output.exists() and any(entry.name != "service" for entry in output.iterdir()):
        raise RuntimeError(f"Output is nonempty but has no experiment signature: {output}")
    if not resume and checkpoints:
        raise RuntimeError("--no-resume requires an output without existing checkpoints")


def write_run_metadata(output, source_output, cfg, manifest, specs, signature):
    output.mkdir(parents=True, exist_ok=True)
    base.exp4.save_json_atomic(output / "experiment_signature.json", {
        "protocol_version": PROTOCOL_VERSION, "sha256": signature})
    base.exp4.save_json_atomic(output / "resolved_config.json", cfg)
    base.exp4.save_json_atomic(output / "target_specs.json",
                              [spec.to_dict() for spec in specs])
    base.exp4.save_json_atomic(output / "level_oo_shared_pool_manifest.json", manifest)
    base.exp4.save_json_atomic(output / "source_manifest_reference.json", {
        "source_output_dir": str(source_output),
        "source_manifest": str(source_output / "level_oo_shared_pool_manifest.json"),
        "source_row_count": len(manifest),
        "selected_specs": [spec.name for spec in specs],
        "reference_protocol_version": base.PROTOCOL_VERSION,
        "pool_reselected": False, "post_generation_evaluation": False,
    })


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path,
                        default=SOURCE_OUTPUT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--lambda-atom", type=float,
                        default=DEFAULT_LAMBDA_ATOM,
                        help="Coefficient for the dynamic atomic guidance loss")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate the selected pool/config without loading models or writing outputs")
    args = parser.parse_args(argv)
    if not math.isfinite(args.lambda_atom) or args.lambda_atom < 0:
        parser.error("--lambda-atom must be a finite non-negative number")
    return args


def main(argv=None):
    args = parse_args(argv)
    source_output, output = args.source_output.resolve(), args.output_dir.resolve()
    source_cfg, manifest, specs = validate_source_output(source_output)
    cfg = load_config(source_cfg, source_output, output, not args.no_resume,
                      args.lambda_atom)
    signature = experiment_signature(cfg, manifest)
    ensure_output_compatibility(output, signature, cfg["resume"])
    print(json.dumps({
        "protocol_version": PROTOCOL_VERSION,
        "target_specs": [spec.name for spec in specs],
        "selected_pool_rows": len(manifest), "seeds": cfg["seeds"],
        "total_tasks": len(manifest) * len(cfg["seeds"]),
        "num_steps": cfg["num_steps"], "learning_rate": cfg["learning_rate"],
        "lambda_atom": cfg["lambda_atom"], "lambda_clip": cfg["lambda_clip"],
        "lambda_reg": cfg["lambda_reg"], "yes_threshold": cfg["yes_threshold"],
        "num_generations": cfg["num_generations"],
        "atomic_weight_rule": cfg["atomic_weight_rule"],
        "output_dir": str(output), "experiment_signature": signature,
        "validate_only": args.validate_only,
    }, indent=2), flush=True)
    if args.validate_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the atomic-guided experiment")
    write_run_metadata(output, source_output, cfg, manifest, specs, signature)
    runtime = GuidedRuntime(cfg)
    dset = base.exp4.load_coco_dataset(cfg["data_path"], split="train")
    all_results, all_traces, all_generations = [], [], []
    for seed in cfg["seeds"]:
        base.exp4.seed_all(seed)
        results, traces, generations = run_seed(
            runtime, dset, manifest, specs, seed, cfg, output, signature)
        all_results.extend(results)
        all_traces.extend(traces)
        all_generations.extend(generations)
    _save_tables(output, all_results, all_traces, all_generations, prefix="all_")
    summary = build_group_summary(all_results, specs)
    base.exp4.save_csv_atomic(output / "loss_group_summary.csv", summary)
    base.exp4.save_json_atomic(output / "loss_group_summary.json", summary)


if __name__ == "__main__":
    main()
