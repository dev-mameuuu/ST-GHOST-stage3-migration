"""Evaluate whether Stage 2 generation feedback improves victim-model answers.

This independent evaluation reuses the 40 archived embeddings selected for
``stage2_level_oo_generation_feedback_step_compare_v4``.  It generates two
paired images for every embedding using the same source image, prompt, and
``generation_index=0`` state:

* ``baseline_step0``: generate directly from the archived terminal embedding;
* ``full_step45``: optimize 21 separate denoising-step image conditions for
  45 updates at Adam learning rate 0.02 using only native victim and clean-image losses.

Each image is then passed directly through the victim's normal image encoder
and asked the saved joint-object trigger prompt.  OWLv2 and any other detector
are deliberately not loaded: this experiment isolates the change in victim
Yes rate.  Feedback uses the same native image path at the generated resolution,
with frozen victim weights. Each condition is the frozen first-stage target
plus a zero-initialized, full-dimensional residual. Each of the current 21
executed denoising steps has its own condition; source latents and both noise
tensors stay fixed.
There is no task-subspace construction, target-feature loss, or projector.
CLIP is used only for the clean constraint and cosine diagnostics. Clean embeddings use the
original Stage-1 336x336 preprocessing.  Each task's clean-cosine threshold is
fixed at min(0.85, baseline_clean_cosine - 0.02), bounded below by -1.
At 0, 5, 10, 15, 20, 25, 30, 35, 40, and 45 completed updates, it also archives
the generation condition schedule, its corresponding image, and the feedback metrics.
Intermediate snapshots reuse the image already rendered for that update;
only the step-0 and step-45 images are queried for the final Yes-rate comparison.

The script is resumable at both image-generation and victim-query granularity.
``--phase generate`` and ``--phase evaluate`` run these phases separately;
evaluation can use the completed tasks even after interrupted generation.
``--condition-mode shared`` provides a matched one-residual control.
``--validate-only`` checks the V4 inputs and existing checkpoints without
loading diffusion or victim models and without writing files.
``--initialization clean`` instead initializes all conditions from the clean
image's CLIP embedding, with a separate default output directory. Archived
Stage-1 embeddings then identify the matched tasks only; they do not enter
generation, feedback metrics, or optimization.
``--ssim`` optionally adds a differentiable RGB SSIM hinge at 256x256,
with a fixed threshold of 0.8 and the same weight as the clean constraint.
This optional experiment has its own default output directory and signature.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from PIL import Image
from tqdm import tqdm

import run_stage1 as stage1
import run_stage2_level_oo_generation_feedback as feedback
import run_stage2_level_oo_generation_feedback_step_compare as comparison


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "stage2_level_oo_generation_feedback_victim_evaluation_staged21_victim_clean_step45_lr002_v1"
SOURCE_DEFAULT = comparison.DEFAULT_OUTPUT
OUTPUT_DEFAULT = ROOT / "outputs" / PROTOCOL_VERSION
CLEAN_OUTPUT_DEFAULT = ROOT / "outputs" / (
    "stage2_level_oo_generation_feedback_victim_evaluation_"
    "staged21_clean_init_victim_clean_step45_lr002_v1")
SSIM_OUTPUT_DEFAULT = ROOT / "outputs" / (
    "stage2_level_oo_generation_feedback_victim_evaluation_"
    "staged21_40_victim_clean_ssim080_step45_lr002_vae_normalized_v1")
SSIM_RESOLUTION = 256
SSIM_THRESHOLD = 0.8
SSIM_CONSTRAINT = "fixed_256_rgb_gaussian_ssim_hinge_squared"
LEARNING_RATE = 0.02
FULL_MODE_UPDATES = 45
FULL_METHOD = f"full_step{FULL_MODE_UPDATES}"
METHODS = (
    ("baseline_step0", 0),
    (FULL_METHOD, FULL_MODE_UPDATES),
)
SAVED_UPDATES = tuple(range(0, FULL_MODE_UPDATES + 1, 5))
STAGE_NAMES = tuple(f"stage_{index}" for index in range(1, 22))
FEATURE_METRIC_FIELDS = (
    "clean_loss",
    "total_loss",
    "target_cosine",
    "clean_cosine",
    "baseline_clean_cosine",
    "effective_clean_threshold",
    "clean_constraint_active",
)
METRIC_FIELDS = FEATURE_METRIC_FIELDS + ("victim_loss", "feedback_yes_probability")
SSIM_METRIC_FIELDS = (
    "ssim_256", "ssim_loss", "ssim_threshold",
    "ssim_constraint_active", "ssim_threshold_met",
)


@dataclass(frozen=True)
class EvaluationPlan:
    """Storage/evaluation policy; the optimization objective stays unchanged."""

    saved_updates: tuple[int, ...] = SAVED_UPDATES
    methods: tuple[tuple[str, int], ...] = METHODS
    summary_every: int = 1

    def __post_init__(self):
        if (not self.saved_updates or self.saved_updates[-1] != FULL_MODE_UPDATES
                or tuple(sorted(set(self.saved_updates))) != self.saved_updates
                or any(step < 0 or step > FULL_MODE_UPDATES for step in self.saved_updates)
                or not self.methods or len(set(self.methods)) != len(self.methods)
                or any(updates not in self.saved_updates
                       or method != ("baseline_step0" if updates == 0 else f"full_step{updates}")
                       for method, updates in self.methods)
                or self.summary_every < 1):
            raise ValueError("Invalid snapshot/evaluation plan")


DEFAULT_PLAN = EvaluationPlan()


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def task_key(task: Mapping[str, object]) -> tuple[int, str, int]:
    return int(task["seed"]), str(task["spec"]), int(task["image_id"])


def safe_name(value: object) -> str:
    return feedback._safe_name(value)


def descriptor(task: Mapping[str, object]) -> dict:
    result = {key: task[key] for key in (
        "seed", "spec", "image_id", "source_path", "embedding_path",
        "embedding_sha256", "terminal_step", "trigger_prompt")}
    if "condition_initialization" in task:
        result["condition_initialization"] = task["condition_initialization"]
    if "source_image_sha256" in task:
        result["source_image_sha256"] = task["source_image_sha256"]
    return result


def condition_initial_embedding(task: Mapping[str, object],
                                clean_embedding: torch.Tensor) -> torch.Tensor:
    mode = task.get("condition_initialization", "stage1")
    if mode == "stage1":
        embedding = task["embedding"]
    elif mode == "clean":
        embedding = clean_embedding
    else:
        raise ValueError(f"Unknown condition initialization: {mode}")
    if (not isinstance(embedding, torch.Tensor) or embedding.shape != (1, 1024)
            or not torch.isfinite(embedding).all().item()):
        raise ValueError("Condition initialization must be a finite [1, 1024] embedding")
    return embedding.detach().float()


def image_state(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Generated image is missing or empty: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size": stat.st_size,
    }


def load_source(source: Path, initialization: str = "stage1") -> dict:
    if initialization not in ("stage1", "clean"):
        raise ValueError(f"Unknown condition initialization: {initialization}")
    source = source.resolve()
    required = ("experiment_signature.json", "resolved_config.json",
                "selected_embeddings.json", "target_specs.json")
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    source_signature = json.loads((source / "experiment_signature.json").read_text())
    cfg = json.loads((source / "resolved_config.json").read_text())
    if (source_signature.get("protocol") != comparison.PROTOCOL
            or cfg.get("protocol_version") != comparison.PROTOCOL):
        raise ValueError(f"Expected V4 step-comparison input at {source}")
    if cfg.get("target_loss") != "direction_plus_relative_norm":
        raise ValueError("The source does not use the V4 directional target loss")
    if (float(cfg.get("learning_rate", float("nan"))) != 0.02
            or str(cfg.get("optimizer")) != "Adam"
            or float(cfg.get("weight_decay", float("nan"))) != 0.0
            or float(cfg.get("lambda_direction", float("nan"))) != 1.0
            or float(cfg.get("lambda_norm", float("nan"))) != 0.1
            or float(cfg.get("lambda_clean", float("nan"))) != 1.0
            or float(cfg.get("clean_threshold", float("nan"))) != 0.85):
        raise ValueError("The source V4 optimization configuration is incompatible")
    selected = json.loads((source / "selected_embeddings.json").read_text())
    spec_rows = json.loads((source / "target_specs.json").read_text())
    if not isinstance(selected, list) or len(selected) != 40:
        raise ValueError("V4 must contain exactly 40 selected embeddings")
    specs = {str(row["name"]): feedback.exp4.CompositeSpec.from_dict(row) for row in spec_rows}
    if len(specs) != 5:
        raise ValueError("Expected the five Level-OO target pairs")

    tasks = []
    seen = set()
    for row in selected:
        required_fields = set(descriptor(row))
        if not required_fields.issubset(row):
            raise ValueError("Selected embedding metadata is incomplete")
        key = task_key(row)
        if key in seen or key[1] not in specs:
            raise ValueError(f"Duplicate or invalid selected task: {key}")
        seen.add(key)
        source_path = Path(str(row["source_path"])).resolve()
        embedding_path = Path(str(row["embedding_path"])).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        embedding = feedback._load_embedding(embedding_path, str(row["embedding_sha256"]))
        task = {**descriptor(row), "source_path": str(source_path),
                "embedding_path": str(embedding_path), "embedding": embedding}
        if initialization == "clean":
            task["condition_initialization"] = "clean"
        tasks.append(task)
    return {
        "source": source,
        "source_signature": source_signature,
        "cfg": cfg,
        "specs": specs,
        "tasks": sorted(tasks, key=task_key),
        "input_sha256": {name: sha256(source / name) for name in required},
    }


@dataclass(frozen=True)
class CleanReference:
    baseline_clean_cosine: float
    clean_threshold: float


def adaptive_clean_threshold(baseline_cosine: float, cfg: Mapping[str, object]) -> float:
    cap = float(cfg["clean_threshold"])
    slack = float(cfg["clean_slack"])
    if not math.isfinite(baseline_cosine) or not -1.000001 <= baseline_cosine <= 1.000001:
        raise ValueError("Invalid baseline clean cosine")
    if not math.isfinite(slack) or not 0 <= slack <= 2 or not -1 <= cap <= 1:
        raise ValueError("Invalid adaptive clean-constraint configuration")
    return max(-1.0, min(cap, baseline_cosine - slack))


def stage_mapping(active_steps: int, mode: str) -> list[int]:
    names = STAGE_NAMES if mode == "staged" else ("shared",)
    if mode not in ("staged", "shared") or active_steps < len(names):
        raise ValueError("Condition mode needs at least one active denoising step per stage")
    return [index * len(names) // active_steps for index in range(active_steps)]


def condition_schedule(state, cfg: Mapping[str, object]) -> dict:
    mode = str(cfg.get("condition_mode", "staged"))
    timesteps = state.timesteps[state.timestep_index:].detach().cpu().tolist()
    return {"condition_mode": mode,
            "stage_names": list(STAGE_NAMES if mode == "staged" else ("shared",)),
            "active_timesteps": timesteps,
            "stage_mapping": stage_mapping(len(timesteps), mode)}


def validate_condition_schedule(schedule: Mapping[str, object]) -> None:
    mode = str(schedule.get("condition_mode", ""))
    timesteps = schedule.get("active_timesteps", [])
    names = list(STAGE_NAMES if mode == "staged" else ("shared",))
    if (schedule.get("stage_names") != names or not isinstance(timesteps, list)
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in timesteps)
            or schedule.get("stage_mapping") != stage_mapping(len(timesteps), mode)):
        raise ValueError("Invalid generation condition schedule")


def render_conditions(runtime, state, conditions: torch.Tensor, cfg: Mapping[str, object]):
    schedule = condition_schedule(state, cfg)
    if conditions.ndim != 3 or conditions.shape[:2] != (len(schedule["stage_names"]), 1):
        raise ValueError("Condition tensor does not match its denoising schedule")
    return feedback._render(runtime, state, conditions, cfg,
                            stage_mapping=schedule["stage_mapping"])


def feature_metrics(z_hat: torch.Tensor, z_star: torch.Tensor, z0: torch.Tensor,
                    cfg: Mapping[str, object], reference: CleanReference) -> dict:
    clean_cosine = float(F.cosine_similarity(z_hat.float(), z0.detach().float(), dim=-1).mean().cpu())
    clean_loss = max(reference.clean_threshold - clean_cosine, 0.0) ** 2
    metrics = {
        "clean_loss": clean_loss,
        "total_loss": float(cfg["lambda_clean"]) * clean_loss,
        "target_cosine": float(F.cosine_similarity(
            z_hat.float(), z_star.detach().float(), dim=-1).mean().cpu()),
        "clean_cosine": clean_cosine,
        "baseline_clean_cosine": reference.baseline_clean_cosine,
        "effective_clean_threshold": reference.clean_threshold,
        "clean_constraint_active": float(clean_cosine < reference.clean_threshold),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Non-finite generated feature metrics")
    return {field: float(metrics[field]) for field in FEATURE_METRIC_FIELDS}


def ssim_enabled(cfg: Mapping[str, object] | None) -> bool:
    return cfg is not None and cfg.get("ssim_constraint") == SSIM_CONSTRAINT


def downsample_for_ssim(image: torch.Tensor) -> torch.Tensor:
    """Downsample a copy; the rendered/saved image retains its full resolution."""
    if image.ndim != 4 or image.shape[:2] != (1, 3):
        raise ValueError("SSIM expects one RGB image tensor [1, 3, H, W]")
    return F.interpolate(image.float(), size=(SSIM_RESOLUTION, SSIM_RESOLUTION),
                         mode="bilinear", align_corners=False, antialias=True)


def rgb_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Differentiable Wang SSIM, matching the previous skimage measurements.

    Both inputs are RGB [0,1] at 256x256. Use an 11x11 Gaussian (sigma=1.5),
    population covariance, C1=0.01^2, C2=0.03^2, and valid-window averaging.
    Separable grouped convolutions avoid any extra pretrained network.
    """
    expected = (1, 3, SSIM_RESOLUTION, SSIM_RESOLUTION)
    if image.shape != expected or reference.shape != expected:
        raise ValueError("SSIM inputs must both have shape [1, 3, 256, 256]")
    # Keep moment calculations in float32 even when called under mixed precision.
    with torch.autocast(device_type=image.device.type, enabled=False):
        image = image.float()
        reference = reference.detach().to(device=image.device, dtype=torch.float32)
        offsets = torch.arange(-5, 6, device=image.device, dtype=torch.float32)
        gaussian = torch.exp(-offsets.square() / (2 * 1.5 ** 2))
        gaussian = gaussian / gaussian.sum()
        moments = torch.cat((image, reference, image.square(), reference.square(),
                             image * reference), dim=1)
        channels = moments.shape[1]
        moments = F.conv2d(moments, gaussian.view(1, 1, 1, 11).expand(channels, 1, 1, 11),
                           groups=channels)
        moments = F.conv2d(moments, gaussian.view(1, 1, 11, 1).expand(channels, 1, 11, 1),
                           groups=channels)
        mu_x, mu_y, ex2, ey2, exy = moments.chunk(5, dim=1)
        mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
        variance_x, variance_y = ex2 - mu_x2, ey2 - mu_y2
        covariance = exy - mu_xy
        numerator = (2 * mu_xy + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)
        denominator = (mu_x2 + mu_y2 + 0.01 ** 2) * (variance_x + variance_y + 0.03 ** 2)
        return (numerator / denominator).mean()


def ssim_terms(image: torch.Tensor, reference: torch.Tensor | None,
               cfg: Mapping[str, object]) -> dict[str, torch.Tensor]:
    if not ssim_enabled(cfg):
        return {}
    if reference is None:
        raise ValueError("SSIM constraint requires the original clean RGB reference")
    score = rgb_ssim(downsample_for_ssim(image), reference)
    threshold = score.new_tensor(float(cfg["ssim_threshold"]))
    return {
        "ssim_256": score,
        "ssim_loss": torch.relu(threshold - score).square(),
        "ssim_threshold": threshold,
        "ssim_constraint_active": (score < threshold).float(),
        "ssim_threshold_met": (score >= threshold).float(),
    }


def victim_terms(victim, image: torch.Tensor, question: str) -> dict[str, torch.Tensor]:
    """Compute native-image Yes probability without decoding or detaching pixels."""
    if image.ndim != 4 or image.shape[:2] != (1, 3):
        raise ValueError("Victim feedback expects one RGB image tensor [1, 3, H, W]")
    inputs = stage1.vllm_standard_preprocessing(
        victim.processor, question, image[0], do_rescale=False)
    pixels = inputs.pop("pixel_values").to(dtype=victim.model.visual.dtype)
    if torch.is_grad_enabled() and image.requires_grad and not pixels.requires_grad:
        raise RuntimeError("Victim image preprocessing detached the generated pixels")

    def last_logits(pixel_values):
        return victim.model(
            **inputs, pixel_values=pixel_values, use_cache=False,
            logits_to_keep=1, return_dict=True).logits[:, -1].float()

    # Checkpoint the frozen victim forward to avoid retaining its full activations.
    logits = (checkpoint(last_logits, pixels, use_reentrant=False)
              if torch.is_grad_enabled() and pixels.requires_grad else last_logits(pixels))
    yes_ids = torch.tensor(victim.yes_ids, device=logits.device, dtype=torch.long)
    if yes_ids.numel() == 0:
        raise ValueError("Victim tokenizer has no Yes token IDs")
    log_probability = torch.logsumexp(
        F.log_softmax(logits, dim=-1).index_select(-1, yes_ids), dim=-1)
    return {
        "victim_loss": -log_probability.mean(),
        "feedback_yes_probability": log_probability.exp().mean(),
    }


def feedback_metrics(runtime, image, z_star, z0, cfg, question,
                     reference: CleanReference, z_hat=None, *,
                     ssim_reference: torch.Tensor | None = None) -> dict:
    metrics = feature_metrics(
        feedback._clip_image(runtime, image) if z_hat is None else z_hat,
        z_star, z0, cfg, reference)
    terms = victim_terms(runtime.victim, image, question)
    metrics.update({key: float(value.detach().cpu()) for key, value in terms.items()})
    metrics["total_loss"] += float(cfg["lambda_victim"]) * metrics["victim_loss"]
    structure = ssim_terms(image, ssim_reference, cfg)
    if structure:
        metrics.update({key: float(value.detach().cpu()) for key, value in structure.items()})
        metrics["total_loss"] += float(cfg["lambda_ssim"]) * metrics["ssim_loss"]
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Non-finite generated feedback metrics")
    return metrics


def condition_diagnostics(residuals: torch.Tensor, names: Sequence[str]) -> dict[str, float]:
    gradients = residuals.grad.detach().float().flatten(1)
    offsets = residuals.detach().float().flatten(1)
    norms = gradients.norm(dim=1)
    diagnostics = {
        "gradient_norm": float(gradients.norm().item()),
        "gradient_cancellation_ratio": float(
            (gradients.sum(dim=0).norm() / norms.sum().clamp_min(1e-20)).item()),
    }
    for index, name in enumerate(names):
        diagnostics[f"{name}_gradient_norm"] = float(norms[index].item())
        diagnostics[f"{name}_residual_norm"] = float(offsets[index].norm().item())
        for other in range(index + 1, len(names)):
            pair = f"{name}_{names[other]}"
            denominator = norms[index] * norms[other]
            diagnostics[f"gradient_cosine_{pair}"] = (
                float((gradients[index] @ gradients[other] / denominator).item())
                if denominator.item() > 0 else 0.0)
            diagnostics[f"condition_gap_{pair}"] = float((offsets[index] - offsets[other]).norm().item())
    return diagnostics


def reproduce_full_condition(runtime, state, z_star: torch.Tensor, z0: torch.Tensor,
                             cfg: Mapping[str, object], updates: int,
                             question: str,
                             reference: CleanReference,
                             save_snapshot: Callable | None = None, *,
                             plan: EvaluationPlan = DEFAULT_PLAN,
                             ssim_reference: torch.Tensor | None = None) -> tuple[torch.Tensor, list[dict]]:
    z_star = z_star.detach()
    z0 = z0.detach()
    schedule = condition_schedule(state, cfg)
    residuals = torch.nn.Parameter(z_star.new_zeros((len(schedule["stage_names"]), *z_star.shape)))
    if float(cfg["weight_decay"]) != 0.0:
        raise ValueError("Matched residual-condition experiments require weight_decay=0")
    optimizer = torch.optim.Adam(
        [residuals], lr=float(cfg["learning_rate"]), weight_decay=0.0)
    trace = []
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        conditions = z_star.unsqueeze(0) + residuals
        image = render_conditions(runtime, state, conditions, cfg)
        z_hat = feedback._clip_image(runtime, image)
        clean_cosine = F.cosine_similarity(z_hat.float(), z0, dim=-1).mean()
        clean_loss = torch.relu(reference.clean_threshold - clean_cosine).square()
        victim = victim_terms(runtime.victim, image, question)
        total = (float(cfg["lambda_clean"]) * clean_loss
                 + float(cfg["lambda_victim"]) * victim["victim_loss"])
        structure = ssim_terms(image, ssim_reference, cfg)
        if structure:
            total = total + float(cfg["lambda_ssim"]) * structure["ssim_loss"]
        if not torch.isfinite(total).all().item():
            raise RuntimeError(f"Non-finite feedback loss at full update {step}")
        total.backward()
        if residuals.grad is None or not torch.isfinite(residuals.grad).all().item():
            raise RuntimeError(f"Non-finite condition gradient at full update {step}")
        diagnostics = condition_diagnostics(residuals, schedule["stage_names"])
        grad_norm = diagnostics["gradient_norm"]
        if grad_norm == 0.0:
            raise RuntimeError(f"Zero condition gradient at full update {step}")
        trace.append({
            "updates_completed": step,
            "effective_clean_threshold": reference.clean_threshold,
            "baseline_clean_cosine": reference.baseline_clean_cosine,
            "clean_cosine_before_update": float(clean_cosine.detach().cpu()),
            "clean_constraint_active": float(clean_cosine.detach().cpu() < reference.clean_threshold),
            "clean_loss_before_update": float(clean_loss.detach().cpu()),
            "victim_loss_before_update": float(victim["victim_loss"].detach().cpu()),
            "feedback_yes_probability_before_update": float(
                victim["feedback_yes_probability"].detach().cpu()),
            "total_loss_before_update": float(total.detach().cpu()),
            **{f"{key}_before_update": float(value.detach().cpu())
               for key, value in structure.items()},
            **diagnostics,
        })
        # The current image belongs to c_step, before the next optimizer update.
        if save_snapshot is not None and step != 0 and step in plan.saved_updates:
            with torch.no_grad():
                metrics = feature_metrics(z_hat.detach(), z_star, z0, cfg, reference)
            metrics.update({
                "victim_loss": trace[-1]["victim_loss_before_update"],
                "feedback_yes_probability": trace[-1]["feedback_yes_probability_before_update"],
                "total_loss": trace[-1]["total_loss_before_update"],
            })
            metrics.update({key: float(value.detach().cpu()) for key, value in structure.items()})
            save_snapshot(step, conditions.detach(), image.detach(), metrics)
        optimizer.step()
        if not torch.isfinite(residuals).all().item():
            raise RuntimeError(f"Non-finite condition after full update {step}")
        # Drop the preceding graph before starting the next diffusion trajectory.
        del image, z_hat, clean_cosine, clean_loss, victim, total, conditions, structure
    return (z_star.unsqueeze(0) + residuals).detach(), trace


def generated_path(output: Path, task: Mapping[str, object], method: str) -> Path:
    return (output / "generated" / method / str(task["spec"]) / feedback.LEVEL
            / f"seed{int(task['seed'])}_{int(task['image_id'])}_g0.png").resolve()


def snapshot_method(updates: int) -> str:
    return "baseline_step0" if updates == 0 else f"full_step{updates}"


def condition_embedding_path(output: Path, task: Mapping[str, object], updates: int) -> Path:
    return (output / "condition_embeddings" / f"step_{updates}" / f"seed_{int(task['seed'])}"
            / f"{safe_name(task['spec'])}__{int(task['image_id'])}__{feedback.LEVEL}.pt").resolve()


def clean_reference_path(output: Path, task: Mapping[str, object]) -> Path:
    return (output / "clean_references" / f"seed_{int(task['seed'])}"
            / f"{safe_name(task['spec'])}__{int(task['image_id'])}__{feedback.LEVEL}.pt").resolve()


def save_clean_reference(output: Path, task: Mapping[str, object], z0: torch.Tensor,
                         z_star: torch.Tensor, reference: CleanReference) -> dict:
    path = clean_reference_path(output, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({
        "format": "stage2_clean_reference_v1", "protocol_version": PROTOCOL_VERSION,
        "task": descriptor(task),
        "clean_embedding": z0.detach().float().cpu(),
        "target_embedding": z_star.detach().float().cpu(),
        "baseline_clean_cosine": reference.baseline_clean_cosine,
        "effective_clean_threshold": reference.clean_threshold,
    }, temporary)
    os.replace(temporary, path)
    return {"path": str(path), "sha256": sha256(path),
            "baseline_clean_cosine": reference.baseline_clean_cosine,
            "effective_clean_threshold": reference.clean_threshold}


def snapshot_checkpoint_path(output: Path, task: Mapping[str, object], updates: int) -> Path:
    return (output / "optimization_snapshots" / f"seed_{int(task['seed'])}"
            / f"{safe_name(task['spec'])}__{int(task['image_id'])}__{feedback.LEVEL}__step{updates}.json")


def save_condition_schedule(path: Path, conditions: torch.Tensor, schedule: Mapping[str, object],
                            task: Mapping[str, object], metadata: Mapping[str, object],
                            initial_embedding: torch.Tensor | None = None) -> dict:
    validate_condition_schedule(schedule)
    tensor = conditions.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tensor.shape != (len(schedule["stage_names"]), 1, 1024) or not torch.isfinite(tensor).all().item():
        raise ValueError("Invalid condition schedule tensor")
    if initial_embedding is None:
        if task.get("condition_initialization", "stage1") != "stage1":
            raise ValueError("Clean-initialized schedules require the actual clean embedding")
        initial_embedding = task["embedding"]
    if initial_embedding.shape != (1, 1024) or not torch.isfinite(initial_embedding).all().item():
        raise ValueError("Invalid initial condition embedding")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save({
                "format": "stage2_condition_schedule_v1", "protocol_version": PROTOCOL_VERSION,
                "conditions": tensor, "condition_schedule": dict(schedule),
                "target_embedding": initial_embedding.detach().float().cpu(),
                "metadata": {**descriptor(task), **metadata},
            }, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {"embedding_path": str(path), "embedding_sha256": sha256(path),
            "embedding_shape": list(tensor.shape), "embedding_dtype": "float32",
            "embedding_device": "cpu"}


def load_condition_schedule(path: Path, expected_sha256: str, schedule: Mapping[str, object],
                            task: Mapping[str, object], updates: int, seed: int,
                            initial_embedding: torch.Tensor | None = None) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != expected_sha256:
        raise ValueError(f"Condition schedule digest mismatch: {path}")
    validate_condition_schedule(schedule)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    conditions = payload.get("conditions")
    metadata = payload.get("metadata", {})
    target = payload.get("target_embedding")
    if initial_embedding is None:
        if task.get("condition_initialization", "stage1") != "stage1":
            raise ValueError("Clean-initialized schedules require the actual clean embedding")
        initial_embedding = task["embedding"]
    if (payload.get("format") != "stage2_condition_schedule_v1"
            or payload.get("protocol_version") != PROTOCOL_VERSION
            or payload.get("condition_schedule") != schedule
            or not isinstance(conditions, torch.Tensor)
            or conditions.shape != (len(schedule["stage_names"]), 1, 1024)
            or conditions.dtype != torch.float32 or not torch.isfinite(conditions).all().item()
            or not isinstance(target, torch.Tensor) or not torch.equal(target, initial_embedding.cpu())
            or any(metadata.get(field) != value for field, value in descriptor(task).items())
            or metadata.get("optimization_updates") != updates
            or metadata.get("generation_seed") != seed):
        raise ValueError(f"Invalid condition schedule: {path}")
    if updates == 0 and not torch.equal(conditions, target.unsqueeze(0).expand_as(conditions)):
        raise ValueError(f"Step-0 conditions differ from the selected initialization: {path}")
    return conditions.detach()


def save_optimization_snapshot(output: Path, task: Mapping[str, object], seed: int,
                               updates: int, condition: torch.Tensor,
                               image: torch.Tensor, metrics: Mapping[str, float],
                               schedule: Mapping[str, object],
                               initial_embedding: torch.Tensor | None = None) -> dict:
    method = snapshot_method(updates)
    path = generated_path(output, task, method)
    feedback._save_image(image, path)
    embedding = save_condition_schedule(
        condition_embedding_path(output, task, updates), condition, schedule, task, {
            "protocol_version": PROTOCOL_VERSION, **descriptor(task),
            "embedding_role": "generation_condition_schedule",
            "optimization_updates": updates, "generation_seed": seed,
            "generation_index": 0,
            "effective_clean_threshold": metrics["effective_clean_threshold"],
            "baseline_clean_cosine": metrics["baseline_clean_cosine"],
            **({"ssim_threshold": metrics["ssim_threshold"],
                "ssim_resolution": SSIM_RESOLUTION} if "ssim_256" in metrics else {}),
        }, initial_embedding=initial_embedding)
    snapshot = {
        "method": method, "optimization_updates": updates, **embedding,
        "generated_path": str(path), "image_state": image_state(path),
        "feature_metrics": dict(metrics),
    }
    feedback._save_json_atomic(snapshot_checkpoint_path(output, task, updates), {
        "protocol_version": PROTOCOL_VERSION, "task": descriptor(task),
        "generation_seed": seed, "snapshot": snapshot,
    })
    return snapshot


def generation_checkpoint_path(output: Path, task: Mapping[str, object]) -> Path:
    return (output / "generation_checkpoints" / f"seed_{int(task['seed'])}"
            / f"{safe_name(task['spec'])}__{int(task['image_id'])}__{feedback.LEVEL}.json")


def victim_checkpoint_path(output: Path, row: Mapping[str, object]) -> Path:
    return (output / "victim_checkpoints" / f"seed_{int(row['seed'])}"
            / f"{safe_name(row['spec'])}__{int(row['image_id'])}__{row['method']}.json")


def generate_task(runtime, task: Mapping[str, object], cfg: Mapping[str, object],
                  output: Path, *, plan: EvaluationPlan = DEFAULT_PLAN) -> dict:
    if ("source_image_sha256" in task
            and sha256(Path(str(task["source_path"]))) != task["source_image_sha256"]):
        raise ValueError(f"Clean source image changed since selection: {task_key(task)}")
    with Image.open(str(task["source_path"])) as handle:
        source = handle.convert("RGB")
    clean_source = source.resize((336, 336), Image.Resampling.LANCZOS)
    with torch.no_grad():
        z0 = runtime.clip.images([clean_source]).detach().float()
    # The same clean encoder/preprocessing supplies both initialization and L_clean.
    z_star = condition_initial_embedding(task, z0).to(runtime.device)
    prompt = str(task["trigger_prompt"]).strip()
    if not prompt:
        raise ValueError(f"No saved trigger prompt for {task_key(task)}")
    seed = feedback.exp4.generation_seed(
        int(task["seed"]), str(task["spec"]), int(task["image_id"]), 0)
    state = None
    ssim_reference = None
    snapshots = []

    def save_snapshot(updates, condition, image, metrics):
        snapshots.append(save_optimization_snapshot(
            output, task, seed, updates, condition, image, metrics, schedule,
            initial_embedding=z_star))

    try:
        state = feedback._make_state(runtime, source, seed, prompt, cfg)
        schedule = condition_schedule(state, cfg)
        baseline_conditions = z_star.unsqueeze(0).expand(len(schedule["stage_names"]), -1, -1)
        with torch.no_grad():
            if ssim_enabled(cfg):
                # Reuse the diffusion source preprocessing in RGB [0,1], before
                # the VAE's signed conversion. Cache only the 256x256 reference.
                source_rgb = runtime.diffusion_preprocess(source).unsqueeze(0).to(runtime.device)
                ssim_reference = downsample_for_ssim(source_rgb).detach()
                del source_rgb
            baseline = render_conditions(runtime, state, baseline_conditions, cfg)
            baseline_z = feedback._clip_image(runtime, baseline).detach().float()
            baseline_cosine = float(
                F.cosine_similarity(baseline_z, z0, dim=-1).mean().clamp(-1, 1).cpu())
            reference = CleanReference(
                baseline_clean_cosine=baseline_cosine,
                clean_threshold=adaptive_clean_threshold(baseline_cosine, cfg))
            if 0 in plan.saved_updates:
                baseline_metrics = feedback_metrics(
                    runtime, baseline, z_star, z0, cfg, prompt, reference, z_hat=baseline_z,
                    ssim_reference=ssim_reference)
        reference_record = save_clean_reference(output, task, z0, z_star, reference)
        if 0 in plan.saved_updates:
            save_snapshot(0, baseline_conditions, baseline, baseline_metrics)
        del baseline, baseline_z

        full_condition, trace = reproduce_full_condition(
            runtime, state, z_star, z0, cfg, FULL_MODE_UPDATES, prompt, reference,
            save_snapshot, plan=plan, ssim_reference=ssim_reference)
        with torch.no_grad():
            full_image = render_conditions(runtime, state, full_condition, cfg)
            full_metrics = feedback_metrics(
                runtime, full_image, z_star, z0, cfg, prompt, reference,
                ssim_reference=ssim_reference)
        save_snapshot(FULL_MODE_UPDATES, full_condition, full_image, full_metrics)
    finally:
        source.close()
        clean_source.close()
        del state, ssim_reference
        torch.cuda.empty_cache()
    return {
        "task": descriptor(task),
        "generation_seed": seed,
        "full_mode_updates": FULL_MODE_UPDATES,
        "clean_reference": reference_record,
        "condition_schedule": schedule,
        "optimization_trace": trace,
        "images": [next(item for item in snapshots if item["method"] == method)
                   for method, _ in plan.methods],
        "optimization_snapshots": snapshots,
    }


def generation_rows(records: Mapping[tuple[int, str, int], Mapping[str, object]]) -> list[dict]:
    rows = []
    for key in sorted(records):
        record = records[key]
        task = record["task"]
        for image in record["images"]:
            rows.append({
                **task,
                "clean_reference_path": record["clean_reference"]["path"],
                "clean_reference_sha256": record["clean_reference"]["sha256"],
                "condition_mode": record["condition_schedule"]["condition_mode"],
                "generation_seed": record["generation_seed"],
                "method": image["method"], "optimization_updates": image["optimization_updates"],
                "generated_path": image["generated_path"],
                "condition_embedding_path": image["embedding_path"],
                "condition_embedding_sha256": image["embedding_sha256"],
                "image_state": image["image_state"], **image["image_state"],
                **image["feature_metrics"],
            })
    return rows


def load_generation_checkpoints(output: Path, signature: str,
                                tasks: Sequence[Mapping[str, object]], *,
                                plan: EvaluationPlan = DEFAULT_PLAN,
                                cfg: Mapping[str, object] | None = None) -> dict:
    expected = {task_key(task): task for task in tasks}
    expected_metrics = set(METRIC_FIELDS + (SSIM_METRIC_FIELDS if ssim_enabled(cfg) else ()))
    records = {}
    for path in sorted((output / "generation_checkpoints").glob("seed_*/*.json")):
        payload = json.loads(path.read_text())
        task = payload.get("task", {})
        key = task_key(task)
        if (payload.get("experiment_signature") != signature or key not in expected
                or key in records or descriptor(task) != descriptor(expected[key])
                or path.resolve() != generation_checkpoint_path(output, expected[key])):
            raise ValueError(f"Incompatible generation checkpoint: {path}")
        schedule = payload.get("condition_schedule", {})
        validate_condition_schedule(schedule)
        reference_record = payload.get("clean_reference", {})
        reference_path = clean_reference_path(output, expected[key])
        if (reference_record.get("path") != str(reference_path)
                or not reference_path.is_file()
                or reference_record.get("sha256") != sha256(reference_path)):
            raise ValueError(f"Clean reference mismatch: {path}")
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
        if (reference.get("format") != "stage2_clean_reference_v1"
                or reference.get("protocol_version") != PROTOCOL_VERSION
                or reference.get("task") != descriptor(expected[key])):
            raise ValueError(f"Invalid clean reference: {reference_path}")
        for field in ("baseline_clean_cosine", "effective_clean_threshold"):
            value = float(reference_record.get(field, float("nan")))
            if not math.isfinite(value) or not -1 <= value <= 1 or reference.get(field) != value:
                raise ValueError(f"Invalid task clean constraint: {reference_path}")
        clean_embedding = reference.get("clean_embedding")
        target_embedding = reference.get("target_embedding")
        if (not isinstance(clean_embedding, torch.Tensor) or clean_embedding.shape != (1, 1024)
                or not torch.isfinite(clean_embedding).all().item()
                or not isinstance(target_embedding, torch.Tensor)
                or not torch.equal(target_embedding,
                                   condition_initial_embedding(expected[key], clean_embedding))):
            raise ValueError(f"Invalid task embedding references: {reference_path}")
        images = payload.get("images", [])
        trace = payload.get("optimization_trace", [])
        if (payload.get("full_mode_updates") != FULL_MODE_UPDATES
                or len(trace) != FULL_MODE_UPDATES
                or [row.get("updates_completed") for row in trace] != list(range(FULL_MODE_UPDATES))
                or any(not math.isfinite(float(value)) for row in trace for value in row.values())):
            raise ValueError(f"Incomplete optimization trace: {path}")
        snapshots = payload.get("optimization_snapshots", [])
        if [item.get("optimization_updates") for item in snapshots] != list(plan.saved_updates):
            raise ValueError(f"Incomplete optimization snapshots: {path}")
        expected_images = [next((item for item in snapshots if item.get("method") == method), None)
                           for method, _ in plan.methods]
        if images != expected_images:
            raise ValueError(f"Incomplete generation checkpoint: {path}")
        for item in snapshots:
            updates = int(item["optimization_updates"])
            if item.get("method") != snapshot_method(updates):
                raise ValueError(f"Incorrect snapshot method in {path}")
            expected_path = generated_path(output, expected[key], item["method"])
            if item.get("generated_path") != str(expected_path) or item.get("image_state") != image_state(expected_path):
                raise ValueError(f"Generated image state mismatch: {path}")
            if set(item.get("feature_metrics", {})) != expected_metrics:
                raise ValueError(f"Missing feature metrics in {path}")
            if any(not math.isfinite(float(value)) for value in item["feature_metrics"].values()):
                raise ValueError(f"Non-finite feedback metrics in {path}")
            for field in ("baseline_clean_cosine", "effective_clean_threshold"):
                if item["feature_metrics"][field] != reference_record[field]:
                    raise ValueError(f"Snapshot clean constraint changed: {path}")
            embedding_path = condition_embedding_path(output, expected[key], updates)
            if item.get("embedding_path") != str(embedding_path):
                raise ValueError(f"Condition embedding path mismatch: {path}")
            embedding_hash = str(item.get("embedding_sha256", ""))
            if not embedding_hash:
                raise ValueError(f"Missing condition embedding digest: {path}")
            embedding = load_condition_schedule(
                embedding_path, embedding_hash, schedule, expected[key], updates, payload["generation_seed"],
                initial_embedding=target_embedding)
            if (item.get("embedding_shape") != list(embedding.shape)
                    or item.get("embedding_dtype") != "float32"
                    or item.get("embedding_device") != "cpu"):
                raise ValueError(f"Condition embedding metadata mismatch: {path}")
            snapshot_path = snapshot_checkpoint_path(output, expected[key], updates)
            if json.loads(snapshot_path.read_text()) != {
                    "protocol_version": PROTOCOL_VERSION, "task": descriptor(expected[key]),
                    "generation_seed": payload["generation_seed"], "snapshot": item}:
                raise ValueError(f"Optimization snapshot manifest mismatch: {snapshot_path}")
        records[key] = payload
    return records


@torch.inference_mode()
def evaluate_image(victim: stage1.VictimRuntime, row: Mapping[str, object]) -> dict:
    path = Path(str(row["generated_path"])).resolve()
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    try:
        answer, decision, probability = victim.answer(image, str(row["trigger_prompt"]))
    finally:
        image.close()
    if decision is not stage1.parse_yes_no(answer):
        raise RuntimeError("Victim answer parser returned an inconsistent value")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise RuntimeError("Victim returned an invalid Yes probability")
    return {
        **row,
        "target_question": str(row["trigger_prompt"]),
        "target_answer": answer,
        "target_decision": decision,
        "target_yes_probability": probability,
        "target_query_yes": int(decision is True),
        "target_answer_unparsed": int(decision is None),
    }


def evaluation_rows(records: Mapping[tuple[int, str, int], Mapping[str, object]]) -> list[dict]:
    return generation_rows(records)


def load_victim_checkpoints(output: Path, signature: str, generated: Sequence[Mapping[str, object]]) -> dict:
    expected = {(int(row["seed"]), str(row["spec"]), int(row["image_id"]), str(row["method"])): row
                for row in generated}
    evaluated = {}
    for path in sorted((output / "victim_checkpoints").glob("seed_*/*.json")):
        payload = json.loads(path.read_text())
        row = payload.get("result", {})
        key = (int(row.get("seed", -1)), str(row.get("spec", "")),
               int(row.get("image_id", -1)), str(row.get("method", "")))
        if (payload.get("experiment_signature") != signature or key not in expected
                or key in evaluated or path.resolve() != victim_checkpoint_path(output, expected[key])):
            raise ValueError(f"Incompatible victim checkpoint: {path}")
        reference = expected[key]
        for field in ("source_path", "embedding_path", "embedding_sha256", "trigger_prompt",
                      "generation_seed", "optimization_updates", "generated_path", "image_state",
                      "clean_reference_path", "clean_reference_sha256", "condition_mode"):
            if row.get(field) != reference.get(field):
                raise ValueError(f"Victim checkpoint input mismatch: {path}")
        if image_state(Path(str(row["generated_path"]))) != row["image_state"]:
            raise ValueError(f"Generated image changed after evaluation: {path}")
        decision = row.get("target_decision")
        if decision not in (True, False, None) or decision is not stage1.parse_yes_no(row.get("target_answer", "")):
            raise ValueError(f"Invalid victim answer in {path}")
        if int(row.get("target_query_yes", -1)) != int(decision is True):
            raise ValueError(f"Invalid victim Yes flag in {path}")
        evaluated[key] = row
    return evaluated


def summaries(generated: Sequence[Mapping[str, object]], evaluated: Mapping[tuple, Mapping[str, object]], *,
              plan: EvaluationPlan = DEFAULT_PLAN,
              tasks: Sequence[Mapping[str, object]] | None = None) -> tuple[list[dict], list[dict]]:
    grouped_generated = defaultdict(list)
    grouped_evaluated = defaultdict(list)
    expected_rows = generated if tasks is None else [
        {**descriptor(task), "method": method} for task in tasks for method, _ in plan.methods]
    for row in expected_rows:
        for spec in (str(row["spec"]), "ALL"):
            grouped_generated[(spec, str(row["method"]))].append(row)
    for row in evaluated.values():
        for spec in (str(row["spec"]), "ALL"):
            grouped_evaluated[(spec, str(row["method"]))].append(row)
    summary = []
    for spec in sorted({key[0] for key in grouped_generated}, key=lambda value: (value == "ALL", value)):
        for method, updates in plan.methods:
            expected = grouped_generated[(spec, method)]
            rows = grouped_evaluated[(spec, method)]
            complete = len(rows) == len(expected)
            yes_count = sum(int(row["target_query_yes"]) for row in rows)
            summary.append({
                "spec": spec, "method": method, "optimization_updates": updates,
                "images_expected": len(expected), "images_evaluated": len(rows),
                "evaluation_complete": int(complete), "yes_count": yes_count,
                "yes_rate": yes_count / len(rows) if rows else None,
                "unparsed_answer_count": sum(int(row["target_answer_unparsed"]) for row in rows),
            })
    paired = []
    if ("baseline_step0", 0) not in plan.methods or (FULL_METHOD, FULL_MODE_UPDATES) not in plan.methods:
        return summary, paired
    for spec in sorted({str(row["spec"]) for row in generated} | {"ALL"}, key=lambda value: (value == "ALL", value)):
        keys = {(int(row["seed"]), str(row["spec"]), int(row["image_id"]))
                for row in generated if spec == "ALL" or row["spec"] == spec}
        pairs = []
        for key in sorted(keys):
            baseline = evaluated.get((*key, "baseline_step0"))
            full = evaluated.get((*key, FULL_METHOD))
            if baseline is not None and full is not None:
                pairs.append((baseline, full))
        complete = len(pairs) == len(keys)
        paired.append({
            "spec": spec, "paired_tasks_expected": len(keys), "paired_tasks_evaluated": len(pairs),
            "evaluation_complete": int(complete),
            "no_to_no": sum(not before["target_query_yes"] and not after["target_query_yes"] for before, after in pairs),
            "no_to_yes": sum(not before["target_query_yes"] and after["target_query_yes"] for before, after in pairs),
            "yes_to_no": sum(before["target_query_yes"] and not after["target_query_yes"] for before, after in pairs),
            "yes_to_yes": sum(before["target_query_yes"] and after["target_query_yes"] for before, after in pairs),
            "yes_rate_gain_percentage_points": (
                100 * sum(after["target_query_yes"] - before["target_query_yes"] for before, after in pairs) / len(pairs)
                if pairs else None),
        })
    return summary, paired


def save_outputs(output: Path, generated_records: Mapping, evaluated: Mapping, *,
                 plan: EvaluationPlan = DEFAULT_PLAN,
                 tasks: Sequence[Mapping[str, object]] | None = None) -> None:
    generated = generation_rows(generated_records)
    feedback._save_csv_atomic(output / "generated_images.csv", generated)
    trace = []
    for key in sorted(generated_records):
        record = generated_records[key]
        trace.extend({**descriptor(record["task"]), "generation_seed": record["generation_seed"],
                      "method": FULL_METHOD, **row} for row in record["optimization_trace"])
    feedback._save_csv_atomic(output / "optimization_trace.csv", trace)
    feedback._save_csv_atomic(output / "clean_references.csv", [
        {**descriptor(generated_records[key]["task"]),
         **generated_records[key]["clean_reference"]}
        for key in sorted(generated_records)])
    snapshots = []
    for key in sorted(generated_records):
        record = generated_records[key]
        for snapshot in record["optimization_snapshots"]:
            snapshots.append({
                **descriptor(record["task"]),
                "source_embedding_path": record["task"]["embedding_path"],
                "source_embedding_sha256": record["task"]["embedding_sha256"],
                "generation_seed": record["generation_seed"],
                **{field: value for field, value in snapshot.items() if field != "feature_metrics"},
                **snapshot["feature_metrics"],
            })
    feedback._save_csv_atomic(output / "optimization_snapshots.csv", snapshots)
    structural_groups = defaultdict(list)
    for row in snapshots:
        if "ssim_256" in row:
            for spec in (str(row["spec"]), "ALL"):
                structural_groups[(spec, int(row["optimization_updates"]))].append(row)
    if structural_groups:
        structural_summary = []
        for (spec, updates), rows in sorted(structural_groups.items()):
            structural_summary.append({
                "spec": spec, "optimization_updates": updates,
                "images_generated": len(rows), "ssim_threshold": rows[0]["ssim_threshold"],
                "ssim_256_mean": sum(row["ssim_256"] for row in rows) / len(rows),
                "ssim_loss_mean": sum(row["ssim_loss"] for row in rows) / len(rows),
                "ssim_threshold_met_count": sum(int(row["ssim_threshold_met"]) for row in rows),
                "ssim_threshold_met_rate": sum(row["ssim_threshold_met"] for row in rows) / len(rows),
            })
        feedback._save_csv_atomic(output / "ssim_summary.csv", structural_summary)
        feedback._save_json_atomic(output / "ssim_summary.json", structural_summary)
    evaluation = [evaluated[key] for key in sorted(evaluated)]
    if evaluation:
        feedback._save_csv_atomic(output / "victim_evaluations.csv", evaluation)
    summary, paired = summaries(generated, evaluated, plan=plan, tasks=tasks)
    feedback._save_csv_atomic(output / "summary.csv", summary)
    feedback._save_json_atomic(output / "summary.json", summary)
    if paired:
        feedback._save_csv_atomic(output / "paired_comparison.csv", paired)
    expected_tasks = 40 if tasks is None else len(tasks)
    expected_evaluations = expected_tasks * len(plan.methods)
    feedback._save_json_atomic(output / "completion_summary.json", {
        "generation_complete": len(generated_records),
        "generation_expected": expected_tasks,
        "optimization_snapshots_complete": len(snapshots),
        "optimization_snapshots_expected": expected_tasks * len(plan.saved_updates),
        "saved_optimization_updates": list(plan.saved_updates),
        "victim_evaluations_complete": len(evaluated),
        "victim_evaluations_expected": expected_evaluations,
        "evaluation_complete": (len(generated_records) == expected_tasks
                                and len(evaluated) == expected_evaluations),
        "all_methods_use_generation_index": 0,
    })


def provenance(source: Mapping[str, object], full_updates: int, *,
               plan: EvaluationPlan = DEFAULT_PLAN) -> dict:
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "vae_image_contract": feedback.VAE_IMAGE_CONTRACT,
        "source_output": str(source["source"]),
        "source_protocol": source.get("source_protocol", comparison.PROTOCOL),
        "source_signature": source["source_signature"],
        "source_input_sha256": source["input_sha256"],
        "selected_tasks": [descriptor(task) for task in source["tasks"]],
        "learning_rate": source["cfg"]["learning_rate"],
        "full_mode_updates": full_updates,
        "saved_optimization_updates": list(plan.saved_updates),
        "condition_embedding_storage": "stage2_condition_schedule_v1 CPU float32 with SHA-256",
        "condition_control": {
            "mode": source["cfg"]["condition_mode"],
            "stage_names": list(STAGE_NAMES if source["cfg"]["condition_mode"] == "staged" else ("shared",)),
            "stage_count": len(STAGE_NAMES) if source["cfg"]["condition_mode"] == "staged" else 1,
            "stage_mapping": "floor(executed_step_index * stage_count / executed_step_count)",
            "parameterization": "frozen_z_star_plus_zero_initialized_full_dimensional_residual",
            "weight_decay": 0.0,
            "fixed_state": "source_latent_diffusion_noise_condition_noise_prompt",
        },
        "methods": [{"name": method, "updates": updates} for method, updates in plan.methods],
        "generation_index": 0,
        "prior_v4_metric_matching": False,
        "feedback_objective": {
            "total_formula": "lambda_victim * L_victim + lambda_clean * L_clean",
            "feature_loss": "none",
            "projector_loaded": False,
            "clean_loss": "fixed_baseline_adaptive_cosine_hinge_squared",
            "clean_threshold_cap": source["cfg"]["clean_threshold"],
            "clean_slack": source["cfg"]["clean_slack"],
            "clean_threshold_formula": "max(-1,min(cap,baseline_clean_cosine-clean_slack))",
            "clean_preprocessing": "Stage1_RGB_resize_336x336_LANCZOS_then_OpenCLIP",
            "lambda_clean": source["cfg"]["lambda_clean"],
            "victim_loss": "negative_log_first_token_yes_probability",
            "lambda_victim": source["cfg"]["lambda_victim"],
            "victim_path": "native_image_encoder_at_generated_resolution",
        },
        "evaluation": "victim_normal_image_encoder_saved_trigger_only",
        "detector_evaluation": False,
    }
    if "selection" in source:
        audit["sample_selection"] = source["selection"]
        audit["experiment_name"] = source["cfg"]["experiment_name"]
    if ssim_enabled(source["cfg"]):
        audit["feedback_objective"].update({
            "total_formula": "lambda_victim * L_victim + lambda_clean * L_clean + lambda_ssim * L_ssim",
            "ssim_loss": "max(0,ssim_threshold-SSIM_256(generated,original_clean))^2",
            "ssim_threshold": source["cfg"]["ssim_threshold"],
            "lambda_ssim": source["cfg"]["lambda_ssim"],
            "ssim_weight_rule": "lambda_ssim = lambda_clean",
            "ssim_protocol": {
                "reference": "original_clean_RGB_using_diffusion_source_preprocessing",
                "both_resize": [SSIM_RESOLUTION, SSIM_RESOLUTION],
                "interpolation": "torch_bilinear_align_corners_False_antialias_True_float32",
                "data_range": 1.0, "channels": "RGB_average",
                "gaussian_window": 11, "sigma": 1.5,
                "use_sample_covariance": False, "K1": 0.01, "K2": 0.03,
                "boundary": "valid_windows_only",
                "gradient": "torch_grouped_separable_convolution_through_generated_RGB",
                "saved_image_resolution": "unchanged",
            },
        })
    if source["cfg"].get("condition_initialization", "stage1") == "clean":
        audit["condition_control"]["parameterization"] = (
            "frozen_clean_z0_plus_zero_initialized_full_dimensional_residual")
        audit["initialization_ablation"] = {
            "condition_initialization": "clean",
            "initial_embedding": "CLIP(original_clean_image)",
            "stage1_embedding_used_for_optimization": False,
            "stage1_archive_role": source.get("selection", {}).get(
                "stage1_archive_role", "matched_task_selection_and_provenance_only"),
            "trigger_prompts": source.get("selection", {}).get(
                "prompt_policy", "same_archived_prompts_as_stage1_initialized_experiment"),
            "target_cosine_reference": "clean_initial_embedding_not_stage1_embedding",
            "clean_threshold_matching": "same_adaptive_rule_using_each_initialization_baseline",
        }
    return audit


def ensure_compatible(output: Path, signature: str) -> None:
    path = output / "experiment_signature.json"
    if path.exists():
        expected = {"protocol_version": PROTOCOL_VERSION, "sha256": signature}
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"Output directory belongs to a different experiment: {output}")
    elif output.exists() and any(entry.name not in {"service", "run.lock"} for entry in output.iterdir()):
        raise ValueError(f"Nonempty output directory lacks an experiment signature: {output}")


@contextmanager
def output_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another victim-evaluation process is writing to this output") from error
        yield


def parse_args(default_initialization: str = "stage1", *,
               default_ssim: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--initialization", choices=("stage1", "clean"),
                        default=default_initialization,
                        help="Initialize conditions from Stage-1 optimized or original clean CLIP embeddings")
    parser.add_argument("--lambda-victim", type=float, default=2.0,
                        help="Weight of the native victim Yes-probability loss (default: 2.0)")
    parser.add_argument("--condition-mode", choices=("staged", "shared"), default="staged",
                        help="21 denoising-step residuals, or a matched shared-residual control")
    parser.add_argument("--phase", choices=("all", "generate", "evaluate"), default="all",
                        help="Run both phases, only generation, or evaluate completed tasks without diffusion")
    parser.add_argument("--clean-slack", type=float, default=0.02,
                        help="Allowed clean-cosine drop from step 0 (default: 0.02)")
    parser.add_argument("--ssim", action="store_true", default=default_ssim,
                        help="Add differentiable 256x256 SSIM loss, weighted identically to L_clean")
    parser.add_argument("--ssim-threshold", type=float, default=SSIM_THRESHOLD,
                        help="Fixed SSIM hinge threshold when --ssim is active (default: 0.8)")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        if args.ssim:
            args.output_dir = (SSIM_OUTPUT_DEFAULT.with_name(SSIM_OUTPUT_DEFAULT.name + "_clean_init")
                               if args.initialization == "clean" else SSIM_OUTPUT_DEFAULT)
        else:
            args.output_dir = CLEAN_OUTPUT_DEFAULT if args.initialization == "clean" else OUTPUT_DEFAULT
    if not math.isfinite(args.lambda_victim) or args.lambda_victim < 0:
        parser.error("--lambda-victim must be finite and nonnegative")
    if not math.isfinite(args.clean_slack) or not 0 <= args.clean_slack <= 2:
        parser.error("--clean-slack must be finite and between 0 and 2")
    if not math.isfinite(args.ssim_threshold) or not 0 <= args.ssim_threshold <= 1:
        parser.error("--ssim-threshold must be finite and between 0 and 1")
    return args


def generate_pending_tasks(source: Mapping[str, object], output: Path, signature: str,
                            generated_records: dict, *,
                            plan: EvaluationPlan = DEFAULT_PLAN) -> None:
    """Keep generation models in a separate scope from final answer decoding."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to regenerate the feedback images")
    runtime_cfg = {**source["cfg"], "output_dir": str(output)}
    runtime = None
    try:
        runtime = feedback.exp4.Runtime(runtime_cfg, attack=False)
        runtime.load_diffusion()
        runtime.pipe.unet.to(dtype=torch.float32).requires_grad_(False)
        runtime.pipe.vae.to(dtype=torch.float32).requires_grad_(False)
        runtime.pipe.text_encoder.requires_grad_(False)
        runtime.victim = stage1.VictimRuntime(runtime_cfg, runtime.device)
        # No Stage-1 projector or task-subspace gradients are needed.
        pending = [task for task in source["tasks"] if task_key(task) not in generated_records]
        for index, task in enumerate(tqdm(pending, desc="Generate staged victim/clean images"), 1):
            record = generate_task(runtime, task, runtime_cfg, output, plan=plan)
            path = generation_checkpoint_path(output, task)
            feedback._save_json_atomic(path, {"experiment_signature": signature, **record})
            generated_records[task_key(task)] = record
            if index % plan.summary_every == 0 or index == len(pending):
                save_outputs(output, generated_records, {}, plan=plan, tasks=source["tasks"])
    finally:
        del runtime
        gc.collect()
        torch.cuda.empty_cache()


def main(default_initialization: str = "stage1", *, default_ssim: bool = False) -> None:
    args = parse_args(default_initialization=default_initialization, default_ssim=default_ssim)
    source = load_source(args.source_output, initialization=args.initialization)
    run_experiment(args, source)


def run_experiment(args: argparse.Namespace, source: dict, *,
                   plan: EvaluationPlan = DEFAULT_PLAN, prepare_only: bool = False) -> None:
    """Run the shared optimizer on an already validated, explicitly selected task set."""
    learning_rate = float(getattr(args, "learning_rate", LEARNING_RATE))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    source["cfg"].update({
        "learning_rate": learning_rate,
        "lambda_victim": args.lambda_victim,
        "condition_mode": args.condition_mode,
        "clean_slack": args.clean_slack,
        "clean_constraint": "fixed_baseline_adaptive_cosine_hinge_squared",
        "target_loss": "none",
        "feedback_loss": "victim_plus_clean",
        "vae_image_contract": feedback.VAE_IMAGE_CONTRACT,
    })
    if getattr(args, "ssim", False):
        source["cfg"].update({
            "ssim_constraint": SSIM_CONSTRAINT,
            "ssim_threshold": args.ssim_threshold,
            "ssim_resolution": SSIM_RESOLUTION,
            "lambda_ssim": float(source["cfg"]["lambda_clean"]),
            "feedback_loss": "victim_plus_clean_plus_ssim",
        })
    # Record the initialization ablation only when clean embeddings are requested.
    if args.initialization == "clean":
        source["cfg"]["condition_initialization"] = "clean"
    output = args.output_dir.resolve()
    if output == source["source"] or output in source["source"].parents or source["source"] in output.parents:
        raise ValueError("The evaluation output must be separate from the V4 source output")
    audit = provenance(source, FULL_MODE_UPDATES, plan=plan)
    signature = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    ensure_compatible(output, signature)
    print(json.dumps({
        "protocol_version": PROTOCOL_VERSION,
        "source_output": str(source["source"]),
        "output_dir": str(output),
        "tasks": len(source["tasks"]),
        "learning_rate": source["cfg"]["learning_rate"],
        "vae_image_contract": audit["vae_image_contract"],
        "condition_initialization": args.initialization,
        "condition_control": audit["condition_control"],
        "generated_images": len(source["tasks"]) * len(plan.methods),
        "saved_snapshot_images": len(source["tasks"]) * len(plan.saved_updates),
        "saved_optimization_updates": list(plan.saved_updates),
        "methods": [{"method": method, "optimization_updates": updates} for method, updates in plan.methods],
        "generation_index": 0,
        "detector_evaluation": False,
        "prior_v4_metric_matching": False,
        "feedback_objective": audit["feedback_objective"],
        "signature": signature,
        "validate_only": args.validate_only,
        "phase": args.phase,
    }, ensure_ascii=False, indent=2), flush=True)
    if args.validate_only:
        generated_records = load_generation_checkpoints(
            output, signature, source["tasks"], plan=plan, cfg=source["cfg"])
        evaluated = load_victim_checkpoints(output, signature, generation_rows(generated_records))
        print(json.dumps({"validated_generation_checkpoints": len(generated_records),
                          "validated_victim_checkpoints": len(evaluated)}, indent=2), flush=True)
        return
    with output_lock(output):
        ensure_compatible(output, signature)
        feedback._save_json_atomic(output / "experiment_signature.json", {
            "protocol_version": PROTOCOL_VERSION, "sha256": signature})
        feedback._save_json_atomic(output / "resolved_config.json", {
            **source["cfg"], "protocol_version": PROTOCOL_VERSION,
            "source_output": str(source["source"]), "output_dir": str(output),
            "generation_index": 0, "full_mode_updates": FULL_MODE_UPDATES,
            "condition_stage_names": audit["condition_control"]["stage_names"],
            "condition_stage_count": audit["condition_control"]["stage_count"],
            "saved_optimization_updates": list(plan.saved_updates),
            "prior_v4_metric_matching": False,
            "victim_evaluation": True, "detector_evaluation": False,
        })
        feedback._save_json_atomic(output / "input_provenance.json", audit)
        feedback._save_json_atomic(output / "target_specs.json", [spec.to_dict() for spec in source["specs"].values()])
        feedback._save_json_atomic(output / "selected_embeddings.json", [descriptor(task) for task in source["tasks"]])
        if "selection" in source:
            feedback._save_json_atomic(output / "selection_summary.json", source["selection"])
        generated_records = load_generation_checkpoints(
            output, signature, source["tasks"], plan=plan, cfg=source["cfg"])
        if prepare_only:
            evaluated = load_victim_checkpoints(output, signature, generation_rows(generated_records))
            save_outputs(output, generated_records, evaluated, plan=plan, tasks=source["tasks"])
            print("Inputs prepared; no diffusion/victim models loaded and no GPU experiment started.", flush=True)
            return
        if args.phase != "evaluate" and len(generated_records) < len(source["tasks"]):
            generate_pending_tasks(source, output, signature, generated_records, plan=plan)
        generated = generation_rows(generated_records)
        evaluated = load_victim_checkpoints(output, signature, generated)
        save_outputs(output, generated_records, evaluated, plan=plan, tasks=source["tasks"])
        if args.phase == "generate":
            return
        if not generated:
            raise RuntimeError("No completed generation checkpoints; run --phase generate first")
        if len(evaluated) == len(generated):
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for direct victim-model evaluation")
        stage1.seed_all(42)
        victim = stage1.VictimRuntime(source["cfg"], torch.device("cuda"))
        try:
            for row in tqdm(generated, desc="Victim evaluation"):
                key = (int(row["seed"]), str(row["spec"]), int(row["image_id"]), str(row["method"]))
                if key in evaluated:
                    continue
                if image_state(Path(str(row["generated_path"]))) != row["image_state"]:
                    raise RuntimeError(f"Generated image changed after validation: {row['generated_path']}")
                result = evaluate_image(victim, row)
                checkpoint = victim_checkpoint_path(output, row)
                feedback._save_json_atomic(checkpoint, {
                    "experiment_signature": signature, "result": result})
                evaluated[key] = result
                if len(evaluated) % plan.summary_every == 0 or len(evaluated) == len(generated):
                    save_outputs(output, generated_records, evaluated, plan=plan, tasks=source["tasks"])
        finally:
            del victim
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
