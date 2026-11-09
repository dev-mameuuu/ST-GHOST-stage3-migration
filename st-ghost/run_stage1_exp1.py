"""Stage 1 Experiment 1: paired Single-O and Multi-O GHOST attacks.

This script keeps the original GHOST protocol wherever possible:

* COCO negatives are ranked by CLIP similarity to the target objects;
* one complete binary query is optimized at a time;
* the loss is ``-log p(Yes) + lambda_clip * Lclip + lambda_reg * Lreg``;
* the optimized CLIP embedding is decoded with Stable Diffusion unCLIP;
* OWLv2 removes generations that contain the target objects.

Multi-O changes only the attack query and its semantic contrast term.  It does
not sum two object attack losses: the complete two-object proposition is one
query and therefore one attack loss, as required by Stage 1 of the protocol.

Run from the repository root, for example::

    python st-ghost/run_stage1_exp1.py \
        --config st-ghost/configs/stage1_exp1_main.json

The default configuration uses five target pairs, 200 paired source images per
pair, and three random seeds.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import binomtest
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# The runtime classes below are the same local-model interfaces used by the
# existing Stage-1 runner.  Only the experiment logic is separate because that
# runner implements the later weighted-sum diagnostic rather than this single-
# query experiment.
from run_stage1 import (  # noqa: E402
    ORIGINAL_SINGLE_PROMPTS,
    Runtime,
    TargetSpec,
    load_coco_dataset,
    save_csv,
    seed_all,
)
from utils import generate_image, get_vae_features  # noqa: E402


CONDITIONS = ("Single-O", "Multi-O")
COLORS = {"Single-O": "#0072B2", "Multi-O": "#E69F00"}
PAPER_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


MULTI_PROMPTS = (
    "Do you see both a {a} and a {b} in the image? Answer with 'Yes' or 'No'.",
    "Are there a {a} and a {b} here? Answer with 'Yes' or 'No'.",
    "Does the image contain both a {a} and a {b}? Answer with 'Yes' or 'No'.",
    "Can you find a {a} and a {b} in this picture? Answer with 'Yes' or 'No'.",
    "Would you say there are both a {a} and a {b} here? Answer with 'Yes' or 'No'.",
    "Are a {a} and a {b} present in this image? Answer with 'Yes' or 'No'.",
)

# These are deliberately different from the optimization prompts.  Multi-O
# uses a complete query for the attack and two independent object questions for
# atom-level diagnostics after decoding.
SINGLE_EVAL = "Can you see any {obj} in this picture? Answer with 'Yes' or 'No'."
MULTI_EVAL = "Can you see both a {a} and a {b} in this picture? Answer with 'Yes' or 'No'."


@dataclass(frozen=True)
class PairSpec:
    name: str
    objects: Tuple[str, str]

    @classmethod
    def from_dict(cls, value: dict) -> "PairSpec":
        objects = tuple(str(item) for item in value.get("objects", ()))
        if len(objects) != 2 or objects[0] == objects[1]:
            raise ValueError(f"{value.get('name', '<unnamed>')} must contain two distinct objects")
        return cls(str(value["name"]), (objects[0], objects[1]))

    def to_dict(self) -> dict:
        return {"name": self.name, "objects": list(self.objects)}

    def as_target_spec(self) -> TargetSpec:
        # AbsenceVerifier only needs ``name`` and ``objects``.  A TargetSpec is
        # used so the existing verifier stays the single source of detector
        # behavior for this repository.
        return TargetSpec(self.name, self.objects, tuple(), tuple())


def single_question(spec: PairSpec, index: int = 0) -> str:
    return ORIGINAL_SINGLE_PROMPTS[index % len(ORIGINAL_SINGLE_PROMPTS)].format(
        obj=spec.objects[0])


def multi_question(spec: PairSpec, index: int = 0) -> str:
    return MULTI_PROMPTS[index % len(MULTI_PROMPTS)].format(
        a=spec.objects[0], b=spec.objects[1])


def optimization_questions(spec: PairSpec, condition: str) -> Tuple[str, ...]:
    if condition == "Single-O":
        return tuple(template.format(obj=spec.objects[0])
                     for template in ORIGINAL_SINGLE_PROMPTS)
    if condition == "Multi-O":
        return tuple(template.format(a=spec.objects[0], b=spec.objects[1])
                     for template in MULTI_PROMPTS)
    raise ValueError(f"Unknown condition: {condition}")


def evaluation_question(spec: PairSpec, condition: str) -> str:
    if condition == "Single-O":
        return SINGLE_EVAL.format(obj=spec.objects[0])
    if condition == "Multi-O":
        return MULTI_EVAL.format(a=spec.objects[0], b=spec.objects[1])
    raise ValueError(f"Unknown condition: {condition}")


def resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    cfg.setdefault("seeds", [cfg.get("seed", 42)])
    cfg["seeds"] = [int(seed) for seed in cfg["seeds"]]
    if len(cfg["seeds"]) != 3:
        raise ValueError("Stage 1 Experiment 1 expects exactly three seeds")
    cfg.setdefault("num_images_per_spec", 200)
    cfg.setdefault("candidate_top_k", 1000)
    cfg.setdefault("clip_selection_batch_size", 64)
    cfg.setdefault("resume", True)
    cfg.setdefault("save_failed_images", False)
    cfg.setdefault("checkpoint_every", 16)
    cfg.setdefault("bootstrap_samples", 10000)
    cfg.setdefault("diffusion_prompt_mode", "ghost_trigger")
    cfg.setdefault("noise_level", 0)
    # These defaults are the Qwen values reported in Table 14 of the GHOST
    # paper, rather than the older debug defaults in GHOST-master/main.py.
    paper_defaults = {
        "num_steps": 100,
        "learning_rate": 0.1,
        "yes_threshold": 0.8,
        "lambda_clip": 15.0,
        "lambda_reg": 10.0,
        "num_generations": 4,
        "guidance_scale": 5.0,
        "diffusion_t": 30,
        "num_inference_steps": 50,
        "owl_threshold": 0.5,
    }
    for key, value in paper_defaults.items():
        cfg.setdefault(key, value)

    for key in (
        "victim_model_path", "clip_checkpoint_path", "projector_path",
        "data_path", "coco_captions_path", "diffusion_model_path",
        "owl_model_path", "output_dir",
    ):
        if key in cfg and cfg[key] is not None:
            cfg[key] = str(resolve_path(cfg[key]))
    required = (
        "victim_model_path", "clip_checkpoint_path", "projector_path",
        "data_path", "coco_captions_path", "diffusion_model_path",
        "owl_model_path", "output_dir",
    )
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError("Missing configuration keys: " + ", ".join(missing))
    if int(cfg["num_images_per_spec"]) <= 0:
        raise ValueError("num_images_per_spec must be positive")
    if int(cfg["candidate_top_k"]) < int(cfg["num_images_per_spec"]):
        raise ValueError("candidate_top_k must cover num_images_per_spec")
    if int(cfg["checkpoint_every"]) <= 0:
        raise ValueError("checkpoint_every must be positive")
    if not 0.0 < float(cfg["yes_threshold"]) <= 1.0:
        raise ValueError("yes_threshold must be in (0, 1]")
    if not 0.0 <= float(cfg["owl_threshold"]) <= 1.0:
        raise ValueError("owl_threshold must be in [0, 1]")
    if not cfg.get("target_specs"):
        raise ValueError("target_specs cannot be empty")
    cfg["target_specs"] = [PairSpec.from_dict(value).to_dict()
                            for value in cfg["target_specs"]]
    return cfg


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: object, default: float = float("nan")) -> float:
    if value is None or value == "" or str(value).lower() == "nan":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def save_json_atomic(path: Path, value: object) -> None:
    """Write a checkpoint so a crash cannot leave a half-written task file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False))
    os.replace(temporary, path)


def save_csv_atomic(path: Path, rows: Sequence[dict]) -> None:
    """Materialize a CSV without exposing a partially written file."""
    if not rows:
        return
    temporary = path.with_name(path.name + ".tmp")
    save_csv(temporary, rows)
    os.replace(temporary, path)


def task_checkpoint_path(seed_dir: Path, spec: str, image_id: int,
                         condition: str) -> Path:
    return seed_dir / "checkpoints" / f"{spec}__{image_id}__{condition}.json"


def task_key(row: dict) -> Tuple[str, int, str]:
    return (str(row.get("spec", "")), as_int(row.get("image_id")),
            str(row.get("condition", "")))


def load_task_checkpoints(seed_dir: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    """Load complete task records; incomplete CSV caches are never authoritative."""
    results: List[dict] = []
    traces: List[dict] = []
    generations: List[dict] = []
    checkpoint_dir = seed_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return results, traces, generations
    for path in sorted(checkpoint_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        results.append(payload["result"])
        traces.extend(payload.get("traces", []))
        generations.extend(payload.get("generations", []))
    return results, traces, generations


def rank_candidates(runtime: Runtime, dset, spec: PairSpec,
                    top_k: int, batch_size: int, output: Path) -> List[dict]:
    """Reproduce GHOST's absent-target CLIP Top-K selection.

    The primary score is the minimum similarity to the two neutral object
    prompts.  The mean is only a deterministic tie breaker, so a source image
    must be semantically close to both objects before it enters the pool.
    """
    candidates = dset.get_imgIds_by_class(
        present_classes=dset.get_all_targets_names(),
        absent_classes=list(spec.objects))
    text_features = runtime.clip.texts(
        [f"a photo of a {obj}" for obj in spec.objects])
    rows: List[dict] = []
    for start in tqdm(range(0, len(candidates), batch_size),
                      desc=f"CLIP rank {spec.name}"):
        image_ids = candidates[start:start + batch_size]
        images = [dset[int(image_id)][0] for image_id in image_ids]
        image_features = F.normalize(runtime.clip.images(images).float(), dim=-1)
        scores = image_features @ text_features.T
        for index, image_id in enumerate(image_ids):
            values = [float(scores[index, col].item()) for col in range(2)]
            rows.append({
                "spec": spec.name,
                "image_id": int(image_id),
                "clip_sim_a": values[0],
                "clip_sim_b": values[1],
                "base_score": min(values),
                "tie_score": sum(values) / 2.0,
            })
    rows.sort(key=lambda row: (row["base_score"], row["tie_score"]), reverse=True)
    ranked = rows[:top_k]
    save_csv(output / f"clip_ranked_{spec.name}.csv", ranked)
    return ranked


def build_manifest(runtime: Runtime, dset, specs: Sequence[PairSpec],
                   cfg: dict, output: Path) -> List[dict]:
    """Build one paired source pool per target pair.

    A candidate is retained only when both target queries have an unambiguous
    initial ``No`` answer.  This is stricter than selecting the two conditions
    independently and guarantees a paired comparison on exactly the same source
    images.
    """
    manifest: List[dict] = []
    required = int(cfg["num_images_per_spec"])
    for spec in specs:
        ranked = rank_candidates(
            runtime, dset, spec, int(cfg["candidate_top_k"]),
            int(cfg["clip_selection_batch_size"]), output)
        accepted: List[dict] = []
        audit: List[dict] = []
        single_q = single_question(spec)
        multi_q = multi_question(spec)
        for rank, candidate in enumerate(
                tqdm(ranked, desc=f"Prescreen {spec.name}"), start=1):
            image, source_path = dset[int(candidate["image_id"])]
            single_answer, single_decision, single_prob = runtime.victim.answer(
                image, single_q)
            multi_answer, multi_decision, multi_prob = runtime.victim.answer(
                image, multi_q)
            row = {
                **candidate,
                "rank": rank,
                "source_path": str(source_path),
                "single_prescreen_question": single_q,
                "single_prescreen_answer": single_answer,
                "single_prescreen_decision": single_decision,
                "single_prescreen_yes_probability": single_prob,
                "multi_prescreen_question": multi_q,
                "multi_prescreen_answer": multi_answer,
                "multi_prescreen_decision": multi_decision,
                "multi_prescreen_yes_probability": multi_prob,
                "paired_prescreen_no": int(single_decision is False and
                                            multi_decision is False),
            }
            audit.append(row)
            if row["paired_prescreen_no"]:
                accepted.append(row)
                row["pool_index"] = len(accepted) - 1
                manifest.append({
                    "spec": spec.name,
                    "image_id": int(candidate["image_id"]),
                    "source_path": str(source_path),
                    "pool_index": row["pool_index"],
                    "rank": rank,
                    "clip_sim_a": candidate["clip_sim_a"],
                    "clip_sim_b": candidate["clip_sim_b"],
                    "base_score": candidate["base_score"],
                    "tie_score": candidate["tie_score"],
                })
                if len(accepted) == required:
                    break
        save_csv(output / f"prescreen_{spec.name}.csv", audit)
        if len(accepted) != required:
            raise RuntimeError(
                f"Only {len(accepted)}/{required} paired No images for {spec.name}; "
                "increase candidate_top_k or revise the target pair.")
    return manifest


def validate_manifest(manifest: Sequence[dict], specs: Sequence[PairSpec],
                      required: int) -> None:
    counts = {spec.name: 0 for spec in specs}
    for row in manifest:
        if row.get("spec") not in counts:
            raise ValueError(f"Manifest contains unknown spec {row.get('spec')}")
        counts[row["spec"]] += 1
    bad = {name: count for name, count in counts.items() if count != required}
    if bad:
        raise ValueError(f"Manifest must contain {required} rows per spec: {bad}")


def generation_seed(seed: int, image_id: int, generation_index: int) -> int:
    # The condition is intentionally excluded so paired conditions share the
    # same diffusion seed for the same source image and attempt index.
    return int(seed) + int(image_id) * 100 + int(generation_index)


def contrast_embeddings(runtime: Runtime, spec: PairSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return GHOST's enhanced semantic suppressors for both target objects."""
    return (runtime.clip.enhanced_text(spec.objects[0]).detach(),
            runtime.clip.enhanced_text(spec.objects[1]).detach())


def contrast_loss(embedding: torch.Tensor,
                 contrasts: Tuple[torch.Tensor, torch.Tensor],
                 condition: str) -> torch.Tensor:
    first, second = contrasts
    first_loss = F.cosine_similarity(embedding.float(), first.float(), dim=-1).mean()
    if condition == "Single-O":
        return first_loss
    second_loss = F.cosine_similarity(embedding.float(), second.float(), dim=-1).mean()
    # The contrast term suppresses both object semantics, but the attack term
    # remains one complete Multi-O query (never two object losses).
    return 0.5 * (first_loss + second_loss)


def verify_presence(runtime: Runtime, image: Image.Image,
                    spec: PairSpec, condition: str) -> dict:
    result = runtime.absence.verify(
        image, spec.as_target_spec(),
        "Single-O" if condition == "Single-O" else "Multi-O")
    presence = result["presence"]
    a, b = spec.objects
    return {
        "presence_a": int(bool(presence.get(a, False))),
        "presence_b": int(bool(presence.get(b, False))) if condition == "Multi-O" else float("nan"),
        "absence_a": int(not bool(presence.get(a, False))),
        "absence_b": int(not bool(presence.get(b, False))) if condition == "Multi-O" else float("nan"),
        "absence_all": int(not bool(presence.get(a, False)) and
                            (condition == "Single-O" or not bool(presence.get(b, False)))),
        "score_a": float(result["max_scores"].get(a, 0.0)),
        "score_b": (float(result["max_scores"].get(b, 0.0))
                    if condition == "Multi-O" else float("nan")),
    }


def evaluate_generated(runtime: Runtime, image: Image.Image, spec: PairSpec,
                       condition: str) -> dict:
    query = evaluation_question(spec, condition)
    target_answer, target_decision, target_prob = runtime.victim.answer(image, query)
    a, b = spec.objects
    answer_a, decision_a, prob_a = runtime.victim.answer(
        image, SINGLE_EVAL.format(obj=a))
    result = {
        "target_question": query,
        "target_answer": target_answer,
        "target_decision": target_decision,
        "target_yes_probability": target_prob,
        "target_query_yes": int(target_decision is True),
        "atom_a_question": SINGLE_EVAL.format(obj=a),
        "atom_a_answer": answer_a,
        "atom_a_decision": decision_a,
        "atom_a_yes": int(decision_a is True),
        "atom_a_yes_probability": prob_a,
    }
    if condition == "Multi-O":
        answer_b, decision_b, prob_b = runtime.victim.answer(
            image, SINGLE_EVAL.format(obj=b))
        result.update({
            "atom_b_question": SINGLE_EVAL.format(obj=b),
            "atom_b_answer": answer_b,
            "atom_b_decision": decision_b,
            "atom_b_yes": int(decision_b is True),
            "atom_b_yes_probability": prob_b,
            "atom_all_yes": int(decision_a is True and decision_b is True),
        })
    else:
        result.update({
            "atom_b_question": "",
            "atom_b_answer": "",
            "atom_b_decision": "",
            "atom_b_yes": float("nan"),
            "atom_b_yes_probability": float("nan"),
            "atom_all_yes": int(decision_a is True),
        })
    result.update(verify_presence(runtime, image, spec, condition))
    result["valid_atom_a"] = int(result["atom_a_yes"] and result["absence_a"])
    result["valid_atom_b"] = (
        int(result["atom_b_yes"] and result["absence_b"])
        if condition == "Multi-O" else float("nan"))
    result["valid_atom_all"] = int(
        result["atom_all_yes"] and result["absence_all"])
    result["valid_attack_success"] = int(
        result["target_query_yes"] and result["absence_all"])
    return result


def optimize_embedding(runtime: Runtime, source: Image.Image, spec: PairSpec,
                       condition: str, seed: int, image_id: int,
                       cfg: dict, contrasts: Tuple[torch.Tensor, torch.Tensor]
                       ) -> Tuple[torch.Tensor, bool, int, str, List[dict]]:
    """Optimize one embedding against exactly one complete binary query."""
    with torch.no_grad():
        initial = runtime.clip.images([source]).detach()
    embedding = nn.Parameter(initial.clone())
    optimizer = torch.optim.AdamW([embedding], lr=float(cfg["learning_rate"]))
    prompts = optimization_questions(spec, condition)
    prompt_rng = random.Random(int(seed) + int(image_id) * 1009)
    trace: List[dict] = []
    threshold = float(cfg["yes_threshold"])
    trigger_step, trigger_prompt = -1, ""
    final_prob = 0.0
    for step in range(int(cfg["num_steps"])):
        prompt = prompts[prompt_rng.randrange(len(prompts))]
        prob_before = runtime.victim.projected_probability(
            source, runtime.projector.model(embedding).half().squeeze(0), prompt)
        attack = -torch.log(prob_before.clamp_min(1e-8))
        clip_term = contrast_loss(embedding, contrasts, condition)
        reg_term = F.mse_loss(embedding.float(), initial.float())
        loss = attack + float(cfg["lambda_clip"]) * clip_term \
            + float(cfg["lambda_reg"]) * reg_term
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            updated_tokens = runtime.projector.model(embedding).half().squeeze(0)
            prob_after = runtime.victim.projected_probability(
                source, updated_tokens, prompt)
            final_prob = float(prob_after.detach().cpu())
        trace.append({
            "seed": seed, "spec": spec.name, "image_id": image_id,
            "condition": condition, "step": step, "prompt": prompt,
            "loss": float(loss.detach()),
            "attack_loss": float(attack.detach()),
            "clip_loss": float(clip_term.detach()),
            "reg_loss": float(reg_term.detach()),
            "yes_before": float(prob_before.detach().cpu()),
            "yes_after": final_prob,
            "threshold_reached": int(final_prob >= threshold),
        })
        if final_prob >= threshold:
            trigger_step, trigger_prompt = step, prompt
            break
    success = trigger_step >= 0
    return embedding.detach(), success, trigger_step, trigger_prompt, trace


def generate_and_evaluate(runtime: Runtime, source: Image.Image,
                          embedding: torch.Tensor, spec: PairSpec,
                          condition: str, seed: int, image_id: int,
                          trigger_step: int, trigger_prompt: str,
                          cfg: dict, output: Path) -> List[dict]:
    if trigger_step < 0:
        return []
    runtime.load_diffusion()
    rows: List[dict] = []
    save_failed = bool(cfg.get("save_failed_images", False))
    for index in range(int(cfg["num_generations"])):
        sample_seed = generation_seed(seed, image_id, index)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
            # VAE posterior sampling and unCLIP noise must use the same
            # seeded scope.  Otherwise paired Single-O/Multi-O runs receive
            # different initial latents before the diffusion process starts.
            vae = get_vae_features([source], runtime.pipe, runtime.diffusion_preprocess)
            generated = generate_image(
                runtime.pipe, vae, embedding.half(),
                t=int(cfg["diffusion_t"]),
                num_inference_steps=int(cfg["num_inference_steps"]),
                noise_level=int(cfg.get("noise_level", 0)),
                prompt=trigger_prompt,
                negative_prompt="low quality, ugly, unrealistic",
                guidance_scale=float(cfg["guidance_scale"]),
            )
            del vae
        evaluation = evaluate_generated(runtime, generated, spec, condition)
        valid = bool(evaluation["valid_attack_success"])
        image_path = ""
        if valid or save_failed:
            directory = output / "generated" / spec.name / condition
            directory.mkdir(parents=True, exist_ok=True)
            suffix = "success" if valid else "failed"
            path = directory / (
                f"seed{seed}_{image_id}_step{trigger_step}_g{index}_"
                f"{suffix}.png")
            generated.save(path)
            image_path = str(path)
        rows.append({
            "seed": seed, "spec": spec.name, "image_id": image_id,
            "condition": condition, "generation_index": index,
            "generation_seed": sample_seed, "trigger_step": trigger_step,
            "trigger_prompt": trigger_prompt, "generated_path": image_path,
            **evaluation,
        })
    return rows


def aggregate_result(spec: PairSpec, condition: str, seed: int, image_id: int,
                     source_path: str, embedding_success: bool,
                     trigger_step: int, trigger_prompt: str,
                     generations: Sequence[dict]) -> dict:
    def any_value(key: str) -> int:
        return int(any(as_int(row.get(key)) for row in generations))

    def any_float(key: str) -> float:
        values = [as_float(row.get(key)) for row in generations]
        values = [value for value in values if not math.isnan(value)]
        return float(any(value > 0 for value in values)) if values else float("nan")

    return {
        "seed": seed,
        "spec": spec.name,
        "image_id": int(image_id),
        "condition": condition,
        "source_path": source_path,
        "embedding_success": int(embedding_success),
        "trigger_step": trigger_step,
        "trigger_prompt": trigger_prompt,
        "generation_evaluated": int(bool(generations)),
        "num_generations_evaluated": len(generations),
        "query_flip_success": any_value("target_query_yes"),
        "valid_thr_success": any_value("valid_attack_success"),
        "atom_a_hit": any_value("atom_a_yes"),
        "atom_b_hit": any_float("atom_b_yes"),
        "atom_all_hit": any_value("atom_all_yes"),
        "valid_atom_a_success": any_value("valid_atom_a"),
        "valid_atom_b_success": any_float("valid_atom_b"),
        "valid_atom_all_success": any_value("valid_atom_all"),
    }


def run_condition(runtime: Runtime, dset, item: dict, spec: PairSpec,
                  condition: str, seed: int, cfg: dict, output: Path,
                  contrasts: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[dict, List[dict], List[dict]]:
    image, source_path = dset[int(item["image_id"])]
    embedding, embedding_success, trigger_step, trigger_prompt, traces = optimize_embedding(
        runtime, image, spec, condition, seed, int(item["image_id"]), cfg, contrasts)
    generations = generate_and_evaluate(
        runtime, image, embedding, spec, condition, seed,
        int(item["image_id"]), trigger_step, trigger_prompt, cfg, output)
    result = aggregate_result(
        spec, condition, seed, int(item["image_id"]), str(source_path), embedding_success,
        trigger_step, trigger_prompt, generations)
    return result, traces, generations


def run_seed(runtime: Runtime, dset, manifest: Sequence[dict],
             specs: Sequence[PairSpec], seed: int, cfg: dict,
             output: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("resume", True):
        checkpoint_results, checkpoint_traces, checkpoint_generations = \
            load_task_checkpoints(seed_dir)
        legacy_results = read_csv(seed_dir / "results.csv")
        legacy_traces = read_csv(seed_dir / "steps.csv")
        legacy_generations = read_csv(seed_dir / "generations.csv")
        checkpoint_keys = {task_key(row) for row in checkpoint_results}
        # A previous version may have CSV rows without task checkpoints.  Keep
        # those rows, while letting an atomic task checkpoint replace any
        # corresponding legacy rows rather than duplicating them.
        results = [row for row in legacy_results
                   if task_key(row) not in checkpoint_keys] + checkpoint_results
        traces = [row for row in legacy_traces
                  if task_key(row) not in checkpoint_keys] + checkpoint_traces
        generations = [row for row in legacy_generations
                       if task_key(row) not in checkpoint_keys] + checkpoint_generations
    else:
        results, traces, generations = [], [], []
    completed = {(row.get("spec"), as_int(row.get("image_id")), row.get("condition"))
                 for row in results}
    items_by_spec: Dict[str, List[dict]] = {spec.name: [] for spec in specs}
    for item in manifest:
        items_by_spec[item["spec"]].append(item)

    contrast_cache = {spec.name: contrast_embeddings(runtime, spec) for spec in specs}
    all_jobs = [(spec, item, condition)
                for spec in specs
                for item in items_by_spec[spec.name]
                for condition in CONDITIONS]
    completed_since_flush = 0
    with tqdm(total=len(all_jobs), desc=f"Stage 1 Exp 1 seed {seed}") as progress:
        for spec, item, condition in all_jobs:
            key = (spec.name, int(item["image_id"]), condition)
            if key in completed:
                progress.update(1)
                continue
            result, new_traces, new_generations = run_condition(
                runtime, dset, item, spec, condition, seed, cfg, seed_dir,
                contrast_cache[spec.name])
            results.append(result)
            traces.extend(new_traces)
            generations.extend(new_generations)
            save_json_atomic(task_checkpoint_path(
                seed_dir, spec.name, int(item["image_id"]), condition), {
                    "result": result,
                    "traces": new_traces,
                    "generations": new_generations,
                })
            completed.add(key)
            completed_since_flush += 1
            if completed_since_flush >= int(cfg["checkpoint_every"]):
                save_csv_atomic(seed_dir / "steps.csv", traces)
                save_csv_atomic(seed_dir / "generations.csv", generations)
                save_csv_atomic(seed_dir / "results.csv", results)
                completed_since_flush = 0
            gc.collect()
            torch.cuda.empty_cache()
            progress.update(1)
    # CSV files are reporting caches.  The per-task JSON files are the resume
    # source of truth, so all three caches can be materialized together here.
    save_csv_atomic(seed_dir / "steps.csv", traces)
    save_csv_atomic(seed_dir / "generations.csv", generations)
    save_csv_atomic(seed_dir / "results.csv", results)
    return results, traces, generations


def binomial_interval(values: Sequence[float]) -> Tuple[float, float, float]:
    binary = [int(bool(value)) for value in values if not math.isnan(float(value))]
    if not binary:
        return float("nan"), float("nan"), float("nan")
    estimate = float(np.mean(binary))
    interval = binomtest(sum(binary), len(binary)).proportion_ci(
        confidence_level=0.95, method="exact")
    return estimate, float(interval.low), float(interval.high)


def summary_row(rows: Sequence[dict], seed_label: object,
                condition: str) -> dict:
    group = [row for row in rows if row.get("condition") == condition]
    def metric(key: str) -> float:
        values = [as_float(row.get(key)) for row in group]
        values = [value for value in values if not math.isnan(value)]
        return float(np.mean(values)) if values else float("nan")
    def sr_ci(key: str) -> Tuple[float, float, float]:
        return binomial_interval([as_float(row.get(key)) for row in group])

    embedding, embedding_low, embedding_high = sr_ci("embedding_success")
    query, query_low, query_high = sr_ci("query_flip_success")
    valid, valid_low, valid_high = sr_ci("valid_thr_success")
    return {
        "seed": seed_label,
        "condition": condition,
        "n": len(group),
        "embedding_sr": embedding,
        "embedding_ci_low": embedding_low,
        "embedding_ci_high": embedding_high,
        "query_flip_sr": query,
        "query_flip_ci_low": query_low,
        "query_flip_ci_high": query_high,
        "valid_thr": valid,
        "valid_thr_ci_low": valid_low,
        "valid_thr_ci_high": valid_high,
        "atom_a_thr": metric("atom_a_hit"),
        "atom_b_thr": metric("atom_b_hit"),
        "atom_all_thr": metric("atom_all_hit"),
        "valid_atom_a_thr": metric("valid_atom_a_success"),
        "valid_atom_b_thr": metric("valid_atom_b_success"),
        "valid_atom_all_thr": metric("valid_atom_all_success"),
        "generated_image_n": sum(as_int(row.get("num_generations_evaluated")) for row in group),
        "embedding_success_n": sum(as_int(row.get("embedding_success")) for row in group),
        "query_flip_success_n": sum(as_int(row.get("query_flip_success")) for row in group),
        "valid_thr_success_n": sum(as_int(row.get("valid_thr_success")) for row in group),
    }


def generation_metrics(generations: Sequence[dict]) -> dict:
    """Summarize detector and MLLM outcomes over generated images only."""
    result = {"absence_eval_n": len(generations)}
    for key, output_key in (("absence_a", "absence_a_rate"),
                            ("absence_b", "absence_b_rate"),
                            ("absence_all", "absence_all_rate"),
                            ("target_query_yes", "query_flip_generation_rate"),
                            ("valid_attack_success", "valid_thr_generation_rate")):
        values = [as_float(row.get(key)) for row in generations]
        values = [value for value in values if not math.isnan(value)]
        result[output_key] = float(np.mean(values)) if values else float("nan")
    return result


def summarize_all(results: Sequence[dict], generations: Sequence[dict],
                  specs: Sequence[PairSpec], seeds: Sequence[int]) -> List[dict]:
    rows: List[dict] = []
    for seed in seeds:
        seed_rows = [row for row in results if as_int(row.get("seed")) == seed]
        for spec in specs:
            spec_rows = [row for row in seed_rows if row.get("spec") == spec.name]
            for condition in CONDITIONS:
                item = summary_row(
                    [row for row in spec_rows if row.get("condition") == condition],
                    seed, condition)
                item["spec"] = spec.name
                item.update(generation_metrics([
                    generation for generation in generations
                    if generation.get("spec") == spec.name
                    and generation.get("condition") == condition
                    and as_int(generation.get("seed")) == seed]))
                rows.append(item)
        for condition in CONDITIONS:
            seed_rows = [row for row in results
                         if as_int(row.get("seed")) == seed
                         and row.get("condition") == condition]
            item = summary_row(seed_rows, seed, condition)
            item["spec"] = "ALL"
            item.update(generation_metrics([
                generation for generation in generations
                if generation.get("condition") == condition
                and as_int(generation.get("seed")) == seed]))
            rows.append(item)
    for spec in specs:
        for condition in CONDITIONS:
            spec_rows = [row for row in results
                         if row.get("spec") == spec.name and row.get("condition") == condition]
            item = summary_row(spec_rows, "AGGREGATE", condition)
            item["spec"] = spec.name
            item.update(generation_metrics([
                generation for generation in generations
                if generation.get("spec") == spec.name
                and generation.get("condition") == condition]))
            # Mean/std over the three seed-specific rates, the primary result
            # for the requested multi-seed protocol.
            seed_items = [row for row in rows
                          if row["spec"] == spec.name and row["condition"] == condition]
            for key in ("embedding_sr", "query_flip_sr", "valid_thr",
                        "atom_a_thr", "atom_b_thr", "atom_all_thr",
                        "valid_atom_a_thr", "valid_atom_b_thr", "valid_atom_all_thr"):
                values = np.asarray([as_float(row[key]) for row in seed_items])
                values = values[~np.isnan(values)]
                item[f"{key}_seed_mean"] = float(values.mean()) if len(values) else float("nan")
                item[f"{key}_seed_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            rows.append(item)
    for condition in CONDITIONS:
        all_rows = [row for row in results if row.get("condition") == condition]
        item = summary_row(all_rows, "AGGREGATE", condition)
        item["spec"] = "ALL"
        item.update(generation_metrics([
            generation for generation in generations
            if generation.get("condition") == condition]))
        # The overall error bar is variation across the three random seeds.
        # First pool the five target specifications within each seed, then
        # calculate mean/std over exactly three seed-level rates.  Averaging
        # the 15 (spec, seed) rows would mix target-pair difficulty into the
        # quantity labelled "across seeds".
        seed_items = []
        for seed in seeds:
            seed_rows = [row for row in results
                         if as_int(row.get("seed")) == seed
                         and row.get("condition") == condition]
            seed_items.append(summary_row(seed_rows, seed, condition))
        for key in ("embedding_sr", "query_flip_sr", "valid_thr",
                    "atom_a_thr", "atom_b_thr", "atom_all_thr",
                    "valid_atom_a_thr", "valid_atom_b_thr", "valid_atom_all_thr"):
            values = np.asarray([as_float(row[key]) for row in seed_items])
            values = values[~np.isnan(values)]
            item[f"{key}_seed_mean"] = float(values.mean()) if len(values) else float("nan")
            item[f"{key}_seed_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        item["seed_level_n"] = len(seed_items)
        rows.append(item)
    return rows


def paired_bootstrap(before: np.ndarray, after: np.ndarray,
                     samples: int, seed: int) -> Tuple[float, float]:
    if len(before) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(before), size=(samples, len(before)))
    deltas = before[indices].mean(axis=1) - after[indices].mean(axis=1)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def clustered_paired_bootstrap(before: np.ndarray, after: np.ndarray,
                               clusters: Sequence[Tuple[str, int]],
                               samples: int, seed: int) -> Tuple[float, float]:
    """Bootstrap source-image clusters, keeping all seed runs together."""
    if len(before) == 0:
        return float("nan"), float("nan")
    grouped: Dict[Tuple[str, int], List[float]] = {}
    for cluster, difference in zip(clusters, before - after):
        grouped.setdefault(cluster, []).append(float(difference))
    cluster_deltas = np.asarray([
        np.mean(values) for values in grouped.values()], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(cluster_deltas),
                           size=(samples, len(cluster_deltas)))
    deltas = cluster_deltas[indices].mean(axis=1)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def paired_comparisons(results: Sequence[dict], cfg: dict,
                       specs: Sequence[PairSpec], seeds: Sequence[int]) -> List[dict]:
    comparisons: List[dict] = []
    for spec_name in [spec.name for spec in specs] + ["ALL"]:
        for seed_label in list(seeds) + ["ALL"]:
            group = [row for row in results
                     if (spec_name == "ALL" or row.get("spec") == spec_name)
                     and (seed_label == "ALL" or as_int(row.get("seed")) == seed_label)]
            lookup = {(as_int(row.get("seed")), row.get("spec"),
                       as_int(row.get("image_id")), row.get("condition")): row
                      for row in group}
            keys = sorted({(as_int(row.get("seed")), row.get("spec"),
                            as_int(row.get("image_id"))) for row in group})
            paired_keys = [
                (seed_key, spec_key, image_id)
                for seed_key, spec_key, image_id in keys
                if (seed_key, spec_key, image_id, "Single-O") in lookup
                and (seed_key, spec_key, image_id, "Multi-O") in lookup]
            pairs = [(lookup[(seed_key, spec_key, image_id, "Single-O")],
                      lookup[(seed_key, spec_key, image_id, "Multi-O")])
                     for seed_key, spec_key, image_id in paired_keys]
            for metric_name, key in (("Embedding", "embedding_success"),
                                     ("QueryFlip", "query_flip_success"),
                                     ("ValidTHR", "valid_thr_success")):
                before = np.asarray([as_float(pair[0].get(key), 0.0) for pair in pairs])
                after = np.asarray([as_float(pair[1].get(key), 0.0) for pair in pairs])
                lost = int(np.sum((before == 1) & (after == 0)))
                gained = int(np.sum((before == 0) & (after == 1)))
                if seed_label == "ALL":
                    cluster_ids = [(spec_key, image_id)
                                   for _, spec_key, image_id in paired_keys]
                    low, high = clustered_paired_bootstrap(
                        before, after, cluster_ids,
                        int(cfg["bootstrap_samples"]),
                        int(cfg["seeds"][0]) + len(comparisons))
                    bootstrap_unit = "source_image_cluster"
                    mcnemar_p = float("nan")
                else:
                    low, high = paired_bootstrap(
                        before, after, int(cfg["bootstrap_samples"]),
                        int(cfg["seeds"][0]) + len(comparisons))
                    bootstrap_unit = "source_image"
                    discordant = lost + gained
                    mcnemar_p = 1.0 if discordant == 0 else float(
                        binomtest(min(lost, gained), discordant, 0.5,
                                  alternative="two-sided").pvalue)
                comparisons.append({
                    "spec": spec_name, "seed": seed_label,
                    "metric": metric_name, "n": len(pairs),
                    "single_rate": float(before.mean()) if len(before) else float("nan"),
                    "multi_rate": float(after.mean()) if len(after) else float("nan"),
                    "paired_delta_single_minus_multi": float((before - after).mean())
                    if len(before) else float("nan"),
                    "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                    "bootstrap_unit": bootstrap_unit,
                    "cluster_n": len(set(cluster_ids)) if seed_label == "ALL" else len(pairs),
                    "lost_successes": lost, "gained_successes": gained,
                    "mcnemar_exact_p": mcnemar_p,
                })
    return comparisons


def save_figure(fig, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_results(results: Sequence[dict], traces: Sequence[dict],
                 generations: Sequence[dict], summary: Sequence[dict],
                 comparisons: Sequence[dict], specs: Sequence[PairSpec],
                 output: Path, cfg: dict) -> None:
    aggregate = {(row.get("spec"), row.get("condition")): row
                 for row in summary if row.get("seed") == "AGGREGATE"}
    with plt.rc_context(PAPER_STYLE):
        # Main outcomes, with the three-seed mean and standard deviation.
        fig, ax = plt.subplots(figsize=(6.2, 4.1))
        x = np.arange(len(CONDITIONS), dtype=float)
        series = (("Embedding threshold", "embedding_sr"),
                  ("Query flip", "query_flip_sr"),
                  ("Valid THR", "valid_thr"))
        offsets = (-0.20, 0.0, 0.20)
        markers = ("o", "s", "D")
        for (label, key), offset, marker in zip(series, offsets, markers):
            means = [as_float(aggregate[("ALL", condition)].get(f"{key}_seed_mean"))
                     for condition in CONDITIONS]
            stds = [as_float(aggregate[("ALL", condition)].get(f"{key}_seed_std"), 0.0)
                    for condition in CONDITIONS]
            positions = x + offset
            ax.errorbar(positions, means, yerr=stds, fmt=marker, markersize=6.5,
                        capsize=3.5, elinewidth=1.1, markeredgecolor="white",
                        markeredgewidth=0.7, color="#222222", label=label, zorder=3)
            for pos, value in zip(positions, means):
                ax.annotate(f"{value:.1%}", (pos, value), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=7.5)
        counts = [int(as_float(aggregate[("ALL", condition)]["n"]))
                  // len(cfg["seeds"]) for condition in CONDITIONS]
        ax.set_xticks(x, [f"{condition}\n$n={count}$"
                         for condition, count in zip(CONDITIONS, counts)])
        ax.set_ylim(-0.03, 1.08)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel("Original-sample rate (mean +/- SD across 3 seeds)")
        ax.set_title("Stage 1 Experiment 1 attack outcomes")
        ax.grid(axis="y")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
        fig.subplots_adjust(bottom=0.24)
        save_figure(fig, output, "attack_outcomes")

        # Paired Single-O minus Multi-O deltas by target specification.
        rows = [row for row in comparisons
                if row.get("seed") == "ALL" and row.get("spec") != "ALL"]
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
        for axis, metric in zip(axes, ("QueryFlip", "ValidTHR")):
            subset = [row for row in rows if row.get("metric") == metric]
            y = np.arange(len(subset))
            values = np.asarray([as_float(row["paired_delta_single_minus_multi"]) for row in subset])
            low = np.asarray([as_float(row["bootstrap_ci_low"]) for row in subset])
            high = np.asarray([as_float(row["bootstrap_ci_high"]) for row in subset])
            axis.errorbar(values, y, xerr=np.vstack((values - low, high - values)),
                          fmt="o", color="#222222", capsize=3, markersize=5)
            axis.axvline(0, color="#888888", linewidth=0.8)
            axis.set_yticks(y, [row["spec"] for row in subset])
            axis.xaxis.set_major_formatter(PercentFormatter(1.0))
            axis.set_title(metric)
            axis.grid(axis="x")
        axes[0].set_xlabel("Single-O minus Multi-O (paired bootstrap 95% CI)")
        axes[1].set_xlabel("Single-O minus Multi-O")
        fig.suptitle("Paired difficulty of the two-object query", y=1.02)
        fig.tight_layout()
        save_figure(fig, output, "paired_difficulty")

        # Atom diagnostics show whether Multi-O failures are complete-query
        # failures or failure to elicit one of the two object atoms.
        multi = aggregate[("ALL", "Multi-O")]
        metric_names = ("atom_a_thr", "atom_b_thr", "atom_all_thr",
                        "query_flip_sr", "valid_thr")
        labels = ("Atom A", "Atom B", "Both atoms", "Multi query", "Valid THR")
        means = [as_float(multi.get(f"{key}_seed_mean")) for key in metric_names]
        stds = [as_float(multi.get(f"{key}_seed_std"), 0.0) for key in metric_names]
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        positions = np.arange(len(labels))
        ax.errorbar(positions, means, yerr=stds, fmt="o", color=COLORS["Multi-O"],
                    capsize=3, markersize=6)
        for pos, value in zip(positions, means):
            ax.annotate(f"{value:.1%}", (pos, value), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(positions, labels, rotation=18)
        ax.set_ylim(-0.03, 1.08)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel("Original-sample rate (mean +/- SD)")
        ax.set_title("Multi-O query versus atom-level diagnostics")
        ax.grid(axis="y")
        fig.tight_layout()
        save_figure(fig, output, "multi_atom_diagnostics")

        # Early-stop-corrected embedding convergence.
        max_step = max((as_int(row.get("step")) for row in traces), default=-1)
        fig, ax = plt.subplots(figsize=(6.6, 4.1))
        for condition in CONDITIONS:
            condition_traces = [row for row in traces if row.get("condition") == condition]
            grouped: Dict[Tuple[int, str, int], Dict[int, float]] = {}
            for row in condition_traces:
                key = (as_int(row.get("seed")), row.get("spec"), as_int(row.get("image_id")))
                grouped.setdefault(key, {})[as_int(row.get("step"))] = as_float(row.get("yes_after"))
            matrix = np.full((len(grouped), max_step + 1), np.nan)
            for index, values in enumerate(grouped.values()):
                for step, value in values.items():
                    matrix[index, step] = value
                observed = np.flatnonzero(~np.isnan(matrix[index]))
                if len(observed):
                    for step in range(observed[0] + 1, matrix.shape[1]):
                        if np.isnan(matrix[index, step]):
                            matrix[index, step] = matrix[index, step - 1]
            if not matrix.size:
                continue
            mean = np.nanmean(matrix, axis=0)
            low = np.nanpercentile(matrix, 2.5, axis=0)
            high = np.nanpercentile(matrix, 97.5, axis=0)
            steps = np.arange(max_step + 1)
            ax.plot(steps, mean, color=COLORS[condition], linewidth=1.8, label=condition)
            ax.fill_between(steps, low, high, color=COLORS[condition], alpha=0.16, linewidth=0)
        ax.axhline(float(cfg["yes_threshold"]), color="#555555", linestyle="--",
                   linewidth=1, label=f"Threshold ({cfg['yes_threshold']})")
        ax.set_xlim(left=0); ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("Optimization step")
        ax.set_ylabel("Yes probability")
        ax.set_title("Single-query embedding convergence")
        ax.grid(axis="both"); ax.legend(loc="best")
        fig.tight_layout()
        save_figure(fig, output, "embedding_convergence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="rank candidates only; do not load Qwen, unCLIP or OWLv2")
    parser.add_argument("--force-manifest", action="store_true",
                        help="rebuild the paired source manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "resolved_config.json", cfg)
    specs = [PairSpec.from_dict(value) for value in cfg["target_specs"]]
    dset = load_coco_dataset(cfg["data_path"], split="train")
    if args.dry_run:
        runtime = Runtime(cfg, attack=False)
        for spec in specs:
            rank_candidates(runtime, dset, spec, int(cfg["candidate_top_k"]),
                            int(cfg["clip_selection_batch_size"]), output)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage 1 attacks")
    runtime = Runtime(cfg, attack=True)
    manifest_path = output / "paired_manifest.json"
    if manifest_path.exists() and not args.force_manifest:
        manifest = json.loads(manifest_path.read_text())
        validate_manifest(manifest, specs, int(cfg["num_images_per_spec"]))
    else:
        manifest = build_manifest(runtime, dset, specs, cfg, output)
        save_json(manifest_path, manifest)
    save_json(output / "target_specs.json", [spec.to_dict() for spec in specs])

    all_results: List[dict] = []
    all_traces: List[dict] = []
    all_generations: List[dict] = []
    for seed in cfg["seeds"]:
        seed_all(seed)
        results, traces, generations = run_seed(
            runtime, dset, manifest, specs, seed, cfg, output)
        all_results.extend(results)
        all_traces.extend(traces)
        all_generations.extend(generations)
    save_csv(output / "all_results.csv", all_results)
    save_csv(output / "all_steps.csv", all_traces)
    save_csv(output / "all_generations.csv", all_generations)
    summary = summarize_all(all_results, all_generations, specs, cfg["seeds"])
    comparisons = paired_comparisons(all_results, cfg, specs, cfg["seeds"])
    save_csv(output / "summary.csv", summary)
    save_csv(output / "paired_comparisons.csv", comparisons)
    plot_results(all_results, all_traces, all_generations, summary,
                 comparisons, specs, output / "figures", cfg)


if __name__ == "__main__":
    main()
