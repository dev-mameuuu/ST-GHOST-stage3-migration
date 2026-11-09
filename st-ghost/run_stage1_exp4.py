"""Stage 1 Experiment 4: shared-pool query-complexity evaluation.

The four conditions are optimized independently on the same Full-00 source
pool:

* Level-O: object query;
* Level-O+A: object plus one attribute query;
* Level-O+R: object-pair relation query;
* Level-O+A+R: the complete composite query.

Every optimization step samples one complete binary-query template, matching
the original GHOST prompt schedule.  The source embedding, optimization
budget, prompt RNG seed, and diffusion seed are paired across levels, while
the optimized embeddings remain independent.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
STAGE1 = ROOT / "st-ghost"
if str(STAGE1) not in sys.path:
    sys.path.insert(0, str(STAGE1))

from run_stage1 import (  # noqa: E402
    Runtime,
    TargetSpec,
    load_coco_dataset,
    parse_yes_no,
    save_csv,
    seed_all,
)
from utils import generate_image, get_vae_features  # noqa: E402


LEVELS = ("Level-O", "Level-O+A", "Level-O+R", "Level-O+A+R")
PROTOCOL_VERSION = "stage1_exp4_ghost_compatible_query_complexity_v4"
POOL_PROTOCOL = "ghost_random_query_no_v1"
COLORS = {
    "Level-O": "#0072B2",
    "Level-O+A": "#009E73",
    "Level-O+R": "#E69F00",
    "Level-O+A+R": "#CC79A7",
}
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

OBJECT_PROMPTS = (
    "Do you see a {a} in the image? Answer with 'Yes' or 'No'.",
    "Is there a {a} here? Answer with 'Yes' or 'No'.",
    "Does the image contain a {a}? Answer with 'Yes' or 'No'.",
    "Can you find a {a} in this picture? Answer with 'Yes' or 'No'.",
    "Would you say there is a {a} here? Answer with 'Yes' or 'No'.",
    "Is a {a} present in this image? Answer with 'Yes' or 'No'.",
)
ATTRIBUTE_PROMPTS = (
    "Is the {a} {attr}? Answer with 'Yes' or 'No'.",
    "Do you see a {a} that is {attr}? Answer with 'Yes' or 'No'.",
    "Would you say the {a} is {attr}? Answer with 'Yes' or 'No'.",
    "Does this image show a {attr} {a}? Answer with 'Yes' or 'No'.",
    "Can you find a {a} that looks {attr}? Answer with 'Yes' or 'No'.",
    "Is there a {attr} {a} in the image? Answer with 'Yes' or 'No'.",
)
RELATION_PROMPTS = (
    "Is the {a} {rel} the {b}? Answer with 'Yes' or 'No'.",
    "Do you see a {a} that is {rel} a {b}? Answer with 'Yes' or 'No'.",
    "Are the {a} and the {b} {rel} each other? Answer with 'Yes' or 'No'.",
    "Does the image show a {a} {rel} a {b}? Answer with 'Yes' or 'No'.",
    "Would you say the {a} is {rel} the {b}? Answer with 'Yes' or 'No'.",
    "Can you find a {a} {rel} a {b} here? Answer with 'Yes' or 'No'.",
)
FULL_PROMPTS = (
    "Is there a {attr_a} {a} {rel} a {attr_b} {b} in the image? Answer with 'Yes' or 'No'.",
    "Do you see a {attr_a} {a} {rel} a {attr_b} {b}? Answer with 'Yes' or 'No'.",
    "Does the image contain a {attr_a} {a} {rel} a {attr_b} {b}? Answer with 'Yes' or 'No'.",
    "Can you find a {attr_a} {a} {rel} a {attr_b} {b} in this picture? Answer with 'Yes' or 'No'.",
    "Would you say there is a {attr_a} {a} {rel} a {attr_b} {b} here? Answer with 'Yes' or 'No'.",
    "Is a {attr_a} {a} {rel} a {attr_b} {b} present in this image? Answer with 'Yes' or 'No'.",
)

# Held-out wording is evaluated for audit and post-generation generalization;
# only the canonical wording participates in Full-00 pool construction.
HELDOUT_TEMPLATES = {
    "Level-O": "Can you see any {a} in this picture? Answer only Yes or No.",
    "Level-O+A": "Can you see a {attr_a} {a} in this picture? Answer only Yes or No.",
    "Level-O+R": "Can you see a {a} that is {rel} a {b} in this picture? Answer only Yes or No.",
    "Level-O+A+R": "Can you see a {attr_a} {a} {rel} a {attr_b} {b} in this picture? Answer only Yes or No.",
}
ATOMIC_OBJECT = "Can you see any {a} in this picture? Answer only Yes or No."
ATOMIC_ATTRIBUTE = "Can you see a {attr} {a} in this picture? Answer only Yes or No."
ATOMIC_RELATION = "Can you see a {a} that is {rel} a {b} in this picture? Answer only Yes or No."


@dataclass(frozen=True)
class CompositeSpec:
    name: str
    objects: Tuple[str, str]
    attributes: Tuple[Tuple[str, str], Tuple[str, str]]
    relation: Tuple[str, str, str]

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CompositeSpec":
        name = str(value.get("name", "")).strip()
        objects = tuple(str(item).strip() for item in value.get("objects", ()))
        attributes = tuple(
            tuple(str(item).strip() for item in pair)
            for pair in value.get("attributes", ())
        )
        relation_values = tuple(
            str(item).strip() for item in value.get("relation", ())
        )
        if not name or len(objects) != 2 or objects[0] == objects[1]:
            raise ValueError("Each target spec needs two distinct objects and a name")
        if len(attributes) != 2 or any(len(pair) != 2 for pair in attributes):
            raise ValueError(f"{name} needs exactly two object-attribute pairs")
        if tuple(pair[0] for pair in attributes) != objects:
            raise ValueError(f"{name} attributes must follow the object order")
        if len(relation_values) != 3 or tuple(relation_values[::2]) != objects:
            raise ValueError(f"{name} relation must be [object_a, relation, object_b]")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Target spec name is not filename-safe: {name}")
        return cls(name, (objects[0], objects[1]),
                   (attributes[0], attributes[1]),
                   (relation_values[0], relation_values[1], relation_values[2]))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "objects": list(self.objects),
            "attributes": [list(pair) for pair in self.attributes],
            "relation": list(self.relation),
        }

    def as_target_spec(self) -> TargetSpec:
        return TargetSpec(
            self.name, self.objects,
            self.attributes, (self.relation,))


def _attribute_phrase(attribute: str, object_name: str) -> str:
    if attribute.startswith("made of ") or attribute.startswith("wearing "):
        return f"a {object_name} {attribute}"
    return f"a {attribute} {object_name}"


def _format_values(spec: CompositeSpec) -> dict:
    a, b = spec.objects
    attr_a, attr_b = spec.attributes[0][1], spec.attributes[1][1]
    _, rel, _ = spec.relation
    return {"a": a, "b": b, "attr_a": attr_a, "attr_b": attr_b, "rel": rel}


def optimization_questions(spec: CompositeSpec, level: str) -> Tuple[str, ...]:
    values = _format_values(spec)
    if level == "Level-O":
        return tuple(template.format(a=values["a"]) for template in OBJECT_PROMPTS)
    if level == "Level-O+A":
        return tuple(template.format(a=values["a"], attr=values["attr_a"])
                     for template in ATTRIBUTE_PROMPTS)
    if level == "Level-O+R":
        return tuple(template.format(**values) for template in RELATION_PROMPTS)
    if level == "Level-O+A+R":
        return tuple(template.format(**values) for template in FULL_PROMPTS)
    raise ValueError(f"Unknown level: {level}")


def canonical_question(spec: CompositeSpec, level: str) -> str:
    return optimization_questions(spec, level)[0]


def evaluation_question(spec: CompositeSpec, level: str) -> str:
    return HELDOUT_TEMPLATES[level].format(**_format_values(spec))


def active_objects(spec: CompositeSpec, level: str) -> Tuple[str, ...]:
    return (spec.objects[0],) if level in ("Level-O", "Level-O+A") else spec.objects


def active_fact_keys(spec: CompositeSpec, level: str) -> Tuple[str, ...]:
    keys = [f"obj:{spec.objects[0]}"]
    if level == "Level-O+A":
        keys.append(f"attr:{spec.objects[0]}:{spec.attributes[0][1]}")
    elif level == "Level-O+R":
        keys.extend((f"obj:{spec.objects[1]}", "rel:" + ":".join(spec.relation)))
    elif level == "Level-O+A+R":
        keys.extend((f"obj:{spec.objects[1]}",
                     f"attr:{spec.objects[0]}:{spec.attributes[0][1]}",
                     f"attr:{spec.objects[1]}:{spec.attributes[1][1]}",
                     "rel:" + ":".join(spec.relation)))
    return tuple(keys)


def atomic_questions(spec: CompositeSpec) -> Dict[str, str]:
    a, b = spec.objects
    attr_a, attr_b = spec.attributes[0][1], spec.attributes[1][1]
    _, rel, _ = spec.relation
    return {
        f"obj:{a}": ATOMIC_OBJECT.format(a=a),
        f"obj:{b}": ATOMIC_OBJECT.format(a=b),
        f"attr:{a}:{attr_a}": ATOMIC_ATTRIBUTE.format(a=a, attr=attr_a),
        f"attr:{b}:{attr_b}": ATOMIC_ATTRIBUTE.format(a=b, attr=attr_b),
        "rel:" + ":".join(spec.relation): ATOMIC_RELATION.format(
            a=a, b=b, rel=rel),
    }


def resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    cfg.setdefault("seeds", [cfg.get("seed", 42)])
    cfg["seeds"] = [int(seed) for seed in cfg["seeds"]]
    if len(cfg["seeds"]) != 3:
        raise ValueError("Stage 1 Experiment 4 expects exactly three seeds")
    cfg.setdefault("num_images_per_spec", 200)
    cfg.setdefault("candidate_top_k", 1500)
    cfg.setdefault("clip_selection_batch_size", 64)
    cfg.setdefault("resume", True)
    cfg.setdefault("save_failed_images", False)
    cfg.setdefault("checkpoint_every", 16)
    cfg.setdefault("bootstrap_samples", 10000)
    cfg.setdefault("permutation_samples", 10000)
    cfg.setdefault("diffusion_prompt_mode", "ghost_trigger")
    cfg.setdefault("noise_level", 0)
    defaults = {
        "num_steps": 100, "learning_rate": 0.1, "lambda_clip": 15.0,
        "lambda_reg": 10.0, "yes_threshold": 0.8, "num_generations": 4,
        "diffusion_t": 30, "num_inference_steps": 50,
        "guidance_scale": 5.0, "owl_threshold": 0.5,
    }
    for key, value in defaults.items():
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
    if int(cfg["num_steps"]) <= 0 or int(cfg["num_generations"]) <= 0:
        raise ValueError("num_steps and num_generations must be positive")
    if int(cfg["checkpoint_every"]) <= 0:
        raise ValueError("checkpoint_every must be positive")
    if not 0.0 < float(cfg["yes_threshold"]) <= 1.0:
        raise ValueError("yes_threshold must be in (0, 1]")
    if not 0.0 <= float(cfg["owl_threshold"]) <= 1.0:
        raise ValueError("owl_threshold must be in [0, 1]")
    if cfg.get("diffusion_prompt_mode") != "ghost_trigger":
        raise ValueError(
            "GHOST-compatible Experiment 4 requires "
            "diffusion_prompt_mode=ghost_trigger")
    if not cfg.get("target_specs"):
        raise ValueError("target_specs cannot be empty")
    specs = [CompositeSpec.from_dict(value) for value in cfg["target_specs"]]
    cfg["target_specs"] = [spec.to_dict() for spec in specs]
    value = cfg.get("levels", LEVELS)
    if isinstance(value, str):
        value = tuple(item.strip() for item in value.split(",") if item.strip())
    elif value is None:
        value = ()
    requested_levels = tuple(value)
    if requested_levels != LEVELS:
        raise ValueError(
            "Experiment 4 requires all levels in fixed order: "
            + ", ".join(LEVELS))
    cfg["levels"] = list(LEVELS)
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


def as_optional_bool(value: object) -> Optional[bool]:
    """Parse a persisted decision while preserving an unknown/null state."""
    if value is None or value == "" or str(value).lower() in {"nan", "none", "null"}:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    try:
        return bool(int(float(value)))
    except (TypeError, ValueError):
        return None


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def save_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False))
    os.replace(temporary, path)


def save_csv_atomic(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    temporary = path.with_name(path.name + ".tmp")
    save_csv(temporary, rows)
    os.replace(temporary, path)


def ensure_compatible_output_dir(output: Path) -> None:
    """Refuse to mix checkpoints and summaries from different protocols."""
    signature_path = output / "experiment_signature.json"
    if not signature_path.exists():
        return
    metadata = json.loads(signature_path.read_text())
    existing = str(metadata.get("protocol_version", ""))
    if existing != PROTOCOL_VERSION:
        raise RuntimeError(
            f"Output directory {output} contains protocol "
            f"{existing or 'unknown'}, but this script uses {PROTOCOL_VERSION}. "
            "Use a new output_dir (for example, "
            "outputs/stage1_exp4_main_v4); existing results were not changed.")


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def generation_seed(seed: int, spec_name: str, image_id: int,
                    generation_index: int) -> int:
    # The level is intentionally excluded: paired levels use the same source
    # VAE sample and diffusion noise for the same attempt index.
    return stable_seed("stage1-exp4-generation", seed, spec_name,
                       image_id, generation_index)


def task_checkpoint_path(seed_dir: Path, spec: str, image_id: int,
                         level: str) -> Path:
    return seed_dir / "checkpoints" / f"{spec}__{image_id}__{level}.json"


def task_key(row: Mapping[str, object]) -> Tuple[str, int, str]:
    return (str(row.get("spec", "")), as_int(row.get("image_id")),
            str(row.get("level", "")))


def experiment_signature(cfg: Mapping[str, object],
                         manifest: Sequence[Mapping[str, object]]) -> str:
    relevant = {
        "protocol_version": PROTOCOL_VERSION,
        "levels": cfg["levels"],
        "target_specs": cfg["target_specs"],
        "manifest": list(manifest),
        "num_steps": cfg["num_steps"],
        "learning_rate": cfg["learning_rate"],
        "lambda_clip": cfg["lambda_clip"],
        "lambda_reg": cfg["lambda_reg"],
        "yes_threshold": cfg["yes_threshold"],
        "num_generations": cfg["num_generations"],
        "diffusion_t": cfg["diffusion_t"],
        "num_inference_steps": cfg["num_inference_steps"],
        "guidance_scale": cfg["guidance_scale"],
        "diffusion_prompt_mode": cfg["diffusion_prompt_mode"],
        "owl_threshold": cfg["owl_threshold"],
    }
    encoded = json.dumps(relevant, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_task_checkpoints(seed_dir: Path, signature: str
                          ) -> Tuple[List[dict], List[dict], List[dict]]:
    results: List[dict] = []
    traces: List[dict] = []
    generations: List[dict] = []
    checkpoint_dir = seed_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return results, traces, generations
    seen = set()
    for path in sorted(checkpoint_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("experiment_signature") != signature:
            raise RuntimeError(
                f"Checkpoint signature mismatch: {path}; use a new output_dir")
        result = payload["result"]
        key = task_key(result)
        if key in seen:
            raise RuntimeError(f"Duplicate checkpoint for {key}")
        seen.add(key)
        results.append(result)
        traces.extend(payload.get("traces", []))
        generations.extend(payload.get("generations", []))
    return results, traces, generations


@torch.no_grad()
def rank_candidates(runtime: Runtime, dset, spec: CompositeSpec,
                    top_k: int, batch_size: int, output: Path) -> List[dict]:
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
            first, second = (float(scores[index, 0]), float(scores[index, 1]))
            rows.append({
                "spec": spec.name, "image_id": int(image_id),
                "clip_sim_a": first, "clip_sim_b": second,
                "base_score": min(first, second),
                "tie_score": (first + second) / 2.0,
                "annotation_a_present": 0,
                "annotation_b_present": 0,
            })
    rows.sort(key=lambda row: (-row["base_score"], -row["tie_score"],
                               row["image_id"]))
    ranked = rows[:top_k]
    save_csv(output / f"clip_ranked_{spec.name}.csv", ranked)
    return ranked


def _prescreen_value(runtime: Runtime, image: Image.Image,
                     question: str) -> Tuple[str, Optional[bool], float]:
    return runtime.victim.answer(image, question)


def build_manifest(runtime: Runtime, dset, specs: Sequence[CompositeSpec],
                   cfg: Mapping[str, object], output: Path) -> List[dict]:
    manifest: List[dict] = []
    required = int(cfg["num_images_per_spec"])
    levels = tuple(cfg["levels"])
    for spec in specs:
        ranked = rank_candidates(runtime, dset, spec, int(cfg["candidate_top_k"]),
                                 int(cfg["clip_selection_batch_size"]), output)
        accepted: List[dict] = []
        audit: List[dict] = []
        for rank, candidate in enumerate(
                tqdm(ranked, desc=f"Prescreen {spec.name} Full-00"), start=1):
            image, source_path = dset[int(candidate["image_id"])]
            row = {**candidate, "rank": rank, "source_path": str(source_path)}
            answers: Dict[str, Tuple[str, Optional[bool], float]] = {}

            def prescreen(question: str) -> Tuple[str, Optional[bool], float]:
                if question not in answers:
                    answers[question] = _prescreen_value(runtime, image, question)
                return answers[question]

            ghost_all_no = True
            canonical_all_no = True
            prompt_index = stable_seed(
                "stage1-exp4-ghost-prescreen", spec.name,
                candidate["image_id"]) % len(OBJECT_PROMPTS)
            for level in levels:
                canonical = canonical_question(spec, level)
                heldout = evaluation_question(spec, level)
                ghost_question = optimization_questions(
                    spec, level)[prompt_index]
                ca, cd, cp = prescreen(canonical)
                ha, hd, hp = prescreen(heldout)
                ga, gd, gp = prescreen(ghost_question)
                prefix = level.lower().replace("+", "_").replace("-", "_")
                row.update({
                    f"{prefix}_ghost_prescreen_question": ghost_question,
                    f"{prefix}_ghost_prescreen_answer": ga,
                    f"{prefix}_ghost_prescreen_decision": gd,
                    f"{prefix}_ghost_prescreen_yes_probability": gp,
                    f"{prefix}_canonical_question": canonical,
                    f"{prefix}_canonical_answer": ca,
                    f"{prefix}_canonical_decision": cd,
                    f"{prefix}_canonical_yes_probability": cp,
                    f"{prefix}_heldout_question": heldout,
                    f"{prefix}_heldout_answer": ha,
                    f"{prefix}_heldout_decision": hd,
                    f"{prefix}_heldout_yes_probability": hp,
                })
                # Original GHOST skips a source only when its sampled attack
                # prompt is already answered Yes.  Apply that rule once per
                # level while retaining canonical and held-out audit fields.
                ghost_all_no = ghost_all_no and gd is False
                canonical_all_no = canonical_all_no and cd is False
            for fact_key, question in atomic_questions(spec).items():
                answer, decision, probability = _prescreen_value(
                    runtime, image, question)
                key = re.sub(r"[^A-Za-z0-9]+", "_", fact_key).strip("_")
                row[f"atomic_{key}_decision"] = decision
                row[f"atomic_{key}_yes_probability"] = probability
                row[f"atomic_{key}_answer"] = answer
            row["full00_ghost_prescreen_no"] = int(ghost_all_no)
            row["full00_canonical_prescreen_no"] = int(canonical_all_no)
            row["full00_prescreen_no"] = int(ghost_all_no)
            audit.append(row)
            if ghost_all_no:
                pool_index = len(accepted)
                accepted.append(row)
                manifest_row = {
                    "spec": spec.name, "image_id": int(candidate["image_id"]),
                    "source_path": str(source_path), "pool_index": pool_index,
                    "rank": rank, "clip_sim_a": candidate["clip_sim_a"],
                    "clip_sim_b": candidate["clip_sim_b"],
                    "base_score": candidate["base_score"],
                    "tie_score": candidate["tie_score"],
                    "annotation_a_present": 0,
                    "annotation_b_present": 0,
                    "pool_protocol": POOL_PROTOCOL,
                }
                # Preserve all source decisions for supplemental true-flip
                # metrics; the primary GHOST ASR uses raw trigger-query Yes.
                for level in levels:
                    prefix = level.lower().replace("+", "_").replace("-", "_")
                    manifest_row[f"{prefix}_ghost_prescreen_question"] = row.get(
                        f"{prefix}_ghost_prescreen_question")
                    manifest_row[f"{prefix}_ghost_initial_decision"] = row.get(
                        f"{prefix}_ghost_prescreen_decision")
                    manifest_row[f"{prefix}_canonical_initial_decision"] = row.get(
                        f"{prefix}_canonical_decision")
                    manifest_row[f"{prefix}_heldout_initial_decision"] = row.get(
                        f"{prefix}_heldout_decision")
                manifest.append(manifest_row)
                if len(accepted) == required:
                    break
        save_csv(output / f"prescreen_{spec.name}.csv", audit)
        if len(accepted) != required:
            raise RuntimeError(
                f"Only {len(accepted)}/{required} Full-00 images for {spec.name}; "
                "increase candidate_top_k or revise the target specification")
    return manifest


def validate_manifest(manifest: Sequence[Mapping[str, object]],
                      specs: Sequence[CompositeSpec], required: int) -> None:
    names = {spec.name for spec in specs}
    counts = {name: 0 for name in names}
    seen = set()
    for row in manifest:
        spec = str(row.get("spec", ""))
        image_id = as_int(row.get("image_id"), -1)
        if spec not in names:
            raise ValueError(f"Unknown target spec in manifest: {spec}")
        key = (spec, image_id)
        if key in seen:
            raise ValueError(f"Duplicate Full-00 image: {key}")
        seen.add(key)
        if as_int(row.get("annotation_a_present")) or as_int(
                row.get("annotation_b_present")):
            raise ValueError(f"Full-00 annotation mismatch for {key}")
        counts[spec] += 1
    bad = {name: count for name, count in counts.items() if count != required}
    if bad:
        raise ValueError(f"Manifest must contain {required} rows per spec: {bad}")


def manifest_has_query_audit(manifest: Sequence[Mapping[str, object]],
                             levels: Sequence[str]) -> bool:
    """Check that a manifest follows the GHOST-compatible v4 pool protocol."""
    if not manifest:
        return False
    required = []
    for level in levels:
        prefix = level.lower().replace("+", "_").replace("-", "_")
        required.extend((f"{prefix}_ghost_prescreen_question",
                         f"{prefix}_ghost_initial_decision",
                         f"{prefix}_heldout_initial_decision",
                         f"{prefix}_canonical_initial_decision"))
    return all(
        row.get("pool_protocol") == POOL_PROTOCOL
        and all(key in row for key in required)
        and all(
            str(row.get(
                f"{level.lower().replace('+', '_').replace('-', '_')}"
                "_ghost_prescreen_question", "")).strip()
            and as_optional_bool(row.get(
                f"{level.lower().replace('+', '_').replace('-', '_')}"
                "_ghost_initial_decision")) is False
            for level in levels
        )
        for row in manifest
    )


def _contrast_target(spec: CompositeSpec, level: str) -> Tuple[str, Tuple[str, ...]]:
    """Return the direct target phrase and objects used to mine captions."""
    a, b = spec.objects
    attr_a, attr_b = spec.attributes[0][1], spec.attributes[1][1]
    _, rel, _ = spec.relation
    if level == "Level-O":
        return f"a {a}", (a,)
    if level == "Level-O+A":
        return _attribute_phrase(attr_a, a), (a,)
    if level == "Level-O+R":
        return f"a {a} {rel} a {b}", (a, b)
    if level == "Level-O+A+R":
        return f"a {attr_a} {a} {rel} a {attr_b} {b}", (a, b)
    raise ValueError(f"Unknown level: {level}")


def _contrast_captions(runtime: Runtime, objects: Sequence[str],
                       max_captions: int = 5) -> List[str]:
    """Mine captions consistently for object-only and pair targets."""
    enhancer = getattr(runtime.clip, "text_enhancer", None)
    if enhancer is None or max_captions <= 0:
        return []
    if len(objects) == 1:
        miner = getattr(enhancer, "mine_captions_with_object", None)
        if callable(miner):
            return list(miner(objects[0], max_captions=max_captions))
        return []
    caption_data = getattr(enhancer, "coco_captions", None) or {}
    required = tuple(str(obj).lower() for obj in objects)
    captions: List[str] = []
    for annotation in caption_data.get("annotations", []):
        caption = str(annotation.get("caption", ""))
        lowered = caption.lower()
        if all(obj in lowered for obj in required):
            captions.append(caption)
            if len(captions) == max_captions:
                break
    return captions


def _raw_clip_texts(runtime: Runtime, texts: Sequence[str]) -> torch.Tensor:
    """Encode raw CLIP text features when the runtime exposes its model."""
    clip = runtime.clip
    model = getattr(clip, "model", None)
    tokenizer = getattr(clip, "tokenizer", None)
    device = getattr(clip, "device", None)
    if model is not None and tokenizer is not None:
        tokens = tokenizer(list(texts))
        if device is not None:
            tokens = tokens.to(device)
        with torch.no_grad():
            return model.encode_text(tokens).float()
    # Lightweight test doubles may only provide the public normalized helper.
    return clip.texts(texts).float()


def contrast_embedding(runtime: Runtime, spec: CompositeSpec,
                        level: str) -> torch.Tensor:
    """Build one uniform direct/template/COCO contrast for every level.

    The direct description receives 0.3 weight, three contextual templates
    receive 0.4 in total, and up to five mined captions receive 0.3 in total.
    If no captions are available, the remaining weights are renormalized.
    """
    target, caption_objects = _contrast_target(spec, level)
    contextual = (
        f"A scene featuring {target}",
        f"An image showing {target}",
        f"A photograph with {target}",
    )
    captions = _contrast_captions(runtime, caption_objects, max_captions=5)
    texts = [f"a photo of {target}", *contextual, *captions]
    embeddings = _raw_clip_texts(runtime, texts)
    weights = [0.3] + [0.4 / len(contextual)] * len(contextual)
    if captions:
        weights.extend([0.3 / len(captions)] * len(captions))
    weight_tensor = torch.tensor(
        weights, device=embeddings.device, dtype=embeddings.dtype)
    weight_tensor /= weight_tensor.sum()
    combined = torch.sum(embeddings * weight_tensor.unsqueeze(-1), dim=0)
    return F.normalize(combined, dim=0).detach()


def contrast_loss(embedding: torch.Tensor, contrast: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(
        embedding.float(), contrast.float().unsqueeze(0), dim=-1).mean()


def verify_objects(runtime: Runtime, image: Image.Image,
                   spec: CompositeSpec, level: str) -> dict:
    detector_level = ("Single-O"
                      if level in ("Level-O", "Level-O+A")
                      else "Multi-O")
    result = runtime.absence.verify(
        image, spec.as_target_spec(), detector_level)
    presence = result["presence"]
    a, b = spec.objects
    a_present = bool(presence.get(a, False))
    b_evaluated = b in active_objects(spec, level)
    b_present = bool(presence.get(b, False)) if b_evaluated else None
    active = active_objects(spec, level)
    required_absent = all(not bool(presence.get(obj, False)) for obj in active)
    return {
        "object_a_present": int(a_present),
        "object_a_absent": int(not a_present),
        "object_a_detection_score": float(result["max_scores"].get(a, 0.0)),
        "object_b_evaluated": int(b_evaluated),
        "object_b_present": int(b_present) if b_evaluated else float("nan"),
        "object_b_absent": int(not b_present) if b_evaluated else float("nan"),
        "object_b_detection_score": (
            float(result["max_scores"].get(b, 0.0))
            if b_evaluated else float("nan")),
        "required_objects_absent": int(required_absent),
    }


def evaluate_generated(runtime: Runtime, image: Image.Image,
                       spec: CompositeSpec, level: str,
                       trigger_prompt: str,
                       heldout_initial_decision: Optional[bool] = None,
                       canonical_initial_decision: Optional[bool] = None) -> dict:
    if not trigger_prompt:
        raise ValueError("trigger_prompt cannot be empty")
    answers: Dict[str, Tuple[str, Optional[bool], float]] = {}

    def answer(question: str) -> Tuple[str, Optional[bool], float]:
        if question not in answers:
            answers[question] = runtime.victim.answer(image, question)
        return answers[question]

    trigger_answer, trigger_decision, trigger_probability = answer(trigger_prompt)
    heldout = evaluation_question(spec, level)
    canonical = canonical_question(spec, level)
    heldout_answer, heldout_decision, heldout_probability = answer(heldout)
    canonical_answer, canonical_decision, canonical_probability = answer(canonical)
    # A held-out Yes is a flip only when the source-image held-out decision
    # was explicitly No.  Unknown/null source states are not counted.
    heldout_flip_success = int(
        heldout_initial_decision is False and heldout_decision is True)
    # Canonical wording is diagnostic under the GHOST-compatible pool.  Count
    # only a verified source No -> generated-image Yes as a true flip.
    canonical_flip_success = int(
        canonical_initial_decision is False and canonical_decision is True)
    row = {
        "target_question": trigger_prompt,
        "target_answer": trigger_answer,
        "target_decision": trigger_decision,
        "target_yes_probability": trigger_probability,
        "target_query_yes": int(trigger_decision is True),
        "trigger_question": trigger_prompt,
        "trigger_answer": trigger_answer,
        "trigger_decision": trigger_decision,
        "trigger_yes_probability": trigger_probability,
        "trigger_query_yes": int(trigger_decision is True),
        "heldout_question": heldout,
        "heldout_answer": heldout_answer,
        "heldout_decision": heldout_decision,
        "heldout_yes_probability": heldout_probability,
        "heldout_query_yes": int(heldout_decision is True),
        "heldout_initial_decision": heldout_initial_decision,
        "heldout_flip_success": heldout_flip_success,
        "canonical_question": canonical,
        "canonical_answer": canonical_answer,
        "canonical_decision": canonical_decision,
        "canonical_yes_probability": canonical_probability,
        "canonical_query_yes": int(canonical_decision is True),
        "canonical_initial_decision": canonical_initial_decision,
        "canonical_flip_success": canonical_flip_success,
        # The original GHOST ASR counts a Yes to the prompt that triggered
        # generation.  Keep the historical query_flip name as an alias even
        # though this is a raw trigger-query success, not necessarily a flip.
        "ghost_query_success": int(trigger_decision is True),
        "query_flip_success": int(trigger_decision is True),
    }
    atomic = atomic_questions(spec)
    active = set(active_fact_keys(spec, level))
    for fact_key, question in atomic.items():
        answer_text, decision, probability = answer(question)
        safe = re.sub(r"[^A-Za-z0-9]+", "_", fact_key).strip("_")
        row[f"atomic_{safe}_question"] = question
        row[f"atomic_{safe}_answer"] = answer_text
        row[f"atomic_{safe}_decision"] = decision
        row[f"atomic_{safe}_yes_probability"] = probability
        row[f"atomic_{safe}_yes"] = int(decision is True)
        row[f"active_atomic_{safe}_yes"] = int(
            decision is True) if fact_key in active else float("nan")
    objects = verify_objects(runtime, image, spec, level)
    row.update(objects)
    active_yes = []
    for fact_key in active:
        safe = re.sub(r"[^A-Za-z0-9]+", "_", fact_key).strip("_")
        active_yes.append(bool(row.get(f"atomic_{safe}_yes", 0)))
    row["active_atomic_all_yes"] = int(all(active_yes))
    row["valid_attack_success"] = int(
        row["target_query_yes"] and row["required_objects_absent"])
    row["ghost_aligned_success"] = row["valid_attack_success"]
    row["heldout_generalization_success"] = int(
        row["valid_attack_success"] and row["heldout_flip_success"])
    row["valid_active_atomic_all"] = int(
        row["active_atomic_all_yes"] and row["required_objects_absent"])
    return row


def generate_and_evaluate(runtime: Runtime, source: Image.Image,
                          embedding: torch.Tensor, spec: CompositeSpec,
                          level: str, seed: int, image_id: int,
                          trigger_step: int, trigger_prompt: str,
                          generation_index: int, cfg: Mapping[str, object],
                          output: Path,
                          heldout_initial_decision: Optional[bool] = None,
                          canonical_initial_decision: Optional[bool] = None) -> dict:
    runtime.load_diffusion()
    sample_seed = generation_seed(seed, spec.name, image_id, generation_index)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)
        vae = get_vae_features([source], runtime.pipe,
                                runtime.diffusion_preprocess)
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
    evaluation = evaluate_generated(
        runtime, generated, spec, level, trigger_prompt,
        heldout_initial_decision=heldout_initial_decision,
        canonical_initial_decision=canonical_initial_decision)
    valid = bool(evaluation["valid_attack_success"])
    generated_path = ""
    if valid or bool(cfg.get("save_failed_images", False)):
        directory = output / "generated" / spec.name / level
        directory.mkdir(parents=True, exist_ok=True)
        suffix = "success" if valid else "failed"
        path = directory / (
            f"seed{seed}_{image_id}_step{trigger_step}_g{generation_index}_"
            f"{suffix}.png")
        generated.save(path)
        generated_path = str(path)
    return {
        "seed": seed, "spec": spec.name, "image_id": image_id,
        "level": level, "generation_index": generation_index,
        "generation_seed": sample_seed, "trigger_step": trigger_step,
        "trigger_prompt": trigger_prompt, "generated_path": generated_path,
        "diffusion_prompt": trigger_prompt,
        **evaluation,
    }


def optimize_embedding(runtime: Runtime, source: Image.Image,
                       spec: CompositeSpec, level: str, seed: int,
                       image_id: int, cfg: Mapping[str, object],
                       contrast: torch.Tensor, output: Path,
                       heldout_initial_decision: Optional[bool] = None,
                       canonical_initial_decision: Optional[bool] = None
                       ) -> Tuple[bool, int, str, List[dict], List[dict]]:
    with torch.no_grad():
        initial = runtime.clip.images([source]).detach()
    embedding = nn.Parameter(initial.clone())
    optimizer = torch.optim.AdamW([embedding], lr=float(cfg["learning_rate"]))
    prompts = optimization_questions(spec, level)
    # Excluding the level gives all four paired conditions the same sequence
    # of template indices while retaining GHOST's random prompt scheduling.
    prompt_rng = random.Random(
        stable_seed("stage1-exp4-prompt", seed, spec.name, image_id))
    threshold = float(cfg["yes_threshold"])
    generation_budget = int(cfg["num_generations"])
    trace: List[dict] = []
    generations: List[dict] = []
    embedding_success = False
    trigger_step, trigger_prompt = -1, ""
    for step in range(int(cfg["num_steps"])):
        prompt = prompts[prompt_rng.randrange(len(prompts))]
        tokens = runtime.projector.model(embedding).half().squeeze(0)
        probability_before = runtime.victim.projected_probability(
            source, tokens, prompt)
        attack_loss = -torch.log(probability_before.clamp_min(1e-8))
        clip_loss = contrast_loss(embedding, contrast)
        regularizer = F.mse_loss(embedding.float(), initial.float())
        loss = (attack_loss + float(cfg["lambda_clip"]) * clip_loss
                + float(cfg["lambda_reg"]) * regularizer)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            updated_tokens = runtime.projector.model(embedding).half().squeeze(0)
            probability_after = runtime.victim.projected_probability(
                source, updated_tokens, prompt)
            yes_after = float(probability_after.detach().cpu())
        threshold_reached = yes_after >= threshold
        generation_attempted = False
        generation_success = False
        generation_index = -1
        if threshold_reached and len(generations) < generation_budget:
            if not embedding_success:
                trigger_step, trigger_prompt = step, prompt
            embedding_success = True
            generation_index = len(generations)
            initial_decision_kwargs = {}
            if heldout_initial_decision is not None:
                initial_decision_kwargs[
                    "heldout_initial_decision"] = heldout_initial_decision
            if canonical_initial_decision is not None:
                initial_decision_kwargs[
                    "canonical_initial_decision"] = canonical_initial_decision
            if not initial_decision_kwargs:
                # Preserve the compact call shape used by older test doubles.
                generation = generate_and_evaluate(
                    runtime, source, embedding.detach(), spec, level, seed,
                    image_id, step, prompt, generation_index, cfg, output)
            else:
                generation = generate_and_evaluate(
                    runtime, source, embedding.detach(), spec, level, seed,
                    image_id, step, prompt, generation_index, cfg, output,
                    **initial_decision_kwargs)
            generations.append(generation)
            generation_attempted = True
            generation_success = bool(generation["valid_attack_success"])
        trace.append({
            "seed": seed, "spec": spec.name, "image_id": image_id,
            "level": level, "step": step, "prompt": prompt,
            "loss": float(loss.detach()), "attack_loss": float(attack_loss.detach()),
            "clip_loss": float(clip_loss.detach()),
            "reg_loss": float(regularizer.detach()),
            "yes_before": float(probability_before.detach().cpu()),
            "yes_after": yes_after,
            "threshold_reached": int(threshold_reached),
            "generation_attempted": int(generation_attempted),
            "generation_index": generation_index,
            "valid_attack_success": int(generation_success),
        })
        if generation_success or len(generations) >= generation_budget:
            break
    return embedding_success, trigger_step, trigger_prompt, trace, generations


def _any_binary(generations: Sequence[Mapping[str, object]], key: str) -> int:
    return int(any(as_int(row.get(key)) for row in generations))


def _any_optional(generations: Sequence[Mapping[str, object]], key: str) -> float:
    values = [as_float(row.get(key)) for row in generations]
    values = [value for value in values if math.isfinite(value)]
    return float(any(bool(value) for value in values)) if values else float("nan")


def _heldout_flip_value(row: Mapping[str, object]) -> int:
    persisted = row.get("heldout_flip_success")
    if persisted is not None and persisted != "":
        return as_int(persisted)
    initial = as_optional_bool(row.get("heldout_initial_decision"))
    generated = as_optional_bool(row.get("heldout_query_yes"))
    return int(initial is False and generated is True)


def _any_heldout_flip(generations: Sequence[Mapping[str, object]]) -> int:
    return int(any(_heldout_flip_value(row) for row in generations))


def _canonical_flip_value(row: Mapping[str, object]) -> int:
    persisted = row.get("canonical_flip_success")
    if persisted is not None and persisted != "":
        return as_int(persisted)
    # Compatibility for hand-built/legacy generation rows.  Actual v4 rows
    # always carry canonical_initial_decision and canonical_flip_success.
    if "canonical_initial_decision" not in row:
        return as_int(row.get("canonical_query_yes"))
    initial = as_optional_bool(row.get("canonical_initial_decision"))
    generated = as_optional_bool(row.get("canonical_query_yes"))
    return int(initial is False and generated is True)


def _any_canonical_flip(generations: Sequence[Mapping[str, object]]) -> int:
    return int(any(_canonical_flip_value(row) for row in generations))


def aggregate_result(spec: CompositeSpec, level: str, seed: int,
                     item: Mapping[str, object], embedding_success: bool,
                     trigger_step: int, trigger_prompt: str,
                     generations: Sequence[Mapping[str, object]]) -> dict:
    successful = next((row for row in generations
                       if as_int(row.get("valid_attack_success"))), None)
    return {
        "seed": seed, "spec": spec.name, "image_id": as_int(item["image_id"]),
        "level": level, "source_path": str(item.get("source_path", "")),
        "pool_index": as_int(item.get("pool_index")),
        "clip_base_score": as_float(item.get("base_score")),
        "heldout_initial_decision": item.get(
            f"{level.lower().replace('+', '_').replace('-', '_')}"
            "_heldout_initial_decision"),
        "canonical_initial_decision": item.get(
            f"{level.lower().replace('+', '_').replace('-', '_')}"
            "_canonical_initial_decision"),
        "embedding_success": int(embedding_success),
        "trigger_step": trigger_step, "trigger_prompt": trigger_prompt,
        "generation_evaluated": int(bool(generations)),
        "num_generations_evaluated": len(generations),
        "ghost_query_success": _any_binary(generations, "target_query_yes"),
        "query_flip_success": _any_binary(generations, "target_query_yes"),
        "heldout_query_flip_success": _any_heldout_flip(generations),
        "canonical_query_flip_success": _any_canonical_flip(generations),
        "valid_thr_success": _any_binary(generations, "valid_attack_success"),
        "heldout_generalization_success": _any_binary(
            generations, "heldout_generalization_success"),
        "required_objects_absent": _any_binary(
            generations, "required_objects_absent"),
        "active_atomic_all_yes": _any_binary(generations, "active_atomic_all_yes"),
        "valid_active_atomic_all": _any_binary(
            generations, "valid_active_atomic_all"),
        "object_a_hit": _any_binary(generations, "object_a_present"),
        "object_b_evaluated": _any_binary(generations, "object_b_evaluated"),
        "object_b_hit": _any_optional(generations, "object_b_present"),
        "object_a_absence": _any_binary(generations, "object_a_absent"),
        "object_b_absence": _any_optional(generations, "object_b_absent"),
        "last_trigger_step": as_int(successful.get("trigger_step"), -1)
        if successful else -1,
        "successful_trigger_prompt": str(successful.get("trigger_prompt", ""))
        if successful else "",
    }


def run_condition(runtime: Runtime, dset, item: Mapping[str, object],
                  spec: CompositeSpec, level: str, seed: int,
                  cfg: Mapping[str, object], output: Path,
                  contrast: torch.Tensor) -> Tuple[dict, List[dict], List[dict]]:
    image, _ = dset[int(item["image_id"])]
    prefix = level.lower().replace("+", "_").replace("-", "_")
    heldout_initial_decision = as_optional_bool(
        item.get(f"{prefix}_heldout_initial_decision"))
    canonical_initial_decision = as_optional_bool(
        item.get(f"{prefix}_canonical_initial_decision"))
    embedding_success, trigger_step, trigger_prompt, traces, generations = \
        optimize_embedding(runtime, image, spec, level, seed,
                           int(item["image_id"]), cfg, contrast, output,
                           heldout_initial_decision,
                           canonical_initial_decision)
    result = aggregate_result(
        spec, level, seed, item, embedding_success, trigger_step,
        trigger_prompt, generations)
    return result, traces, generations


def run_seed(runtime: Runtime, dset, manifest: Sequence[dict],
             specs: Sequence[CompositeSpec], seed: int,
             cfg: Mapping[str, object], output: Path,
             signature: str) -> Tuple[List[dict], List[dict], List[dict]]:
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("resume", True):
        results, traces, generations = load_task_checkpoints(seed_dir, signature)
    else:
        results, traces, generations = [], [], []
    completed = {task_key(row) for row in results}
    items_by_spec: Dict[str, List[dict]] = {spec.name: [] for spec in specs}
    for item in manifest:
        items_by_spec[str(item["spec"])].append(item)
    for rows in items_by_spec.values():
        rows.sort(key=lambda row: as_int(row.get("pool_index")))
    spec_by_name = {spec.name: spec for spec in specs}
    contrasts = {
        (spec.name, level): contrast_embedding(runtime, spec, level)
        for spec in specs for level in cfg["levels"]
    }
    jobs = [(spec_by_name[str(item["spec"])], item, level)
            for item in manifest for level in cfg["levels"]]
    completed_since_flush = 0
    with tqdm(total=len(jobs), desc=f"Stage 1 Exp 4 seed {seed}") as progress:
        for spec, item, level in jobs:
            key = (spec.name, as_int(item["image_id"]), level)
            if key in completed:
                progress.update(1)
                continue
            result, new_traces, new_generations = run_condition(
                runtime, dset, item, spec, level, seed, cfg, seed_dir,
                contrasts[(spec.name, level)])
            results.append(result)
            traces.extend(new_traces)
            generations.extend(new_generations)
            save_json_atomic(task_checkpoint_path(
                seed_dir, spec.name, int(item["image_id"]), level), {
                    "experiment_signature": signature,
                    "result": result, "traces": new_traces,
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
    save_csv_atomic(seed_dir / "steps.csv", traces)
    save_csv_atomic(seed_dir / "generations.csv", generations)
    save_csv_atomic(seed_dir / "results.csv", results)
    return results, traces, generations


def binomial_interval(values: Sequence[object]) -> Tuple[float, float, float]:
    binary = [int(bool(as_int(value))) for value in values
              if math.isfinite(as_float(value))]
    if not binary:
        return float("nan"), float("nan"), float("nan")
    interval = binomtest(sum(binary), len(binary)).proportion_ci(
        confidence_level=0.95, method="exact")
    return (float(np.mean(binary)), float(interval.low), float(interval.high))


def summary_row(rows: Sequence[Mapping[str, object]], seed: object,
                level: str) -> dict:
    def metric(key: str) -> Tuple[float, float, float]:
        return binomial_interval([row.get(key) for row in rows])

    fields = {
        "embedding_success": "embedding_sr",
        "ghost_query_success": "ghost_query_sr",
        "query_flip_success": "query_flip_sr",
        "heldout_query_flip_success": "heldout_query_flip_sr",
        "canonical_query_flip_success": "canonical_query_flip_sr",
        "valid_thr_success": "valid_thr",
        "heldout_generalization_success": "heldout_generalization_sr",
        "active_atomic_all_yes": "active_atomic_all_sr",
        "valid_active_atomic_all": "valid_active_atomic_all_sr",
        "required_objects_absent": "object_state_preservation_sr",
    }
    result = {"seed": seed, "level": level, "n": len(rows)}
    for source, target in fields.items():
        rate, low, high = metric(source)
        result[target] = rate
        result[f"{target}_ci_low"] = low
        result[f"{target}_ci_high"] = high
    result.update({
        "generated_image_n": sum(as_int(row.get("num_generations_evaluated"))
                                  for row in rows),
        "embedding_success_n": sum(as_int(row.get("embedding_success"))
                                    for row in rows),
        "query_flip_success_n": sum(as_int(row.get("query_flip_success"))
                                     for row in rows),
        "valid_thr_success_n": sum(as_int(row.get("valid_thr_success"))
                                    for row in rows),
    })
    for key in ("object_a_hit", "object_b_evaluated", "object_b_hit",
                "object_a_absence", "object_b_absence"):
        values = [as_float(row.get(key)) for row in rows]
        values = [value for value in values if math.isfinite(value)]
        result[key + "_sr"] = float(np.mean(values)) if values else float("nan")
    return result


def generation_summary(generations: Sequence[Mapping[str, object]]) -> dict:
    result = {"generation_eval_n": len(generations)}
    for source, target in (
        ("target_query_yes", "query_flip_generation_rate"),
        ("target_query_yes", "trigger_query_generation_rate"),
        ("canonical_flip_success", "canonical_flip_generation_rate"),
        ("heldout_query_yes", "heldout_query_generation_rate"),
        ("heldout_flip_success", "heldout_flip_generation_rate"),
        ("valid_attack_success", "valid_thr_generation_rate"),
        ("required_objects_absent", "object_absence_generation_rate"),
    ):
        values = [as_float(row.get(source)) for row in generations]
        values = [value for value in values if math.isfinite(value)]
        result[target] = float(np.mean(values)) if values else float("nan")
    return result


def clustered_metric_bootstrap(
        results: Sequence[Mapping[str, object]], level: str, metric_key: str,
        samples: int = 10000, seed: int = 0,
        spec_name: Optional[str] = None) -> Tuple[float, float]:
    """Return a 95% bootstrap CI clustered by (spec, image_id).

    Results from the three random seeds are averaged within each source-image
    cluster before resampling, so repeated evaluations of one image do not
    inflate the effective sample size.
    """
    grouped: Dict[Tuple[str, int], List[float]] = {}
    for row in results:
        if row.get("level") != level:
            continue
        row_spec = str(row.get("spec", ""))
        if spec_name not in (None, "ALL") and row_spec != spec_name:
            continue
        value = as_float(row.get(metric_key))
        if math.isfinite(value):
            cluster = (row_spec, as_int(row.get("image_id"), -1))
            grouped.setdefault(cluster, []).append(value)
    cluster_values = np.asarray(
        [float(np.mean(values)) for values in grouped.values()], dtype=float)
    if not len(cluster_values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        cluster_values, size=(max(1, int(samples)), len(cluster_values)),
        replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def summarize_all(results: Sequence[dict], generations: Sequence[dict],
                  specs: Sequence[CompositeSpec], seeds: Sequence[int],
                  levels: Sequence[str], bootstrap_samples: int = 10000,
                  bootstrap_seed: int = 0) -> List[dict]:
    rows: List[dict] = []
    for seed in seeds:
        for spec in list(specs) + [None]:
            spec_name = spec.name if spec else "ALL"
            for level in levels:
                group = [row for row in results
                         if as_int(row.get("seed")) == seed
                         and row.get("level") == level
                         and (spec is None or row.get("spec") == spec_name)]
                item = summary_row(group, seed, level)
                item["spec"] = spec_name
                item.update(generation_summary([
                    row for row in generations
                    if as_int(row.get("seed")) == seed
                    and row.get("level") == level
                    and (spec is None or row.get("spec") == spec_name)]))
                rows.append(item)
    for spec in list(specs) + [None]:
        spec_name = spec.name if spec else "ALL"
        for level in levels:
            group = [row for row in results
                     if row.get("level") == level
                     and (spec is None or row.get("spec") == spec_name)]
            item = summary_row(group, "AGGREGATE", level)
            item["spec"] = spec_name
            item.update(generation_summary([
                row for row in generations
                if row.get("level") == level
                and (spec is None or row.get("spec") == spec_name)]))
            seed_items = [row for row in rows
                          if row["seed"] in seeds
                          and row["level"] == level
                          and row["spec"] == spec_name]
            for key in ("embedding_sr", "ghost_query_sr", "query_flip_sr", "valid_thr",
                        "heldout_query_flip_sr", "canonical_query_flip_sr"):
                values = np.asarray([as_float(row.get(key)) for row in seed_items])
                values = values[np.isfinite(values)]
                item[f"{key}_seed_mean"] = float(values.mean()) if len(values) else float("nan")
                item[f"{key}_seed_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            cluster_metrics = {
                "embedding_sr": "embedding_success",
                "ghost_query_sr": "ghost_query_success",
                "query_flip_sr": "query_flip_success",
                "valid_thr": "valid_thr_success",
                "heldout_query_flip_sr": "heldout_query_flip_success",
                "canonical_query_flip_sr": "canonical_query_flip_success",
            }
            for metric_name, source_key in cluster_metrics.items():
                low, high = clustered_metric_bootstrap(
                    results, level, source_key, bootstrap_samples,
                    bootstrap_seed + len(rows), spec_name=spec_name)
                item[f"{metric_name}_cluster_ci_low"] = low
                item[f"{metric_name}_cluster_ci_high"] = high
            item["seed_level_n"] = len(seed_items)
            rows.append(item)
    return rows


def clustered_bootstrap_delta(before: np.ndarray, after: np.ndarray,
                              clusters: Sequence[Tuple[str, int]],
                              samples: int, seed: int) -> Tuple[float, float]:
    grouped: Dict[Tuple[str, int], List[float]] = {}
    for cluster, delta in zip(clusters, before - after):
        grouped.setdefault(cluster, []).append(float(delta))
    values = np.asarray([np.mean(value) for value in grouped.values()])
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    draws = values[indices].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def paired_comparisons(results: Sequence[dict], cfg: Mapping[str, object],
                       specs: Sequence[CompositeSpec], seeds: Sequence[int],
                       levels: Sequence[str]) -> List[dict]:
    comparisons: List[dict] = []
    pairs = [(levels[i], levels[j]) for i in range(len(levels))
             for j in range(i + 1, len(levels))]
    for spec_name in [spec.name for spec in specs] + ["ALL"]:
        for seed_label in list(seeds) + ["ALL"]:
            group = [row for row in results
                     if (spec_name == "ALL" or row.get("spec") == spec_name)
                     and (seed_label == "ALL" or as_int(row.get("seed")) == seed_label)]
            lookup = {(as_int(row.get("seed")), row.get("spec"),
                       as_int(row.get("image_id")), row.get("level")): row
                      for row in group}
            keys = sorted({(as_int(row.get("seed")), row.get("spec"),
                            as_int(row.get("image_id"))) for row in group})
            for first, second in pairs:
                paired = [key for key in keys
                          if (key + (first,)) in lookup
                          and (key + (second,)) in lookup]
                for metric_name, metric_key in (
                        ("Embedding", "embedding_success"),
                        ("GhostQuery", "query_flip_success"),
                        ("ValidTHR", "valid_thr_success")):
                    before = np.asarray([
                        as_float(lookup[key + (first,)].get(metric_key), 0.0)
                        for key in paired])
                    after = np.asarray([
                        as_float(lookup[key + (second,)].get(metric_key), 0.0)
                        for key in paired])
                    lost = int(np.sum((before == 1) & (after == 0)))
                    gained = int(np.sum((before == 0) & (after == 1)))
                    if seed_label == "ALL":
                        clusters = [(key[1], key[2]) for key in paired]
                        low, high = clustered_bootstrap_delta(
                            before, after, clusters,
                            int(cfg["bootstrap_samples"]),
                            int(cfg["seeds"][0]) + len(comparisons))
                        pvalue = float("nan")
                        unit = "source_image_cluster"
                    else:
                        differences = before - after
                        if len(differences):
                            rng = np.random.default_rng(
                                int(cfg["seeds"][0]) + len(comparisons))
                            indices = rng.integers(
                                0, len(differences),
                                size=(int(cfg["bootstrap_samples"]), len(differences)))
                            draws = differences[indices].mean(axis=1)
                            low, high = np.percentile(draws, [2.5, 97.5])
                        else:
                            low = high = float("nan")
                        discordant = lost + gained
                        pvalue = (1.0 if not discordant else float(
                            binomtest(min(lost, gained), discordant,
                                      0.5).pvalue))
                        unit = "source_image"
                    comparisons.append({
                        "spec": spec_name, "seed": seed_label,
                        "comparison": f"{first}_minus_{second}",
                        "metric": metric_name, "n": len(paired),
                        "first_rate": float(before.mean()) if len(before) else float("nan"),
                        "second_rate": float(after.mean()) if len(after) else float("nan"),
                        "delta_first_minus_second": float((before - after).mean())
                        if len(before) else float("nan"),
                        "bootstrap_ci_low": float(low),
                        "bootstrap_ci_high": float(high),
                        "bootstrap_unit": unit,
                        "cluster_n": len(set((key[1], key[2]) for key in paired))
                        if seed_label == "ALL" else len(paired),
                        "lost_successes": lost, "gained_successes": gained,
                        "mcnemar_exact_p": pvalue,
                    })
    return comparisons


def transition_rows(results: Sequence[dict], specs: Sequence[CompositeSpec],
                    seeds: Sequence[int], levels: Sequence[str]) -> List[dict]:
    rows: List[dict] = []
    for spec_name in [spec.name for spec in specs] + ["ALL"]:
        for seed_label in list(seeds) + ["ALL"]:
            group = [row for row in results
                     if (spec_name == "ALL" or row.get("spec") == spec_name)
                     and (seed_label == "ALL" or as_int(row.get("seed")) == seed_label)]
            lookup = {(as_int(row.get("seed")), row.get("spec"),
                       as_int(row.get("image_id")), row.get("level")): row
                      for row in group}
            keys = sorted({(as_int(row.get("seed")), row.get("spec"),
                            as_int(row.get("image_id"))) for row in group})
            for first in levels:
                for second in levels:
                    if first == second:
                        continue
                    paired = [key for key in keys
                              if key + (first,) in lookup and key + (second,) in lookup]
                    first_values = np.asarray([
                        as_int(lookup[key + (first,)].get("valid_thr_success"))
                        for key in paired])
                    second_values = np.asarray([
                        as_int(lookup[key + (second,)].get("valid_thr_success"))
                        for key in paired])
                    from_only = int(np.sum((first_values == 1) & (second_values == 0)))
                    to_only = int(np.sum((first_values == 0) & (second_values == 1)))
                    both = int(np.sum((first_values == 1) & (second_values == 1)))
                    neither = int(np.sum((first_values == 0) & (second_values == 0)))
                    rows.append({
                        "spec": spec_name, "seed": seed_label,
                        "from_level": first, "to_level": second,
                        "metric": "ValidTHR", "n": len(paired),
                        "both_success": both, "from_only": from_only,
                        "to_only": to_only, "neither": neither,
                        "from_only_rate": from_only / len(paired) if paired else float("nan"),
                        "to_only_rate": to_only / len(paired) if paired else float("nan"),
                    })
    return rows


def save_figure(fig, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_results(summary: Sequence[dict], traces: Sequence[dict],
                 comparisons: Sequence[dict], output: Path,
                 cfg: Mapping[str, object]) -> None:
    aggregate = {(row.get("spec"), row.get("level")): row
                 for row in summary if row.get("seed") == "AGGREGATE"}
    levels = tuple(cfg["levels"])
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        x = np.arange(len(levels), dtype=float)
        for label, key, marker in (
                ("Embedding threshold", "embedding_sr", "o"),
                ("GHOST trigger-query success", "query_flip_sr", "s"),
                ("Valid THR", "valid_thr", "D")):
            values = [as_float(aggregate[("ALL", level)].get(
                f"{key}_seed_mean", aggregate[("ALL", level)].get(key)))
                      for level in levels]
            low = [as_float(aggregate[("ALL", level)].get(
                f"{key}_cluster_ci_low"), value) for level, value in zip(levels, values)]
            high = [as_float(aggregate[("ALL", level)].get(
                f"{key}_cluster_ci_high"), value) for level, value in zip(levels, values)]
            yerr = np.vstack((np.asarray(values) - np.asarray(low),
                              np.asarray(high) - np.asarray(values)))
            ax.errorbar(x, values, yerr=yerr, fmt=marker, color="#222222",
                        capsize=3, markersize=6, label=label)
            for pos, value in zip(x, values):
                ax.annotate(f"{value:.1%}", (pos, value), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=7.5)
        ax.set_xticks(x, [level.replace("Level-", "") for level in levels])
        ax.set_ylim(-0.03, 1.08)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel("Original-sample rate (mean; 95% clustered bootstrap CI)")
        ax.set_title("Stage 1 Experiment 4 query complexity")
        ax.grid(axis="y")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
        fig.subplots_adjust(bottom=0.23)
        save_figure(fig, output, "attack_outcomes")

        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        for level in levels:
            grouped: Dict[Tuple[int, str, int], Dict[int, float]] = {}
            for row in traces:
                if row.get("level") != level:
                    continue
                key = (as_int(row.get("seed")), str(row.get("spec")),
                       as_int(row.get("image_id")))
                grouped.setdefault(key, {})[as_int(row.get("step"))] = as_float(
                    row.get("yes_after"))
            max_step = max((max(values) for values in grouped.values()), default=-1)
            if max_step < 0:
                continue
            matrix = np.full((len(grouped), max_step + 1), np.nan)
            for index, values in enumerate(grouped.values()):
                for step, value in values.items():
                    matrix[index, step] = value
                observed = np.flatnonzero(np.isfinite(matrix[index]))
                if len(observed):
                    for step in range(observed[0] + 1, matrix.shape[1]):
                        if not np.isfinite(matrix[index, step]):
                            matrix[index, step] = matrix[index, step - 1]
            x_steps = np.arange(max_step + 1)
            ax.plot(x_steps, np.nanmean(matrix, axis=0), color=COLORS[level],
                    linewidth=1.6, label=level)
        ax.axhline(float(cfg["yes_threshold"]), color="#555555", linestyle="--",
                   linewidth=1, label="threshold")
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("Optimization step")
        ax.set_ylabel("Yes probability")
        ax.set_title("Embedding convergence by query complexity")
        ax.grid(axis="both")
        ax.legend(loc="best")
        fig.tight_layout()
        save_figure(fig, output, "embedding_convergence")

        subset = [row for row in comparisons
                  if row.get("seed") == "ALL" and row.get("spec") == "ALL"
                  and row.get("metric") == "ValidTHR"]
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        y = np.arange(len(subset))
        values = np.asarray([as_float(row["delta_first_minus_second"]) for row in subset])
        low = np.asarray([as_float(row["bootstrap_ci_low"]) for row in subset])
        high = np.asarray([as_float(row["bootstrap_ci_high"]) for row in subset])
        if len(subset):
            ax.errorbar(values, y, xerr=np.vstack((values - low, high - values)),
                        fmt="o", color="#222222", capsize=3, markersize=5)
            ax.set_yticks(y, [row["comparison"] for row in subset])
        ax.axvline(0, color="#888888", linewidth=0.8)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("Earlier-level minus later-level Valid-THR")
        ax.set_title("Paired difficulty differences")
        ax.grid(axis="x")
        fig.tight_layout()
        save_figure(fig, output, "paired_complexity_differences")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="rank Full-00 candidates without loading attack models")
    parser.add_argument("--force-manifest", action="store_true",
                        help="rebuild the Full-00 manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output = Path(cfg["output_dir"])
    ensure_compatible_output_dir(output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "resolved_config.json", cfg)
    specs = [CompositeSpec.from_dict(value) for value in cfg["target_specs"]]
    dset = load_coco_dataset(cfg["data_path"], split="train")
    if args.dry_run:
        runtime = Runtime(cfg, attack=False)
        for spec in specs:
            rank_candidates(runtime, dset, spec, int(cfg["candidate_top_k"]),
                            int(cfg["clip_selection_batch_size"]), output)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage 1 Experiment 4")
    runtime = Runtime(cfg, attack=True)
    manifest_path = output / "full00_manifest.json"
    if manifest_path.exists() and not args.force_manifest:
        manifest = json.loads(manifest_path.read_text())
        validate_manifest(manifest, specs, int(cfg["num_images_per_spec"]))
        if not manifest_has_query_audit(manifest, cfg["levels"]):
            raise RuntimeError(
                "Existing Full-00 manifest predates the GHOST-compatible v4 "
                "pool protocol; rerun with --force-manifest or use a new "
                "output_dir")
    else:
        manifest = build_manifest(runtime, dset, specs, cfg, output)
        validate_manifest(manifest, specs, int(cfg["num_images_per_spec"]))
        save_json(manifest_path, manifest)
    save_json(output / "target_specs.json", [spec.to_dict() for spec in specs])
    save_json(output / "levels.json", cfg["levels"])
    signature = experiment_signature(cfg, manifest)
    save_json(output / "experiment_signature.json", {
        "protocol_version": PROTOCOL_VERSION, "sha256": signature,
    })
    all_results: List[dict] = []
    all_traces: List[dict] = []
    all_generations: List[dict] = []
    for seed in cfg["seeds"]:
        seed_all(seed)
        results, traces, generations = run_seed(
            runtime, dset, manifest, specs, seed, cfg, output, signature)
        all_results.extend(results)
        all_traces.extend(traces)
        all_generations.extend(generations)
    save_csv(output / "all_results.csv", all_results)
    save_csv(output / "all_steps.csv", all_traces)
    save_csv(output / "all_generations.csv", all_generations)
    summary = summarize_all(
        all_results, all_generations, specs, cfg["seeds"], cfg["levels"],
        bootstrap_samples=int(cfg["bootstrap_samples"]),
        bootstrap_seed=int(cfg["seeds"][0]))
    comparisons = paired_comparisons(
        all_results, cfg, specs, cfg["seeds"], cfg["levels"])
    transitions = transition_rows(
        all_results, specs, cfg["seeds"], cfg["levels"])
    save_csv(output / "summary.csv", summary)
    save_csv(output / "paired_comparisons.csv", comparisons)
    save_csv(output / "transition_matrix.csv", transitions)
    plot_results(summary, all_traces, comparisons, output / "figures", cfg)


if __name__ == "__main__":
    main()
