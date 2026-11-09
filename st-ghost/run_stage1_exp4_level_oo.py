"""Stage 1 Experiment 4 paired Level-OO control.

Level-OO asks whether both target objects are present, without attributes or
relations.  It reuses the exact source-image manifest from the completed
GHOST-compatible v4 run.  The source pool is never re-filtered for the new
query: its initial Level-OO answer is recorded as an audit field and strict
No-to-Yes flips are reported as a supplemental metric.

The source CLIP image, optimization budget, prompt-index RNG, and diffusion
seed schedule are deliberately aligned with ``run_stage1_exp4.py``.  The
optimization and generation code is kept local because the four-Level runner
intentionally rejects unknown levels.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
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


LEVEL = "Level-OO"
PROTOCOL_VERSION = "stage1_exp4_level_oo_paired_v1"
SOURCE_PROTOCOL_VERSION = exp4.PROTOCOL_VERSION
SOURCE_OUTPUT_DEFAULT = ROOT / "outputs" / "stage1_exp4_main_v4_retry"
OUTPUT_DEFAULT = ROOT / "outputs" / "stage1_exp4_level_oo_v1"

OO_PROMPTS = (
    "Do you see both {a} and {b} in the image? Answer with 'Yes' or 'No'.",
    "Is there both {a} and {b} here? Answer with 'Yes' or 'No'.",
    "Does the image contain both {a} and {b}? Answer with 'Yes' or 'No'.",
    "Can you find both {a} and {b} in this picture? Answer with 'Yes' or 'No'.",
    "Would you say there are both {a} and {b} here? Answer with 'Yes' or 'No'.",
    "Are both {a} and {b} present in this image? Answer with 'Yes' or 'No'.",
)
OO_HELDOUT_TEMPLATE = (
    "Can you see both {a} and {b} in this picture? Answer only Yes or No."
)


def _article(noun: str) -> str:
    return ("an " if noun[:1].lower() in "aeiou" else "a ") + noun


def _oo_values(spec: exp4.CompositeSpec) -> dict:
    a, b = spec.objects
    return {"a": _article(a), "b": _article(b)}


def oo_optimization_questions(spec: exp4.CompositeSpec) -> Tuple[str, ...]:
    values = _oo_values(spec)
    return tuple(template.format(**values) for template in OO_PROMPTS)


def oo_canonical_question(spec: exp4.CompositeSpec) -> str:
    return oo_optimization_questions(spec)[0]


def oo_heldout_question(spec: exp4.CompositeSpec) -> str:
    return OO_HELDOUT_TEMPLATE.format(**_oo_values(spec))


def oo_atomic_questions(spec: exp4.CompositeSpec) -> Dict[str, str]:
    a, b = spec.objects
    return {
        f"obj:{a}": exp4.ATOMIC_OBJECT.format(a=a),
        f"obj:{b}": exp4.ATOMIC_OBJECT.format(a=b),
    }


def oo_active_fact_keys(spec: exp4.CompositeSpec) -> Tuple[str, ...]:
    return tuple(f"obj:{obj}" for obj in spec.objects)


def oo_contrast_embedding(runtime: exp4.Runtime,
                          spec: exp4.CompositeSpec) -> torch.Tensor:
    """Use the same weighted direct/template/COCO contrast as v4."""
    a, b = spec.objects
    target = f"{_article(a)} and {_article(b)}"
    contextual = (
        f"A scene featuring {target}",
        f"An image showing {target}",
        f"A photograph with {target}",
    )
    captions = exp4._contrast_captions(runtime, spec.objects, max_captions=5)
    texts = [f"a photo of {target}", *contextual, *captions]
    embeddings = exp4._raw_clip_texts(runtime, texts)
    weights = [0.3] + [0.4 / len(contextual)] * len(contextual)
    if captions:
        weights.extend([0.3 / len(captions)] * len(captions))
    weight_tensor = torch.tensor(
        weights, device=embeddings.device, dtype=embeddings.dtype)
    weight_tensor /= weight_tensor.sum()
    combined = torch.sum(embeddings * weight_tensor.unsqueeze(-1), dim=0)
    return F.normalize(combined, dim=0).detach()


def _safe_fact(fact_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", fact_key).strip("_")


def _answer(runtime: exp4.Runtime, image: Image.Image, question: str,
            answers: Dict[str, Tuple[str, Optional[bool], float]]) \
        -> Tuple[str, Optional[bool], float]:
    if question not in answers:
        answers[question] = runtime.victim.answer(image, question)
    return answers[question]


def build_oo_manifest(runtime: exp4.Runtime, dset,
                      source_manifest: Sequence[Mapping[str, object]],
                      specs: Sequence[exp4.CompositeSpec],
                      output: Path) -> List[dict]:
    """Attach Level-OO audit decisions without changing source membership."""
    spec_by_name = {spec.name: spec for spec in specs}
    rows: List[dict] = []
    for item in tqdm(source_manifest, desc="Level-OO source audit"):
        spec_name = str(item.get("spec", ""))
        spec = spec_by_name[spec_name]
        image_id = exp4.as_int(item.get("image_id"), -1)
        image, source_path = dset[image_id]
        expected_path = Path(str(item.get("source_path", ""))).resolve()
        actual_path = Path(str(source_path)).resolve()
        if expected_path != actual_path:
            raise RuntimeError(
                f"Shared-pool source mismatch for {spec_name}/{image_id}: "
                f"manifest={expected_path}, dataset={actual_path}")
        answers: Dict[str, Tuple[str, Optional[bool], float]] = {}
        prompt_index = exp4.stable_seed(
            "stage1-exp4-level-oo-prescreen", spec_name, image_id) % len(OO_PROMPTS)
        ghost_question = oo_optimization_questions(spec)[prompt_index]
        canonical_question = oo_canonical_question(spec)
        heldout_question = oo_heldout_question(spec)
        ghost_answer, ghost_decision, ghost_probability = _answer(
            runtime, image, ghost_question, answers)
        canonical_answer, canonical_decision, canonical_probability = _answer(
            runtime, image, canonical_question, answers)
        heldout_answer, heldout_decision, heldout_probability = _answer(
            runtime, image, heldout_question, answers)
        row = {
            **dict(item),
            "level": LEVEL,
            "pool_protocol": str(item.get("pool_protocol", "")),
            "oo_prescreen_question": ghost_question,
            "oo_prescreen_answer": ghost_answer,
            "oo_prescreen_initial_decision": ghost_decision,
            "oo_prescreen_yes_probability": ghost_probability,
            "oo_prescreen_no": int(ghost_decision is False),
            "oo_canonical_question": canonical_question,
            "oo_canonical_initial_decision": canonical_decision,
            "oo_canonical_yes_probability": canonical_probability,
            "oo_canonical_answer": canonical_answer,
            "oo_heldout_question": heldout_question,
            "oo_heldout_initial_decision": heldout_decision,
            "oo_heldout_yes_probability": heldout_probability,
            "oo_heldout_answer": heldout_answer,
        }
        for fact_key, question in oo_atomic_questions(spec).items():
            answer_text, decision, probability = _answer(
                runtime, image, question, answers)
            safe = _safe_fact(fact_key)
            row[f"atomic_{safe}_question"] = question
            row[f"atomic_{safe}_answer"] = answer_text
            row[f"atomic_{safe}_initial_decision"] = decision
            row[f"atomic_{safe}_yes_probability"] = probability
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    exp4.save_json(output / "level_oo_shared_pool_manifest.json", rows)
    exp4.save_csv(output / "level_oo_source_audit.csv", rows)
    return rows


def validate_oo_audit(oo_manifest: Sequence[Mapping[str, object]],
                      source_manifest: Sequence[Mapping[str, object]],
                      specs: Sequence[exp4.CompositeSpec],
                      required: int) -> None:
    """Ensure a resumed OO audit still refers to the exact source pool."""
    if len(oo_manifest) != len(source_manifest):
        raise RuntimeError(
            f"Expected {len(source_manifest)} OO audit rows, "
            f"got {len(oo_manifest)}")
    exp4.validate_manifest(oo_manifest, specs, required)
    source_by_key = {
        (str(row.get("spec", "")), exp4.as_int(row.get("image_id"), -1)):
        row for row in source_manifest
    }
    seen = set()
    for row in oo_manifest:
        key = (str(row.get("spec", "")),
               exp4.as_int(row.get("image_id"), -1))
        if key in seen:
            raise RuntimeError(f"Duplicate OO audit row: {key}")
        seen.add(key)
        source = source_by_key.get(key)
        if source is None:
            raise RuntimeError(f"OO audit row is not in the source pool: {key}")
        if row.get("level") != LEVEL:
            raise RuntimeError(f"OO audit row has unexpected level: {key}")
        if (Path(str(row.get("source_path", ""))).resolve()
                != Path(str(source.get("source_path", ""))).resolve()):
            raise RuntimeError(f"OO audit source image mismatch: {key}")
        if exp4.as_int(row.get("pool_index"), -1) != exp4.as_int(
                source.get("pool_index"), -1):
            raise RuntimeError(f"OO audit pool index mismatch: {key}")


def validate_source_pool(base_output: Path, cfg: Mapping[str, object],
                         manifest: Sequence[Mapping[str, object]],
                         specs: Sequence[exp4.CompositeSpec]) -> None:
    signature_path = base_output / "experiment_signature.json"
    if not signature_path.exists():
        raise FileNotFoundError(f"Missing source signature: {signature_path}")
    signature = json.loads(signature_path.read_text())
    if signature.get("protocol_version") != SOURCE_PROTOCOL_VERSION:
        raise RuntimeError(
            "The source output is not the completed GHOST-compatible v4 run: "
            f"{signature.get('protocol_version')!r}")
    expected_levels = tuple(exp4.LEVELS)
    if tuple(cfg.get("levels", ())) != expected_levels:
        raise RuntimeError(
            "The source resolved_config.json does not contain the four v4 "
            f"levels in order: {expected_levels}")
    exp4.validate_manifest(
        manifest, specs, int(cfg["num_images_per_spec"]))
    if not exp4.manifest_has_query_audit(manifest, exp4.LEVELS):
        raise RuntimeError(
            "Source manifest is missing the complete v4 GHOST query audit")
    expected_rows = len(specs) * int(cfg["num_images_per_spec"])
    if len(manifest) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} source-pool rows, got {len(manifest)}")
    for row in manifest:
        if row.get("pool_protocol") != exp4.POOL_PROTOCOL:
            raise RuntimeError("Source manifest has an unexpected pool protocol")
        source_path = Path(str(row.get("source_path", "")))
        if not source_path.is_file():
            raise FileNotFoundError(source_path)


def load_inputs(base_output: Path) -> Tuple[dict, List[dict], List[exp4.CompositeSpec]]:
    cfg_path = base_output / "resolved_config.json"
    manifest_path = base_output / "full00_manifest.json"
    if not cfg_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "The completed v4 output must contain resolved_config.json and "
            "full00_manifest.json")
    cfg = json.loads(cfg_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    specs = [exp4.CompositeSpec.from_dict(value)
             for value in cfg["target_specs"]]
    validate_source_pool(base_output, cfg, manifest, specs)
    return cfg, manifest, specs


def load_oo_config(base_cfg: Mapping[str, object], base_output: Path,
                   output: Path, resume: bool) -> dict:
    cfg = dict(base_cfg)
    cfg["output_dir"] = str(output.resolve())
    cfg["levels"] = [LEVEL]
    cfg["resume"] = bool(resume)
    cfg["source_output_dir"] = str(base_output.resolve())
    cfg["source_protocol_version"] = SOURCE_PROTOCOL_VERSION
    cfg["protocol_version"] = PROTOCOL_VERSION
    cfg["level_oo_prompts"] = list(OO_PROMPTS)
    cfg["level_oo_heldout_template"] = OO_HELDOUT_TEMPLATE
    return cfg


def ensure_output_compatibility(output: Path, signature: str) -> None:
    path = output / "experiment_signature.json"
    if not path.exists():
        return
    existing = json.loads(path.read_text())
    if existing.get("sha256") != signature:
        raise RuntimeError(
            f"Output directory {output} belongs to a different Level-OO run; "
            "use a new output directory")


def experiment_signature(cfg: Mapping[str, object],
                         manifest: Sequence[Mapping[str, object]]) -> str:
    relevant = {
        "protocol_version": PROTOCOL_VERSION,
        "source_protocol_version": SOURCE_PROTOCOL_VERSION,
        "source_output_dir": cfg["source_output_dir"],
        "levels": [LEVEL],
        "target_specs": cfg["target_specs"],
        "manifest": list(manifest),
        "level_oo_prompts": list(OO_PROMPTS),
        "level_oo_heldout_template": OO_HELDOUT_TEMPLATE,
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
        "owl_threshold": cfg["owl_threshold"],
    }
    payload = json.dumps(relevant, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_oo_objects(runtime: exp4.Runtime, image: Image.Image,
                      spec: exp4.CompositeSpec) -> dict:
    result = runtime.absence.verify(
        image, spec.as_target_spec(), "Multi-O")
    presence = result["presence"]
    scores = result["max_scores"]
    a, b = spec.objects
    a_present = bool(presence.get(a, False))
    b_present = bool(presence.get(b, False))
    return {
        "object_a_present": int(a_present),
        "object_a_absent": int(not a_present),
        "object_a_detection_score": float(scores.get(a, 0.0)),
        "object_b_evaluated": 1,
        "object_b_present": int(b_present),
        "object_b_absent": int(not b_present),
        "object_b_detection_score": float(scores.get(b, 0.0)),
        "required_objects_absent": int(not a_present and not b_present),
    }


def evaluate_oo_generated(
        runtime: exp4.Runtime, image: Image.Image, spec: exp4.CompositeSpec,
        trigger_prompt: str, heldout_initial_decision: Optional[bool] = None,
        canonical_initial_decision: Optional[bool] = None) -> dict:
    if not trigger_prompt:
        raise ValueError("trigger_prompt cannot be empty")
    answers: Dict[str, Tuple[str, Optional[bool], float]] = {}
    trigger_answer, trigger_decision, trigger_probability = _answer(
        runtime, image, trigger_prompt, answers)
    heldout = oo_heldout_question(spec)
    canonical = oo_canonical_question(spec)
    heldout_answer, heldout_decision, heldout_probability = _answer(
        runtime, image, heldout, answers)
    canonical_answer, canonical_decision, canonical_probability = _answer(
        runtime, image, canonical, answers)
    heldout_flip = int(
        heldout_initial_decision is False and heldout_decision is True)
    canonical_flip = int(
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
        "ghost_query_success": int(trigger_decision is True),
        "query_flip_success": int(trigger_decision is True),
        "oo_query_initial_decision": canonical_initial_decision,
        "oo_query_flip_success": canonical_flip,
        "canonical_question": canonical,
        "canonical_answer": canonical_answer,
        "canonical_decision": canonical_decision,
        "canonical_yes_probability": canonical_probability,
        "canonical_query_yes": int(canonical_decision is True),
        "canonical_initial_decision": canonical_initial_decision,
        "canonical_flip_success": canonical_flip,
        "heldout_question": heldout,
        "heldout_answer": heldout_answer,
        "heldout_decision": heldout_decision,
        "heldout_yes_probability": heldout_probability,
        "heldout_query_yes": int(heldout_decision is True),
        "heldout_initial_decision": heldout_initial_decision,
        "heldout_flip_success": heldout_flip,
    }
    atomic = oo_atomic_questions(spec)
    active = set(oo_active_fact_keys(spec))
    for fact_key, question in atomic.items():
        answer_text, decision, probability = _answer(
            runtime, image, question, answers)
        safe = _safe_fact(fact_key)
        row[f"atomic_{safe}_question"] = question
        row[f"atomic_{safe}_answer"] = answer_text
        row[f"atomic_{safe}_decision"] = decision
        row[f"atomic_{safe}_yes_probability"] = probability
        row[f"atomic_{safe}_yes"] = int(decision is True)
        row[f"active_atomic_{safe}_yes"] = (
            int(decision is True) if fact_key in active else float("nan"))
    row.update(verify_oo_objects(runtime, image, spec))
    row["active_atomic_all_yes"] = int(all(
        bool(row.get(f"atomic_{_safe_fact(key)}_yes", 0)) for key in active))
    row["valid_attack_success"] = int(
        row["target_query_yes"] and row["required_objects_absent"])
    row["ghost_aligned_success"] = row["valid_attack_success"]
    row["heldout_generalization_success"] = int(
        row["valid_attack_success"] and row["heldout_flip_success"])
    row["valid_active_atomic_all"] = int(
        row["active_atomic_all_yes"] and row["required_objects_absent"])
    return row


def generate_oo_and_evaluate(runtime: exp4.Runtime, source: Image.Image,
                             embedding: torch.Tensor, spec: exp4.CompositeSpec,
                             seed: int, image_id: int, trigger_step: int,
                             trigger_prompt: str, generation_index: int,
                             cfg: Mapping[str, object], output: Path,
                             heldout_initial_decision: Optional[bool],
                             canonical_initial_decision: Optional[bool]) -> dict:
    runtime.load_diffusion()
    sample_seed = exp4.generation_seed(
        seed, spec.name, image_id, generation_index)
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
    evaluation = evaluate_oo_generated(
        runtime, generated, spec, trigger_prompt,
        heldout_initial_decision=heldout_initial_decision,
        canonical_initial_decision=canonical_initial_decision)
    valid = bool(evaluation["valid_attack_success"])
    generated_path = ""
    if valid or bool(cfg.get("save_failed_images", False)):
        directory = output / "generated" / spec.name / LEVEL
        directory.mkdir(parents=True, exist_ok=True)
        suffix = "success" if valid else "failed"
        path = directory / (
            f"seed{seed}_{image_id}_step{trigger_step}_g{generation_index}_"
            f"{suffix}.png")
        generated.save(path)
        generated_path = str(path)
    return {
        "seed": seed, "spec": spec.name, "image_id": image_id,
        "level": LEVEL, "generation_index": generation_index,
        "generation_seed": sample_seed, "trigger_step": trigger_step,
        "trigger_prompt": trigger_prompt, "generated_path": generated_path,
        "diffusion_prompt": trigger_prompt,
        **evaluation,
    }


def optimize_oo_embedding(runtime: exp4.Runtime, source: Image.Image,
                          spec: exp4.CompositeSpec, seed: int, image_id: int,
                          cfg: Mapping[str, object], contrast: torch.Tensor,
                          output: Path, heldout_initial_decision: Optional[bool],
                          canonical_initial_decision: Optional[bool]
                          ) -> Tuple[bool, int, str, List[dict], List[dict]]:
    with torch.no_grad():
        initial = runtime.clip.images([source]).detach()
    embedding = nn.Parameter(initial.clone())
    optimizer = torch.optim.AdamW(
        [embedding], lr=float(cfg["learning_rate"]))
    prompts = oo_optimization_questions(spec)
    # This is intentionally the exact v4 seed formula, which excludes level.
    prompt_rng = random.Random(exp4.stable_seed(
        "stage1-exp4-prompt", seed, spec.name, image_id))
    threshold = float(cfg["yes_threshold"])
    generation_budget = int(cfg["num_generations"])
    traces: List[dict] = []
    generations: List[dict] = []
    embedding_success = False
    trigger_step, trigger_prompt = -1, ""
    for step in range(int(cfg["num_steps"])):
        prompt = prompts[prompt_rng.randrange(len(prompts))]
        tokens = runtime.projector.model(embedding).half().squeeze(0)
        probability_before = runtime.victim.projected_probability(
            source, tokens, prompt)
        attack_loss = -torch.log(probability_before.clamp_min(1e-8))
        clip_loss = exp4.contrast_loss(embedding, contrast)
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
            generation = generate_oo_and_evaluate(
                runtime, source, embedding.detach(), spec, seed, image_id,
                step, prompt, generation_index, cfg, output,
                heldout_initial_decision, canonical_initial_decision)
            generations.append(generation)
            generation_attempted = True
            generation_success = bool(generation["valid_attack_success"])
        traces.append({
            "seed": seed, "spec": spec.name, "image_id": image_id,
            "level": LEVEL, "step": step, "prompt": prompt,
            "loss": float(loss.detach()),
            "attack_loss": float(attack_loss.detach()),
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
    return embedding_success, trigger_step, trigger_prompt, traces, generations


def aggregate_oo_result(spec: exp4.CompositeSpec, seed: int,
                        item: Mapping[str, object], embedding_success: bool,
                        trigger_step: int, trigger_prompt: str,
                        generations: Sequence[Mapping[str, object]]) -> dict:
    successful = next((row for row in generations
                       if exp4.as_int(row.get("valid_attack_success"))), None)
    return {
        "seed": seed, "spec": spec.name, "image_id": exp4.as_int(item["image_id"]),
        "level": LEVEL, "source_path": str(item.get("source_path", "")),
        "pool_index": exp4.as_int(item.get("pool_index")),
        "clip_base_score": exp4.as_float(item.get("base_score")),
        "oo_prescreen_initial_decision": item.get(
            "oo_prescreen_initial_decision"),
        "oo_canonical_initial_decision": item.get(
            "oo_canonical_initial_decision"),
        "oo_heldout_initial_decision": item.get(
            "oo_heldout_initial_decision"),
        "embedding_success": int(embedding_success),
        "trigger_step": trigger_step, "trigger_prompt": trigger_prompt,
        "generation_evaluated": int(bool(generations)),
        "num_generations_evaluated": len(generations),
        "ghost_query_success": exp4._any_binary(
            generations, "target_query_yes"),
        "query_flip_success": exp4._any_binary(
            generations, "target_query_yes"),
        "heldout_query_flip_success": exp4._any_heldout_flip(generations),
        "canonical_query_flip_success": exp4._any_canonical_flip(generations),
        "valid_thr_success": exp4._any_binary(
            generations, "valid_attack_success"),
        "heldout_generalization_success": exp4._any_binary(
            generations, "heldout_generalization_success"),
        "required_objects_absent": exp4._any_binary(
            generations, "required_objects_absent"),
        "active_atomic_all_yes": exp4._any_binary(
            generations, "active_atomic_all_yes"),
        "valid_active_atomic_all": exp4._any_binary(
            generations, "valid_active_atomic_all"),
        "object_a_hit": exp4._any_binary(generations, "object_a_present"),
        "object_b_evaluated": exp4._any_binary(
            generations, "object_b_evaluated"),
        "object_b_hit": exp4._any_optional(
            generations, "object_b_present"),
        "object_a_absence": exp4._any_binary(
            generations, "object_a_absent"),
        "object_b_absence": exp4._any_optional(
            generations, "object_b_absent"),
        "last_trigger_step": exp4.as_int(successful.get("trigger_step"), -1)
        if successful else -1,
        "successful_trigger_prompt": str(
            successful.get("trigger_prompt", "")) if successful else "",
    }


def run_oo_condition(runtime: exp4.Runtime, dset, item: Mapping[str, object],
                     spec: exp4.CompositeSpec, seed: int,
                     cfg: Mapping[str, object], output: Path,
                     contrast: torch.Tensor) -> Tuple[dict, List[dict], List[dict]]:
    image, source_path = dset[exp4.as_int(item["image_id"])]
    expected_path = Path(str(item["source_path"])).resolve()
    if Path(str(source_path)).resolve() != expected_path:
        raise RuntimeError(f"Source image changed for {spec.name}/{item['image_id']}")
    embedding_success, trigger_step, trigger_prompt, traces, generations = \
        optimize_oo_embedding(
            runtime, image, spec, seed, exp4.as_int(item["image_id"]), cfg,
            contrast, output,
            exp4.as_optional_bool(item.get("oo_heldout_initial_decision")),
            exp4.as_optional_bool(item.get("oo_canonical_initial_decision")))
    result = aggregate_oo_result(
        spec, seed, item, embedding_success, trigger_step, trigger_prompt,
        generations)
    return result, traces, generations


def run_oo_seed(runtime: exp4.Runtime, dset, manifest: Sequence[dict],
                specs: Sequence[exp4.CompositeSpec], seed: int,
                cfg: Mapping[str, object], output: Path,
                signature: str) -> Tuple[List[dict], List[dict], List[dict]]:
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("resume", True):
        results, traces, generations = exp4.load_task_checkpoints(
            seed_dir, signature)
    else:
        results, traces, generations = [], [], []
    completed = {exp4.task_key(row) for row in results}
    spec_by_name = {spec.name: spec for spec in specs}
    contrasts = {
        spec.name: oo_contrast_embedding(runtime, spec) for spec in specs}
    jobs = [(spec_by_name[str(item["spec"])], item) for item in manifest]
    with tqdm(total=len(jobs), desc=f"Stage 1 Exp 4 {LEVEL} seed {seed}") as progress:
        for spec, item in jobs:
            key = (spec.name, exp4.as_int(item["image_id"]), LEVEL)
            if key in completed:
                progress.update(1)
                continue
            result, new_traces, new_generations = run_oo_condition(
                runtime, dset, item, spec, seed, cfg, seed_dir,
                contrasts[spec.name])
            results.append(result)
            traces.extend(new_traces)
            generations.extend(new_generations)
            exp4.save_json_atomic(exp4.task_checkpoint_path(
                seed_dir, spec.name, exp4.as_int(item["image_id"]), LEVEL), {
                    "experiment_signature": signature,
                    "result": result, "traces": new_traces,
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


def save_pair_plot(oo_summary: Sequence[Mapping[str, object]],
                   base_summary: Sequence[Mapping[str, object]],
                   output: Path) -> None:
    labels = ["Level-O", LEVEL, "Level-O+R"]
    by_key = {(str(row.get("spec")), str(row.get("level"))): row
              for row in base_summary if row.get("seed") == "AGGREGATE"}
    by_key.update({("ALL", str(row.get("level"))): row
                   for row in oo_summary if row.get("seed") == "AGGREGATE"})
    values = []
    for level in labels:
        row = by_key[("ALL", level)]
        values.append((
            exp4.as_float(row.get("valid_thr_seed_mean", row.get("valid_thr"))),
            exp4.as_float(row.get("valid_thr_cluster_ci_low")),
            exp4.as_float(row.get("valid_thr_cluster_ci_high")),
        ))
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    x = np.arange(len(labels))
    y = np.asarray([item[0] for item in values])
    low = np.asarray([item[1] for item in values])
    high = np.asarray([item[2] for item in values])
    ax.errorbar(x, y, yerr=np.vstack((y - low, high - y)), fmt="o",
                color="#222222", capsize=3)
    ax.set_xticks(x, ["O", "OO", "O+R"])
    ax.set_ylim(-0.005, max(0.02, float(high.max()) * 1.25))
    ax.set_ylabel("Valid-THR original-sample rate")
    ax.set_title("Paired object-count / relation control")
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "level_oo_valid_thr.png", dpi=320,
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output / "level_oo_valid_thr.pdf",
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output", type=Path,
                        default=SOURCE_OUTPUT_DEFAULT,
                        help="Completed v4 output containing the shared manifest")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not load Level-OO checkpoints")
    parser.add_argument("--save-failed-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_output = args.base_output.resolve()
    output = args.output_dir.resolve()
    base_cfg, source_manifest, specs = load_inputs(base_output)
    cfg = load_oo_config(base_cfg, base_output, output, not args.no_resume)
    if args.save_failed_images:
        cfg["save_failed_images"] = True
    output.mkdir(parents=True, exist_ok=True)
    signature = experiment_signature(cfg, source_manifest)
    ensure_output_compatibility(output, signature)
    exp4.save_json(output / "resolved_config.json", cfg)
    exp4.save_json(output / "source_manifest_reference.json", {
        "source_output_dir": str(base_output),
        "source_manifest": str(base_output / "full00_manifest.json"),
        "source_protocol_version": SOURCE_PROTOCOL_VERSION,
        "source_row_count": len(source_manifest),
    })
    exp4.save_json(output / "experiment_signature.json", {
        "protocol_version": PROTOCOL_VERSION, "sha256": signature,
    })
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Level-OO")
    runtime = exp4.Runtime(cfg, attack=True)
    oo_manifest_path = output / "level_oo_shared_pool_manifest.json"
    if oo_manifest_path.exists() and cfg.get("resume", True):
        oo_manifest = json.loads(oo_manifest_path.read_text())
        validate_oo_audit(oo_manifest, source_manifest, specs,
                          int(cfg["num_images_per_spec"]))
    else:
        oo_manifest = build_oo_manifest(
            runtime, exp4.load_coco_dataset(cfg["data_path"], split="train"),
            source_manifest, specs, output)
        validate_oo_audit(oo_manifest, source_manifest, specs,
                          int(cfg["num_images_per_spec"]))
    exp4.save_json(output / "target_specs.json",
                   [spec.to_dict() for spec in specs])
    dset = exp4.load_coco_dataset(cfg["data_path"], split="train")
    all_results: List[dict] = []
    all_traces: List[dict] = []
    all_generations: List[dict] = []
    for seed in cfg["seeds"]:
        exp4.seed_all(seed)
        results, traces, generations = run_oo_seed(
            runtime, dset, oo_manifest, specs, seed, cfg, output, signature)
        all_results.extend(results)
        all_traces.extend(traces)
        all_generations.extend(generations)
    exp4.save_csv(output / "all_results.csv", all_results)
    exp4.save_csv(output / "all_steps.csv", all_traces)
    exp4.save_csv(output / "all_generations.csv", all_generations)
    summary = exp4.summarize_all(
        all_results, all_generations, specs, cfg["seeds"], [LEVEL],
        bootstrap_samples=int(cfg["bootstrap_samples"]),
        bootstrap_seed=int(cfg["seeds"][0]))
    exp4.save_csv(output / "summary.csv", summary)

    base_results = exp4.read_csv(base_output / "all_results.csv")
    base_summary = exp4.read_csv(base_output / "summary.csv")
    combined = base_results + all_results
    comparison_levels = (exp4.LEVELS[0], LEVEL, exp4.LEVELS[2])
    comparisons = exp4.paired_comparisons(
        combined, cfg, specs, cfg["seeds"], comparison_levels)
    transitions = exp4.transition_rows(
        combined, specs, cfg["seeds"], comparison_levels)
    exp4.save_csv(output / "paired_comparisons_vs_v4.csv", comparisons)
    exp4.save_csv(output / "transition_matrix_vs_v4.csv", transitions)
    save_pair_plot(summary, base_summary, output / "figures")


if __name__ == "__main__":
    main()
