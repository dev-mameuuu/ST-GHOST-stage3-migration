"""Stage 3: original GHOST retry protocol on 2,000 distinct clean images.

Five pairs, 400 images/pair, Multi-O only, one run/image. No fixed seeds or
seed sweeps are used. Hyperparameters, Multi-O prompts, probability interface,
contrast loss and model wrappers come from run_stage1_exp1.py. This is its
two-object extension of GHOST, not the original paper's single-object task.

Restore GHOST-master/main.py's interleaved optimization/generation protocol:
update the embedding, test the Qwen pre-update probability, generate ONE
candidate if it is strictly above threshold, then continue the SAME AdamW
optimizer if trigger-Yes AND object absence did not both hold. Stop on valid
success, four total generations, or 100 updates (defaults inherited from exp1).
The Qwen branch in GHOST-master reuses pre-update inputs for its threshold
check; this detail is preserved and both probabilities are recorded.

Original utils.get_vae_features/generate_image are deliberately reused,
including their legacy [0,1] VAE input and clamp-only decoder postprocessing.
No Stage-2 VAE correction or trajectory optimization is silently introduced.
This choice is recorded in resolved_config.json and must be accounted for
when comparing to experiments using the corrected VAE image contract.

Completed tasks and in-task AdamW/RNG state are checkpointed atomically. RNG
state is restored only for interrupted-task continuation; no seed is imposed.
Use --validate-only for a CPU input check, --prepare-only to build the reusable
clean-image manifest, --max-tasks 1 for a GPU trial, and --summarize-only to
rebuild CSV/JSON reports without loading models. A normal run starts attacks.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import random
import signal
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

import run_stage1_exp1 as exp1


ROOT = exp1.ROOT
PROTOCOL = "stage3_original_ghost_comparison_multi_o_no_seed_v1"
DEFAULT_CONFIG = ROOT / "st-ghost/configs/stage3_original_ghost_comparison.json"
VAE_CONTRACT = {
    "implementation": "GHOST-master/utils.py (same as run_stage1_exp1.py)",
    "encoder_input_range": "[0,1] (legacy reference behavior)",
    "decoder_postprocess": "clamp(0,1), without signed-to-RGB rescaling",
    "precision": "float16 reference pipeline",
}
# The only older implementation eligible for continuation with this patch.
# Config and every other source hash must still match exactly (see below).
NUMERICAL_PATCH_PREDECESSOR = "4474b85b72998e3ec505d8fcdb46b4f936e0c66842c77ea8b460bf8ec8e3eb27"
NUMERICAL_FAILURE_POLICY = "stop_task_restore_last_finite_embedding_keep_prior_generations_count_in_denominator"
SOURCE_IMAGE_SIZE = (336, 336)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_source_image(path):
    with Image.open(path) as image:
        return image.convert("RGB").resize(SOURCE_IMAGE_SIZE, Image.Resampling.LANCZOS)


def json_save(path, payload):
    exp1.save_json_atomic(Path(path), payload)


def tensor_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is writing {output}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_config(path):
    raw = read_json(path)
    base = exp1.resolve_path(raw.pop("base_config", "st-ghost/configs/stage1_exp1_main.json"))
    cfg = exp1.load_config(base)
    cfg.update(raw)
    # The inherited three-seed list is intentionally removed, never executed.
    cfg.pop("seeds", None)
    cfg.pop("seed", None)
    cfg.update(protocol_version=PROTOCOL, conditions=["Multi-O"],
               repeats_per_image=1, random_seed=None,
               generation_protocol="one_candidate_per_threshold_step_then_retry",
               threshold_probability="qwen_pre_update", threshold_comparison=">",
               prescreen_protocol="exp1_single_and_multi_both_no",
               globally_unique_images=True, vae_image_contract=VAE_CONTRACT,
               base_config=str(base))
    cfg.setdefault("ranking_cache_dir", "outputs/stage1_exp1_main")
    for field in ("output_dir", "ranking_cache_dir"):
        cfg[field] = str(exp1.resolve_path(cfg[field]))
    if int(cfg["num_images_per_spec"]) <= 0 or int(cfg["candidate_top_k"]) < int(cfg["num_images_per_spec"]):
        raise ValueError("candidate_top_k must cover a positive num_images_per_spec")
    if any(int(cfg[key]) <= 0 for key in ("num_steps", "num_generations", "checkpoint_every")):
        raise ValueError("Step, generation and checkpoint budgets must be positive")
    if not 0 < float(cfg["yes_threshold"]) < 1 or float(cfg["learning_rate"]) <= 0:
        raise ValueError("Require 0 < yes_threshold < 1 and learning_rate > 0")
    if Path(cfg["output_dir"]) == Path(cfg["ranking_cache_dir"]):
        raise ValueError("Use a separate output directory from the ranking cache")
    if len(cfg["target_specs"]) != 5:
        raise ValueError("This comparison expects the five exp1 target pairs")
    return cfg


def experiment_identity(cfg):
    paths = [Path(__file__).resolve(), Path(exp1.__file__),
             ROOT / "st-ghost/run_stage1.py", ROOT / "GHOST-master/main.py",
             ROOT / "GHOST-master/utils.py", ROOT / "GHOST-master/data.py",
             Path(cfg["base_config"])]
    provenance = {str(path): digest(path) for path in paths}
    payload = {"config": cfg, "implementation_sha256": provenance}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return signature, payload


def compatible_identity(output, current_signature, current_identity):
    """Keep archived data IDs only for the explicitly supported safety patch."""
    signature_path = output / "experiment_signature.json"
    if not signature_path.exists():
        return current_signature, current_identity
    archived_signature = read_json(signature_path)["sha256"]
    if archived_signature == current_signature:
        return current_signature, current_identity
    archived = read_json(output / "input_provenance.json")
    archived_digest = hashlib.sha256(json.dumps(archived, sort_keys=True).encode()).hexdigest()
    runner_path = str(Path(__file__).resolve())
    expected = {**current_identity, "implementation_sha256": {
        **current_identity["implementation_sha256"], runner_path: NUMERICAL_PATCH_PREDECESSOR}}
    if archived_digest != archived_signature or archived != expected:
        raise ValueError("Config/implementation changed incompatibly; use a fresh output directory")
    # Do not rewrite historical provenance to pretend old tasks used new code.
    # main() records the current implementation separately under output_lock.
    return archived_signature, archived


class NumericalFailure(FloatingPointError):
    def __init__(self, message, phase, details=None):
        super().__init__(message)
        self.phase = phase
        self.details = details or {}


def tensor_diagnostics(tensor):
    value = tensor.detach()
    finite = torch.isfinite(value)
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "nonfinite_count": int((~finite).sum().cpu()),
            "finite_absmax": float(value[finite].abs().max().cpu()) if finite.any() else None}


def checked_probability(runtime, source, embedding, prompt, phase):
    if not torch.isfinite(embedding).all():
        raise NumericalFailure("Nonfinite embedding", phase, tensor_diagnostics(embedding))
    projected = runtime.projector.model(embedding)
    if not torch.isfinite(projected).all():
        raise NumericalFailure("Nonfinite projector output", phase, tensor_diagnostics(projected))
    tokens = projected.half().squeeze(0)
    if not torch.isfinite(tokens).all():
        raise NumericalFailure("Projector output overflows FP16", phase, {
            "projector_output": tensor_diagnostics(projected),
            "victim_tokens": tensor_diagnostics(tokens), "fp16_max": torch.finfo(torch.float16).max})
    probability = runtime.victim.projected_probability(source, tokens, prompt)
    if not torch.isfinite(probability).all():
        raise NumericalFailure("Nonfinite victim probability", phase, tensor_diagnostics(tokens))
    return probability


def optimization_update(runtime, source, embedding, initial, optimizer, prompt, cfg, contrasts):
    """An unchanged AdamW update, with explicit checks at numerical boundaries."""
    prob_before = checked_probability(runtime, source, embedding, prompt, "before_update")
    attack = -torch.log(prob_before + 1e-8)
    contrast = exp1.contrast_loss(embedding, contrasts, "Multi-O")
    reg = F.mse_loss(embedding.float(), initial.float())
    loss = attack + float(cfg["lambda_clip"]) * contrast + float(cfg["lambda_reg"]) * reg
    if not torch.isfinite(loss):
        raise NumericalFailure("Nonfinite embedding objective", "objective")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(embedding.grad.float().norm().detach().cpu())
    if not math.isfinite(grad_norm):
        raise NumericalFailure("Nonfinite embedding gradient", "backward", tensor_diagnostics(embedding.grad))
    optimizer.step()
    with torch.no_grad():
        after = checked_probability(runtime, source, embedding, prompt, "after_update")
    # Only scalars escape; release the victim graph before image generation.
    return {"loss": float(loss.detach().cpu()), "attack_loss": float(attack.detach().cpu()),
            "clip_loss": float(contrast.detach().cpu()), "reg_loss": float(reg.detach().cpu()),
            "yes_before": float(prob_before.detach().cpu()), "yes_after": float(after.cpu()),
            "embedding_gradient_norm": grad_norm}


def read_rankings(cfg, dset, output, runtime=None):
    """Reuse matching exp1 CLIP Top-K files; validate annotation/file membership."""
    cache = Path(cfg["ranking_cache_dir"])
    reference = read_json(cache / "resolved_config.json")
    for field in ("clip_checkpoint_path", "data_path"):
        if exp1.resolve_path(reference[field]) != exp1.resolve_path(cfg[field]):
            raise ValueError(f"Incompatible ranking cache: {field}")
    reference_pairs = {s["name"]: s["objects"] for s in reference["target_specs"]}
    pools = {}
    for value in cfg["target_specs"]:
        spec = exp1.PairSpec.from_dict(value)
        if reference_pairs.get(spec.name) != list(spec.objects):
            raise ValueError(f"Ranking target mismatch: {spec.name}")
        path = output / f"clip_ranked_{spec.name}.csv"
        if not path.exists():
            path = cache / path.name
        rows = exp1.read_csv(path)
        if len(rows) < int(cfg["candidate_top_k"]):
            if runtime is None:
                raise ValueError(f"Need a longer CLIP ranking for {spec.name}; run --rank-only")
            rows = exp1.rank_candidates(runtime, dset, spec, int(cfg["candidate_top_k"]),
                                        int(cfg["clip_selection_batch_size"]), output)
            path = output / f"clip_ranked_{spec.name}.csv"
        parsed, seen = [], set()
        for rank, row in enumerate(rows[:int(cfg["candidate_top_k"])], 1):
            image_id = int(row["image_id"])
            if row["spec"] != spec.name or image_id in seen:
                raise ValueError(f"Duplicate or mismatched CLIP candidate: {path}")
            seen.add(image_id)
            if set(dset.get_targets(image_id)) & set(spec.objects):
                raise ValueError(f"Candidate contains an annotated target: {image_id}")
            source = Path(dset.image_dir) / dset.im_dict[image_id]["file_name"]
            if not source.is_file():
                raise FileNotFoundError(source)
            parsed.append({**row, "image_id": image_id, "rank": rank,
                           "source_path": str(source.resolve()),
                           **{k: float(row[k]) for k in (
                               "clip_sim_a", "clip_sim_b", "base_score", "tie_score")}})
        if parsed != sorted(parsed, key=lambda r: (r["base_score"], r["tie_score"]), reverse=True):
            raise ValueError(f"CLIP candidate order is invalid: {path}")
        pools[spec.name] = {"rows": parsed, "path": str(path), "sha256": digest(path)}
    return pools


def augment_assignment(name, accepted, owners, assignments):
    """Add one slot while preserving other groups' counts, reassigning overlaps."""
    visited = set()

    def visit(group):
        if group in visited:
            return False
        visited.add(group)
        for row in accepted[group]:
            image_id = row["image_id"]
            if image_id not in owners:
                owners[image_id] = group
                assignments[group].add(image_id)
                return True
        for row in accepted[group]:
            image_id = row["image_id"]
            previous = owners[image_id]
            if previous != group and visit(previous):
                assignments[previous].remove(image_id)
                assignments[group].add(image_id)
                owners[image_id] = group
                return True
        return False

    return visit(name)


def check_distinct_capacity(pools, quota):
    accepted = {name: value["rows"] for name, value in pools.items()}
    assignments = {name: set() for name in pools}
    owners = {}
    for _ in range(quota):
        for name in pools:
            if not augment_assignment(name, accepted, owners, assignments):
                raise ValueError(f"Ranked pools cannot supply {quota} distinct images per pair")
    return {"distinct_ranked_candidates": len({r["image_id"] for p in pools.values() for r in p["rows"]}),
            "distinct_allocation_counts_before_prescreen": {name: len(ids) for name, ids in assignments.items()}}


def validate_manifest(manifest, cfg, dset=None):
    names = {s["name"] for s in cfg["target_specs"]}
    counts = Counter(row["spec"] for row in manifest)
    if dict(counts) != {name: int(cfg["num_images_per_spec"]) for name in names}:
        raise ValueError("Manifest must have the requested count for every target pair")
    if len({row["image_id"] for row in manifest}) != len(manifest):
        raise ValueError("The comparison requires globally distinct source images")
    specs = {s["name"]: exp1.PairSpec.from_dict(s) for s in cfg["target_specs"]}
    for row in manifest:
        spec = specs[row["spec"]]
        if (row["single_prescreen_decision"] is not False
                or row["multi_prescreen_decision"] is not False
                or row["single_prescreen_question"] != exp1.single_question(spec)
                or row["multi_prescreen_question"] != exp1.multi_question(spec)
                or digest(row["source_path"]) != row["source_sha256"]):
            raise ValueError(f"Invalid clean-image manifest entry: {row['image_id']}")
        if dset is not None and set(dset.get_targets(row["image_id"])) & set(spec.objects):
            raise ValueError("Manifest source contains annotated targets")


def build_manifest(victim, dset, cfg, output, signature, pools):
    names = [s["name"] for s in cfg["target_specs"]]
    specs = {s["name"]: exp1.PairSpec.from_dict(s) for s in cfg["target_specs"]}
    accepted = {name: [] for name in names}
    audit = {name: [] for name in names}
    assignments = {name: set() for name in names}
    owners, cursors = {}, {name: 0 for name in names}
    quota = int(cfg["num_images_per_spec"])
    # Round-robin allocation plus augmenting paths avoids target-order bias
    # from simply removing earlier groups' Top-400 images from later groups.
    with tqdm(total=quota * len(names), desc="Distinct clean sources") as progress:
        for _ in range(quota):
            for name in names:
                while not augment_assignment(name, accepted, owners, assignments):
                    if cursors[name] == len(pools[name]["rows"]):
                        raise RuntimeError(f"Insufficient distinct initial-No sources for {name}; enlarge candidate_top_k")
                    candidate = pools[name]["rows"][cursors[name]]
                    cursors[name] += 1
                    audit_path = output / "prescreen_checkpoints" / name / f"{candidate['image_id']}.json"
                    if audit_path.exists():
                        payload = read_json(audit_path)
                        row = payload["result"]
                        if (payload["signature"] != signature
                                or any(row.get(k) != v for k, v in candidate.items())
                                or digest(row["source_path"]) != row["source_sha256"]):
                            raise ValueError(f"Incompatible prescreen checkpoint: {audit_path}")
                    else:
                        image, _ = dset[candidate["image_id"]]
                        row = {**candidate, "source_sha256": digest(candidate["source_path"])}
                        try:
                            for prefix, question in (
                                ("single", exp1.single_question(specs[name])),
                                ("multi", exp1.multi_question(specs[name]))):
                                answer, decision, probability = victim.answer(image, question)
                                row.update({f"{prefix}_prescreen_question": question,
                                            f"{prefix}_prescreen_answer": answer,
                                            f"{prefix}_prescreen_decision": decision,
                                            f"{prefix}_prescreen_yes_probability": probability})
                        finally:
                            image.close()
                        row["paired_prescreen_no"] = int(row["single_prescreen_decision"] is False
                                                          and row["multi_prescreen_decision"] is False)
                        json_save(audit_path, {"signature": signature, "result": row})
                    if (row["paired_prescreen_no"] != int(row["single_prescreen_decision"] is False
                                                           and row["multi_prescreen_decision"] is False)
                            or row["single_prescreen_question"] != exp1.single_question(specs[name])
                            or row["multi_prescreen_question"] != exp1.multi_question(specs[name])):
                        raise ValueError(f"Invalid prescreen decisions/questions: {audit_path}")
                    audit[name].append(row)
                    if row["paired_prescreen_no"]:
                        accepted[name].append(row)
                progress.update(1)
    manifest = []
    for name in names:
        for row in accepted[name]:
            if row["image_id"] in assignments[name]:
                manifest.append({**row, "pool_index": sum(r["spec"] == name for r in manifest)})
        exp1.save_csv_atomic(output / f"prescreen_{name}.csv", audit[name])
    validate_manifest(manifest, cfg, dset)
    payload = {"signature": signature,
               "ranking_provenance": {name: {k: v for k, v in p.items() if k != "rows"}
                                      for name, p in pools.items()}, "images": manifest}
    json_save(output / "clean_manifest.json", payload)
    # Flat file for pairing with future methods; it has no seed/task repeats.
    json_save(output / "selected_images.json", manifest)
    exp1.save_csv_atomic(output / "selected_images.csv", manifest)
    return manifest


def embedding_artifact(output, item, role, embedding, signature, **metadata):
    path = output / "embeddings" / item["spec"] / f"{item['image_id']}_{role}.pt"
    tensor_save(path, {"signature": signature, "spec": item["spec"],
                       "image_id": item["image_id"], "condition": "Multi-O",
                       "role": role, "embedding": embedding.detach().cpu(), **metadata})
    return str(path)


def evaluate_candidate(runtime, image, spec, prompt):
    # Exp1's fixed query is held out; the original GHOST stopping rule uses
    # the actual trigger query. Keep these outcomes separate in every record.
    diagnostic = exp1.evaluate_generated(runtime, image, spec, "Multi-O")
    for key in list(diagnostic):
        if key.startswith("target_"):
            diagnostic[f"heldout_{key}"] = diagnostic.pop(key)
    diagnostic["heldout_valid_attack_success"] = diagnostic.pop("valid_attack_success")
    answer, decision, probability = runtime.victim.answer(image, prompt)
    return {**diagnostic, "target_question": prompt, "target_answer": answer,
            "target_decision": decision, "target_yes_probability": probability,
            "target_query_yes": int(decision is True),
            "target_answer_unparsed": int(decision is None),
            "valid_attack_success": int(decision is True and diagnostic["absence_all"])}


def draw_candidate(runtime, source, embedding, vae, spec, prompt, item,
                   step, index, cfg, output, signature):
    started = time.monotonic()
    with torch.no_grad():
        generated = exp1.generate_image(
            runtime.pipe, vae, embedding.detach().half(), t=int(cfg["diffusion_t"]),
            num_inference_steps=int(cfg["num_inference_steps"]),
            noise_level=int(cfg["noise_level"]), prompt=prompt,
            negative_prompt="low quality, ugly, unrealistic",
            guidance_scale=float(cfg["guidance_scale"]))
    try:
        evaluation = evaluate_candidate(runtime, generated, spec, prompt)
        stem = f"{item['image_id']}_step{step:03d}_g{index}"
        path = output / "generated" / spec.name / f"{stem}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.png")
        generated.save(temporary)
        os.replace(temporary, path)
        embedding_path = embedding_artifact(output, item, f"step{step:03d}_g{index}",
                                             embedding, signature, step=step,
                                             optimization_updates=step + 1, trigger_prompt=prompt)
        return {"spec": spec.name, "image_id": item["image_id"], "condition": "Multi-O",
                "generation_index": index, "trigger_step": step,
                "optimization_updates": step + 1, "trigger_prompt": prompt,
                "source_path": item["source_path"], "generation_seed": None,
                "generated_path": str(path), "image_sha256": digest(path),
                "condition_embedding_path": embedding_path,
                "condition_embedding_sha256": digest(embedding_path),
                "generation_and_evaluation_seconds": time.monotonic() - started, **evaluation}
    finally:
        generated.close()


def result_row(item, trace, generations, stop_reason, artifacts, elapsed):
    last = trace[-1] if trace else {}
    threshold_rows = [r for r in trace if r["threshold_reached"]]
    probs = [r["target_yes_probability"] for r in generations]
    finite_probs = [p for p in probs if math.isfinite(p)]
    return {"spec": item["spec"], "image_id": item["image_id"], "condition": "Multi-O",
            "source_path": item["source_path"], "source_sha256": item["source_sha256"],
            "embedding_success": int(bool(threshold_rows)),
            "first_threshold_step": threshold_rows[0]["step"] if threshold_rows else -1,
            "first_threshold_prompt": threshold_rows[0]["prompt"] if threshold_rows else "",
            "optimization_updates": len(trace), "stop_reason": stop_reason,
            "initial_embedding_yes_probability": trace[0]["yes_before"] if trace else None,
            "final_embedding_yes_probability": last.get("yes_after"),
            "max_embedding_yes_probability": max((r["yes_after"] for r in trace), default=None),
            "post_update_threshold_ever_reached": int(any(r["post_update_threshold_reached"] for r in trace)),
            "initial_attack_loss": trace[0]["attack_loss"] if trace else None,
            "final_attack_loss": -math.log(max(last["yes_after"], 1e-8)) if trace else None,
            "generation_evaluated": int(bool(generations)),
            "num_generations_evaluated": len(generations),
            "query_flip_success": int(any(r["target_query_yes"] for r in generations)),
            "valid_thr_success": int(any(r["valid_attack_success"] for r in generations)),
            "heldout_query_flip_success": int(any(r["heldout_target_query_yes"] for r in generations)),
            "heldout_valid_thr_success": int(any(r["heldout_valid_attack_success"] for r in generations)),
            "max_generated_yes_probability": max(finite_probs) if finite_probs else None,
            "mean_generated_yes_probability": sum(finite_probs) / len(finite_probs) if finite_probs else None,
            "last_generated_yes_probability": probs[-1] if probs else None,
            "first_valid_generation_index": next((r["generation_index"] for r in generations
                                                   if r["valid_attack_success"]), -1),
            "task_seconds": elapsed, **artifacts}


def save_task_state(path, signature, item, embedding, initial, optimizer,
                    trace, generations, artifacts, elapsed, best_probability):
    tensor_save(path, {"signature": signature, "item": item,
                       "embedding": cpu_tree(embedding), "initial": cpu_tree(initial),
                       "optimizer": cpu_tree(optimizer.state_dict()), "trace": trace,
                       "generations": generations, "artifacts": artifacts,
                       "elapsed_seconds": elapsed, "best_probability": best_probability,
                       "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state_all()})


def run_task(runtime, item, cfg, output, signature, contrasts):
    name = f"{item['spec']}__{item['image_id']}"
    state_path = output / "in_progress" / f"{name}.pt"
    checkpoint_path = output / "task_checkpoints" / f"{name}.json"
    spec = exp1.PairSpec.from_dict(next(s for s in cfg["target_specs"] if s["name"] == item["spec"]))
    if digest(item["source_path"]) != item["source_sha256"]:
        raise ValueError("Source image changed since clean-image selection")
    source = load_source_image(item["source_path"])
    started, previous_seconds, vae = time.monotonic(), 0.0, None
    try:
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            if state["signature"] != signature or state["item"] != item:
                raise ValueError(f"Incompatible in-task checkpoint: {state_path}")
            initial = state["initial"].to(runtime.device)
            embedding = torch.nn.Parameter(state["embedding"].to(runtime.device))
            trace, generations, artifacts = state["trace"], state["generations"], state["artifacts"]
            previous_seconds, best_probability = state["elapsed_seconds"], state["best_probability"]
        else:
            state = None
            with torch.no_grad():
                initial = runtime.clip.images([source]).detach()
            embedding = torch.nn.Parameter(initial.clone())
            trace, generations, best_probability = [], [], -1.0
            artifacts = {"initial_embedding_path": embedding_artifact(
                output, item, "initial", initial, signature)}
        optimizer = torch.optim.AdamW([embedding], lr=float(cfg["learning_rate"]))
        if state is not None:
            optimizer.load_state_dict(state["optimizer"])
        # One posterior sample per source, as in the original reference. It is
        # stored with the task so retries and interrupted resumes share it.
        vae_path = output / "in_progress" / f"{name}_vae.pt"
        if generations or len(trace) < int(cfg["num_steps"]):
            runtime.load_diffusion()
            if hasattr(runtime.pipe, "set_progress_bar_config"):
                runtime.pipe.set_progress_bar_config(disable=True)
            if vae_path.exists():
                vae_state = torch.load(vae_path, map_location="cpu", weights_only=True)
                if vae_state["signature"] != signature or vae_state["source_sha256"] != item["source_sha256"]:
                    raise ValueError("Incompatible cached VAE posterior")
                vae = vae_state["vae"].to(runtime.device)
            else:
                with torch.no_grad():
                    vae = exp1.get_vae_features([source], runtime.pipe, runtime.diffusion_preprocess).detach()
                if not torch.isfinite(vae).all():
                    raise FloatingPointError("Nonfinite reference VAE encoding")
                tensor_save(vae_path, {"signature": signature, "source_sha256": item["source_sha256"],
                                       "vae": vae.cpu()})
        if state is not None:
            # Restore after lazy model/VAE initialization, which can consume RNG.
            random.setstate(state["python_rng"])
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        prompts = exp1.optimization_questions(spec, "Multi-O")
        stop_reason, numerical_failure = None, None
        for step in range(len(trace), int(cfg["num_steps"])):
            if any(g["valid_attack_success"] for g in generations):
                stop_reason = "valid_attack_success"
                break
            if len(generations) >= int(cfg["num_generations"]):
                stop_reason = "generation_budget_exhausted"
                break
            prompt = random.choice(prompts)
            last_finite_embedding = embedding.detach().clone()
            try:
                update = optimization_update(runtime, source, embedding, initial, optimizer, prompt, cfg, contrasts)
            except NumericalFailure as error:
                numerical_failure = {"spec": spec.name, "image_id": item["image_id"],
                                     "step": step, "attempted_update": step + 1, "prompt": prompt,
                                     "phase": error.phase, "error": str(error), "details": error.details,
                                     "updates_retained": len(trace), "generations_retained": len(generations),
                                     "policy": NUMERICAL_FAILURE_POLICY, "recorded_at": utc_now()}
                artifacts["numerical_rejected_embedding_path"] = embedding_artifact(
                    output, item, "numerical_rejected", embedding, signature, **numerical_failure)
                with torch.no_grad():
                    embedding.copy_(last_finite_embedding)
                optimizer.zero_grad(set_to_none=True)
                failure_dir = output / "numerical_failures"
                failure_path = failure_dir / f"{name}.json"
                artifacts["numerical_failure_path"] = str(failure_path)
                # Preserve the last saved optimizer/RNG and posterior for audit
                # or explicit future retry, rather than deleting this evidence.
                for previous, key in ((state_path, "numerical_checkpoint_path"),
                                      (vae_path, "numerical_vae_path")):
                    if previous.exists():
                        archived_path = failure_dir / previous.name
                        archived_path.parent.mkdir(parents=True, exist_ok=True)
                        # Copy atomically; normal task cleanup removes originals
                        # only after the completed checkpoint is written.
                        temporary = archived_path.with_name(archived_path.name + ".tmp")
                        temporary.write_bytes(previous.read_bytes())
                        os.replace(temporary, archived_path)
                        artifacts[key] = str(archived_path)
                json_save(failure_path, {"signature": signature, **numerical_failure, **artifacts})
                print(json.dumps({"event": "task_numerical_failure", **numerical_failure}), flush=True)
                stop_reason = "numerical_failure"
                break
            before_value, after_value = update["yes_before"], update["yes_after"]
            reached = before_value > float(cfg["yes_threshold"])
            row = {"spec": spec.name, "image_id": item["image_id"], "condition": "Multi-O",
                   "step": step, "optimization_updates": step + 1, "prompt": prompt,
                   **update,
                   "threshold_check_yes_probability": before_value,
                   "threshold_reached": int(reached),
                   "post_update_threshold_reached": int(after_value > float(cfg["yes_threshold"])),
                   "embedding_displacement_l2": float((embedding.detach() - initial).float().norm().cpu()),
                   "generation_triggered": 0, "generation_index": -1}
            if after_value > best_probability:
                best_probability = after_value
                artifacts["best_embedding_path"] = embedding_artifact(
                    output, item, "best", embedding, signature, step=step,
                    yes_probability=after_value, prompt=prompt)
            if reached:
                if "first_threshold_embedding_path" not in artifacts:
                    artifacts["first_threshold_embedding_path"] = embedding_artifact(
                        output, item, "first_threshold", embedding, signature, step=step,
                        threshold_check_yes_probability=before_value, post_update_yes_probability=after_value)
                generation = draw_candidate(runtime, source, embedding, vae, spec, prompt,
                                             item, step, len(generations), cfg, output, signature)
                generation.update(embedding_yes_before=before_value, embedding_yes_after=after_value)
                generations.append(generation)
                row.update(generation_triggered=1, generation_index=generation["generation_index"])
            trace.append(row)
            elapsed = previous_seconds + time.monotonic() - started
            save_task_state(state_path, signature, item, embedding, initial, optimizer,
                            trace, generations, artifacts, elapsed, best_probability)
            print(json.dumps({"event": "optimization_step", **row,
                              "generations_so_far": len(generations)}), flush=True)
            if reached:
                print(json.dumps({"event": "candidate_evaluated", **generations[-1]}), flush=True)
        if numerical_failure is not None:
            stop_reason = "numerical_failure"
        elif any(g["valid_attack_success"] for g in generations):
            stop_reason = "valid_attack_success"
        elif len(generations) >= int(cfg["num_generations"]):
            stop_reason = "generation_budget_exhausted"
        else:
            stop_reason = "optimization_budget_exhausted"
        artifacts["final_embedding_path"] = embedding_artifact(
            output, item, "final", embedding, signature, optimization_updates=len(trace),
            stop_reason=stop_reason)
        result = result_row(item, trace, generations, stop_reason, artifacts,
                            previous_seconds + time.monotonic() - started)
        payload = {"signature": signature, "status": "complete", "result": result,
                   "traces": trace, "generations": generations, "numerical_failure": numerical_failure}
        json_save(checkpoint_path, payload)
        state_path.unlink(missing_ok=True)
        vae_path.unlink(missing_ok=True)
        print(json.dumps({"event": "task_completed", **result}), flush=True)
        return payload
    finally:
        source.close()


def csv_save(path, rows):
    # Materialize an empty file too, avoiding stale data after a fresh report.
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        exp1.save_csv_atomic(path, rows)
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text("")
        os.replace(temporary, path)


def mean_finite(rows, key):
    values = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    return sum(values) / len(values) if values else None


def summarize_group(results, generations, spec, expected):
    def rate(field):
        return sum(r[field] for r in results) / len(results) if results else None
    return {"spec": spec, "condition": "Multi-O", "tasks_expected": expected,
            "tasks_completed": len(results), "completion_rate": len(results) / expected,
            "numerical_failure_tasks": sum(r["stop_reason"] == "numerical_failure" for r in results),
            "embedding_success_n": sum(r["embedding_success"] for r in results),
            "embedding_success_rate": rate("embedding_success"),
            "query_yes_tasks": sum(r["query_flip_success"] for r in results),
            "task_query_yes_rate": rate("query_flip_success"),
            "valid_success_tasks": sum(r["valid_thr_success"] for r in results),
            "task_valid_success_rate": rate("valid_thr_success"),
            "heldout_task_query_yes_rate": rate("heldout_query_flip_success"),
            "heldout_task_valid_success_rate": rate("heldout_valid_thr_success"),
            "images_generated": len(generations),
            "generated_query_yes_rate": mean_finite(generations, "target_query_yes"),
            "generated_valid_success_rate": mean_finite(generations, "valid_attack_success"),
            "generated_mean_yes_probability": mean_finite(generations, "target_yes_probability"),
            "generated_mean_heldout_yes_probability": mean_finite(generations, "heldout_target_yes_probability"),
            "generated_both_objects_absent_rate": mean_finite(generations, "absence_all"),
            "generated_unparsed_answers": sum(r["target_answer_unparsed"] for r in generations),
            "mean_optimization_updates": mean_finite(results, "optimization_updates"),
            "mean_initial_embedding_yes_probability": mean_finite(results, "initial_embedding_yes_probability"),
            "mean_final_embedding_yes_probability": mean_finite(results, "final_embedding_yes_probability"),
            "mean_task_seconds": mean_finite(results, "task_seconds")}


def write_reports(output, cfg, signature, manifest=None):
    results, traces, generations = [], [], []
    expected_keys = {(r["spec"], r["image_id"]) for r in manifest} if manifest is not None else None
    seen = set()
    for path in sorted((output / "task_checkpoints").glob("*.json")):
        record = read_json(path)
        row = record["result"]
        key = (row["spec"], row["image_id"])
        if (record["signature"] != signature or record["status"] != "complete" or key in seen
                or (expected_keys is not None and key not in expected_keys)):
            raise ValueError(f"Incompatible completed-task checkpoint: {path}")
        seen.add(key)
        results.append(row)
        traces.extend(record["traces"])
        generations.extend(record["generations"])
    summary = []
    for name in [s["name"] for s in cfg["target_specs"]] + ["ALL"]:
        subset = [r for r in results if name == "ALL" or r["spec"] == name]
        images = [r for r in generations if name == "ALL" or r["spec"] == name]
        expected = int(cfg["num_images_per_spec"]) * (len(cfg["target_specs"]) if name == "ALL" else 1)
        summary.append(summarize_group(subset, images, name, expected))
    for name, rows in (("all_results.csv", results), ("all_steps.csv", traces),
                       ("all_generations.csv", generations), ("summary.csv", summary)):
        csv_save(output / name, rows)
    summary_payload = {"protocol_version": PROTOCOL, "updated_at": utc_now(),
                       "rate_denominator": "completed tasks (partial runs are not the full experiment)",
                       "numerical_failure_policy": NUMERICAL_FAILURE_POLICY,
                       "evaluation_complete": len(results) == summary[-1]["tasks_expected"],
                       "overall": summary[-1], "per_spec": summary[:-1],
                       "stop_reasons": dict(Counter(r["stop_reason"] for r in results))}
    json_save(output / "summary.json", summary_payload)
    # A view of the completed data at each attempt index (later attempts are
    # conditional on earlier failures, not independent fixed-budget samples).
    by_attempt = []
    for index in range(int(cfg["num_generations"])):
        rows = [r for r in generations if r["generation_index"] == index]
        by_attempt.append({"generation_index": index, "images": len(rows),
                           "mean_yes_probability": mean_finite(rows, "target_yes_probability"),
                           "query_yes_rate": mean_finite(rows, "target_query_yes"),
                           "valid_success_rate": mean_finite(rows, "valid_attack_success")})
    csv_save(output / "summary_by_generation_attempt.csv", by_attempt)
    return summary_payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="CPU config/ranking/manifest validation only")
    mode.add_argument("--rank-only", action="store_true", help="Prepare/check CLIP rankings; no attacks")
    mode.add_argument("--prepare-only", action="store_true", help="Build clean-image manifest; no embedding optimization")
    mode.add_argument("--summarize-only", action="store_true", help="Rebuild reports without loading models")
    parser.add_argument("--max-tasks", type=int, help="Run at most this many additional tasks, then stop cleanly")
    args = parser.parse_args()
    if args.max_tasks is not None and args.max_tasks <= 0:
        parser.error("--max-tasks must be positive")
    return args


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output = Path(cfg["output_dir"])
    current_signature, current_identity = experiment_identity(cfg)
    signature, identity = compatible_identity(output, current_signature, current_identity)
    signature_path = output / "experiment_signature.json"
    if signature_path.exists():
        if read_json(signature_path)["sha256"] != signature:
            raise ValueError("Config/implementation changed; use a fresh output directory")
    elif output.exists() and any(p.name != "run.lock" for p in output.iterdir()):
        raise ValueError("Nonempty output has no experiment signature; refusing to mix results")
    if args.summarize_only and not signature_path.exists():
        raise FileNotFoundError("No initialized experiment to summarize; run preparation or attacks first")
    manifest_path = output / "clean_manifest.json"
    manifest = None
    if manifest_path.exists():
        payload = read_json(manifest_path)
        if payload["signature"] != signature:
            raise ValueError("Clean manifest belongs to another experiment")
        manifest = payload["images"]
        validate_manifest(manifest, cfg)
    if args.summarize_only:
        with output_lock(output):
            print(json.dumps(write_reports(output, cfg, signature, manifest), indent=2), flush=True)
        return
    dset = None
    if args.validate_only or args.rank_only or manifest is None:
        dset = exp1.load_coco_dataset(cfg["data_path"], split="train")
    if args.validate_only:
        pools = read_rankings(cfg, dset, output)
        capacity = check_distinct_capacity(pools, int(cfg["num_images_per_spec"]))
        if manifest is not None:
            validate_manifest(manifest, cfg, dset)
        print(json.dumps({"tasks_planned": int(cfg["num_images_per_spec"]) * len(cfg["target_specs"]),
                          "conditions": cfg["conditions"], "repeats_per_image": 1,
                          "random_seed": None, "hyperparameters": {k: cfg[k] for k in (
                              "num_steps", "learning_rate", "yes_threshold", "lambda_clip", "lambda_reg",
                              "num_generations", "guidance_scale", "diffusion_t", "num_inference_steps", "owl_threshold")},
                          "ranking_candidates": {name: len(p["rows"]) for name, p in pools.items()},
                          "manifest_ready": manifest is not None, "signature": signature,
                          **capacity}, indent=2), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for screening/attacks")
    with output_lock(output):
        json_save(signature_path, {"protocol_version": PROTOCOL, "sha256": signature})
        json_save(output / "resolved_config.json", cfg)
        json_save(output / "input_provenance.json", identity)
        json_save(output / "implementation_revisions" / f"{digest(Path(__file__))}.json", {
            "recorded_at": utc_now(), "data_signature": signature,
            "implementation_signature": current_signature, "current_identity": current_identity,
            "resumes_predecessor": signature != current_signature,
            "numerical_failure_policy": NUMERICAL_FAILURE_POLICY})
        runtime = None
        status = {"pid": os.getpid(), "started_at": utc_now(), "status": "running"}
        json_save(output / "run_status.json", status)

        def terminate(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}; completed steps/tasks are checkpointed")

        signal.signal(signal.SIGTERM, terminate)
        try:
            # Validation/cached ranking needs no CLIP runtime. Only load it if
            # a longer ranking was explicitly requested via --rank-only.
            if args.rank_only:
                runtime = exp1.Runtime(cfg, attack=False)
                read_rankings(cfg, dset, output, runtime)
            else:
                if manifest is None:
                    pools = read_rankings(cfg, dset, output)
                    check_distinct_capacity(pools, int(cfg["num_images_per_spec"]))
                    from run_stage1 import VictimRuntime
                    victim = VictimRuntime(cfg, torch.device("cuda"))
                    try:
                        manifest = build_manifest(victim, dset, cfg, output, signature, pools)
                    finally:
                        del victim
                        gc.collect()
                        torch.cuda.empty_cache()
                if dset is not None:
                    del dset
                    dset = None
                    gc.collect()
                if not args.prepare_only:
                    summary = write_reports(output, cfg, signature, manifest)
                    completed = {(r["spec"], r["image_id"]) for r in exp1.read_csv(output / "all_results.csv")}
                    # CSV image ids are strings. Canonicalize before comparison.
                    completed = {(name, int(image_id)) for name, image_id in completed}
                    pending = [item for item in manifest if (item["spec"], item["image_id"]) not in completed]
                    if args.max_tasks is not None:
                        pending = pending[:args.max_tasks]
                    if pending:
                        runtime = exp1.Runtime(cfg, attack=True)
                        contrast_cache = {s["name"]: exp1.contrast_embeddings(runtime, exp1.PairSpec.from_dict(s))
                                          for s in cfg["target_specs"]}
                        for index, item in enumerate(tqdm(pending, desc="Stage 3 original GHOST"), 1):
                            run_task(runtime, item, cfg, output, signature, contrast_cache[item["spec"]])
                            if index % int(cfg["checkpoint_every"]) == 0 or index == len(pending):
                                summary = write_reports(output, cfg, signature, manifest)
                                print(json.dumps({"event": "progress", **summary["overall"]}), flush=True)
                            gc.collect()
                            torch.cuda.empty_cache()
            summary = write_reports(output, cfg, signature, manifest)
            json_save(output / "run_status.json", {**status, "finished_at": utc_now(),
                      "status": "completed" if summary["evaluation_complete"] else (
                          "prepared" if args.prepare_only or args.rank_only else "partial"),
                      "tasks_completed": summary["overall"]["tasks_completed"]})
            print(json.dumps(summary, indent=2), flush=True)
        except BaseException as error:
            write_reports(output, cfg, signature, manifest)
            json_save(output / "run_status.json", {**status, "finished_at": utc_now(),
                      "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      "error": repr(error)})
            raise
        finally:
            del runtime
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
