"""Four-candidate evaluation of the saved 500-task, step-45 condition schedules.

Keep each task's final [21, 1, 1024] tensor frozen. Candidate g0 copies the
original optimized-seed PNG and answer; g1/g2/g3 use the existing deterministic
generation_seed(seed, spec, image_id, index) rule. All generation settings,
float32 UNet/VAE, signed VAE normalization, prompts and victim decoding match
the completed 500-task SSIM experiment. No optimization is repeated.

Generation and victim decoding run in separate model scopes. Each candidate
has an atomic checkpoint, allowing resume after interruption. Report both
per-image Yes rates and per-task any-of-four Yes rates, with g0 (the seed used
during optimization) distinguished from the three new seeds. No detector is
run, so a Yes answer alone is not a detector-verified hallucination.

From the project root:
  .venv-stage1/bin/python st-ghost/run_stage2_level_oo_generation_feedback_victim_evaluation_500_four_seeds.py --validate-only
  .venv-stage1/bin/python -u st-ghost/run_stage2_level_oo_generation_feedback_victim_evaluation_500_four_seeds.py
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import run_stage2_level_oo_generation_feedback_victim_evaluation as evaluation


ROOT = evaluation.ROOT
feedback = evaluation.feedback
stage1 = evaluation.stage1
PROTOCOL_VERSION = "stage2_staged21_500_frozen_step45_four_seeds_v1"
SOURCE_DEFAULT = ROOT / "outputs" / (
    "stage2_level_oo_generation_feedback_victim_evaluation_"
    "staged21_500_victim_clean_ssim080_step45_lr002_vae_normalized_v1")
OUTPUT_DEFAULT = ROOT / "outputs" / (
    "stage2_level_oo_generation_feedback_victim_evaluation_"
    "staged21_500_victim_clean_ssim080_step45_lr002_vae_normalized_four_seeds_v1")
INDICES = (0, 1, 2, 3)
ANSWER_FIELDS = ("target_question", "target_answer", "target_decision",
                 "target_yes_probability", "target_query_yes", "target_answer_unparsed")


def read_json(path: Path):
    return json.loads(path.read_text())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def key(row: dict) -> tuple[int, str, int, int]:
    return (*evaluation.task_key(row), int(row["generation_index"]))


def validate_answer(row: dict) -> None:
    decision = stage1.parse_yes_no(row["target_answer"])
    probability = row["target_yes_probability"]
    if (row["target_decision"] is not decision
            or row["target_query_yes"] != int(decision is True)
            or row["target_answer_unparsed"] != int(decision is None)
            or row["target_question"] != row["trigger_prompt"]
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError(f"Invalid victim answer: {evaluation.task_key(row)}")


def load_source(source: Path, *, expected_sample_count: int = 500,
                expected_initialization: str = "stage1") -> tuple[dict, list[dict], dict]:
    """Read only the selected manifest, final tensors, final PNGs and answers."""
    cfg = read_json(source / "resolved_config.json")
    source_signature = read_json(source / "experiment_signature.json")["sha256"]
    completion = read_json(source / "completion_summary.json")
    if expected_sample_count < 1 or expected_initialization not in ("stage1", "clean"):
        raise ValueError("Invalid expected source size/initialization")
    count = expected_sample_count
    if (cfg.get("sample_count") != count or cfg.get("condition_stage_count") != 21
            or cfg.get("condition_mode") != "staged" or cfg.get("full_mode_updates") != 45
            or cfg.get("condition_initialization", "stage1") != expected_initialization
            or cfg.get("vae_image_contract") != feedback.VAE_IMAGE_CONTRACT
            or not evaluation.ssim_enabled(cfg)
            or completion.get("generation_complete") != count
            or completion.get("victim_evaluations_complete") != count
            or completion.get("evaluation_complete") is not True):
        raise ValueError(f"Source must be a completed {count}-task {expected_initialization}-init/SSIM/corrected-VAE experiment")
    tasks = read_json(source / "selected_embeddings.json")
    if (len(tasks) != count or len({evaluation.task_key(t) for t in tasks}) != count
            or len({t["image_id"] for t in tasks}) != count
            or len({t["source_path"] for t in tasks}) != count):
        raise ValueError(f"Expected {count} distinct selected tasks/images")
    with (source / "generated_images.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    generated = {evaluation.task_key(row): row for row in rows}
    if len(rows) != count or set(generated) != {evaluation.task_key(t) for t in tasks}:
        raise ValueError("Final generated-image table and selected tasks differ")
    entries = []
    for task in sorted(tasks, key=evaluation.task_key):
        row = generated[evaluation.task_key(task)]
        if (any(str(row.get(field)) != str(value)
                for field, value in evaluation.descriptor(task).items())
                or row["method"] != "full_step45" or int(row["optimization_updates"]) != 45):
            raise ValueError(f"Final image/task mismatch: {evaluation.task_key(task)}")
        original_seed = feedback.exp4.generation_seed(
            int(task["seed"]), task["spec"], int(task["image_id"]), 0)
        if int(row["generation_seed"]) != original_seed:
            raise ValueError("Source candidate is not the original g0 seed")
        condition_path = evaluation.condition_embedding_path(source, task, 45)
        if str(condition_path) != row["condition_embedding_path"]:
            raise ValueError("Unexpected final condition path")
        payload = torch.load(condition_path, map_location="cpu", weights_only=True)
        schedule = payload["condition_schedule"]
        if task.get("condition_initialization", "stage1") != expected_initialization:
            raise ValueError("Task/source condition initialization differs")
        if expected_initialization == "clean":
            # Clean initialization has no Stage-1 tensor. Validate the reference
            # against its signed generation checkpoint before using it.
            reference_path = evaluation.clean_reference_path(source, task)
            generation_path = evaluation.generation_checkpoint_path(source, task)
            generation = read_json(generation_path)
            reference_record = generation.get("clean_reference", {})
            if (generation.get("experiment_signature") != source_signature
                    or generation.get("task") != evaluation.descriptor(task)
                    or reference_record.get("path") != str(reference_path)
                    or reference_record.get("sha256") != evaluation.sha256(reference_path)):
                raise ValueError(f"Invalid clean reference provenance: {reference_path}")
            reference = torch.load(reference_path, map_location="cpu", weights_only=True)
            initial = reference.get("clean_embedding")
            if (reference.get("format") != "stage2_clean_reference_v1"
                    or reference.get("protocol_version") != evaluation.PROTOCOL_VERSION
                    or reference.get("task") != evaluation.descriptor(task)
                    or not isinstance(initial, torch.Tensor) or initial.shape != (1, 1024)
                    or initial.dtype != torch.float32 or not torch.isfinite(initial).all().item()
                    or not isinstance(reference.get("target_embedding"), torch.Tensor)
                    or not torch.equal(initial, reference["target_embedding"])):
                raise ValueError(f"Invalid clean initial embedding: {reference_path}")
        else:
            initial = feedback._load_embedding(Path(task["embedding_path"]), task["embedding_sha256"])
        conditions = evaluation.load_condition_schedule(
            condition_path, row["condition_embedding_sha256"], schedule,
            task, 45, original_seed, initial_embedding=initial)
        if (conditions.shape != (21, 1, 1024) or len(schedule["active_timesteps"]) != 21
                or schedule["stage_mapping"] != list(range(21))):
            raise ValueError("Expected exactly one saved condition per executed denoising step")
        original_image = Path(row["generated_path"]).resolve()
        if original_image != evaluation.generated_path(source, task, "full_step45"):
            raise ValueError("Unexpected original image path")
        image_state = evaluation.image_state(original_image)
        if image_state["sha256"] != row["sha256"] or image_state["size"] != int(row["size"]):
            raise ValueError(f"Source PNG digest mismatch: {original_image}")
        victim_path = evaluation.victim_checkpoint_path(source, {**task, "method": "full_step45"})
        victim_payload = read_json(victim_path)
        original_result = victim_payload["result"]
        if (victim_payload["experiment_signature"] != source_signature
                or any(original_result.get(field) != value
                       for field, value in evaluation.descriptor(task).items())
                or original_result.get("condition_embedding_sha256") != row["condition_embedding_sha256"]
                or original_result.get("generated_path") != str(original_image)
                or original_result.get("image_state") != image_state
                or original_result.get("generation_seed") != original_seed):
            raise ValueError(f"Source victim result mismatch: {victim_path}")
        validate_answer(original_result)
        seeds = [feedback.exp4.generation_seed(
            int(task["seed"]), task["spec"], int(task["image_id"]), index) for index in INDICES]
        if len(set(seeds)) != 4:
            raise ValueError("Candidate seed collision within a task")
        source_image_sha256 = evaluation.sha256(Path(task["source_path"]))
        if task.get("source_image_sha256", source_image_sha256) != source_image_sha256:
            raise ValueError("Clean source image changed since task selection")
        entries.append({
            **evaluation.descriptor(task),
            "source_image_sha256": source_image_sha256,
            "condition_embedding_path": str(condition_path),
            "condition_embedding_sha256": row["condition_embedding_sha256"],
            "condition_schedule": schedule,
            "optimization_generation_seed": original_seed,
            "generation_seeds": seeds,
            "original_generated_path": str(original_image),
            "original_image_state": image_state,
            "original_ssim_256": float(row["ssim_256"]),
            "original_clean_cosine": float(row["clean_cosine"]),
            "original_answer": {field: original_result[field] for field in ANSWER_FIELDS},
            "original_victim_checkpoint": str(victim_path),
            "original_victim_checkpoint_sha256": evaluation.sha256(victim_path),
        })
    files = [source / name for name in (
        "resolved_config.json", "experiment_signature.json", "selected_embeddings.json",
        "generated_images.csv", "completion_summary.json")]
    files += [Path(feedback.__file__), Path(evaluation.__file__), Path(stage1.__file__)]
    provenance = {"source_output": str(source), "source_experiment_signature": source_signature,
                  "input_file_sha256": {str(path): evaluation.sha256(path) for path in files}}
    return cfg, entries, provenance


def identity(entry: dict, index: int, output: Path) -> dict:
    image = (output / "generated" / "full_step45" / entry["spec"] / feedback.LEVEL
             / f"seed{entry['seed']}_{entry['image_id']}_g{index}.png").resolve()
    return {**evaluation.descriptor(entry), "source_image_sha256": entry["source_image_sha256"],
            "method": "full_step45", "optimization_updates": 45, "additional_optimization_updates": 0,
            "condition_embedding_path": entry["condition_embedding_path"],
            "condition_embedding_sha256": entry["condition_embedding_sha256"],
            "generation_index": index, "generation_seed": entry["generation_seeds"][index],
            "optimization_generation_seed": entry["optimization_generation_seed"],
            "seed_used_during_optimization": int(index == 0), "generated_path": str(image),
            "generation_origin": "reused_original_g0" if index == 0 else "new_seed_frozen_conditions"}


def checkpoint_path(output: Path, row: dict) -> Path:
    return (output / "candidate_checkpoints" / f"seed_{row['seed']}"
            / f"{row['spec']}__{row['image_id']}__g{row['generation_index']}.json")


def write_record(output: Path, signature: str, row: dict) -> None:
    feedback._save_json_atomic(checkpoint_path(output, row), {
        "experiment_signature": signature, "result": row})


def load_records(output: Path, signature: str, entries: list[dict]) -> dict:
    expected = {key(row): row for entry in entries for index in INDICES
                for row in [identity(entry, index, output)]}
    records = {}
    for path in sorted((output / "candidate_checkpoints").glob("seed_*/*.json")):
        payload = read_json(path)
        row = payload["result"]
        candidate_key = key(row)
        if (payload["experiment_signature"] != signature or candidate_key not in expected
                or candidate_key in records or path != checkpoint_path(output, row)
                or any(row.get(field) != value for field, value in expected[candidate_key].items())
                or evaluation.image_state(Path(row["generated_path"])) != row.get("image_state")):
            raise ValueError(f"Incompatible candidate checkpoint: {path}")
        if any(field in row for field in ANSWER_FIELDS):
            if not all(field in row for field in ANSWER_FIELDS):
                raise ValueError(f"Incomplete victim answer: {path}")
            validate_answer(row)
        records[candidate_key] = row
    return records


def import_originals(output: Path, signature: str, entries: list[dict], records: dict) -> None:
    for entry in entries:
        row = identity(entry, 0, output)
        if key(row) in records:
            continue
        path = Path(row["generated_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.png")
        shutil.copy2(entry["original_generated_path"], temporary)
        os.replace(temporary, path)
        state = evaluation.image_state(path)
        if state["sha256"] != entry["original_image_state"]["sha256"]:
            raise ValueError("Copied g0 PNG differs from the original")
        row.update(image_state=state, ssim_256=entry["original_ssim_256"],
                   clean_cosine=entry["original_clean_cosine"], **entry["original_answer"])
        write_record(output, signature, row)
        records[key(row)] = row


def mean(rows: list[dict], field: str):
    values = [float(row[field]) for row in rows if field in row]
    return sum(values) / len(values) if values else None


def save_outputs(output: Path, entries: list[dict], records: dict, *,
                 protocol_version: str = PROTOCOL_VERSION) -> dict:
    count = len(entries)
    if not count:
        raise ValueError("Cannot summarize an empty task set")
    rows = [records[candidate_key] for candidate_key in sorted(records)]
    evaluated = [row for row in rows if "target_query_yes" in row]
    tasks = []
    for entry in entries:
        task_rows = [records[(*evaluation.task_key(entry), index)] for index in INDICES
                     if (*evaluation.task_key(entry), index) in records]
        answers = [row for row in task_rows if "target_query_yes" in row]
        hits = [row["generation_index"] for row in answers if row["target_query_yes"]]
        tasks.append({**evaluation.descriptor(entry), "images_generated": len(task_rows),
                      "images_evaluated": len(answers), "original_g0_query_yes":
                          entry["original_answer"]["target_query_yes"],
                      "successful_generation_indices": hits, "any_hit_observed": int(bool(hits)),
                      "four_candidate_query_yes": int(bool(hits)) if len(answers) == 4 else None,
                      "new_seeds_any_hit": int(any(index > 0 for index in hits)),
                      "max_evaluated_yes_probability": max(
                          (row["target_yes_probability"] for row in answers), default=None)})
    by_index = []
    for index in INDICES:
        generated = [row for row in rows if row["generation_index"] == index]
        answers = [row for row in evaluated if row["generation_index"] == index]
        by_index.append({"generation_index": index, "seed_used_during_optimization": index == 0,
                         "images_expected": count, "images_generated": len(generated),
                         "images_evaluated": len(answers), "query_yes": sum(
                             row["target_query_yes"] for row in answers),
                         "per_image_query_yes_rate": mean(answers, "target_query_yes"),
                         "mean_yes_probability": mean(answers, "target_yes_probability"),
                         "mean_ssim_256": mean(generated, "ssim_256"),
                         "mean_clean_cosine": mean(generated, "clean_cosine")})
    completed = [task for task in tasks if task["images_evaluated"] == 4]
    baseline_hits = sum(entry["original_answer"]["target_query_yes"] for entry in entries)
    observed_hits = sum(task["any_hit_observed"] for task in tasks)
    cumulative = []
    for budget in range(1, 5):
        groups = [[records.get((*evaluation.task_key(entry), index)) for index in range(budget)]
                  for entry in entries]
        complete_groups = [group for group in groups if all(
            row is not None and "target_query_yes" in row for row in group)]
        successes = sum(any(row["target_query_yes"] for row in group) for group in complete_groups)
        cumulative.append({"candidate_budget": budget, "tasks_evaluated": len(complete_groups),
                           "tasks_query_yes": successes,
                           "task_any_hit_rate": successes / len(complete_groups) if complete_groups else None})
    summary = {
        "protocol_version": protocol_version, "updated_at": utc_now(),
        "success_definition": "victim_answers_yes_no_detector_verification",
        "tasks_expected": count, "candidates_per_task": 4, "images_expected": count * 4,
        "images_generated": len(rows), "images_evaluated": len(evaluated),
        "new_seed_images_generated": sum(row["generation_index"] > 0 for row in rows),
        "new_seed_images_expected": count * 3, "four_candidate_tasks_complete": len(completed),
        "evaluation_complete": len(completed) == count,
        "baseline_g0_success_tasks": baseline_hits, "baseline_g0_success_rate": baseline_hits / count,
        "observed_any_hit_tasks": observed_hits,
        "observed_any_hit_rate_lower_bound_all_tasks": observed_hits / count,
        "additional_hit_tasks_over_g0": observed_hits - baseline_hits,
        "completed_tasks_any_hit_rate": mean(completed, "four_candidate_query_yes"),
        "all_images_query_yes_rate": mean(evaluated, "target_query_yes"),
        "new_seed_images_query_yes_rate": mean(
            [row for row in evaluated if row["generation_index"] > 0], "target_query_yes"),
        "by_generation_index": by_index, "cumulative_candidate_budget": cumulative,
    }
    if count == 500:
        summary["observed_any_hit_rate_lower_bound_all_500"] = observed_hits / count
    feedback._save_csv_atomic(output / "generated_images.csv", rows)
    feedback._save_csv_atomic(output / "victim_evaluations.csv", evaluated)
    feedback._save_csv_atomic(output / "task_results.csv", tasks)
    feedback._save_csv_atomic(output / "summary_by_generation_index.csv", by_index)
    feedback._save_json_atomic(output / "summary.json", summary)
    return summary


@torch.no_grad()
def generate(output: Path, signature: str, cfg: dict, entries: list[dict],
             records: dict, max_new_images: int | None, *,
             protocol_version: str = PROTOCOL_VERSION) -> None:
    pending = [(entry, index) for entry in entries for index in INDICES[1:]
               if (*evaluation.task_key(entry), index) not in records]
    if max_new_images is not None:
        pending = pending[:max_new_images]
    if not pending:
        return
    runtime = None
    try:
        # Reuse the exact reference runtime and rendering code; no victim or
        # projector is loaded here, and autograd is disabled throughout.
        runtime = feedback.exp4.Runtime(cfg, attack=False)
        runtime.load_diffusion()
        runtime.pipe.unet.to(dtype=torch.float32).requires_grad_(False)
        runtime.pipe.vae.to(dtype=torch.float32).requires_grad_(False)
        runtime.pipe.text_encoder.requires_grad_(False)
        previous_key = None
        for completed, (entry, index) in enumerate(tqdm(pending, desc="Frozen conditions: new seeds"), 1):
            if evaluation.task_key(entry) != previous_key:
                if previous_key is not None:
                    source.close()
                if evaluation.sha256(Path(entry["source_path"])) != entry["source_image_sha256"]:
                    raise ValueError("Clean source image changed after input validation")
                with Image.open(entry["source_path"]) as handle:
                    source = handle.convert("RGB")
                condition_path = Path(entry["condition_embedding_path"])
                if evaluation.sha256(condition_path) != entry["condition_embedding_sha256"]:
                    raise ValueError("Frozen conditions changed after input validation")
                conditions = torch.load(condition_path, map_location="cpu", weights_only=True)["conditions"]
                conditions = conditions.to(runtime.device)
                source_rgb = runtime.diffusion_preprocess(source).unsqueeze(0).to(runtime.device)
                reference = evaluation.downsample_for_ssim(source_rgb)
                with source.resize((336, 336), Image.Resampling.LANCZOS) as clean:
                    z0 = runtime.clip.images([clean]).detach().float()
                previous_key = evaluation.task_key(entry)
            row = identity(entry, index, output)
            started = time.monotonic()
            state = feedback._make_state(
                runtime, source, row["generation_seed"], entry["trigger_prompt"], cfg)
            if evaluation.condition_schedule(state, cfg) != entry["condition_schedule"]:
                raise ValueError("New-seed denoising schedule differs from saved 21-step schedule")
            image = evaluation.render_conditions(runtime, state, conditions, cfg)
            z_hat = feedback._clip_image(runtime, image)
            row["clean_cosine"] = float(F.cosine_similarity(z_hat, z0, dim=-1).mean().cpu())
            row["ssim_256"] = float(evaluation.rgb_ssim(
                evaluation.downsample_for_ssim(image), reference).cpu())
            path = Path(row["generated_path"])
            temporary = path.with_suffix(".tmp.png")
            feedback._save_image(image, temporary)
            os.replace(temporary, path)
            row["image_state"] = evaluation.image_state(path)
            row["generation_seconds"] = time.monotonic() - started
            write_record(output, signature, row)
            records[key(row)] = row
            print(json.dumps({"event": "generated", "task": list(evaluation.task_key(entry)),
                              "generation_index": index, "generation_seed": row["generation_seed"],
                              "seconds": row["generation_seconds"]}), flush=True)
            del state, image, z_hat
            if completed % 10 == 0 or completed == len(pending):
                save_outputs(output, entries, records, protocol_version=protocol_version)
        source.close()
    finally:
        del runtime
        gc.collect()
        torch.cuda.empty_cache()


def evaluate(output: Path, signature: str, cfg: dict, entries: list[dict], records: dict, *,
             protocol_version: str = PROTOCOL_VERSION) -> None:
    pending = [records[candidate_key] for candidate_key in sorted(records)
               if "target_query_yes" not in records[candidate_key]]
    if not pending:
        return
    victim = None
    try:
        stage1.seed_all(42)
        victim = stage1.VictimRuntime(cfg, torch.device("cuda"))
        for completed, row in enumerate(tqdm(pending, desc="Victim: new-seed images"), 1):
            if evaluation.image_state(Path(row["generated_path"])) != row["image_state"]:
                raise ValueError("Generated PNG changed before victim evaluation")
            result = evaluation.evaluate_image(victim, row)
            validate_answer(result)
            write_record(output, signature, result)
            records[key(result)] = result
            if completed % 10 == 0 or completed == len(pending):
                summary = save_outputs(output, entries, records, protocol_version=protocol_version)
                print(json.dumps({"event": "evaluated", "images_evaluated": summary["images_evaluated"],
                                  "observed_any_hit_tasks": summary["observed_any_hit_tasks"],
                                  "additional_hit_tasks_over_g0": summary["additional_hit_tasks_over_g0"]}), flush=True)
    finally:
        del victim
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--phase", choices=("all", "generate", "evaluate"), default="all")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-new-images", type=int, help="Limit only new generation for a smoke test; resume without this flag")
    args = parser.parse_args()
    if args.max_new_images is not None and args.max_new_images < 1:
        parser.error("--max-new-images must be positive")
    run_experiment(args)


def run_experiment(args: argparse.Namespace, *, expected_sample_count: int = 500,
                   expected_initialization: str = "stage1",
                   protocol_version: str = PROTOCOL_VERSION) -> None:
    """Reuse frozen-condition four-seed evaluation for a validated task set."""
    output, source = args.output_dir.resolve(), args.source_output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Use a separate output directory outside the source experiment")
    cfg, entries, provenance = load_source(
        source, expected_sample_count=expected_sample_count,
        expected_initialization=expected_initialization)
    image_count = len(entries) * len(INDICES)
    cfg = {**cfg, "protocol_version": protocol_version, "output_dir": str(output),
           "source_output": str(source), "num_generations": 4,
           "generation_indices": list(INDICES), "additional_optimization_updates": 0,
           "conditions_frozen": True, "reuse_original_g0": True,
           "candidate_seed_rule": "run_stage1_exp4.generation_seed(seed,spec,image_id,index)",
           "candidate_noise_policy": "resample_vae_posterior_diffusion_noise_and_embedding_noise_with_seed"}
    signature = hashlib.sha256(json.dumps(
        {"config": cfg, "entries": entries, "provenance": provenance},
        sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    expected_signature = {"protocol_version": protocol_version, "sha256": signature}
    if (output / "experiment_signature.json").exists():
        if read_json(output / "experiment_signature.json") != expected_signature:
            raise ValueError("Output signature differs; refusing to mix experiments")
    elif output.exists() and any(path.name not in {"run.lock", "service"} for path in output.iterdir()):
        raise ValueError("Nonempty output lacks a matching experiment signature")
    if args.validate_only:
        records = load_records(output, signature, entries)
        print(json.dumps({"source_tasks_validated": len(entries), "condition_shape": [21, 1, 1024],
                          "images_planned": image_count, "original_g0_yes": sum(
                              entry["original_answer"]["target_query_yes"] for entry in entries),
                          "candidate_checkpoints_validated": len(records),
                          "experiment_signature": signature}, indent=2), flush=True)
        return
    with evaluation.output_lock(output):
        feedback._save_json_atomic(output / "experiment_signature.json", expected_signature)
        feedback._save_json_atomic(output / "resolved_config.json", cfg)
        feedback._save_json_atomic(output / "input_provenance.json", provenance)
        feedback._save_json_atomic(output / "selected_tasks.json", entries)
        records = load_records(output, signature, entries)
        import_originals(output, signature, entries, records)
        summary = save_outputs(output, entries, records, protocol_version=protocol_version)
        if args.prepare_only:
            print(json.dumps(summary, indent=2), flush=True)
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        started = utc_now()
        status = {"started_at": started, "pid": os.getpid(), "phase": args.phase,
                  "status": "running", "output_dir": str(output)}
        feedback._save_json_atomic(output / "run_status.json", status)
        print(json.dumps({"event": "started", **status,
                          "existing_images": len(records), "new_images_pending": image_count - len(records)}), flush=True)
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}; completed candidates are checkpointed")
        signal.signal(signal.SIGTERM, interrupted)
        try:
            if args.phase in ("all", "generate"):
                generate(output, signature, cfg, entries, records, args.max_new_images,
                         protocol_version=protocol_version)
            if args.phase in ("all", "evaluate"):
                evaluate(output, signature, cfg, entries, records, protocol_version=protocol_version)
        except BaseException as error:
            summary = save_outputs(output, entries, records, protocol_version=protocol_version)
            feedback._save_json_atomic(output / "run_status.json", {
                **status, "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "finished_at": utc_now(), "error": repr(error),
                "images_generated": summary["images_generated"],
                "images_evaluated": summary["images_evaluated"]})
            raise
        summary = save_outputs(output, entries, records, protocol_version=protocol_version)
        feedback._save_json_atomic(output / "run_status.json", {
            **status, "status": "completed" if summary["evaluation_complete"] else "partial",
            "finished_at": utc_now(), "images_generated": summary["images_generated"],
            "images_evaluated": summary["images_evaluated"]})
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
