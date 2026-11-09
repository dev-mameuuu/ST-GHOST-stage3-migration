"""Run the complete Stage 2 atomic-guided experiment and archive embeddings.

This runner uses the same five-pair source pool, weaker-target guidance,
optimization budget, prompt schedule, generation settings, and three seeds as
``run_stage2_level_oo_atomic_guided_all.py``.  It deliberately reruns all
tasks: the older combined output imported two-pair checkpoints that did not
contain the terminal embedding tensor.

For every task, including tasks that do not reach the embedding threshold, the
terminal optimized embedding is saved as an independent CPU ``float32``
``.pt`` file.  The JSON checkpoint records its path, shape, norm, and
SHA-256 digest.  A later run validates these files before skipping a completed
task, so deleting or replacing an embedding cannot silently pass resume.

This script performs no post-generation victim query or detector evaluation.
It only archives the terminal embedding and the same generated images as the
atomic-guided experiment.

Validate configuration and the selected source pool without loading models::

    .venv-stage1/bin/python st-ghost/run_stage2_level_oo_atomic_guided_embeddings.py \
        --validate-only

Run the full archive-producing experiment::

    .venv-stage1/bin/python st-ghost/run_stage2_level_oo_atomic_guided_embeddings.py

The default output is ``outputs/stage2_level_oo_atomic_guided_embeddings_v1``.
The runner resumes completed tasks when their checkpoint and embedding file
are intact; use ``--no-resume`` to require a new output directory.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

import torch
from tqdm import tqdm

STAGE1 = Path(__file__).resolve().parent
if str(STAGE1) not in sys.path:
    sys.path.insert(0, str(STAGE1))

import run_stage2_level_oo_atomic_guided as guided  # noqa: E402


base = guided.base
LEVEL = guided.LEVEL
PROTOCOL_VERSION = "stage2_level_oo_atomic_guided_embeddings_v1"
OUTPUT_DEFAULT = base.ROOT / "outputs" / PROTOCOL_VERSION
SOURCE_OUTPUT_DEFAULT = guided.SOURCE_OUTPUT_DEFAULT
ALL_GUIDED_SPECS = guided.ALL_GUIDED_SPECS
DEFAULT_LAMBDA_ATOM = guided.DEFAULT_LAMBDA_ATOM


def load_config(source_cfg: Mapping[str, object], source_output: Path,
                output: Path, resume: bool,
                lambda_atom: float = DEFAULT_LAMBDA_ATOM) -> dict:
    """Use the all-pairs guided settings and add the archive protocol fields."""
    if not math.isfinite(lambda_atom) or lambda_atom < 0:
        raise ValueError("lambda_atom must be finite and non-negative")
    cfg = guided.load_config(
        source_cfg, source_output, output, resume,
        lambda_atom=lambda_atom, spec_names=ALL_GUIDED_SPECS)
    cfg["protocol_version"] = PROTOCOL_VERSION
    cfg["reference_protocol_version"] = guided.PROTOCOL_VERSION
    cfg["embedding_archive"] = "terminal_task_embedding"
    cfg["embedding_storage_dtype"] = "float32"
    cfg["embedding_storage_device"] = "cpu"
    return cfg


def experiment_signature(cfg: Mapping[str, object],
                         manifest: Sequence[Mapping[str, object]]) -> str:
    """Hash the full guided protocol plus the embedding archive contract."""
    reference = guided.experiment_signature(cfg, manifest)
    payload = json.dumps({
        "protocol_version": PROTOCOL_VERSION,
        "reference_guided_signature": reference,
        "config": {key: value for key, value in cfg.items()
                   if key not in ("output_dir", "resume", "checkpoint_every")},
        "manifest": list(manifest),
        "embedding_archive": {
            "state": "terminal_after_optimization",
            "dtype": "float32",
            "device": "cpu",
            "format": "torch_save_dict_v1",
        },
    }, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _embedding_path(output: Path, seed: int, spec: str, image_id: int) -> Path:
    safe = base._safe_name(spec)
    return (output / "embeddings" / f"seed_{seed}"
            / f"{safe}__{image_id}__{LEVEL}.pt").resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_embedding_atomic(path: Path, embedding: torch.Tensor,
                           metadata: Mapping[str, object]) -> dict:
    """Write a CPU tensor atomically, then return its auditable file metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = embedding.detach().to(device="cpu", dtype=torch.float32).contiguous()
    payload = {
        "format": "stage2_terminal_embedding_v1",
        "embedding": tensor,
        "metadata": dict(metadata),
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.",
                suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {
        "embedding_path": str(path),
        "embedding_sha256": _sha256(path),
        "embedding_shape": list(tensor.shape),
        "embedding_dtype": str(tensor.dtype).replace("torch.", ""),
        "embedding_device": "cpu",
        "embedding_norm": float(torch.linalg.vector_norm(tensor).item()),
        "embedding_numel": int(tensor.numel()),
    }


def _validate_embedding_record(result: Mapping[str, object], output: Path,
                               signature: str) -> None:
    """Reject a resumed task if its archived tensor is missing or changed."""
    try:
        seed = int(result["seed"])
        spec = str(result["spec"])
        image_id = int(result["image_id"])
        path = Path(str(result["embedding_path"])).resolve()
        expected = _embedding_path(output, seed, spec, image_id)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Checkpoint has incomplete embedding metadata") from exc
    if path != expected:
        raise RuntimeError(f"Embedding path mismatch: {path}")
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Checkpoint references missing embedding: {path}")
    expected_digest = str(result.get("embedding_sha256", ""))
    if not expected_digest or _sha256(path) != expected_digest:
        raise RuntimeError(f"Embedding digest mismatch: {path}")
    if result.get("embedding_signature") != signature:
        raise RuntimeError(f"Embedding signature mismatch: {path}")
    if result.get("embedding_dtype") != "float32" or result.get("embedding_device") != "cpu":
        raise RuntimeError(f"Unsupported embedding storage metadata: {path}")


def _ensure_output_compatibility(output: Path, signature: str,
                                 resume: bool) -> None:
    signature_path = output / "experiment_signature.json"
    checkpoints = list(output.glob("seed_*/checkpoints/*.json"))
    if signature_path.exists():
        existing = json.loads(signature_path.read_text())
        if (existing.get("protocol_version") != PROTOCOL_VERSION
                or existing.get("sha256") != signature):
            raise RuntimeError(f"Use a new output directory: incompatible run at {output}")
    elif output.exists() and any(entry.name != "service" for entry in output.iterdir()):
        raise RuntimeError(f"Output is nonempty but has no experiment signature: {output}")
    if not resume and (checkpoints or output.exists() and any(output.iterdir())):
        raise RuntimeError("--no-resume requires an empty output directory")


def _write_run_metadata(output: Path, source_output: Path, cfg: Mapping[str, object],
                        manifest: Sequence[Mapping[str, object]], specs,
                        signature: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    base.exp4.save_json_atomic(output / "experiment_signature.json", {
        "protocol_version": PROTOCOL_VERSION, "sha256": signature,
    })
    base.exp4.save_json_atomic(output / "resolved_config.json", dict(cfg))
    base.exp4.save_json_atomic(output / "target_specs.json",
                               [spec.to_dict() for spec in specs])
    base.exp4.save_json_atomic(output / "level_oo_shared_pool_manifest.json",
                               list(manifest))
    base.exp4.save_json_atomic(output / "source_manifest_reference.json", {
        "source_output_dir": str(source_output),
        "source_manifest": str(source_output / "level_oo_shared_pool_manifest.json"),
        "source_row_count": len(manifest),
        "selected_specs": [spec.name for spec in specs],
        "reference_protocol_version": base.PROTOCOL_VERSION,
        "pool_reselected": False,
        "post_generation_evaluation": False,
        "embedding_archive": "terminal_task_embedding",
    })


def _validate_resumed_results(results: Sequence[Mapping[str, object]],
                              generations: Sequence[Mapping[str, object]],
                              manifest: Sequence[Mapping[str, object]],
                              seed: int, output: Path, signature: str) -> None:
    expected_by_key = {
        (str(row["spec"]), base._as_int(row["image_id"])): row
        for row in manifest
    }
    expected = set(expected_by_key)
    completed = {(str(row["spec"]), base._as_int(row["image_id"]))
                 for row in results}
    if len(completed) != len(results) or not completed.issubset(expected):
        raise RuntimeError(f"Unexpected or duplicate checkpoints in seed_{seed}")
    if any(base._as_int(row["seed"]) != seed or row["level"] != LEVEL
           for row in results):
        raise RuntimeError(f"Checkpoint seed/level mismatch in seed_{seed}")
    for row in results:
        key = (str(row["spec"]), base._as_int(row["image_id"]))
        if Path(str(row.get("source_path", ""))).resolve() != Path(
                str(expected_by_key[key]["source_path"])).resolve():
            raise RuntimeError(f"Checkpoint source image mismatch in seed_{seed}")
        _validate_embedding_record(row, output, signature)
    for row in generations:
        path = Path(row["generated_path"])
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Checkpoint references missing image: {path}")


def _save_tables(output: Path, results: Sequence[Mapping[str, object]],
                 traces: Sequence[Mapping[str, object]],
                 generations: Sequence[Mapping[str, object]], prefix: str = "") -> None:
    for name, rows in (("results", results), ("steps", traces),
                       ("generations", generations)):
        base.exp4.save_csv_atomic(output / f"{prefix}{name}.csv", rows)


def build_group_summary(results, specs):
    valid = [row for row in results if not row.get("numerical_error")]
    summaries = guided.build_group_summary(valid, specs)
    for summary in summaries:
        rows = [row for row in results
                if summary["spec"] == "ALL" or row["spec"] == summary["spec"]]
        if summary["group"] == "embedding_success":
            rows = [row for row in rows if row["embedding_success"]]
        elif summary["group"] == "embedding_failure":
            rows = [row for row in rows if not row["embedding_success"]]
        elif summary["group"] == "embedding_success_generation_completed":
            rows = [row for row in rows
                    if row["embedding_success"] and row["image_generation_completed"]]
        summary["n_valid_loss_tasks"] = summary["n"]
        summary["n"] = len(rows)
        summary["n_numerical_failure"] = sum(bool(row.get("numerical_error"))
                                             for row in rows)
    return summaries


def run_seed(runtime, dset, manifest, specs, seed, cfg, output, signature):
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("resume", True):
        results, traces, generations = base._load_checkpoints(seed_dir, signature)
    else:
        results, traces, generations = [], [], []
    _validate_resumed_results(results, generations, manifest, seed, output, signature)
    completed = {(str(row["spec"]), base._as_int(row["image_id"]))
                 for row in results}
    spec_by_name = {spec.name: spec for spec in specs}
    contrasts = {}
    with tqdm(total=len(manifest), desc=f"Stage 2 embedding archive seed {seed}") as progress:
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
            result, task_traces, task_generations, terminal_embedding = guided.optimize_task(
                runtime, image, spec, seed, image_id, cfg,
                contrasts[spec.name], seed_dir, return_embedding=True)
            result.update({
                "source_path": str(source_path),
                "pool_index": base._as_int(item.get("pool_index")),
                "clip_base_score": base._as_float(item.get("base_score")),
            })
            embedding_metadata = _save_embedding_atomic(
                _embedding_path(output, seed, spec.name, image_id),
                terminal_embedding,
                {
                    "experiment_signature": signature,
                    "seed": seed,
                    "spec": spec.name,
                    "image_id": image_id,
                    "level": LEVEL,
                    "embedding_state": "terminal_after_optimization",
                    "embedding_success": int(result["embedding_success"]),
                    "terminal_step": int(result["terminal_step"]),
                    "source_path": str(source_path),
                })
            result.update(embedding_metadata)
            result.update({
                "embedding_signature": signature,
                "embedding_state": "terminal_after_optimization",
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
            del terminal_embedding
            gc.collect()
            torch.cuda.empty_cache()
            progress.update(1)
    _save_tables(seed_dir, results, traces, generations)
    return results, traces, generations


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, default=SOURCE_OUTPUT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--no-resume", action="store_true",
                        help="Require a new empty output directory")
    parser.add_argument("--lambda-atom", type=float, default=DEFAULT_LAMBDA_ATOM,
                        help="Dynamic atomic guidance coefficient; default matches atomic_guided_all")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate source/config without loading models or writing output")
    args = parser.parse_args(argv)
    if not math.isfinite(args.lambda_atom) or args.lambda_atom < 0:
        parser.error("--lambda-atom must be a finite non-negative number")
    return args


def main(argv=None):
    args = parse_args(argv)
    source_output, output = args.source_output.resolve(), args.output_dir.resolve()
    source_cfg, manifest, specs = guided.validate_source_output(
        source_output, ALL_GUIDED_SPECS)
    cfg = load_config(source_cfg, source_output, output,
                      not args.no_resume, args.lambda_atom)
    signature = experiment_signature(cfg, manifest)
    _ensure_output_compatibility(output, signature, cfg["resume"])
    print(json.dumps({
        "protocol_version": PROTOCOL_VERSION,
        "reference_protocol_version": guided.PROTOCOL_VERSION,
        "target_specs": [spec.name for spec in specs],
        "selected_pool_rows": len(manifest),
        "total_tasks": len(manifest) * len(cfg["seeds"]),
        "seeds": cfg["seeds"],
        "num_steps": cfg["num_steps"],
        "learning_rate": cfg["learning_rate"],
        "lambda_atom": cfg["lambda_atom"],
        "lambda_clip": cfg["lambda_clip"],
        "lambda_reg": cfg["lambda_reg"],
        "yes_threshold": cfg["yes_threshold"],
        "num_generations": cfg["num_generations"],
        "embedding_storage": "CPU float32 torch-save dict",
        "output_dir": str(output),
        "experiment_signature": signature,
        "validate_only": args.validate_only,
    }, indent=2), flush=True)
    if args.validate_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the atomic-guided embedding archive")
    _write_run_metadata(output, source_output, cfg, manifest, specs, signature)
    runtime = guided.GuidedRuntime(cfg)
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
    base.exp4.save_json_atomic(output / "completion_summary.json", {
        "expected_tasks": len(manifest) * len(cfg["seeds"]),
        "completed_tasks": len(all_results),
        "complete": len(all_results) == len(manifest) * len(cfg["seeds"]),
        "embedding_files": sum(1 for row in all_results if row.get("embedding_path")),
        "embedding_archive": "terminal_after_optimization",
        "by_spec": {spec.name: {
            "completed": sum(row["spec"] == spec.name for row in all_results),
            "expected": int(cfg["num_images_per_spec"]) * len(cfg["seeds"]),
        } for spec in specs},
    })


if __name__ == "__main__":
    main()
