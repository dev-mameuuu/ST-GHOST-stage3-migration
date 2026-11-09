"""Stage-3 ST-GHOST: 1,000 clean initializations and four final candidates.

Select 200 distinct images per target pair from the original-GHOST 2,000-image
manifest, independently of attack outcomes. Initialize all 21 denoising-step
conditions from CLIP(original image), then run 45 Adam updates at lr=0.03.
Reuse the previous clean-init objective, 2*L_victim + L_clean + L_ssim, the
SSIM_256 threshold of 0.8, and the corrected signed VAE image contract.

Freeze the final [21, 1, 1024] conditions. Candidate g0 is the final image made
with the optimization seed; g1/g2/g3 use three other deterministic seeds, with
no further optimization. An image/task succeeds when any of its four images
makes the victim answer Yes. This is a Yes-answer metric, without a detector.
Each clean image is one task, not four optimization runs.

Outputs are separate from historical experiments:
  optimization/   final conditions, g0 images, loss traces and g0 answers
  four_seeds/     all four images, answers and task-level any-of-four summaries

From the project root:
  .venv-stage1/bin/python st-ghost/run_stage3_st_ghost.py --prepare-only
  .venv-stage1/bin/python -u st-ghost/run_stage3_st_ghost.py

--validate-only is read-only; --prepare-only freezes inputs without GPU models.
--phase optimize stops after optimization and g0 evaluation; generate also
renders the other three seeds; evaluate queries saved images without diffusion.
Default all runs the complete experiment. Completed tasks/candidates resume.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import run_stage2_level_oo_generation_feedback_victim_evaluation as evaluation
import run_stage2_level_oo_generation_feedback_victim_evaluation_500_four_seeds as four_seeds


ROOT = evaluation.ROOT
PROTOCOL_VERSION = "stage3_st_ghost_clean_init_staged21_1000_step45_lr003_four_seeds_v1"
EXPERIMENT_NAME = PROTOCOL_VERSION
OUTPUT_DEFAULT = ROOT / "outputs" / EXPERIMENT_NAME
SOURCE_DEFAULT = ROOT / "outputs" / "stage3_original_ghost_comparison"
REFERENCE_DEFAULT = ROOT / "outputs" / (
    "stage2_level_oo_generation_feedback_victim_evaluation_"
    "staged21_750_clean_init_victim_clean_ssim080_step45_lr002_vae_normalized_v1")
SOURCE_PROTOCOL = "stage3_original_ghost_comparison_multi_o_no_seed_v1"
SAMPLE_COUNT = 1000
SOURCE_COUNT = 2000
LEARNING_RATE = 0.03
TASK_SEED = 42
PLAN = evaluation.EvaluationPlan(
    saved_updates=(45,), methods=((evaluation.FULL_METHOD, 45),), summary_every=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-output", type=Path, default=SOURCE_DEFAULT,
                        help="Original-GHOST experiment containing selected_images.json (2,000 images)")
    parser.add_argument("--reference-output", type=Path, default=REFERENCE_DEFAULT,
                        help="Previous 750-task clean-init configuration; no embeddings/results are loaded")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--phase", choices=("all", "optimize", "generate", "evaluate"), default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.selection_seed = 42
    args.initialization = "clean"
    args.learning_rate = LEARNING_RATE
    args.lambda_victim = 2.0
    args.condition_mode = "staged"
    args.clean_slack = 0.02
    args.ssim = True
    args.ssim_threshold = 0.8
    return args


def select_images(manifest: list[dict], spec_names: list[str], *, count: int = SAMPLE_COUNT,
                  source_count: int = SOURCE_COUNT, selection_seed: int = 42) -> list[dict]:
    """Balanced sampling from the frozen clean-image pool, never attack outcomes."""
    names = sorted(spec_names)
    if (len(set(names)) != len(names) or not names or count < 1
            or count > source_count or count % len(names) or source_count % len(names)):
        raise ValueError("Expected positive balanced source/selection sizes and distinct target pairs")
    if (len(manifest) != source_count
            or len({int(row["image_id"]) for row in manifest}) != source_count
            or len({str(Path(row["source_path"]).resolve()) for row in manifest}) != source_count):
        raise ValueError(f"Source must contain {source_count} distinct original images")
    counts = Counter(row["spec"] for row in manifest)
    if counts != Counter({name: source_count // len(names) for name in names}):
        raise ValueError("Source target-pair counts differ from the balanced original-GHOST pool")
    rng = random.Random(selection_seed)
    selected = []
    for name in names:
        pool = sorted((row for row in manifest if row["spec"] == name), key=lambda row: int(row["image_id"]))
        selected.extend(rng.sample(pool, count // len(names)))
    return sorted(selected, key=lambda row: (row["spec"], int(row["image_id"])))


def load_source(args: argparse.Namespace) -> dict:
    original = args.source_output.resolve()
    reference = args.reference_output.resolve()
    output = args.output_dir.resolve()
    for protected in (original, reference):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("Use a separate output directory outside source/reference experiments")
    manifest_path = original / "selected_images.json"
    signature_path = original / "experiment_signature.json"
    original_config_path = original / "resolved_config.json"
    reference_config_path = reference / "resolved_config.json"
    manifest = json.loads(manifest_path.read_text())
    signature = json.loads(signature_path.read_text())
    original_cfg = json.loads(original_config_path.read_text())
    cfg = json.loads(reference_config_path.read_text())
    if signature.get("protocol_version") != SOURCE_PROTOCOL or not signature.get("sha256"):
        raise ValueError("Expected the recent Stage-3 original-GHOST source manifest/signature")
    expected_reference = {
        "condition_initialization": "clean", "condition_mode": "staged", "condition_stage_count": 21,
        "optimizer": "Adam", "weight_decay": 0.0, "full_mode_updates": 45,
        "saved_optimization_updates": [45], "lambda_victim": 2.0, "lambda_clean": 1.0,
        "clean_threshold": 0.85, "clean_slack": 0.02,
        "ssim_constraint": evaluation.SSIM_CONSTRAINT, "ssim_threshold": 0.8,
        "ssim_resolution": 256, "lambda_ssim": 1.0,
        "diffusion_compute_dtype": "float32", "vae_image_contract": evaluation.feedback.VAE_IMAGE_CONTRACT,
    }
    differences = [key for key, value in expected_reference.items() if cfg.get(key) != value]
    if differences:
        raise ValueError(f"Clean-init reference configuration mismatch: {', '.join(differences)}")
    specs = {value["name"]: evaluation.feedback.exp4.CompositeSpec.from_dict(value)
             for value in cfg["target_specs"]}
    original_specs = {value["name"]: value["objects"] for value in original_cfg["target_specs"]}
    if len(specs) != 5 or original_specs != {name: list(spec.objects) for name, spec in specs.items()}:
        raise ValueError("Source and clean-init reference must use the same five object pairs")
    selected = select_images(manifest, list(specs), selection_seed=args.selection_seed)
    tasks = []
    for row in selected:
        path = Path(row["source_path"]).resolve()
        if evaluation.sha256(path) != row["source_sha256"]:
            raise ValueError(f"Clean source image SHA-256 mismatch: {path}")
        prompt = str(row.get("multi_prescreen_question", "")).strip()
        if (not prompt or row.get("multi_prescreen_decision") is not False
                or row.get("paired_prescreen_no") != 1):
            raise ValueError(f"Missing original Multi-O clean-image prescreen: {path}")
        seeds = [evaluation.feedback.exp4.generation_seed(
            TASK_SEED, row["spec"], int(row["image_id"]), index) for index in four_seeds.INDICES]
        if len(set(seeds)) != 4:
            raise ValueError(f"Candidate seed collision: {path}")
        tasks.append({
            "seed": TASK_SEED, "spec": row["spec"], "image_id": int(row["image_id"]),
            "source_path": str(path), "source_image_sha256": row["source_sha256"],
            # Explicitly absent: there is no optimized Stage-1 embedding input.
            "embedding_path": "", "embedding_sha256": "", "terminal_step": -1,
            "trigger_prompt": prompt, "condition_initialization": "clean",
        })
    for key in ("embedding_archive", "source_output_dir", "source_output", "output_dir"):
        cfg.pop(key, None)
    cfg.update({
        "experiment_name": EXPERIMENT_NAME, "sample_count": len(tasks),
        "learning_rate": args.learning_rate, "seed": TASK_SEED, "seeds": [TASK_SEED],
        "selection_seed": args.selection_seed, "num_images_per_spec": SAMPLE_COUNT // len(specs),
        "source_protocol_version": SOURCE_PROTOCOL,
        "condition_initialization": "clean", "num_generations": 1,
        "final_candidates_per_task": 4, "final_generation_indices": list(four_seeds.INDICES),
        "optimization_generation_index": 0, "checkpoint_steps": [45],
        "saved_optimization_updates": [45], "evaluated_optimization_updates": [45],
        "stage1_archive_role": "none_clean_images_selected_from_stage3_manifest",
        "success_definition": "at_least_one_of_four_victim_yes_no_detector_verification",
    })
    files = (manifest_path, signature_path, original_config_path, reference_config_path,
             Path(__file__), Path(evaluation.__file__), Path(four_seeds.__file__),
             Path(evaluation.feedback.__file__), Path(evaluation.stage1.__file__))
    return {
        "source": original, "source_protocol": SOURCE_PROTOCOL, "source_signature": signature,
        "cfg": cfg, "specs": specs, "tasks": tasks, "selected_images": selected,
        "input_sha256": {str(path): evaluation.sha256(path) for path in files},
        "selection": {
            "sample_count": len(tasks), "source_sample_count": len(manifest),
            "unique_selected_original_images": len(tasks), "duplicate_original_images": 0,
            "selected_by_spec": dict(sorted(Counter(task["spec"] for task in tasks).items())),
            "sampling": "uniform_within_each_of_five_pairs_from_frozen_2000_image_manifest",
            "selection_seed": args.selection_seed, "optimization_base_seed": TASK_SEED,
            "original_ghost_attack_outcomes_used_for_selection": False,
            "stage1_embedding_used_for_selection": False,
            "stage1_archive_role": "none_clean_images_selected_from_stage3_manifest",
            "prompt_policy": "original_ghost_manifest_multi_prescreen_question",
            "reference_config": str(reference_config_path),
        },
    }


def four_seed_args(args: argparse.Namespace) -> argparse.Namespace:
    root = args.output_dir.resolve()
    return argparse.Namespace(
        source_output=root / "optimization", output_dir=root / "four_seeds",
        phase=args.phase if args.phase in ("generate", "evaluate") else "all",
        validate_only=args.validate_only, prepare_only=False, max_new_images=None)


def run(args: argparse.Namespace) -> None:
    source = load_source(args)
    root = args.output_dir.resolve()
    optimizer_args = argparse.Namespace(**vars(args))
    optimizer_args.output_dir = root / "optimization"
    optimizer_args.phase = "evaluate" if args.phase == "evaluate" else "all"
    print(json.dumps({"experiment_name": EXPERIMENT_NAME, "selection": source["selection"],
                      "learning_rate": args.learning_rate, "optimization_updates": 45,
                      "independent_denoising_conditions": 21, "candidates_per_task": 4,
                      "planned_final_images": SAMPLE_COUNT * 4,
                      "four_seed_output": str(root / "four_seeds")}, indent=2), flush=True)
    # Freeze both the engine inputs and the top-level plan before GPU work.
    evaluation.run_experiment(optimizer_args, source, plan=PLAN, prepare_only=True)
    completion_path = root / "optimization" / "completion_summary.json"
    if args.validate_only:
        complete = completion_path.is_file() and json.loads(completion_path.read_text()).get("evaluation_complete") is True
        if complete:
            four_seeds.run_experiment(four_seed_args(args), expected_sample_count=SAMPLE_COUNT,
                                     expected_initialization="clean", protocol_version=PROTOCOL_VERSION)
        else:
            print("Inputs validated; four-seed validation awaits completed optimization/g0 evaluation.", flush=True)
        return
    with evaluation.output_lock(root):
        evaluation.feedback._save_json_atomic(root / "selected_images.json", source["selected_images"])
        evaluation.feedback._save_json_atomic(root / "experiment_plan.json", {
            "protocol_version": PROTOCOL_VERSION, "source_output": str(source["source"]),
            "sample_selection": source["selection"], "learning_rate": args.learning_rate,
            "optimization_updates": 45, "condition_stage_count": 21,
            "condition_initialization": "clean", "candidates_per_task": 4,
            "candidate_policy": "optimization_seed_g0_plus_three_new_seeds_frozen_conditions",
            "planned_final_images": SAMPLE_COUNT * 4,
            "success_definition": source["cfg"]["success_definition"],
            "optimization_output": str(root / "optimization"),
            "candidate_output": str(root / "four_seeds"),
        })
    if args.prepare_only:
        return
    evaluation.run_experiment(optimizer_args, source, plan=PLAN)
    if args.phase == "optimize":
        return
    complete = completion_path.is_file() and json.loads(completion_path.read_text()).get("evaluation_complete") is True
    if not complete:
        raise RuntimeError("Complete all 1,000 optimization tasks and g0 evaluations before four-seed sampling")
    four_seeds.run_experiment(four_seed_args(args), expected_sample_count=SAMPLE_COUNT,
                             expected_initialization="clean", protocol_version=PROTOCOL_VERSION)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
