#!/usr/bin/env python3
"""Generate local Stage-3 configs/launchers; never load models or start jobs.

Run from any directory. Paths in the input are absolute or relative to the
checkout root. Existing run names cannot be reassigned to another model/config.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = (
    "victim_model_path", "clip_checkpoint_path", "diffusion_model_path",
    "owl_model_path", "data_path", "coco_captions_path", "projector_path",
)


def read(path: Path):
    return json.loads(path.read_text())


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def resolve(value: str) -> Path:
    p = Path(os.path.expandvars(value)).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_settings(path: Path) -> dict:
    settings = read(path)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}", settings.get("run_name", "")):
        raise ValueError("run_name must be a short name without directory separators")
    if settings.get("victim_family") != "qwen2_5_vl":
        raise ValueError("Current runners implement Qwen2.5-VL only. Adapt the model interfaces first; see docs/VICTIM_ADAPTATION.md")
    for key in PATH_KEYS:
        if not isinstance(settings.get(key), str) or not settings[key].strip():
            raise ValueError(f"Missing path: {key}")
        settings[key] = str(resolve(settings[key]))
    for key in ("target_dim", "num_tokens"):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer matching the new victim/projector")
    return settings


def check_assets(settings: dict) -> None:
    for key in PATH_KEYS:
        if not Path(settings[key]).exists():
            raise FileNotFoundError(f"{key}: {settings[key]}")
    for key in ("victim_model_path", "owl_model_path"):
        if not (Path(settings[key]) / "config.json").is_file():
            raise FileNotFoundError(f"Missing model config under {settings[key]}")
    if not (Path(settings["diffusion_model_path"]) / "model_index.json").is_file():
        raise FileNotFoundError("diffusion_model_path must contain the complete unCLIP pipeline")
    victim_config = read(Path(settings["victim_model_path"]) / "config.json")
    if victim_config.get("model_type") != "qwen2_5_vl":
        raise ValueError("Victim config is not qwen2_5_vl; adapt interfaces before running")
    if victim_config.get("hidden_size") != settings["target_dim"]:
        raise ValueError("target_dim does not match victim hidden_size")
    clip = Path(settings["clip_checkpoint_path"])
    expected = read(ROOT / "migration_inputs/external_assets.json")["clip_checkpoint"]["sha256"]
    if not clip.is_file() or digest(clip) != expected:
        raise ValueError("CLIP checkpoint differs from the one that produced the included rankings; recompute rankings before reuse")
    coco = Path(settings["data_path"])
    for name in ("instances_train2017.json", "captions_train2017.json"):
        if not (coco / "annotations" / name).is_file():
            raise FileNotFoundError(coco / "annotations" / name)
    image_dir = coco / "images/train2017"
    if not image_dir.is_dir():
        image_dir = coco / "train2017"
    if not image_dir.is_dir():
        raise FileNotFoundError("Expected COCO/images/train2017 or COCO/train2017")
    ids = set()
    for path in (ROOT / "migration_inputs/ranking_cache").glob("clip_ranked_*.csv"):
        with path.open(newline="") as handle:
            ids.update(int(row["image_id"]) for row in csv.DictReader(handle))
    missing = [image_dir / f"{i:012d}.jpg" for i in sorted(ids)
               if not (image_dir / f"{i:012d}.jpg").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} CLIP candidate images; first: {missing[0]}")
    print(f"Assets found; CLIP SHA-256 matches; {len(ids)} ranked candidate images found.")
    print("Projector tensor dimensions and differentiable victim preprocessing still require a runtime check.")


def launcher(path: Path, arguments: list[object]) -> None:
    command = " ".join(shlex.quote(str(v)) for v in arguments)
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd -- {shlex.quote(str(ROOT))}\n"
        f"stage3_default_python={shlex.quote(str(ROOT / '.venv-stage1/bin/python'))}\n"
        'stage3_python="${STGHOST_PYTHON:-$stage3_default_python}"\n'
        f'exec "$stage3_python" -u {command} "$@"\n'
    )
    path.chmod(0o755)


def configure(settings: dict) -> Path:
    run = ROOT / "runtime" / settings["run_name"]
    identity = run / "server_settings.json"
    if run.exists() and (not identity.exists() or read(identity) != settings):
        raise ValueError("This run_name already exists with different settings; choose a new run_name")
    # A moved checkout changes absolute source paths/signatures. Do not rewrite
    # a profile containing results; configure a fresh run name on the new host.
    marker = run / "checkout_path.json"
    if marker.exists() and read(marker) != {"root": str(ROOT)}:
        raise ValueError("This profile was configured in another checkout; choose a fresh run_name")
    run.mkdir(parents=True, exist_ok=True)
    save(identity, settings)
    save(marker, {"root": str(ROOT)})
    paths = {key: settings[key] for key in PATH_KEYS}
    configs = run / "configs"
    base = read(ROOT / "st-ghost/configs/stage1_exp1_main.json")
    base.update(paths, target_dim=settings["target_dim"], num_tokens=settings["num_tokens"],
                output_dir=str(run / "unused_stage1_output"))
    save(configs / "base.json", base)
    original = read(ROOT / "st-ghost/configs/stage3_original_ghost_comparison.json")
    original.update(base_config=str(configs / "base.json"), output_dir=str(run / "original_ghost"),
                    ranking_cache_dir=str(run / "ranking_cache"))
    save(configs / "original_ghost.json", original)
    cache = read(ROOT / "migration_inputs/ranking_cache/resolved_config.json")
    cache.update(paths)
    save(run / "ranking_cache/resolved_config.json", cache)
    for source in (ROOT / "migration_inputs/ranking_cache").glob("clip_ranked_*.csv"):
        shutil.copy2(source, run / "ranking_cache" / source.name)
    reference = read(ROOT / "migration_inputs/reference/stghost_reference.json")
    for key in ("source_output_dir", "source_output", "embedding_archive", "output_dir"):
        reference.pop(key, None)
    reference.update(paths, target_dim=settings["target_dim"], num_tokens=settings["num_tokens"])
    save(run / "stghost_reference/resolved_config.json", reference)
    train = read(ROOT / "st-ghost/configs/train_projector_qwen.json")
    train.update(qwen_model_path=settings["victim_model_path"],
                 clip_checkpoint_path=settings["clip_checkpoint_path"],
                 coco_root=settings["data_path"],
                 output_dir=str(Path(settings["projector_path"]).parent),
                 target_dim=settings["target_dim"], expected_num_tokens=settings["num_tokens"])
    save(configs / "projector.json", train)
    launcher(run / "run_original_ghost.sh", [ROOT / "st-ghost/run_stage3_original_ghost_comparison.py",
                                             "--config", configs / "original_ghost.json"])
    launcher(run / "run_st_ghost.sh", [ROOT / "st-ghost/run_stage3_st_ghost.py",
        "--source-output", run / "original_ghost", "--reference-output", run / "stghost_reference",
        "--output-dir", run / "stghost"])
    launcher(run / "train_projector.sh", [ROOT / "st-ghost/train_projector_qwen.py",
                                          "--config", configs / "projector.json"])
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=ROOT / "server_paths.local.json")
    parser.add_argument("--check-assets", action="store_true",
                        help="Check local assets and CLIP checksum before generating configs")
    args = parser.parse_args()
    settings = load_settings(args.paths)
    if args.check_assets:
        check_assets(settings)
    run = configure(settings)
    print(f"Prepared: {run}")
    print("No model was loaded and no experiment was started.")
    print(f"Next: bash {shlex.quote(str(run / 'run_original_ghost.sh'))} --validate-only")


if __name__ == "__main__":
    main()
