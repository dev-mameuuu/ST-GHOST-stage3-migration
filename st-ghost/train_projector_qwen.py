"""Train a CLIP-to-Qwen2.5-VL visual-token projector from local assets.

This follows GHOST-master/train_projector.py: a TokenMLP maps one OpenCLIP
ViT-H/14 global image vector to the visual tokens produced by Qwen2.5-VL.
Only the projector is optimized; Qwen and CLIP remain frozen.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
GHOST = ROOT / "GHOST-master"
if str(GHOST) not in sys.path:
    sys.path.insert(0, str(GHOST))
from projector.projector import TokenMLP  # noqa: E402


class LocalImageDataset(Dataset):
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, directory: Path, image_size: int):
        self.paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in self.EXTENSIONS)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {directory}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as source:
            image = source.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        return image, str(path)


def collate_images(batch):
    images, paths = zip(*batch)
    return list(images), list(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qwen_model_path", type=Path)
    parser.add_argument("--clip_checkpoint_path", type=Path)
    parser.add_argument("--coco_root", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--resume_checkpoint", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    cfg = json.loads(args.config.read_text())
    for key in ("qwen_model_path", "clip_checkpoint_path", "coco_root", "output_dir",
                "epochs", "batch_size", "resume_checkpoint"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = str(value) if isinstance(value, Path) else value
    for key in ("qwen_model_path", "clip_checkpoint_path", "coco_root", "output_dir"):
        if not cfg.get(key):
            raise ValueError(f"Missing required config value: {key}")
    for key in ("qwen_model_path", "clip_checkpoint_path", "coco_root"):
        if not Path(cfg[key]).exists():
            raise FileNotFoundError(f"Configured local path does not exist: {cfg[key]}")
    if not 0 < float(cfg["train_fraction"]) <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def clip_features(images: Sequence[Image.Image], model, preprocess, device: torch.device) -> torch.Tensor:
    pixels = torch.stack([preprocess(image) for image in images]).to(device)
    return model.encode_image(pixels)


def qwen_features(images: Sequence[Image.Image], model, processor, device: torch.device) -> torch.Tensor:
    batch = processor.image_processor(images=list(images), return_tensors="pt")
    pixels = batch["pixel_values"].to(device=device, dtype=model.visual.dtype)
    grid = batch["image_grid_thw"].to(device)
    tokens = model.visual(pixels, grid_thw=grid)
    batch_size = len(images)
    if tokens.shape[0] % batch_size:
        raise RuntimeError(f"Cannot reshape Qwen tokens {tuple(tokens.shape)} for batch {batch_size}")
    return tokens.view(batch_size, tokens.shape[0] // batch_size, -1)


def load_models(cfg: dict, device: torch.device):
    logging.info("Loading Qwen from %s", cfg["qwen_model_path"])
    processor = AutoProcessor.from_pretrained(cfg["qwen_model_path"], local_files_only=True)
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["qwen_model_path"], local_files_only=True, torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval().to(device)
    qwen.requires_grad_(False)
    clip_path = Path(cfg["clip_checkpoint_path"])
    if clip_path.is_dir():
        candidates = [clip_path / name for name in
                      ("open_clip_model.safetensors", "open_clip_pytorch_model.bin",
                       "pytorch_model.bin", "model.safetensors")]
        clip_path = next((path for path in candidates if path.exists()), None)
        if clip_path is None:
            raise FileNotFoundError(f"No OpenCLIP checkpoint found in {cfg['clip_checkpoint_path']}")
    logging.info("Loading OpenCLIP checkpoint from %s", clip_path)
    clip, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained=str(clip_path))
    clip = clip.eval().to(device); clip.requires_grad_(False)
    return qwen, processor, clip, preprocess


def build_loaders(cfg: dict):
    root = Path(cfg["coco_root"])
    train = LocalImageDataset(root / "train2017", int(cfg["image_size"]))
    val = LocalImageDataset(root / "val2017", int(cfg["image_size"]))
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    permutation = torch.randperm(len(train), generator=generator).tolist()
    count = max(1, round(len(train) * float(cfg["train_fraction"])))
    selected = permutation[:count]
    train_subset = Subset(train, selected)
    common = dict(num_workers=int(cfg["num_workers"]), collate_fn=collate_images,
                  pin_memory=True, persistent_workers=int(cfg["num_workers"]) > 0)
    train_loader = DataLoader(train_subset, batch_size=int(cfg["batch_size"]), shuffle=True,
                              generator=generator, **common)
    val_loader = DataLoader(val, batch_size=int(cfg["val_batch_size"]), shuffle=False, **common)
    return train_loader, val_loader, selected, len(train), len(val)


@torch.no_grad()
def validate(projector, loader, qwen, processor, clip, preprocess, device, cfg) -> float:
    projector.eval(); total = 0.0
    for images, _ in tqdm(loader, desc="Validation", leave=False):
        source = clip_features(images, clip, preprocess, device)
        target = qwen_features(images, qwen, processor, device)
        prediction = projector(source).float()
        if prediction.shape != target.shape:
            raise RuntimeError(f"Projector/Qwen shape mismatch: {prediction.shape} vs {target.shape}")
        total += F.mse_loss(prediction, target.float()).item()
    return total / len(loader)


def save_checkpoint(path: Path, projector, optimizer, scheduler, epoch: int,
                    global_step: int, best_val: float, cfg: dict) -> None:
    torch.save({
        "projector_state_dict": projector.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch, "global_step": global_step, "best_val_loss": best_val,
        "projector_config": {key: cfg[key] for key in
                             ("clip_dim", "context_dim", "hidden_dim", "expected_num_tokens", "target_dim")},
    }, path)


def plot_history(history: list[dict], output: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], marker="o", label="validation")
    plt.xlabel("Epoch"); plt.ylabel("MSE loss"); plt.title("Qwen projector training")
    plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(output / "loss_curve.png", dpi=180); plt.close()


def main() -> None:
    args = parse_args(); cfg = load_config(args)
    output = Path(cfg["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "train.log")])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen projector training")
    device = torch.device("cuda"); seed_everything(int(cfg["seed"]))
    train_loader, val_loader, indices, full_train, full_val = build_loaders(cfg)
    (output / "train_indices.json").write_text(json.dumps(indices))
    (output / "resolved_config.json").write_text(json.dumps(cfg, indent=2))
    logging.info("COCO train subset: %d/%d; validation: %d", len(indices), full_train, full_val)
    qwen, processor, clip, preprocess = load_models(cfg, device)
    projector = TokenMLP(
        clip_dim=int(cfg["clip_dim"]), context_dim=int(cfg["context_dim"]),
        num_tokens=int(cfg["expected_num_tokens"]), hidden_dim=int(cfg["hidden_dim"]),
        target_dim=int(cfg["target_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"])) \
        if cfg.get("scheduler") == "cosine" else None
    start_epoch, global_step, best_val = 1, 0, float("inf")
    if cfg.get("resume_checkpoint"):
        state = torch.load(cfg["resume_checkpoint"], map_location="cpu")
        projector.load_state_dict(state["projector_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler and state.get("scheduler_state_dict"):
            scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state["epoch"]) + 1; global_step = int(state["global_step"])
        best_val = float(state["best_val_loss"])
    if args.validate_only:
        print(f"validation_loss={validate(projector, val_loader, qwen, processor, clip, preprocess, device, cfg):.8f}")
        return
    history = []
    metrics_path = output / "metrics.csv"
    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        projector.train(); running = 0.0; started = time.time()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for step, (images, _) in enumerate(progress, 1):
            with torch.no_grad():
                source = clip_features(images, clip, preprocess, device)
                target = qwen_features(images, qwen, processor, device)
            prediction = projector(source).float()
            if prediction.shape != target.shape:
                raise RuntimeError(f"Projector/Qwen shape mismatch: {prediction.shape} vs {target.shape}")
            loss = F.mse_loss(prediction, target.float())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch}, step={step}: {loss}")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad_norm = float(clip_grad_norm_(projector.parameters(), float(cfg["grad_clip_norm"])))
            optimizer.step(); global_step += 1; running += loss.item()
            if step % int(cfg["log_every"]) == 0:
                progress.set_postfix(loss=f"{loss.item():.5f}", avg=f"{running / step:.5f}", grad=f"{grad_norm:.2f}")
        train_loss = running / len(train_loader)
        val_loss = validate(projector, val_loader, qwen, processor, clip, preprocess, device, cfg)
        if scheduler: scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.time() - started}
        history.append(row)
        with metrics_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys()); writer.writeheader(); writer.writerows(history)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(output / "best.pt", projector, optimizer, scheduler, epoch, global_step, best_val, cfg)
            torch.save(projector.state_dict(), output / "best_state_dict.pt")
        save_checkpoint(output / "last.pt", projector, optimizer, scheduler, epoch, global_step, best_val, cfg)
        if cfg.get("save_every_epoch"):
            save_checkpoint(output / f"epoch_{epoch:02d}.pt", projector, optimizer, scheduler,
                            epoch, global_step, best_val, cfg)
        plot_history(history, output)
        logging.info("epoch=%d train=%.8f val=%.8f best=%.8f time=%.1fs",
                     epoch, train_loss, val_loss, best_val, row["seconds"])


if __name__ == "__main__":
    main()
