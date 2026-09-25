"""
Improved YOLOv8 Alphabet Detection Training Script
===================================================
Optimized for small datasets with severe class imbalance.

Key improvements:
  1. Uses YOLOv8s (small) with rock-solid SGD optimizer to prevent gradient divergence
  2. Dataset analysis with accurate path resolution and class distribution inspection
  3. Character-safe augmentations: NO horizontal/vertical flips (fliplr=0, flipud=0)
     to preserve alphabet orientation (B != mirrored B, M != W, etc.)
  4. Moderate rotation (±8°), slight translation, and scale jitter for camera realism
  5. Disabled mixup/copy_paste to prevent confusing blended glyphs
  6. Cosine LR schedule with warm-up for smooth convergence
  7. Automatic best.pt model backup to models/ directory
"""

import argparse
import collections
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


def analyze_dataset(data_yaml: Path):
    """Prints dataset statistics and class distribution for sanity checking."""
    import yaml

    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    root_path = cfg.get("path", "")
    if root_path:
        base_dir = Path(root_path)
    else:
        base_dir = data_yaml.parent

    class_names = cfg.get("names", [])
    nc = cfg.get("nc", len(class_names))

    print("\n" + "=" * 60)
    print("📊 DATASET ANALYSIS")
    print("=" * 60)
    print(f"  Classes (nc): {nc}")
    print(f"  Names: {class_names}")

    split_map = {
        "train": cfg.get("train", "train/images"),
        "val": cfg.get("val", cfg.get("valid", "valid/images")),
        "test": cfg.get("test", "test/images"),
    }

    for split, split_rel in split_map.items():
        if not split_rel:
            continue
        p = Path(split_rel)
        if p.is_absolute():
            img_dir = p
        else:
            # Check relative to base_dir
            candidate1 = (base_dir / p).resolve()
            candidate2 = (data_yaml.parent / p).resolve()
            img_dir = candidate1 if candidate1.exists() else candidate2

        lbl_dir = Path(str(img_dir).replace("images", "labels"))

        img_count = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0

        counts = collections.Counter()
        if lbl_dir.exists():
            for lf in lbl_dir.glob("*.txt"):
                for line in lf.read_text().strip().splitlines():
                    parts = line.split()
                    if parts:
                        try:
                            counts[int(parts[0])] += 1
                        except ValueError:
                            pass

        total_ann = sum(counts.values())
        print(f"\n  [{split.upper()}] {img_count} images, {total_ann} annotations")

        if counts and split == "train":
            print("  Class distribution:")
            max_c = max(counts.values()) if counts else 1
            for cid in range(nc):
                name = class_names[cid] if cid < len(class_names) else f"ID_{cid}"
                c = counts.get(cid, 0)
                bar = "█" * max(1, int(c / max_c * 30)) if c > 0 else "░"
                flag = " ⚠️ LOW" if 0 < c < 5 else (" ❌ MISSING" if c == 0 else "")
                print(f"    [{cid:2d}] {name:2s} : {c:4d} {bar}{flag}")

    print("=" * 60 + "\n")


def setup_callbacks(model: YOLO):
    """Registers custom callbacks to display detailed per-epoch metrics."""

    best_map50 = [0.0]

    def on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1
        epochs = trainer.epochs

        current_lr = trainer.optimizer.param_groups[0]["lr"]
        metrics = getattr(trainer, "metrics", {}) or {}

        precision = metrics.get("metrics/precision(B)", 0.0)
        recall = metrics.get("metrics/recall(B)", 0.0)
        map50 = metrics.get("metrics/mAP50(B)", 0.0)
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)

        val_box_loss = metrics.get("val/box_loss", 0.0)
        val_cls_loss = metrics.get("val/cls_loss", 0.0)
        val_dfl_loss = metrics.get("val/dfl_loss", 0.0)

        is_best = map50 > best_map50[0]
        if is_best:
            best_map50[0] = map50

        print("\n" + "=" * 70)
        print(f"📊 EPOCH [{epoch}/{epochs}] METRICS SUMMARY {'🏆 NEW BEST' if is_best else ''}")
        print("-" * 70)
        print(f"  • Learning Rate (LR)     : {current_lr:.6f}")
        print(f"  • Precision (B)          : {precision:.4f} ({precision * 100:.2f}%)")
        print(f"  • Recall (B)             : {recall:.4f} ({recall * 100:.2f}%)")
        print(f"  • mAP@50 (Accuracy)      : {map50:.4f} ({map50 * 100:.2f}%)")
        print(f"  • mAP@50-95              : {map50_95:.4f} ({map50_95 * 100:.2f}%)")
        print(f"  • Validation Losses      : box={val_box_loss:.4f}, cls={val_cls_loss:.4f}, dfl={val_dfl_loss:.4f}")
        print(f"  • Best mAP@50 so far     : {best_map50[0]:.4f} ({best_map50[0] * 100:.2f}%)")
        print("=" * 70 + "\n")

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def train_yolov8(
    data_yaml: Path | str,
    weights_path: Path | str,
    output_models_dir: Path | str,
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 8,
    device: str | int | None = None,
    patience: int = 25,
    resume: bool = False,
):
    data_yaml = Path(data_yaml).resolve()
    output_models_dir = Path(output_models_dir).resolve()
    output_models_dir.mkdir(parents=True, exist_ok=True)

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")

    # Show dataset analysis
    analyze_dataset(data_yaml)

    # Determine Device (GPU preference)
    if device is None:
        if torch.cuda.is_available():
            device = 0
            gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[Device] Utilizing GPU: {gpu_name} ({total_mem:.2f} GB VRAM)")
        else:
            device = "cpu"
            print("[Device] CUDA not available, using CPU.")
    else:
        print(f"[Device] Using specified device: {device}")

    # Resolve Weights Path
    weights_path = Path(weights_path).resolve()
    if not weights_path.exists():
        print(f"Weights not found at {weights_path}, downloading via Ultralytics...")
        weights_arg = weights_path.name
    else:
        weights_arg = str(weights_path)

    print(f"\nInitializing YOLOv8 model with weights: {weights_arg}")
    model = YOLO(weights_arg)

    # Attach epoch monitoring callback
    setup_callbacks(model)

    # ───────────────────────────────────────────────────────────────
    #  TRAINING HYPERPARAMETERS — Stabilized for Alphabet Detection
    # ───────────────────────────────────────────────────────────────
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=str(output_models_dir.parent / "runs" / "train"),
        name="yolov8_training_v2",
        save=True,
        exist_ok=True,
        verbose=True,
        workers=2,
        patience=patience,
        resume=resume,

        # ── Optimizer & Learning Rate Schedule ──
        # SGD with Cosine annealing is proven stable against loss spikes
        optimizer="SGD",
        lr0=0.01,           # Initial LR for SGD
        lrf=0.01,           # Cosine decay down to 0.0001
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_bias_lr=0.01,
        cos_lr=True,

        # ── Character-Safe Augmentations ──
        # DO NOT flip letters: 'B', 'D', 'E', 'F' become mirrored; 'M' becomes 'W'
        fliplr=0.0,         # 0% horizontal flip (CRITICAL: preserve letter orientation)
        flipud=0.0,         # 0% vertical flip (CRITICAL: preserve letter orientation)
        mixup=0.0,          # 0% mixup (do not blend separate letters together)
        copy_paste=0.0,     # 0% copy-paste (do not paste foreign glyphs)

        # Realistic camera jitter for holding flashcards in hand
        degrees=8.0,        # ±8° slight tilt
        translate=0.10,     # ±10% translation
        scale=0.25,         # ±25% zoom (preserves character strokes)
        shear=2.0,          # ±2° slight perspective angle
        perspective=0.0,
        hsv_h=0.015,        # Subtle hue jitter
        hsv_s=0.4,          # Saturation change for varying lighting
        hsv_v=0.4,          # Value/brightness change
        erasing=0.1,        # 10% subtle occlusion
        mosaic=0.5,         # Moderate mosaic
        close_mosaic=10,    # Disable mosaic 10 epochs before end for clean fine-tuning

        # ── Balanced Loss Gains ──
        box=7.5,            # Standard box loss gain
        cls=0.5,            # Standard cls gain (prevents loss explosion)
        dfl=1.5,            # Distribution focal loss gain

        # ── NMS ──
        iou=0.6,
    )

    print(f"\n🚀 Starting stabilized training for {epochs} epochs (patience={patience})")
    print(f"   Dataset: {data_yaml}")
    print(f"   Optimizer: SGD (lr0=0.01, lrf=0.01, momentum=0.937, cos_lr=True)")
    print(f"   Augmentations: fliplr=0, flipud=0, mixup=0, copy_paste=0, mosaic=0.5")
    print(f"   Batch: {batch_size}, ImgSz: {imgsz}\n")

    results = model.train(**train_kwargs)

    # Locate best.pt from the training results
    run_save_dir = (
        Path(results.save_dir)
        if hasattr(results, "save_dir")
        else (output_models_dir.parent / "runs" / "train" / "yolov8_training_v2")
    )
    best_pt_source = run_save_dir / "weights" / "best.pt"
    dest_best_pt = output_models_dir / "best.pt"

    if best_pt_source.exists():
        shutil.copy2(best_pt_source, dest_best_pt)
        print("\n" + "#" * 70)
        print(f"✅ SUCCESS: Best model weights saved to:")
        print(f"   --> {dest_best_pt}")
        print("#" * 70 + "\n")
    else:
        print(f"⚠️ Warning: Could not find best.pt at {best_pt_source}")

    return dest_best_pt


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    default_data_yaml = base_dir / "data" / "My First Project.v1i.yolov8" / "data.yaml"
    default_models_dir = base_dir / "models"

    if (default_models_dir / "yolov8s.pt").exists():
        default_weights = default_models_dir / "yolov8s.pt"
    elif (base_dir / "yolov8s.pt").exists():
        default_weights = base_dir / "yolov8s.pt"
    elif (default_models_dir / "yolov8n.pt").exists():
        default_weights = default_models_dir / "yolov8n.pt"
    else:
        default_weights = "yolov8s.pt"

    parser = argparse.ArgumentParser(
        description="Improved YOLOv8 Alphabet Detection Training (optimized for small datasets)"
    )
    parser.add_argument("--data", type=str, default=str(default_data_yaml), help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default=str(default_weights), help="Pretrained weights path (default: yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8, optimal for 6GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience (default: 25)")
    parser.add_argument("--save-dir", type=str, default=str(default_models_dir), help="Directory to save best.pt")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")

    args = parser.parse_args()

    train_yolov8(
        data_yaml=args.data,
        weights_path=args.weights,
        output_models_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        resume=args.resume,
    )
