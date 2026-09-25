import argparse
import shutil
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

def setup_callbacks(model: YOLO):
    """
    Registers custom callbacks to display detailed per-epoch metrics:
    Learning Rate (LR), Precision, Recall, mAP@50, mAP@50-95, and Losses.
    """
    def on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1
        epochs = trainer.epochs

        # Retrieve current learning rate from optimizer
        current_lr = trainer.optimizer.param_groups[0]["lr"]

        # Retrieve validation metrics dictionary
        metrics = getattr(trainer, "metrics", {}) or {}

        precision = metrics.get("metrics/precision(B)", 0.0)
        recall = metrics.get("metrics/recall(B)", 0.0)
        map50 = metrics.get("metrics/mAP50(B)", 0.0)
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)

        # Retrieve validation losses
        val_box_loss = metrics.get("val/box_loss", 0.0)
        val_cls_loss = metrics.get("val/cls_loss", 0.0)
        val_dfl_loss = metrics.get("val/dfl_loss", 0.0)

        print("\n" + "=" * 70)
        print(f"📊 EPOCH [{epoch}/{epochs}] METRICS SUMMARY")
        print("-" * 70)
        print(f"  • Learning Rate (LR) : {current_lr:.6f}")
        print(f"  • Precision (B)      : {precision:.4f} ({precision * 100:.2f}%)")
        print(f"  • Recall (B)         : {recall:.4f} ({recall * 100:.2f}%)")
        print(f"  • mAP@50 (Accuracy)  : {map50:.4f} ({map50 * 100:.2f}%)")
        print(f"  • mAP@50-95          : {map50_95:.4f} ({map50_95 * 100:.2f}%)")
        print(f"  • Validation Losses  : box={val_box_loss:.4f}, cls={val_cls_loss:.4f}, dfl={val_dfl_loss:.4f}")
        print("=" * 70 + "\n")

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def train_yolov8(
    data_yaml: Path | str,
    weights_path: Path | str,
    output_models_dir: Path | str,
    epochs: int = 25,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str | int | None = None,
):
    data_yaml = Path(data_yaml).resolve()
    output_models_dir = Path(output_models_dir).resolve()
    output_models_dir.mkdir(parents=True, exist_ok=True)

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")

    # Determine Device (GPU preference)
    if device is None:
        if torch.cuda.is_available():
            device = 0
            gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[Device Selection] Utilizing GPU: {gpu_name} ({total_mem:.2f} GB VRAM)")
        else:
            device = "cpu"
            print("[Device Selection] CUDA not available, using CPU.")
    else:
        print(f"[Device Selection] Using specified device: {device}")

    # Resolve Weights Path
    weights_path = Path(weights_path).resolve()
    if not weights_path.exists():
        print(f"Base weights not found at {weights_path}, will attempt download via Ultralytics.")
        weights_arg = weights_path.name
    else:
        weights_arg = str(weights_path)

    print(f"\nInitializing YOLOv8 model with weights: {weights_arg}")
    model = YOLO(weights_arg)

    # Attach epoch monitoring callback
    setup_callbacks(model)

    print(f"Starting training for {epochs} epochs on dataset: {data_yaml}")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=str(output_models_dir.parent / "runs" / "train"),
        name="yolov8_training",
        save=True,
        exist_ok=True,
        verbose=True,
        workers=2,
    )

    # Locate best.pt from the training results
    run_save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (output_models_dir.parent / "runs" / "train" / "yolov8_training")
    best_pt_source = run_save_dir / "weights" / "best.pt"

    dest_best_pt = output_models_dir / "best.pt"

    if best_pt_source.exists():
        shutil.copy2(best_pt_source, dest_best_pt)
        print("\n" + "#" * 70)
        print(f"SUCCESS: Best model weights saved to:")
        print(f"  --> {dest_best_pt}")
        print("#" * 70 + "\n")
    else:
        print(f"Warning: Could not find best.pt at {best_pt_source}")

    return dest_best_pt


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    default_data_yaml = base_dir / "data" / "My First Project.v1i.yolov8" / "data.yaml"
    default_models_dir = base_dir / "models"
    
    # Select default pretrained model (prefer yolov8s.pt if downloaded, else yolov8n.pt)
    if (default_models_dir / "yolov8s.pt").exists():
        default_weights = default_models_dir / "yolov8s.pt"
    else:
        default_weights = default_models_dir / "yolov8n.pt"

    parser = argparse.ArgumentParser(description="Train YOLOv8 on custom dataset with GPU acceleration.")
    parser.add_argument("--data", type=str, default=str(default_data_yaml), help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default=str(default_weights), help="Pretrained weights path")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--save-dir", type=str, default=str(default_models_dir), help="Directory to save best.pt")

    args = parser.parse_args()

    train_yolov8(
        data_yaml=args.data,
        weights_path=args.weights,
        output_models_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
    )
