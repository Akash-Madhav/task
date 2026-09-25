import argparse
import json
from pathlib import Path
import torch
from ultralytics import YOLO

def evaluate_best_model(
    weights_path: Path | str = "models/best.pt",
    data_yaml: Path | str = "data/My First Project.v1i.yolov8/data.yaml",
    split: str = "val",
    imgsz: int = 640,
    batch_size: int = 16,
    device: str | int | None = None,
):
    weights_path = Path(weights_path).resolve()
    data_yaml = Path(data_yaml).resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"Model file not found at: {weights_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data config not found at: {data_yaml}")

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 75)
    print(f"🔍 EVALUATING YOLOv8 MODEL: {weights_path.name}")
    print(f"   Dataset Split : {split}")
    print(f"   Data Config   : {data_yaml}")
    print(f"   Device        : {'GPU (CUDA:0)' if device == 0 else device}")
    print("=" * 75 + "\n")

    # Load trained model
    model = YOLO(str(weights_path))

    # Run validation / evaluation
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project="runs/evaluation",
        name=f"eval_{split}",
        save_json=False,
        plots=True,
        verbose=True,
    )

    # Overall metrics
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    fitness = float(metrics.fitness)

    # Speed metrics
    speed = metrics.speed  # {'preprocess': ms, 'inference': ms, 'loss': ms, 'postprocess': ms}
    total_time_per_image = sum(speed.values())
    fps = 1000.0 / total_time_per_image if total_time_per_image > 0 else 0.0

    print("\n" + "=" * 75)
    print(f"📈 OVERALL PERFORMANCE METRICS ({split.upper()} SET)")
    print("-" * 75)
    print(f"  • mAP@50 (Accuracy / Detection rate) : {map50:.4f} ({map50 * 100:.2f}%)")
    print(f"  • mAP@50-95 (Strict Accuracy)        : {map50_95:.4f} ({map50_95 * 100:.2f}%)")
    print(f"  • Precision                          : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"  • Recall                             : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"  • Model Fitness Score                : {fitness:.4f}")
    print("-" * 75)
    print(f"  ⚡ Inference Speed: {speed.get('inference', 0.0):.2f} ms/image  (~{fps:.1f} FPS on GPU)")
    print("=" * 75)

    # Per-Class Breakdown
    # metrics.box.ap50 and metrics.box.ap are arrays corresponding to class indices present in the val set
    per_class_results = {}
    classes_present = metrics.box.ap_class_index if hasattr(metrics.box, "ap_class_index") else range(len(metrics.box.maps))

    print("\n📋 PER-CLASS ACCURACY BREAKDOWN:")
    print("-" * 75)
    print(f"{'Class':<10} | {'Precision':<12} | {'Recall':<12} | {'mAP@50':<12} | {'mAP@50-95'}")
    print("-" * 75)

    for i, cls_idx in enumerate(classes_present):
        cls_name = metrics.names[cls_idx]
        p_val = float(metrics.box.p[i]) if i < len(metrics.box.p) else 0.0
        r_val = float(metrics.box.r[i]) if i < len(metrics.box.r) else 0.0
        map50_val = float(metrics.box.all_ap[i, 0]) if hasattr(metrics.box, "all_ap") else float(metrics.box.maps[cls_idx])
        map50_95_val = float(metrics.box.maps[cls_idx])

        per_class_results[cls_name] = {
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "map50": round(map50_val, 4),
            "map50_95": round(map50_95_val, 4),
        }

        print(f"{cls_name:<10} | {p_val:<12.4f} | {r_val:<12.4f} | {map50_val:<12.4f} | {map50_95_val:.4f}")

    print("-" * 75)

    save_dir = Path(metrics.save_dir)
    print(f"\n📊 Visual diagnostic plots (Confusion Matrix, PR Curve, F1 Curve) saved to:")
    print(f"   --> {save_dir}")

    # Export report to JSON
    report = {
        "model": str(weights_path),
        "split": split,
        "overall_metrics": {
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "fitness": round(fitness, 4),
        },
        "speed_ms": speed,
        "fps": round(fps, 1),
        "per_class_metrics": per_class_results,
        "plots_directory": str(save_dir),
    }

    report_file = save_dir / "evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Evaluate best.pt and view accuracy metrics.")
    parser.add_argument("--weights", type=str, default=str(base_dir / "models" / "best.pt"), help="Path to best.pt")
    parser.add_argument("--data", type=str, default=str(base_dir / "data" / "My First Project.v1i.yolov8" / "data.yaml"), help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"], help="Dataset split to evaluate")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")

    args = parser.parse_args()

    evaluate_best_model(
        weights_path=args.weights,
        data_yaml=args.data,
        split=args.split,
        batch_size=args.batch,
        imgsz=args.imgsz,
    )
