import argparse
from pathlib import Path
from ultralytics import YOLO

def download_model(model_name: str = "yolov8n.pt", target_dir: Path | str | None = None) -> Path:
    """
    Downloads and saves a YOLOv8 model to the specified target directory.

    Args:
        model_name: Name of the YOLOv8 model (e.g., 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', etc.).
        target_dir: Directory where the model file should be saved. Defaults to 'models/'.

    Returns:
        Path: Full path to the saved model file.
    """
    if target_dir is None:
        target_dir = Path(__file__).resolve().parent / "models"
    else:
        target_dir = Path(target_dir).resolve()

    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    # Full destination path for model file
    model_path = target_dir / model_name

    if model_path.exists():
        print(f"Model already exists at: {model_path}")
    else:
        print(f"Downloading {model_name} into: {target_dir} ...")

    # Initializing YOLO with the destination path automatically downloads
    # the weights into that path if not already present.
    model = YOLO(str(model_path))

    print(f"Model successfully loaded and ready at: {model_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save YOLOv8 models into the models directory.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLOv8 model weight filename (default: yolov8n.pt). Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "models"),
        help="Target folder to save the model (default: task/models)"
    )

    args = parser.parse_args()
    download_model(model_name=args.model, target_dir=args.save_dir)
