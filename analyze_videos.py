import argparse
import json
from collections import defaultdict
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

def format_timestamp(seconds: float) -> str:
    """Convert seconds into MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 100)
    return f"{mins:02d}:{secs:02d}.{millis:02d}"


def analyze_video(
    video_path: Path,
    model: YOLO,
    output_dir: Path,
    conf_threshold: float = 0.35,
    frame_stride: int = 2,
    save_annotated_frames: bool = True,
    save_annotated_video: bool = False,
):
    video_path = Path(video_path).resolve()
    video_name = video_path.stem
    video_out_dir = output_dir / video_name
    frames_dir = video_out_dir / "detections"
    if save_annotated_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print(f"\n{'='*75}")
    print(f"🎥 ANALYZING VIDEO: {video_path.name}")
    print(f"   Resolution: {width}x{height} | FPS: {fps:.1f} | Total Frames: {total_frames} | Duration: {duration_sec:.2f}s")
    print(f"   Frame Stride: Every {frame_stride} frame(s) | Conf Threshold: {conf_threshold}")
    print(f"{'='*75}")

    video_writer = None
    if save_annotated_video:
        video_out_dir.mkdir(parents=True, exist_ok=True)
        annotated_video_path = video_out_dir / f"{video_name}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps / frame_stride, (width, height))

    # Stats tracking per class
    # class_name -> list of detections: {'frame_idx', 'time_sec', 'confidence', 'bbox'}
    detections_by_class = defaultdict(list)
    frames_with_detections = 0
    analyzed_frames_count = 0
    saved_frames_count = 0

    # Sequential timeline of dominant detections
    timeline = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            analyzed_frames_count += 1
            time_sec = frame_idx / fps

            # Run inference
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                device=0 if torch.cuda.is_available() else "cpu",
                verbose=False,
            )[0]

            boxes = results.boxes
            current_frame_has_detection = False
            annotated_frame = None

            if len(boxes) > 0:
                current_frame_has_detection = True
                frames_with_detections += 1

                frame_classes = []
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()

                    detections_by_class[cls_name].append({
                        "frame_idx": frame_idx,
                        "time_sec": round(time_sec, 2),
                        "conf": round(conf, 4),
                        "bbox": [round(x, 1) for x in xyxy],
                    })
                    frame_classes.append((cls_name, conf))

                # Track timeline event
                best_detection = max(frame_classes, key=lambda x: x[1])
                timeline.append({
                    "frame_idx": frame_idx,
                    "time_sec": round(time_sec, 2),
                    "timestamp": format_timestamp(time_sec),
                    "class": best_detection[0],
                    "confidence": round(best_detection[1], 4),
                })

                # Save sample annotated frame (e.g. up to 100 frames to avoid disk bloat)
                if save_annotated_frames and saved_frames_count < 100:
                    annotated_frame = results.plot()
                    saved_frame_filename = frames_dir / f"frame_{frame_idx:05d}_{best_detection[0]}_{best_detection[1]:.2f}.jpg"
                    cv2.imwrite(str(saved_frame_filename), annotated_frame)
                    saved_frames_count += 1

            if video_writer is not None:
                if annotated_frame is None:
                    annotated_frame = results.plot()
                video_writer.write(annotated_frame)

        frame_idx += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()

    # Consolidate class statistics
    class_summary = {}
    for cls_name, items in detections_by_class.items():
        confs = [x["conf"] for x in items]
        timestamps = [x["time_sec"] for x in items]
        first_seen = min(timestamps)
        last_seen = max(timestamps)
        count = len(items)
        coverage_pct = (count / analyzed_frames_count) * 100 if analyzed_frames_count > 0 else 0

        class_summary[cls_name] = {
            "detection_count": count,
            "frame_presence_pct": round(coverage_pct, 2),
            "avg_confidence": round(sum(confs) / count, 4),
            "max_confidence": round(max(confs), 4),
            "first_seen_time": format_timestamp(first_seen),
            "last_seen_time": format_timestamp(last_seen),
        }

    # Sort classes by frequency of occurrence
    sorted_summary = dict(sorted(class_summary.items(), key=lambda item: item[1]["detection_count"], reverse=True))

    # Consolidate contiguous timeline segments
    consolidated_segments = []
    if timeline:
        current_segment = {
            "class": timeline[0]["class"],
            "start_time": timeline[0]["time_sec"],
            "end_time": timeline[0]["time_sec"],
            "max_conf": timeline[0]["confidence"],
            "frames": 1,
        }
        for item in timeline[1:]:
            if item["class"] == current_segment["class"] and (item["time_sec"] - current_segment["end_time"]) <= 1.0:
                current_segment["end_time"] = item["time_sec"]
                current_segment["max_conf"] = max(current_segment["max_conf"], item["confidence"])
                current_segment["frames"] += 1
            else:
                if current_segment["frames"] >= 2:  # Filter out single-frame blips
                    consolidated_segments.append(current_segment)
                current_segment = {
                    "class": item["class"],
                    "start_time": item["time_sec"],
                    "end_time": item["time_sec"],
                    "max_conf": item["confidence"],
                    "frames": 1,
                }
        if current_segment["frames"] >= 2:
            consolidated_segments.append(current_segment)

    # Print results
    print(f"\n📊 SUMMARY REPORT FOR: {video_path.name}")
    print(f"Total analyzed frames: {analyzed_frames_count} | Frames with detections: {frames_with_detections}")
    print("-" * 75)
    print(f"{'Class':<8} | {'Detections':<12} | {'Presence %':<12} | {'Avg Conf':<10} | {'Max Conf':<10} | {'Time Range'}")
    print("-" * 75)
    if not sorted_summary:
        print("  No classes detected above confidence threshold.")
    else:
        for cls_name, stats in sorted_summary.items():
            print(
                f"{cls_name:<8} | {stats['detection_count']:<12} | {stats['frame_presence_pct']:<11}% | "
                f"{stats['avg_confidence']:<10.2f} | {stats['max_confidence']:<10.2f} | "
                f"{stats['first_seen_time']} -> {stats['last_seen_time']}"
            )

    print("\n⏳ NOTABLE ACTIVITY TIMELINE (Active intervals):")
    if not consolidated_segments:
        print("  No sustained active intervals detected.")
    else:
        for seg in consolidated_segments:
            start_str = format_timestamp(seg["start_time"])
            end_str = format_timestamp(seg["end_time"])
            print(f"  • [{start_str} - {end_str}] Class '{seg['class']}' (Max Conf: {seg['max_conf']:.2f}, {seg['frames']} frames)")

    result_data = {
        "video_name": video_path.name,
        "video_duration_seconds": round(duration_sec, 2),
        "total_frames": total_frames,
        "analyzed_frames": analyzed_frames_count,
        "frames_with_detections": frames_with_detections,
        "classes_detected": list(sorted_summary.keys()),
        "classes_summary": sorted_summary,
        "active_intervals": [
            {
                "class": s["class"],
                "start": format_timestamp(s["start_time"]),
                "end": format_timestamp(s["end_time"]),
                "max_confidence": s["max_conf"],
            }
            for s in consolidated_segments
        ],
    }

    # Save video json analysis
    video_out_dir.mkdir(parents=True, exist_ok=True)
    with open(video_out_dir / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    return result_data


def analyze_all_videos(
    videos_dir: Path | str,
    weights_path: Path | str,
    output_dir: Path | str,
    conf_threshold: float = 0.35,
    frame_stride: int = 2,
):
    videos_dir = Path(videos_dir).resolve()
    weights_path = Path(weights_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    print(f"Loading trained YOLOv8 model from: {weights_path}")
    model = YOLO(str(weights_path))

    video_extensions = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}
    video_files = [f for f in videos_dir.iterdir() if f.suffix.lower() in video_extensions]

    if not video_files:
        print(f"No video files found in {videos_dir}")
        return

    print(f"Found {len(video_files)} video(s) to analyze.")

    all_results = {}
    for vid in video_files:
        res = analyze_video(
            video_path=vid,
            model=model,
            output_dir=output_dir,
            conf_threshold=conf_threshold,
            frame_stride=frame_stride,
        )
        if res:
            all_results[vid.name] = res

    # Save overall summary
    with open(output_dir / "overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'#'*75}")
    print(f" All videos analyzed successfully! Results saved to: {output_dir}")
    print(f"{'#'*75}\n")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Extract frames and analyze object presence in videos using trained YOLOv8.")
    parser.add_argument("--videos-dir", type=str, default=str(base_dir / "videos"), help="Folder containing input videos")
    parser.add_argument("--weights", type=str, default=str(base_dir / "models" / "best.pt"), help="Trained model path (best.pt)")
    parser.add_argument("--output-dir", type=str, default=str(base_dir / "runs" / "video_analysis"), help="Output directory")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--stride", type=int, default=2, help="Frame stride (process every Nth frame, default: 2)")

    args = parser.parse_args()

    analyze_all_videos(
        videos_dir=args.videos_dir,
        weights_path=args.weights,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        frame_stride=args.stride,
    )
