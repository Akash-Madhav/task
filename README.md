# 🔤 Alphabet Vision & Analyzer

An AI-powered computer vision and deep learning desktop application built with **CustomTkinter**, **OpenCV**, and **YOLOv8**. The application performs real-time alphabet detection from a webcam or media files, renders bounding boxes with high-tech corner accents, and provides linguistic, spatial, and phonetic analysis of detected letters.

---

## 📸 Key Features

* **🎨 Modern CustomTkinter Interface**: Sleek dark-mode aesthetic with responsive scaling, custom color palettes, and real-time dashboard cards.
* **🎥 OpenCV Video Engine**: Seamless webcam streaming (Camera 0, 1, 2) with mirror view, pause/resume, image/video file loader, and sample test generator.
* **⚡ YOLOv8 Object Detection**: Powered by a custom-trained alphabet model (`models/best.pt`), optimized for real-time inference with NVIDIA CUDA GPU acceleration (~8–10 ms latency).
* **📐 High-Tech Bounding Box Visualizer**:
  * Curved / chamfered corner-bracket styling.
  * Class pill tags and dynamic confidence percentages.
  * Unique, high-contrast color codes for each alphabet.
* **🧠 Live Continuous Multi-Letter Tracking & Analysis Engine**:
  * **Dynamic Letter Badge Chips**: Real-time colored chips for every alphabet simultaneously detected in view, showing NATO phonetic code, vowel/consonant type, alphabet position, and multi-occurrence count.
  * **Multi-Letter Spatial & Linguistic Ledger**: Comprehensive breakdown for all visible letters including bounding box coordinates `[X1, Y1, X2, Y2]`, dimensions `Width × Height`, center point `(Cx, Cy)`, frame coverage percentage, ASCII codes, and sample vocabulary words.
  * **Average Confidence Gauge**: Live animated indicator displaying the mean detection certainty across all detected alphabets.
* **📊 Detection History & CSV Export**: Chronological log of detected alphabets with timestamps, confidence scores, and one-click export to CSV.
* **🔊 Audio Announcements**: Optional text-to-speech announcement of newly detected letters powered by Windows SAPI voice engine with intelligent cooldown debouncing.
* **🎛️ Real-Time Sensitivity Sliders**: Live adjustment of confidence threshold (5%–95%) and IoU overlap threshold.

---

## 🗂️ Project Structure

```plaintext
task/
├── alphabet_detector_app.py        # Main CustomTkinter + OpenCV GUI application
├── run_app.bat                     # Quick Windows launcher script
├── train_improved.py               # Primary YOLOv8s training pipeline (SGD, safe augmentations, 79.1% mAP)
├── train.py                        # Baseline YOLOv8 training script with epoch callback tracking
├── evaluate_model.py               # Validation & model benchmark evaluation script
├── analyze_videos.py               # Video offline inference and timeline analysis
├── download_model.py               # Helper script to download base YOLO weights
├── requirements.txt                # Python project dependencies
├── README.md                       # Comprehensive project documentation
├── EXECUTION.md                    # Step-by-step setup & execution guide
├── data/
│   └── My First Project.v1i.yolov8/ # Roboflow alphabet dataset (train/valid/test + data.yaml)
├── models/
│   ├── best.pt                     # Custom-trained YOLOv8 alphabet detection weights (79.1% mAP@50)
│   ├── yolov8s.pt                  # Base YOLOv8 small model weights (fine-tuning backbone)
│   └── yolov8n.pt                  # Base YOLOv8 nano model weights
├── videos/                         # Test video files for offline analysis
├── runs/                           # Training, evaluation, and video inference output logs
└── captures/                       # Snapshots saved from the desktop application
```

---

## 🔡 Supported Alphabet Classes

The current custom-trained YOLOv8 model recognizes 17 alphabet classes:

| Class ID | Letter | NATO Phonetic | Type | ASCII Dec |
| :---: | :---: | :---: | :---: | :---: |
| 0 | **A** | Alpha | Vowel | 65 |
| 1 | **B** | Bravo | Consonant | 66 |
| 2 | **C** | Charlie | Consonant | 67 |
| 3 | **D** | Delta | Consonant | 68 |
| 4 | **E** | Echo | Vowel | 69 |
| 5 | **F** | Foxtrot | Consonant | 70 |
| 6 | **G** | Golf | Consonant | 71 |
| 7 | **H** | Hotel | Consonant | 72 |
| 8 | **J** | Juliett | Consonant | 74 |
| 9 | **K** | Kilo | Consonant | 75 |
| 10 | **L** | Lima | Consonant | 76 |
| 11 | **O** | Oscar | Vowel | 79 |
| 12 | **P** | Papa | Consonant | 80 |
| 13 | **T** | Tango | Consonant | 84 |
| 14 | **U** | Uniform | Vowel | 85 |
| 15 | **V** | Victor | Consonant | 86 |
| 16 | **Z** | Zulu | Consonant | 90 |

*(Note: Custom weights covering additional alphabets can be loaded directly from the UI).*

---

## 🚀 Quick Start

### 1. Launch GUI Directly
Double-click `run_app.bat` or run:

```bash
python alphabet_detector_app.py
```

### 2. Basic Operations
* **Start Camera**: Activates the webcam feed and begins real-time inference.
* **Show Alphabet**: Hold up a printed or drawn alphabet in front of your camera.
* **Test Sample**: Click *Test Sample* to immediately run inference on random samples from the dataset test directory without needing a webcam.
* **Capture Snapshot**: Saves current annotated frame to `captures/`.
* **Export CSV**: Open *Detection History* tab and click *Export CSV*.

---

## 🛠️ Complete Pipeline Scripts

| Script | Purpose | Example Command |
| :--- | :--- | :--- |
| `alphabet_detector_app.py` | Real-time Desktop GUI with OpenCV | `python alphabet_detector_app.py` |
| `train_improved.py` | Primary YOLOv8s trainer with SGD & safe augmentations (79.1% mAP@50) | `python train_improved.py --weights models/yolov8s.pt --epochs 100 --batch 8` |
| `train.py` | Baseline YOLOv8 training script | `python train.py --weights models/yolov8s.pt --epochs 50 --batch 8` |
| `evaluate_model.py` | Benchmark precision, recall, and mAP | `python evaluate_model.py --split test` |
| `analyze_videos.py` | Batch video analysis and timeline logging | `python analyze_videos.py --conf 0.30` |
| `download_model.py` | Download base weights (`yolov8s.pt`, `yolov8n.pt`) | `python download_model.py --model yolov8s.pt` |

For full setup instructions, virtual environment configuration, and troubleshooting, see [EXECUTION.md](file:///c:/Users/akash/task/EXECUTION.md).
