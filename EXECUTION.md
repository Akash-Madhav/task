# 📋 Execution & Setup Guide: Alphabet Vision & Analyzer

This guide details the complete setup, execution, training, evaluation, and troubleshooting procedures for the **Alphabet Vision & Analyzer** project.

---

## 💻 1. System Requirements

* **Operating System**: Windows 10 / 11 (64-bit)
* **Python**: Version `3.10` to `3.14`
* **GPU (Recommended)**: NVIDIA GPU with CUDA support (e.g., RTX 3050, 4060, etc.) for high-speed inference (~8–10 ms/frame). CPU is also supported automatically.
* **Camera**: Standard USB or built-in webcam.

---

## 📦 2. Environment Setup & Installation

### Option A: Using Existing Python Environment
If you are already running Python in your command prompt:

```powershell
# Navigate to the workspace directory
cd c:\Users\akash\task

# Install all required packages
pip install -r requirements.txt
```

### Option B: Using a Dedicated Virtual Environment (Recommended)

```powershell
# 1. Open PowerShell and navigate to the project directory
cd c:\Users\akash\task

# 2. Create a virtual environment named .venv
python -m venv .venv

# 3. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 3. Running the CustomTkinter OpenCV Application

### Method 1: Using the Batch Launcher (One-Click)
In Windows File Explorer, double-click:
```plaintext
run_app.bat
```

### Method 2: From PowerShell or Command Prompt
With your environment active, run:
```powershell
python alphabet_detector_app.py
```

---

## 🎮 4. Application User Guide

### 1. Video Feeds & Input Selection
* **Start Camera / Stop Camera**: Toggles the live OpenCV webcam stream. The camera stream is mirrored for natural physical interaction.
* **Camera Dropdown**: Select between `Camera 0`, `Camera 1`, or `Camera 2` if multiple cameras are connected.
* **Pause / Resume**: Freezes the active stream without disconnecting the camera.
* **Open Image/Video**: Opens a file dialog allowing you to test any local picture (`.jpg`, `.png`, `.webp`) or video (`.mp4`, `.avi`, `.mov`).
* **Test Sample**: Instantly selects and runs detection on a random image from the test dataset (`data/My First Project.v1i.yolov8/test/images`). Great for verification without a webcam!
* **Capture Snapshot**: Takes an annotated screenshot of the current frame and saves it to the `captures/` folder.

### 2. Live Continuous Multi-Letter Analysis
* **Live Detections Summary & Chips**: Displays colored badge chips for every letter currently detected in frame simultaneously, complete with NATO phonetic identifiers, vowel/consonant tags, alphabet positions, and frequency count badges.
* **Count Badge**: Shows the total count of detected objects and unique letters in real time.
* **Average Confidence Bar**: Tracks the mean detection confidence percentage across all visible objects.
* **All Detected Objects & Spatial Data**: Formatted multi-letter detail ledger providing:
  * Letter label, phonetic designation, vowel/consonant classification, and confidence score.
  * Spatial geometry: bounding box coordinates `[X1, Y1, X2, Y2]`, dimensions (width × height in pixels), centroid `(Cx, Cy)`, and video frame coverage percentage.
  * Computational data: ASCII decimal and hexadecimal values plus vocabulary example words.

### 3. Settings & Sensitivity Tab
* **Confidence Threshold**: Drag the slider (from 5% to 95%) to adjust detection sensitivity.
* **IoU Threshold**: Adjust overlap suppression to avoid duplicate boxes.
* **Visual Overlays**: Check or uncheck *Bounding Boxes*, *Labels*, *Confidence*, or *Corner Accents*.
* **Voice Feedback**: Check *Speak Detected Alphabet* to enable native Windows SAPI voice announcements with automatic cooldown debouncing.
* **Model Weights**: Browse and load custom `.pt` model files at runtime.
* **Compute Device**: Switch between `CUDA` and `CPU` with a single click.

### 4. Detection History Tab
* View a live, timestamped chronological table of all detected letters.
* Click **Export CSV** to save detection records with timestamps, confidence scores, and bounding box coordinates.
* Click **Clear Log** to reset the history.

---

## 🧠 5. Training the YOLOv8 Alphabet Model

The project includes both the primary stabilized training pipeline (`train_improved.py`) and the original baseline trainer (`train.py`).

### Primary Recommended Training: `train_improved.py`
Optimized for small, imbalanced alphabet datasets. Uses **SGD optimizer with cosine decay**, character-safe augmentations (disables letter-flipping `fliplr=0, flipud=0` and `mixup=0`), and automatically saves the best model to `models/best.pt`.

```powershell
# Run stabilized training with YOLOv8s (small) base weights
python train_improved.py --weights models/yolov8s.pt --epochs 100 --batch 8 --patience 25
```

#### Results Achieved:
* **Validation mAP@50**: **79.13%** (Precision: 74.28%, Recall: 66.00%)
* **Unseen Test Set mAP@50**: **60.01%** (Precision: 71.52%, GPU inference ~15 ms/frame)
* **Training Time**: ~22 minutes for 81 epochs on RTX 3050 Laptop GPU (automatic early stopping)

### Alternative Baseline Training: `train.py`
Standard YOLOv8 baseline training script:

```powershell
python train.py --weights models/yolov8s.pt --epochs 50 --batch 8 --imgsz 640
```

### Training CLI Arguments:
* `--data`: Path to `data.yaml` (Default: `data/My First Project.v1i.yolov8/data.yaml`).
* `--weights`: Base model to start training from (Default: `models/yolov8s.pt`).
* `--epochs`: Number of training epochs (Default: `100` for `train_improved.py`, `25` for `train.py`).
* `--imgsz`: Input image size (Default: `640`).
* `--batch`: Batch size (Default: `8`, optimal for 6GB VRAM GPUs).
* `--patience`: Early stopping patience (Default: `25` epochs).
* `--device`: Hardware device `0` for GPU, or `cpu`.

Trained weights are automatically saved to `models/best.pt` upon completion, which the desktop app loads automatically.

---

## 📊 6. Evaluating the Model

To compute detailed benchmarks (Precision, Recall, mAP@50, mAP@50-95, confusion matrices, and speed metrics):

```powershell
# Evaluate on the validation set
python evaluate_model.py --split val

# Evaluate on the unseen test set
python evaluate_model.py --split test
```

Evaluation plots, curves, and summaries are saved to `runs/evaluation/`.

---

## 📹 7. Offline Video Analysis

To run detection on recorded videos and generate timeline detection logs and annotated video outputs:

```powershell
# Analyze all videos in the videos/ directory
python analyze_videos.py --conf 0.30 --stride 2

# Save annotated video output
python analyze_videos.py --save-video
```

Results are stored in `runs/video_analysis/<video_name>/`.

---

## 🔧 8. Troubleshooting & FAQ

### Issue: "Webcam not detected / Black screen"
* **Solution**: Ensure your webcam is not in use by another application (Zoom, Teams, etc.). Select `Camera 1` or `Camera 2` from the dropdown menu, or click **Test Sample** to test with static images.

### Issue: "CUDA out of memory"
* **Solution**: In the GUI's *Settings & Sensitivity* tab, switch the **Compute Acceleration Device** to `CPU`, or in `train.py` reduce the `--batch` size to `8` or `4`.

### Issue: "Windows SAPI speech not working"
* **Solution**: Ensure `pywin32` is installed (`pip install pywin32`). The speech engine utilizes standard Windows built-in voice synthesizer without external internet dependencies.

### Issue: "Low detection accuracy"
* **Solution**: 
  1. Ensure the alphabet card or drawing has clear contrast against the background.
  2. Adjust the **Confidence Threshold** slider in the Settings tab down to `20%–25%`.
  3. Ensure the letter is within the 17 trained alphabet classes (`A, B, C, D, E, F, G, H, J, K, L, O, P, T, U, V, Z`).
