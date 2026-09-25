"""
Alphabet Vision & Analyzer - CustomTkinter + OpenCV + YOLOv8
Real-time alphabet detection, bounding box visualization, and detailed phonetic/linguistic analysis.
"""

import os
import sys
import time
import math
import csv
import threading
import random
from datetime import datetime
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import customtkinter as ctk

# Try importing torch & ultralytics
try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Try importing win32com for text-to-speech
try:
    import win32com.client
    SAPI_AVAILABLE = True
except ImportError:
    SAPI_AVAILABLE = False


# ============================================================================
# ALPHABET KNOWLEDGE BASE & LINGUISTIC ANALYZER
# ============================================================================
NATO_PHONETIC = {
    'A': 'Alpha',   'B': 'Bravo',    'C': 'Charlie', 'D': 'Delta',
    'E': 'Echo',    'F': 'Foxtrot',  'G': 'Golf',    'H': 'Hotel',
    'I': 'India',   'J': 'Juliett',  'K': 'Kilo',    'L': 'Lima',
    'M': 'Mike',    'N': 'November', 'O': 'Oscar',   'P': 'Papa',
    'Q': 'Quebec',  'R': 'Romeo',    'S': 'Sierra',  'T': 'Tango',
    'U': 'Uniform', 'V': 'Victor',   'W': 'Whiskey', 'X': 'X-ray',
    'Y': 'Yankee',  'Z': 'Zulu'
}

SAMPLE_WORDS = {
    'A': 'Apple, Anchor, Airplane',
    'B': 'Balloon, Banana, Bridge',
    'C': 'Cat, Castle, Camera',
    'D': 'Dolphin, Diamond, Drum',
    'E': 'Elephant, Eagle, Engine',
    'F': 'Falcon, Flame, Flower',
    'G': 'Guitar, Galaxy, Garden',
    'H': 'Horizon, Harmony, Helmet',
    'I': 'Island, Igloo, Iceberg',
    'J': 'Jaguar, Jungle, Journey',
    'K': 'Kangaroo, Kingdom, Kite',
    'L': 'Lion, Lantern, Lightning',
    'M': 'Mountain, Moon, Magnet',
    'N': 'Nebula, Nautilus, Navigator',
    'O': 'Ocean, Owl, Orbit',
    'P': 'Phoenix, Planet, Pyramid',
    'Q': 'Quartz, Quantum, Quest',
    'R': 'Rocket, Rainbow, River',
    'S': 'Satellite, Star, Symphony',
    'T': 'Telescope, Tiger, Tornado',
    'U': 'Universe, Umbrella, Urchin',
    'V': 'Volcano, Velocity, Violin',
    'W': 'Wave, Whale, Windmill',
    'X': 'Xylophone, Xenon, X-ray',
    'Y': 'Yacht, Yellow, Yield',
    'Z': 'Zephyr, Zenith, Zebra'
}

VOWELS = set(['A', 'E', 'I', 'O', 'U'])

# Distinct visually appealing colors for each alphabet (BGR format for OpenCV)
CLASS_COLORS = {
    'A': (60, 220, 20),    # Bright Emerald
    'B': (235, 140, 20),   # Vibrant Cyan/Blue
    'C': (30, 180, 255),   # Sunset Orange
    'D': (240, 50, 160),   # Neon Purple
    'E': (40, 240, 240),   # Sun Yellow
    'F': (180, 20, 240),   # Hot Pink
    'G': (100, 255, 120),  # Mint Green
    'H': (240, 200, 40),   # Azure Blue
    'I': (180, 140, 255),  # Lavender
    'J': (60, 130, 255),   # Coral Orange
    'K': (255, 100, 100),  # Soft Sky Blue
    'L': (50, 210, 170),   # Seafoam
    'M': (150, 80, 240),   # Magenta
    'N': (220, 220, 80),   # Pale Cyan
    'O': (30, 230, 255),   # Amber
    'P': (210, 80, 210),   # Violet
    'Q': (120, 220, 80),   # Spring Green
    'R': (240, 120, 80),   # Cornflower Blue
    'S': (80, 240, 180),   # Lime
    'T': (200, 160, 40),   # Deep Cyan
    'U': (40, 160, 240),   # Tangerine
    'V': (180, 70, 220),   # Deep Purple
    'W': (240, 240, 100),  # Electric Ice
    'X': (90, 190, 240),   # Apricot
    'Y': (210, 210, 50),   # Teal
    'Z': (50, 255, 230)    # Chartreuse
}


def get_alphabet_metadata(letter: str):
    """Returns rich linguistic and computational metadata for a given alphabet."""
    char = letter.upper() if letter else '?'
    is_alpha = char.isalpha() and len(char) == 1
    
    if not is_alpha:
        return {
            'letter': char,
            'phonetic': 'Unknown',
            'type': 'Symbol/Other',
            'position': 'N/A',
            'ascii_dec': ord(char) if char else 0,
            'ascii_hex': hex(ord(char)).upper() if char else '0x00',
            'sample_words': 'None'
        }
    
    pos = ord(char) - ord('A') + 1
    letter_type = "Vowel" if char in VOWELS else "Consonant"
    phonetic = NATO_PHONETIC.get(char, "Unknown")
    sample_words = SAMPLE_WORDS.get(char, "N/A")
    
    return {
        'letter': char,
        'phonetic': phonetic,
        'type': letter_type,
        'position': f"{pos} of 26",
        'pos_num': pos,
        'ascii_dec': ord(char),
        'ascii_hex': f"0x{ord(char):02X}",
        'sample_words': sample_words
    }


def get_letter_hex_color(letter: str) -> str:
    """Returns hexadecimal color code for a given letter based on CLASS_COLORS."""
    char = letter.upper() if letter else '?'
    bgr = CLASS_COLORS.get(char, (137, 180, 250))  # Default Catppuccin Blue
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


# ============================================================================
# THREADED SPEECH ENGINE
# ============================================================================
class VoiceAnnouncer:
    """Non-blocking Windows SAPI Text-to-Speech manager with debounce."""
    def __init__(self):
        self.lock = threading.Lock()
        self.enabled = False
        self.last_spoken_letter = ""
        self.last_spoken_time = 0.0
        self.cooldown = 2.0  # seconds between repeated announcements
        self.queue = deque(maxlen=3)
        self.worker_thread = None
        self.running = False
        
        if SAPI_AVAILABLE:
            try:
                self.running = True
                self.worker_thread = threading.Thread(target=self._speech_loop, daemon=True)
                self.worker_thread.start()
            except Exception as e:
                print(f"[VoiceAnnouncer] SAPI initialization failed: {e}")

    def _speech_loop(self):
        # COM must be initialized in the thread that uses it
        import pythoncom
        pythoncom.CoInitialize()
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        voice.Rate = 1  # slightly faster speech rate
        
        while self.running:
            text_to_say = None
            with self.lock:
                if self.queue:
                    text_to_say = self.queue.popleft()
            
            if text_to_say and self.enabled:
                try:
                    voice.Speak(text_to_say)
                except Exception as e:
                    print(f"[VoiceAnnouncer] Error speaking: {e}")
            else:
                time.sleep(0.08)

    def announce_letter(self, letter: str, phonetic: str = ""):
        if not self.enabled or not SAPI_AVAILABLE:
            return
        
        now = time.time()
        with self.lock:
            # Debounce: avoid repeating the same letter too fast
            if letter == self.last_spoken_letter and (now - self.last_spoken_time) < self.cooldown:
                return
            self.last_spoken_letter = letter
            self.last_spoken_time = now
            phrase = f"Alphabet {letter}. {phonetic}" if phonetic else f"Alphabet {letter}"
            self.queue.clear()
            self.queue.append(phrase)

    def stop(self):
        self.running = False


# ============================================================================
# MODERN OPENCV VISUALIZER & BOUNDING BOX RENDERER
# ============================================================================
class FrameAnnotator:
    """Renders sleek, modern bounding boxes with high-tech corner brackets and badges."""
    
    @staticmethod
    def draw_corner_rect(img, x1, y1, x2, y2, color, thickness=2, corner_len=18):
        """Draws rounded/bracketed corners for high-tech aesthetics."""
        w = x2 - x1
        h = y2 - y1
        c_len = min(corner_len, w // 3, h // 3)
        
        # Main subtle boundary (semi-transparent rectangle)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, thickness // 2), cv2.LINE_AA)
        
        # Corner highlight brackets (thicker)
        t = thickness + 1
        # Top-Left
        cv2.line(img, (x1, y1), (x1 + c_len, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + c_len), color, t, cv2.LINE_AA)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - c_len, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + c_len), color, t, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + c_len, y2), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - c_len), color, t, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - c_len, y2), color, t, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - c_len), color, t, cv2.LINE_AA)

    @staticmethod
    def draw_detection_badge(img, label, conf, x1, y1, color):
        """Draws a sleek pill badge with label and confidence above the bounding box."""
        badge_text = f"{label} {int(conf * 100)}%"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        thickness = 1
        
        (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)
        
        # Position badge above box, or inside if too close to top
        margin = 6
        badge_h = th + margin * 2
        badge_w = tw + margin * 2
        
        by1 = max(0, y1 - badge_h - 4)
        by2 = by1 + badge_h
        bx1 = x1
        bx2 = x1 + badge_w
        
        if by1 < 5:
            by1 = y1 + 4
            by2 = by1 + badge_h
            
        # Draw badge background
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, -1, cv2.LINE_AA)
        
        # Inner dark border or contrast text
        # Compute brightness to choose black or white text
        b, g, r = color
        luminance = (0.299 * r + 0.587 * g + 0.114 * b)
        text_color = (15, 15, 20) if luminance > 140 else (255, 255, 255)
        
        text_origin = (bx1 + margin, by2 - margin - 2)
        cv2.putText(img, badge_text, text_origin, font, font_scale, text_color, thickness, cv2.LINE_AA)


# ============================================================================
# VISION PIPELINE WORKER (THREADED)
# ============================================================================
class VisionWorker:
    """Manages OpenCV capture, YOLO inference, and threaded frame streaming."""
    
    def __init__(self, on_frame_processed_callback=None):
        self.callback = on_frame_processed_callback
        self.lock = threading.Lock()
        self.running = False
        self.paused = False
        
        # Shared slot for GUI pull model (decouples capture from rendering)
        self.latest_data = None
        self.new_frame_available = False
        
        self.cap = None
        self.source_type = "camera"  # "camera", "file", "sample"
        self.camera_index = 0
        self.media_path = None
        
        # YOLO Model
        self.model = None
        self.model_path = "models/best.pt"
        self.device = "cuda" if (torch.cuda.is_available() if YOLO_AVAILABLE else False) else "cpu"
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        
        # Visualization options
        self.show_bbox = True
        self.show_label = True
        self.show_conf = True
        self.tech_corners = True
        
        # FPS and latency tracking
        self.fps = 0.0
        self.inference_time_ms = 0.0
        self._frame_count = 0
        self._fps_start_time = time.time()
        
        # Thread
        self.worker_thread = None
        
        # Load model initially
        self.load_model(self.model_path)

    def get_latest_frame(self):
        """Thread-safe retrieval of latest processed frame for GUI polling."""
        with self.lock:
            if not self.new_frame_available:
                return None
            self.new_frame_available = False
            return self.latest_data

    def load_model(self, model_path: str):
        with self.lock:
            self.model_path = model_path
            if not YOLO_AVAILABLE:
                print("[VisionWorker] YOLO / Ultralytics not installed.")
                return False
            try:
                resolved_path = Path(model_path).resolve()
                if not resolved_path.exists():
                    print(f"[VisionWorker] Weights file not found at: {resolved_path}")
                    return False
                
                print(f"[VisionWorker] Loading YOLO model from: {resolved_path} (Device: {self.device})")
                self.model = YOLO(str(resolved_path))
                print(f"[VisionWorker] Model loaded successfully. Classes: {list(self.model.names.values())}")
                return True
            except Exception as e:
                print(f"[VisionWorker] Failed to load model: {e}")
                self.model = None
                return False

    def set_device(self, device: str):
        with self.lock:
            self.device = device
            if self.model:
                try:
                    self.model.to(device)
                except Exception as e:
                    print(f"[VisionWorker] Failed to set device: {e}")

    def start_camera(self, camera_idx: int = 0):
        self.stop_stream()
        with self.lock:
            self.source_type = "camera"
            self.camera_index = camera_idx
            self.running = True
            self.paused = False
        
        self.worker_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.worker_thread.start()

    def start_file(self, filepath: str):
        self.stop_stream()
        with self.lock:
            self.source_type = "file"
            self.media_path = filepath
            self.running = True
            self.paused = False
        
        self.worker_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.worker_thread.start()

    def stop_stream(self):
        with self.lock:
            self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.5)
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None

    def toggle_pause(self):
        with self.lock:
            self.paused = not self.paused
        return self.paused

    def _predict_frame(self, frame):
        """Runs YOLO inference on a single frame and returns sorted detections."""
        detections = []
        infer_ms = 0.0

        if self.model is not None:
            t0 = time.time()
            try:
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False
                )
                infer_ms = (time.time() - t0) * 1000.0

                if results and len(results) > 0:
                    r = results[0]
                    names = self.model.names
                    boxes = r.boxes
                    
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            label = names.get(cls_id, str(cls_id))
                            
                            detections.append({
                                'label': label,
                                'conf': conf,
                                'box': xyxy.tolist(),  # [x1, y1, x2, y2]
                                'cls_id': cls_id
                            })
            except Exception as e:
                print(f"[VisionWorker] Inference error: {e}")

        # Sort detections by confidence descending
        detections.sort(key=lambda d: d['conf'], reverse=True)
        return detections, infer_ms

    def _annotate_frame(self, frame, detections):
        """Draws bounding boxes and labels on a copy of frame."""
        annotated_frame = frame.copy()
        if self.show_bbox:
            for det in detections:
                label = det['label']
                conf = det['conf']
                x1, y1, x2, y2 = det['box']
                color = CLASS_COLORS.get(label.upper(), (0, 255, 128))
                
                if self.tech_corners:
                    FrameAnnotator.draw_corner_rect(annotated_frame, x1, y1, x2, y2, color, thickness=2)
                else:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                    
                if self.show_label or self.show_conf:
                    FrameAnnotator.draw_detection_badge(annotated_frame, label, conf if self.show_conf else 1.0, x1, y1, color)
        return annotated_frame

    def _stream_loop(self):
        # Handle still image files directly without busy looping VideoCapture
        if self.source_type == "file" and self.media_path:
            resolved_media = Path(self.media_path).resolve()
            ext = resolved_media.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                try:
                    frame = cv2.imdecode(np.fromfile(str(resolved_media), dtype=np.uint8), cv2.IMREAD_COLOR)
                except Exception:
                    frame = cv2.imread(str(resolved_media))
                if frame is None:
                    print(f"[VisionWorker] Error opening image {resolved_media}")
                    self.running = False
                    return
                h, w = frame.shape[:2]
                detections, infer_ms = self._predict_frame(frame)
                annotated_frame = self._annotate_frame(frame, detections)
                meta = {
                    'fps': 0.0,
                    'latency_ms': infer_ms,
                    'frame_width': w,
                    'frame_height': h,
                    'source': 'image'
                }
                with self.lock:
                    self.latest_data = (annotated_frame, frame, detections, meta)
                    self.new_frame_available = True
                
                if self.callback:
                    try:
                        self.callback(annotated_frame, frame, detections, meta)
                    except Exception:
                        pass
                
                while self.running:
                    time.sleep(0.05)
                return

        # Open video or camera source
        if self.source_type == "camera":
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
        else:
            self.cap = cv2.VideoCapture(self.media_path)

        if not self.cap or not self.cap.isOpened():
            print(f"[VisionWorker] Error opening source {self.source_type}")
            self.running = False
            return

        # Configure camera resolution
        if self.source_type == "camera":
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        is_video_file = (self.source_type == "file")
        prev_time = time.time()

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret:
                if is_video_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.03)
                    continue

            # Flip horizontal for natural webcam mirror effect
            if self.source_type == "camera":
                frame = cv2.flip(frame, 1)

            # Perform YOLO Inference & annotation
            detections, infer_ms = self._predict_frame(frame)
            annotated_frame = self._annotate_frame(frame, detections)

            # Calculate FPS
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._fps_start_time
            if elapsed >= 0.5:
                self.fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start_time = now
            self.inference_time_ms = infer_ms

            # Dispatch metadata
            h, w = frame.shape[:2]
            meta = {
                'fps': self.fps,
                'latency_ms': self.inference_time_ms,
                'frame_width': w,
                'frame_height': h,
                'source': self.source_type
            }

            # Put in shared slot for GUI pull model
            with self.lock:
                self.latest_data = (annotated_frame, frame, detections, meta)
                self.new_frame_available = True

            if self.callback:
                try:
                    self.callback(annotated_frame, frame, detections, meta)
                except Exception:
                    pass

            # Target ~30 FPS rate control to prevent UI lag and excessive GPU load
            dt = time.time() - prev_time
            target_delay = 0.033  # ~30 fps
            if dt < target_delay:
                time.sleep(target_delay - dt)
            prev_time = time.time()

        if self.cap:
            self.cap.release()
            self.cap = None


# ============================================================================
# MAIN CUSTOMTKINTER APPLICATION
# ============================================================================
class AlphabetDetectorApp(ctk.CTk):
    """Modern CustomTkinter GUI for Real-time Alphabet Detection & Analysis."""
    
    def __init__(self):
        super().__init__()
        
        # Configure App Appearance
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        self.title("Alphabet Vision & Analyzer | YOLOv8 + OpenCV")
        self.geometry("1420x880")
        self.minsize(1100, 720)
        
        # Lifecycle and polling state
        self._app_alive = True
        self._poll_timer_id = None
        self._visible_letters = set()
        self._letter_chips = []
        self._last_letter_counts = {}
        self._last_det_signature = ""
        self._last_logged_times = {}
        
        # State variables
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.latest_detections = []
        self.detection_history = []
        self.max_history = 100
        
        # Sound engine
        self.announcer = VoiceAnnouncer()
        
        # Video pipeline (GUI pulls frames via _poll_frame loop)
        self.worker = VisionWorker()
        
        # Setup UI
        self._build_header()
        self._build_main_content()
        self._build_statusbar()
        
        # Setup clean exit
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Start GUI polling loop (~30 FPS fixed timer)
        self._poll_frame()
        
        # Auto-start with sample image or camera 0 if available
        self.after(400, self._auto_init)

    def _auto_init(self):
        """Attempts to open default camera, or displays a sample test image."""
        # Check if camera 0 opens
        test_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cam_available = test_cap.isOpened()
        if cam_available:
            ret, _ = test_cap.read()
            cam_available = ret
            test_cap.release()
        
        if cam_available:
            self.start_camera_feed(0)
        else:
            self.load_random_sample_image()
            self.status_label.configure(text="Webcam not detected - Sample test image loaded. Click 'Start Camera' or select input.")

    # ------------------------------------------------------------------------
    # UI CONSTRUCTION
    # ------------------------------------------------------------------------
    def _build_header(self):
        """Header bar with app title, status indicators, and quick metrics."""
        self.header_frame = ctk.CTkFrame(self, height=64, corner_radius=0, fg_color="#181825")
        self.header_frame.pack(fill="x", side="top", padx=0, pady=0)
        self.header_frame.pack_propagate(False)

        # Title & Subtitle
        title_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        title_box.pack(side="left", padx=20, pady=10)
        
        title_label = ctk.CTkLabel(
            title_box, 
            text="ALPHABET VISION & ANALYZER", 
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color="#cdd6f4"
        )
        title_label.pack(anchor="w")
        
        sub_label = ctk.CTkLabel(
            title_box, 
            text="Deep Learning Real-Time Alphabet Detection & Linguistic Breakdown", 
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#a6adc8"
        )
        sub_label.pack(anchor="w")

        # Right badges (Model, Hardware, FPS)
        badges_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        badges_box.pack(side="right", padx=20, pady=10)

        # Hardware Badge (GPU/CPU)
        gpu_name = torch.cuda.get_device_name(0) if (YOLO_AVAILABLE and torch.cuda.is_available()) else "CPU"
        self.hw_badge = ctk.CTkLabel(
            badges_box,
            text=f"Hardware: {gpu_name}",
            fg_color="#313244",
            corner_radius=8,
            padx=12,
            pady=4,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#89dceb"
        )
        self.hw_badge.pack(side="right", padx=6)

        # Model Badge
        model_name = Path(self.worker.model_path).name
        self.model_badge = ctk.CTkLabel(
            badges_box,
            text=f"Model: {model_name}",
            fg_color="#313244",
            corner_radius=8,
            padx=12,
            pady=4,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#a6e3a1"
        )
        self.model_badge.pack(side="right", padx=6)

        # Live FPS Badge
        self.fps_badge = ctk.CTkLabel(
            badges_box,
            text="FPS: 0.0 | 0.0 ms",
            fg_color="#313244",
            corner_radius=8,
            padx=12,
            pady=4,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#f9e2af"
        )
        self.fps_badge.pack(side="right", padx=6)

    def _build_main_content(self):
        """Split screen: Video viewport on Left, Analysis Dashboard on Right."""
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=16, pady=12)

        # Left Pane: Video Screen + Video Controls (Width: 62%)
        self.left_pane = ctk.CTkFrame(self.main_container, corner_radius=12, fg_color="#1e1e2e")
        self.left_pane.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self._build_video_viewport()
        self._build_video_controls()

        # Right Pane: Analysis & Settings (Width: 38%)
        self.right_pane = ctk.CTkFrame(self.main_container, width=520, corner_radius=12, fg_color="#1e1e2e")
        self.right_pane.pack(side="right", fill="both", padx=(10, 0))
        self.right_pane.pack_propagate(False)

        self._build_analysis_dashboard()

    def _build_video_viewport(self):
        """Video canvas displaying OpenCV stream."""
        self.video_container = ctk.CTkFrame(self.left_pane, fg_color="#11111b", corner_radius=10)
        self.video_container.pack(fill="both", expand=True, padx=12, pady=(12, 8))

        # Raw tk.Label for high-performance direct PhotoImage rendering (avoids CTkImage overhead & crash)
        self.video_label = tk.Label(self.video_container, bg="#11111b", bd=0, highlightthickness=0)
        self.video_label.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_video_controls(self):
        """Controls below video viewport: Start/Pause/Stop, Camera dropdown, Browse media, Snapshot."""
        controls_frame = ctk.CTkFrame(self.left_pane, height=64, fg_color="#181825", corner_radius=10)
        controls_frame.pack(fill="x", side="bottom", padx=12, pady=(0, 12))

        # Left action buttons
        self.btn_camera = ctk.CTkButton(
            controls_frame, 
            text="Start Camera", 
            command=self.toggle_camera,
            width=110,
            fg_color="#89b4fa",
            hover_color="#74c7ec",
            text_color="#11111b",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
        )
        self.btn_camera.pack(side="left", padx=(12, 6), pady=12)

        self.btn_pause = ctk.CTkButton(
            controls_frame, 
            text="Pause", 
            command=self.toggle_pause,
            width=80,
            fg_color="#45475a",
            hover_color="#585b70"
        )
        self.btn_pause.pack(side="left", padx=6, pady=12)

        # Camera selection dropdown
        self.camera_select = ctk.CTkOptionMenu(
            controls_frame,
            values=["Camera 0", "Camera 1", "Camera 2"],
            command=self.on_camera_select,
            width=105,
            fg_color="#313244",
            button_color="#45475a"
        )
        self.camera_select.set("Camera 0")
        self.camera_select.pack(side="left", padx=6, pady=12)

        # Load File buttons
        self.btn_open_file = ctk.CTkButton(
            controls_frame,
            text="Open Image/Video",
            command=self.open_media_file,
            width=130,
            fg_color="#313244",
            hover_color="#45475a"
        )
        self.btn_open_file.pack(side="left", padx=6, pady=12)

        # Test Sample Button
        self.btn_sample = ctk.CTkButton(
            controls_frame,
            text="Test Sample",
            command=self.load_random_sample_image,
            width=100,
            fg_color="#313244",
            hover_color="#45475a"
        )
        self.btn_sample.pack(side="left", padx=6, pady=12)

        # Snapshot button (Right side)
        self.btn_snapshot = ctk.CTkButton(
            controls_frame,
            text="Capture Snapshot",
            command=self.capture_snapshot,
            width=130,
            fg_color="#a6e3a1",
            hover_color="#94e2d5",
            text_color="#11111b",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
        )
        self.btn_snapshot.pack(side="right", padx=12, pady=12)

    def _build_analysis_dashboard(self):
        """Right analysis panel with Tabview: Analysis, Controls, History."""
        self.tabview = ctk.CTkTabview(self.right_pane, fg_color="#1e1e2e")
        self.tabview.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_analysis = self.tabview.add("Alphabet Analysis")
        self.tab_settings = self.tabview.add("Settings & Sensitivity")
        self.tab_history = self.tabview.add("Detection History")

        self._build_tab_analysis(self.tab_analysis)
        self._build_tab_settings(self.tab_settings)
        self._build_tab_history(self.tab_history)

    def _build_tab_analysis(self, parent):
        """Live multi-letter tracking dashboard showing all visible alphabets simultaneously."""
        # 1. Header Card: Live Summary & Multi-Letter Chip Badges
        self.summary_card = ctk.CTkFrame(parent, fg_color="#181825", corner_radius=12)
        self.summary_card.pack(fill="x", padx=6, pady=8)

        card_top = ctk.CTkFrame(self.summary_card, fg_color="transparent")
        card_top.pack(fill="x", padx=14, pady=(12, 6))

        ctk.CTkLabel(
            card_top,
            text="LIVE DETECTIONS",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#89b4fa"
        ).pack(side="left")

        self.count_badge = ctk.CTkLabel(
            card_top,
            text="0 letters detected",
            fg_color="#313244",
            corner_radius=8,
            padx=12,
            pady=3,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#cdd6f4"
        )
        self.count_badge.pack(side="right")

        # Container for dynamic letter chips
        self.chips_container = ctk.CTkFrame(self.summary_card, fg_color="transparent")
        self.chips_container.pack(fill="x", padx=14, pady=(4, 14))

        self.no_chips_label = ctk.CTkLabel(
            self.chips_container,
            text="Awaiting alphabet in camera view...",
            font=ctk.CTkFont(family="Segoe UI", size=13, slant="italic"),
            text_color="#6c7086"
        )
        self.no_chips_label.pack(anchor="w", pady=6)

        # 2. Average Confidence Bar Section
        conf_frame = ctk.CTkFrame(parent, fg_color="#181825", corner_radius=10)
        conf_frame.pack(fill="x", padx=6, pady=(0, 8))

        conf_header = ctk.CTkFrame(conf_frame, fg_color="transparent")
        conf_header.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(
            conf_header,
            text="AVERAGE DETECTION CONFIDENCE",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#a6adc8"
        ).pack(side="left")

        self.conf_percent_label = ctk.CTkLabel(
            conf_header,
            text="0%",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#a6e3a1"
        )
        self.conf_percent_label.pack(side="right")

        self.conf_progress = ctk.CTkProgressBar(
            conf_frame,
            height=10,
            corner_radius=5,
            progress_color="#a6e3a1",
            fg_color="#313244"
        )
        self.conf_progress.set(0.0)
        self.conf_progress.pack(fill="x", padx=12, pady=(0, 12))

        # 3. All Visible Detections List & Spatial Data (Fills remaining height)
        multi_frame = ctk.CTkFrame(parent, fg_color="#181825", corner_radius=10)
        multi_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        multi_header = ctk.CTkFrame(multi_frame, fg_color="transparent")
        multi_header.pack(fill="x", padx=12, pady=(10, 6))

        ctk.CTkLabel(
            multi_header,
            text="ALL DETECTED OBJECTS & SPATIAL DATA",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#a6adc8"
        ).pack(side="left")

        self.multi_detections_box = ctk.CTkTextbox(
            multi_frame,
            height=280,
            fg_color="#11111b",
            text_color="#cdd6f4",
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="none"
        )
        self.multi_detections_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.multi_detections_box.insert("0.0", "No detections currently in view.\n")

    def _update_letter_chips(self, detections):
        """Updates the horizontal row of visual badge chips for all currently detected letters."""
        for chip in self._letter_chips:
            chip.destroy()
        self._letter_chips.clear()

        if not detections:
            self.no_chips_label.pack(anchor="w", pady=6)
            return

        self.no_chips_label.pack_forget()

        # Count occurrences of each letter
        letter_counts = {}
        for d in detections:
            lbl = d['label']
            letter_counts[lbl] = letter_counts.get(lbl, 0) + 1

        # Create chip for each unique letter (alphabetical order)
        for letter in sorted(letter_counts.keys()):
            count = letter_counts[letter]
            info = get_alphabet_metadata(letter)
            hex_color = get_letter_hex_color(letter)

            chip = ctk.CTkFrame(
                self.chips_container,
                fg_color="#1e1e2e",
                border_width=2,
                border_color=hex_color,
                corner_radius=10
            )
            chip.pack(side="left", padx=4, pady=4)

            # Left side: Big letter
            lbl_char = ctk.CTkLabel(
                chip,
                text=letter,
                font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
                text_color=hex_color
            )
            lbl_char.pack(side="left", padx=(10, 6), pady=6)

            # Right side: phonetic + count or type
            sub_frame = ctk.CTkFrame(chip, fg_color="transparent")
            sub_frame.pack(side="left", padx=(0, 10), pady=6)

            phonetic_text = info['phonetic']
            if count > 1:
                phonetic_text += f" (x{count})"

            lbl_sub = ctk.CTkLabel(
                sub_frame,
                text=phonetic_text,
                font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
                text_color="#cdd6f4",
                anchor="w"
            )
            lbl_sub.pack(anchor="w")

            lbl_type = ctk.CTkLabel(
                sub_frame,
                text=f"{info['type']} • #{info['pos_num']}",
                font=ctk.CTkFont(family="Segoe UI", size=9),
                text_color="#a6adc8",
                anchor="w"
            )
            lbl_type.pack(anchor="w")

            self._letter_chips.append(chip)

    def _build_tab_settings(self, parent):
        """Sensitivity sliders, model file picker, and visualization styling."""
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=4, pady=4)

        # Confidence Threshold Slider
        ctk.CTkLabel(
            scroll, 
            text="Confidence Threshold", 
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#cdd6f4"
        ).pack(anchor="w", padx=8, pady=(8, 2))

        conf_row = ctk.CTkFrame(scroll, fg_color="transparent")
        conf_row.pack(fill="x", padx=8, pady=(0, 8))

        self.slider_conf = ctk.CTkSlider(
            conf_row, 
            from_=0.05, 
            to=0.95, 
            number_of_steps=90,
            command=self._on_conf_change
        )
        self.slider_conf.set(self.worker.conf_threshold)
        self.slider_conf.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.lbl_conf_val = ctk.CTkLabel(
            conf_row, 
            text=f"{int(self.worker.conf_threshold * 100)}%", 
            width=45,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#89b4fa"
        )
        self.lbl_conf_val.pack(side="right")

        # IoU / NMS Threshold Slider
        ctk.CTkLabel(
            scroll, 
            text="IoU / NMS Overlap Threshold", 
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#cdd6f4"
        ).pack(anchor="w", padx=8, pady=(8, 2))

        iou_row = ctk.CTkFrame(scroll, fg_color="transparent")
        iou_row.pack(fill="x", padx=8, pady=(0, 8))

        self.slider_iou = ctk.CTkSlider(
            iou_row, 
            from_=0.10, 
            to=0.90, 
            number_of_steps=80,
            command=self._on_iou_change
        )
        self.slider_iou.set(self.worker.iou_threshold)
        self.slider_iou.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.lbl_iou_val = ctk.CTkLabel(
            iou_row, 
            text=f"{int(self.worker.iou_threshold * 100)}%", 
            width=45,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#89b4fa"
        )
        self.lbl_iou_val.pack(side="right")

        # Visual Overlays Toggles
        ctk.CTkLabel(
            scroll, 
            text="Visual Overlays", 
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#cdd6f4"
        ).pack(anchor="w", padx=8, pady=(12, 6))

        self.chk_bbox = ctk.CTkCheckBox(scroll, text="Show Bounding Boxes", command=self._on_toggle_visuals)
        self.chk_bbox.select()
        self.chk_bbox.pack(anchor="w", padx=12, pady=4)

        self.chk_labels = ctk.CTkCheckBox(scroll, text="Show Alphabet Labels & Badges", command=self._on_toggle_visuals)
        self.chk_labels.select()
        self.chk_labels.pack(anchor="w", padx=12, pady=4)

        self.chk_conf = ctk.CTkCheckBox(scroll, text="Show Confidence Percentages", command=self._on_toggle_visuals)
        self.chk_conf.select()
        self.chk_conf.pack(anchor="w", padx=12, pady=4)

        self.chk_corners = ctk.CTkCheckBox(scroll, text="High-Tech Corner Accents", command=self._on_toggle_visuals)
        self.chk_corners.select()
        self.chk_corners.pack(anchor="w", padx=12, pady=4)

        # Speech & Audio Announcements
        ctk.CTkLabel(
            scroll, 
            text="Audio Feedback", 
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#cdd6f4"
        ).pack(anchor="w", padx=8, pady=(16, 6))

        self.chk_voice = ctk.CTkCheckBox(
            scroll, 
            text="Speak Detected Alphabet (Windows SAPI Voice)",
            command=self._on_toggle_voice
        )
        if SAPI_AVAILABLE:
            self.chk_voice.pack(anchor="w", padx=12, pady=4)
        else:
            self.chk_voice.configure(state="disabled")

        # Model Weights Selection
        ctk.CTkLabel(
            scroll, 
            text="YOLO Model Weights (.pt)", 
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#cdd6f4"
        ).pack(anchor="w", padx=8, pady=(16, 6))

        model_row = ctk.CTkFrame(scroll, fg_color="transparent")
        model_row.pack(fill="x", padx=8, pady=(0, 8))

        self.entry_model_path = ctk.CTkEntry(model_row, placeholder_text="Path to .pt weights")
        self.entry_model_path.insert(0, self.worker.model_path)
        self.entry_model_path.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_browse_model = ctk.CTkButton(
            model_row, 
            text="Browse", 
            width=70, 
            command=self.browse_model_file
        )
        self.btn_browse_model.pack(side="left", padx=(0, 6))

        self.btn_load_model = ctk.CTkButton(
            model_row, 
            text="Reload", 
            width=70, 
            command=self.reload_model_file
        )
        self.btn_load_model.pack(side="left")

        # Device Selection (GPU / CPU)
        ctk.CTkLabel(
            scroll, 
            text="Compute Acceleration Device", 
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#cdd6f4"
        ).pack(anchor="w", padx=8, pady=(16, 6))

        devices = ["cuda"] if (torch.cuda.is_available() if YOLO_AVAILABLE else False) else []
        devices.append("cpu")

        self.device_select = ctk.CTkSegmentedButton(
            scroll,
            values=devices,
            command=self._on_device_change
        )
        self.device_select.set(self.worker.device)
        self.device_select.pack(fill="x", padx=8, pady=(0, 12))

    def _build_tab_history(self, parent):
        """Detection log table with export option."""
        top_bar = ctk.CTkFrame(parent, fg_color="transparent")
        top_bar.pack(fill="x", padx=6, pady=8)

        ctk.CTkLabel(
            top_bar,
            text="CHRONOLOGICAL LOG",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#cdd6f4"
        ).pack(side="left")

        self.btn_clear_log = ctk.CTkButton(
            top_bar,
            text="Clear Log",
            width=80,
            fg_color="#313244",
            hover_color="#45475a",
            command=self.clear_history
        )
        self.btn_clear_log.pack(side="right", padx=(6, 0))

        self.btn_export_csv = ctk.CTkButton(
            top_bar,
            text="Export CSV",
            width=90,
            fg_color="#89b4fa",
            hover_color="#74c7ec",
            text_color="#11111b",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            command=self.export_history_csv
        )
        self.btn_export_csv.pack(side="right")

        # History Text Area
        self.history_box = ctk.CTkTextbox(
            parent,
            fg_color="#11111b",
            text_color="#cdd6f4",
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.history_box.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.history_box.insert("0.0", f"{'TIME':<10} | {'LETTER':<6} | {'CONF':<8} | {'PHONETIC':<10} | {'TYPE':<10}\n")
        self.history_box.insert("end", "-" * 55 + "\n")

    def _build_statusbar(self):
        """Bottom status bar."""
        self.statusbar = ctk.CTkFrame(self, height=28, fg_color="#181825", corner_radius=0)
        self.statusbar.pack(fill="x", side="bottom")
        self.statusbar.pack_propagate(False)

        self.status_label = ctk.CTkLabel(
            self.statusbar,
            text="Ready. Select camera or file to begin.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#a6adc8"
        )
        self.status_label.pack(side="left", padx=16)

        self.status_res_label = ctk.CTkLabel(
            self.statusbar,
            text="Resolution: --",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#a6adc8"
        )
        self.status_res_label.pack(side="right", padx=16)

    # ------------------------------------------------------------------------
    # EVENT HANDLERS & CALLBACKS
    # ------------------------------------------------------------------------
    def _poll_frame(self):
        """Fixed-rate GUI poll loop (~30 FPS) decoupling camera processing from GUI rendering."""
        if not self._app_alive:
            return

        try:
            data = self.worker.get_latest_frame()
            if data is not None:
                annotated_frame, raw_frame, detections, meta = data
                self.latest_raw_frame = raw_frame
                self.latest_annotated_frame = annotated_frame
                self.latest_detections = detections
                self._update_gui(annotated_frame, detections, meta)
        except Exception as e:
            pass
        finally:
            if self._app_alive:
                self._poll_timer_id = self.after(33, self._poll_frame)

    def _update_gui(self, frame_bgr, detections, meta):
        """Updates video display, multi-letter summary, detailed detections, and history."""
        if not self._app_alive:
            return

        # 1. Update video canvas with high-performance ImageTk.PhotoImage
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            
            # Viewport target dimensions
            target_w = max(100, self.video_container.winfo_width() - 10)
            target_h = max(100, self.video_container.winfo_height() - 10)

            # Preserve aspect ratio
            scale = min(target_w / w, target_h / h)
            disp_w = max(1, int(w * scale))
            disp_h = max(1, int(h * scale))

            pil_img = Image.fromarray(frame_rgb).resize((disp_w, disp_h), Image.Resampling.BILINEAR)
            photo = ImageTk.PhotoImage(image=pil_img)
            
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # keep reference to prevent GC
        except Exception:
            pass

        # 2. Update status and FPS
        fps = meta.get('fps', 0.0)
        lat = meta.get('latency_ms', 0.0)
        frame_w = meta.get('frame_width', 0)
        frame_h = meta.get('frame_height', 0)
        self.fps_badge.configure(text=f"FPS: {fps:.1f} | {lat:.1f} ms")
        self.status_res_label.configure(text=f"Source: {frame_w}x{frame_h}")

        # 3. Multi-Letter Tracking & Analysis Updates
        if detections and len(detections) > 0:
            current_letters = {d['label'] for d in detections}
            letter_counts = {}
            for d in detections:
                lbl = d['label']
                letter_counts[lbl] = letter_counts.get(lbl, 0) + 1

            # Voice announcement for newly-appeared letters
            new_letters = current_letters - self._visible_letters
            for l in sorted(new_letters):
                info = get_alphabet_metadata(l)
                self.announcer.announce_letter(l, info['phonetic'])

            # Update letter chips if the visible letters or their counts changed
            chips_changed = (letter_counts != self._last_letter_counts)
            if chips_changed:
                self._visible_letters = current_letters
                self._last_letter_counts = letter_counts
                self._update_letter_chips(detections)

                n_total = len(detections)
                n_uniq = len(current_letters)
                badge_text = f"{n_total} letter{'s' if n_total != 1 else ''} ({n_uniq} unique)"
                self.count_badge.configure(text=badge_text)

            # Signature check for details and confidence
            det_sig = "|".join(f"{d['label']}:{d['conf']:.2f}:{d['box']}" for d in detections)
            sig_changed = (det_sig != self._last_det_signature)

            if sig_changed:
                self._last_det_signature = det_sig

                # Average Confidence
                avg_conf = sum(d['conf'] for d in detections) / len(detections)
                self.conf_progress.set(avg_conf)
                self.conf_percent_label.configure(text=f"{int(avg_conf * 100)}%")

                # Build rich multi-detection details
                lines = []
                frame_area = max(1, frame_w * frame_h)
                for i, d in enumerate(detections, 1):
                    letter = d['label']
                    conf = d['conf']
                    box = d['box']
                    info = get_alphabet_metadata(letter)

                    x1, y1, x2, y2 = box
                    bw = x2 - x1
                    bh = y2 - y1
                    cx = x1 + bw // 2
                    cy = y1 + bh // 2
                    cov_pct = (bw * bh / frame_area) * 100.0

                    lines.append(f"[{i}] Alphabet '{letter}' ({info['phonetic']}) | {info['type']} | Conf: {int(conf * 100)}%")
                    lines.append(f"    • Box: [{x1}, {y1}, {x2}, {y2}] | Dim: {bw}x{bh} px | Center: ({cx}, {cy}) | Coverage: {cov_pct:.1f}%")
                    lines.append(f"    • ASCII: {info['ascii_dec']} ({info['ascii_hex']}) | Words: {info['sample_words']}")
                    lines.append("")

                self.multi_detections_box.delete("0.0", "end")
                self.multi_detections_box.insert("0.0", "\n".join(lines).rstrip())

                # Status line update
                letters_display = ", ".join(f"'{l}'" for l in sorted(current_letters))
                self.status_label.configure(
                    text=f"Live tracking {len(detections)} letter(s): {letters_display} | Avg Conf: {int(avg_conf * 100)}%"
                )

                # Log detections (debounced per letter)
                self._log_detection(detections)

        else:
            if self._visible_letters:
                self._visible_letters = set()
                self._last_letter_counts = {}
                self._last_det_signature = ""
                self._update_letter_chips([])
                self.count_badge.configure(text="0 letters detected")
                self.conf_progress.set(0.0)
                self.conf_percent_label.configure(text="0%")
                self.multi_detections_box.delete("0.0", "end")
                self.multi_detections_box.insert("0.0", "No detections currently in view.\n")
                self.status_label.configure(text="Awaiting alphabet in camera view...")

    def _log_detection(self, detections):
        """Logs detections into chronological history table with per-letter debouncing."""
        now = time.time()
        time_str = datetime.now().strftime("%H:%M:%S")

        for d in detections:
            letter = d['label']
            conf = d['conf']
            box = d['box']

            # Debounce same letter within 1.5 seconds
            last_t = self._last_logged_times.get(letter, 0.0)
            if now - last_t < 1.5:
                continue
            self._last_logged_times[letter] = now

            info = get_alphabet_metadata(letter)
            entry = {
                'timestamp': now,
                'time_str': time_str,
                'letter': letter,
                'conf': f"{int(conf * 100)}%",
                'phonetic': info['phonetic'],
                'type': info['type'],
                'box': box
            }
            self.detection_history.append(entry)

            # Insert at top of history textbox (row 2 after header and divider)
            log_line = f"{time_str:<10} | {letter:<6} | {int(conf * 100):>3}%     | {info['phonetic']:<10} | {info['type']:<10}\n"
            self.history_box.insert("2.0", log_line)

    # ------------------------------------------------------------------------
    # CONTROLS & SETTINGS CALLBACKS
    # ------------------------------------------------------------------------
    def toggle_camera(self):
        if not self.worker.running or self.worker.source_type != "camera":
            cam_idx = int(self.camera_select.get().split()[-1])
            self.start_camera_feed(cam_idx)
        else:
            self.worker.stop_stream()
            self.btn_camera.configure(text="Start Camera", fg_color="#89b4fa")
            self.status_label.configure(text="Camera stopped.")

    def start_camera_feed(self, cam_idx: int):
        self.worker.start_camera(cam_idx)
        self.btn_camera.configure(text="Stop Camera", fg_color="#f38ba8")
        self.status_label.configure(text=f"Live feed started on Camera {cam_idx}.")

    def on_camera_select(self, choice):
        cam_idx = int(choice.split()[-1])
        if self.worker.running and self.worker.source_type == "camera":
            self.start_camera_feed(cam_idx)

    def toggle_pause(self):
        is_paused = self.worker.toggle_pause()
        self.btn_pause.configure(
            text="Resume" if is_paused else "Pause",
            fg_color="#fab387" if is_paused else "#45475a"
        )

    def open_media_file(self):
        from tkinter import filedialog
        filetypes = [
            ("All Supported Media", "*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.mp4;*.avi;*.mov;*.mkv"),
            ("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
            ("Videos", "*.mp4;*.avi;*.mov;*.mkv")
        ]
        path = filedialog.askopenfilename(title="Select Image or Video File", filetypes=filetypes)
        if path:
            self.worker.start_file(path)
            self.btn_camera.configure(text="Start Camera", fg_color="#89b4fa")
            self.status_label.configure(text=f"Loaded media: {Path(path).name}")

    def load_random_sample_image(self):
        """Loads a random test image from the dataset test folder if available."""
        test_dir = Path("data/My First Project.v1i.yolov8/test/images")
        if not test_dir.exists():
            test_dir = Path("data")
        
        image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        if not image_files:
            # Search recursively in workspace
            image_files = list(Path(".").glob("**/*.jpg"))

        if image_files:
            chosen = random.choice(image_files)
            self.worker.start_file(str(chosen))
            self.btn_camera.configure(text="Start Camera", fg_color="#89b4fa")
            self.status_label.configure(text=f"Loaded sample test image: {chosen.name}")
        else:
            self.status_label.configure(text="No sample images found in dataset folder.")

    def capture_snapshot(self):
        """Saves current annotated frame and raw frame to captures/ directory."""
        if self.latest_annotated_frame is None:
            self.status_label.configure(text="No frame available to capture.")
            return

        captures_dir = Path("captures")
        captures_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        annotated_path = captures_dir / f"alphabet_detected_{timestamp}.jpg"
        
        cv2.imwrite(str(annotated_path), self.latest_annotated_frame)
        self.status_label.configure(text=f"Snapshot saved: {annotated_path.name}")
        print(f"[Snapshot] Saved frame to {annotated_path.resolve()}")

    def _on_conf_change(self, val):
        self.worker.conf_threshold = float(val)
        self.lbl_conf_val.configure(text=f"{int(val * 100)}%")

    def _on_iou_change(self, val):
        self.worker.iou_threshold = float(val)
        self.lbl_iou_val.configure(text=f"{int(val * 100)}%")

    def _on_toggle_visuals(self):
        self.worker.show_bbox = bool(self.chk_bbox.get())
        self.worker.show_label = bool(self.chk_labels.get())
        self.worker.show_conf = bool(self.chk_conf.get())
        self.worker.tech_corners = bool(self.chk_corners.get())

    def _on_toggle_voice(self):
        self.announcer.enabled = bool(self.chk_voice.get())

    def _on_device_change(self, device):
        self.worker.set_device(device)
        self.hw_badge.configure(text=f"Device: {device.upper()}")

    def browse_model_file(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select YOLO Weights (.pt)",
            filetypes=[("PyTorch Model", "*.pt")]
        )
        if path:
            self.entry_model_path.delete(0, "end")
            self.entry_model_path.insert(0, path)
            self.reload_model_file()

    def reload_model_file(self):
        path = self.entry_model_path.get().strip()
        success = self.worker.load_model(path)
        if success:
            model_name = Path(path).name
            self.model_badge.configure(text=f"Model: {model_name}")
            self.status_label.configure(text=f"Successfully loaded model weights: {model_name}")
        else:
            self.status_label.configure(text=f"Failed to load model from: {path}")

    def clear_history(self):
        self.detection_history.clear()
        self.history_box.delete("0.0", "end")
        self.history_box.insert("0.0", f"{'TIME':<10} | {'LETTER':<6} | {'CONF':<8} | {'PHONETIC':<10} | {'TYPE':<10}\n")
        self.history_box.insert("end", "-" * 55 + "\n")
        self.status_label.configure(text="History cleared.")

    def export_history_csv(self):
        if not self.detection_history:
            self.status_label.configure(text="No detection history to export.")
            return

        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            title="Export Detections to CSV",
            defaultextension=".csv",
            filetypes=[("CSV File", "*.csv")]
        )
        if path:
            try:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Time", "Letter", "Confidence", "Phonetic", "Type", "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"])
                    for row in self.detection_history:
                        box = row['box']
                        writer.writerow([row['timestamp'], row['time_str'], row['letter'], row['conf'], row['phonetic'], row['type'], box[0], box[1], box[2], box[3]])
                self.status_label.configure(text=f"Exported {len(self.detection_history)} rows to {Path(path).name}")
            except Exception as e:
                self.status_label.configure(text=f"Export error: {e}")

    def on_close(self):
        """Gracefully release camera, cancel polling timer, and stop background worker threads."""
        print("[App] Closing AlphabetDetectorApp...")
        self._app_alive = False
        if self._poll_timer_id is not None:
            try:
                self.after_cancel(self._poll_timer_id)
            except Exception:
                pass
            self._poll_timer_id = None

        self.worker.stop_stream()
        self.announcer.stop()
        try:
            self.destroy()
        except Exception:
            pass


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    app = AlphabetDetectorApp()
    app.mainloop()
