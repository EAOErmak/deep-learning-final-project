import threading
import time
from pathlib import Path
from dataclasses import dataclass

try:
    import mss
    import numpy as np
    from ultralytics import YOLO
except ImportError:
    mss = None
    np = None
    YOLO = None

@dataclass
class VisionTarget:
    screen_dx: float  # [-1.0, 1.0] relative to screen center
    screen_dy: float
    confidence: float
    label: str

class YoloVisionModule:
    def __init__(self, model_path: Path, capture_fps: int = 60):
        self.model_path = model_path
        self.capture_fps = capture_fps
        self.model = None
        self.sct = None
        
        self._thread = None
        self._stop_event = threading.Event()
        
        # Shared state
        self.latest_target: VisionTarget | None = None
        self.player_team: str | None = None
        self.lock = threading.Lock()
        self.is_running = False

    def update_context(self, player_team: str | None):
        self.player_team = player_team

    def start(self):
        if YOLO is None:
            print("Vision modules not installed. Skipping YOLO.")
            return

        if not self.model_path.exists():
            print(f"YOLO model not found at {self.model_path}. Skipping YOLO.")
            return

        print(f"Loading YOLO model from {self.model_path}...")
        self.model = YOLO(str(self.model_path))
        self.sct = mss.mss()
        
        self.is_running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("YOLO Vision Pipeline started.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.is_running = False

    def get_latest_target(self) -> VisionTarget | None:
        with self.lock:
            return self.latest_target

    def _capture_loop(self):
        monitor = self.sct.monitors[1]  # Primary monitor
        width = monitor["width"]
        height = monitor["height"]
        center_x = width / 2.0
        center_y = height / 2.0
        
        # Capture 640x640 crop in the center to save performance
        crop_size = 640
        left = int(center_x - crop_size / 2)
        top = int(center_y - crop_size / 2)
        bbox = {"top": top, "left": left, "width": crop_size, "height": crop_size}

        frame_time = 1.0 / self.capture_fps

        while not self._stop_event.is_set():
            start_t = time.perf_counter()
            
            sct_img = self.sct.grab(bbox)
            img = np.array(sct_img)[:, :, :3]  # Drop alpha channel

            # Inference
            results = self.model.predict(img, verbose=False)
            
            best_target = None
            min_dist = float('inf')
            
            # Determine enemy labels based on player team
            enemy_labels = []
            if self.player_team == "CT":
                enemy_labels = ["th", "t_head"]
            elif self.player_team == "T":
                enemy_labels = ["ch", "ct_head"]
            else:
                # Fallback: target any head if team is unknown
                enemy_labels = ["ch", "th", "ct_head", "t_head"]
            
            if len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    label = self.model.names[cls_id]
                    
                    label_l = label.lower()
                    is_enemy = any(e in label_l for e in enemy_labels)
                    
                    # Ensure it's an enemy AND it's a head
                    if not is_enemy or ("head" not in label_l and label_l not in ["ch", "th"]):
                        continue
                        
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    box_center_x = (x1 + x2) / 2.0
                    box_center_y = (y1 + y2) / 2.0
                    
                    full_x = left + box_center_x
                    full_y = top + box_center_y
                    
                    dist = (full_x - center_x)**2 + (full_y - center_y)**2
                    
                    if dist < min_dist:
                        min_dist = dist
                        # Normalize identically to world_to_screen_delta
                        screen_dx = (full_x - center_x) / center_x
                        screen_dy = (full_y - center_y) / center_x  # Use center_x to preserve aspect ratio aspect
                        
                        best_target = VisionTarget(
                            screen_dx=screen_dx,
                            screen_dy=screen_dy,
                            confidence=conf,
                            label=label
                        )
            
            with self.lock:
                self.latest_target = best_target
                
            elapsed = time.perf_counter() - start_t
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
