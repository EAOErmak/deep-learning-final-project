import threading
import time
from pathlib import Path
from dataclasses import dataclass
import logging

try:
    import mss
    import numpy as np
    from ultralytics import YOLO
except ImportError:
    mss = None
    np = None
    YOLO = None

from cs2_ai.vision.window_capture import CaptureRegion, WindowCaptureLocator
from cs2_ai.vision.yolo_overlay import OverlayDetection, YoloOverlayWindow

@dataclass
class VisionTarget:
    screen_dx: float  # [-1.0, 1.0] relative to screen center
    screen_dy: float
    confidence: float
    label: str


@dataclass
class VisionDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float
    is_enemy: bool
    is_head: bool
    is_selected: bool = False

class YoloVisionModule:
    def __init__(
        self,
        model_path: Path,
        capture_fps: int = 60,
        window_keywords: tuple[str, ...] = ('counter-strike', 'cs2'),
        show_overlay: bool = False,
    ):
        self.model_path = model_path
        self.capture_fps = capture_fps
        self.model = None
        self.sct = None
        self.logger = logging.getLogger(__name__)
        self.window_locator = WindowCaptureLocator(window_keywords=window_keywords)
        self.overlay = YoloOverlayWindow() if show_overlay else None
        
        self._thread = None
        self._stop_event = threading.Event()
        
        # Shared state
        self.latest_target: VisionTarget | None = None
        self.latest_detections: list[VisionDetection] = []
        self.latest_window_region: CaptureRegion | None = None
        self.latest_capture_region: CaptureRegion | None = None
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
        if self.overlay is not None:
            self.overlay.start()
        
        self.is_running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("YOLO Vision Pipeline started.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.overlay is not None:
            self.overlay.stop()
        self.is_running = False

    def get_latest_target(self) -> VisionTarget | None:
        with self.lock:
            return self.latest_target

    def get_debug_snapshot(self) -> tuple[CaptureRegion | None, CaptureRegion | None, list[VisionDetection], VisionTarget | None]:
        with self.lock:
            return (
                self.latest_window_region,
                self.latest_capture_region,
                list(self.latest_detections),
                self.latest_target,
            )

    def _capture_loop(self):
        frame_time = 1.0 / self.capture_fps

        while not self._stop_event.is_set():
            start_t = time.perf_counter()

            window_region = self.window_locator.find_client_region()
            capture_region = self._resolve_capture_region(window_region)
            left = capture_region.left
            top = capture_region.top
            width = capture_region.width
            height = capture_region.height
            if window_region is not None:
                center_x = window_region.left + window_region.width / 2.0
                center_y = window_region.top + window_region.height / 2.0
            else:
                center_x = left + width / 2.0
                center_y = top + height / 2.0
            bbox = capture_region.as_mss_bbox()

            sct_img = self.sct.grab(bbox)
            img = np.array(sct_img)[:, :, :3]  # Drop alpha channel

            # Inference
            results = self.model.predict(img, verbose=False)
            
            best_target = None
            min_dist = float('inf')
            detections: list[VisionDetection] = []
            
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
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    label_l = label.lower()
                    is_enemy = any(e in label_l for e in enemy_labels)
                    is_head = "head" in label_l or label_l in ["ch", "th"]

                    full_x1 = left + float(x1)
                    full_y1 = top + float(y1)
                    full_x2 = left + float(x2)
                    full_y2 = top + float(y2)
                    if window_region is not None:
                        rel_x1 = full_x1 - window_region.left
                        rel_y1 = full_y1 - window_region.top
                        rel_x2 = full_x2 - window_region.left
                        rel_y2 = full_y2 - window_region.top
                    else:
                        rel_x1 = float(x1)
                        rel_y1 = float(y1)
                        rel_x2 = float(x2)
                        rel_y2 = float(y2)

                    detections.append(
                        VisionDetection(
                            x1=rel_x1,
                            y1=rel_y1,
                            x2=rel_x2,
                            y2=rel_y2,
                            label=label,
                            confidence=conf,
                            is_enemy=is_enemy,
                            is_head=is_head,
                        )
                    )
                    
                    # Ensure it's an enemy AND it's a head
                    if not is_enemy or not is_head:
                        continue

                    box_center_x = (x1 + x2) / 2.0
                    box_center_y = (y1 + y2) / 2.0
                    
                    full_x = left + box_center_x
                    full_y = top + box_center_y
                    
                    dist = (full_x - center_x)**2 + (full_y - center_y)**2
                    
                    if dist < min_dist:
                        min_dist = dist
                        # Normalize identically to world_to_screen_delta
                        norm_base = max(width / 2.0, 1.0)
                        screen_dx = (full_x - center_x) / norm_base
                        screen_dy = (full_y - center_y) / norm_base  # Use width to preserve aspect ratio handling
                        
                        best_target = VisionTarget(
                            screen_dx=screen_dx,
                            screen_dy=screen_dy,
                            confidence=conf,
                            label=label
                        )

            if best_target is not None:
                selected_idx = self._select_detection_index(detections, best_target, window_region, capture_region)
                if selected_idx is not None:
                    detections[selected_idx].is_selected = True
            
            with self.lock:
                self.latest_target = best_target
                self.latest_detections = detections
                self.latest_window_region = window_region
                self.latest_capture_region = capture_region
            if self.overlay is not None:
                overlay_detections = [
                    OverlayDetection(
                        x1=item.x1,
                        y1=item.y1,
                        x2=item.x2,
                        y2=item.y2,
                        label=item.label,
                        confidence=item.confidence,
                        is_enemy=item.is_enemy,
                        is_head=item.is_head,
                        is_selected=item.is_selected,
                    )
                    for item in detections
                ]
                self.overlay.update(
                    window_region=window_region,
                    capture_region=capture_region if window_region is not None else None,
                    detections=overlay_detections,
                )
                
            elapsed = time.perf_counter() - start_t
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _resolve_capture_region(self, window_region: CaptureRegion | None) -> CaptureRegion:
        if window_region is not None:
            crop_size = min(640, window_region.width, window_region.height)
            crop_left = window_region.left + max(0, (window_region.width - crop_size) // 2)
            crop_top = window_region.top + max(0, (window_region.height - crop_size) // 2)
            return CaptureRegion(left=crop_left, top=crop_top, width=crop_size, height=crop_size)

        monitor = self.sct.monitors[1]  # Primary monitor
        width = int(monitor['width'])
        height = int(monitor['height'])
        crop_size = min(640, width, height)
        left = int(width / 2.0 - crop_size / 2)
        top = int(height / 2.0 - crop_size / 2)
        return CaptureRegion(left=left, top=top, width=crop_size, height=crop_size)

    def _select_detection_index(
        self,
        detections: list[VisionDetection],
        target: VisionTarget,
        window_region: CaptureRegion | None,
        capture_region: CaptureRegion,
    ) -> int | None:
        if window_region is None:
            return None
        norm_base = max(capture_region.width / 2.0, 1.0)
        target_center_x = window_region.width / 2.0 + target.screen_dx * norm_base
        target_center_y = window_region.height / 2.0 + target.screen_dy * norm_base
        best_idx = None
        best_dist = float('inf')
        for idx, detection in enumerate(detections):
            center_x = (detection.x1 + detection.x2) / 2.0
            center_y = (detection.y1 + detection.y2) / 2.0
            dist = (center_x - target_center_x) ** 2 + (center_y - target_center_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx
