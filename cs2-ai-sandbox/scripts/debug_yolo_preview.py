from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cv2
    import mss
    import numpy as np
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(f'Missing vision dependency: {exc}')

from cs2_ai.vision.window_capture import CaptureRegion, WindowCaptureLocator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Show a live YOLO preview window with detections from the CS2 window.')
    parser.add_argument('--yolo-weights', type=Path, default=Path('weights/yolov10s_cs2.pt'))
    parser.add_argument('--capture-fps', type=int, default=20)
    parser.add_argument('--window-keyword', action='append', default=None)
    parser.add_argument('--player-team', choices=['CT', 'T', 'unknown'], default='unknown')
    parser.add_argument('--conf-threshold', type=float, default=0.25)
    return parser.parse_args()


def resolve_capture_region(sct: 'mss.mss', window_region: CaptureRegion | None) -> CaptureRegion:
    if window_region is not None:
        crop_size = min(640, window_region.width, window_region.height)
        crop_left = window_region.left + max(0, (window_region.width - crop_size) // 2)
        crop_top = window_region.top + max(0, (window_region.height - crop_size) // 2)
        return CaptureRegion(left=crop_left, top=crop_top, width=crop_size, height=crop_size)

    monitor = sct.monitors[1]
    width = int(monitor['width'])
    height = int(monitor['height'])
    crop_size = min(640, width, height)
    left = int(width / 2.0 - crop_size / 2)
    top = int(height / 2.0 - crop_size / 2)
    return CaptureRegion(left=left, top=top, width=crop_size, height=crop_size)


def resolve_enemy_labels(player_team: str) -> list[str]:
    if player_team == 'CT':
        return ['th', 't_head']
    if player_team == 'T':
        return ['ch', 'ct_head']
    return ['ch', 'th', 'ct_head', 't_head']


def main() -> int:
    args = parse_args()
    if not args.yolo_weights.exists():
        raise SystemExit(f'YOLO weights not found: {args.yolo_weights}')

    window_keywords = tuple(args.window_keyword) if args.window_keyword else ('counter-strike', 'cs2')
    locator = WindowCaptureLocator(window_keywords=window_keywords)
    model = YOLO(str(args.yolo_weights))
    sct = mss.MSS()
    frame_time = 1.0 / max(args.capture_fps, 1)
    enemy_labels = resolve_enemy_labels(args.player_team)
    window_name = 'YOLO Debug Preview'

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 900)

    try:
        while True:
            started_at = time.perf_counter()
            window_region = locator.find_client_region()
            capture_region = resolve_capture_region(sct, window_region)
            raw_frame = np.array(sct.grab(capture_region.as_mss_bbox()))
            frame = np.ascontiguousarray(raw_frame[:, :, :3])
            results = model.predict(frame, verbose=False, conf=args.conf_threshold)

            best_idx = None
            best_dist = float('inf')
            center_x = capture_region.width / 2.0
            center_y = capture_region.height / 2.0

            if len(results) > 0:
                boxes = results[0].boxes
                for idx, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    label = str(model.names[cls_id])
                    label_l = label.lower()
                    is_enemy = any(token in label_l for token in enemy_labels)
                    is_head = 'head' in label_l or label_l in {'ch', 'th'}
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

                    color = (255, 166, 77)
                    if is_enemy and is_head:
                        color = (82, 82, 255)
                    elif is_enemy:
                        color = (71, 179, 255)

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(
                        frame,
                        f'{label} {conf:.2f}',
                        (int(x1), max(18, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

                    if is_enemy and is_head:
                        box_center_x = (x1 + x2) / 2.0
                        box_center_y = (y1 + y2) / 2.0
                        dist = (box_center_x - center_x) ** 2 + (box_center_y - center_y) ** 2
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = idx

                if best_idx is not None:
                    selected = boxes[best_idx]
                    x1, y1, x2, y2 = [int(v) for v in selected.xyxy[0].tolist()]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 3)

            cv2.line(frame, (int(center_x - 8), int(center_y)), (int(center_x + 8), int(center_y)), (255, 255, 255), 1)
            cv2.line(frame, (int(center_x), int(center_y - 8)), (int(center_x), int(center_y + 8)), (255, 255, 255), 1)

            header = 'CS2 window found' if window_region is not None else 'CS2 window not found, using monitor center'
            cv2.putText(frame, header, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 216, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                f'crop={capture_region.width}x{capture_region.height} | team={args.player_team} | q/esc to exit',
                (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

            elapsed = time.perf_counter() - started_at
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
