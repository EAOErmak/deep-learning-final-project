from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO on screenshots and save annotated images with a center crosshair.")
    parser.add_argument("--weights", type=Path, default=Path("weights/yolov10s_cs2.pt"))
    parser.add_argument("--input-dir", type=Path, default=Path("yolo_debug/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("yolo_debug/output"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument(
        "--enemy-team",
        choices=["ct", "t", "any"],
        default="any",
        help="Optional label filter for runtime-style enemy selection.",
    )
    return parser.parse_args()


def draw_crosshair(image) -> None:
    height, width = image.shape[:2]
    cx = width // 2
    cy = height // 2
    color = (0, 255, 255)
    size = max(10, min(width, height) // 40)
    cv2.drawMarker(
        image,
        (cx, cy),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=size,
        thickness=2,
        line_type=cv2.LINE_AA,
    )


def runtime_enemy_labels(enemy_team: str) -> list[str]:
    if enemy_team == "ct":
        return ["ch", "ct_head"]
    if enemy_team == "t":
        return ["th", "t_head"]
    return ["ch", "th", "ct_head", "t_head"]


def is_runtime_enemy(label: str, enemy_team: str) -> bool:
    label_l = label.lower()
    return any(expected in label_l for expected in runtime_enemy_labels(enemy_team))


def annotate_image(model: YOLO, image_path: Path, output_path: Path, conf: float, line_width: int, enemy_team: str) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    results = model.predict(image, conf=conf, verbose=False)
    rendered = image.copy()
    draw_crosshair(rendered)

    if results:
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            label = str(model.names[cls_id])
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]

            runtime_match = is_runtime_enemy(label, enemy_team)
            color = (0, 220, 0) if runtime_match else (0, 120, 255)
            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, line_width)

            text = f"{label} {score:.2f}"
            if runtime_match:
                text += " enemy"
            cv2.putText(
                rendered,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), rendered)


def main() -> int:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    model = YOLO(str(args.weights))
    image_paths = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        print(f"No screenshots found in {args.input_dir}")
        return 0

    print(f"Loaded weights: {args.weights}")
    print(f"Processing {len(image_paths)} screenshot(s) from {args.input_dir}")
    for image_path in image_paths:
        output_path = args.output_dir / image_path.name
        annotate_image(model, image_path, output_path, args.conf, args.line_width, args.enemy_team)
        print(f"saved -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
