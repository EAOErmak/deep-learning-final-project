from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs2_ai.vision.yolo_pipeline import YoloVisionModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run YOLO with a live transparent overlay above the CS2 window.')
    parser.add_argument('--yolo-weights', type=Path, default=Path('weights/yolov10s_cs2.pt'))
    parser.add_argument('--capture-fps', type=int, default=30)
    parser.add_argument('--window-keyword', action='append', default=None)
    parser.add_argument('--player-team', choices=['CT', 'T', 'unknown'], default='unknown')
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
    args = parse_args()
    window_keywords = tuple(args.window_keyword) if args.window_keyword else ('counter-strike', 'cs2')

    module = YoloVisionModule(
        args.yolo_weights,
        capture_fps=args.capture_fps,
        window_keywords=window_keywords,
        show_overlay=True,
    )
    if args.player_team != 'unknown':
        module.update_context(args.player_team)
    module.start()
    if not module.is_running:
        return 1

    running = True
    last_target_signature = None

    def stop_handler(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    logging.info('YOLO overlay started | window_keywords=%s | player_team=%s', window_keywords, args.player_team)
    try:
        while running:
            target = module.get_latest_target()
            if target is not None:
                signature = (target.label, round(target.screen_dx, 4), round(target.screen_dy, 4), round(target.confidence, 4))
                if signature != last_target_signature:
                    last_target_signature = signature
                    logging.info(
                        'Latest target | label=%s dx=%.4f dy=%.4f conf=%.4f',
                        target.label,
                        target.screen_dx,
                        target.screen_dy,
                        target.confidence,
                    )
            time.sleep(0.25)
    finally:
        module.stop()
        logging.info('YOLO overlay stopped.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
