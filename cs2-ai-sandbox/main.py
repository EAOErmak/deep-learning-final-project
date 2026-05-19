from __future__ import annotations

import logging
import signal
import time
from typing import Any

from dummy_agent import DummyAgent
from feature_encoder import encode_state
from input_controller import InputController
from state_reader import MockStateReader

LOOP_HZ = 10.0
LOOP_SLEEP_SECONDS = 1.0 / LOOP_HZ


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    configure_logging()

    running = True

    def stop_handler(_signum: int, _frame: Any) -> None:
        nonlocal running
        logging.info("Shutdown requested. Stopping sandbox loop.")
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    state_reader = MockStateReader()
    agent = DummyAgent()
    input_controller = InputController()

    logging.info("CS2 AI sandbox started. Press Ctrl+C to stop.")

    try:
        while running:
            loop_started_at = time.perf_counter()

            raw_state = state_reader.read_state()
            features = encode_state(raw_state)
            action = agent.predict(features)
            input_controller.apply(action)

            logging.info("Features: %s", features)
            logging.info("Action:   %s", action)

            elapsed = time.perf_counter() - loop_started_at
            time.sleep(max(0.0, LOOP_SLEEP_SECONDS - elapsed))
    finally:
        input_controller.stop_all()
        logging.info("All inputs released.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
