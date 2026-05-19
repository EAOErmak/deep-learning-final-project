from __future__ import annotations

import argparse
import logging
import signal
import time
from typing import Any

from dummy_agent import DummyAgent
from feature_encoder import encode_state
from game_state import GameState
from gsi_server import GSIServer
from input_controller import InputController
from runtime_agent import PipelineRuntimeAgent
from state_reader import StateReader


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CS2 AI sandbox runtime')
    parser.add_argument('--state-source', choices=['mock', 'gsi'], default='mock')
    parser.add_argument('--gsi-port', type=int, default=3000)
    parser.add_argument('--hz', type=float, default=10.0)
    return parser.parse_args()


def build_agent(state_source: str) -> Any:
    if state_source == 'gsi':
        logging.info('Using PipelineRuntimeAgent for live GSI sandbox mode.')
        return PipelineRuntimeAgent()
    logging.info('Using DummyAgent fallback for mock mode.')
    return DummyAgent()


def main() -> int:
    configure_logging()
    args = parse_args()
    loop_sleep_seconds = 1.0 / max(args.hz, 0.1)
    running = True
    gsi_server: GSIServer | None = None

    def stop_handler(_signum: int, _frame: Any) -> None:
        nonlocal running
        logging.info('Shutdown requested. Stopping sandbox loop.')
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    if args.state_source == 'gsi':
        gsi_server = GSIServer(port=args.gsi_port)
        gsi_server.start()

    state_reader = StateReader(mode=args.state_source, gsi_server=gsi_server)
    agent = build_agent(args.state_source)
    input_controller = InputController()

    logging.info('CS2 AI sandbox started | state_source=%s | hz=%.2f', args.state_source, args.hz)

    try:
        while running:
            loop_started_at = time.perf_counter()
            raw_state = state_reader.read_state()
            if raw_state is None:
                logging.info('Waiting for GSI payload...')
                time.sleep(loop_sleep_seconds)
                continue

            features = encode_state(raw_state)
            if isinstance(raw_state, GameState) and hasattr(agent, 'predict_state'):
                action = agent.predict_state(raw_state, features)
            else:
                action = agent.predict(features)

            input_controller.apply(action)
            logging.info('Features: %s', features)
            logging.info('Action:   %s', action)
            elapsed = time.perf_counter() - loop_started_at
            time.sleep(max(0.0, loop_sleep_seconds - elapsed))
    finally:
        input_controller.stop_all()
        if gsi_server is not None:
            gsi_server.stop()
        logging.info('All inputs released.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
