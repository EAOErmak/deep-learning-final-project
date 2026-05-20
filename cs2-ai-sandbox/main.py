from __future__ import annotations

from pathlib import Path

from project_cache import configure_project_pycache

configure_project_pycache(Path(__file__).resolve().parent)

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
from neural_runtime_agent import NeuralRuntimeAgent
from runtime_agent import PipelineRuntimeAgent
from state_reader import StateReader


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CS2 AI sandbox runtime')
    parser.add_argument('--state-source', choices=['mock', 'gsi'], default='mock')
    parser.add_argument('--agent-mode', choices=['auto', 'dummy', 'pipeline', 'neural-random', 'neural-checkpoint', 'neural-pipeline'], default='auto')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--aim-checkpoint', type=str, default=None)
    parser.add_argument('--movement-checkpoint', type=str, default=None)
    parser.add_argument('--tracker-checkpoint', type=str, default=None)
    parser.add_argument('--gsi-port', type=int, default=3000)
    parser.add_argument('--hz', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--disable-window-guard', action='store_true')
    parser.add_argument('--window-keyword', action='append', default=None)
    parser.add_argument('--min-live-readiness', choices=['basic', 'spatial', 'observer'], default='basic')
    parser.add_argument(
        '--allow-basic-neural',
        action='store_true',
        help='Allow trained neural agents to run on basic GSI without spatial fields. Use only for debugging.',
    )
    return parser.parse_args()


def resolve_agent_mode(state_source: str, agent_mode: str) -> str:
    if agent_mode == 'auto':
        return 'pipeline' if state_source == 'gsi' else 'dummy'
    return agent_mode


def resolve_min_live_readiness(args: argparse.Namespace, resolved_agent_mode: str) -> str:
    if (
        args.state_source == 'gsi'
        and resolved_agent_mode in {'neural-checkpoint', 'neural-pipeline'}
        and args.min_live_readiness == 'basic'
        and not args.allow_basic_neural
    ):
        logging.warning(
            'Promoting min_live_readiness from basic to spatial for trained neural mode. '
            'Use --allow-basic-neural only for debugging with incomplete GSI payloads.'
        )
        return 'spatial'
    return args.min_live_readiness


def build_agent(state_source: str, agent_mode: str, seed: int, args: argparse.Namespace) -> Any:
    resolved_mode = resolve_agent_mode(state_source, agent_mode)

    if resolved_mode == 'dummy':
        logging.info('Using DummyAgent fallback.')
        return DummyAgent()
    if resolved_mode == 'pipeline':
        logging.info('Using PipelineRuntimeAgent for live/runtime sandbox mode.')
        return PipelineRuntimeAgent()
    if resolved_mode == 'neural-random':
        logging.info('Using NeuralRuntimeAgent with random untrained PyTorch weights.')
        return NeuralRuntimeAgent(seed=seed)
    if resolved_mode == 'neural-checkpoint':
        checkpoint = args.checkpoint
        if not checkpoint:
            raise ValueError('--checkpoint is required for --agent-mode neural-checkpoint')
        logging.info('Using NeuralRuntimeAgent with trained checkpoint.')
        return NeuralRuntimeAgent(seed=seed, checkpoint_path=checkpoint)
    if resolved_mode == 'neural-pipeline':
        logging.info('Using FullNeuralRuntimeAgent with modular NeuralAIPipeline.')
        from neural_runtime_agent import FullNeuralRuntimeAgent
        return FullNeuralRuntimeAgent(
            seed=seed, 
            aim_checkpoint=args.aim_checkpoint, 
            movement_checkpoint=args.movement_checkpoint, 
            tracker_checkpoint=args.tracker_checkpoint
        )
    raise ValueError(f'Unsupported agent mode: {agent_mode}')


def is_live_runtime_ready(game_state: GameState, min_readiness: str) -> tuple[bool, str]:
    player = game_state.controlled_player
    if player is None:
        return False, 'controlled_player missing'
    if (player.activity or '').lower() != 'playing':
        return False, f'player activity is {(player.activity or "unknown")!r}, not playing'
    if min_readiness == 'basic':
        return True, 'basic live state available'
    if min_readiness == 'spatial':
        if not game_state.capabilities.has_spatial_state:
            return False, 'spatial state missing: no player.position/player.forward'
        return True, 'spatial live state available'
    if min_readiness == 'observer':
        if not game_state.capabilities.has_spatial_state:
            return False, 'spatial state missing: no player.position/player.forward'
        if not game_state.capabilities.has_allplayers:
            return False, 'observer context missing: no allplayers'
        return True, 'observer-grade live state available'
    return False, f'unsupported readiness mode: {min_readiness}'


def main() -> int:
    configure_logging()
    args = parse_args()
    resolved_agent_mode = resolve_agent_mode(args.state_source, args.agent_mode)
    min_live_readiness = resolve_min_live_readiness(args, resolved_agent_mode)
    loop_sleep_seconds = 1.0 / max(args.hz, 0.1)
    running = True
    gsi_server: GSIServer | None = None
    last_readiness_log_at = 0.0

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
    agent = build_agent(args.state_source, resolved_agent_mode, args.seed, args)
    window_keywords = tuple(args.window_keyword) if args.window_keyword else ('counter-strike', 'cs2')
    input_controller = InputController(
        window_guard_enabled=not args.disable_window_guard,
        allowed_window_keywords=window_keywords,
    )

    logging.info(
        'CS2 AI sandbox started | state_source=%s | agent_mode=%s | hz=%.2f | window_guard=%s | window_keywords=%s | min_live_readiness=%s',
        args.state_source,
        resolved_agent_mode,
        args.hz,
        not args.disable_window_guard,
        window_keywords,
        min_live_readiness,
    )

    try:
        while running:
            loop_started_at = time.perf_counter()
            raw_state = state_reader.read_state()
            if raw_state is None:
                logging.info('Waiting for GSI payload...')
                time.sleep(loop_sleep_seconds)
                continue

            if isinstance(raw_state, GameState) and args.state_source == 'gsi':
                ready, reason = is_live_runtime_ready(raw_state, min_live_readiness)
                if not ready:
                    input_controller.stop_all()
                    now = time.monotonic()
                    if now - last_readiness_log_at >= 1.0:
                        logging.info('Live runtime not ready | reason=%s | capabilities=%s', reason, raw_state.capabilities)
                        last_readiness_log_at = now
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

