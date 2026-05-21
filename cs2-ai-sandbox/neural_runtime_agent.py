from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cs2_ai.ml.models.decision_dqn import DecisionDQN
from cs2_ai.ml.models.movement_gru import MovementGRU
from cs2_ai.ml.utils.torch_utils import get_device, set_seed, torch_available
from cs2_ai.features.feature_contract import validate_checkpoint_schema
from game_state import GameState

ActionDict = dict[str, Any]


class NeuralRuntimeAgent:
    """Legacy feature-vector runtime agent.

    This path is kept for backward compatibility with old single-checkpoint
    experiments. Current supervised training in this project is modular and
    should use FullNeuralRuntimeAgent instead.
    """

    FEATURE_ORDER = [
        'self_x',
        'self_y',
        'self_z',
        'self_vel_x',
        'self_vel_y',
        'self_vel_z',
        'self_hp',
        'self_armor',
        'self_money',
        'self_alive',
        'ammo',
        'ammo_reserve',
        'yaw',
        'pitch',
        'team_is_ct',
        'round_live',
        'round_freeze',
        'round_warmup',
        'bomb_planted',
        'visible_players_count',
        'available_players_count',
        'has_spatial_state',
        'has_enemy_context',
        'enemy_visible',
        'enemy_rel_x',
        'enemy_rel_y',
        'enemy_rel_z',
        'enemy_hp',
        'enemy_distance',
    ]

    def __init__(self, seed: int = 42, mouse_scale: float = 18.0, checkpoint_path: str | None = None) -> None:
        if not torch_available():
            raise RuntimeError('PyTorch is not available. Install torch to use NeuralRuntimeAgent.')

        import torch

        set_seed(seed)
        self.torch = torch
        self.device = get_device()
        self.mouse_scale = mouse_scale
        self.logger = logging.getLogger(__name__)
        self.model = DecisionDQN(input_dim=len(self.FEATURE_ORDER), action_dim=10).to(self.device)
        self.model.eval()
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            self.logger.info('NeuralRuntimeAgent initialized | device=%s | checkpoint=%s | input_dim=%s', self.device, checkpoint_path, len(self.FEATURE_ORDER))
        else:
            self.logger.info('NeuralRuntimeAgent initialized | device=%s | seed=%s | input_dim=%s', self.device, seed, len(self.FEATURE_ORDER))
            self.logger.warning('NeuralRuntimeAgent is a legacy feature-vector path. Use neural-pipeline for current modular checkpoints.')

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_file}')
        checkpoint = self.torch.load(checkpoint_file, map_location=self.device)
        if isinstance(checkpoint, dict):
            model_type = checkpoint.get('model_type')
            input_dim = checkpoint.get('input_dim')
            if model_type in {'movement_bc', 'decision_dqn_movement', 'aim_attention', 'enemy_tracker_lstm'} or input_dim not in {None, len(self.FEATURE_ORDER)}:
                raise ValueError(
                    f'Checkpoint {checkpoint_file} is not compatible with NeuralRuntimeAgent legacy feature-vector mode. '
                    'Use --agent-mode neural-pipeline with modular checkpoints instead.'
                )
        state_dict = checkpoint.get('model_state_dict') if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError(f'Unsupported checkpoint format: {checkpoint_file}')
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, features: dict[str, float | int | bool]) -> ActionDict:
        return self._predict_from_features(features)

    def predict_state(
        self,
        _game_state: GameState,
        features: dict[str, float | int | bool] | None = None,
    ) -> ActionDict:
        return self._predict_from_features(features or {})

    def _predict_from_features(self, features: dict[str, float | int | bool]) -> ActionDict:
        vector = [float(features.get(name, 0.0)) for name in self.FEATURE_ORDER]
        x = self.torch.tensor(vector, dtype=self.torch.float32, device=self.device).unsqueeze(0)

        with self.torch.no_grad():
            logits = self.model(x).squeeze(0)

        move_logits = logits[:4]
        aux_logits = logits[4:8]
        mouse_logits = logits[8:10]

        move_probs = self.torch.softmax(move_logits, dim=0)
        move_idx = int(self.torch.multinomial(move_probs, num_samples=1).item())
        aux_probs = self.torch.sigmoid(aux_logits)
        aux_samples = self.torch.bernoulli(aux_probs).to(dtype=self.torch.int32)
        mouse_values = self.torch.tanh(mouse_logits) * self.mouse_scale

        action: ActionDict = {
            'forward': move_idx == 0,
            'back': move_idx == 1,
            'left': move_idx == 2,
            'right': move_idx == 3,
            'jump': bool(aux_samples[0].item()),
            'crouch': bool(aux_samples[1].item()),
            'walk': bool(aux_samples[2].item()),
            'fire': bool(aux_samples[3].item()),
            'mouse_dx': int(round(float(mouse_values[0].item()))),
            'mouse_dy': int(round(float(mouse_values[1].item()))),
        }

        self.logger.info(
            'NeuralRuntimeAgent outputs | move_probs=%s | aux_probs=%s | mouse=%s',
            [round(float(v), 4) for v in move_probs.detach().cpu().tolist()],
            [round(float(v), 4) for v in aux_probs.detach().cpu().tolist()],
            [round(float(v), 2) for v in mouse_values.detach().cpu().tolist()],
        )
        self.logger.info('NeuralRuntimeAgent action dict: %s', action)
        return action

class FullNeuralRuntimeAgent:
    def __init__(
        self,
        seed: int = 42,
        aim_checkpoint: str | None = None,
        movement_checkpoint: str | None = None,
        tracker_checkpoint: str | None = None,
        yolo_weights: str | None = None,
        window_keywords: tuple[str, ...] = ('counter-strike', 'cs2'),
        show_yolo_overlay: bool = False,
    ) -> None:
        if not torch_available():
            raise RuntimeError('PyTorch is not available.')

        import torch
        from cs2_ai.ml.models.aim_attention import AIM_HEAD_MODE_LEGACY, AimAttentionModel
        from cs2_ai.ml.models.enemy_tracker_lstm import EnemyTrackerLSTM
        from cs2_ai.pipeline.neural_ai_pipeline import NeuralAIPipeline
        from cs2_ai.features.enemy_tracker_features import EnemyTrackerFeatureExtractor
        from cs2_ai.features.movement_features import MovementFeatureExtractor
        from cs2_ai.features.aim_features import AIM_FEATURE_MODE_VISION_LIKE, AimFeatureExtractor, aim_schema_supports_vision_metadata
        from cs2_ai.config import MAX_ENEMIES

        set_seed(seed)
        self.device = get_device()
        self.logger = logging.getLogger(__name__)
        if not aim_checkpoint or not movement_checkpoint or not tracker_checkpoint:
            raise ValueError('neural-pipeline requires --aim-checkpoint, --movement-checkpoint, and --tracker-checkpoint.')
        checkpoints = {
            'aim': self._load_checkpoint(Path(aim_checkpoint), expected_model_type='aim_attention'),
            'movement': self._load_checkpoint(Path(movement_checkpoint), expected_model_type={'decision_dqn_movement', 'movement_gru_chunk', 'movement_gru'}),
            'tracker': self._load_checkpoint(Path(tracker_checkpoint), expected_model_type='enemy_tracker_lstm'),
        }
        if not aim_schema_supports_vision_metadata(checkpoints['aim'].get('feature_schema')):
            raise ValueError('Aim checkpoint was trained without vision features; retrain aim model with vision-enabled schema.')
        seq_lens = {name: int(checkpoint['feature_schema']['seq_len']) for name, checkpoint in checkpoints.items()}
        self.logger.info(
            'Loaded neural pipeline seq_len metadata | aim=%s movement=%s tracker=%s',
            seq_lens['aim'],
            seq_lens['movement'],
            seq_lens['tracker'],
        )
        self.aim_extractor = AimFeatureExtractor(seq_len=seq_lens['aim'], feature_mode=AIM_FEATURE_MODE_VISION_LIKE)
        self.movement_extractor = MovementFeatureExtractor(seq_len=seq_lens['movement'])
        self.tracker_extractor = EnemyTrackerFeatureExtractor(seq_len=seq_lens['tracker'])
        validate_checkpoint_schema(checkpoints['aim'], self.aim_extractor.schema(), str(aim_checkpoint))
        validate_checkpoint_schema(checkpoints['movement'], self.movement_extractor.schema(), str(movement_checkpoint))
        validate_checkpoint_schema(checkpoints['tracker'], self.tracker_extractor.schema(), str(tracker_checkpoint))

        self.aim_model = AimAttentionModel(
            input_dim=self.aim_extractor.feature_dim(),
            head_mode=str(checkpoints['aim'].get('aim_head_mode', AIM_HEAD_MODE_LEGACY)),
        ).to(self.device)
        self.movement_model = self._build_movement_model(checkpoints['movement']).to(self.device)
        self.tracker_model = self._build_tracker_model(checkpoints['tracker'], EnemyTrackerLSTM, MAX_ENEMIES).to(self.device)
        self.aim_model.load_state_dict(checkpoints['aim']['model_state_dict'])
        self.movement_model.load_state_dict(checkpoints['movement']['model_state_dict'])
        self.tracker_model.load_state_dict(checkpoints['tracker']['model_state_dict'])

        self.aim_model.eval()
        self.movement_model.eval()
        self.tracker_model.eval()

        self.pipeline = NeuralAIPipeline(
            self.aim_model,
            self.movement_model,
            self.tracker_model,
            memory_len=max(seq_lens.values()),
            seq_lens=seq_lens,
            device=self.device,
            strict_readiness=True,
        )
        self.runtime_adapter = None
        
        if yolo_weights and Path(yolo_weights).exists():
            from cs2_ai.vision.yolo_pipeline import YoloVisionModule
            self.vision_module = YoloVisionModule(
                Path(yolo_weights),
                window_keywords=window_keywords,
                show_overlay=show_yolo_overlay,
            )
            self.vision_module.start()
        else:
            self.vision_module = None
            if yolo_weights:
                self.logger.warning('YOLO weights path not found or unavailable: %s', yolo_weights)
            else:
                self.logger.warning('YOLO vision disabled: aim runtime will receive vision_target=None and suppress live firing.')
            
        self.logger.info('FullNeuralRuntimeAgent initialized | seq_lens=%s', seq_lens)

    def _load_checkpoint(self, checkpoint_path: Path, expected_model_type: str | set[str]) -> dict[str, Any]:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        import torch

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
            raise ValueError(f'Unsupported checkpoint format: {checkpoint_path}')
        expected_types = {expected_model_type} if isinstance(expected_model_type, str) else set(expected_model_type)
        if checkpoint.get('model_type') not in expected_types:
            raise ValueError(
                f'Checkpoint {checkpoint_path} has model_type={checkpoint.get("model_type")!r}, expected one of {sorted(expected_types)!r}.'
            )
        return checkpoint

    def _build_movement_model(self, checkpoint: dict[str, Any]):
        model_type = str(checkpoint.get('model_type', 'decision_dqn_movement'))
        action_dim = int(checkpoint.get('action_dim', 6))
        if model_type in {'movement_gru_chunk', 'movement_gru'}:
            return MovementGRU(
                input_dim=self.movement_extractor.feature_dim(),
                action_dim=action_dim,
                chunk_len=int(checkpoint.get('chunk_len', 8)),
                hidden_dim=int(checkpoint.get('hidden_dim', 256)),
                num_layers=int(checkpoint.get('num_layers', checkpoint.get('gru_num_layers', 2))),
                dropout=float(checkpoint.get('dropout', checkpoint.get('gru_dropout', 0.1))),
            )
        return DecisionDQN(
            input_dim=self.movement_extractor.feature_dim(),
            action_dim=action_dim,
            hidden_dim=int(checkpoint.get('hidden_dim', 256)),
        )

    def _build_tracker_model(self, checkpoint: dict[str, Any], tracker_cls, max_enemies: int):
        import torch

        model = tracker_cls(
            input_dim=self.tracker_extractor.feature_dim(),
            hidden_dim=int(checkpoint.get('hidden_dim', 128)),
            num_layers=int(checkpoint.get('num_layers', 2)),
            output_enemies=int(checkpoint.get('output_enemies', max_enemies)),
            dropout=float(checkpoint.get('dropout', 0.1)),
            output_mode=str(checkpoint.get('output_mode', 'each_tick')),
        )
        state_dict = checkpoint.get('model_state_dict', {})
        if isinstance(state_dict, dict) and not any(str(key).startswith('shared_head.') for key in state_dict):
            self.logger.warning(
                'Tracker checkpoint appears to be legacy and has no shared_head weights; using Identity shared head for compatibility.'
            )
            model.shared_head = torch.nn.Identity()
        return model

    def predict_state(self, game_state: GameState, _features: dict[str, float | int | bool] | None = None) -> ActionDict:
        from runtime_agent import PipelineRuntimeAgent

        if self.runtime_adapter is None:
            self.runtime_adapter = PipelineRuntimeAgent()

        ai_state = self.runtime_adapter.to_ai_game_state(game_state)
        
        vision_target = None
        if self.vision_module and getattr(self.vision_module, 'is_running', False):
            controlled = game_state.controlled_player
            team_name = (controlled.team or '') if controlled is not None else ''
            self.vision_module.update_context(team_name)
            vision_target = self.vision_module.get_latest_target()
        elif self.vision_module:
            self.logger.debug('Vision module exists but is not running; aim will use vision_target=None.')
            
        action_plan = self.pipeline.step(ai_state, vision_target=vision_target)
        action = self.runtime_adapter.action_plan_to_action_dict(action_plan)
        
        self.logger.info('FullNeuralRuntimeAgent action dict: %s', action)
        return action
