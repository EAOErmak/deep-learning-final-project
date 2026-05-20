from __future__ import annotations

from cs2_ai.modules.action_coordinator import ActionCoordinator
from cs2_ai.modules.buy import RuleBasedBuyModule
from cs2_ai.modules.decision_maker import RuleBasedDecisionMaker
from cs2_ai.modules.input_controller import DryRunInputController
from cs2_ai.schemas.game_state import GameState, GameStateSequence
from cs2_ai.schemas.module_outputs import AimShootOutput, EnemyPrediction, EnemyTrackerOutput, MovementOutput
from cs2_ai.state.belief_state import BeliefState
from cs2_ai.state.memory import TickMemory
from cs2_ai.features.enemy_tracker_features import EnemyTrackerFeatureExtractor
from cs2_ai.features.movement_features import MovementFeatureExtractor
from cs2_ai.features.aim_features import AimFeatureExtractor
from cs2_ai.ml.utils.torch_utils import torch_available

if torch_available():
    import torch

class NeuralAIPipeline:
    def __init__(self, aim_model, movement_model, tracker_model, memory_len: int = 16, device: str = 'cpu'):
        self.aim_model = aim_model
        self.movement_model = movement_model
        self.tracker_model = tracker_model
        self.device = device

        self.memory = TickMemory(max_len=memory_len)
        self.belief_state = BeliefState()
        self.decision_maker = RuleBasedDecisionMaker()
        self.buy_module = RuleBasedBuyModule()
        self.coordinator = ActionCoordinator()
        self.input_controller = DryRunInputController()
        
        self.tracker_extractor = EnemyTrackerFeatureExtractor()
        self.movement_extractor = MovementFeatureExtractor()
        self.aim_extractor = AimFeatureExtractor()
        
        self.last_enemy_tracker_output = None
        self.last_belief_state = None
        self.last_decision_output = None
        self.last_movement_output = None
        self.last_aim_output = None
        self.last_buy_output = None
        self.last_action_plan = None

    def step(self, game_state: GameState):
        self.memory.push(game_state)
        sequence = GameStateSequence(perspective_steamid=game_state.perspective_steamid, states=self.memory.get_sequence())
        
        # 1. Enemy Tracker
        tracker_features = torch.tensor(self.tracker_extractor.extract(sequence), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            positions, confidences = self.tracker_model(tracker_features)
            # positions is [B, SeqLen, Enemies, 3], we only need the last timestep for live inference
            positions = positions[:, -1, :, :].squeeze(0).cpu().numpy()
            confidences = torch.sigmoid(confidences[:, -1, :]).squeeze(0).cpu().numpy()
            
        predictions = []
        for i in range(len(confidences)):
            if confidences[i] > 0.5:
                predictions.append(EnemyPrediction(
                    enemy_slot=i,
                    steamid=None,
                    predicted_position=list(positions[i] * 10000.0), # Denormalize assuming normalization divided by 10000
                    predicted_velocity=[0.0, 0.0, 0.0],
                    confidence=float(confidences[i]),
                    last_seen_seconds=0.0
                ))
        self.last_enemy_tracker_output = EnemyTrackerOutput(predictions=predictions)
        
        # 2. Belief State
        self.last_belief_state = self.belief_state.update(game_state, self.last_enemy_tracker_output)
        
        # 3. Decision Maker
        self.last_decision_output = self.decision_maker.decide(game_state, self.last_belief_state)
        
        # 4. Movement
        movement_features = torch.tensor(self.movement_extractor.extract(sequence, self.last_decision_output, self.last_belief_state), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            movement_logits = self.movement_model(movement_features)
            # movement_logits is [1, SeqLen, 8], we only need the last timestep
            movement_logits = movement_logits[:, -1, :].squeeze(0)
            
        binary_probs = torch.sigmoid(movement_logits[:6])
        move_values = torch.tanh(movement_logits[6:8]) * 450.0 # Denormalize 450 max speed
        
        # parse binary logits [FORWARD, BACK, LEFT, RIGHT, WALK, ducking]
        forward = bool(binary_probs[0] > 0.5)
        back = bool(binary_probs[1] > 0.5)
        left = bool(binary_probs[2] > 0.5)
        right = bool(binary_probs[3] > 0.5)
        walk = bool(binary_probs[4] > 0.5)
        ducking = bool(binary_probs[5] > 0.5)
        
        move_dir = [0.0, 0.0]
        if forward: move_dir[0] += 1.0
        if back: move_dir[0] -= 1.0
        if right: move_dir[1] -= 1.0 # CS2 mapping: right is negative y usually? Or depends on coordinate system.
        if left: move_dir[1] += 1.0
        
        self.last_movement_output = MovementOutput(
            move_direction=move_dir,
            movement_mode="walk" if walk else "run",
            target_position=None,
            should_jump=False,
            should_crouch=ducking
        )
        
        # 5. Aim Shoot
        aim_features = torch.tensor(self.aim_extractor.extract(sequence, self.last_belief_state), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            aim_delta, shoot_logits, rightclick_logits = self.aim_model(aim_features)
            
        aim_delta = aim_delta.squeeze(0).cpu().numpy() * 500.0 # Denormalize
        shoot = bool(torch.sigmoid(shoot_logits).squeeze(0).item() > 0.5)
        rightclick = bool(torch.sigmoid(rightclick_logits).squeeze(0).item() > 0.5)
        
        self.last_aim_output = AimShootOutput(
            aim_delta=list(aim_delta),
            aim_position=None,
            shoot=shoot,
            rightclick=rightclick,
            burst_length=3 if shoot else 0,
            counter_strafe=False,
            confidence=1.0
        )
        
        # 6. Buy and Combine
        self.last_buy_output = self.buy_module.decide(game_state)
        self.last_action_plan = self.coordinator.build_action_plan(
            game_state, self.last_decision_output, self.last_movement_output, self.last_aim_output, self.last_buy_output
        )
        
        self.input_controller.execute(self.last_action_plan)
        return self.last_action_plan
