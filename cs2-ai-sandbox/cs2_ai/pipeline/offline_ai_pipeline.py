from __future__ import annotations

from cs2_ai.modules.action_coordinator import ActionCoordinator
from cs2_ai.modules.aim_shoot import SimpleAimShootModule
from cs2_ai.modules.buy import RuleBasedBuyModule
from cs2_ai.modules.decision_maker import RuleBasedDecisionMaker
from cs2_ai.modules.enemy_tracker import RuleBasedEnemyTracker
from cs2_ai.modules.input_controller import DryRunInputController
from cs2_ai.modules.movement import SimpleMovementModule
from cs2_ai.schemas.game_state import GameState, GameStateSequence
from cs2_ai.state.belief_state import BeliefState
from cs2_ai.state.memory import TickMemory


class OfflineAIPipeline:
    def __init__(self, memory_len: int = 128):
        self.memory = TickMemory(max_len=memory_len)
        self.enemy_tracker = RuleBasedEnemyTracker()
        self.belief_state = BeliefState()
        self.decision_maker = RuleBasedDecisionMaker()
        self.movement_module = SimpleMovementModule()
        self.aim_module = SimpleAimShootModule()
        self.buy_module = RuleBasedBuyModule()
        self.coordinator = ActionCoordinator()
        self.input_controller = DryRunInputController()
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
        self.last_enemy_tracker_output = self.enemy_tracker.predict(sequence)
        self.last_belief_state = self.belief_state.update(game_state, self.last_enemy_tracker_output)
        self.last_decision_output = self.decision_maker.decide(game_state, self.last_belief_state)
        self.last_movement_output = self.movement_module.decide(game_state, self.last_belief_state, self.last_decision_output)
        self.last_aim_output = self.aim_module.decide(game_state, self.last_belief_state, self.last_decision_output)
        self.last_buy_output = self.buy_module.decide(game_state)
        self.last_action_plan = self.coordinator.build_action_plan(game_state, self.last_decision_output, self.last_movement_output, self.last_aim_output, self.last_buy_output)
        self.input_controller.execute(self.last_action_plan)
        return self.last_action_plan
