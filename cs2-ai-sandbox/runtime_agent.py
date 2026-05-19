from __future__ import annotations

import logging
from typing import Any

from cs2_ai.config import weapon_to_id
from cs2_ai.pipeline.offline_ai_pipeline import OfflineAIPipeline
from cs2_ai.schemas.game_state import BombState as AIBombState
from cs2_ai.schemas.game_state import GameState as AIGameState
from cs2_ai.schemas.game_state import PlayerInputState as AIPlayerInputState
from cs2_ai.schemas.game_state import PlayerState as AIPlayerState
from cs2_ai.schemas.game_state import RoundState as AIRoundState
from cs2_ai.schemas.module_outputs import ActionPlan
from game_state import GameState, PlayerState

ActionDict = dict[str, Any]


class PipelineRuntimeAgent:
    def __init__(self) -> None:
        self.pipeline = OfflineAIPipeline(memory_len=16)
        self.logger = logging.getLogger(__name__)

    def predict_state(self, game_state: GameState, _features: dict[str, float | int | bool] | None = None) -> ActionDict:
        ai_state = self._to_ai_game_state(game_state)
        action_plan = self.pipeline.step(ai_state)
        action = self._action_plan_to_action_dict(action_plan)
        self.logger.info('PipelineRuntimeAgent action dict: %s', action)
        return action

    def _to_ai_game_state(self, game_state: GameState) -> AIGameState:
        controlled = game_state.controlled_player
        if controlled is None:
            raise ValueError('PipelineRuntimeAgent requires a controlled_player in live GameState.')

        self_player = self._to_ai_player(controlled)
        teammates: list[AIPlayerState] = []
        enemies: list[AIPlayerState] = []
        self_team_num = self_player.team_num

        for player in game_state.players:
            if player.id == controlled.id:
                continue
            ai_player = self._to_ai_player(player)
            if ai_player.team_num == self_team_num:
                teammates.append(ai_player)
            else:
                enemies.append(ai_player)

        phase = (game_state.round_state.phase or game_state.map_state.phase or '').lower()
        round_state = AIRoundState(
            tick=int(game_state.timestamp * 64) if game_state.timestamp else 0,
            round_number=int(game_state.map_state.round_number or 0),
            round_start_time=0.0,
            round_in_progress=phase in {'live', 'over', 'bomb'} or phase == '',
            is_freeze_period=phase in {'freezetime', 'freeze'},
            is_warmup_period=phase == 'warmup',
            game_phase=0,
            round_win_status=0,
            round_win_reason=0,
            ct_losing_streak=int(game_state.map_state.ct_consecutive_round_losses or 0),
            t_losing_streak=int(game_state.map_state.t_consecutive_round_losses or 0),
        )
        bomb_state = AIBombState(
            is_bomb_planted=(game_state.round_state.bomb_state or '').lower() == 'planted',
            is_bomb_dropped=False,
            bomb_position=None,
        )
        return AIGameState(
            tick=round_state.tick,
            perspective_steamid=self_player.steamid,
            self_player=self_player,
            self_input=AIPlayerInputState(
                forward=False,
                back=False,
                left=False,
                right=False,
                fire=False,
                rightclick=False,
                reload=False,
                use=False,
                zoom=False,
                walk=False,
                usercmd_mouse_dx=0.0,
                usercmd_mouse_dy=0.0,
                usercmd_forward_move=0.0,
                usercmd_left_move=0.0,
            ),
            teammates=teammates,
            enemies=enemies,
            round=round_state,
            bomb=bomb_state,
        )

    def _to_ai_player(self, player: PlayerState) -> AIPlayerState:
        team_num = self._team_to_num(player.team)
        position = [player.position.x, player.position.y, player.position.z] if player.position is not None else [0.0, 0.0, 0.0]
        forward = [player.forward.x, player.forward.y, player.forward.z] if player.forward is not None else [0.0, 0.0, 0.0]
        velocity = [player.velocity.x, player.velocity.y, player.velocity.z] if player.velocity is not None else [0.0, 0.0, 0.0]
        return AIPlayerState(
            steamid=self._safe_steamid(player.id),
            name=player.name or 'unknown',
            team_num=team_num,
            position=position,
            velocity=velocity,
            health=float(player.health or 0),
            armor=float(player.armor or 0),
            has_helmet=bool(player.helmet),
            is_alive=bool(player.is_alive),
            money=float(player.money or 0),
            weapon=player.weapon or 'none',
            weapon_id=weapon_to_id(player.weapon),
            ammo=float(player.ammo or 0),
            total_ammo=float((player.ammo or 0) + (player.ammo_reserve or 0)),
            pitch=self._approximate_pitch(forward),
            yaw=self._approximate_yaw(forward),
            is_scoped=False,
            is_walking=False,
            is_airborne=False,
            duck_amount=0.0,
            ducking=False,
            shots_fired=0,
            flash_duration=float(player.flashed or 0),
            spotted=True,
            last_place_name='unknown',
            in_bomb_zone=False,
            in_buy_zone=False,
            which_bomb_zone=0,
        )

    def _action_plan_to_action_dict(self, action_plan: ActionPlan) -> ActionDict:
        action: ActionDict = {
            'forward': False,
            'back': False,
            'left': False,
            'right': False,
            'jump': False,
            'crouch': False,
            'walk': False,
            'fire': False,
            'mouse_dx': 0,
            'mouse_dy': 0,
        }
        for command in action_plan.keyboard_inputs:
            if command.command == 'W':
                action['forward'] = True
            elif command.command == 'S':
                action['back'] = True
            elif command.command == 'A':
                action['left'] = True
            elif command.command == 'D':
                action['right'] = True
            elif command.command == 'SHIFT':
                action['walk'] = True
            elif command.command == 'CTRL':
                action['crouch'] = True
        for command in action_plan.mouse_inputs:
            if command.command == 'mouse_move_yaw':
                action['mouse_dx'] = int(round(float(command.value)))
            elif command.command == 'mouse_move_pitch':
                action['mouse_dy'] = int(round(float(command.value)))
            elif command.command == 'mouse_left':
                action['fire'] = True
        return action

    def _team_to_num(self, team: str | None) -> int:
        if not team:
            return 0
        team_upper = team.upper()
        if team_upper in {'CT', 'COUNTERTERRORIST'}:
            return 3
        if team_upper in {'T', 'TERRORIST'}:
            return 2
        return 0

    def _safe_steamid(self, player_id: str) -> int:
        try:
            return int(str(player_id))
        except (TypeError, ValueError):
            return 0

    def _approximate_yaw(self, forward: list[float]) -> float:
        x, y, _ = forward
        if x == 0.0 and y == 0.0:
            return 0.0
        import math
        return math.degrees(math.atan2(y, x))

    def _approximate_pitch(self, forward: list[float]) -> float:
        x, y, z = forward
        import math
        horizontal = math.sqrt(x * x + y * y)
        if horizontal == 0.0 and z == 0.0:
            return 0.0
        return math.degrees(math.atan2(z, horizontal))
