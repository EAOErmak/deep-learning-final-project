from __future__ import annotations

import numpy as np
import pandas as pd

from cs2_ai.config import MAX_TEAMMATES, MONEY_SCALE, weapon_to_id
from cs2_ai.state.game_state_builder import GameStateBuilder


class BuyFeatureExtractor:
    FEATURE_NAMES = [
        'self_money',
        'self_armor',
        'self_helmet',
        'self_weapon_id',
        'team_money_0',
        'team_money_1',
        'team_money_2',
        'team_money_3',
        'team_money_4',
        'team_armor_0',
        'team_armor_1',
        'team_armor_2',
        'team_armor_3',
        'team_armor_4',
        'team_helmet_0',
        'team_helmet_1',
        'team_helmet_2',
        'team_helmet_3',
        'team_helmet_4',
        'round_number',
        'losing_streak',
        'in_buy_zone',
        'team_is_ct',
    ]

    def __init__(self) -> None:
        self.game_state_builder = GameStateBuilder()

    def feature_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def feature_names(self) -> list[str]:
        return list(self.FEATURE_NAMES)

    def extract_from_tick_rows(self, tick_rows: pd.DataFrame, perspective_steamid: int) -> np.ndarray:
        game_state = self.game_state_builder.build_from_tick_rows(tick_rows, perspective_steamid)
        return self.extract_from_game_state(game_state)

    def extract_from_game_state(self, game_state) -> np.ndarray:
        self_player = game_state.self_player
        team_players = [game_state.self_player, *game_state.teammates]
        team_money = [player.money / MONEY_SCALE for player in team_players]
        team_armor = [player.armor / 100.0 for player in team_players]
        team_helmets = [1.0 if player.has_helmet else 0.0 for player in team_players]
        while len(team_money) < MAX_TEAMMATES + 1:
            team_money.append(0.0)
            team_armor.append(0.0)
            team_helmets.append(0.0)
        losing_streak = game_state.round.ct_losing_streak if self_player.team_num == 3 else game_state.round.t_losing_streak
        values = [
            self_player.money / MONEY_SCALE,
            self_player.armor / 100.0,
            1.0 if self_player.has_helmet else 0.0,
            weapon_to_id(self_player.weapon) / 17.0,
            *team_money[:5],
            *team_armor[:5],
            *team_helmets[:5],
            float(game_state.round.round_number) / 30.0,
            float(losing_streak) / 10.0,
            1.0 if self_player.in_buy_zone else 0.0,
            1.0 if self_player.team_num == 3 else 0.0,
        ]
        return np.asarray(values, dtype=np.float32)


def build_buy_target_from_freeze_sequence(player_rows: pd.DataFrame) -> dict:
    if player_rows.empty:
        return {'final_weapon_after_freeze': 'none', 'money_spent': 0.0, 'buy_type': 'eco'}
    sorted_rows = player_rows.sort_values('tick')
    start_balance = float(sorted_rows.iloc[0].get('balance', 0.0))
    end_balance = float(sorted_rows.iloc[-1].get('balance', 0.0))
    money_spent = max(0.0, start_balance - end_balance)
    final_weapon = str(sorted_rows.iloc[-1].get('active_weapon_name', 'none') or 'none')
    if final_weapon == 'AWP':
        buy_type = 'awp'
    elif money_spent >= 4500:
        buy_type = 'full'
    elif money_spent >= 2700:
        buy_type = 'half'
    elif money_spent >= 1500:
        buy_type = 'force'
    else:
        buy_type = 'eco'
    return {'final_weapon_after_freeze': final_weapon, 'money_spent': money_spent, 'buy_type': buy_type}