from __future__ import annotations

from cs2_ai.schemas.game_state import GameState
from cs2_ai.schemas.module_outputs import BuyOutput


class RuleBasedBuyModule:
    def reset(self) -> None:
        return None

    def decide(self, game_state: GameState) -> BuyOutput:
        if not game_state.round.is_freeze_period:
            return BuyOutput(should_buy=False, buy_type="none", buy_list=[])
        if not game_state.self_player.in_buy_zone:
            return BuyOutput(should_buy=False, buy_type="none", buy_list=[])
        money = game_state.self_player.money
        if money >= 4750:
            primary = "AWP" if money >= 6500 else ("M4A1-S" if game_state.self_player.team_num == 3 else "AK-47")
            return BuyOutput(True, "full_buy", [primary, "kevlar", "helmet", "flashbang"])
        if money >= 2700:
            return BuyOutput(True, "half_buy", ["MP9" if game_state.self_player.team_num == 3 else "MAC-10", "kevlar"])
        if money >= 1500:
            return BuyOutput(True, "force_buy", ["P250", "kevlar"])
        return BuyOutput(True, "eco", [])
