from __future__ import annotations

from dataclasses import asdict
from pprint import pprint

from cs2_ai.schemas.module_outputs import ActionPlan


class DryRunInputController:
    def execute(self, action_plan: ActionPlan) -> None:
        print("DryRunInputController.execute")
        pprint({
            "duration_ms": action_plan.duration_ms,
            "keyboard_inputs": [asdict(command) for command in action_plan.keyboard_inputs],
            "mouse_inputs": [asdict(command) for command in action_plan.mouse_inputs],
        })
