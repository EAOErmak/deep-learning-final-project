from __future__ import annotations

from typing import Any

from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button, Controller as MouseController


ActionDict = dict[str, Any]


class InputController:
    """
    Executes keyboard and mouse actions.

    The agent never touches OS input directly. It only emits action
    dictionaries, and this controller translates them into safe,
    standard keyboard and mouse events.
    """

    def __init__(self) -> None:
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self._pressed_keys: set[str] = set()
        self._mouse_left_down = False

        self._keymap: dict[str, str | Key] = {
            "forward": "w",
            "back": "s",
            "left": "a",
            "right": "d",
            "crouch": Key.ctrl_l,
            "walk": Key.shift_l,
        }

    def move_forward_start(self) -> None:
        self._press_key("forward")

    def move_forward_stop(self) -> None:
        self._release_key("forward")

    def move_back_start(self) -> None:
        self._press_key("back")

    def move_back_stop(self) -> None:
        self._release_key("back")

    def move_left_start(self) -> None:
        self._press_key("left")

    def move_left_stop(self) -> None:
        self._release_key("left")

    def move_right_start(self) -> None:
        self._press_key("right")

    def move_right_stop(self) -> None:
        self._release_key("right")

    def jump(self) -> None:
        self.keyboard.press(Key.space)
        self.keyboard.release(Key.space)

    def crouch_start(self) -> None:
        self._press_key("crouch")

    def crouch_stop(self) -> None:
        self._release_key("crouch")

    def walk_start(self) -> None:
        self._press_key("walk")

    def walk_stop(self) -> None:
        self._release_key("walk")

    def fire_start(self) -> None:
        if not self._mouse_left_down:
            self.mouse.press(Button.left)
            self._mouse_left_down = True

    def fire_stop(self) -> None:
        if self._mouse_left_down:
            self.mouse.release(Button.left)
            self._mouse_left_down = False

    def mouse_move(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        self.mouse.move(dx, dy)

    def stop_all(self) -> None:
        for action_name in ("forward", "back", "left", "right", "crouch", "walk"):
            self._release_key(action_name)
        self.fire_stop()

    def apply(self, action: ActionDict) -> None:
        self._apply_hold_action("forward", action.get("forward", False))
        self._apply_hold_action("back", action.get("back", False))
        self._apply_hold_action("left", action.get("left", False))
        self._apply_hold_action("right", action.get("right", False))
        self._apply_hold_action("crouch", action.get("crouch", False))
        self._apply_hold_action("walk", action.get("walk", False))

        if action.get("jump", False):
            self.jump()

        if action.get("fire", False):
            self.fire_start()
        else:
            self.fire_stop()

        dx = int(action.get("mouse_dx", 0))
        dy = int(action.get("mouse_dy", 0))
        self.mouse_move(dx, dy)

    def _apply_hold_action(self, action_name: str, is_active: bool) -> None:
        if is_active:
            self._press_key(action_name)
        else:
            self._release_key(action_name)

    def _press_key(self, action_name: str) -> None:
        if action_name in self._pressed_keys:
            return

        key = self._resolve_key(self._keymap[action_name])
        self.keyboard.press(key)
        self._pressed_keys.add(action_name)

    def _release_key(self, action_name: str) -> None:
        if action_name not in self._pressed_keys:
            return

        key = self._resolve_key(self._keymap[action_name])
        self.keyboard.release(key)
        self._pressed_keys.discard(action_name)

    @staticmethod
    def _resolve_key(key: str | Key) -> Key | KeyCode:
        if isinstance(key, Key):
            return key
        return KeyCode.from_char(key)
