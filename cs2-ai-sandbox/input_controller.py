from __future__ import annotations

import ctypes
import logging
import sys
import time
from typing import Any

from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button, Controller as MouseController


ActionDict = dict[str, Any]

if sys.platform.startswith('win'):
    user32 = ctypes.windll.user32
    INPUT_MOUSE = 0
    INPUT_KEYBOARD = 1
    KEYEVENTF_SCANCODE = 0x0008
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_EXTENDEDKEY = 0x0001
    MOUSEEVENTF_MOVE = 0x0001
    MAPVK_VK_TO_VSC = 0

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ('dx', ctypes.c_long),
            ('dy', ctypes.c_long),
            ('mouseData', ctypes.c_ulong),
            ('dwFlags', ctypes.c_ulong),
            ('time', ctypes.c_ulong),
            ('dwExtraInfo', ctypes.POINTER(ctypes.c_ulong)),
        ]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ('wVk', ctypes.c_ushort),
            ('wScan', ctypes.c_ushort),
            ('dwFlags', ctypes.c_ulong),
            ('time', ctypes.c_ulong),
            ('dwExtraInfo', ctypes.POINTER(ctypes.c_ulong)),
        ]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [
            ('mi', MOUSEINPUT),
            ('ki', KEYBDINPUT),
        ]

    class INPUT(ctypes.Structure):
        _anonymous_ = ('u',)
        _fields_ = [
            ('type', ctypes.c_ulong),
            ('u', _INPUTUNION),
        ]


class InputController:
    """
    Executes keyboard and mouse actions.

    The agent never touches OS input directly. It only emits action
    dictionaries, and this controller translates them into safe,
    standard keyboard and mouse events.
    """

    def __init__(
        self,
        window_guard_enabled: bool = True,
        allowed_window_keywords: tuple[str, ...] = ('counter-strike', 'cs2'),
        backend: str = 'auto',
    ) -> None:
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self._pressed_keys: set[str] = set()
        self._mouse_left_down = False
        self._window_guard_enabled = window_guard_enabled and sys.platform.startswith('win')
        self._allowed_window_keywords = tuple(keyword.lower() for keyword in allowed_window_keywords)
        self._logger = logging.getLogger(__name__)
        self._last_guard_log_at = 0.0
        self._backend = self._resolve_backend(backend)

        self._keymap: dict[str, str | Key] = {
            'forward': 'w',
            'back': 's',
            'left': 'a',
            'right': 'd',
            'crouch': Key.ctrl_l,
            'walk': Key.shift_l,
        }
        self._vk_keymap: dict[str, int] = {
            'forward': 0x57,
            'back': 0x53,
            'left': 0x41,
            'right': 0x44,
            'crouch': 0xA2,
            'walk': 0xA0,
            'jump': 0x20,
        }
        self._logger.info('InputController initialized | backend=%s | window_guard=%s', self._backend, self._window_guard_enabled)

    def move_forward_start(self) -> None:
        self._press_key('forward')

    def move_forward_stop(self) -> None:
        self._release_key('forward')

    def move_back_start(self) -> None:
        self._press_key('back')

    def move_back_stop(self) -> None:
        self._release_key('back')

    def move_left_start(self) -> None:
        self._press_key('left')

    def move_left_stop(self) -> None:
        self._release_key('left')

    def move_right_start(self) -> None:
        self._press_key('right')

    def move_right_stop(self) -> None:
        self._release_key('right')

    def jump(self) -> None:
        if self._backend == 'sendinput':
            self._send_key_event(self._vk_keymap['jump'], is_key_up=False)
            self._send_key_event(self._vk_keymap['jump'], is_key_up=True)
            return
        self.keyboard.press(Key.space)
        self.keyboard.release(Key.space)

    def crouch_start(self) -> None:
        self._press_key('crouch')

    def crouch_stop(self) -> None:
        self._release_key('crouch')

    def walk_start(self) -> None:
        self._press_key('walk')

    def walk_stop(self) -> None:
        self._release_key('walk')

    def fire_start(self) -> None:
        if not self._mouse_left_down:
            if self._backend == 'sendinput':
                self.mouse.press(Button.left)
            else:
                self.mouse.press(Button.left)
            self._mouse_left_down = True

    def fire_stop(self) -> None:
        if self._mouse_left_down:
            if self._backend == 'sendinput':
                self.mouse.release(Button.left)
            else:
                self.mouse.release(Button.left)
            self._mouse_left_down = False

    def mouse_move(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        if self._backend == 'sendinput':
            self._send_mouse_move(dx, dy)
            return
        self.mouse.move(dx, dy)

    def stop_all(self) -> None:
        for action_name in ('forward', 'back', 'left', 'right', 'crouch', 'walk'):
            self._release_key(action_name)
        self.fire_stop()

    def apply(self, action: ActionDict) -> None:
        if self._window_guard_enabled and not self._is_allowed_foreground_window():
            self.stop_all()
            self._log_window_guard_skip()
            return

        self._apply_hold_action('forward', action.get('forward', False))
        self._apply_hold_action('back', action.get('back', False))
        self._apply_hold_action('left', action.get('left', False))
        self._apply_hold_action('right', action.get('right', False))
        self._apply_hold_action('crouch', action.get('crouch', False))
        self._apply_hold_action('walk', action.get('walk', False))

        if action.get('jump', False):
            self.jump()

        if action.get('fire', False):
            self.fire_start()
        else:
            self.fire_stop()

        dx = int(action.get('mouse_dx', 0))
        dy = int(action.get('mouse_dy', 0))
        self.mouse_move(dx, dy)

    def _apply_hold_action(self, action_name: str, is_active: bool) -> None:
        if is_active:
            self._press_key(action_name)
        else:
            self._release_key(action_name)

    def _press_key(self, action_name: str) -> None:
        if action_name in self._pressed_keys:
            return

        if self._backend == 'sendinput':
            self._send_key_event(self._vk_keymap[action_name], is_key_up=False)
            self._pressed_keys.add(action_name)
            return

        key = self._resolve_key(self._keymap[action_name])
        self.keyboard.press(key)
        self._pressed_keys.add(action_name)

    def _release_key(self, action_name: str) -> None:
        if action_name not in self._pressed_keys:
            return

        if self._backend == 'sendinput':
            self._send_key_event(self._vk_keymap[action_name], is_key_up=True)
            self._pressed_keys.discard(action_name)
            return

        key = self._resolve_key(self._keymap[action_name])
        self.keyboard.release(key)
        self._pressed_keys.discard(action_name)

    def _is_allowed_foreground_window(self) -> bool:
        title = self.get_active_window_title().lower()
        return bool(title) and any(keyword in title for keyword in self._allowed_window_keywords)

    def _log_window_guard_skip(self) -> None:
        now = time.monotonic()
        if now - self._last_guard_log_at < 1.0:
            return
        self._last_guard_log_at = now
        self._logger.info(
            'Window guard blocked input | active_window=%r | allowed_keywords=%s',
            self.get_active_window_title(),
            self._allowed_window_keywords,
        )

    @staticmethod
    def get_active_window_title() -> str:
        if not sys.platform.startswith('win'):
            return ''
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ''
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ''
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value

    @staticmethod
    def _resolve_key(key: str | Key) -> Key | KeyCode:
        if isinstance(key, Key):
            return key
        return KeyCode.from_char(key)

    def _resolve_backend(self, backend: str) -> str:
        normalized = backend.lower()
        if normalized not in {'auto', 'pynput', 'sendinput'}:
            raise ValueError(f'Unsupported input backend: {backend}')
        if normalized == 'auto':
            return 'sendinput' if sys.platform.startswith('win') else 'pynput'
        if normalized == 'sendinput' and not sys.platform.startswith('win'):
            self._logger.warning('sendinput backend requested on non-Windows platform; falling back to pynput.')
            return 'pynput'
        return normalized

    def _send_key_event(self, virtual_key: int, *, is_key_up: bool) -> None:
        if not sys.platform.startswith('win'):
            return
        scan_code = user32.MapVirtualKeyW(virtual_key, MAPVK_VK_TO_VSC)
        flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if is_key_up else 0)
        if virtual_key in {0xA0, 0xA1, 0xA2, 0xA3, 0x25, 0x26, 0x27, 0x28}:
            flags |= KEYEVENTF_EXTENDEDKEY
        event = INPUT(
            type=INPUT_KEYBOARD,
            ki=KEYBDINPUT(
                wVk=0,
                wScan=scan_code,
                dwFlags=flags,
                time=0,
                dwExtraInfo=None,
            ),
        )
        user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))

    def _send_mouse_move(self, dx: int, dy: int) -> None:
        if not sys.platform.startswith('win'):
            return
        event = INPUT(
            type=INPUT_MOUSE,
            mi=MOUSEINPUT(
                dx=dx,
                dy=dy,
                mouseData=0,
                dwFlags=MOUSEEVENTF_MOVE,
                time=0,
                dwExtraInfo=None,
            ),
        )
        user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))
