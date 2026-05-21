from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    def as_mss_bbox(self) -> dict[str, int]:
        return {
            'left': self.left,
            'top': self.top,
            'width': self.width,
            'height': self.height,
        }


class WindowCaptureLocator:
    def __init__(self, *, window_keywords: tuple[str, ...] = ('counter-strike', 'cs2')) -> None:
        self.logger = logging.getLogger(__name__)
        self.window_keywords = tuple(keyword.lower() for keyword in window_keywords if keyword)
        self._warned_unavailable = False
        self._warned_not_found = False

    def find_client_region(self) -> CaptureRegion | None:
        if not sys.platform.startswith('win'):
            self._warn_unavailable_once('Window capture is only implemented on Windows; falling back to screen coordinates.')
            return None

        user32 = ctypes.windll.user32
        hwnd = self._find_matching_window(user32)
        if not hwnd:
            self._warn_not_found_once()
            return None

        rect = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            self.logger.warning('GetClientRect failed for matching CS2 window; falling back to screen coordinates.')
            return None

        origin = wintypes.POINT(0, 0)
        if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            self.logger.warning('ClientToScreen failed for matching CS2 window; falling back to screen coordinates.')
            return None

        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width <= 0 or height <= 0:
            self.logger.warning('Matching CS2 window has non-positive client area; falling back to screen coordinates.')
            return None

        self._warned_not_found = False
        return CaptureRegion(left=int(origin.x), top=int(origin.y), width=width, height=height)

    def _find_matching_window(self, user32) -> int:
        if not self.window_keywords:
            return 0

        matches: list[tuple[int, int]] = []

        enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

        def callback(hwnd: int, _lparam: int) -> bool:
            if not user32.IsWindowVisible(hwnd):
                return True

            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True

            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value.strip()
            if not title:
                return True

            title_l = title.lower()
            if not any(keyword in title_l for keyword in self.window_keywords):
                return True

            rect = wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return True

            area = max(0, int(rect.right - rect.left)) * max(0, int(rect.bottom - rect.top))
            if area <= 0:
                return True

            matches.append((hwnd, area))
            return True

        user32.EnumWindows(enum_proc(callback), 0)
        if not matches:
            return 0

        matches.sort(key=lambda item: item[1], reverse=True)
        return int(matches[0][0])

    def _warn_unavailable_once(self, message: str) -> None:
        if self._warned_unavailable:
            return
        self._warned_unavailable = True
        self.logger.warning(message)

    def _warn_not_found_once(self) -> None:
        if self._warned_not_found:
            return
        self._warned_not_found = True
        self.logger.warning(
            'No visible window matched keywords %s; falling back to screen coordinates.',
            self.window_keywords,
        )
