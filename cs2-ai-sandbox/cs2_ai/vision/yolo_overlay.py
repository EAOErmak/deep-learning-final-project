from __future__ import annotations

import ctypes
import logging
import sys
import threading
from ctypes import wintypes
from dataclasses import dataclass

from cs2_ai.vision.window_capture import CaptureRegion


@dataclass(frozen=True, slots=True)
class OverlayDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float
    is_enemy: bool
    is_head: bool
    is_selected: bool = False


if sys.platform.startswith('win'):
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    kernel32 = ctypes.windll.kernel32
    HANDLE = wintypes.HANDLE
    HICON = HANDLE
    HCURSOR = HANDLE
    HBRUSH = HANDLE

    WM_DESTROY = 0x0002
    WM_PAINT = 0x000F
    WM_ERASEBKGND = 0x0014
    WM_APP_UPDATE = 0x8000 + 1

    WS_POPUP = 0x80000000
    WS_VISIBLE = 0x10000000

    WS_EX_LAYERED = 0x00080000
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_TOPMOST = 0x00000008
    WS_EX_TOOLWINDOW = 0x00000080
    WS_EX_NOACTIVATE = 0x08000000

    SW_HIDE = 0
    SW_SHOWNA = 8
    SWP_NOACTIVATE = 0x0010
    SWP_SHOWWINDOW = 0x0040

    LWA_COLORKEY = 0x00000001

    PS_SOLID = 0
    TRANSPARENT = 1

    COLORKEY = 0x00FF00FF  # magenta in COLORREF

    class PAINTSTRUCT(ctypes.Structure):
        _fields_ = [
            ('hdc', wintypes.HDC),
            ('fErase', wintypes.BOOL),
            ('rcPaint', wintypes.RECT),
            ('fRestore', wintypes.BOOL),
            ('fIncUpdate', wintypes.BOOL),
            ('rgbReserved', ctypes.c_byte * 32),
        ]

    class WNDCLASSW(ctypes.Structure):
        _fields_ = [
            ('style', ctypes.c_uint),
            ('lpfnWndProc', ctypes.WINFUNCTYPE(ctypes.c_longlong, wintypes.HWND, ctypes.c_uint, wintypes.WPARAM, wintypes.LPARAM)),
            ('cbClsExtra', ctypes.c_int),
            ('cbWndExtra', ctypes.c_int),
            ('hInstance', wintypes.HINSTANCE),
            ('hIcon', HICON),
            ('hCursor', HCURSOR),
            ('hbrBackground', HBRUSH),
            ('lpszMenuName', wintypes.LPCWSTR),
            ('lpszClassName', wintypes.LPCWSTR),
        ]


class YoloOverlayWindow:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._window_region: CaptureRegion | None = None
        self._capture_region: CaptureRegion | None = None
        self._detections: list[OverlayDetection] = []
        self._available = sys.platform.startswith('win')
        self._hwnd: int | None = None
        self._class_name = f'CodexYoloOverlay_{id(self)}'
        self._wndproc_ref = None
        self._class_atom = 0
        self.is_running = False

    def start(self) -> None:
        if not self._available:
            self.logger.warning('YOLO overlay unavailable on this platform/runtime. Skipping overlay.')
            return
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name='yolo-overlay')
        self._thread.start()
        self.is_running = True

    def stop(self) -> None:
        self._stop_event.set()
        hwnd = self._hwnd
        if hwnd:
            user32.PostMessageW(hwnd, WM_DESTROY, 0, 0)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.is_running = False

    def update(
        self,
        *,
        window_region: CaptureRegion | None,
        capture_region: CaptureRegion | None,
        detections: list[OverlayDetection],
    ) -> None:
        with self._lock:
            self._window_region = window_region
            self._capture_region = capture_region
            self._detections = list(detections)
        hwnd = self._hwnd
        if hwnd:
            user32.PostMessageW(hwnd, WM_APP_UPDATE, 0, 0)

    def _run(self) -> None:
        try:
            self._create_window()
            msg = wintypes.MSG()
            while not self._stop_event.is_set():
                result = user32.GetMessageW(ctypes.byref(msg), 0, 0, 0)
                if result <= 0:
                    break
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        except Exception:
            self.logger.exception('YOLO overlay crashed.')
        finally:
            self.is_running = False

    def _create_window(self) -> None:
        hinstance = kernel32.GetModuleHandleW(None)

        @ctypes.WINFUNCTYPE(ctypes.c_longlong, wintypes.HWND, ctypes.c_uint, wintypes.WPARAM, wintypes.LPARAM)
        def wndproc(hwnd, msg, wparam, lparam):
            if msg == WM_APP_UPDATE:
                self._sync_window()
                user32.InvalidateRect(hwnd, None, True)
                return 0
            if msg == WM_ERASEBKGND:
                return 1
            if msg == WM_PAINT:
                self._paint(hwnd)
                return 0
            if msg == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        self._wndproc_ref = wndproc
        wndclass = WNDCLASSW()
        wndclass.lpfnWndProc = wndproc
        wndclass.hInstance = hinstance
        wndclass.lpszClassName = self._class_name
        wndclass.hbrBackground = gdi32.CreateSolidBrush(COLORKEY)
        self._class_atom = user32.RegisterClassW(ctypes.byref(wndclass))
        if not self._class_atom:
            raise OSError('RegisterClassW failed for YOLO overlay.')

        hwnd = user32.CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            self._class_name,
            'YOLO Overlay',
            WS_POPUP | WS_VISIBLE,
            0,
            0,
            1,
            1,
            0,
            0,
            hinstance,
            0,
        )
        if not hwnd:
            raise OSError('CreateWindowExW failed for YOLO overlay.')
        self._hwnd = int(hwnd)
        if not user32.SetLayeredWindowAttributes(hwnd, COLORKEY, 0, LWA_COLORKEY):
            raise OSError('SetLayeredWindowAttributes failed for YOLO overlay.')
        self._sync_window()
        user32.ShowWindow(hwnd, SW_SHOWNA)
        user32.UpdateWindow(hwnd)

    def _sync_window(self) -> None:
        hwnd = self._hwnd
        if not hwnd:
            return
        with self._lock:
            window_region = self._window_region
        if window_region is None:
            user32.ShowWindow(hwnd, SW_HIDE)
            return
        user32.SetWindowPos(
            hwnd,
            -1,
            window_region.left,
            window_region.top,
            max(window_region.width, 1),
            max(window_region.height, 1),
            SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )
        user32.ShowWindow(hwnd, SW_SHOWNA)

    def _paint(self, hwnd: int) -> None:
        paint_struct = PAINTSTRUCT()
        hdc = user32.BeginPaint(hwnd, ctypes.byref(paint_struct))
        try:
            with self._lock:
                window_region = self._window_region
                capture_region = self._capture_region
                detections = list(self._detections)
            if window_region is None:
                return

            rect = wintypes.RECT(0, 0, max(window_region.width, 1), max(window_region.height, 1))
            brush = gdi32.CreateSolidBrush(COLORKEY)
            user32.FillRect(hdc, ctypes.byref(rect), brush)
            gdi32.DeleteObject(brush)
            gdi32.SetBkMode(hdc, TRANSPARENT)

            if capture_region is not None:
                crop_left = capture_region.left - window_region.left
                crop_top = capture_region.top - window_region.top
                crop_right = crop_left + capture_region.width
                crop_bottom = crop_top + capture_region.height
                self._draw_rect(hdc, crop_left, crop_top, crop_right, crop_bottom, 0x00FFD800, 1)

            center_x = int(window_region.width / 2.0)
            center_y = int(window_region.height / 2.0)
            self._draw_line(hdc, center_x - 8, center_y, center_x + 8, center_y, 0x00FFFFFF, 1)
            self._draw_line(hdc, center_x, center_y - 8, center_x, center_y + 8, 0x00FFFFFF, 1)

            for detection in detections:
                color = 0x00FFA64D
                if detection.is_enemy and detection.is_head:
                    color = 0x005252FF
                elif detection.is_enemy:
                    color = 0x0047B3FF
                if detection.is_selected:
                    color = 0x007FFF00

                self._draw_rect(
                    hdc,
                    int(detection.x1),
                    int(detection.y1),
                    int(detection.x2),
                    int(detection.y2),
                    color,
                    2 if detection.is_selected else 1,
                )
                self._draw_text(
                    hdc,
                    int(detection.x1 + 4),
                    max(0, int(detection.y1 - 14)),
                    f'{detection.label} {detection.confidence:.2f}',
                    color,
                )
        finally:
            user32.EndPaint(hwnd, ctypes.byref(paint_struct))

    def _draw_rect(self, hdc, left: int, top: int, right: int, bottom: int, color: int, width: int) -> None:
        pen = gdi32.CreatePen(PS_SOLID, width, color)
        old_pen = gdi32.SelectObject(hdc, pen)
        old_brush = gdi32.SelectObject(hdc, gdi32.GetStockObject(5))  # NULL_BRUSH
        gdi32.Rectangle(hdc, left, top, right, bottom)
        gdi32.SelectObject(hdc, old_brush)
        gdi32.SelectObject(hdc, old_pen)
        gdi32.DeleteObject(pen)

    def _draw_line(self, hdc, x1: int, y1: int, x2: int, y2: int, color: int, width: int) -> None:
        pen = gdi32.CreatePen(PS_SOLID, width, color)
        old_pen = gdi32.SelectObject(hdc, pen)
        gdi32.MoveToEx(hdc, x1, y1, None)
        gdi32.LineTo(hdc, x2, y2)
        gdi32.SelectObject(hdc, old_pen)
        gdi32.DeleteObject(pen)

    def _draw_text(self, hdc, x: int, y: int, text: str, color: int) -> None:
        gdi32.SetTextColor(hdc, color)
        gdi32.TextOutW(hdc, x, y, text, len(text))
