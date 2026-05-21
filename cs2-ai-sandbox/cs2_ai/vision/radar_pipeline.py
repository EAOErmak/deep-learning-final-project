from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Iterable

from game_state import GameState, LiveCapabilities, PlayerState, Vector3
from cs2_ai.vision.window_capture import CaptureRegion, WindowCaptureLocator

try:
    import cv2
    import mss
    import numpy as np
except ImportError:
    cv2 = None
    mss = None
    np = None


@dataclass(slots=True)
class RadarTeammate:
    slot: int
    rel_x: float
    rel_y: float
    confidence: float


@dataclass(slots=True)
class RadarObservation:
    self_position: Vector3
    self_forward: Vector3
    teammates: list[RadarTeammate]
    confidence: float


class RadarVisionModule:
    def __init__(
        self,
        *,
        left: int = 24,
        top: int = 24,
        size: int = 220,
        world_scale: float = 2000.0,
        min_blob_area: int = 6,
        max_blob_area: int = 200,
        window_keywords: tuple[str, ...] = ('counter-strike', 'cs2'),
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.left = left
        self.top = top
        self.size = size
        self.world_scale = world_scale
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.sct = None
        self.is_running = False
        self.window_locator = WindowCaptureLocator(window_keywords=window_keywords)

    def start(self) -> None:
        if mss is None or cv2 is None or np is None:
            self.logger.warning('Radar vision dependencies unavailable. Skipping radar vision.')
            return
        self.sct = mss.mss()
        self.is_running = True
        self.logger.info(
            'Radar vision enabled | bbox=(left=%s top=%s size=%s) | world_scale=%s',
            self.left,
            self.top,
            self.size,
            self.world_scale,
        )

    def stop(self) -> None:
        self.sct = None
        self.is_running = False

    def capture(self) -> RadarObservation | None:
        if not self.is_running or self.sct is None or np is None or cv2 is None:
            return None

        bbox = self._resolve_capture_region().as_mss_bbox()
        frame = np.array(self.sct.grab(bbox))[:, :, :3]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        teammate_mask = self._build_teammate_mask(hsv)
        teammates = self._extract_teammates(teammate_mask)
        return RadarObservation(
            self_position=Vector3(0.0, 0.0, 0.0),
            # Assume rotating radar: "up" is current facing direction.
            self_forward=Vector3(0.0, 1.0, 0.0),
            teammates=teammates,
            confidence=1.0 if teammates else 0.0,
        )

    def _build_teammate_mask(self, hsv) -> 'np.ndarray':
        masks = [
            cv2.inRange(hsv, np.array([35, 40, 120]), np.array([95, 255, 255])),
            cv2.inRange(hsv, np.array([90, 30, 120]), np.array([125, 255, 255])),
        ]
        combined = masks[0]
        for mask in masks[1:]:
            combined = cv2.bitwise_or(combined, mask)
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.dilate(combined, kernel, iterations=1)
        return combined

    def _extract_teammates(self, mask) -> list[RadarTeammate]:
        if np is None or cv2 is None:
            return []
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        center = self.size / 2.0
        teammates: list[RadarTeammate] = []
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < self.min_blob_area or area > self.max_blob_area:
                continue
            cx, cy = centroids[label_idx]
            rel_x = float((cx - center) / center * self.world_scale)
            rel_y = float((center - cy) / center * self.world_scale)
            distance = abs(rel_x) + abs(rel_y)
            if distance < 40.0:
                # Ignore the center player marker / noise near the self icon.
                continue
            teammates.append(
                RadarTeammate(
                    slot=len(teammates),
                    rel_x=rel_x,
                    rel_y=rel_y,
                    confidence=min(1.0, area / max(self.min_blob_area, 1)),
                )
            )
        teammates.sort(key=lambda item: abs(item.rel_x) + abs(item.rel_y))
        return teammates[:4]

    def _resolve_capture_region(self) -> CaptureRegion:
        window_region = self.window_locator.find_client_region()
        if window_region is not None:
            max_left = max(window_region.width - self.size, 0)
            max_top = max(window_region.height - self.size, 0)
            left = window_region.left + min(max(self.left, 0), max_left)
            top = window_region.top + min(max(self.top, 0), max_top)
            return CaptureRegion(left=left, top=top, width=self.size, height=self.size)

        return CaptureRegion(left=self.left, top=self.top, width=self.size, height=self.size)


def augment_live_state_with_radar(
    game_state: GameState,
    radar: RadarObservation | None,
) -> GameState:
    if radar is None or game_state.controlled_player is None:
        return game_state

    controlled = game_state.controlled_player
    players = [player for player in game_state.players if player.id == controlled.id or player.team != controlled.team]
    radar_controlled = replace(
        controlled,
        position=controlled.position if controlled.position is not None else radar.self_position,
        forward=controlled.forward if controlled.forward is not None else radar.self_forward,
    )
    teammates = _build_radar_teammates(radar_controlled, radar.teammates)
    players = [radar_controlled, *teammates, *[player for player in players if player.id != controlled.id]]
    capabilities = replace(
        game_state.capabilities,
        has_player_position=radar_controlled.position is not None,
        has_player_forward=radar_controlled.forward is not None,
        has_spatial_state=radar_controlled.position is not None and radar_controlled.forward is not None,
    )
    return replace(
        game_state,
        controlled_player=radar_controlled,
        players=players,
        capabilities=capabilities,
    )


def _build_radar_teammates(controlled: PlayerState, teammates: Iterable[RadarTeammate]) -> list[PlayerState]:
    result: list[PlayerState] = []
    team_name = controlled.team or 'CT'
    for idx, teammate in enumerate(teammates):
        result.append(
            PlayerState(
                id=f'radar_teammate_{idx}',
                name=f'radar_teammate_{idx}',
                team=team_name,
                position=Vector3(teammate.rel_x, teammate.rel_y, 0.0),
                forward=Vector3(0.0, 1.0, 0.0),
                health=100,
                armor=0,
                money=0,
                weapon=None,
                ammo=None,
                is_alive=True,
                velocity=None,
                helmet=None,
                flashed=None,
                smoked=None,
                burning=None,
                round_kills=None,
                round_killhs=None,
                equip_value=None,
                ammo_reserve=None,
                observer_slot=None,
                activity='playing',
            )
        )
    return result
