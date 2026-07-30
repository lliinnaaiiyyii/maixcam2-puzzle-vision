from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .calibration import homography_from_corners
from .config import AppConfig, AutoCalibrationConfig
from .models import Point


Corners = tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class AutoCalibrationObservation:
    corners: Corners | None
    stable_frames: int
    ready: bool


def _order_corners(points: np.ndarray) -> Corners | None:
    if points.shape != (4, 2):
        return None
    sums = points.sum(axis=1)
    differences = points[:, 1] - points[:, 0]
    indices = (int(np.argmin(sums)), int(np.argmin(differences)), int(np.argmax(sums)), int(np.argmax(differences)))
    if len(set(indices)) != 4:
        return None
    return tuple(tuple(float(value) for value in points[index]) for index in indices)  # type: ignore[return-value]


def _scaled_frame(frame_bgr: np.ndarray, processing_width: int) -> tuple[np.ndarray, float]:
    height, width = frame_bgr.shape[:2]
    if width <= processing_width:
        return frame_bgr, 1.0
    scale = processing_width / width
    return cv2.resize(frame_bgr, (processing_width, int(round(height * scale))), interpolation=cv2.INTER_AREA), scale


def detect_a4_corners(frame_bgr: np.ndarray, config: AutoCalibrationConfig) -> Corners | None:
    """Return a complete green A4 contour as TL, TR, BR, BL image points."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("detect_a4_corners expects BGR image data")
    scaled, scale = _scaled_frame(frame_bgr, config.processing_width)
    hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
    lower = np.asarray((config.hue_min, config.saturation_min, config.value_min), dtype=np.uint8)
    upper = np.asarray((config.hue_max, 255, 255), dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    frame_area = float(mask.shape[0] * mask.shape[1])
    if cv2.contourArea(contour) < config.min_area_ratio * frame_area:
        return None
    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
    if len(approximation) != 4 or not cv2.isContourConvex(approximation):
        return None
    points = approximation[:, 0, :].astype(np.float64) / scale
    margin = config.border_margin_px / scale
    height, width = frame_bgr.shape[:2]
    if np.any(points[:, 0] <= margin) or np.any(points[:, 1] <= margin):
        return None
    if np.any(points[:, 0] >= width - 1 - margin) or np.any(points[:, 1] >= height - 1 - margin):
        return None
    return _order_corners(points)


class AutoBoardCalibrator:
    """Accept a corner set only after it stays stable across consecutive frames."""

    def __init__(self, config: AutoCalibrationConfig) -> None:
        self._config = config
        self._corners: np.ndarray | None = None
        self._stable_frames = 0

    def reset(self) -> None:
        self._corners = None
        self._stable_frames = 0

    def observe(self, frame_bgr: np.ndarray) -> AutoCalibrationObservation:
        corners = detect_a4_corners(frame_bgr, self._config)
        if corners is None:
            self.reset()
            return AutoCalibrationObservation(None, 0, False)
        current = np.asarray(corners, dtype=np.float64)
        if self._corners is None or np.max(np.linalg.norm(current - self._corners, axis=1)) > self._config.corner_stability_px:
            self._corners = current
            self._stable_frames = 1
        else:
            self._corners = (self._corners * self._stable_frames + current) / (self._stable_frames + 1)
            self._stable_frames += 1
        stable_corners = _order_corners(self._corners)
        ready = self._stable_frames >= self._config.stable_frames
        return AutoCalibrationObservation(stable_corners, self._stable_frames, ready)


def config_from_auto_calibration(config: AppConfig, corners: Corners) -> AppConfig:
    homography = homography_from_corners(corners, config.board.size_mm)
    return replace(config, board=replace(config.board, homography_image_to_board=homography))
