from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .config import load_config
from .models import Point


class BoardCalibration:
    def __init__(self, homography_image_to_board: tuple[tuple[float, float, float], ...]) -> None:
        if len(homography_image_to_board) != 3 or any(len(row) != 3 for row in homography_image_to_board):
            raise ValueError("homography must be 3x3")
        self._homography = tuple(tuple(float(value) for value in row) for row in homography_image_to_board)
        if not all(math.isfinite(value) for row in self._homography for value in row):
            raise ValueError("homography must be finite")

    def image_to_board(self, point_px: Point) -> Point:
        x, y = point_px
        matrix = self._homography
        denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
        if abs(denominator) < 1e-9:
            raise ValueError("image point projects to homography horizon")
        return (
            (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator,
            (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator,
        )


def homography_from_corners(
    image_corners: tuple[Point, Point, Point, Point],
    board_size_mm: Point = (210.0, 297.0),
) -> tuple[tuple[float, float, float], ...]:
    """Map image corners ordered TL, TR, BR, BL to the configured A4 millimetres."""
    source = np.asarray(image_corners, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(source).all():
        raise ValueError("image corners must be finite")
    width, height = board_size_mm
    if width <= 0 or height <= 0:
        raise ValueError("board size must be positive")
    destination = np.asarray(((0.0, 0.0), (width, 0.0), (width, height), (0.0, height)), dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source, destination)
    return tuple(tuple(float(value) for value in row) for row in homography)


def update_homography_from_corners(
    config_path: str | Path,
    image_corners: tuple[Point, Point, Point, Point],
) -> tuple[tuple[float, float, float], ...]:
    """Write a corner-derived calibration while preserving every other config field."""
    path = Path(config_path)
    config = load_config(path)
    homography = homography_from_corners(image_corners, config.board.size_mm)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["board"]["homography_image_to_board"] = [list(row) for row in homography]
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return homography
