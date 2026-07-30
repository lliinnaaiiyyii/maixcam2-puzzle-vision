from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    exposure_us: int
    gain: int
    white_balance: tuple[float, ...]


@dataclass(frozen=True)
class BoardConfig:
    size_mm: tuple[float, float]
    pixels_per_mm: float
    source_roi_mm: tuple[float, float, float, float]
    target_roi_mm: tuple[float, float, float, float]
    target_center_mm: tuple[float, float]
    homography_image_to_board: tuple[tuple[float, float, float], ...]

    @property
    def is_calibrated(self) -> bool:
        return self.homography_image_to_board != ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@dataclass(frozen=True)
class SegmentationConfig:
    delta_e_threshold: float
    min_piece_area_mm2: float
    morphology_radius_mm: float
    polygon_epsilon_mm: float
    collinear_angle_deg: float
    min_edge_mm: float


@dataclass(frozen=True)
class SolverConfig:
    edge_length_tolerance_mm: float = 2.0
    edge_length_tolerance_ratio: float = 0.05
    max_overlap_ratio: float = 0.02
    min_rectangle_side_mm: float = 20.0
    min_rectangle_short_side_mm: float = 50.0
    max_rectangle_short_side_mm: float = 90.0
    min_rectangle_long_side_mm: float = 90.0
    max_rectangle_long_side_mm: float = 120.0
    rectangle_size_tolerance_mm: float = 1.0
    min_rectangle_fill_ratio: float = 0.94
    max_internal_hole_mm2: float = 25.0
    ambiguity_margin: float = 0.04
    max_states: int = 1200
    max_seam_residual_mm: float = 2.5
    pattern_margin: float = 0.03
    min_confidence: float = 0.75


@dataclass(frozen=True)
class PatternConfig:
    enabled: bool = True
    ambiguity_score_band: float = 0.04
    texture_margin: float = 0.03
    strip_width_px: int = 10
    strip_length_px: int = 40


@dataclass(frozen=True)
class AutoCalibrationConfig:
    enabled: bool = True
    hue_min: int = 35
    hue_max: int = 95
    saturation_min: int = 80
    value_min: int = 60
    min_area_ratio: float = 0.15
    border_margin_px: int = 4
    stable_frames: int = 3
    corner_stability_px: float = 6.0
    processing_width: int = 640


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    board: BoardConfig
    segmentation: SegmentationConfig
    solver: SolverConfig
    pattern: PatternConfig
    auto_calibration: AutoCalibrationConfig = field(default_factory=AutoCalibrationConfig)


def rectangle_size_is_within_topic_limit(width: float, height: float, config: SolverConfig) -> bool:
    if not math.isfinite(width) or not math.isfinite(height):
        return False
    short_side, long_side = sorted((width, height))
    tolerance = config.rectangle_size_tolerance_mm
    return (
        short_side >= max(config.min_rectangle_side_mm, config.min_rectangle_short_side_mm - tolerance)
        and short_side <= config.max_rectangle_short_side_mm + tolerance
        and long_side >= config.min_rectangle_long_side_mm - tolerance
        and long_side <= config.max_rectangle_long_side_mm + tolerance
    )


def _tuple(values: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{label} must contain {length} values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def _roi(values: Any, board_size: tuple[float, float], label: str) -> tuple[float, float, float, float]:
    x, y, width, height = _tuple(values, 4, label)
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > board_size[0] or y + height > board_size[1]:
        raise ValueError(f"{label} is outside board bounds")
    return x, y, width, height


def _homography(values: Any) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError("homography_image_to_board must be 3x3")
    matrix = tuple(_tuple(row, 3, "homography_image_to_board row") for row in values)
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if abs(determinant) < 1e-12:
        raise ValueError("homography_image_to_board is singular")
    return matrix


def load_config(path: str | Path) -> AppConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        camera_raw = raw["camera"]
        board_raw = raw["board"]
        segmentation_raw = raw["segmentation"]
    except (KeyError, TypeError) as error:
        raise ValueError("config must include camera, board, and segmentation") from error

    size_mm = _tuple(board_raw["size_mm"], 2, "board.size_mm")
    if size_mm not in ((210.0, 297.0), (297.0, 210.0)):
        raise ValueError("board.size_mm must be [210, 297] or [297, 210] for A4")
    pixels_per_mm = float(board_raw["pixels_per_mm"])
    if not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise ValueError("board.pixels_per_mm must be positive")

    camera = CameraConfig(
        width=int(camera_raw["width"]),
        height=int(camera_raw["height"]),
        exposure_us=int(camera_raw.get("exposure_us", -1)),
        gain=int(camera_raw.get("gain", -1)),
        white_balance=tuple(float(value) for value in camera_raw.get("white_balance", ())),
    )
    if camera.width <= 0 or camera.height <= 0:
        raise ValueError("camera dimensions must be positive")
    board = BoardConfig(
        size_mm=size_mm,
        pixels_per_mm=pixels_per_mm,
        source_roi_mm=_roi(board_raw["source_roi_mm"], size_mm, "board.source_roi_mm"),
        target_roi_mm=_roi(board_raw["target_roi_mm"], size_mm, "board.target_roi_mm"),
        target_center_mm=_tuple(board_raw["target_center_mm"], 2, "board.target_center_mm"),
        homography_image_to_board=_homography(board_raw["homography_image_to_board"]),
    )
    segmentation = SegmentationConfig(
        delta_e_threshold=float(segmentation_raw["delta_e_threshold"]),
        min_piece_area_mm2=float(segmentation_raw["min_piece_area_mm2"]),
        morphology_radius_mm=float(segmentation_raw["morphology_radius_mm"]),
        polygon_epsilon_mm=float(segmentation_raw["polygon_epsilon_mm"]),
        collinear_angle_deg=float(segmentation_raw["collinear_angle_deg"]),
        min_edge_mm=float(segmentation_raw.get("min_edge_mm", 20.0)),
    )
    if segmentation.min_piece_area_mm2 <= 0 or segmentation.morphology_radius_mm < 0 or segmentation.min_edge_mm <= 0:
        raise ValueError("segmentation settings are outside topic constraints")
    solver = SolverConfig(**raw.get("solver", {}))
    if not math.isfinite(solver.min_rectangle_side_mm) or solver.min_rectangle_side_mm <= 0:
        raise ValueError("solver.min_rectangle_side_mm must be positive")
    rectangle_limits = (
        solver.min_rectangle_short_side_mm,
        solver.max_rectangle_short_side_mm,
        solver.min_rectangle_long_side_mm,
        solver.max_rectangle_long_side_mm,
        solver.rectangle_size_tolerance_mm,
    )
    if not all(math.isfinite(value) for value in rectangle_limits):
        raise ValueError("solver rectangle size limits must be finite")
    if (
        solver.min_rectangle_short_side_mm <= 0
        or solver.max_rectangle_short_side_mm < solver.min_rectangle_short_side_mm
        or solver.min_rectangle_long_side_mm < solver.min_rectangle_short_side_mm
        or solver.max_rectangle_long_side_mm < solver.min_rectangle_long_side_mm
        or solver.rectangle_size_tolerance_mm < 0
    ):
        raise ValueError("solver rectangle size limits are invalid")
    auto_calibration = AutoCalibrationConfig(**raw.get("auto_calibration", {}))
    if not 0 <= auto_calibration.hue_min < auto_calibration.hue_max <= 179:
        raise ValueError("auto_calibration hue range must be inside [0, 179]")
    if not 0 <= auto_calibration.saturation_min <= 255 or not 0 <= auto_calibration.value_min <= 255:
        raise ValueError("auto_calibration saturation/value minimum must be inside [0, 255]")
    if not 0 < auto_calibration.min_area_ratio <= 1:
        raise ValueError("auto_calibration.min_area_ratio must be inside (0, 1]")
    if auto_calibration.border_margin_px < 0 or auto_calibration.stable_frames < 1 or auto_calibration.processing_width < 64:
        raise ValueError("auto_calibration dimensions are invalid")
    if not math.isfinite(auto_calibration.corner_stability_px) or auto_calibration.corner_stability_px <= 0:
        raise ValueError("auto_calibration.corner_stability_px must be positive")
    return AppConfig(
        camera=camera,
        board=board,
        segmentation=segmentation,
        solver=solver,
        pattern=PatternConfig(**raw.get("pattern", {})),
        auto_calibration=auto_calibration,
    )
