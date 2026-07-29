from __future__ import annotations

import cv2
import numpy as np

from .config import AppConfig
from .models import PlanResult, SolveStatus
from .pattern import make_layout_pattern_scorer
from .planner import plan_assembly
from .segmentation import extract_pieces
from .solver import solve_layout


def rectify_frame(frame_bgr: np.ndarray, config: AppConfig) -> np.ndarray | None:
    expected_width = int(round(config.board.size_mm[0] * config.board.pixels_per_mm))
    expected_height = int(round(config.board.size_mm[1] * config.board.pixels_per_mm))
    if frame_bgr.shape[:2] == (expected_height, expected_width):
        return frame_bgr
    if not config.board.is_calibrated:
        return None
    homography = np.asarray(config.board.homography_image_to_board, dtype=np.float64)
    board_scale = np.array(
        ((config.board.pixels_per_mm, 0.0, 0.0), (0.0, config.board.pixels_per_mm, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return cv2.warpPerspective(frame_bgr, board_scale @ homography, (expected_width, expected_height))


def solve_frame(frame_bgr: np.ndarray, config: AppConfig) -> PlanResult:
    board = rectify_frame(frame_bgr, config)
    if board is None:
        return PlanResult.failure(SolveStatus.NO_BOARD, {"reason": "missing_or_invalid_homography"})
    pieces = extract_pieces(board, config)
    if not pieces:
        return PlanResult.failure(SolveStatus.SEGMENTATION_FAILED, {"piece_count": 0})
    if not 1 <= len(pieces) <= 4:
        return PlanResult.failure(SolveStatus.INVALID_PIECE_COUNT, {"piece_count": len(pieces)})
    pattern_scorer = (
        make_layout_pattern_scorer(board, pieces, config.board.pixels_per_mm)
        if config.pattern.enabled
        else None
    )
    assembly = solve_layout(pieces, config.solver, pattern_scorer)
    return plan_assembly(pieces, assembly, config)
