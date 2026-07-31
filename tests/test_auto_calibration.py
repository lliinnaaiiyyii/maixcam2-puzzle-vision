import cv2
import numpy as np

from maixcam2_puzzle_vision.auto_calibration import AutoBoardCalibrator, detect_a4_corners
from maixcam2_puzzle_vision.config import AutoCalibrationConfig


EXPECTED_CORNERS = np.asarray(((160, 52), (1064, 58), (1056, 680), (182, 690)), dtype=np.float64)


def _green_a4_frame() -> np.ndarray:
    frame = np.full((720, 1280, 3), (35, 65, 95), dtype=np.uint8)
    cv2.fillConvexPoly(frame, EXPECTED_CORNERS.astype(np.int32), (20, 230, 40), cv2.LINE_AA)
    cv2.fillConvexPoly(frame, np.asarray(((250, 200), (390, 240), (320, 410)), dtype=np.int32), (235, 235, 235))
    cv2.fillConvexPoly(frame, np.asarray(((420, 150), (490, 150), (490, 330), (420, 330)), dtype=np.int32), (235, 235, 235))
    return frame


def _distant_green_a4_frame() -> tuple[np.ndarray, np.ndarray]:
    frame = np.full((720, 1280, 3), (35, 65, 95), dtype=np.uint8)
    corners = np.asarray(((430, 220), (750, 220), (750, 420), (430, 420)), dtype=np.int32)
    cv2.fillConvexPoly(frame, corners, (20, 230, 40), cv2.LINE_AA)
    return frame, corners.astype(np.float64)


def test_detect_a4_corners_finds_green_quadrilateral_around_white_pieces() -> None:
    corners = detect_a4_corners(_green_a4_frame(), AutoCalibrationConfig())

    assert corners is not None
    np.testing.assert_allclose(np.asarray(corners), EXPECTED_CORNERS, atol=6.0)


def test_detect_a4_corners_accepts_distant_green_a4() -> None:
    frame, expected_corners = _distant_green_a4_frame()

    corners = detect_a4_corners(frame, AutoCalibrationConfig())

    assert corners is not None
    np.testing.assert_allclose(np.asarray(corners), expected_corners, atol=4.0)


def test_auto_board_calibrator_requires_three_stable_frames() -> None:
    calibrator = AutoBoardCalibrator(AutoCalibrationConfig(stable_frames=3))
    frame = _green_a4_frame()

    first = calibrator.observe(frame)
    second = calibrator.observe(frame)
    third = calibrator.observe(frame)

    assert first.ready is False
    assert second.ready is False
    assert third.ready is True
    assert third.corners is not None


def test_auto_board_calibrator_rejects_frame_without_green_a4() -> None:
    calibrator = AutoBoardCalibrator(AutoCalibrationConfig())
    frame = np.full((720, 1280, 3), (35, 65, 95), dtype=np.uint8)

    observation = calibrator.observe(frame)

    assert observation.ready is False
    assert observation.corners is None
    assert observation.stable_frames == 0
