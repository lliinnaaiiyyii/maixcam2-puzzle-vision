import math

from maixcam2_puzzle_vision.config import (
    AppConfig,
    BoardConfig,
    CameraConfig,
    PatternConfig,
    SegmentationConfig,
    SolverConfig,
)
from maixcam2_puzzle_vision.geometry import polygon_centroid
from maixcam2_puzzle_vision.models import AssemblyResult, PieceObservation, RigidTransform2D, SolveStatus
from maixcam2_puzzle_vision.planner import plan_assembly


def _landscape_config() -> AppConfig:
    return AppConfig(
        camera=CameraConfig(1280, 720, -1, -1, ()),
        board=BoardConfig(
            size_mm=(297.0, 210.0),
            pixels_per_mm=2.0,
            source_roi_mm=(3.0, 3.0, 142.0, 204.0),
            target_roi_mm=(152.0, 3.0, 142.0, 204.0),
            target_center_mm=(223.0, 105.0),
            homography_image_to_board=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        segmentation=SegmentationConfig(24.0, 150.0, 1.0, 5.0, 4.0, 8.0),
        solver=SolverConfig(),
        pattern=PatternConfig(),
    )


def _rotate_about(point: tuple[float, float], center: tuple[float, float], angle_deg: float) -> tuple[float, float]:
    radians = math.radians(angle_deg)
    delta_x, delta_y = point[0] - center[0], point[1] - center[1]
    return (
        center[0] + delta_x * math.cos(radians) - delta_y * math.sin(radians),
        center[1] + delta_x * math.sin(radians) + delta_y * math.cos(radians),
    )


def test_planner_aligns_horizontal_rectangle_long_side_with_tall_target_roi() -> None:
    polygon = ((20.0, 20.0), (120.0, 20.0), (120.0, 80.0), (20.0, 80.0))
    piece = PieceObservation(0, polygon, polygon_centroid(polygon))
    assembly = AssemblyResult(
        status=SolveStatus.OK,
        transforms={0: RigidTransform2D()},
        fill_ratio=1.0,
        score=0.0,
    )

    result = plan_assembly((piece,), assembly, _landscape_config())

    assert result.status is SolveStatus.OK
    assert result.rectangle_size_mm == (60.0, 100.0)
    command = result.commands[0]
    target_polygon = tuple(_rotate_about(point, command.pick_xy_mm, command.delta_theta_deg) for point in polygon)
    target_polygon = tuple(
        (point[0] - command.pick_xy_mm[0] + command.place_xy_mm[0], point[1] - command.pick_xy_mm[1] + command.place_xy_mm[1])
        for point in target_polygon
    )
    xs = [point[0] for point in target_polygon]
    ys = [point[1] for point in target_polygon]

    assert math.isclose(max(xs) - min(xs), 60.0, abs_tol=1e-6)
    assert math.isclose(max(ys) - min(ys), 100.0, abs_tol=1e-6)
    assert min(xs) >= 152.0
    assert max(xs) <= 294.0
    assert min(ys) >= 3.0
    assert max(ys) <= 207.0
