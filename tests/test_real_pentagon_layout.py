from maixcam2_puzzle_vision.config import SolverConfig
from maixcam2_puzzle_vision.geometry import polygon_centroid
from maixcam2_puzzle_vision.models import PieceObservation, SolveStatus
from maixcam2_puzzle_vision import solver
from maixcam2_puzzle_vision.solver import solve_layout


def test_minimum_rectangle_projects_points_without_per_point_rotation(monkeypatch) -> None:
    polygons = (
        ((0.0, 0.0), (60.0, 0.0), (60.0, 40.0), (0.0, 40.0)),
        ((60.0, 0.0), (100.0, 0.0), (100.0, 40.0), (60.0, 40.0)),
    )
    rotations = 0
    original_rotation = solver.rotate_point

    def count_rotation(*args):
        nonlocal rotations
        rotations += 1
        return original_rotation(*args)

    monkeypatch.setattr(solver, "rotate_point", count_rotation)

    width, height, _angle = solver._minimum_rectangle(polygons)

    assert sorted((width, height)) == [40.0, 100.0]
    assert rotations == 0


def test_solver_recovers_captured_triangle_quad_pentagon_quad_layout() -> None:
    # Polygons extracted from the captured MaixCAM2 frame containing one pentagon.
    polygons = (
        ((90.0, 54.0), (55.5, 64.0), (81.0, 25.5)),
        ((59.5, 86.0), (56.0, 102.5), (22.0, 96.5), (43.0, 61.0)),
        ((101.0, 76.5), (107.5, 92.0), (69.0, 142.0), (56.0, 134.5), (76.0, 69.0)),
        ((96.5, 157.5), (59.5, 176.5), (57.0, 155.0), (89.5, 140.5)),
    )
    pieces = tuple(
        PieceObservation(piece_id, polygon, polygon_centroid(polygon))
        for piece_id, polygon in enumerate(polygons)
    )

    result = solve_layout(
        pieces,
        SolverConfig(
            max_states=1200,
            ambiguity_margin=0.08,
            allow_multiple_valid_rectangles=True,
        ),
    )

    assert result.status is SolveStatus.OK
    assert result.rectangle_size_mm is not None
    assert 70.0 <= result.rectangle_size_mm[0] <= 75.0
    assert 56.0 <= result.rectangle_size_mm[1] <= 60.0
    assert result.fill_ratio >= 0.94
