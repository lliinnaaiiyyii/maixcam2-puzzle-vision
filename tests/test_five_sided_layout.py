import math

from maixcam2_puzzle_vision.config import SolverConfig
from maixcam2_puzzle_vision.geometry import apply_transform_polygon, polygon_centroid
from maixcam2_puzzle_vision.models import PieceObservation, RigidTransform2D, SolveStatus
from maixcam2_puzzle_vision.solver import solve_layout


def test_solver_falls_back_to_multiple_partial_seams_for_a_pentagon() -> None:
    # The pentagon's bottom edge is split between pieces 2 and 3 at x=50 mm.
    assembled = (
        ((0.0, 0.0), (60.0, 0.0), (70.0, 20.0), (100.0, 40.0), (0.0, 40.0)),
        ((60.0, 0.0), (100.0, 0.0), (100.0, 40.0), (70.0, 20.0)),
        ((100.0, 40.0), (100.0, 60.0), (50.0, 60.0), (50.0, 40.0)),
        ((50.0, 40.0), (50.0, 60.0), (0.0, 60.0), (0.0, 40.0)),
    )
    source_poses = (
        RigidTransform2D(math.radians(11.0), 31.0, 19.0),
        RigidTransform2D(math.radians(-27.0), 180.0, 55.0),
        RigidTransform2D(math.radians(42.0), -15.0, 160.0),
        RigidTransform2D(math.radians(-33.0), 120.0, 115.0),
    )
    pieces = []
    for piece_id, (polygon, pose) in enumerate(zip(assembled, source_poses)):
        transformed = apply_transform_polygon(polygon, pose)
        pieces.append(PieceObservation(piece_id, transformed, polygon_centroid(transformed)))

    result = solve_layout(
        tuple(pieces),
        SolverConfig(
            max_states=1200,
            ambiguity_margin=0.08,
            allow_multiple_valid_rectangles=True,
        ),
    )

    assert result.status is SolveStatus.OK
    assert result.rectangle_size_mm is not None
    assert math.isclose(result.rectangle_size_mm[0], 100.0, abs_tol=1e-3)
    assert math.isclose(result.rectangle_size_mm[1], 60.0, abs_tol=1e-3)
    assert result.diagnostics["strategy"] == "pentagon_partial_beam"
