import math

import maixcam2_puzzle_vision.solver as solver
from maixcam2_puzzle_vision.config import SolverConfig
from maixcam2_puzzle_vision.geometry import apply_transform_polygon, polygon_centroid
from maixcam2_puzzle_vision.models import PieceObservation, RigidTransform2D, SolveStatus


def _piece(
    piece_id: int,
    polygon: tuple[tuple[float, float], ...],
    transform: RigidTransform2D,
) -> PieceObservation:
    placed = apply_transform_polygon(polygon, transform)
    return PieceObservation(piece_id, placed, polygon_centroid(placed))


def _partial_gap_observations() -> tuple[PieceObservation, ...]:
    polygons = (
        ((0.0, 0.0), (30.0, 0.0), (30.0, 60.0), (0.0, 60.0)),
        ((30.0, 0.0), (100.0, 0.0), (100.0, 20.0), (30.0, 20.0)),
        ((30.0, 20.0), (100.0, 20.0), (100.0, 60.0), (70.0, 60.0)),
        ((30.0, 20.0), (70.0, 60.0), (30.0, 60.0)),
    )
    transforms = (
        RigidTransform2D(math.radians(8), 24.0, 30.0),
        RigidTransform2D(math.radians(-31), 150.0, 42.0),
        RigidTransform2D(math.radians(57), 44.0, 145.0),
        RigidTransform2D(math.radians(-67), 152.0, 146.0),
    )
    return tuple(
        _piece(index, polygon, transform)
        for index, (polygon, transform) in enumerate(zip(polygons, transforms))
    )


def test_solver_retries_full_skeleton_budget_after_fast_limit_is_inconclusive(monkeypatch) -> None:
    monkeypatch.setattr(solver, "_FAST_SKELETON_STATE_LIMIT", 1)

    result = solver.solve_layout(
        _partial_gap_observations(),
        SolverConfig(max_states=4000, ambiguity_margin=0.0),
    )

    assert result.status is SolveStatus.OK
    assert result.diagnostics["skeleton_state_limits"] == [1, 800]
