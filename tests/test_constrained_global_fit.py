import math

import maixcam2_puzzle_vision.solver as solver
from maixcam2_puzzle_vision.config import SolverConfig
from maixcam2_puzzle_vision.geometry import apply_transform_polygon, polygon_centroid
from maixcam2_puzzle_vision.models import PieceObservation, RigidTransform2D, SolveStatus


def _piece(
    piece_id: int,
    polygon: tuple[tuple[float, float], ...],
    transform: RigidTransform2D = RigidTransform2D(),
) -> PieceObservation:
    placed = apply_transform_polygon(polygon, transform)
    return PieceObservation(piece_id, placed, polygon_centroid(placed))


def _scattered_observations() -> tuple[PieceObservation, ...]:
    polygons = (
        ((102.5, 36.0), (104.0, 98.0), (81.5, 96.5), (78.5, 36.0)),
        ((58.5, 60.5), (68.5, 110.0), (36.0, 116.5), (18.5, 104.5), (45.5, 52.0)),
        ((114.5, 123.5), (92.5, 166.0), (74.5, 120.0), (97.0, 111.0)),
        ((70.0, 168.5), (34.5, 170.0), (32.0, 129.5)),
    )
    return tuple(_piece(index, polygon) for index, polygon in enumerate(polygons))


def _partial_gap_observations() -> tuple[PieceObservation, ...]:
    polygons = (
        ((0.0, 0.0), (30.0, 0.0), (30.0, 60.0), (0.0, 60.0)),
        ((30.0, 0.0), (100.0, 0.0), (100.0, 20.0), (30.0, 20.0)),
        ((30.0, 20.0), (100.0, 20.0), (100.0, 60.0), (70.0, 60.0)),
        ((30.0, 20.0), (70.0, 60.0), (30.0, 60.0)),
    )
    transforms = (
        RigidTransform2D(math.radians(8.0), 24.0, 30.0),
        RigidTransform2D(math.radians(-31.0), 150.0, 42.0),
        RigidTransform2D(math.radians(57.0), 44.0, 145.0),
        RigidTransform2D(math.radians(-67.0), 152.0, 146.0),
    )
    return tuple(
        _piece(index, polygon, transform)
        for index, (polygon, transform) in enumerate(zip(polygons, transforms))
    )


def test_solver_recovers_scattered_hand_cut_layout_with_bounded_global_fit() -> None:
    result = solver.solve_layout(_scattered_observations(), SolverConfig(max_states=4000))

    assert result.status is SolveStatus.OK
    assert result.diagnostics["strategy"] == "constrained_global_rigid_fit"
    assert result.fill_ratio >= 0.90


def test_exact_variable_rectangle_does_not_enter_global_fit() -> None:
    result = solver.solve_layout(
        _partial_gap_observations(),
        SolverConfig(max_states=4000, ambiguity_margin=0.0),
    )

    assert result.status is SolveStatus.OK
    assert result.diagnostics["strategy"] != "constrained_global_rigid_fit"
    assert result.rectangle_size_mm is not None
