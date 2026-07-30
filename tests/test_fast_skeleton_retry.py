import math

import maixcam2_puzzle_vision.solver as solver
from maixcam2_puzzle_vision.config import SolverConfig
from maixcam2_puzzle_vision.geometry import apply_transform_polygon, polygon_centroid
from maixcam2_puzzle_vision.models import AssemblyResult, PieceObservation, RigidTransform2D, SolveStatus


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


def test_solver_stops_skeleton_gap_early_for_multiple_valid_rectangles(monkeypatch) -> None:
    monkeypatch.setattr(solver, "_FAST_SKELETON_STATE_LIMIT", 1)
    monkeypatch.setattr(solver, "_EARLY_SKELETON_BATCH_SIZE", 1)

    result = solver.solve_layout(
        _partial_gap_observations(),
        SolverConfig(
            max_states=4000,
            ambiguity_margin=0.08,
            allow_multiple_valid_rectangles=True,
            early_accept_valid_layout_count=1,
        ),
    )

    assert result.status is SolveStatus.OK
    assert result.diagnostics["early_accept_triggered"] is True
    assert result.diagnostics["states_visited"] < 800
    assert result.diagnostics["skeleton_state_limits"] == [solver._STRICT_SKELETON_STATE_LIMIT]


def test_contest_mode_tries_strict_skeleton_matching_before_relaxed_fallback(monkeypatch) -> None:
    calls: list[tuple[int | None, bool]] = []

    def fake_solver(
        _pieces: tuple[PieceObservation, ...],
        _config: SolverConfig,
        state_limit: int | None = None,
        enable_global_fit: bool = True,
        relaxed_skeleton: bool = True,
    ) -> AssemblyResult:
        calls.append((state_limit, relaxed_skeleton))
        return AssemblyResult(SolveStatus.OK)

    monkeypatch.setattr(solver, "_solve_skeleton_gap_layout", fake_solver)

    result = solver._solve_skeleton_gap_with_retry(
        (),
        SolverConfig(max_states=4000, allow_multiple_valid_rectangles=True),
    )

    assert result is not None
    assert result.status is SolveStatus.OK
    assert calls == [(solver._STRICT_SKELETON_STATE_LIMIT, False)]
    assert result.diagnostics["skeleton_state_limits"] == [solver._STRICT_SKELETON_STATE_LIMIT]
