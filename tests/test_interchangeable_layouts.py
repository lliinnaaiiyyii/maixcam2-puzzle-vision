from maixcam2_puzzle_vision.geometry import polygon_centroid
from maixcam2_puzzle_vision.models import PieceObservation, RigidTransform2D
from maixcam2_puzzle_vision.solver import (
    _has_dominant_interchangeable_cluster,
    _layouts_are_equivalent,
    _layouts_are_interchangeably_equivalent,
)


def _piece(piece_id: int, polygon: tuple[tuple[float, float], ...]) -> PieceObservation:
    return PieceObservation(piece_id, polygon, polygon_centroid(polygon))


def test_interchangeable_layouts_allow_identical_pieces_to_swap_positions() -> None:
    pieces_by_id = {
        0: _piece(0, ((0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0))),
        1: _piece(1, ((40.0, 0.0), (60.0, 0.0), (60.0, 20.0), (40.0, 20.0))),
        2: _piece(2, ((0.0, 40.0), (60.0, 40.0), (60.0, 60.0), (0.0, 60.0))),
    }
    first = {0: RigidTransform2D(), 1: RigidTransform2D(), 2: RigidTransform2D()}
    second = {
        0: RigidTransform2D(0.0, 40.0, 0.0),
        1: RigidTransform2D(0.0, -40.0, 0.0),
        2: RigidTransform2D(),
    }

    assert _layouts_are_equivalent(first, second) is False
    assert _layouts_are_interchangeably_equivalent(first, second, pieces_by_id) is True


def test_interchangeable_layouts_reject_different_shapes_swapped_into_same_slots() -> None:
    pieces_by_id = {
        0: _piece(0, ((0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0))),
        1: _piece(1, ((40.0, 0.0), (70.0, 0.0), (70.0, 20.0), (40.0, 20.0))),
        2: _piece(2, ((0.0, 40.0), (60.0, 40.0), (60.0, 60.0), (0.0, 60.0))),
    }
    first = {0: RigidTransform2D(), 1: RigidTransform2D(), 2: RigidTransform2D()}
    second = {
        0: RigidTransform2D(0.0, 40.0, 0.0),
        1: RigidTransform2D(0.0, -40.0, 0.0),
        2: RigidTransform2D(),
    }

    assert _layouts_are_interchangeably_equivalent(first, second, pieces_by_id) is False


def test_dominant_interchangeable_cluster_accepts_equivalent_majority() -> None:
    pieces_by_id = {
        0: _piece(0, ((0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0))),
        1: _piece(1, ((40.0, 0.0), (60.0, 0.0), (60.0, 20.0), (40.0, 20.0))),
        2: _piece(2, ((0.0, 40.0), (60.0, 40.0), (60.0, 60.0), (0.0, 60.0))),
    }
    best = {0: RigidTransform2D(), 1: RigidTransform2D(), 2: RigidTransform2D()}
    equivalent_swap = {
        0: RigidTransform2D(0.0, 40.0, 0.0),
        1: RigidTransform2D(0.0, -40.0, 0.0),
        2: RigidTransform2D(),
    }
    different_slot = {0: RigidTransform2D(0.0, 0.0, 28.0), 1: RigidTransform2D(), 2: RigidTransform2D()}

    accepted, equivalent_count, close_count = _has_dominant_interchangeable_cluster(
        best,
        (equivalent_swap, equivalent_swap, different_slot),
        pieces_by_id,
    )

    assert accepted is True
    assert equivalent_count == 3
    assert close_count == 4


def test_dominant_interchangeable_cluster_rejects_non_equivalent_majority() -> None:
    pieces_by_id = {
        0: _piece(0, ((0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0))),
        1: _piece(1, ((40.0, 0.0), (60.0, 0.0), (60.0, 20.0), (40.0, 20.0))),
        2: _piece(2, ((0.0, 40.0), (60.0, 40.0), (60.0, 60.0), (0.0, 60.0))),
    }
    best = {0: RigidTransform2D(), 1: RigidTransform2D(), 2: RigidTransform2D()}
    equivalent_swap = {
        0: RigidTransform2D(0.0, 40.0, 0.0),
        1: RigidTransform2D(0.0, -40.0, 0.0),
        2: RigidTransform2D(),
    }
    different_slots = tuple(
        {0: RigidTransform2D(0.0, 0.0, y_offset), 1: RigidTransform2D(), 2: RigidTransform2D()}
        for y_offset in (28.0, 36.0, 44.0)
    )

    accepted, equivalent_count, close_count = _has_dominant_interchangeable_cluster(
        best,
        (equivalent_swap, *different_slots),
        pieces_by_id,
    )

    assert accepted is False
    assert equivalent_count == 2
    assert close_count == 5
