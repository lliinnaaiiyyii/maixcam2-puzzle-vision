from maixcam2_puzzle_vision.geometry import polygon_centroid
from maixcam2_puzzle_vision.models import PieceObservation, RigidTransform2D
from maixcam2_puzzle_vision.solver import _layouts_are_equivalent, _layouts_are_interchangeably_equivalent


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
