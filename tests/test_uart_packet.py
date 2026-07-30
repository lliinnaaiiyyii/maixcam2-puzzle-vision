from maixcam2_puzzle_vision.models import MoveCommand, PlanResult, SolveStatus
from maixcam2_puzzle_vision.serial_protocol import build_uart_packet


def command(
    piece_id: int,
    pick_xy_mm: tuple[float, float],
    place_xy_mm: tuple[float, float],
    delta_theta_deg: float,
) -> MoveCommand:
    return MoveCommand(
        piece_id=piece_id,
        pick_xy_mm=pick_xy_mm,
        place_xy_mm=place_xy_mm,
        delta_theta_deg=delta_theta_deg,
        confidence=0.9,
    )


def test_build_uart_packet_uses_bracket_frame_and_deterministic_piece_order() -> None:
    result = PlanResult(
        status=SolveStatus.OK,
        confidence=0.9,
        commands=(
            command(1, (3.0, 4.0), (153.0, 104.0), 45.5),
            command(0, (1.25, 2.5), (151.0, 102.125), -90.0),
        ),
    )

    packet = build_uart_packet(result)

    assert packet == "[2,1.25,2.5,151,102.125,-90,3,4,153,104,45.5]"
    assert "\n" not in packet
    assert packet.isascii()


def test_build_uart_packet_returns_none_for_unsuccessful_result() -> None:
    result = PlanResult.failure(SolveStatus.NO_RECTANGLE_SOLUTION, {"candidate_count": 0})

    assert build_uart_packet(result) is None
