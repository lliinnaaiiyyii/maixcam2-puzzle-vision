from __future__ import annotations

from .models import PlanResult, SolveStatus


def _format_number(value: float) -> str:
    rounded = round(float(value), 3)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def build_uart_packet(result: PlanResult) -> str | None:
    """Build the lower-controller frame for a successful puzzle solution."""
    if result.status is not SolveStatus.OK:
        return None

    commands = sorted(result.commands, key=lambda command: command.piece_id)
    fields = [str(len(commands))]
    for command in commands:
        fields.extend(
            (
                _format_number(command.pick_xy_mm[0]),
                _format_number(command.pick_xy_mm[1]),
                _format_number(command.place_xy_mm[0]),
                _format_number(command.place_xy_mm[1]),
                _format_number(command.delta_theta_deg),
            )
        )
    return "[" + ",".join(fields) + "]"
