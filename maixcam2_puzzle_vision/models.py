from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


Point = tuple[float, float]


class SolveStatus(str, Enum):
    OK = "OK"
    NO_BOARD = "NO_BOARD"
    SEGMENTATION_FAILED = "SEGMENTATION_FAILED"
    INVALID_PIECE_COUNT = "INVALID_PIECE_COUNT"
    INVALID_POLYGON = "INVALID_POLYGON"
    NO_RECTANGLE_SOLUTION = "NO_RECTANGLE_SOLUTION"
    AMBIGUOUS = "AMBIGUOUS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


@dataclass(frozen=True)
class RigidTransform2D:
    angle_rad: float = 0.0
    tx_mm: float = 0.0
    ty_mm: float = 0.0

    @property
    def angle_deg(self) -> float:
        from math import degrees

        return degrees(self.angle_rad)


@dataclass(frozen=True)
class PieceObservation:
    piece_id: int
    polygon_mm: tuple[Point, ...]
    centroid_mm: Point
    contour_px: tuple[Point, ...] = ()


@dataclass(frozen=True)
class MoveCommand:
    piece_id: int
    pick_xy_mm: Point
    place_xy_mm: Point
    delta_theta_deg: float
    confidence: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "piece_id": self.piece_id,
            "pick_x_mm": round(self.pick_xy_mm[0], 3),
            "pick_y_mm": round(self.pick_xy_mm[1], 3),
            "place_x_mm": round(self.place_xy_mm[0], 3),
            "place_y_mm": round(self.place_xy_mm[1], 3),
            "delta_theta_deg": round(self.delta_theta_deg, 3),
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True)
class AssemblyResult:
    status: SolveStatus
    transforms: dict[int, RigidTransform2D] = field(default_factory=dict)
    rectangle_size_mm: Point | None = None
    fill_ratio: float = 0.0
    score: float = 1.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanResult:
    status: SolveStatus
    confidence: float
    commands: tuple[MoveCommand, ...] = ()
    rectangle_size_mm: Point | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(cls, status: SolveStatus, diagnostics: dict[str, Any]) -> "PlanResult":
        if status is SolveStatus.OK:
            raise ValueError("failure status cannot be OK")
        return cls(status=status, confidence=0.0, diagnostics=dict(diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "confidence": round(self.confidence, 3),
            "rectangle_size_mm": (
                None
                if self.rectangle_size_mm is None
                else [round(value, 3) for value in self.rectangle_size_mm]
            ),
            "commands": [command.to_dict() for command in self.commands] if self.status is SolveStatus.OK else [],
            "diagnostics": self.diagnostics,
        }
