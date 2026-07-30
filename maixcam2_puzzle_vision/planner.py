from __future__ import annotations

import math

from .config import AppConfig
from .geometry import (
    apply_transform,
    apply_transform_polygon,
    compose,
    oriented_bounding_box,
    point_in_polygon,
    polygon_centroid,
)
from .models import AssemblyResult, MoveCommand, PieceObservation, PlanResult, RigidTransform2D, SolveStatus


def _inside_roi(point: tuple[float, float], roi: tuple[float, float, float, float]) -> bool:
    x, y, width, height = roi
    return x <= point[0] <= x + width and y <= point[1] <= y + height


def _polygon_inside_roi(
    polygon: tuple[tuple[float, float], ...],
    roi: tuple[float, float, float, float],
) -> bool:
    return all(_inside_roi(point, roi) for point in polygon)


def _interior_pick(polygon: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    centroid = polygon_centroid(polygon)
    if point_in_polygon(centroid, polygon):
        return centroid
    vertices = tuple(polygon)
    for index in range(1, len(vertices) - 1):
        candidate = (
            (vertices[0][0] + vertices[index][0] + vertices[index + 1][0]) / 3.0,
            (vertices[0][1] + vertices[index][1] + vertices[index + 1][1]) / 3.0,
        )
        if point_in_polygon(candidate, polygon):
            return candidate
    raise ValueError("piece does not have a usable interior pick point")


def _wrap_degrees(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def plan_assembly(pieces: tuple[PieceObservation, ...], assembly: AssemblyResult, config: AppConfig) -> PlanResult:
    if assembly.status is not SolveStatus.OK:
        return PlanResult.failure(assembly.status, assembly.diagnostics)
    placed = tuple(apply_transform_polygon(piece.polygon_mm, assembly.transforms[piece.piece_id]) for piece in pieces)
    angle, min_x, min_y, width, height = oriented_bounding_box(placed)
    normalizer = RigidTransform2D(-angle, -min_x, -min_y)
    target_roi_width, target_roi_height = config.board.target_roi_mm[2:]
    rectangle_is_wide = width > height
    target_roi_is_wide = target_roi_width > target_roi_height
    target_orientation = RigidTransform2D()
    if width != height and target_roi_width != target_roi_height and rectangle_is_wide != target_roi_is_wide:
        # The normalized rectangle spans [0, width] x [0, height]. Rotate it once
        # so its long side matches the target ROI's long side, then move it positive.
        target_orientation = RigidTransform2D(math.pi / 2.0, height, 0.0)
        width, height = height, width
    target_origin = (config.board.target_center_mm[0] - width / 2.0, config.board.target_center_mm[1] - height / 2.0)
    target_transform = compose(RigidTransform2D(0.0, *target_origin), compose(target_orientation, normalizer))
    target_polygons = tuple(
        apply_transform_polygon(piece.polygon_mm, compose(target_transform, assembly.transforms[piece.piece_id]))
        for piece in pieces
    )
    if not all(_polygon_inside_roi(polygon, config.board.target_roi_mm) for polygon in target_polygons):
        return PlanResult.failure(
            SolveStatus.NO_RECTANGLE_SOLUTION,
            {**assembly.diagnostics, "planner_error": "assembled_rectangle_outside_target_roi"},
        )
    confidence = max(0.0, min(1.0, 0.5 * assembly.fill_ratio + 0.4 * max(0.0, 1.0 - assembly.score) + 0.1))
    if confidence < config.solver.min_confidence:
        return PlanResult(
            status=SolveStatus.LOW_CONFIDENCE,
            confidence=confidence,
            diagnostics={**assembly.diagnostics, "reason": "confidence_below_threshold"},
        )
    commands = []
    for piece in sorted(pieces, key=lambda observation: observation.piece_id):
        pick = _interior_pick(piece.polygon_mm)
        final_transform = compose(target_transform, assembly.transforms[piece.piece_id])
        place = apply_transform(pick, final_transform)
        if not _inside_roi(pick, config.board.source_roi_mm) or not _inside_roi(place, config.board.target_roi_mm):
            return PlanResult.failure(
                SolveStatus.NO_RECTANGLE_SOLUTION,
                {**assembly.diagnostics, "planner_error": "pick_or_place_outside_roi"},
            )
        commands.append(
            MoveCommand(
                piece_id=piece.piece_id,
                pick_xy_mm=pick,
                place_xy_mm=place,
                delta_theta_deg=_wrap_degrees(math.degrees(final_transform.angle_rad)),
                confidence=confidence,
            )
        )
    return PlanResult(
        status=SolveStatus.OK,
        confidence=confidence,
        commands=tuple(commands),
        rectangle_size_mm=(width, height),
        diagnostics=assembly.diagnostics,
    )
