from __future__ import annotations

import math
from typing import Iterable

from .models import Point, RigidTransform2D


EPSILON = 1e-9


def rotate_point(point: Point, angle_rad: float) -> Point:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return point[0] * cosine - point[1] * sine, point[0] * sine + point[1] * cosine


def apply_transform(point: Point, transform: RigidTransform2D) -> Point:
    x, y = rotate_point(point, transform.angle_rad)
    return x + transform.tx_mm, y + transform.ty_mm


def apply_transform_polygon(polygon: Iterable[Point], transform: RigidTransform2D) -> tuple[Point, ...]:
    return tuple(apply_transform(point, transform) for point in polygon)


def _apply_point(self: RigidTransform2D, point: Point) -> Point:
    return apply_transform(point, self)


def _apply_polygon(self: RigidTransform2D, polygon: Iterable[Point]) -> tuple[Point, ...]:
    return apply_transform_polygon(polygon, self)


RigidTransform2D.apply_point = _apply_point  # type: ignore[attr-defined]
RigidTransform2D.apply_polygon = _apply_polygon  # type: ignore[attr-defined]


def compose(first: RigidTransform2D, second: RigidTransform2D) -> RigidTransform2D:
    """Return the transform that applies second, then first."""
    tx, ty = rotate_point((second.tx_mm, second.ty_mm), first.angle_rad)
    return RigidTransform2D(first.angle_rad + second.angle_rad, tx + first.tx_mm, ty + first.ty_mm)


def inverse(transform: RigidTransform2D) -> RigidTransform2D:
    inverse_angle = -transform.angle_rad
    tx, ty = rotate_point((-transform.tx_mm, -transform.ty_mm), inverse_angle)
    return RigidTransform2D(inverse_angle, tx, ty)


def align_reversed_edges(edge_a: tuple[Point, Point], edge_b: tuple[Point, Point]) -> RigidTransform2D:
    """Map edge_b onto edge_a when the two polygon boundary directions oppose."""
    a_start, a_end = edge_a
    b_start, b_end = edge_b
    source_angle = math.atan2(b_end[1] - b_start[1], b_end[0] - b_start[0])
    target_angle = math.atan2(a_start[1] - a_end[1], a_start[0] - a_end[0])
    angle = target_angle - source_angle
    mapped_start = rotate_point(b_start, angle)
    return RigidTransform2D(angle, a_end[0] - mapped_start[0], a_end[1] - mapped_start[1])


def polygon_edges(polygon: tuple[Point, ...]) -> tuple[tuple[Point, Point], ...]:
    return tuple((polygon[index], polygon[(index + 1) % len(polygon)]) for index in range(len(polygon)))


def edge_length(edge: tuple[Point, Point]) -> float:
    return math.dist(edge[0], edge[1])


def polygon_signed_area(polygon: Iterable[Point]) -> float:
    points = tuple(polygon)
    if len(points) < 3:
        return 0.0
    return 0.5 * sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )


def polygon_centroid(polygon: Iterable[Point]) -> Point:
    points = tuple(polygon)
    area = polygon_signed_area(points)
    if abs(area) < EPSILON:
        return sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)
    factor = 1.0 / (6.0 * area)
    x = sum(
        (points[index][0] + points[(index + 1) % len(points)][0])
        * (points[index][0] * points[(index + 1) % len(points)][1] - points[(index + 1) % len(points)][0] * points[index][1])
        for index in range(len(points))
    )
    y = sum(
        (points[index][1] + points[(index + 1) % len(points)][1])
        * (points[index][0] * points[(index + 1) % len(points)][1] - points[(index + 1) % len(points)][0] * points[index][1])
        for index in range(len(points))
    )
    return x * factor, y * factor


def point_in_polygon(point: Point, polygon: Iterable[Point]) -> bool:
    points = tuple(polygon)
    inside = False
    previous = points[-1]
    for current in points:
        intersects = (current[1] > point[1]) != (previous[1] > point[1])
        if intersects:
            boundary_x = (previous[0] - current[0]) * (point[1] - current[1]) / (previous[1] - current[1]) + current[0]
            if point[0] <= boundary_x:
                inside = not inside
        previous = current
    return inside


def _cross(origin: Point, first: Point, second: Point) -> float:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])


def _line_intersection(first_start: Point, first_end: Point, second_start: Point, second_end: Point) -> Point:
    first_dx, first_dy = first_end[0] - first_start[0], first_end[1] - first_start[1]
    second_dx, second_dy = second_end[0] - second_start[0], second_end[1] - second_start[1]
    denominator = first_dx * second_dy - first_dy * second_dx
    if abs(denominator) < EPSILON:
        return first_end
    parameter = ((second_start[0] - first_start[0]) * second_dy - (second_start[1] - first_start[1]) * second_dx) / denominator
    return first_start[0] + parameter * first_dx, first_start[1] + parameter * first_dy


def convex_clip(subject: Iterable[Point], clip: Iterable[Point]) -> tuple[Point, ...]:
    output = tuple(subject)
    clip_points = tuple(clip)
    if polygon_signed_area(clip_points) < 0:
        clip_points = tuple(reversed(clip_points))
    for clip_start, clip_end in polygon_edges(clip_points):
        input_points = output
        output = ()
        if not input_points:
            break
        previous = input_points[-1]
        previous_inside = _cross(clip_start, clip_end, previous) >= -EPSILON
        for current in input_points:
            current_inside = _cross(clip_start, clip_end, current) >= -EPSILON
            if current_inside != previous_inside:
                output += (_line_intersection(previous, current, clip_start, clip_end),)
            if current_inside:
                output += (current,)
            previous, previous_inside = current, current_inside
    return output


def polygon_intersection_area(first: Iterable[Point], second: Iterable[Point]) -> float:
    return abs(polygon_signed_area(convex_clip(first, second)))


def bounding_dimensions(polygons: Iterable[Iterable[Point]]) -> Point:
    points = tuple(point for polygon in polygons for point in polygon)
    if not points:
        return 0.0, 0.0
    return max(point[0] for point in points) - min(point[0] for point in points), max(point[1] for point in points) - min(point[1] for point in points)


def oriented_bounding_box(polygons: Iterable[Iterable[Point]]) -> tuple[float, float, float, float, float]:
    polygon_tuple = tuple(tuple(polygon) for polygon in polygons)
    points = tuple(point for polygon in polygon_tuple for point in polygon)
    if not points:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    angles = {
        round(math.atan2(end[1] - start[1], end[0] - start[0]), 10)
        for polygon in polygon_tuple
        for start, end in polygon_edges(polygon)
    }
    best: tuple[float, float, float, float, float] | None = None
    for angle in angles:
        rotated = tuple(rotate_point(point, -angle) for point in points)
        min_x, max_x = min(point[0] for point in rotated), max(point[0] for point in rotated)
        min_y, max_y = min(point[1] for point in rotated), max(point[1] for point in rotated)
        candidate = (angle, min_x, min_y, max_x - min_x, max_y - min_y)
        if best is None or candidate[3] * candidate[4] < best[3] * best[4]:
            best = candidate
    assert best is not None
    return best


def translate_to_minimum(polygons: Iterable[Iterable[Point]]) -> Point:
    points = tuple(point for polygon in polygons for point in polygon)
    return min(point[0] for point in points), min(point[1] for point in points)
