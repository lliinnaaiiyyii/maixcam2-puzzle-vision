from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, permutations, product
from typing import Callable

from .config import SolverConfig
from .geometry import (
    align_reversed_edges,
    apply_transform_polygon,
    compose,
    edge_length,
    inverse,
    polygon_centroid,
    polygon_edges,
    polygon_intersection_area,
    polygon_signed_area,
    rotate_point,
)
from .models import AssemblyResult, PieceObservation, RigidTransform2D, SolveStatus


@dataclass(frozen=True)
class SeamCandidate:
    piece_a: int
    edge_a: int
    piece_b: int
    edge_b: int
    transform_b_to_a: RigidTransform2D
    length_error_mm: float


@dataclass(frozen=True)
class _State:
    transforms: dict[int, RigidTransform2D]
    used_edges: frozenset[tuple[int, int]]
    seams: tuple[SeamCandidate, ...]


def _cross(edge: tuple[tuple[float, float], tuple[float, float]], point: tuple[float, float]) -> float:
    (x0, y0), (x1, y1) = edge
    return (x1 - x0) * (point[1] - y0) - (y1 - y0) * (point[0] - x0)


def _opposite_sides(
    edge: tuple[tuple[float, float], tuple[float, float]],
    first_polygon: tuple[tuple[float, float], ...],
    second_polygon: tuple[tuple[float, float], ...],
) -> bool:
    return _cross(edge, polygon_centroid(first_polygon)) * _cross(edge, polygon_centroid(second_polygon)) < -1e-6


def _lengths_match(first: float, second: float, config: SolverConfig) -> bool:
    tolerance = max(config.edge_length_tolerance_mm, config.edge_length_tolerance_ratio * max(first, second))
    return abs(first - second) <= tolerance


def build_seam_candidates(pieces: tuple[PieceObservation, ...], config: SolverConfig) -> tuple[SeamCandidate, ...]:
    candidates: list[SeamCandidate] = []
    for first, second in combinations(pieces, 2):
        for first_index, first_edge in enumerate(polygon_edges(first.polygon_mm)):
            first_length = edge_length(first_edge)
            for second_index, second_edge in enumerate(polygon_edges(second.polygon_mm)):
                second_length = edge_length(second_edge)
                if not _lengths_match(first_length, second_length, config):
                    continue
                transform = align_reversed_edges(first_edge, second_edge)
                transformed_second = apply_transform_polygon(second.polygon_mm, transform)
                if not _opposite_sides(first_edge, first.polygon_mm, transformed_second):
                    continue
                if polygon_intersection_area(first.polygon_mm, transformed_second) > 1e-4:
                    continue
                error = abs(first_length - second_length)
                candidates.append(SeamCandidate(first.piece_id, first_index, second.piece_id, second_index, transform, error))
                candidates.append(
                    SeamCandidate(
                        second.piece_id,
                        second_index,
                        first.piece_id,
                        first_index,
                        inverse(transform),
                        error,
                    )
                )
    return tuple(candidates)


def _minimum_rectangle(polygons: tuple[tuple[tuple[float, float], ...], ...]) -> tuple[float, float, float]:
    points = tuple(point for polygon in polygons for point in polygon)
    if not points:
        return 0.0, 0.0, 0.0
    angles = {round(math.atan2(end[1] - start[1], end[0] - start[0]), 10) for polygon in polygons for start, end in polygon_edges(polygon)}
    best: tuple[float, float, float] | None = None
    for angle in angles:
        rotated = tuple(rotate_point(point, -angle) for point in points)
        width = max(point[0] for point in rotated) - min(point[0] for point in rotated)
        height = max(point[1] for point in rotated) - min(point[1] for point in rotated)
        candidate = width, height, angle
        if best is None or width * height < best[0] * best[1]:
            best = candidate
    assert best is not None
    return best


def _seam_residual(state: _State, pieces_by_id: dict[int, PieceObservation]) -> float:
    if not state.seams:
        return 0.0
    residuals = []
    for seam in state.seams:
        first_edge = polygon_edges(pieces_by_id[seam.piece_a].polygon_mm)[seam.edge_a]
        second_edge = polygon_edges(pieces_by_id[seam.piece_b].polygon_mm)[seam.edge_b]
        first_transform = state.transforms[seam.piece_a]
        second_transform = state.transforms[seam.piece_b]
        first_start, first_end = apply_transform_polygon(first_edge, first_transform)
        second_start, second_end = apply_transform_polygon(second_edge, second_transform)
        residuals.extend((math.dist(first_end, second_start), math.dist(first_start, second_end)))
    return sum(residuals) / len(residuals)


def _score_state(state: _State, pieces_by_id: dict[int, PieceObservation], config: SolverConfig) -> tuple[float, tuple[float, float], float] | None:
    polygons = tuple(
        apply_transform_polygon(pieces_by_id[piece_id].polygon_mm, transform)
        for piece_id, transform in state.transforms.items()
    )
    width, height, _ = _minimum_rectangle(polygons)
    short_side, long_side = sorted((width, height))
    if not 50.0 <= short_side <= 90.0 or not 90.0 <= long_side <= 120.0:
        return None
    rectangle_area = width * height
    if rectangle_area <= 1e-6:
        return None
    areas = [abs(polygon_signed_area(polygon)) for polygon in polygons]
    total_area = sum(areas)
    fill_ratio = total_area / rectangle_area
    if fill_ratio < config.min_rectangle_fill_ratio:
        return None
    overlap = sum(polygon_intersection_area(first, second) for first, second in combinations(polygons, 2))
    if overlap > config.max_overlap_ratio * min(areas):
        return None
    residual = _seam_residual(state, pieces_by_id)
    if residual > config.max_seam_residual_mm:
        return None
    score = (1.0 - min(fill_ratio, 1.0)) + overlap / max(total_area, 1e-6) + residual / 100.0
    return score, (long_side, short_side), fill_ratio


def _layout_signature(state: _State) -> tuple[tuple[int, float, float, float], ...]:
    anchor_id = min(state.transforms)
    anchor_inverse = inverse(state.transforms[anchor_id])
    signature = []
    for piece_id, transform in sorted(state.transforms.items()):
        relative = compose(anchor_inverse, transform)
        angle = (relative.angle_rad + math.pi) % (2.0 * math.pi) - math.pi
        signature.append((piece_id, round(angle, 4), round(relative.tx_mm, 3), round(relative.ty_mm, 3)))
    return tuple(signature)


def _align_edge_to_segment(
    segment_start: tuple[float, float],
    segment_end: tuple[float, float],
    source_edge: tuple[tuple[float, float], tuple[float, float]],
) -> RigidTransform2D:
    """Map a source edge onto a possibly shorter target segment without scaling."""
    source_start, source_end = source_edge
    source_angle = math.atan2(source_end[1] - source_start[1], source_end[0] - source_start[0])
    target_angle = math.atan2(segment_end[1] - segment_start[1], segment_end[0] - segment_start[0])
    angle = target_angle - source_angle
    rotated_start = rotate_point(source_start, angle)
    return RigidTransform2D(angle, segment_start[0] - rotated_start[0], segment_start[1] - rotated_start[1])


def _solve_segmented_anchor_layout(
    pieces: tuple[PieceObservation, ...], config: SolverConfig
) -> AssemblyResult | None:
    """Recover the Figure 2 layout where three pieces partition one triangle edge."""
    if len(pieces) != 4:
        return None
    candidates: list[tuple[float, dict[int, RigidTransform2D], tuple[float, float], float, float]] = []
    for anchor in pieces:
        if len(anchor.polygon_mm) != 3:
            continue
        remaining = tuple(piece for piece in pieces if piece.piece_id != anchor.piece_id)
        for anchor_edge in polygon_edges(anchor.polygon_mm):
            anchor_length = edge_length(anchor_edge)
            for ordered_pieces in permutations(remaining):
                edge_ranges = tuple(range(len(piece.polygon_mm)) for piece in ordered_pieces)
                for edge_indexes in product(*edge_ranges):
                    selected_edges = tuple(
                        polygon_edges(piece.polygon_mm)[edge_index]
                        for piece, edge_index in zip(ordered_pieces, edge_indexes)
                    )
                    selected_lengths = tuple(edge_length(edge) for edge in selected_edges)
                    selected_total = sum(selected_lengths)
                    if selected_total <= 1e-6 or abs(selected_total - anchor_length) > max(4.0, anchor_length * 0.12):
                        continue
                    transforms = {anchor.piece_id: RigidTransform2D()}
                    offset = 0.0
                    for piece, source_edge, source_length in zip(ordered_pieces, selected_edges, selected_lengths):
                        next_offset = offset + source_length * anchor_length / selected_total
                        start_fraction = next_offset / anchor_length
                        end_fraction = offset / anchor_length
                        segment_start = (
                            anchor_edge[0][0] + (anchor_edge[1][0] - anchor_edge[0][0]) * start_fraction,
                            anchor_edge[0][1] + (anchor_edge[1][1] - anchor_edge[0][1]) * start_fraction,
                        )
                        segment_end = (
                            anchor_edge[0][0] + (anchor_edge[1][0] - anchor_edge[0][0]) * end_fraction,
                            anchor_edge[0][1] + (anchor_edge[1][1] - anchor_edge[0][1]) * end_fraction,
                        )
                        transforms[piece.piece_id] = _align_edge_to_segment(segment_start, segment_end, source_edge)
                        offset = next_offset
                    polygons = tuple(
                        apply_transform_polygon(piece.polygon_mm, transforms[piece.piece_id])
                        for piece in pieces
                    )
                    width, height, _ = _minimum_rectangle(polygons)
                    short_side, long_side = sorted((width, height))
                    if not 50.0 <= short_side <= 90.0 or not 90.0 <= long_side <= 120.0:
                        continue
                    rectangle_area = width * height
                    if rectangle_area <= 1e-6:
                        continue
                    areas = tuple(abs(polygon_signed_area(polygon)) for polygon in polygons)
                    total_area = sum(areas)
                    fill_ratio = total_area / rectangle_area
                    if fill_ratio < max(0.90, config.min_rectangle_fill_ratio - 0.02):
                        continue
                    overlap = sum(polygon_intersection_area(first, second) for first, second in combinations(polygons, 2))
                    if overlap > 0.03 * total_area:
                        continue
                    expected_aspect = 100.0 / 60.0
                    score = (
                        (1.0 - min(fill_ratio, 1.0))
                        + 5.0 * overlap / max(total_area, 1e-6)
                        + 0.1 * abs(long_side / short_side - expected_aspect)
                    )
                    candidates.append((score, transforms, (long_side, short_side), fill_ratio, overlap))
    if not candidates:
        return None
    candidates.sort(key=lambda candidate: candidate[0])
    best = candidates[0]
    diagnostics = {
        "strategy": "segmented_triangle_anchor",
        "candidate_count": len(candidates),
        "best_score": best[0],
        "overlap_mm2": best[4],
    }
    if len(candidates) > 1:
        diagnostics["second_score"] = candidates[1][0]
        if candidates[1][0] - best[0] < config.ambiguity_margin:
            return AssemblyResult(SolveStatus.AMBIGUOUS, diagnostics=diagnostics)
    return AssemblyResult(
        SolveStatus.OK,
        transforms=best[1],
        rectangle_size_mm=best[2],
        fill_ratio=best[3],
        score=best[0],
        diagnostics=diagnostics,
    )


def solve_layout(
    pieces: tuple[PieceObservation, ...],
    config: SolverConfig,
    pattern_scorer: Callable[[dict[int, RigidTransform2D]], float] | None = None,
) -> AssemblyResult:
    if not 1 <= len(pieces) <= 4:
        return AssemblyResult(SolveStatus.INVALID_PIECE_COUNT)
    pieces_by_id = {piece.piece_id: piece for piece in pieces}
    candidates = build_seam_candidates(pieces, config)
    candidates_by_target: dict[int, tuple[SeamCandidate, ...]] = {}
    for candidate in candidates:
        candidates_by_target[candidate.piece_a] = candidates_by_target.get(candidate.piece_a, ()) + (candidate,)

    layouts: dict[tuple[tuple[int, float, float, float], ...], tuple[float, _State, tuple[float, float], float]] = {}
    states_visited = 0
    state_limit_reached = False

    def search(state: _State) -> None:
        nonlocal states_visited, state_limit_reached
        if states_visited >= config.max_states:
            state_limit_reached = True
            return
        states_visited += 1
        if len(state.transforms) == len(pieces):
            scored = _score_state(state, pieces_by_id, config)
            if scored is not None:
                signature = _layout_signature(state)
                entry = (scored[0], state, scored[1], scored[2])
                if signature not in layouts or entry[0] < layouts[signature][0]:
                    layouts[signature] = entry
            return
        for target_id, target_transform in tuple(state.transforms.items()):
            for candidate in candidates_by_target.get(target_id, ()):
                if candidate.piece_b in state.transforms or (candidate.piece_a, candidate.edge_a) in state.used_edges or (candidate.piece_b, candidate.edge_b) in state.used_edges:
                    continue
                source_transform = compose(target_transform, candidate.transform_b_to_a)
                source_polygon = apply_transform_polygon(pieces_by_id[candidate.piece_b].polygon_mm, source_transform)
                collision = False
                for existing_id, existing_transform in state.transforms.items():
                    existing_polygon = apply_transform_polygon(pieces_by_id[existing_id].polygon_mm, existing_transform)
                    allowed_overlap = config.max_overlap_ratio * min(
                        abs(polygon_signed_area(source_polygon)), abs(polygon_signed_area(existing_polygon))
                    )
                    if polygon_intersection_area(source_polygon, existing_polygon) > allowed_overlap + 1e-6:
                        collision = True
                        break
                if collision:
                    continue
                search(
                    _State(
                        {**state.transforms, candidate.piece_b: source_transform},
                        state.used_edges | {(candidate.piece_a, candidate.edge_a), (candidate.piece_b, candidate.edge_b)},
                        state.seams + (candidate,),
                    )
                )

    for root in pieces:
        search(_State({root.piece_id: RigidTransform2D()}, frozenset(), ()))
    diagnostics = {
        "candidate_count": len(candidates),
        "states_visited": states_visited,
        "state_limit_reached": state_limit_reached,
        "valid_layout_count": len(layouts),
    }
    if not layouts:
        segmented = _solve_segmented_anchor_layout(pieces, config)
        if segmented is not None:
            return segmented
        return AssemblyResult(SolveStatus.NO_RECTANGLE_SOLUTION, diagnostics=diagnostics)
    ranked = sorted(layouts.values(), key=lambda entry: entry[0])
    best = ranked[0]
    diagnostics["best_score"] = best[0]
    if len(ranked) > 1:
        diagnostics["second_score"] = ranked[1][0]
        if ranked[1][0] - best[0] < config.ambiguity_margin:
            if pattern_scorer is None:
                return AssemblyResult(SolveStatus.AMBIGUOUS, diagnostics=diagnostics)
            close_layouts = [entry for entry in ranked if entry[0] - best[0] < config.ambiguity_margin]
            texture_ranked = sorted((float(pattern_scorer(entry[1].transforms)), entry) for entry in close_layouts)
            diagnostics["best_texture_score"] = texture_ranked[0][0]
            if len(texture_ranked) > 1:
                diagnostics["second_texture_score"] = texture_ranked[1][0]
                if texture_ranked[1][0] - texture_ranked[0][0] < config.pattern_margin:
                    return AssemblyResult(SolveStatus.AMBIGUOUS, diagnostics=diagnostics)
            best = texture_ranked[0][1]
    return AssemblyResult(
        SolveStatus.OK,
        transforms=best[1].transforms,
        rectangle_size_mm=best[2],
        fill_ratio=best[3],
        score=best[0],
        diagnostics=diagnostics,
    )
