from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class PartialSeamCandidate:
    piece_a: int
    edge_a: int
    piece_b: int
    edge_b: int
    transform_b_to_a: RigidTransform2D
    interval_a: tuple[float, float]
    interval_b: tuple[float, float]
    length_gap_ratio: float


@dataclass(frozen=True)
class _PartialState:
    transforms: dict[int, RigidTransform2D]
    used_intervals: dict[tuple[int, int], tuple[tuple[float, float], ...]]
    penalty: float


_PARTIAL_POSITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
_MIN_PARTIAL_SEAM_RATIO = 0.15
_MAX_SOURCE_EDGES_PER_TARGET = 4


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


def _interpolate(
    edge: tuple[tuple[float, float], tuple[float, float]], fraction: float
) -> tuple[float, float]:
    start, end = edge
    return (
        start[0] + (end[0] - start[0]) * fraction,
        start[1] + (end[1] - start[1]) * fraction,
    )


def _partial_alignment(
    target_edge: tuple[tuple[float, float], tuple[float, float]],
    target_start_fraction: float,
    source_edge: tuple[tuple[float, float], tuple[float, float]],
) -> RigidTransform2D:
    target_length = edge_length(target_edge)
    source_length = edge_length(source_edge)
    source_start_fraction = target_start_fraction
    source_end_fraction = source_start_fraction + source_length / max(target_length, 1e-6)
    # A cut edge has opposite boundary directions in its two adjacent pieces.
    return _align_edge_to_segment(
        _interpolate(target_edge, source_end_fraction),
        _interpolate(target_edge, source_start_fraction),
        source_edge,
    )


def _select_partial_candidates(
    candidates: list[PartialSeamCandidate],
) -> tuple[PartialSeamCandidate, ...]:
    """Keep a small, diverse set of source edges for every long target edge."""
    grouped: dict[tuple[int, int, int], dict[int, list[PartialSeamCandidate]]] = {}
    for candidate in candidates:
        edge_groups = grouped.setdefault(
            (candidate.piece_a, candidate.edge_a, candidate.piece_b),
            {},
        )
        edge_groups.setdefault(candidate.edge_b, []).append(candidate)
    selected: list[PartialSeamCandidate] = []
    for edge_groups in grouped.values():
        source_groups = sorted(
            edge_groups.values(),
            key=lambda group: (
                group[0].length_gap_ratio,
                -(
                    group[0].interval_a[1]
                    - group[0].interval_a[0]
                ),
            ),
        )
        for group in source_groups[:_MAX_SOURCE_EDGES_PER_TARGET]:
            selected.extend(group)
    return tuple(selected)


def build_partial_seam_candidates(
    pieces: tuple[PieceObservation, ...], config: SolverConfig
) -> tuple[PartialSeamCandidate, ...]:
    """Build bounded whole-edge and short-to-long shared-cut hypotheses."""
    forward: list[PartialSeamCandidate] = []
    for first, second in combinations(pieces, 2):
        for first_index, first_edge in enumerate(polygon_edges(first.polygon_mm)):
            first_length = edge_length(first_edge)
            for second_index, second_edge in enumerate(polygon_edges(second.polygon_mm)):
                second_length = edge_length(second_edge)
                if first_length >= second_length:
                    target, target_index, target_edge, target_length = first, first_index, first_edge, first_length
                    source, source_index, source_edge, source_length = second, second_index, second_edge, second_length
                else:
                    target, target_index, target_edge, target_length = second, second_index, second_edge, second_length
                    source, source_index, source_edge, source_length = first, first_index, first_edge, first_length
                if source_length / max(target_length, 1e-6) < _MIN_PARTIAL_SEAM_RATIO:
                    continue
                is_whole_edge = _lengths_match(target_length, source_length, config)
                positions = (0.0,) if is_whole_edge else _PARTIAL_POSITIONS
                for position in positions:
                    interval_start = position * (1.0 - source_length / target_length)
                    interval_end = interval_start + source_length / target_length
                    transform = _partial_alignment(target_edge, interval_start, source_edge)
                    transformed_source = apply_transform_polygon(source.polygon_mm, transform)
                    if not _opposite_sides(target_edge, target.polygon_mm, transformed_source):
                        continue
                    allowed_overlap = config.max_overlap_ratio * min(
                        abs(polygon_signed_area(target.polygon_mm)),
                        abs(polygon_signed_area(transformed_source)),
                    )
                    if polygon_intersection_area(target.polygon_mm, transformed_source) > allowed_overlap + 1e-6:
                        continue
                    forward.append(
                        PartialSeamCandidate(
                            target.piece_id,
                            target_index,
                            source.piece_id,
                            source_index,
                            transform,
                            (interval_start, interval_end),
                            (0.0, 1.0),
                            0.0 if is_whole_edge else 1.0 - source_length / target_length,
                        )
                    )
    selected = _select_partial_candidates(forward)
    candidates: list[PartialSeamCandidate] = []
    for candidate in selected:
        candidates.append(candidate)
        candidates.append(
            PartialSeamCandidate(
                candidate.piece_b,
                candidate.edge_b,
                candidate.piece_a,
                candidate.edge_a,
                inverse(candidate.transform_b_to_a),
                candidate.interval_b,
                candidate.interval_a,
                candidate.length_gap_ratio,
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.length_gap_ratio,
                candidate.piece_a,
                candidate.edge_a,
                candidate.piece_b,
                candidate.edge_b,
                candidate.interval_a,
            ),
        )
    )


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


def _has_usable_rectangle_size(width: float, height: float, config: SolverConfig) -> bool:
    return min(width, height) >= config.min_rectangle_side_mm


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
    if not _has_usable_rectangle_size(width, height, config):
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


def _transforms_signature(transforms: dict[int, RigidTransform2D]) -> tuple[tuple[int, float, float, float], ...]:
    anchor_id = min(transforms)
    anchor_inverse = inverse(transforms[anchor_id])
    signature = []
    for piece_id, transform in sorted(transforms.items()):
        relative = compose(anchor_inverse, transform)
        angle = (relative.angle_rad + math.pi) % (2.0 * math.pi) - math.pi
        signature.append((piece_id, round(angle, 3), round(relative.tx_mm, 1), round(relative.ty_mm, 1)))
    return tuple(signature)


def _layout_cluster_signature(
    transforms: dict[int, RigidTransform2D],
) -> tuple[tuple[int, float, float, float], ...]:
    """Merge the same physical layout reached through different seam orders."""
    anchor_id = min(transforms)
    anchor_inverse = inverse(transforms[anchor_id])
    signature = []
    for piece_id, transform in sorted(transforms.items()):
        relative = compose(anchor_inverse, transform)
        angle = (relative.angle_rad + math.pi) % (2.0 * math.pi) - math.pi
        signature.append(
            (
                piece_id,
                round(angle / math.radians(2.0)) * 2.0,
                round(relative.tx_mm / 2.0) * 2.0,
                round(relative.ty_mm / 2.0) * 2.0,
            )
        )
    return tuple(signature)


def _score_transforms(
    transforms: dict[int, RigidTransform2D],
    pieces_by_id: dict[int, PieceObservation],
    config: SolverConfig,
    seam_penalty: float = 0.0,
    minimum_fill_ratio: float | None = None,
) -> tuple[float, tuple[float, float], float] | None:
    polygons = tuple(
        apply_transform_polygon(pieces_by_id[piece_id].polygon_mm, transform)
        for piece_id, transform in transforms.items()
    )
    width, height, _ = _minimum_rectangle(polygons)
    short_side, long_side = sorted((width, height))
    if not _has_usable_rectangle_size(width, height, config):
        return None
    rectangle_area = width * height
    if rectangle_area <= 1e-6:
        return None
    areas = [abs(polygon_signed_area(polygon)) for polygon in polygons]
    total_area = sum(areas)
    fill_ratio = total_area / rectangle_area
    if fill_ratio < (config.min_rectangle_fill_ratio if minimum_fill_ratio is None else minimum_fill_ratio):
        return None
    overlap = sum(polygon_intersection_area(first, second) for first, second in combinations(polygons, 2))
    if overlap > max(config.max_overlap_ratio, 0.02) * min(areas):
        return None
    score = (
        (1.0 - min(fill_ratio, 1.0))
        + overlap / max(total_area, 1e-6)
        + 0.02 * seam_penalty
    )
    return score, (long_side, short_side), fill_ratio


def _minimum_rectangle_area(
    transforms: dict[int, RigidTransform2D],
    pieces_by_id: dict[int, PieceObservation],
) -> float:
    polygons = tuple(
        apply_transform_polygon(pieces_by_id[piece_id].polygon_mm, transform)
        for piece_id, transform in transforms.items()
    )
    width, height, _ = _minimum_rectangle(polygons)
    return width * height


def _intervals_overlap(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return max(first[0], second[0]) < min(first[1], second[1]) - 1e-3


def _interval_is_available(
    used_intervals: dict[tuple[int, int], tuple[tuple[float, float], ...]],
    key: tuple[int, int],
    interval: tuple[float, float],
) -> bool:
    return not any(_intervals_overlap(interval, used) for used in used_intervals.get(key, ()))


def _add_used_interval(
    used_intervals: dict[tuple[int, int], tuple[tuple[float, float], ...]],
    key: tuple[int, int],
    interval: tuple[float, float],
) -> dict[tuple[int, int], tuple[tuple[float, float], ...]]:
    updated = dict(used_intervals)
    updated[key] = updated.get(key, ()) + (interval,)
    return updated


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
                    if not _has_usable_rectangle_size(width, height, config):
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
                    score = (
                        (1.0 - min(fill_ratio, 1.0))
                        + 5.0 * overlap / max(total_area, 1e-6)
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


def _solve_skeleton_gap_layout(
    pieces: tuple[PieceObservation, ...],
    config: SolverConfig,
) -> AssemblyResult | None:
    """Attach one partial-edge piece to a connected whole-edge skeleton."""
    if len(pieces) < 3:
        return None
    pieces_by_id = {piece.piece_id: piece for piece in pieces}
    skeleton_config = replace(
        config,
        edge_length_tolerance_mm=max(config.edge_length_tolerance_mm, 4.0),
        edge_length_tolerance_ratio=max(config.edge_length_tolerance_ratio, 0.12),
    )
    full_candidates = build_seam_candidates(pieces, skeleton_config)
    partial_candidates = build_partial_seam_candidates(pieces, config)
    if not full_candidates or not partial_candidates:
        return None
    by_target: dict[int, tuple[SeamCandidate, ...]] = {}
    for candidate in full_candidates:
        by_target[candidate.piece_a] = by_target.get(candidate.piece_a, ()) + (candidate,)

    skeletons: list[_State] = []
    seen: set[tuple[tuple[int, float, float, float], ...]] = set()
    states_visited = 0
    state_limit = min(config.max_states, 800)

    def search(state: _State) -> None:
        nonlocal states_visited
        if states_visited >= state_limit:
            return
        states_visited += 1
        if len(state.transforms) == len(pieces) - 1:
            signature = _transforms_signature(state.transforms)
            if signature not in seen:
                seen.add(signature)
                skeletons.append(state)
            return
        for target_id, target_transform in tuple(state.transforms.items()):
            for candidate in by_target.get(target_id, ()):
                if (
                    candidate.piece_b in state.transforms
                    or (candidate.piece_a, candidate.edge_a) in state.used_edges
                    or (candidate.piece_b, candidate.edge_b) in state.used_edges
                ):
                    continue
                source_transform = compose(target_transform, candidate.transform_b_to_a)
                source_polygon = apply_transform_polygon(pieces_by_id[candidate.piece_b].polygon_mm, source_transform)
                if any(
                    polygon_intersection_area(
                        source_polygon,
                        apply_transform_polygon(pieces_by_id[existing_id].polygon_mm, existing_transform),
                    )
                    > config.max_overlap_ratio
                    * min(
                        abs(polygon_signed_area(source_polygon)),
                        abs(polygon_signed_area(pieces_by_id[existing_id].polygon_mm)),
                    )
                    + 1e-6
                    for existing_id, existing_transform in state.transforms.items()
                ):
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
    if not skeletons:
        return None

    partial_by_target: dict[int, tuple[PartialSeamCandidate, ...]] = {}
    for candidate in partial_candidates:
        partial_by_target[candidate.piece_a] = partial_by_target.get(candidate.piece_a, ()) + (candidate,)
    layouts: dict[tuple[tuple[int, float, float, float], ...], tuple[float, dict[int, RigidTransform2D], tuple[float, float], float]] = {}
    pruned_overlap = 0
    for skeleton in skeletons:
        remaining_id = next(piece.piece_id for piece in pieces if piece.piece_id not in skeleton.transforms)
        placed_polygons = {
            piece_id: apply_transform_polygon(pieces_by_id[piece_id].polygon_mm, transform)
            for piece_id, transform in skeleton.transforms.items()
        }
        for target_id, target_transform in skeleton.transforms.items():
            for candidate in partial_by_target.get(target_id, ()):
                if candidate.piece_b != remaining_id:
                    continue
                if (
                    (candidate.piece_a, candidate.edge_a) in skeleton.used_edges
                    or (candidate.piece_b, candidate.edge_b) in skeleton.used_edges
                ):
                    continue
                source_transform = compose(target_transform, candidate.transform_b_to_a)
                source_polygon = apply_transform_polygon(pieces_by_id[remaining_id].polygon_mm, source_transform)
                collision = False
                for existing_id, existing_polygon in placed_polygons.items():
                    if existing_id == target_id:
                        continue
                    allowed_overlap = max(config.max_overlap_ratio, 0.02) * min(
                        abs(polygon_signed_area(source_polygon)),
                        abs(polygon_signed_area(existing_polygon)),
                    )
                    if polygon_intersection_area(source_polygon, existing_polygon) > allowed_overlap + 1e-6:
                        collision = True
                        break
                if collision:
                    pruned_overlap += 1
                    continue
                transforms = {**skeleton.transforms, remaining_id: source_transform}
                scored = _score_transforms(
                    transforms,
                    pieces_by_id,
                    config,
                    candidate.length_gap_ratio,
                    minimum_fill_ratio=max(0.90, config.min_rectangle_fill_ratio - 0.04),
                )
                if scored is None:
                    continue
                signature = _layout_cluster_signature(transforms)
                entry = (scored[0], transforms, scored[1], scored[2])
                if signature not in layouts or entry[0] < layouts[signature][0]:
                    layouts[signature] = entry
    diagnostics = {
        "strategy": "whole_edge_skeleton_plus_partial_gap",
        "candidate_count": len(full_candidates) + len(partial_candidates),
        "skeleton_edge_tolerance_ratio": skeleton_config.edge_length_tolerance_ratio,
        "skeleton_count": len(skeletons),
        "states_visited": states_visited,
        "state_limit_reached": states_visited >= state_limit,
        "valid_layout_count": len(layouts),
        "pruned_overlap": pruned_overlap,
    }
    if not layouts:
        return AssemblyResult(SolveStatus.NO_RECTANGLE_SOLUTION, diagnostics=diagnostics)
    ranked = sorted(layouts.values(), key=lambda entry: entry[0])
    best = ranked[0]
    diagnostics["best_score"] = best[0]
    if len(ranked) > 1:
        diagnostics["second_score"] = ranked[1][0]
        if ranked[1][0] - best[0] < config.ambiguity_margin:
            return AssemblyResult(SolveStatus.AMBIGUOUS, diagnostics=diagnostics)
    return AssemblyResult(
        SolveStatus.OK,
        transforms=best[1],
        rectangle_size_mm=best[2],
        fill_ratio=best[3],
        score=best[0],
        diagnostics=diagnostics,
    )


def _solve_partial_seam_layout(
    pieces: tuple[PieceObservation, ...],
    config: SolverConfig,
) -> AssemblyResult | None:
    if len(pieces) < 2:
        return None
    pieces_by_id = {piece.piece_id: piece for piece in pieces}
    candidates = build_partial_seam_candidates(pieces, config)
    if not candidates:
        return None
    candidates_by_target: dict[int, tuple[PartialSeamCandidate, ...]] = {}
    for candidate in candidates:
        candidates_by_target[candidate.piece_a] = candidates_by_target.get(candidate.piece_a, ()) + (candidate,)

    layouts: dict[tuple[tuple[int, float, float, float], ...], tuple[float, dict[int, RigidTransform2D], tuple[float, float], float]] = {}
    queue: list[tuple[float, int, _PartialState]] = []
    best_seen: dict[tuple[tuple[int, float, float, float], ...], float] = {}
    push_index = 0
    for root in pieces:
        state = _PartialState({root.piece_id: RigidTransform2D()}, {}, 0.0)
        best_seen[_transforms_signature(state.transforms)] = 0.0
        heapq.heappush(queue, (0.0, push_index, state))
        push_index += 1

    states_visited = 0
    state_limit_reached = False
    pruned_overlap = 0
    pruned_duplicate = 0
    pruned_interval = 0
    pruned_area = 0
    total_area = sum(abs(polygon_signed_area(piece.polygon_mm)) for piece in pieces)
    maximum_rectangle_area = total_area / max(config.min_rectangle_fill_ratio, 1e-6)
    state_limit = min(config.max_states, 320)
    while queue and states_visited < state_limit:
        _priority, _tie_breaker, state = heapq.heappop(queue)
        signature = _transforms_signature(state.transforms)
        if state.penalty > best_seen.get(signature, float("inf")) + 1e-6:
            pruned_duplicate += 1
            continue
        states_visited += 1
        if len(state.transforms) == len(pieces):
            scored = _score_transforms(state.transforms, pieces_by_id, config, state.penalty)
            if scored is not None:
                entry = (scored[0], state.transforms, scored[1], scored[2])
                layout_signature = _layout_cluster_signature(state.transforms)
                if layout_signature not in layouts or entry[0] < layouts[layout_signature][0]:
                    layouts[layout_signature] = entry
            continue

        placed_polygons = {
            piece_id: apply_transform_polygon(pieces_by_id[piece_id].polygon_mm, transform)
            for piece_id, transform in state.transforms.items()
        }
        for target_id, target_transform in tuple(state.transforms.items()):
            for candidate in candidates_by_target.get(target_id, ()):
                source_id = candidate.piece_b
                if source_id in state.transforms:
                    continue
                target_key = (candidate.piece_a, candidate.edge_a)
                source_key = (candidate.piece_b, candidate.edge_b)
                if not _interval_is_available(state.used_intervals, target_key, candidate.interval_a):
                    pruned_interval += 1
                    continue
                if not _interval_is_available(state.used_intervals, source_key, candidate.interval_b):
                    pruned_interval += 1
                    continue
                source_transform = compose(target_transform, candidate.transform_b_to_a)
                source_polygon = apply_transform_polygon(pieces_by_id[source_id].polygon_mm, source_transform)
                collision = False
                for existing_id, existing_polygon in placed_polygons.items():
                    if existing_id == target_id:
                        continue
                    allowed_overlap = max(config.max_overlap_ratio, 0.02) * min(
                        abs(polygon_signed_area(source_polygon)),
                        abs(polygon_signed_area(existing_polygon)),
                    )
                    if polygon_intersection_area(source_polygon, existing_polygon) > allowed_overlap + 1e-6:
                        collision = True
                        break
                if collision:
                    pruned_overlap += 1
                    continue
                used = _add_used_interval(state.used_intervals, target_key, candidate.interval_a)
                used = _add_used_interval(used, source_key, candidate.interval_b)
                updated_transforms = {**state.transforms, source_id: source_transform}
                if _minimum_rectangle_area(updated_transforms, pieces_by_id) > maximum_rectangle_area:
                    pruned_area += 1
                    continue
                penalty = state.penalty + candidate.length_gap_ratio
                next_signature = _transforms_signature(updated_transforms)
                if best_seen.get(next_signature, float("inf")) <= penalty + 1e-6:
                    pruned_duplicate += 1
                    continue
                best_seen[next_signature] = penalty
                heapq.heappush(
                    queue,
                    (
                        penalty - 0.1 * len(updated_transforms),
                        push_index,
                        _PartialState(updated_transforms, used, penalty),
                    ),
                )
                push_index += 1
    if queue:
        state_limit_reached = True
    diagnostics = {
        "strategy": "partial_seam",
        "candidate_count": len(candidates),
        "states_visited": states_visited,
        "state_limit_reached": state_limit_reached,
        "valid_layout_count": len(layouts),
        "pruned_overlap": pruned_overlap,
        "pruned_duplicate": pruned_duplicate,
        "pruned_interval": pruned_interval,
        "pruned_area": pruned_area,
    }
    if not layouts:
        return AssemblyResult(SolveStatus.NO_RECTANGLE_SOLUTION, diagnostics=diagnostics)
    ranked = sorted(layouts.values(), key=lambda entry: entry[0])
    best = ranked[0]
    diagnostics["best_score"] = best[0]
    if len(ranked) > 1:
        diagnostics["second_score"] = ranked[1][0]
        if ranked[1][0] - best[0] < config.ambiguity_margin:
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
        skeleton_gap = _solve_skeleton_gap_layout(pieces, config)
        if skeleton_gap is not None and skeleton_gap.status is not SolveStatus.NO_RECTANGLE_SOLUTION:
            return skeleton_gap
        if skeleton_gap is not None and skeleton_gap.diagnostics.get("skeleton_count", 0) > 0:
            return AssemblyResult(
                SolveStatus.NO_RECTANGLE_SOLUTION,
                diagnostics={
                    **diagnostics,
                    "skeleton_gap": skeleton_gap.diagnostics,
                    "reason": "whole_edge_skeleton_rejected_without_rectangle",
                },
            )
        partial = _solve_partial_seam_layout(pieces, config)
        if partial is not None and partial.status is not SolveStatus.NO_RECTANGLE_SOLUTION:
            return partial
        if skeleton_gap is not None:
            diagnostics["skeleton_gap"] = skeleton_gap.diagnostics
        if partial is not None:
            diagnostics["partial_seam"] = partial.diagnostics
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
