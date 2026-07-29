from __future__ import annotations

import cv2
import numpy as np

from .config import AppConfig
from .geometry import edge_length, polygon_edges, polygon_signed_area
from .models import PieceObservation


def _merge_collinear(points: np.ndarray, angle_limit_deg: float) -> np.ndarray:
    merged = np.asarray(points, dtype=np.float64)
    while len(merged) > 3:
        keep: list[np.ndarray] = []
        changed = False
        for index, current in enumerate(merged):
            previous_vector = merged[index - 1] - current
            next_vector = merged[(index + 1) % len(merged)] - current
            denominator = np.linalg.norm(previous_vector) * np.linalg.norm(next_vector)
            if denominator <= 1e-9:
                changed = True
                continue
            angle = np.degrees(np.arccos(np.clip(np.dot(previous_vector, next_vector) / denominator, -1.0, 1.0)))
            if abs(180.0 - angle) < angle_limit_deg:
                changed = True
            else:
                keep.append(current)
        if not changed or len(keep) < 3:
            break
        merged = np.asarray(keep, dtype=np.float64)
    return merged


def _roi_bounds(config: AppConfig) -> tuple[int, int, int, int]:
    x, y, width, height = config.board.source_roi_mm
    scale = config.board.pixels_per_mm
    return int(round(x * scale)), int(round(y * scale)), int(round((x + width) * scale)), int(round((y + height) * scale))


def _protected_border(roi: np.ndarray) -> np.ndarray:
    if roi.shape[0] < 3 or roi.shape[1] < 3:
        raise ValueError("source ROI is too small for background estimation")
    return np.concatenate((roi[0], roi[-1], roi[:, 0], roi[:, -1]), axis=0)


def extract_pieces(frame_bgr: np.ndarray, config: AppConfig) -> tuple[PieceObservation, ...]:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("extract_pieces expects BGR image data")
    x0, y0, x1, y1 = _roi_bounds(config)
    if x0 < 0 or y0 < 0 or x1 > frame_bgr.shape[1] or y1 > frame_bgr.shape[0]:
        return ()
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    roi = lab[y0:y1, x0:x1]
    background = np.median(_protected_border(roi), axis=0)
    distance = np.linalg.norm(roi - background, axis=2)
    mask = np.where(distance > config.segmentation.delta_e_threshold, 255, 0).astype(np.uint8)
    radius = max(1, int(round(config.segmentation.morphology_radius_mm * config.board.pixels_per_mm)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area_px = config.segmentation.min_piece_area_mm2 * config.board.pixels_per_mm**2
    observations: list[PieceObservation] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area_px:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if x <= 0 or y <= 0 or x + width >= mask.shape[1] or y + height >= mask.shape[0]:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon_px = cv2.approxPolyDP(
            contour,
            config.segmentation.polygon_epsilon_mm * config.board.pixels_per_mm,
            True,
        )[:, 0, :].astype(np.float64)
        polygon_px = _merge_collinear(polygon_px, config.segmentation.collinear_angle_deg)
        polygon_mm = tuple(tuple(point) for point in (polygon_px + np.array((x0, y0))) / config.board.pixels_per_mm)
        if not 3 <= len(polygon_mm) <= 5:
            continue
        if polygon_signed_area(polygon_mm) < 0:
            polygon_mm = tuple(reversed(polygon_mm))
        if any(edge_length(edge) < config.segmentation.min_edge_mm for edge in polygon_edges(polygon_mm)):
            continue
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-9:
            continue
        centroid_mm = (
            (moments["m10"] / moments["m00"] + x0) / config.board.pixels_per_mm,
            (moments["m01"] / moments["m00"] + y0) / config.board.pixels_per_mm,
        )
        contour_points = tuple(tuple(point) for point in contour[:, 0, :] + np.array((x0, y0)))
        observations.append(PieceObservation(0, polygon_mm, centroid_mm, contour_points))
    observations.sort(key=lambda piece: (piece.centroid_mm[1], piece.centroid_mm[0]))
    return tuple(
        PieceObservation(index, piece.polygon_mm, piece.centroid_mm, piece.contour_px)
        for index, piece in enumerate(observations)
    )
