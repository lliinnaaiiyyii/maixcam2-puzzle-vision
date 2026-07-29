from __future__ import annotations

import numpy as np
import cv2

from .geometry import apply_transform_polygon
from .models import PieceObservation, RigidTransform2D


def _gradient(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim == 3:
        values = values.mean(axis=2)
    if values.ndim != 2:
        raise ValueError("seam strip must be a grayscale or BGR image")
    horizontal = np.diff(values, axis=1, prepend=values[:, :1])
    vertical = np.diff(values, axis=0, prepend=values[:1, :])
    magnitude = np.hypot(horizontal, vertical)
    deviation = float(magnitude.std())
    if deviation < 1e-6:
        return np.zeros_like(magnitude)
    return (magnitude - float(magnitude.mean())) / deviation


def seam_mismatch(left_strip: np.ndarray, right_strip: np.ndarray) -> float:
    """Return illumination-normalized mismatch after reversing the neighboring strip."""
    left = _gradient(left_strip)
    right = _gradient(np.fliplr(np.asarray(right_strip)))
    if left.shape != right.shape:
        raise ValueError("seam strips must have matching shapes")
    return float(np.mean(np.abs(left - right)))


def contact_gradient_score(layers: list[np.ndarray], masks: list[np.ndarray]) -> float:
    """Measure discontinuity on contact bands between already-placed pieces."""
    if len(layers) != len(masks) or len(layers) < 2:
        return float("inf")
    composite = np.zeros_like(layers[0])
    for layer, mask in zip(layers, masks):
        composite[mask > 0] = layer[mask > 0]
    gray = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY) if composite.ndim == 3 else composite
    horizontal = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    vertical = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3))
    gradient = np.hypot(horizontal, vertical)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    scores = []
    for first_index in range(len(masks)):
        for second_index in range(first_index + 1, len(masks)):
            contact = cv2.bitwise_and(cv2.dilate(masks[first_index], kernel), cv2.dilate(masks[second_index], kernel))
            if int(np.count_nonzero(contact)) >= 4:
                scores.append(float(np.mean(gradient[contact > 0])))
    return float(np.mean(scores)) if scores else float("inf")


def make_layout_pattern_scorer(
    frame_bgr: np.ndarray,
    pieces: tuple[PieceObservation, ...],
    pixels_per_mm: float,
):
    """Return a scorer for geometrically close layouts of the captured pieces."""

    def score(transforms: dict[int, RigidTransform2D]) -> float:
        placed = [apply_transform_polygon(piece.polygon_mm, transforms[piece.piece_id]) for piece in pieces]
        all_points = [point for polygon in placed for point in polygon]
        margin = 6
        min_x = min(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_x = max(point[0] for point in all_points)
        max_y = max(point[1] for point in all_points)
        width = max(1, int(round((max_x - min_x) * pixels_per_mm)) + margin * 2)
        height = max(1, int(round((max_y - min_y) * pixels_per_mm)) + margin * 2)
        layers: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for piece in pieces:
            transform = transforms[piece.piece_id]
            cosine = float(np.cos(transform.angle_rad))
            sine = float(np.sin(transform.angle_rad))
            matrix = np.array(
                (
                    (cosine, -sine, pixels_per_mm * (transform.tx_mm - min_x) + margin),
                    (sine, cosine, pixels_per_mm * (transform.ty_mm - min_y) + margin),
                ),
                dtype=np.float32,
            )
            source_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            source_points = np.rint(np.asarray(piece.polygon_mm) * pixels_per_mm).astype(np.int32)
            cv2.fillPoly(source_mask, [source_points], 255)
            layers.append(cv2.warpAffine(frame_bgr, matrix, (width, height), flags=cv2.INTER_LINEAR))
            masks.append(cv2.warpAffine(source_mask, matrix, (width, height), flags=cv2.INTER_NEAREST))
        return contact_gradient_score(layers, masks)

    return score
