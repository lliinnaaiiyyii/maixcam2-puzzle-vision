# Vertex-Constrained Seam Search Design

## Goal

Reduce automatic solve time for hand-cut four-piece puzzles by trying complete shared edges before the expensive partial-seam search, while retaining support for old variable-size rectangular puzzles and segmented seams.

## Scope

- Keep the calibrated A4 coordinate system, white-piece segmentation, rectangle-size flexibility, overlap checks, fill-ratio checks, and machine-coordinate output unchanged.
- Accept triangles, quadrilaterals, pentagons, and other simplified convex piece contours.
- Do not assume a fixed final rectangle width or height.
- Keep partial seams available only after the complete-edge strategy fails to produce a valid rectangle.

## Geometry Rule

For a complete shared edge `A=(a0,a1)` and the reverse-oriented edge `B=(b0,b1)`, the candidate rigid transform must satisfy:

```
distance(a0, T(b1)) <= endpoint_tolerance_mm
distance(a1, T(b0)) <= endpoint_tolerance_mm
dot(unit(a1-a0), unit(T(b0)-T(b1))) <= -cos(angle_tolerance)
```

The endpoint residual is mathematically equivalent to the length mismatch for a rigid transform. It therefore does not replace length matching; it makes the complete-edge assumption explicit and supplies a cheap, inspectable candidate score.

For a segmented seam, a short edge may terminate in the interior of a long edge after contour simplification. That layout must use point-to-segment validation instead of the two-vertex rule.

## Search Order

1. Build and rank complete-edge candidates by endpoint residual, anti-parallel direction error, and edge length error.
2. Search only complete-edge layouts first. Validate each final layout with the current rectangle fill, overlap, and target-ROI checks.
3. Return the best unambiguous complete-edge rectangle immediately.
4. Only when step 2 finds no valid rectangle, run the existing `whole_edge_skeleton_plus_partial_gap` and `partial_seam` fallbacks.
5. Retain diagnostics for candidate counts, endpoint residual rejection, strategy used, and elapsed solve time.

## Components

- `maixcam2_puzzle_vision/solver.py`
  - Add a complete-edge candidate ranking/filter that works with `PieceObservation.polygon_mm`.
  - Run the standard complete-edge search before partial search and retain the existing fallback behavior.
  - Add diagnostic counters without modifying JSON command fields.
- `tests_maixcam2_puzzle_vision/test_solver.py`
  - Verify a variable-size rectangle containing a pentagon-capable contour is solved through the complete-edge strategy.
  - Verify a known segmented seam remains solvable through fallback.
  - Verify a near-length but nonmatching full edge is rejected by the endpoint residual limit.
- `tests_maixcam2_puzzle_vision/test_pipeline.py`
  - Replay the supplied calibration image and assert that the fast path either returns a valid solution or records a deterministic fallback status.

## Acceptance Criteria

- Full-edge puzzles do not enter partial-seam search after a valid rectangle is found.
- Prior smaller and larger rectangle regression tests remain valid.
- The existing segmented-triangle and partial-gap fixtures continue to produce `OK`.
- The new calibration replay exposes candidate counts and strategy, allowing solve-time comparisons on MaixCAM.
- No configuration forces a target rectangle size.

## Risks and Safeguards

- Hand-cut edges can differ by several millimetres; endpoint tolerance must be derived from the existing length tolerance and not made unbounded.
- A hard two-vertex rule would reject segmented cuts, so it is restricted to the primary search.
- A fast candidate cannot bypass the current fill, overlap, and ROI validation.
