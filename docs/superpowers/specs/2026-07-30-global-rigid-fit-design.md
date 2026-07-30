# Constrained Global Rigid Fit Design

## Goal

Recover valid rectangular assemblies when small image-space measurement errors make exact shared-edge alignment contradictory, without changing the existing support for variable-size quadrilateral puzzles or segmented seams.

## Scope

- Preserve calibrated A4 coordinates, white-piece segmentation, command JSON, target ROI checks, and the existing complete-edge, segmented-anchor, and partial-gap solvers.
- Keep the output rectangle dimensions inferred from the pieces; do not introduce fixed width, height, or piece-shape assumptions.
- Apply the new step only after existing exact solvers report no valid rectangle.
- Accept a new layout only when it still meets explicit overlap, fill, seam-residual, ambiguity, and confidence checks.

## Problem

The failed replay contains four correctly segmented pieces, but their independently extracted vertices differ slightly from a previously successful replay. Exact edge alignment produces either too much overlap or insufficient rectangle fill. A full skeleton search completes before its state limit, so this is a geometric consistency failure rather than a missed or truncated candidate.

## Design

### Candidate Topology

Reuse the existing compatible complete-edge and partial-edge candidates to construct a connected four-piece topology. Candidate generation, edge-length tolerances, and the variable-size rectangle model remain unchanged.

### Initial Layout

Use the existing rigid edge-alignment transforms as an initial pose. The candidate still includes its original seam intervals, so a partial edge may meet the interior of a longer edge. No rule requires two coincident endpoints for a segmented seam.

### Bounded Fit

For a complete candidate layout, refine only the non-anchor piece poses with bounded translations and rotations. The fit minimizes:

1. reversed seam endpoint residuals for complete shared edges;
2. point-to-segment residuals for partial shared edges;
3. rectangle-boundary residuals after choosing the minimum-area oriented bounding rectangle;
4. overlap penalties.

The search uses a small deterministic coordinate-descent neighbourhood around the initial transforms. It runs without SciPy or other desktop-only dependencies, so it remains compatible with MaixCAM2.

### Acceptance

The best fitted candidate is accepted only when all of these hold:

- its rectangle fill ratio meets the existing partial-gap fallback threshold, `max(0.90, config.min_rectangle_fill_ratio - 0.04)`;
- every pairwise overlap is below the existing configured ratio;
- mean seam residual stays below a new bounded tolerance derived from the existing seam tolerance;
- the pose change for each piece is within the fitting bound;
- no inequivalent candidate lies inside the existing ambiguity margin.

Otherwise the solver retains the current `NO_RECTANGLE_SOLUTION` result. The planner therefore never receives a forced or low-quality robot command.

## Compatibility

- Existing variable-size four-sided pieces retain their current exact complete-edge route and return before the new fallback runs.
- Segmented triangle/pentagon puzzles retain the current partial-gap route. The fitter consumes its topology and intervals rather than replacing it with an all-vertices rule.
- The existing calibration, source and target ROI, and A4 coordinate frame are unchanged.

## Diagnostics

Fallback results add a `constrained_global_fit` diagnostic object with the topology count, fit attempts, accepted layout count, best seam residual, and rejection reason. Existing diagnostic keys are preserved.

## Tests

- A regression fixture from the scattered four-piece frame must produce a valid, unambiguous rectangle through the new fallback.
- Existing variable-size rectangular fixtures must continue to use their exact strategy and retain their inferred rectangle dimensions.
- The existing segmented pentagon/triangle reference fixture must stay solvable.
- A deliberately inconsistent topology must remain `NO_RECTANGLE_SOLUTION` rather than producing commands.
