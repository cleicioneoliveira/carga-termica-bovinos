from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ==========================================================
# POLYGON SMOOTHING
# ==========================================================
#
# This module applies smoothing to polygon boundaries in
# psychrometric space.
#
# ----------------------------------------------------------
# SCIENTIFIC CONTEXT
# ----------------------------------------------------------
#
# The polygons generated from density fields often exhibit
# irregular, jagged boundaries due to:
#
#   - discretization of the histogram grid
#   - sparse data in edge regions
#   - local fluctuations in density
#
# Smoothing is introduced as a post-processing step to:
#
#   - improve visual interpretability
#   - generate publication-quality figures
#
# ----------------------------------------------------------
# IMPORTANT WARNING
# ----------------------------------------------------------
#
# Smoothing alters the geometric representation of the data.
#
# Therefore:
#
#   - it should NOT be interpreted as a change in the data
#   - it should NOT be used for quantitative analysis
#   - it is strictly a visualization refinement
#
# Any scientific conclusions must be based on the unsmoothed
# polygon.
#
# ----------------------------------------------------------
# METHODOLOGY
# ----------------------------------------------------------
#
# A 1D Gaussian filter is applied along the polygon boundary.
#
# In this implementation:
#
#   - smoothing is applied ONLY to the W coordinate (y-axis)
#   - T (x-axis) remains unchanged
#
# This preserves the monotonic structure of temperature while
# reducing oscillations in humidity ratio.
#
# ----------------------------------------------------------
# LIMITATIONS
# ----------------------------------------------------------
#
#   - may distort sharp features
#   - may shift boundary positions
#   - assumes ordered polygon vertices
#
# For strongly irregular polygons, alternative methods such
# as spline fitting or contour smoothing may be preferable.
#
def smooth_polygon(poly: np.ndarray, cfg: dict) -> np.ndarray:
    """Smooth polygon boundary."""

    # ------------------------------------------------------
    # CHECK ENABLE FLAG
    # ------------------------------------------------------
    #
    if not cfg["smoothing"]["enabled"]:
        return poly

    if poly.shape[0] < 3:
        return poly

    # ------------------------------------------------------
    # APPLY GAUSSIAN SMOOTHING
    # ------------------------------------------------------
    #
    # Only applied along W dimension
    #
    sigma = cfg["smoothing"]["sigma"]

    smoothed = poly.copy()

    smoothed[:, 1] = gaussian_filter1d(
        smoothed[:, 1],
        sigma=sigma,
        mode="nearest"
    )

    return smoothed


# ==========================================================
# MULTI-POLYGON SMOOTHING
# ==========================================================
#
# Applies smoothing independently to multiple polygons
# (e.g., core, transition, limit zones).
#
# ----------------------------------------------------------
# DESIGN DECISION
# ----------------------------------------------------------
#
# Each polygon is smoothed independently to preserve:
#
#   - relative geometry between zones
#   - independence of density thresholds
#
# ----------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------
#
# dict[str, np.ndarray]
#   same structure as input, but with smoothed boundaries
#
def smooth_polygons(
    polys: dict,
    cfg: dict,
) -> dict:
    """Apply smoothing to multiple polygons."""

    return {
        name: smooth_polygon(poly, cfg)
        for name, poly in polys.items()
    }
