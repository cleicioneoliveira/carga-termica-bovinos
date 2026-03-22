from __future__ import annotations

import numpy as np


# ==========================================================
# MULTI-ZONE SEGMENTATION (DENSITY-BASED)
# ==========================================================
#
# This module splits the density field into multiple regions
# representing different levels of thermal comfort.
#
# ----------------------------------------------------------
# SCIENTIFIC CONTEXT
# ----------------------------------------------------------
#
# Traditional approaches attempt to define a single "comfort
# zone". However, empirical data rarely supports a binary
# separation between comfort and discomfort.
#
# Instead, comfort should be interpreted as a continuum.
#
# This function implements a multi-zone approach:
#
#   - CORE        → highest density (optimal comfort)
#   - TRANSITION  → intermediate density (acceptable)
#   - LIMIT       → low-density boundary (tolerance)
#
# These zones are derived from the density distribution
# itself, not from predefined physiological thresholds.
#
# ----------------------------------------------------------
# METHODOLOGICAL INTERPRETATION
# ----------------------------------------------------------
#
# The density field represents the probability distribution
# of observed comfort conditions in psychrometric space.
#
# By applying percentiles:
#
#   - we identify statistically dominant regions
#   - we avoid arbitrary thresholds
#   - we preserve dataset-driven structure
#
# IMPORTANT:
#
# These zones are NOT:
#   - independent clusters
#   - hard biological limits
#
# They are:
#   - statistical layers of the same distribution
#
# ----------------------------------------------------------
# INPUT ASSUMPTIONS
# ----------------------------------------------------------
#
# points:
#   array of shape (N, 2)
#   each row = (T, W) coordinate
#
# values:
#   2D density matrix (same grid used to generate points)
#
# cfg:
#   configuration dictionary containing:
#       cfg["zones"]["core_percentile"]
#       cfg["zones"]["transition_percentile"]
#       cfg["zones"]["limit_percentile"]
#
# ----------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------
#
# dict:
#   {
#       "core": np.ndarray,
#       "transition": np.ndarray,
#       "limit": np.ndarray
#   }
#
# Each entry contains a point cloud corresponding to the zone.
#
def build_zones(
    points: np.ndarray,
    values: np.ndarray,
    cfg: dict,
) -> dict[str, np.ndarray]:
    """Split density into core/transition/limit zones."""

    # ------------------------------------------------------
    # EXTRACT VALID DENSITY VALUES
    # ------------------------------------------------------
    #
    flat = values[~np.isnan(values)]

    if flat.size == 0:
        raise ValueError("No valid density values available for zone definition.")

    # ------------------------------------------------------
    # COMPUTE PERCENTILE THRESHOLDS
    # ------------------------------------------------------
    #
    # These thresholds define the boundaries between zones.
    #
    p_core = np.percentile(flat, cfg["zones"]["core_percentile"])
    p_mid = np.percentile(flat, cfg["zones"]["transition_percentile"])
    p_low = np.percentile(flat, cfg["zones"]["limit_percentile"])

    # ------------------------------------------------------
    # INITIALIZE STRUCTURE
    # ------------------------------------------------------
    #
    zones = {
        "core": [],
        "transition": [],
        "limit": [],
    }

    # ------------------------------------------------------
    # ASSIGN POINTS TO ZONES
    # ------------------------------------------------------
    #
    # IMPORTANT:
    #
    # The mapping between `points` and `values` depends on the
    # exact iteration order used in extract_points().
    #
    # Therefore:
    #   - iteration order MUST remain consistent
    #   - idx is used to synchronize both structures
    #
    idx = 0

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isnan(values[i, j]):

                v = values[i, j]

                # --------------------------------------------------
                # HIERARCHICAL ASSIGNMENT
                # --------------------------------------------------
                #
                # Zones are mutually exclusive in this implementation.
                #
                # That means:
                #   core ⊂ transition ⊂ limit (conceptually)
                #
                # but here we assign each point to a single class
                # for visualization purposes.
                #
                if v >= p_core:
                    zones["core"].append(points[idx])

                elif v >= p_mid:
                    zones["transition"].append(points[idx])

                elif v >= p_low:
                    zones["limit"].append(points[idx])

                # values below p_low are discarded

                idx += 1

    # ------------------------------------------------------
    # CONVERT TO NUMPY ARRAYS
    # ------------------------------------------------------
    #
    result = {}

    for k, v in zones.items():
        if len(v) == 0:
            result[k] = np.empty((0, 2))
        else:
            result[k] = np.array(v)

    return result
