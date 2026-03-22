from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
import alphashape


# ==========================================================
# POLYGON EXTRACTION FROM POINT CLOUD
# ==========================================================
#
# This module converts a set of points in psychrometric space
# into a polygon representing the empirical comfort zone.
#
# ----------------------------------------------------------
# SCIENTIFIC CONTEXT
# ----------------------------------------------------------
#
# After density filtering and clustering, the remaining
# points represent regions of interest in the (T, W) space.
#
# However, these points are discrete.
#
# To obtain a continuous representation, we extract a polygon
# that encloses or approximates this region.
#
# ----------------------------------------------------------
# AVAILABLE METHODS
# ----------------------------------------------------------
#
# 1. CONVEX HULL
#
#   - Simplest enclosing polygon
#   - Guarantees no self-intersection
#   - Computationally stable
#
#   Limitations:
#       - ignores concavities
#       - overestimates area
#
#   Use case:
#       - baseline comparison
#       - reproducibility reference
#
# ----------------------------------------------------------
#
# 2. ALPHA SHAPE (CONCAVE HULL)
#
#   - Captures non-convex structures
#   - Follows data distribution more closely
#
#   Limitations:
#       - sensitive to alpha parameter
#       - may generate multiple disconnected polygons
#       - may fail for sparse or noisy data
#
#   Use case:
#       - realistic comfort zone estimation
#       - scientific interpretation
#
# ----------------------------------------------------------
# INPUT ASSUMPTIONS
# ----------------------------------------------------------
#
# points:
#   array of shape (N, 2)
#   coordinates in (T, W)
#
# cfg:
#   configuration dictionary:
#       cfg["geometry"]["method"]
#       cfg["geometry"]["alpha"]
#
# ----------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------
#
# np.ndarray of shape (M, 2)
# representing ordered polygon vertices
#
def build_polygon(points: np.ndarray, cfg: dict) -> np.ndarray:
    """Build polygon using convex hull or alpha shape."""

    if points.shape[0] < 3:
        raise ValueError("At least 3 points are required to build a polygon.")

    method = cfg["geometry"]["method"]

    # ------------------------------------------------------
    # CONVEX HULL
    # ------------------------------------------------------
    #
    if method == "convex":
        try:
            hull = ConvexHull(points)
            return points[hull.vertices]
        except Exception as e:
            raise RuntimeError(f"ConvexHull failed: {e}")

    # ------------------------------------------------------
    # ALPHA SHAPE
    # ------------------------------------------------------
    #
    if method == "alpha":
        try:
            shape = alphashape.alphashape(points, cfg["geometry"]["alpha"])
        except Exception as e:
            raise RuntimeError(f"Alpha shape computation failed: {e}")

        if shape is None:
            raise RuntimeError("Alpha shape returned None.")

        # --------------------------------------------------
        # HANDLE GEOMETRY TYPES
        # --------------------------------------------------
        #
        # Polygon:
        #   single connected region
        #
        # MultiPolygon:
        #   multiple disconnected regions
        #   → we select the largest (dominant cluster)
        #
        if shape.geom_type == "Polygon":
            x, y = shape.exterior.xy

        elif shape.geom_type == "MultiPolygon":
            largest = max(shape.geoms, key=lambda p: p.area)
            x, y = largest.exterior.xy

        else:
            raise RuntimeError(f"Unsupported geometry type: {shape.geom_type}")

        return np.column_stack([x, y])

    # ------------------------------------------------------
    # INVALID METHOD
    # ------------------------------------------------------
    #
    raise ValueError("Invalid geometry method. Use 'convex' or 'alpha'.")


# ==========================================================
# MULTI-ZONE POLYGON EXTRACTION
# ==========================================================
#
# Builds one polygon per zone (core, transition, limit).
#
# ----------------------------------------------------------
# SCIENTIFIC INTERPRETATION
# ----------------------------------------------------------
#
# Each zone corresponds to a different density threshold.
#
# Therefore:
#
#   - core polygon → most reliable region
#   - transition → intermediate
#   - limit → boundary of tolerance
#
# These polygons can be:
#
#   - plotted together
#   - compared across datasets
#   - used for downstream metrics
#
# ----------------------------------------------------------
# IMPORTANT DESIGN DECISION
# ----------------------------------------------------------
#
# Small point clouds are ignored because:
#
#   - they produce unstable geometries
#   - they often correspond to noise
#
# The threshold (10 points) is empirical and may be tuned.
#
def build_zone_polygons(
    zones: dict,
    cfg: dict,
) -> dict[str, np.ndarray]:
    """Build polygon for each zone."""

    polys = {}

    for name, pts in zones.items():

        # --------------------------------------------------
        # MINIMUM POINT REQUIREMENT
        # --------------------------------------------------
        #
        # Avoid unstable geometry
        #
        if pts.shape[0] < 10:
            continue

        try:
            polys[name] = build_polygon(pts, cfg)
        except Exception:
            # silently skip problematic zones
            continue

    return polys
