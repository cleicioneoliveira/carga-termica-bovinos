from __future__ import annotations

import numpy as np
import pandas as pd

from psychchart.psychrometrics import Psychrometrics


# ==========================================================
# TEMPERATURE + HUMIDITY → HUMIDITY RATIO (W)
# ==========================================================
#
# This function converts raw environmental observations into
# psychrometric coordinates:
#
#   T → dry-bulb temperature (°C)
#   W → humidity ratio (kg/kg dry air)
#
# ----------------------------------------------------------
# SCIENTIFIC CONTEXT
# ----------------------------------------------------------
#
# The psychrometric space (T, W) is used instead of (T, RH)
# because:
#
#   - W is thermodynamically consistent
#   - W removes nonlinear distortions present in RH
#   - W is directly linked to enthalpy and heat load
#
# This transformation is essential for:
#
#   - building density fields
#   - extracting geometrically meaningful regions
#
# ----------------------------------------------------------
# DATA CLEANING STRATEGY
# ----------------------------------------------------------
#
# The function also applies strict filtering:
#
#   - removes non-numeric values
#   - removes missing values
#   - removes physically inconsistent observations
#
# This step is critical to avoid contaminating the density
# field with unrealistic points.
#
def compute_T_W(df: pd.DataFrame, pressure: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert temperature and RH into humidity ratio (W)."""

    df = df.copy()

    # ------------------------------------------------------
    # TYPE CONVERSION
    # ------------------------------------------------------
    #
    # Ensures numeric consistency
    #
    df["temperatura"] = pd.to_numeric(df["temperatura"], errors="coerce")
    df["umidade"] = pd.to_numeric(df["umidade"], errors="coerce")

    # ------------------------------------------------------
    # REMOVE MISSING VALUES
    # ------------------------------------------------------
    #
    df = df.dropna(subset=["temperatura", "umidade"])

    # ------------------------------------------------------
    # PHYSICAL FILTERING
    # ------------------------------------------------------
    #
    # Removes unrealistic environmental conditions.
    #
    # These limits are not strict physical laws, but
    # practical bounds for field data.
    #
    df = df[
        (df["umidade"] >= 0) &
        (df["umidade"] <= 100) &
        (df["temperatura"] > -10) &
        (df["temperatura"] < 60)
    ]

    # ------------------------------------------------------
    # VARIABLE EXTRACTION
    # ------------------------------------------------------
    #
    T = df["temperatura"].values
    RH = df["umidade"].values / 100.0

    # ------------------------------------------------------
    # PSYCHROMETRIC TRANSFORMATION
    # ------------------------------------------------------
    #
    # Uses saturation vapor pressure relations internally.
    #
    W = Psychrometrics.humidity_ratio(T, RH, pressure)

    # ------------------------------------------------------
    # FINAL CONSISTENCY FILTER
    # ------------------------------------------------------
    #
    mask = np.isfinite(T) & np.isfinite(W)

    return T[mask], W[mask]


# ==========================================================
# DENSITY FIELD (2D HISTOGRAM)
# ==========================================================
#
# Converts comfort points into a 2D density field.
#
# ----------------------------------------------------------
# INTERPRETATION
# ----------------------------------------------------------
#
# Each cell represents the probability mass of observations
# in psychrometric space.
#
# The resulting matrix is later used to:
#
#   - define comfort regions
#   - extract polygons
#   - derive multi-zone structures
#
# ----------------------------------------------------------
# IMPORTANT NOTES
# ----------------------------------------------------------
#
#   - normalization ensures comparability
#   - thresholding removes noise
#   - resolution is controlled by number of bins
#
def build_density(
    df: pd.DataFrame,
    pressure: float,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build normalized 2D density field."""

    T, W = compute_T_W(df, pressure)

    # ------------------------------------------------------
    # HISTOGRAM CONSTRUCTION
    # ------------------------------------------------------
    #
    H, T_edges, W_edges = np.histogram2d(
        T,
        W,
        bins=cfg["density"]["bins"]
    )

    # ------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------
    #
    # Converts counts into probability mass
    #
    total = np.sum(H)
    if total == 0:
        raise ValueError("Density histogram is empty.")

    H = H / total

    # ------------------------------------------------------
    # NOISE REMOVAL
    # ------------------------------------------------------
    #
    # Remove very low-density cells
    #
    H[H < cfg["density"]["min_density"]] = np.nan

    # Transpose aligns axes with plotting convention
    return T_edges, W_edges, H.T


# ==========================================================
# GRID → POINT CLOUD
# ==========================================================
#
# Converts the density grid into a set of points
# representing valid regions.
#
# ----------------------------------------------------------
# INTERPRETATION
# ----------------------------------------------------------
#
# Each retained point corresponds to a cell center
# with non-null density.
#
# This transformation is required because:
#
#   - clustering operates on point clouds
#   - geometric algorithms require coordinate sets
#
def extract_points(
    T_edges: np.ndarray,
    W_edges: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Extract valid density points."""

    T_centers = (T_edges[:-1] + T_edges[1:]) / 2
    W_centers = (W_edges[:-1] + W_edges[1:]) / 2

    points = []

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isnan(values[i, j]):
                points.append([T_centers[j], W_centers[i]])

    if len(points) == 0:
        raise ValueError("No valid density points extracted.")

    return np.array(points)


# ==========================================================
# DENSITY-BASED FILTERING
# ==========================================================
#
# Removes low-density regions based on percentile threshold.
#
# ----------------------------------------------------------
# SCIENTIFIC PURPOSE
# ----------------------------------------------------------
#
# This step defines how conservative the comfort zone is:
#
#   - low percentile → inclusive region
#   - high percentile → core region only
#
# ----------------------------------------------------------
# IMPORTANT
# ----------------------------------------------------------
#
# This is NOT a physical threshold.
# It is a statistical selection criterion.
#
def filter_density(
    points: np.ndarray,
    values: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Apply percentile-based density filtering."""

    if not cfg["density"]["use_filter"]:
        return points

    flat = values[~np.isnan(values)]

    if flat.size == 0:
        raise ValueError("No valid density values for filtering.")

    threshold = np.percentile(flat, cfg["density"]["percentile"])

    filtered = []

    idx = 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isnan(values[i, j]):
                if values[i, j] >= threshold:
                    filtered.append(points[idx])
                idx += 1

    if len(filtered) == 0:
        raise ValueError("Filtering removed all points.")

    return np.array(filtered)
