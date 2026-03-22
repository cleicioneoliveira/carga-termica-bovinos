from __future__ import annotations

import pandas as pd

from config import CONFIG, ZONE_COLORS

from pipeline.density import (
    build_density,
    extract_points,
    filter_density,
)

from pipeline.zones import build_zones
from pipeline.geometry import build_zone_polygons
from pipeline.smoothing import smooth_polygons

from plot.plot_psychro import plot_psychro

from extract_comfort_periods import (
    standardize_columns,
    calculate_heat_load,
    define_comfort,
    extract_comfort_periods
)


# ==========================================================
# DATASET PREPARATION
# ==========================================================
#
# This function transforms the raw dataset into a dataset
# containing only valid comfort records.
#
# ----------------------------------------------------------
# SCIENTIFIC CONTEXT
# ----------------------------------------------------------
#
# The original dataset contains:
#
#   - environmental variables
#   - behavioral indicators (rumination, activity, etc.)
#   - temporal information
#
# This step performs:
#
#   1. column standardization
#   2. heat load computation
#   3. comfort classification
#   4. extraction of continuous comfort periods
#
# The output is a filtered dataset representing only
# conditions associated with thermal comfort.
#
# IMPORTANT:
#
# This is one of the most critical steps in the pipeline.
# Any bias here propagates to all subsequent stages.
#
def build_comfort_dataset(cfg: dict) -> pd.DataFrame:
    """Build comfort dataset from raw data."""

    df = pd.read_parquet(cfg["dataset_path"])

    if df.empty:
        raise ValueError("Input dataset is empty.")

    df = standardize_columns(df)
    df = calculate_heat_load(df)
    df = define_comfort(df)

    df = extract_comfort_periods(df)

    if df.empty:
        raise ValueError("No comfort records found after filtering.")

    return df


# ==========================================================
# MAIN PIPELINE EXECUTION
# ==========================================================
#
# This function orchestrates the full workflow:
#
#   dataset → density → filtering → zones → geometry → plot
#
# ----------------------------------------------------------
# PIPELINE FLOW
# ----------------------------------------------------------
#
#   1. Build comfort dataset
#   2. Convert to psychrometric space (T, W)
#   3. Build density field
#   4. Extract valid points
#   5. Apply density filtering (optional)
#   6. Segment into zones (core/transition/limit)
#   7. Build polygons
#   8. Apply smoothing (optional)
#   9. Generate final plot
#
# ----------------------------------------------------------
# DESIGN PRINCIPLES
# ----------------------------------------------------------
#
#   - fully reproducible (driven by CONFIG)
#   - modular (each stage isolated)
#   - interpretable (clear separation of steps)
#
def run_pipeline() -> None:
    """Execute full pipeline."""

    cfg = CONFIG

    print("[INFO] Building dataset...")
    df = build_comfort_dataset(cfg)

    print(f"[INFO] Comfort records: {len(df):,}")

    # ------------------------------------------------------
    # DENSITY FIELD
    # ------------------------------------------------------
    #
    print("[INFO] Building density...")
    T_edges, W_edges, values = build_density(
        df,
        pressure=101325,
        cfg=cfg
    )

    # ------------------------------------------------------
    # POINT EXTRACTION
    # ------------------------------------------------------
    #
    print("[INFO] Extracting points...")
    points = extract_points(T_edges, W_edges, values)

    print(f"[INFO] Points extracted: {len(points):,}")

    # ------------------------------------------------------
    # DENSITY FILTERING
    # ------------------------------------------------------
    #
    print("[INFO] Filtering density...")
    points = filter_density(points, values, cfg)

    print(f"[INFO] Points after filtering: {len(points):,}")

    if len(points) < 10:
        raise RuntimeError("Too few points after filtering. Check density parameters.")

    # ------------------------------------------------------
    # ZONE SEGMENTATION
    # ------------------------------------------------------
    #
    print("[INFO] Building zones...")
    zones = build_zones(points, values, cfg)

    for name, pts in zones.items():
        print(f"[INFO] Zone '{name}': {len(pts):,} points")

    # ------------------------------------------------------
    # GEOMETRY
    # ------------------------------------------------------
    #
    print("[INFO] Building polygons...")
    polygons = build_zone_polygons(zones, cfg)

    if not polygons:
        raise RuntimeError("No polygons were generated.")

    # ------------------------------------------------------
    # SMOOTHING
    # ------------------------------------------------------
    #
    print("[INFO] Smoothing polygons...")
    polygons = smooth_polygons(polygons, cfg)

    # ------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------
    #
    print("[INFO] Plotting...")
    plot_psychro(
        T_edges,
        W_edges,
        values,
        polygons,
        ZONE_COLORS,
        cfg
    )

    print("[INFO] Pipeline completed successfully.")


# ==========================================================
# ENTRY POINT
# ==========================================================
#
if __name__ == "__main__":
    run_pipeline()
