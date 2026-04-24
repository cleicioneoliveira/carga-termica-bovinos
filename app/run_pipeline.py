from __future__ import annotations

import argparse
import pandas as pd

from .config import CONFIG, ZONE_COLORS

from .pipeline.density import (
    build_density,
    extract_points,
    filter_density,
)

from .pipeline.zones import build_zones
from .pipeline.geometry import build_zone_polygons
from .pipeline.smoothing import smooth_polygons
from .pipeline.thermal_comfort import (
    load_and_prepare_dataset,
    run_manual_mode,
    run_auto_mode,
)

from .plot.plot_psychro import plot_psychro


from .util.profiling import run_with_profile

# ==========================================================
# CONFIG PADRÃO
# ==========================================================
DEFAULT_THI_THRESHOLD = 72
DEFAULT_WINDOWS = list(range(1, 25, 1))   # 1, 2, ..., 24
DEFAULT_WINDOW = 15
DEFAULT_MIN_DURATION = 3
DEFAULT_THERMAL_MODE = 'manual'
DEFAULT_THERMAL_CRITERIA = 'mean_corr'
DEFAULT_OUTPUT_DIR = 'outputs_conforto'

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
#   2. cleaning and type conversion
#   3. THI computation
#   4. heat load computation
#   5. comfort classification
#   6. extraction of continuous comfort periods
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
    """Build comfort dataset from raw data using thermal_comfort_pipeline."""

    dataset_path = cfg["dataset_path"]

    thi_threshold = cfg.get("thi_threshold", DEFAULT_THI_THRESHOLD)
    min_duration = cfg.get("min_duration", DEFAULT_MIN_DURATION)

    thermal_mode = cfg.get("thermal_mode", DEFAULT_THERMAL_MODE)
    output_dir = cfg.get("thermal_output_dir", DEFAULT_OUTPUT_DIR)

    df = load_and_prepare_dataset(
        dataset_path=dataset_path,
        thi_threshold=thi_threshold,
    )

    if df.empty:
        raise ValueError("Input dataset is empty after preparation.")

    if thermal_mode == "manual":
        window = cfg.get("thermal_window", DEFAULT_WINDOW)

        _, df_periods = run_manual_mode(
            df=df,
            window=window,
            min_duration=min_duration,
            output_dir=output_dir,
        )

    elif thermal_mode == "auto":
        windows = cfg.get("thermal_windows", DEFAULT_WINDOWS)
        criterion = cfg.get("thermal_criterion", DEFAULT_THERMAL_CRITERIA)

        _, _, _, df_periods = run_auto_mode(
            df=df,
            windows=windows,
            criterion=criterion,
            min_duration=min_duration,
            output_dir=output_dir,
        )

    else:
        raise ValueError(
            f"Invalid thermal_mode: {thermal_mode!r}. "
            "Use 'manual' or 'auto'."
        )

    if df_periods.empty:
        raise ValueError("No comfort records found after filtering.")

    return df_periods


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

# ==========================================================
# CLI
# ==========================================================
def build_parser() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(
        description="Análise de carga térmica e extração de períodos de conforto."
    )

    #--------------------------------------------
    # Profilling flags
    #--------------------------------------------

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Ativa profiling com cProfile."
    )
    
    parser.add_argument(
        "--profile-file",
        default="outputs_conforto/profile.prof",
        help="Arquivo de saída do profiling."
    )
    
    parser.add_argument(
        "--profile-sort",
        default="cumulative",
        choices=["cumulative", "time", "calls"],
        help="Critério de ordenação do profiling."
    )
    
    parser.add_argument(
        "--profile-lines",
        type=int,
        default=30,
        help="Quantidade de linhas mostradas no resumo do profiling."
    )

    return parser


if __name__ == "__main__":
    
    parser = build_parser()
    args = parser.parse_args()

    if args.profile:
        run_with_profile(
            run_pipeline,
            profile_file=args.profile_file,
            sort_by=args.profile_sort,
            lines=args.profile_lines,
        )
    else:
        run_pipeline()
