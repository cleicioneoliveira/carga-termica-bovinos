from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CONFIG, ZONE_COLORS, load_config
from .pipeline.density import build_density, extract_points, filter_density
from .pipeline.geometry import build_zone_polygons
from .pipeline.smoothing import smooth_polygons
from .pipeline.thermal_comfort import (
    load_and_prepare_dataset,
    run_auto_mode,
    run_manual_mode,
)
from .pipeline.zones import build_zones
from .plot.plot_psychro import plot_psychro
from .util.profiling import run_with_profile


DEFAULT_THI_THRESHOLD = 72
DEFAULT_WINDOWS = list(range(1, 25))
DEFAULT_WINDOW = 15
DEFAULT_MIN_DURATION = 3
DEFAULT_THERMAL_MODE = "manual"
DEFAULT_THERMAL_CRITERION = "mean_corr"
DEFAULT_OUTPUT_DIR = "outputs_conforto"

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure a compact console logger for command-line execution."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
        force=True,
    )


def build_comfort_dataset(cfg: dict[str, Any]) -> pd.DataFrame:
    """Build the comfort-period dataset used by the psychrometric stage.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration loaded from YAML.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing only valid comfort-period records.
    """
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
        criterion = cfg.get("thermal_criterion", DEFAULT_THERMAL_CRITERION)
        _, _, _, df_periods = run_auto_mode(
            df=df,
            windows=windows,
            criterion=criterion,
            min_duration=min_duration,
            output_dir=output_dir,
        )

    else:
        raise ValueError(
            f"Invalid thermal_mode: {thermal_mode!r}. Use 'manual' or 'auto'."
        )

    if df_periods.empty:
        raise ValueError("No comfort records found after filtering.")

    return df_periods


def run_pipeline(cfg: dict[str, Any] | None = None) -> None:
    """Execute the full thermal-comfort and psychrometric pipeline."""
    cfg = cfg or CONFIG

    logger.info("Building dataset...")
    df = build_comfort_dataset(cfg)
    logger.info("Comfort records: %s", f"{len(df):,}")

    logger.info("Building density...")
    T_edges, W_edges, values = build_density(
        df,
        pressure=101325,
        cfg=cfg,
    )

    logger.info("Extracting points...")
    points = extract_points(T_edges, W_edges, values)
    logger.info("Points extracted: %s", f"{len(points):,}")

    logger.info("Filtering density...")
    points = filter_density(points, values, cfg)
    logger.info("Points after filtering: %s", f"{len(points):,}")

    if len(points) < 10:
        raise RuntimeError("Too few points after filtering. Check density parameters.")

    logger.info("Building zones...")
    zones = build_zones(points, values, cfg)
    for name, pts in zones.items():
        logger.info("Zone '%s': %s points", name, f"{len(pts):,}")

    logger.info("Building polygons...")
    polygons = build_zone_polygons(zones, cfg)
    if not polygons:
        raise RuntimeError("No polygons were generated.")

    if cfg.get("smoothing", {}).get("enabled", True):
        logger.info("Smoothing polygons...")
        polygons = smooth_polygons(polygons, cfg)

    logger.info("Plotting...")
    plot_psychro(T_edges, W_edges, values, polygons, ZONE_COLORS, cfg)

    logger.info("Pipeline completed successfully.")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface parser."""
    parser = argparse.ArgumentParser(
        description="Analyze bovine thermal load and extract comfort periods."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file. Defaults to app/config.yaml.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset_path from the YAML configuration.",
    )
    parser.add_argument(
        "--thermal-mode",
        choices=["manual", "auto"],
        default=None,
        help="Override thermal_mode from the YAML configuration.",
    )
    parser.add_argument(
        "--thermal-window",
        type=int,
        default=None,
        help="Override thermal_window when using manual mode.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display Matplotlib windows after saving figures.",
    )
    parser.add_argument(
        "--verbose-chart",
        action="store_true",
        help="Do not suppress stdout emitted by the chart renderer.",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable polygon smoothing for this run.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override log_level from the YAML configuration.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile profiling.",
    )
    parser.add_argument(
        "--profile-file",
        default="outputs_conforto/profile.prof",
        help="Profiling output file.",
    )
    parser.add_argument(
        "--profile-sort",
        default="cumulative",
        choices=["cumulative", "time", "calls"],
        help="Profiling sort criterion.",
    )
    parser.add_argument(
        "--profile-lines",
        type=int,
        default=30,
        help="Number of profiling summary lines to display.",
    )

    return parser


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply command-line overrides without mutating the loaded config."""
    updated = dict(cfg)
    updated["density"] = dict(cfg.get("density", {}))
    updated["smoothing"] = dict(cfg.get("smoothing", {}))

    if args.dataset:
        updated["dataset_path"] = args.dataset

    if args.thermal_mode:
        updated["thermal_mode"] = args.thermal_mode

    if args.thermal_window is not None:
        updated["thermal_window"] = args.thermal_window

    if args.no_smooth:
        updated["smoothing"]["enabled"] = False

    if args.show_plots:
        updated["show_plots"] = True

    if args.verbose_chart:
        updated["suppress_chart_stdout"] = False

    if args.log_level:
        updated["log_level"] = args.log_level

    return updated


def main() -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    cfg = apply_cli_overrides(cfg, args)
    configure_logging(cfg.get("log_level", "INFO"))

    if args.profile:
        run_with_profile(
            lambda: run_pipeline(cfg),
            profile_file=args.profile_file,
            sort_by=args.profile_sort,
            lines=args.profile_lines,
        )
    else:
        run_pipeline(cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
