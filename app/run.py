from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import pandas as pd

from config_schema import PipelineConfig
from config import ZONE_COLORS

from pipeline.density import build_density, extract_points, filter_density
from pipeline.zones import build_zones
from pipeline.geometry import build_zone_polygons
from pipeline.smoothing import smooth_polygons
from plot.plot_psychro import plot_psychro

from extract_comfort_periods import (
    standardize_columns,
    calculate_heat_load,
    define_comfort,
    extract_comfort_periods,
)


# ==========================================================
# DATASET
# ==========================================================
def build_comfort_dataset(cfg: PipelineConfig) -> pd.DataFrame:
    print(f"[INFO] Loading dataset: {cfg.dataset_path}")

    df = pd.read_parquet(cfg.dataset_path)

    df = standardize_columns(df)
    df = calculate_heat_load(df)
    df = define_comfort(df)
    df = extract_comfort_periods(df)

    print(f"[INFO] Comfort records: {len(df):,}")

    return df


# ==========================================================
# CONFIG LOADER
# ==========================================================
def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return PipelineConfig(**data)


# ==========================================================
# CLI
# ==========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thermal comfort pipeline")

    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--dataset", type=str, help="Override dataset path")
    parser.add_argument("--no-cluster", action="store_true")
    parser.add_argument("--no-smooth", action="store_true")

    return parser.parse_args()


# ==========================================================
# CONFIG OVERRIDES (IMPORTANT)
# ==========================================================
def apply_overrides(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """
    IMPORTANT:
    Pydantic models should be treated as IMMUTABLE for reproducibility.

    Therefore, we create a new instance instead of mutating the original.
    """

    cfg_dict = cfg.model_dump()

    if args.dataset:
        cfg_dict["dataset_path"] = args.dataset

    if args.no_cluster:
        cfg_dict["clustering"]["enabled"] = False

    if args.no_smooth:
        cfg_dict["smoothing"]["enabled"] = False

    return PipelineConfig(**cfg_dict)


# ==========================================================
# MAIN PIPELINE
# ==========================================================
def run_pipeline(cfg: PipelineConfig) -> None:

    # snapshot da config (garante consistência)
    cfg_dict = cfg.model_dump()

    print("[INFO] Building dataset...")
    df = build_comfort_dataset(cfg)

    print("[INFO] Building density field...")
    T_edges, W_edges, values = build_density(df, 101325, cfg_dict)

    print("[INFO] Extracting points...")
    points = extract_points(T_edges, W_edges, values)

    print("[INFO] Filtering density...")
    points = filter_density(points, values, cfg_dict)

    print("[INFO] Building zones...")
    zones = build_zones(points, values, cfg_dict)

    print("[INFO] Extracting geometry...")
    polygons = build_zone_polygons(zones, cfg_dict)

    if cfg.smoothing.enabled:
        print("[INFO] Applying smoothing...")
        polygons = smooth_polygons(polygons, cfg_dict)

    print("[INFO] Plotting...")
    plot_psychro(
        T_edges,
        W_edges,
        values,
        polygons,
        ZONE_COLORS,
        cfg_dict,
    )

    print("[INFO] Pipeline completed successfully.")


# ==========================================================
# ENTRYPOINT
# ==========================================================
def main() -> int:
    args = parse_args()

    # base config
    cfg = (
        load_config(args.config)
        if args.config
        else PipelineConfig(dataset_path=Path("dataset.parquet"))
    )

    # apply overrides safely
    cfg = apply_overrides(cfg, args)

    run_pipeline(cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
