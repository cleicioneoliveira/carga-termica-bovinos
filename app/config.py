from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
DEFAULT_CONFIG_PATH = APP_DIR / "config.yaml"


ZONE_COLORS: dict[str, str] = {
    "core": "red",
    "transition": "orange",
    "limit": "yellow",
}


def resolve_project_path(path: str | Path, *, base_dir: Path = PROJECT_ROOT) -> Path:
    """Resolve absolute and project-relative paths consistently.

    Parameters
    ----------
    path : str or Path
        Path to resolve. Absolute paths are returned unchanged after expansion.
        Relative paths are resolved from ``base_dir``.
    base_dir : Path, optional
        Directory used to resolve relative paths. Defaults to the repository root.

    Returns
    -------
    Path
        Resolved path.
    """
    resolved = Path(path).expanduser()

    if not resolved.is_absolute():
        resolved = base_dir / resolved

    return resolved.resolve()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the pipeline configuration from a YAML file.

    ``app/config.yaml`` is the official source of truth. This module exists only
    to centralize loading, validation and compatibility for older imports.
    """
    config_path = resolve_project_path(path or DEFAULT_CONFIG_PATH)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    if not isinstance(cfg, dict):
        raise TypeError(f"Configuration root must be a mapping: {config_path}")

    cfg = _with_defaults(cfg)
    validate_config(cfg)
    return cfg


def _with_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a validated configuration dictionary with safe defaults."""
    result = deepcopy(cfg)

    result.setdefault("thermal_mode", "auto")
    result.setdefault("thi_threshold", 72)
    result.setdefault("min_duration", 3)
    result.setdefault("thermal_output_dir", "outputs_conforto")
    result.setdefault("thermal_window", 15)
    result.setdefault("thermal_windows", list(range(1, 25)))
    result.setdefault("thermal_criterion", "mean_corr")
    result.setdefault("chart_config_path", "app/chart_config.yaml")
    result.setdefault("output_fig", "fig_comfort_polygon.png")
    result.setdefault("show_plots", False)
    result.setdefault("suppress_chart_stdout", True)
    result.setdefault("log_level", "INFO")

    result.setdefault("density", {})
    result["density"].setdefault("bins", 40)
    result["density"].setdefault("min_density", 0.001)
    result["density"].setdefault("percentile", 65)
    result["density"].setdefault("use_filter", False)

    result.setdefault("clustering", {})
    result["clustering"].setdefault("enabled", False)
    result["clustering"].setdefault("eps", 0.5)
    result["clustering"].setdefault("min_samples", 10)

    result.setdefault("geometry", {})
    result["geometry"].setdefault("method", "alpha")
    result["geometry"].setdefault("alpha", 1.2)

    result.setdefault("smoothing", {})
    result["smoothing"].setdefault("enabled", True)
    result["smoothing"].setdefault("sigma", 2)

    result.setdefault("zones", {})
    result["zones"].setdefault("core_percentile", 85)
    result["zones"].setdefault("transition_percentile", 60)
    result["zones"].setdefault("limit_percentile", 30)

    return result


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate the minimum configuration required by the pipeline."""
    required_keys = ["dataset_path", "density", "geometry", "smoothing", "zones"]
    missing = [key for key in required_keys if key not in cfg]
    if missing:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing)}")

    thermal_mode = cfg.get("thermal_mode")
    if thermal_mode not in {"manual", "auto"}:
        raise ValueError("thermal_mode must be either 'manual' or 'auto'")

    criterion = cfg.get("thermal_criterion")
    if criterion not in {"mean_corr", "median_corr"}:
        raise ValueError("thermal_criterion must be 'mean_corr' or 'median_corr'")

    if int(cfg["density"]["bins"]) < 10:
        raise ValueError("density.bins must be >= 10")

    if cfg["geometry"]["method"] not in {"alpha", "convex"}:
        raise ValueError("geometry.method must be 'alpha' or 'convex'")

    zones = cfg["zones"]
    if not zones["core_percentile"] >= zones["transition_percentile"] >= zones["limit_percentile"]:
        raise ValueError(
            "Zone percentiles must satisfy: core_percentile >= "
            "transition_percentile >= limit_percentile"
        )

    log_level = str(cfg.get("log_level", "INFO")).upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level not in valid_levels:
        raise ValueError(f"log_level must be one of {sorted(valid_levels)}")


# Backward-compatible import used by older scripts.
CONFIG = load_config()
