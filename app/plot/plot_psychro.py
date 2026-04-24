from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from psychchart import PsychChart
from psychchart.config import DensityFieldConfig
from psychchart.loader import load_chart_config


# ==========================================================
# PSYCHROMETRIC PLOTTING WITH COMFORT ZONES
# ==========================================================
#
# This module generates the final visualization of the
# empirical comfort zones in psychrometric space.
#
# ----------------------------------------------------------
# SCIENTIFIC CONTEXT
# ----------------------------------------------------------
#
# The plot integrates:
#
#   1. Density field of comfort observations
#   2. Extracted polygon(s) representing comfort regions
#
# The result is a visual synthesis of:
#
#   - environmental conditions (T, W)
#   - statistical distribution of comfort
#   - geometric representation of comfort zones
#
# ----------------------------------------------------------
# VISUAL COMPONENTS
# ----------------------------------------------------------
#
# 1. Psychrometric chart
#   - thermodynamic reference grid
#
# 2. Density field
#   - represents probability mass of comfort observations
#
# 3. Zone polygons
#   - core / transition / limit regions
#
# ----------------------------------------------------------
# IMPORTANT INTERPRETATION NOTE
# ----------------------------------------------------------
#
# The density field represents empirical evidence.
#
# The polygons represent derived structures based on:
#
#   - filtering
#   - clustering
#   - geometric extraction
#
# Therefore:
#
#   - density = data
#   - polygons = model interpretation
#
# ----------------------------------------------------------
# INPUTS
# ----------------------------------------------------------
#
# T_edges:
#   bin edges for temperature axis
#
# W_edges:
#   bin edges for humidity axis
#
# values:
#   2D density matrix
#
# polygons:
#   dict[str, np.ndarray]
#   polygon vertices per zone
#
# colors:
#   dict[str, str]
#   color per zone
#
# cfg:
#   global configuration dictionary
#

from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHART_CONFIG_PATH = APP_DIR / "chart_config.yaml"


def _resolve_chart_config_path(cfg: dict) -> Path:
    """
    Resolve the psychrometric chart configuration path.

    Resolution order:
    1. Use cfg["chart_config_path"] if provided and the file exists.
    2. Otherwise, fall back to the default chart_config.yaml
       located at the project root.

    Relative paths are resolved from the project root.
    """
    custom_path = cfg.get("chart_config_path")

    if custom_path:
        path = Path(custom_path).expanduser()
        if not path.is_absolute():
            path = APP_DIR / path
        path = path.resolve()

        if path.exists():
            return path

        print(
            f"[WARNING] Provided chart configuration file does not exist: {path}. "
            f"Falling back to default configuration: {DEFAULT_CHART_CONFIG_PATH.resolve()}"
        )

    default_path = DEFAULT_CHART_CONFIG_PATH.resolve()

    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        "No valid psychrometric chart configuration file was found. "
        f"Checked custom path: {custom_path!r} and default path: {default_path}"
    )

def _resolve_output_dir(cfg: dict) -> Path:
    """
    Resolve the output directory for generated figures.

    If a relative path is provided, it is resolved from the
    project root. The directory is created automatically if
    it does not exist.
    """
    output_dir = Path(cfg.get("thermal_output_dir", "outputs_conforto")).expanduser()

    if not output_dir.is_absolute():
        output_dir = APP_DIR / output_dir

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def plot_psychro(
    T_edges,
    W_edges,
    values,
    polygons: dict,
    colors: dict,
    cfg: dict,
):
    """Plot psychrometric chart with zones."""

    # ------------------------------------------------------
    # LOAD CHART CONFIGURATION
    # ------------------------------------------------------
    #
    # This defines:
    #   - axes limits
    #   - grid styling
    #   - psychrometric curves
    #
    chart_config_path = _resolve_chart_config_path(cfg)

    if not chart_config_path.exists():
        raise FileNotFoundError(
            f"Psychrometric chart config file not found: {chart_config_path}"
        )

    chart_cfg = load_chart_config(str(chart_config_path))
    chart = PsychChart(**chart_cfg)

    # ------------------------------------------------------
    # BUILD DENSITY FIELD OBJECT
    # ------------------------------------------------------
    #
    # PsychChart expects a structured object.
    # We dynamically construct it here.
    #
    density_field = type("DensityFieldWrapper", (), {
        "data": type("DensityData", (), {
            "T_edges": T_edges,
            "W_edges": W_edges,
            "values": values
        }),
        "cfg": DensityFieldConfig(
            cmap="viridis",
            alpha=0.7,
            normalize=True,
            colorbar=True
        )
    })

    chart.density_fields = [density_field]

    # ------------------------------------------------------
    # DRAW BASE CHART
    # ------------------------------------------------------
    #
    ax = chart.draw()

    # ------------------------------------------------------
    # PLOT ZONE POLYGONS
    # ------------------------------------------------------
    #
    # Each zone is plotted as a semi-transparent region.
    #
    # Transparency allows overlapping interpretation.
    #
    for name, poly in polygons.items():

        # Skip empty polygons
        if poly is None or len(poly) == 0:
            continue

        ax.fill(
            poly[:, 0],
            poly[:, 1],
            alpha=0.3,
            color=colors.get(name, "gray"),
            label=name
        )

    # ------------------------------------------------------
    # LEGEND
    # ------------------------------------------------------
    #
    # Labels correspond to zone names:
    #   core / transition / limit
    #
    ax.legend()

    # ------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------
    #
    # High-resolution export for publication
    #
    output_dir = _resolve_output_dir(cfg)
    plt.savefig(output_dir / cfg["output_fig"], dpi=300, bbox_inches="tight")
    
    # Display
    plt.show()
