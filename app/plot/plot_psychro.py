from __future__ import annotations

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
    chart_cfg = load_chart_config("chart_config.yaml")

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
    plt.savefig(cfg["output_fig"], dpi=300)

    # Display
    plt.show()
