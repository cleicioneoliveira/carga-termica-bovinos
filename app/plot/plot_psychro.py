"""
Psychrometric plotting utilities for the thermal load pipeline.

This module uses the external ``psychchart`` package only to draw the
psychrometric background. The empirical thermal-comfort zones are then drawn
on top of the chart using matplotlib patches.

Important
---------
``psychchart.load_chart_config()`` already returns a dictionary compatible with
``PsychChart(**chart_cfg)``. Do not unwrap the original YAML manually and do
not pass a top-level ``chart`` key directly to ``PsychChart``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Polygon as MplPolygon

from psychchart import PsychChart
from psychchart.loader import load_chart_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_chart_config_path(cfg: dict[str, Any]) -> Path:
    """
    Resolve the psychchart YAML configuration file.

    Priority
    --------
    1. cfg["chart_config_path"]
    2. cfg["psychchart_config"]
    3. cfg["chart_config"]
    4. cfg["chart"]["config"]
    5. app/chart_config.yaml relative to the repository root
    6. app/plot/chart_config.yaml relative to this file
    """
    candidates: list[Path] = []

    for key in ("chart_config_path", "psychchart_config", "chart_config"):
        value = cfg.get(key)
        if value:
            path = Path(value).expanduser()
            candidates.append(path)
            if not path.is_absolute():
                candidates.append(PROJECT_ROOT / path)
                candidates.append(Path.cwd() / path)

    chart_block = cfg.get("chart")
    if isinstance(chart_block, dict):
        value = chart_block.get("config")
        if value:
            path = Path(value).expanduser()
            candidates.append(path)
            if not path.is_absolute():
                candidates.append(PROJECT_ROOT / path)
                candidates.append(Path.cwd() / path)

    candidates.extend(
        [
            PROJECT_ROOT / "app" / "chart_config.yaml",
            Path(__file__).resolve().parent / "chart_config.yaml",
            Path.cwd() / "app" / "chart_config.yaml",
            Path.cwd() / "app" / "plot" / "chart_config.yaml",
        ]
    )

    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if resolved.exists():
            return resolved

    checked_str = "\n - ".join(dict.fromkeys(checked))
    raise FileNotFoundError(
        "Could not find psychchart configuration file. Checked:\n"
        f" - {checked_str}"
    )



def _clear_existing_titles(fig, ax) -> None:
    """
    Remove titles created previously by PsychChart or matplotlib.

    PsychChart may set the axis title internally from cfg.title. This function
    clears all title locations before the pipeline applies its final title.
    """
    ax.set_title("", loc="left")
    ax.set_title("", loc="center")
    ax.set_title("", loc="right")

    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.remove()
        fig._suptitle = None



def _line_is_axis_aligned(line) -> bool:
    """
    Return True for vertical or horizontal helper/grid-like lines.
    """
    try:
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
    except Exception:
        return False

    if xdata.size < 2 or ydata.size < 2:
        return False

    is_vertical = np.nanmax(xdata) - np.nanmin(xdata) < 1.0e-9
    is_horizontal = np.nanmax(ydata) - np.nanmin(ydata) < 1.0e-9

    return bool(is_vertical or is_horizontal)



def _apply_nature_style_to_psychchart(ax, cfg: dict[str, Any]) -> None:
    """
    Apply a cleaner academic/Nature-like visual style to the psychrometric chart.
    """
    fig = ax.figure

    fig.set_facecolor("white")
    ax.set_facecolor("white")

    _clear_existing_titles(fig, ax)

    title = cfg.get("title", "Carta psicrométrica com zonas empíricas de conforto")
    if title:
        ax.set_title(
            title,
            loc="left",
            pad=8,
            fontsize=8,
            fontweight="bold",
            color="black",
        )

    ax.set_xlabel(
        cfg.get("xlabel", "Temperatura do ar, Tbs (°C)"),
        fontsize=7,
        color="black",
    )
    ax.set_ylabel(
        cfg.get("ylabel", "Razão de umidade, W (g kg⁻¹ de ar seco)"),
        fontsize=7,
        color="black",
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=6,
        width=0.6,
        length=2.5,
        direction="out",
        colors="black",
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=0.4,
        length=1.5,
        direction="out",
        colors="black",
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("black")

    for line in ax.lines:
        if _line_is_axis_aligned(line):
            line.set_color("0.82")
            line.set_linewidth(cfg.get("helper_linewidth", 0.22))
            line.set_alpha(0.45)
            line.set_linestyle("-")
            continue

        line.set_linewidth(cfg.get("psychrometric_linewidth", 0.35))
        line.set_alpha(cfg.get("psychrometric_line_alpha", 0.55))

    for text in ax.texts:
        text.set_fontsize(cfg.get("psychrometric_label_fontsize", 5.5))
        text.set_alpha(cfg.get("psychrometric_label_alpha", 0.75))
        text.set_color(cfg.get("psychrometric_label_color", "0.25"))

    if cfg.get("grid", True):
        ax.grid(
            True,
            which="major",
            linewidth=cfg.get("grid_linewidth", 0.25),
            alpha=cfg.get("grid_alpha", 0.25),
        )
    else:
        ax.grid(False)



def _to_points_array(points: Any) -> np.ndarray:
    """
    Convert polygon points to a numeric Nx2 numpy array.

    Accepted formats
    ----------------
    - [[T, W], [T, W], ...]
    - [(T, W), (T, W), ...]
    - numpy array with shape (N, 2)
    """
    arr = np.asarray(points, dtype=float)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            "Each polygon must be an array-like object with shape (N, 2)."
        )

    return arr



def _convert_w_to_gkg_if_needed(points: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """
    Convert moisture coordinate from kg/kg to g/kg when needed.
    """
    moisture_unit = str(cfg.get("moisture_unit", cfg.get("w_unit", "g/kg"))).lower()

    out = points.copy()

    if moisture_unit in {"kg/kg", "kg kg-1", "kg kg⁻¹"}:
        out[:, 1] *= 1000.0
        return out

    if np.nanmax(np.abs(out[:, 1])) < 0.2:
        out[:, 1] *= 1000.0

    return out



def _draw_zone_polygons(
    ax,
    polygons: dict[str, Any],
    colors: dict[str, str],
    cfg: dict[str, Any],
) -> list[Patch]:
    """
    Draw empirical comfort-zone polygons and return legend handles.
    """
    handles: list[Patch] = []

    zone_alpha = cfg.get("zone_alpha", 0.38)
    zone_edgecolor = cfg.get("zone_edgecolor", "0.15")
    zone_linewidth = cfg.get("zone_linewidth", 0.45)

    for zone_name, points in polygons.items():
        pts = _to_points_array(points)
        pts = _convert_w_to_gkg_if_needed(pts, cfg)

        color = colors.get(zone_name, "0.7")

        patch = MplPolygon(
            pts,
            closed=True,
            facecolor=color,
            edgecolor=zone_edgecolor,
            linewidth=zone_linewidth,
            alpha=zone_alpha,
            zorder=20,
        )
        ax.add_patch(patch)

        handles.append(
            Patch(
                facecolor=color,
                edgecolor=zone_edgecolor,
                linewidth=zone_linewidth,
                alpha=zone_alpha,
                label=str(zone_name),
            )
        )

        if cfg.get("zone_labels", True):
            centroid = np.nanmean(pts, axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                str(zone_name),
                ha="center",
                va="center",
                fontsize=cfg.get("zone_label_fontsize", 6),
                fontweight=cfg.get("zone_label_fontweight", "bold"),
                color=cfg.get("zone_label_color", "black"),
                zorder=30,
            )

    return handles



def _add_legend(ax, handles: list[Patch], cfg: dict[str, Any]) -> None:
    """
    Add legend for comfort-zone polygons.
    """
    if not handles or not cfg.get("legend", True):
        return

    legend = ax.legend(
        handles=handles,
        loc=cfg.get("legend_loc", "upper left"),
        bbox_to_anchor=cfg.get("legend_bbox_to_anchor", None),
        frameon=True,
        framealpha=0.92,
        fontsize=cfg.get("legend_fontsize", 6),
        title=cfg.get("legend_title", "Zonas"),
        title_fontsize=cfg.get("legend_title_fontsize", 6),
        borderpad=0.5,
        labelspacing=0.35,
        handlelength=1.2,
        handletextpad=0.5,
    )

    legend.get_frame().set_linewidth(0.4)
    legend.get_frame().set_edgecolor("0.4")



def _resolve_output_path(cfg: dict[str, Any]) -> Path | None:
    """
    Resolve the final figure path from the runtime configuration.
    """
    output = cfg.get("output") or cfg.get("output_file")
    if output:
        output_path = Path(output).expanduser()
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        return output_path.resolve()

    output_dir = cfg.get("thermal_output_dir")
    output_fig = cfg.get("output_fig")
    if output_dir and output_fig:
        output_path = Path(output_dir).expanduser() / output_fig
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        return output_path.resolve()

    return None



def _save_or_show(fig, cfg: dict[str, Any]) -> None:
    """
    Save the figure to disk or show it interactively.
    """
    output_path = _resolve_output_path(cfg)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stem = output_path.with_suffix("")
        dpi = cfg.get("dpi", cfg.get("save_dpi", 600))
        transparent = cfg.get("transparent", False)

        fig.savefig(
            stem.with_suffix(".png"),
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
        )
        fig.savefig(
            stem.with_suffix(".pdf"),
            bbox_inches="tight",
            transparent=transparent,
        )
        fig.savefig(
            stem.with_suffix(".svg"),
            bbox_inches="tight",
            transparent=transparent,
        )
        plt.close(fig)
        return

    if cfg.get("show_plots", False):
        plt.show()
    else:
        plt.close(fig)



def plot_psychro(
    T_edges,
    W_edges,
    values,
    polygons: dict,
    colors: dict,
    cfg: dict,
):
    """
    Plot psychrometric chart with empirical comfort zones.

    Parameters
    ----------
    T_edges, W_edges, values
        Kept in the signature for compatibility with the pipeline.
        The current version uses PsychChart for the background and overlays
        the empirical polygons.
    polygons
        Dictionary mapping zone names to polygon coordinates.
    colors
        Dictionary mapping zone names to matplotlib-compatible colors.
    cfg
        Runtime plotting configuration.
    """
    chart_config_path = _resolve_chart_config_path(cfg)
    chart_cfg = load_chart_config(str(chart_config_path))

    chart = PsychChart(**chart_cfg)

    if hasattr(chart, "cfg"):
        chart.cfg.title = None

    rc_params = {
        "font.family": cfg.get("font_family", "Arial"),
        "font.size": 6,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }

    with plt.rc_context(rc_params):
        chart.draw()

        fig = chart.fig
        ax = chart.ax

        if fig is None or ax is None:
            raise RuntimeError("PsychChart did not create a valid figure/axis.")

        width = cfg.get("fig_width", cfg.get("width", 7.2))
        height = cfg.get("fig_height", cfg.get("height", 5.0))
        fig.set_size_inches(width, height)

        _apply_nature_style_to_psychchart(ax, cfg)

        handles = _draw_zone_polygons(ax, polygons, colors, cfg)
        _add_legend(ax, handles, cfg)

        fig.tight_layout()
        _save_or_show(fig, cfg)
