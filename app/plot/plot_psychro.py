from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from psychchart import PsychChart
from psychchart.config import DensityFieldConfig
from psychchart.loader import load_chart_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHART_CONFIG_PATH = PROJECT_ROOT / "app" / "chart_config.yaml"


def _resolve_path(path: str | Path, *, base_dir: Path = PROJECT_ROOT) -> Path:
    """Resolve absolute and project-relative paths."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved.resolve()


def _resolve_chart_config_path(cfg: dict) -> Path:
    """Resolve the psychrometric chart configuration path.

    Resolution order:
    1. Use cfg["chart_config_path"] when provided and valid.
    2. Fall back to app/chart_config.yaml.

    Relative paths are resolved from the repository root.
    """
    custom_path = cfg.get("chart_config_path")

    if custom_path:
        path = _resolve_path(custom_path)
        if path.exists():
            return path

        print(
            "[WARNING] Provided chart configuration file does not exist: "
            f"{path}. Falling back to default configuration: "
            f"{DEFAULT_CHART_CONFIG_PATH.resolve()}"
        )

    default_path = DEFAULT_CHART_CONFIG_PATH.resolve()
    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        "No valid psychrometric chart configuration file was found. "
        f"Checked custom path: {custom_path!r} and default path: {default_path}"
    )


def _resolve_output_dir(cfg: dict) -> Path:
    """Resolve and create the output directory for generated figures."""
    output_dir = cfg.get("thermal_output_dir", "outputs_conforto")
    resolved = _resolve_path(output_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def plot_psychro(
    T_edges,
    W_edges,
    values,
    polygons: dict,
    colors: dict,
    cfg: dict,
):
    """Plot psychrometric chart with empirical comfort zones.

    Parameters
    ----------
    T_edges : array-like
        Temperature bin edges.
    W_edges : array-like
        Humidity-ratio bin edges.
    values : array-like
        Density matrix.
    polygons : dict
        Polygon vertices by zone name.
    colors : dict
        Matplotlib-compatible color per zone.
    cfg : dict
        Pipeline configuration.
    """
    chart_config_path = _resolve_chart_config_path(cfg)
    chart_cfg = load_chart_config(str(chart_config_path))
    chart = PsychChart(**chart_cfg)

    density_field = type(
        "DensityFieldWrapper",
        (),
        {
            "data": type(
                "DensityData",
                (),
                {
                    "T_edges": T_edges,
                    "W_edges": W_edges,
                    "values": values,
                },
            ),
            "cfg": DensityFieldConfig(
                cmap="viridis",
                alpha=0.7,
                normalize=True,
                colorbar=True,
            ),
        },
    )

    chart.density_fields = [density_field]
    ax = chart.draw()

    for name, poly in polygons.items():
        if poly is None or len(poly) == 0:
            continue

        ax.fill(
            poly[:, 0],
            poly[:, 1],
            alpha=0.3,
            color=colors.get(name, "gray"),
            label=name,
        )

    ax.legend()

    output_dir = _resolve_output_dir(cfg)
    output_path = output_dir / cfg.get("output_fig", "fig_comfort_polygon.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Figure saved to: {output_path}")

    plt.show()
