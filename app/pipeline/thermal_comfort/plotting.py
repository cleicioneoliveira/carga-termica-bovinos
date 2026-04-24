from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pandas as pd
import seaborn as sns

from .analysis import find_series_max_point, find_consensus_negative_end, find_zero_crossing

def plot_window_results_academic(
    df_results: pd.DataFrame,
    output_dir: str | Path,
    x_tick_interval: int = 3,
) -> None:
    """
    Gera gráfico acadêmico da correlação por janela temporal.
    """
    required_columns = {"window_h", "mean_corr", "median_corr"}
    missing_columns = required_columns - set(df_results.columns)
    if missing_columns:
        raise ValueError(
            f"DataFrame inválido. Colunas ausentes: {sorted(missing_columns)}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df_plot = (
        df_results.loc[:, ["window_h", "mean_corr", "median_corr"]]
        .dropna()
        .sort_values("window_h")
        .reset_index(drop=True)
    )

    if df_plot.empty:
        raise ValueError("Não há dados válidos para plotagem após limpeza.")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    color_mean = "#1f77b4"
    color_median = "#ff7f0e"

    ax.plot(
        df_plot["window_h"],
        df_plot["mean_corr"],
        marker="o",
        markersize=5,
        linewidth=1.5,
        color=color_mean,
        label="Média (Mean)",
        zorder=3,
    )

    ax.plot(
        df_plot["window_h"],
        df_plot["median_corr"],
        marker="s",
        markersize=5,
        linestyle="--",
        linewidth=1.5,
        color=color_median,
        label="Mediana (Median)",
        zorder=3,
    )

    ax.fill_between(
        df_plot["window_h"],
        df_plot["mean_corr"],
        df_plot["median_corr"],
        color="gray",
        alpha=0.15,
        label="Intervalo Inter-método",
        zorder=1,
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.3, zorder=2)

    negative_phase_end = find_consensus_negative_end(
        df_plot["window_h"],
        df_plot["mean_corr"],
        df_plot["median_corr"],
    )

    if negative_phase_end is not None:
        ax.axvspan(
            float(df_plot["window_h"].iloc[0]),
            negative_phase_end,
            color="gray",
            alpha=0.05,
            label="Fase Negativa",
            zorder=0,
        )

    mean_crossing = find_zero_crossing(df_plot["window_h"], df_plot["mean_corr"])
    median_crossing = find_zero_crossing(df_plot["window_h"], df_plot["median_corr"])

    if mean_crossing is not None:
        ax.axvline(
            mean_crossing,
            color=color_mean,
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
            label=f"Cruzamento média ≈ {mean_crossing:.2f} h",
        )

    if median_crossing is not None:
        ax.axvline(
            median_crossing,
            color=color_median,
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
            label=f"Cruzamento mediana ≈ {median_crossing:.2f} h",
        )

    mean_max_x, mean_max_y = find_series_max_point(df_plot, "window_h", "mean_corr")
    median_max_x, median_max_y = find_series_max_point(df_plot, "window_h", "median_corr")

    ax.scatter(mean_max_x, mean_max_y, s=40, color=color_mean, zorder=5)
    ax.annotate(
        f"Máx. média\n({mean_max_x:.1f} h, {mean_max_y:.2f})",
        xy=(mean_max_x, mean_max_y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )

    ax.scatter(median_max_x, median_max_y, s=40, color=color_median, zorder=5)
    ax.annotate(
        f"Máx. mediana\n({median_max_x:.1f} h, {median_max_y:.2f})",
        xy=(median_max_x, median_max_y),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=9,
    )

    ax.set_xlabel("Janela temporal (horas)", fontweight="bold")
    ax.set_ylabel("Coeficiente de Correlação", fontweight="bold")
    ax.set_title(
        "Impacto da Escala Temporal na Resposta de Ofegação",
        pad=20,
        fontweight="bold",
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_interval))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.tick_params(direction="out", length=6, width=1)

    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=False)

    plt.tight_layout()
    plt.savefig(output_path / "temporal_scale_academic.png", dpi=600, bbox_inches="tight")
    plt.savefig(output_path / "temporal_scale_academic.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_psychrometric(
    df: pd.DataFrame,
    output_dir: str | Path,
    kde_sample_size: int = 5000,
    kde_levels_fill: int = 10,
    kde_levels_contour: list[float] | None = None,
    bw_adjust: float = 1.2,
    scatter_sample_size: int | None = None,
    debug_timers: bool = False,
) -> None:
    """
    Gera o gráfico psicrométrico com KDE, contorno e scatter.
    """
    if kde_levels_contour is None:
        kde_levels_contour = [0.2, 0.4, 0.6, 0.8]

    t0 = time.perf_counter()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    required_cols = ["temperatura", "umidade"]
    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes para plot: {missing}")

    plot_df = df.dropna(subset=["temperatura", "umidade"]).copy()

    if plot_df.empty:
        raise ValueError("Não há dados válidos para gerar o gráfico psicrométrico.")

    if debug_timers:
        t1 = time.perf_counter()
        print(f"[TIMER] preparação inicial: {t1 - t0:.4f} s")

    if len(plot_df) > kde_sample_size:
        df_kde = plot_df.sample(n=kde_sample_size, random_state=42)
    else:
        df_kde = plot_df

    if scatter_sample_size is not None and len(plot_df) > scatter_sample_size:
        df_scatter = plot_df.sample(n=scatter_sample_size, random_state=42)
    else:
        df_scatter = plot_df

    if debug_timers:
        t2 = time.perf_counter()
        print(f"[TIMER] amostragem: {t2 - t1:.4f} s")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    sns.kdeplot(
        data=df_kde,
        x="temperatura",
        y="umidade",
        fill=True,
        cmap="viridis",
        levels=kde_levels_fill,
        thresh=0.05,
        alpha=0.6,
        bw_adjust=bw_adjust,
        ax=ax,
    )

    sns.kdeplot(
        data=df_kde,
        x="temperatura",
        y="umidade",
        levels=kde_levels_contour,
        color="black",
        linewidths=1,
        bw_adjust=bw_adjust,
        ax=ax,
    )

    ax.scatter(
        df_scatter["temperatura"],
        df_scatter["umidade"],
        s=5,
        alpha=0.2,
        color="blue",
        label="Dados de conforto",
    )

    ax.set_xlabel("Temperatura (°C)")
    ax.set_ylabel("Umidade Relativa (%)")
    ax.set_title("Região empírica de conforto térmico baseada em dados")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "fig_psychrometric_comfort.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_path / "fig_psychrometric_comfort.pdf", bbox_inches="tight")
    plt.close(fig)

    if debug_timers:
        t_end = time.perf_counter()
        print(f"[TIMER] total plot_psychrometric: {t_end - t0:.4f} s")
