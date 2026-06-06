from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import pandas as pd
import seaborn as sns



from .analysis import find_series_max_point, find_consensus_negative_end, find_zero_crossing


def _finish_figure(fig, *, show_plot: bool) -> None:
    """Display or close a Matplotlib figure after saving."""
    if show_plot:
        fig.show()
    else:
        plt.close(fig)


def plot_window_results_academic(
    df_results: pd.DataFrame,
    output_dir: str | Path,
    x_tick_interval: int = 3,
    show_plot: bool = False,
) -> None:
    """
    Gera gráfico acadêmico da correlação por janela temporal
    com estilo mais editorial, próximo ao padrão de revistas científicas.
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

    # Garante tipo numérico
    x = pd.to_numeric(df_plot["window_h"])
    y_mean = pd.to_numeric(df_plot["mean_corr"])
    y_median = pd.to_numeric(df_plot["median_corr"])

    # ------------------------------------------------------------------
    # Estatísticas auxiliares
    # ------------------------------------------------------------------
    negative_phase_end = find_consensus_negative_end(x, y_mean, y_median)
    mean_crossing = find_zero_crossing(x, y_mean)
    median_crossing = find_zero_crossing(x, y_median)

    mean_max_x, mean_max_y = find_series_max_point(df_plot, "window_h", "mean_corr")
    median_max_x, median_max_y = find_series_max_point(df_plot, "window_h", "median_corr")

    # ------------------------------------------------------------------
    # Estilo (isolado via rc_context)
    # ------------------------------------------------------------------
    rc_params = {
        "font.family": "Arial",
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
        fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=450)

        # Cores sóbrias e acessíveis
        color_mean = "#0072B2"
        color_median = "#D55E00"

        # ------------------------------------------------------------------
        # Faixa negativa
        # ------------------------------------------------------------------
        if negative_phase_end is not None:
            ax.axvspan(
                float(x.iloc[0]),
                negative_phase_end,
                color="0.85",
                alpha=0.35,
                lw=0,
                zorder=0,
            )

        # ------------------------------------------------------------------
        # Intervalo entre métodos
        # ------------------------------------------------------------------
        ax.fill_between(
            x,
            y_mean,
            y_median,
            color=color_mean,
            alpha=0.12,
            lw=0,
            zorder=1,
        )

        # Linha horizontal em zero
        ax.axhline(
            0,
            color="0.25",
            linewidth=0.6,
            zorder=2,
        )

        # ------------------------------------------------------------------
        # Séries principais
        # ------------------------------------------------------------------
        ax.plot(
            x,
            y_mean,
            marker="o",
            markersize=2.8,
            linewidth=1.1,
            color=color_mean,
            markerfacecolor=color_mean,
            markeredgecolor=color_mean,
            label="Média",
            zorder=4,
        )

        ax.plot(
            x,
            y_median,
            marker="s",
            markersize=2.6,
            linestyle=(0, (4, 2)),
            linewidth=1.1,
            color=color_median,
            markerfacecolor=color_median,
            markeredgecolor=color_median,
            label="Mediana",
            zorder=4,
        )

        # ------------------------------------------------------------------
        # Linhas verticais de cruzamento
        # ------------------------------------------------------------------
        if mean_crossing is not None:
            ax.axvline(
                mean_crossing,
                color=color_mean,
                linestyle=(0, (1.2, 2.0)),
                linewidth=0.8,
                zorder=3,
            )

        if median_crossing is not None:
            ax.axvline(
                median_crossing,
                color=color_median,
                linestyle=(0, (1.2, 2.0)),
                linewidth=0.8,
                zorder=3,
            )

        # ------------------------------------------------------------------
        # Pontos de máximo: maiores e vazados
        # ------------------------------------------------------------------
        ax.scatter(
            mean_max_x,
            mean_max_y,
            s=55,
            facecolors="white",
            edgecolors=color_mean,
            linewidths=1.2,
            zorder=6,
        )

        ax.scatter(
            median_max_x,
            median_max_y,
            s=55,
            facecolors="white",
            edgecolors=color_median,
            linewidths=1.2,
            zorder=6,
        )

        # ------------------------------------------------------------------
        # Anotações com linhas curvas
        # ------------------------------------------------------------------
        # Posições ajustadas para ficar limpo visualmente no caso típico 1-24 h
        ax.annotate(
            f"Máx. média\n{mean_max_x:.0f} h; r = {mean_max_y:.3f}",
            xy=(mean_max_x, mean_max_y),
            xytext=(mean_max_x + 2.0, mean_max_y + 0.010),
            fontsize=6,
            color="black",
            ha="left",
            va="top",
            arrowprops=dict(
                arrowstyle="-",
                color=color_mean,
                lw=0.7,
                connectionstyle="arc3,rad=0.20",
                shrinkA=0,
                shrinkB=4,
            ),
            zorder=7,
        )

        ax.annotate(
            f"Máx. mediana\n{median_max_x:.0f} h; r = {median_max_y:.3f}",
            xy=(median_max_x, median_max_y),
            xytext=(median_max_x + 1.4, median_max_y - 0.022),
            fontsize=6,
            color="black",
            ha="left",
            va="top",
            arrowprops=dict(
                arrowstyle="-",
                color=color_median,
                lw=0.7,
                connectionstyle="arc3,rad=-0.18",
                shrinkA=0,
                shrinkB=4,
            ),
            zorder=7,
        )

        # Rótulos discretos das linhas de cruzamento
        if mean_crossing is not None:
            ax.text(
                mean_crossing - 0.10,
                min(y_mean.min(), y_median.min()) - 0.003,
                f"{mean_crossing:.2f} h",
                fontsize=5.5,
                color="black",
                ha="right",
                va="bottom",
                rotation=90,
            )

        if median_crossing is not None:
            ax.text(
                median_crossing + 0.10,
                min(y_mean.min(), y_median.min()) - 0.003,
                f"{median_crossing:.2f} h",
                fontsize=5.5,
                color="black",
                ha="left",
                va="bottom",
                rotation=90,
            )

        # ------------------------------------------------------------------
        # Eixos e aparência
        # ------------------------------------------------------------------
        ax.set_xlabel("Janela temporal (h)")
        ax.set_ylabel("Coeficiente de correlação")

        ax.set_title(
            "Impacto da Escala Temporal na Resposta de Ofegação",
            loc="left",
            pad=8,
            fontweight="bold",
        )

        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_interval))

        # Limites automáticos com pequena folga
        y_all_min = min(y_mean.min(), y_median.min())
        y_all_max = max(y_mean.max(), y_median.max())
        y_pad = 0.010

        ax.set_xlim(float(x.min()) - 0.2, float(x.max()) + 0.7)
        ax.set_ylim(y_all_min - y_pad, y_all_max + y_pad)

        # Sem grade pesada
        ax.grid(False)

        # Moldura limpa
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(direction="out", length=3, width=0.6)

        # ------------------------------------------------------------------
        # Legenda customizada
        # ------------------------------------------------------------------
        legend_handles = [
            Line2D(
                [0], [0],
                color=color_mean,
                lw=1.1,
                marker="o",
                ms=2.8,
                label="Média",
            ),
            Line2D(
                [0], [0],
                color=color_median,
                lw=1.1,
                linestyle=(0, (4, 2)),
                marker="s",
                ms=2.6,
                label="Mediana",
            ),
            mpatches.Patch(
                facecolor=color_mean,
                alpha=0.12,
                edgecolor="none",
                label="Intervalo inter-método",
            ),
            mpatches.Patch(
                facecolor="0.85",
                alpha=0.35,
                edgecolor="none",
                label="Fase negativa",
            ),
        ]

        if mean_crossing is not None:
            legend_handles.append(
                Line2D(
                    [0], [0],
                    color=color_mean,
                    lw=0.8,
                    linestyle=(0, (1.2, 2.0)),
                    label=f"Cruzamento média ≈ {mean_crossing:.2f} h",
                )
            )

        if median_crossing is not None:
            legend_handles.append(
                Line2D(
                    [0], [0],
                    color=color_median,
                    lw=0.8,
                    linestyle=(0, (1.2, 2.0)),
                    label=f"Cruzamento mediana ≈ {median_crossing:.2f} h",
                )
            )

        ax.legend(
            handles=legend_handles,
            loc="upper left",
            frameon=False,
            handlelength=2.4,
            borderaxespad=0.4,
            labelspacing=0.55,
        )

        plt.tight_layout(pad=0.6)

        plt.savefig(
            output_path / "temporal_scale_academic.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.savefig(
            output_path / "temporal_scale_academic.pdf",
            bbox_inches="tight",
        )
        plt.savefig(
            output_path / "temporal_scale_academic.svg",
            bbox_inches="tight",
        )

        _finish_figure(fig, show_plot=show_plot)

def plot_psychrometric(
    df: pd.DataFrame,
    output_dir: str | Path,
    kde_sample_size: int = 5000,
    kde_levels_fill: int = 10,
    kde_levels_contour: list[float] | None = None,
    bw_adjust: float = 1.2,
    scatter_sample_size: int | None = None,
    debug_timers: bool = False,
    show_plot: bool = False,
) -> None:
    """
    Gera gráfico psicrométrico com KDE, contornos e pontos observados,
    usando estilo acadêmico/editorial próximo ao padrão de revistas científicas.
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

    plot_df["temperatura"] = pd.to_numeric(plot_df["temperatura"], errors="coerce")
    plot_df["umidade"] = pd.to_numeric(plot_df["umidade"], errors="coerce")
    plot_df = plot_df.dropna(subset=["temperatura", "umidade"])

    if plot_df.empty:
        raise ValueError("Não há dados numéricos válidos para gerar o gráfico psicrométrico.")

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

    # ------------------------------------------------------------------
    # Estilo editorial, isolado para não afetar outros gráficos
    # ------------------------------------------------------------------
    rc_params = {
        "font.family": "Arial",
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
        fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=450)

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        kde_cmap = "Blues"
        contour_color = "0.15"
        scatter_color = "0.10"

        # --------------------------------------------------------------
        # KDE preenchido
        # --------------------------------------------------------------
        sns.kdeplot(
            data=df_kde,
            x="temperatura",
            y="umidade",
            fill=True,
            cmap=kde_cmap,
            levels=kde_levels_fill,
            thresh=0.05,
            alpha=0.70,
            bw_adjust=bw_adjust,
            ax=ax,
            zorder=1,
        )

        # --------------------------------------------------------------
        # Contornos
        # --------------------------------------------------------------
        sns.kdeplot(
            data=df_kde,
            x="temperatura",
            y="umidade",
            levels=kde_levels_contour,
            color=contour_color,
            linewidths=0.55,
            bw_adjust=bw_adjust,
            ax=ax,
            zorder=3,
        )

        # --------------------------------------------------------------
        # Pontos observados
        # --------------------------------------------------------------
        ax.scatter(
            df_scatter["temperatura"],
            df_scatter["umidade"],
            s=3.5,
            alpha=0.18,
            color=scatter_color,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

        # --------------------------------------------------------------
        # Eixos
        # --------------------------------------------------------------
        ax.set_xlabel("Temperatura do ar (°C)")
        ax.set_ylabel("Umidade relativa (%)")

        ax.set_title(
            "Região empírica de conforto térmico",
            loc="left",
            pad=8,
            fontweight="bold",
        )

        # Limites com pequena margem
        x_min = float(plot_df["temperatura"].min())
        x_max = float(plot_df["temperatura"].max())
        y_min = float(plot_df["umidade"].min())
        y_max = float(plot_df["umidade"].max())

        x_pad = max((x_max - x_min) * 0.04, 0.5)
        y_pad = max((y_max - y_min) * 0.04, 2.0)

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(max(0, y_min - y_pad), min(100, y_max + y_pad))

        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

        ax.tick_params(direction="out", length=3, width=0.6)

        # Sem grid pesada
        ax.grid(False)

        # Moldura limpa
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # --------------------------------------------------------------
        # Legenda customizada, sem caixa
        # --------------------------------------------------------------
        legend_handles = [
            Patch(
                facecolor="#9ecae1",
                edgecolor="none",
                alpha=0.70,
                label="Densidade KDE",
            ),
            Line2D(
                [0], [0],
                color=contour_color,
                lw=0.55,
                label="Contornos KDE",
            ),
            Line2D(
                [0], [0],
                marker="o",
                linestyle="none",
                markerfacecolor=scatter_color,
                markeredgecolor="none",
                alpha=0.35,
                markersize=3,
                label="Dados observados",
            ),
        ]

        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=False,
            handlelength=1.8,
            borderaxespad=0.4,
            labelspacing=0.55,
        )

        plt.tight_layout(pad=0.6)

        plt.savefig(
            output_path / "fig_psychrometric_comfort.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.savefig(
            output_path / "fig_psychrometric_comfort.pdf",
            bbox_inches="tight",
        )
        plt.savefig(
            output_path / "fig_psychrometric_comfort.svg",
            bbox_inches="tight",
        )

        _finish_figure(fig, show_plot=show_plot)

    if debug_timers:
        t_end = time.perf_counter()
        print(f"[TIMER] total plot_psychrometric: {t_end - t0:.4f} s")
