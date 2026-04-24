#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate all main figures for the thermal stress article.

Figures generated:
1. ATL distribution
2. Example time series for one animal
3. ATL vs panting with LOWESS + physiological saturation point
4. Relative Panting Index (RPI) vs ATL
5. Logistic vs LOWESS model comparison
6. Dynamic psychrometric chart with ATL / RPI zones

Expected input:
- Parquet dataset containing at least:
    animal_id
    data_hora
    temperatura
    umidade
    ofegacao

If your columns have different names, use command line arguments.

Example:
python generate_article_figures.py \
    --input dataset_final_plus_station.parquet \
    --output-dir figures_article \
    --temp-col temperatura_do_ar_bulbo_seco_horaria_c \
    --rh-col umidade_relativa_do_ar_horaria \
    --panting-col ofegacao_hora
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.nonparametric.smoothers_lowess import lowess


# ============================================================
# CONFIG
# ============================================================

@dataclass
class FigureConfig:
    input_path: str
    output_dir: str = "figures_article"
    animal_col: str = "animal_id"
    datetime_col: str = "data_hora"
    temp_col: str = "temperatura"
    rh_col: str = "umidade"
    panting_col: str = "ofegacao"

    thi_threshold: float = 72.0
    window_hours: int = 15
    resample_freq: str = "1h"

    lowess_frac: float = 0.15
    panting_event_threshold: Optional[float] = None
    panting_quantile: float = 0.75

    scatter_sample_n: int = 15000
    random_state: int = 42


# ============================================================
# BASIC UTILITIES
# ============================================================

def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def calc_thi(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    """
    THI formula widely used in dairy cattle studies.
    """
    return (1.8 * temp_c + 32.0) - (0.55 - 0.0055 * rh) * (1.8 * temp_c - 26.8)


def standardize_columns(df: pd.DataFrame, cfg: FigureConfig) -> pd.DataFrame:
    rename_map = {
        cfg.animal_col: "animal_id",
        cfg.datetime_col: "data_hora",
        cfg.temp_col: "temperatura",
        cfg.rh_col: "umidade",
        cfg.panting_col: "ofegacao",
    }
    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    df = df.rename(columns=rename_map).copy()
    df["data_hora"] = pd.to_datetime(df["data_hora"])
    df = df.sort_values(["animal_id", "data_hora"]).reset_index(drop=True)
    return df


# ============================================================
# DATA PREPARATION
# ============================================================

def resample_animal(group: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """
    Resample environmental data and preserve realistic gaps.

    Interpolation is limited to short gaps only.
    """
    g = group.copy().set_index("data_hora").sort_index()

    env = (
        g[["temperatura", "umidade"]]
        .resample(freq)
        .mean()
        .interpolate(limit=2, limit_area="inside")
    )

    pant = g["ofegacao"].resample(freq).mean()

    out = pd.concat([env, pant], axis=1)
    out["animal_id"] = group["animal_id"].iloc[0]
    return out.reset_index()


def build_analysis_dataset(df: pd.DataFrame, cfg: FigureConfig) -> pd.DataFrame:
    """
    Build analysis-ready dataset with THI, thermal excess, ATL, LOWESS expected panting, RPI.
    """
    # Resample per animal
    resampled = (
        df.groupby("animal_id", group_keys=False, observed=False)
        .apply(lambda x: resample_animal(x, cfg.resample_freq))
        .reset_index(drop=True)
    )

    # Compute THI and heat excess
    resampled["thi"] = calc_thi(resampled["temperatura"], resampled["umidade"])
    resampled["heat_excess"] = np.maximum(0.0, resampled["thi"] - cfg.thi_threshold)

    heat_col = f"atl_{cfg.window_hours}h"
    resampled[heat_col] = (
        resampled.groupby("animal_id", observed=False)["heat_excess"]
        .transform(lambda s: s.rolling(cfg.window_hours, min_periods=cfg.window_hours).sum())
    )

    # Remove missing values required for analysis
    analysis = resampled.dropna(subset=[heat_col, "ofegacao"]).copy()

    # Define panting event threshold
    if cfg.panting_event_threshold is None:
        pant_thr = float(analysis["ofegacao"].quantile(cfg.panting_quantile))
    else:
        pant_thr = float(cfg.panting_event_threshold)

    analysis["panting_event"] = (analysis["ofegacao"] >= pant_thr).astype(int)

    # LOWESS expected panting
    lowess_fit = lowess(
        endog=analysis["ofegacao"].values,
        exog=analysis[heat_col].values,
        frac=cfg.lowess_frac,
        return_sorted=True,
    )

    x_lowess = lowess_fit[:, 0]
    y_lowess = lowess_fit[:, 1]

    # Interpolate expected panting for every observation
    analysis["panting_expected"] = np.interp(
        analysis[heat_col].values,
        x_lowess,
        y_lowess,
        left=y_lowess[0],
        right=y_lowess[-1],
    )

    eps = 1e-6
    analysis["rpi"] = analysis["ofegacao"] / np.maximum(analysis["panting_expected"], eps)

    return analysis


# ============================================================
# MODELING
# ============================================================

def fit_logistic_model(df: pd.DataFrame, heat_col: str) -> dict:
    X = df[[heat_col]].values
    y = df["panting_event"].values

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)

    beta0 = float(model.intercept_[0])
    beta1 = float(model.coef_[0][0])

    threshold_50 = np.nan
    if abs(beta1) > 1e-12:
        threshold_50 = -beta0 / beta1

    return {
        "model": model,
        "auc": auc,
        "beta0": beta0,
        "beta1": beta1,
        "threshold_50": threshold_50,
    }


def estimate_saturation_point(df: pd.DataFrame, heat_col: str) -> float:
    """
    Estimate the physiological saturation point from LOWESS expected response.
    Heuristic:
    - sort by ATL
    - smooth already done
    - compute gradient
    - find first region after the main growth phase where slope becomes <= 0
    """
    tmp = df[[heat_col, "panting_expected"]].drop_duplicates().sort_values(heat_col).copy()
    x = tmp[heat_col].values
    y = tmp["panting_expected"].values

    if len(x) < 10:
        return float(np.nan)

    dy = np.gradient(y, x)

    # Focus only on the upper portion of ATL to avoid noisy early behavior
    x_q = np.quantile(x, 0.50)
    mask = x >= x_q

    x2 = x[mask]
    dy2 = dy[mask]

    idx = np.where(dy2 <= 0)[0]
    if len(idx) == 0:
        # fallback: maximum expected panting
        return float(x[np.argmax(y)])

    return float(x2[idx[0]])


# ============================================================
# PLOTTING
# ============================================================

def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_atl_distribution(df: pd.DataFrame, heat_col: str, outdir: Path) -> None:
    p25 = df[heat_col].quantile(0.25)
    p75 = df[heat_col].quantile(0.75)
    p90 = df[heat_col].quantile(0.90)

    plt.figure(figsize=(8, 5))
    plt.hist(df[heat_col], bins=60, edgecolor="black", alpha=0.8)

    plt.axvline(p25, linestyle="--", linewidth=2, label=f"P25 = {p25:.1f}")
    plt.axvline(p75, linestyle="--", linewidth=2, label=f"P75 = {p75:.1f}")
    plt.axvline(p90, linestyle="--", linewidth=2, label=f"P90 = {p90:.1f}")

    plt.xlabel("Accumulated thermal load (ATL)")
    plt.ylabel("Frequency")
    plt.title("Distribution of accumulated thermal load")
    plt.legend()

    savefig(outdir / "01_atl_distribution.png")


def plot_example_timeseries(df: pd.DataFrame, heat_col: str, outdir: Path) -> None:
    valid_counts = (
        df.dropna(subset=[heat_col, "thi"])
        .groupby("animal_id", observed=False)
        .size()
        .sort_values(ascending=False)
    )

    if valid_counts.empty:
        return

    animal_id = valid_counts.index[0]
    d = (
        df[df["animal_id"] == animal_id]
        .dropna(subset=[heat_col, "thi"])
        .sort_values("data_hora")
        .copy()
    )

    plt.figure(figsize=(12, 5))
    ax1 = plt.gca()
    ax1.plot(d["data_hora"], d[heat_col], linewidth=1.8, label="ATL")
    ax1.set_ylabel("ATL")

    ax2 = ax1.twinx()
    ax2.plot(d["data_hora"], d["thi"], linewidth=1.2, alpha=0.7, label="THI")
    ax2.set_ylabel("THI")

    ax1.set_xlabel("Datetime")
    ax1.set_title(f"Example thermal time series (animal {animal_id})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    savefig(outdir / "02_example_timeseries.png")


def plot_atl_vs_panting_lowess(
    df: pd.DataFrame,
    heat_col: str,
    saturation_point: float,
    outdir: Path,
    sample_n: int,
    random_state: int,
) -> None:
    d = df[[heat_col, "ofegacao", "panting_expected"]].dropna().copy()
    if len(d) > sample_n:
        d_sample = d.sample(sample_n, random_state=random_state)
    else:
        d_sample = d

    curve = (
        d[[heat_col, "panting_expected"]]
        .drop_duplicates()
        .sort_values(heat_col)
    )

    plt.figure(figsize=(8, 5))
    plt.scatter(d_sample[heat_col], d_sample["ofegacao"], s=8, alpha=0.15, label="Observed")
    plt.plot(curve[heat_col], curve["panting_expected"], linewidth=3, label="LOWESS expected")

    if np.isfinite(saturation_point):
        plt.axvline(
            saturation_point,
            linestyle="--",
            linewidth=2,
            label=f"Saturation point ≈ {saturation_point:.1f}",
        )

    plt.xlabel("Accumulated thermal load (ATL)")
    plt.ylabel("Panting response")
    plt.title("Panting response as a function of accumulated thermal load")
    plt.legend()

    savefig(outdir / "03_atl_vs_panting_lowess.png")


def plot_rpi_vs_atl(
    df: pd.DataFrame,
    heat_col: str,
    saturation_point: float,
    outdir: Path,
    sample_n: int,
    random_state: int,
) -> None:
    d = df[[heat_col, "rpi"]].dropna().copy()
    if len(d) > sample_n:
        d_sample = d.sample(sample_n, random_state=random_state)
    else:
        d_sample = d

    # Median trend by bins
    d["bin"] = pd.qcut(d[heat_col], q=min(30, d[heat_col].nunique()), duplicates="drop")
    grouped = d.groupby("bin", observed=False)["rpi"].median()
    centers = np.array([i.mid for i in grouped.index])

    plt.figure(figsize=(8, 5))
    plt.scatter(d_sample[heat_col], d_sample["rpi"], s=8, alpha=0.10, label="Observed RPI")
    plt.plot(centers, grouped.values, linewidth=3, label="Median RPI trend")
    plt.axhline(1.0, linestyle="--", linewidth=2, label="RPI = 1")

    if np.isfinite(saturation_point):
        plt.axvline(
            saturation_point,
            linestyle="--",
            linewidth=2,
            label=f"Saturation point ≈ {saturation_point:.1f}",
        )

    plt.xlabel("Accumulated thermal load (ATL)")
    plt.ylabel("Relative Panting Index (RPI)")
    plt.title("Relative panting efficiency across thermal load")
    plt.legend()

    savefig(outdir / "04_rpi_vs_atl.png")


def plot_model_comparison(
    df: pd.DataFrame,
    heat_col: str,
    logistic_result: dict,
    outdir: Path,
) -> None:
    curve = (
        df[[heat_col, "panting_expected"]]
        .drop_duplicates()
        .sort_values(heat_col)
    )

    x_grid = np.linspace(df[heat_col].min(), df[heat_col].max(), 400).reshape(-1, 1)
    y_logit = logistic_result["model"].predict_proba(x_grid)[:, 1]

    # Build LOWESS probability-like curve from binary event
    tmp = df[[heat_col, "panting_event"]].dropna().copy()
    tmp["bin"] = pd.qcut(tmp[heat_col], q=min(25, tmp[heat_col].nunique()), duplicates="drop")
    grouped = tmp.groupby("bin", observed=False)["panting_event"].mean()
    centers = np.array([i.mid for i in grouped.index])

    plt.figure(figsize=(8, 5))
    plt.plot(x_grid[:, 0], y_logit, linewidth=3, label=f"Logistic (AUC = {logistic_result['auc']:.3f})")
    plt.plot(centers, grouped.values, linewidth=3, label="Observed event probability by ATL bin")

    thr = logistic_result["threshold_50"]
    if np.isfinite(thr):
        plt.axvline(thr, linestyle="--", linewidth=2, label=f"Logistic 50% threshold ≈ {thr:.1f}")

    plt.xlabel("Accumulated thermal load (ATL)")
    plt.ylabel("Probability of panting event")
    plt.title("Comparison between logistic model and observed event pattern")
    plt.legend()

    savefig(outdir / "05_logistic_vs_observed_probability.png")


def plot_dynamic_psychrometric_chart(
    df: pd.DataFrame,
    heat_col: str,
    saturation_point: float,
    outdir: Path,
) -> None:
    """
    Dynamic psychrometric-like scatter plot in T x RH space,
    colored by zone derived from ATL and RPI.
    """

    d = df[["temperatura", "umidade", heat_col, "rpi"]].dropna().copy()

    # Define dynamic zones
    p25 = d[heat_col].quantile(0.25)
    p75 = d[heat_col].quantile(0.75)

    def classify(row) -> str:
        atl = row[heat_col]
        rpi = row["rpi"]

        if atl <= p25 and 0.8 <= rpi <= 1.2:
            return "Homeostasis"
        if atl <= p75 and rpi > 1.0:
            return "Physiological alert"
        if np.isfinite(saturation_point) and atl >= saturation_point and rpi < 1.0:
            return "Thermal fatigue"
        return "Transition"

    d["zone"] = d.apply(classify, axis=1)

    # Plot
    plt.figure(figsize=(8, 6))

    for zone in ["Homeostasis", "Transition", "Physiological alert", "Thermal fatigue"]:
        sub = d[d["zone"] == zone]
        if sub.empty:
            continue
        plt.scatter(
            sub["temperatura"],
            sub["umidade"],
            s=8,
            alpha=0.18,
            label=zone,
        )

    plt.xlabel("Air temperature (°C)")
    plt.ylabel("Relative humidity (%)")
    plt.title("Dynamic psychrometric zones based on ATL and RPI")
    plt.legend()

    savefig(outdir / "06_dynamic_psychrometric_zones.png")


def save_summary(
    df: pd.DataFrame,
    heat_col: str,
    logistic_result: dict,
    saturation_point: float,
    panting_threshold: float,
    outdir: Path,
) -> None:
    summary = {
        "n_rows_analysis": len(df),
        "heat_variable": heat_col,
        "panting_event_threshold": panting_threshold,
        "atl_p25": float(df[heat_col].quantile(0.25)),
        "atl_p75": float(df[heat_col].quantile(0.75)),
        "atl_p90": float(df[heat_col].quantile(0.90)),
        "saturation_point": float(saturation_point) if np.isfinite(saturation_point) else np.nan,
        "logistic_auc": float(logistic_result["auc"]),
        "logistic_beta0": float(logistic_result["beta0"]),
        "logistic_beta1": float(logistic_result["beta1"]),
        "logistic_threshold_50": float(logistic_result["threshold_50"])
        if np.isfinite(logistic_result["threshold_50"]) else np.nan,
    }

    pd.DataFrame([summary]).to_csv(outdir / "figure_summary_metrics.csv", index=False)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all article figures automatically.")
    parser.add_argument("--input", required=True, help="Path to input parquet dataset.")
    parser.add_argument("--output-dir", default="figures_article")
    parser.add_argument("--animal-col", default="animal_id")
    parser.add_argument("--datetime-col", default="data_hora")
    parser.add_argument("--temp-col", default="temperatura")
    parser.add_argument("--rh-col", default="umidade")
    parser.add_argument("--panting-col", default="ofegacao")
    parser.add_argument("--thi-threshold", type=float, default=72.0)
    parser.add_argument("--window-hours", type=int, default=15)
    parser.add_argument("--lowess-frac", type=float, default=0.15)
    parser.add_argument("--panting-threshold", type=float, default=None)
    parser.add_argument("--panting-quantile", type=float, default=0.75)

    args = parser.parse_args()

    cfg = FigureConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        animal_col=args.animal_col,
        datetime_col=args.datetime_col,
        temp_col=args.temp_col,
        rh_col=args.rh_col,
        panting_col=args.panting_col,
        thi_threshold=args.thi_threshold,
        window_hours=args.window_hours,
        lowess_frac=args.lowess_frac,
        panting_event_threshold=args.panting_threshold,
        panting_quantile=args.panting_quantile,
    )

    outdir = ensure_output_dir(cfg.output_dir)

    print("[INFO] Loading dataset...")
    df = pd.read_parquet(cfg.input_path)

    print("[INFO] Standardizing columns...")
    df = standardize_columns(df, cfg)

    print("[INFO] Building analysis dataset...")
    analysis = build_analysis_dataset(df, cfg)

    heat_col = f"atl_{cfg.window_hours}h"
    panting_threshold = (
        cfg.panting_event_threshold
        if cfg.panting_event_threshold is not None
        else float(analysis["ofegacao"].quantile(cfg.panting_quantile))
    )

    print("[INFO] Fitting logistic model...")
    logistic_result = fit_logistic_model(analysis, heat_col)

    print("[INFO] Estimating physiological saturation point...")
    saturation_point = estimate_saturation_point(analysis, heat_col)

    print("[INFO] Generating figures...")
    plot_atl_distribution(analysis, heat_col, outdir)
    plot_example_timeseries(analysis, heat_col, outdir)
    plot_atl_vs_panting_lowess(
        analysis, heat_col, saturation_point, outdir,
        cfg.scatter_sample_n, cfg.random_state
    )
    plot_rpi_vs_atl(
        analysis, heat_col, saturation_point, outdir,
        cfg.scatter_sample_n, cfg.random_state
    )
    plot_model_comparison(analysis, heat_col, logistic_result, outdir)
    plot_dynamic_psychrometric_chart(analysis, heat_col, saturation_point, outdir)

    print("[INFO] Saving summary metrics...")
    save_summary(
        analysis,
        heat_col,
        logistic_result,
        saturation_point,
        panting_threshold,
        outdir,
    )

    print("[DONE] Figures saved to:", outdir.resolve())
    print("[DONE] Logistic AUC:", round(logistic_result["auc"], 4))
    print("[DONE] Physiological saturation point:", round(saturation_point, 4) if np.isfinite(saturation_point) else "NaN")
    print("[DONE] Panting event threshold:", panting_threshold)


if __name__ == "__main__":
    main()
