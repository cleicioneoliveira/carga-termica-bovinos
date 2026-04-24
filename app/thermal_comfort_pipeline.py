from __future__ import annotations

import argparse
from typing import Optional, Tuple, Literal, Any, Sequence
from pathlib import Path
import time

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import ttest_1samp, wilcoxon
import seaborn as sns


from local_io.inmet import read_inmet_csv
#from time.merge import merge_time_series 
from util.profiling import run_with_profile, profiled
from pipeline.thermal_comfort.ITU import calculate_itu
#from .gemini.entalpia import calcular_entalpia as calculate_itu

import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from dataclasses import dataclass


# ==========================================================
# CONFIG PADRÃO
# ==========================================================
DEFAULT_THI_THRESHOLD = 72
DEFAULT_WINDOWS = list(range(3, 25, 3))   # 3, 6, ..., 24
DEFAULT_WINDOW = 15
DEFAULT_MIN_DURATION = 3




# ==========================================================
# MODELS
# ==========================================================


@dataclass(frozen=True)
class ModelSummary:
    """Resumo objetivo de ajuste de modelo."""
    formula: str
    n_obs: int
    aic: float | None
    bic: float | None
    rsquared: float | None
    rsquared_adj: float | None


def prepare_model_data(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    numeric_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Prepara os dados para modelagem, removendo valores inválidos e
    garantindo tipo numérico nas colunas de modelagem.
    """
    missing = set(required_columns).difference(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {', '.join(sorted(missing))}")

    clean = df.loc[:, required_columns].copy()
    clean = clean.replace([np.inf, -np.inf], np.nan)

    if numeric_columns is not None:
        for col in numeric_columns:
            if col not in clean.columns:
                raise ValueError(f"Coluna numérica ausente em prepare_model_data: {col}")
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna()

    if clean.empty:
        raise ValueError("Não há dados válidos após limpeza.")

    return clean


def calculate_vif(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """
    Calcula VIF para avaliar colinearidade entre preditores numéricos.
    """
    if not features:
        raise ValueError("Nenhuma feature informada para cálculo de VIF.")

    missing = set(features).difference(df.columns)
    if missing:
        raise ValueError(f"Features ausentes para VIF: {', '.join(sorted(missing))}")

    x = df.loc[:, features].copy()

    for col in features:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x = x.replace([np.inf, -np.inf], np.nan).dropna()

    if x.empty:
        raise ValueError("Não há dados válidos para cálculo de VIF após coerção numérica.")

    constant_features = [col for col in x.columns if x[col].nunique(dropna=True) <= 1]
    if constant_features:
        raise ValueError(
            "Não é possível calcular VIF com colunas constantes: "
            + ", ".join(constant_features)
        )

    x_matrix = x.to_numpy(dtype=float)

    vif_values = [
        float(variance_inflation_factor(x_matrix, i))
        for i in range(x_matrix.shape[1])
    ]

    return pd.DataFrame(
        {
            "feature": list(features),
            "vif": vif_values,
        }
    ).sort_values("vif", ascending=False, ignore_index=True)


def fit_ols_model(
    df: pd.DataFrame,
    formula: str,
) -> tuple[object, ModelSummary]:
    """
    Ajusta um modelo OLS via fórmula.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame preparado.
    formula : str
        Fórmula statsmodels.

    Returns
    -------
    tuple[object, ModelSummary]
        Resultado do ajuste e resumo objetivo.
    """
    model = smf.ols(formula=formula, data=df).fit()

    summary = ModelSummary(
        formula=formula,
        n_obs=int(model.nobs),
        aic=float(model.aic),
        bic=float(model.bic),
        rsquared=float(model.rsquared),
        rsquared_adj=float(model.rsquared_adj),
    )
    return model, summary


def fit_mixed_model(
    df: pd.DataFrame,
    formula: str,
    group_col: str,
) -> tuple[object, ModelSummary]:
    """
    Ajusta um modelo linear misto com intercepto aleatório por grupo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame preparado.
    formula : str
        Fórmula dos efeitos fixos.
    group_col : str
        Coluna identificadora do grupo.

    Returns
    -------
    tuple[object, ModelSummary]
        Resultado do ajuste e resumo objetivo.
    """
    model = smf.mixedlm(formula=formula, data=df, groups=df[group_col]).fit()

    summary = ModelSummary(
        formula=formula,
        n_obs=int(df.shape[0]),
        aic=float(model.aic) if model.aic is not None else None,
        bic=float(model.bic) if model.bic is not None else None,
        rsquared=None,
        rsquared_adj=None,
    )
    return model, summary





#
#helpers
#

def _find_series_max_point(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> tuple[float, float]:
    """
    Retorna o ponto de máximo de uma série.

    Args:
        df: DataFrame contendo os dados.
        x_col: Nome da coluna do eixo X.
        y_col: Nome da coluna da série.

    Returns:
        Tupla no formato (x_do_maximo, valor_maximo).
    """
    if df.empty:
        raise ValueError("O DataFrame está vazio.")

    valid_series = df[y_col].dropna()
    if valid_series.empty:
        raise ValueError(f"A coluna '{y_col}' não possui valores válidos.")

    idx = valid_series.idxmax()
    x_at_max = float(df.loc[idx, x_col])
    y_max = float(df.loc[idx, y_col])

    return x_at_max, y_max

def _find_zero_crossing(x: pd.Series, y: pd.Series) -> Optional[float]:
    """
    Retorna o primeiro ponto em que a série cruza y = 0, usando interpolação linear.

    Args:
        x: Série com os valores do eixo X.
        y: Série com os valores do eixo Y.

    Returns:
        O valor interpolado de x onde ocorre o primeiro cruzamento com zero.
        Retorna None se não houver cruzamento.
    """
    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)

    if len(x_values) != len(y_values):
        raise ValueError("x e y devem ter o mesmo comprimento.")

    if len(x_values) < 2:
        return None

    for i in range(len(y_values) - 1):
        y0 = y_values[i]
        y1 = y_values[i + 1]
        x0 = x_values[i]
        x1 = x_values[i + 1]

        if y0 == 0:
            return float(x0)

        if y0 * y1 < 0:
            # Interpolação linear:
            # x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))

        if y1 == 0:
            return float(x1)

    return None


def _find_consensus_negative_end(
    x: pd.Series,
    y1: pd.Series,
    y2: pd.Series,
) -> Optional[float]:
    """
    Retorna o fim da fase em que y1 e y2 estão simultaneamente abaixo de zero.

    O retorno é interpolado entre os dois pontos onde essa condição deixa de ser
    verdadeira pela primeira vez.

    Args:
        x: Série com os valores do eixo X.
        y1: Primeira série Y.
        y2: Segunda série Y.

    Returns:
        O valor interpolado de x onde termina a fase negativa comum.
        Retorna None se a fase negativa comum não existir ou não puder ser inferida.
    """
    x_values = x.to_numpy(dtype=float)
    y1_values = y1.to_numpy(dtype=float)
    y2_values = y2.to_numpy(dtype=float)

    if not (len(x_values) == len(y1_values) == len(y2_values)):
        raise ValueError("x, y1 e y2 devem ter o mesmo comprimento.")

    if len(x_values) < 2:
        return None

    both_negative = (y1_values < 0) & (y2_values < 0)

    if not both_negative.any():
        return None

    last_negative_idx = np.where(both_negative)[0][-1]

    if last_negative_idx == len(x_values) - 1:
        return float(x_values[last_negative_idx])

    x0 = x_values[last_negative_idx]
    x1 = x_values[last_negative_idx + 1]

    y1_0 = y1_values[last_negative_idx]
    y1_1 = y1_values[last_negative_idx + 1]

    y2_0 = y2_values[last_negative_idx]
    y2_1 = y2_values[last_negative_idx + 1]

    crossings: list[float] = []

    if y1_0 < 0 <= y1_1:
        crossings.append(x0 - y1_0 * (x1 - x0) / (y1_1 - y1_0))
    elif y1_0 == 0:
        crossings.append(x0)

    if y2_0 < 0 <= y2_1:
        crossings.append(x0 - y2_0 * (x1 - x0) / (y2_1 - y2_0))
    elif y2_0 == 0:
        crossings.append(x0)

    if crossings:
        return float(min(crossings))

    return float(x0)

# ==========================================================
# PADRONIZAÇÃO E LIMPEZA
# ==========================================================

def standardize_columns_(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "temperatura_do_ar_bulbo_seco_horaria_c": "temperatura",
        "umidade_relativa_do_ar_horaria": "umidade",
        "ofegacao_hora": "ofegacao",
    }
    return df.rename(columns=mapping)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "temperatura_compost_1": "temperatura",
        "humidade_compost_1": "umidade",
        "ofegacao_hora": "ofegacao",
        "ruminacao_hora": "ruminacao",
        "atividade_hora": "atividade",
        "ocio_hora": "ocio",
    }


    return df.rename(columns=mapping)


def convert_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_columns = ["animal_id", "data_hora", "temperatura", "umidade", "ofegacao"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    numeric_columns = [
        "temperatura",
        "umidade",
        "ofegacao",
        "atividade",
        "ruminacao",
        "ocio",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")

    df = df.dropna(subset=["animal_id", "data_hora", "temperatura", "umidade", "ofegacao"])
    df = df.sort_values(["animal_id", "data_hora"]).reset_index(drop=True)

    return df


# ==========================================================
# THI E CARGA TÉRMICA
# ==========================================================
def calculate_specific_humidity(
    temperatura_c: pd.Series | np.ndarray,
    umidade_relativa: pd.Series | np.ndarray,
    pressure_kpa: float = 101.325
) -> np.ndarray:
    """
    Calcula umidade específica (kg/kg) a partir de temperatura e UR.
    """
    t = np.asarray(temperatura_c, dtype=float)
    rh = np.asarray(umidade_relativa, dtype=float)

    # pressão de saturação do vapor (kPa)
    es = 0.6108 * np.exp((17.27 * t) / (t + 237.3))

    # pressão real de vapor (kPa)
    e = (rh / 100.0) * es

    # razão de mistura (kg/kg de ar seco)
    r = 0.622 * e / (pressure_kpa - e)

    # umidade específica (kg/kg de ar úmido)
    q = r / (1.0 + r)

    return q

def calcular_dpv(temp, ur):
    """
    Calcula o Déficit de Pressão de Vapor (DPV) em kPa.
    
    Parâmetros:
    temp (float ou np.array): Temperatura do ar em °C.
    ur (float ou np.array): Umidade Relativa em %.
    
    Retorna:
    dpv (float ou np.array): Déficit de pressão de vapor em kPa.
    """
    # 1. Cálculo da Pressão de Saturação de Vapor (es) em kPa
    # Fórmula de Tetens
    es = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
    
    # 2. Cálculo da Pressão Real de Vapor (ea) em kPa
    ea = es * (ur / 100.0)
    
    # 3. O DPV é a diferença entre o que o ar suporta e o que ele tem
    dpv = es - ea
    
    return dpv


def add_thi_and_heat_excess(
    df: pd.DataFrame,
    thi_threshold: float = DEFAULT_THI_THRESHOLD
) -> pd.DataFrame:
    df = df.copy()

    print(f"THI Threshold : {thi_threshold}")
    df["thi"] = calculate_itu(df["temperatura"], df["umidade"])
    df["heat_excess"] = np.maximum(0, df["thi"] - thi_threshold)
    return df


def add_heat_load(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()
    heat_col = f"heat_load_{window}h"

    df[heat_col] = (
        df.groupby("animal_id", observed=False)["heat_excess"]
        .transform(lambda x: x.rolling(window, min_periods=1).sum())
    )

    return df


# ==========================================================
# ANÁLISE DE JANELAS
# ==========================================================



def analyze_per_animal(
    df: pd.DataFrame,
    heat_col: str,
    *,
    id_col: str = "animal_id",
    current_behavior: str = "ofegacao",
    min_samples_per_animal: int = 50,
) -> np.ndarray:
    """
    Calcula, para cada animal, a correlação de Pearson entre uma variável
    térmica e uma variável comportamental/fisiológica, retornando apenas
    os coeficientes válidos.

    A função agrupa o conjunto de dados pela coluna identificadora informada
    em ``id_col`` e, para cada grupo, seleciona apenas as observações válidas
    das colunas de interesse. Valores infinitos são convertidos para ``NaN``
    e removidos antes do cálculo. Animais com número insuficiente de amostras
    válidas ou com séries constantes em qualquer uma das variáveis são
    ignorados, pois nesses casos a correlação é indefinida.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada contendo, obrigatoriamente, as colunas
        informadas em ``id_col``, ``heat_col`` e ``current_behavior``.
    heat_col : str
        Nome da coluna que representa a variável térmica a ser correlacionada
        por animal, por exemplo temperatura, umidade específica, entalpia,
        ITU ou carga térmica acumulada.
    id_col : str, optional
        Nome da coluna identificadora do animal. O padrão é ``"animal_id"``.
    current_behavior : str, optional
        Nome da coluna da variável resposta a ser analisada para cada animal.
        O padrão é ``"ofegacao"``.
    min_samples_per_animal : int, optional
        Número mínimo de observações válidas exigidas para que um animal seja
        incluído no cálculo da correlação. O padrão é ``50``.

    Returns
    -------
    np.ndarray
        Vetor unidimensional contendo os coeficientes de correlação de Pearson
        calculados para cada animal elegível. O array pode ser vazio caso
        nenhum animal atenda aos critérios mínimos da análise.

    Raises
    ------
    ValueError
        Lançado quando uma ou mais colunas obrigatórias não estão presentes
        no DataFrame de entrada.
    ValueError
        Lançado quando ``min_samples_per_animal`` for menor que 2.

    Notes
    -----
    - A correlação é calculada individualmente para cada animal, e não sobre
      o conjunto total de dados.
    - Observações com ``NaN``, ``+inf`` ou ``-inf`` nas colunas analisadas
      são descartadas antes do cálculo.
    - Animais com menos de ``min_samples_per_animal`` observações válidas
      são excluídos.
    - Animais cujas séries sejam constantes em ``heat_col`` ou
      ``current_behavior`` também são excluídos, pois a correlação de Pearson
      não é definida nesses casos.

    Examples
    --------
    >>> corrs = analyze_per_animal(
    ...     df=data,
    ...     heat_col="cta_15h",
    ...     current_behavior="ofegacao",
    ...     min_samples_per_animal=50,
    ... )
    >>> corrs.shape
    (n,)

    >>> corrs = analyze_per_animal(
    ...     df=data,
    ...     heat_col="entalpia",
    ...     current_behavior="ruminacao",
    ...     min_samples_per_animal=30,
    ... )
    """
    if min_samples_per_animal < 2:
        raise ValueError("min_samples_per_animal deve ser maior ou igual a 2.")

    required_columns = {id_col, heat_col, current_behavior}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colunas obrigatórias ausentes no DataFrame: {missing}")

    correlations: list[float] = []

    for _, group in df.groupby(id_col, observed=True):
        subset = group[[heat_col, current_behavior]].replace([np.inf, -np.inf], np.nan)
        subset = subset.dropna()

        if len(subset) < min_samples_per_animal:
            continue

        x = subset[heat_col]
        y = subset[current_behavior]

        if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
            continue

        corr = x.corr(y)

        if pd.notna(corr) and np.isfinite(corr):
            correlations.append(float(corr))

    return np.asarray(correlations, dtype=float)

def compute_significance(corr_values: np.ndarray) -> tuple[float, float]:
    """
    Calcula p-valores para testar se a distribuição das correlações
    difere de zero usando t-test e Wilcoxon.

    Parameters
    ----------
    corr_values : np.ndarray
        Array com correlações por animal.

    Returns
    -------
    tuple[float, float]
        (p_ttest, p_wilcoxon)
    """
    values = np.asarray(corr_values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan

    p_t = np.nan
    p_w = np.nan

    try:
        p_t = float(ttest_1samp(values, popmean=0.0, nan_policy="omit").pvalue)
    except (ValueError, RuntimeError, FloatingPointError):
        p_t = np.nan

    try:
        # Wilcoxon requer diferenças não degeneradas em vários cenários.
        p_w = float(wilcoxon(values).pvalue)
    except (ValueError, RuntimeError, FloatingPointError):
        p_w = np.nan

    return p_t, p_w


def run_window_analysis(
    df: pd.DataFrame,
    windows: list[int],
    *,
    min_samples_per_animal: int = 50,
) -> pd.DataFrame:
    """
    Executa a análise para múltiplas janelas de tempo e retorna
    estatísticas agregadas das correlações por janela.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    windows : list[int]
        Lista de janelas em horas.
    min_samples_per_animal : int, optional
        Número mínimo de observações válidas por animal.

    Returns
    -------
    pd.DataFrame
        DataFrame com uma linha por janela analisada.
    """
    results: list[dict[str, float | int]] = []

    for window in windows:
        print(f"[INFO] Testando janela: {window}h")

        temp_df = add_heat_load(df, window)
        heat_col = f"heat_load_{window}h"

        corr_values = analyze_per_animal(
            temp_df,
            heat_col,
            min_samples_per_animal=min_samples_per_animal,
        )

        valid_corrs = corr_values[np.isfinite(corr_values)]

        if valid_corrs.size > 0:
            mean_corr = float(valid_corrs.mean())
            median_corr = float(np.median(valid_corrs))
            std_corr = float(valid_corrs.std(ddof=1)) if valid_corrs.size > 1 else np.nan
            min_corr = float(valid_corrs.min())
            max_corr = float(valid_corrs.max())
            q25_corr = float(np.percentile(valid_corrs, 25))
            q75_corr = float(np.percentile(valid_corrs, 75))
            iqr_corr = float(q75_corr - q25_corr)

            abs_mean_corr = float(np.mean(np.abs(valid_corrs)))
            abs_median_corr = float(np.median(np.abs(valid_corrs)))

            positives = int(np.sum(valid_corrs > 0))
            negatives = int(np.sum(valid_corrs < 0))
            zeros = int(np.sum(valid_corrs == 0))

            positive_ratio = float(positives / valid_corrs.size)
            negative_ratio = float(negatives / valid_corrs.size)
            zero_ratio = float(zeros / valid_corrs.size)
        else:
            mean_corr = np.nan
            median_corr = np.nan
            std_corr = np.nan
            min_corr = np.nan
            max_corr = np.nan
            q25_corr = np.nan
            q75_corr = np.nan
            iqr_corr = np.nan
            abs_mean_corr = np.nan
            abs_median_corr = np.nan
            positives = 0
            negatives = 0
            zeros = 0
            positive_ratio = np.nan
            negative_ratio = np.nan
            zero_ratio = np.nan

        p_t, p_w = compute_significance(valid_corrs)

        results.append(
            {
                "window_h": int(window),
                "n_animals": int(valid_corrs.size),
                "mean_corr": mean_corr,
                "median_corr": median_corr,
                "std_corr": std_corr,
                "min_corr": min_corr,
                "max_corr": max_corr,
                "q25_corr": q25_corr,
                "q75_corr": q75_corr,
                "iqr_corr": iqr_corr,
                "abs_mean_corr": abs_mean_corr,
                "abs_median_corr": abs_median_corr,
                "positives": positives,
                "negatives": negatives,
                "zeros": zeros,
                "positive_ratio": positive_ratio,
                "negative_ratio": negative_ratio,
                "zero_ratio": zero_ratio,
                "p_ttest": p_t,
                "p_wilcoxon": p_w,
            }
        )

    return pd.DataFrame(results)


def choose_best_window(
    df_results: pd.DataFrame,
    criterion: Literal["mean_corr", "median_corr"] = "mean_corr",
    *,
    use_absolute: bool = False,
) -> int:
    """
    Escolhe a melhor janela com base em um critério de correlação.

    Parameters
    ----------
    df_results : pd.DataFrame
        Resultado produzido por `run_window_analysis`.
    criterion : {"mean_corr", "median_corr"}, optional
        Métrica usada para escolher a melhor janela.
    use_absolute : bool, optional
        Se True, escolhe a janela com maior magnitude de correlação,
        independentemente do sinal.

    Returns
    -------
    int
        Valor de `window_h` da melhor janela.
    """
    valid_criteria = {"mean_corr", "median_corr"}
    if criterion not in valid_criteria:
        raise ValueError(
            f"Critério inválido: {criterion}. Use um de {sorted(valid_criteria)}"
        )

    required_columns = {"window_h", criterion}
    missing_columns = required_columns.difference(df_results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colunas obrigatórias ausentes em df_results: {missing}")

    temp = df_results.dropna(subset=[criterion]).copy()
    if temp.empty:
        raise ValueError(
            "Não foi possível escolher a melhor janela: resultados válidos ausentes."
        )

    if use_absolute:
        best_idx = temp[criterion].abs().idxmax()
    else:
        best_idx = temp[criterion].idxmax()

    return int(temp.loc[best_idx, "window_h"])

#------------------------------
# PLOTTING
#-----------------------------

def plot_window_results(df_results: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(
        df_results["window_h"],
        df_results["mean_corr"],
        marker="o",
        linewidth=2,
        label="Média"
    )
    ax.plot(
        df_results["window_h"],
        df_results["median_corr"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Mediana"
    )

    ax.set_xlabel("Janela temporal (horas)")
    ax.set_ylabel("Correlação")
    ax.set_title("Influência da escala temporal sobre a resposta de ofegação")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(output_dir / "temporal_scale_clean.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "temporal_scale_clean.pdf", bbox_inches="tight")
    plt.close(fig)
    


def plot_window_results_academic(
    df_results: pd.DataFrame,
    output_dir: str | Path,
    x_tick_interval: int = 3,
) -> None:
    """
    Gera gráfico acadêmico da correlação por janela temporal.

    Args:
        df_results: DataFrame contendo as colunas:
            - window_h
            - mean_corr
            - median_corr
        output_dir: Diretório de saída para salvar as figuras.
        x_tick_interval: Intervalo entre os ticks principais do eixo X.
    """
    required_columns = {"window_h", "mean_corr", "median_corr"}
    missing_columns = required_columns - set(df_results.columns)
    if missing_columns:
        raise ValueError(
            f"DataFrame inválido. Colunas ausentes: {sorted(missing_columns)}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    negative_phase_end = _find_consensus_negative_end(
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

    mean_crossing = _find_zero_crossing(df_plot["window_h"], df_plot["mean_corr"])
    median_crossing = _find_zero_crossing(df_plot["window_h"], df_plot["median_corr"])

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

    mean_max_x, mean_max_y = _find_series_max_point(df_plot, "window_h", "mean_corr")
    median_max_x, median_max_y = _find_series_max_point(df_plot, "window_h", "median_corr")

    ax.scatter(
        mean_max_x,
        mean_max_y,
        s=40,
        color=color_mean,
        zorder=5,
    )
    ax.annotate(
        f"Máx. média\n({mean_max_x:.1f} h, {mean_max_y:.2f})",
        xy=(mean_max_x, mean_max_y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )

    ax.scatter(
        median_max_x,
        median_max_y,
        s=40,
        color=color_median,
        zorder=5,
    )
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
    plt.savefig(output_dir / "temporal_scale_academic.png", dpi=600, bbox_inches="tight")
    plt.savefig(output_dir / "temporal_scale_academic.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

# ==========================================================
# PLOT PSYCHROMETRIC SPACE (SIMPLIFICADO)
# ==========================================================

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
    Gera o gráfico psicrométrico com KDE + contorno + scatter.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com colunas 'temperatura' e 'umidade'.
    output_dir : str | Path
        Diretório de saída.
    kde_sample_size : int
        Número máximo de pontos usados no KDE para acelerar o cálculo.
    kde_levels_fill : int
        Quantidade de níveis do KDE preenchido.
    kde_levels_contour : list[float] | None
        Níveis do contorno. Se None, usa [0.2, 0.4, 0.6, 0.8].
    bw_adjust : float
        Ajuste da largura de banda do KDE.
    scatter_sample_size : int | None
        Se definido, limita o número de pontos do scatter.
    debug_timers : bool
        Se True, imprime tempos por etapa.
    """
    if kde_levels_contour is None:
        kde_levels_contour = [0.2, 0.4, 0.6, 0.8]

    t0 = time.perf_counter()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = ["temperatura", "umidade"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes para plot: {missing}")

    plot_df = df.dropna(subset=["temperatura", "umidade"]).copy()

    if plot_df.empty:
        raise ValueError("Não há dados válidos para gerar o gráfico psicrométrico.")

    if debug_timers:
        t1 = time.perf_counter()
        print(f"[TIMER] preparação inicial: {t1 - t0:.4f} s")

    # ------------------------------------------------------
    # Amostragem para KDE
    # ------------------------------------------------------
    if len(plot_df) > kde_sample_size:
        df_kde = plot_df.sample(n=kde_sample_size, random_state=42)
    else:
        df_kde = plot_df

    # ------------------------------------------------------
    # Amostragem opcional para scatter
    # ------------------------------------------------------
    if scatter_sample_size is not None and len(plot_df) > scatter_sample_size:
        df_scatter = plot_df.sample(n=scatter_sample_size, random_state=42)
    else:
        df_scatter = plot_df

    if debug_timers:
        t2 = time.perf_counter()
        print(f"[TIMER] amostragem: {t2 - t1:.4f} s")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if debug_timers:
        t3 = time.perf_counter()
        print(f"[TIMER] criação figura: {t3 - t2:.4f} s")

    # ------------------------------------------------------
    # KDE preenchido
    # ------------------------------------------------------
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

    if debug_timers:
        t4 = time.perf_counter()
        print(f"[TIMER] kde preenchido: {t4 - t3:.4f} s")

    # ------------------------------------------------------
    # Contorno
    # ------------------------------------------------------
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

    if debug_timers:
        t5 = time.perf_counter()
        print(f"[TIMER] kde contorno: {t5 - t4:.4f} s")

    # ------------------------------------------------------
    # Scatter
    # ------------------------------------------------------
    ax.scatter(
        df_scatter["temperatura"],
        df_scatter["umidade"],
        s=5,
        alpha=0.2,
        color="blue",
        label="Dados de conforto",
    )

    if debug_timers:
        t6 = time.perf_counter()
        print(f"[TIMER] scatter: {t6 - t5:.4f} s")

    # ------------------------------------------------------
    # Labels e layout
    # ------------------------------------------------------
    ax.set_xlabel("Temperatura (°C)")
    ax.set_ylabel("Umidade Relativa (%)")
    ax.set_title("Região empírica de conforto térmico baseada em dados")

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if debug_timers:
        t7 = time.perf_counter()
        print(f"[TIMER] labels/layout: {t7 - t6:.4f} s")

    # ------------------------------------------------------
    # Salvamento
    # ------------------------------------------------------
    plt.savefig(output_dir / "fig_psychrometric_comfort.png", dpi=300, bbox_inches="tight")

    if debug_timers:
        t8 = time.perf_counter()
        print(f"[TIMER] save PNG: {t8 - t7:.4f} s")

    plt.savefig(output_dir / "fig_psychrometric_comfort.pdf", bbox_inches="tight")

    if debug_timers:
        t9 = time.perf_counter()
        print(f"[TIMER] save PDF: {t9 - t8:.4f} s")

    plt.close(fig)

    if debug_timers:
        t10 = time.perf_counter()
        print(f"[TIMER] close fig: {t10 - t9:.4f} s")
        print(f"[TIMER] total plot_psychrometric: {t10 - t0:.4f} s")


# ==========================================================
# DEFINIÇÃO DE CONFORTO
# ==========================================================

def define_comfort(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()
    heat_col = f"heat_load_{window}h"

    if heat_col not in df.columns:
        raise ValueError(f"Coluna {heat_col} não encontrada.")

    df["heat_p25"] = (
        df.groupby("animal_id", observed=False)[heat_col]
        .transform(lambda x: x.dropna().quantile(0.25) if len(x.dropna()) > 0 else np.nan)
    )

    df["pant_p25"] = (
        df.groupby("animal_id", observed=False)["ofegacao"]
        .transform(lambda x: x.dropna().quantile(0.25) if len(x.dropna()) > 0 else np.nan)
    )

    df["comfort_flag"] = (
        (df[heat_col] <= df["heat_p25"]) &
        (df["ofegacao"] <= df["pant_p25"])
    )

    return df


# ==========================================================
# EXTRAÇÃO DE BLOCOS CONTÍNUOS
# ==========================================================
def extract_comfort_periods(
    df: pd.DataFrame,
    min_duration: int = DEFAULT_MIN_DURATION
) -> pd.DataFrame:
    df = df.sort_values(["animal_id", "data_hora"]).copy()

    change = (
        df.groupby("animal_id", observed=False)["comfort_flag"]
        .transform(lambda s: s.ne(s.shift()).fillna(True))
        .astype(int)
    )

    df["block"] = change.groupby(df["animal_id"], observed=False).cumsum()

    block_info = (
        df.groupby(["animal_id", "block"], observed=False)
        .agg(
            comfort_flag_first=("comfort_flag", "first"),
            block_duration_h=("comfort_flag", "size"),
        )
        .reset_index()
    )

    valid_blocks = block_info[
        block_info["comfort_flag_first"].fillna(False)
        & (block_info["block_duration_h"] >= min_duration)
    ][["animal_id", "block", "block_duration_h"]]

    if valid_blocks.empty:
        return pd.DataFrame(columns=list(df.columns) + ["block_duration_h"])

    result = df.merge(valid_blocks, on=["animal_id", "block"], how="inner")
    return result.reset_index(drop=True)


# ==========================================================
# PIPELINE
# ==========================================================

def load_and_prepare_dataset(dataset_path: str | Path, thi_threshold: float) -> pd.DataFrame:
    print("[INFO] Carregando dataset...")
#    df_inmet = read_inmet_csv("/media/extra/wrk/DADOS_CLEO/data/CACOAL.csv")
    df = pd.read_parquet(dataset_path)
    
#    # --------------------------------------------------
#    # 1. limpar dados
#    # --------------------------------------------------
#
#    df_ts = df.dropna(subset=["data_hora"]).copy()
#    df_inmet_clean = df_inmet.dropna(subset=["datetime"]).copy()
#    # --------------------------------------------------
#    # 2. garantir datetime
#    # --------------------------------------------------
#    df_ts["data_hora"] = pd.to_datetime(df_ts["data_hora"])
#    df_inmet_clean["datetime"] = pd.to_datetime(df_inmet_clean["datetime"])
#    print("INMET NaT:", df_inmet["datetime"].isna().sum())
#    # --------------------------------------------------
#    # 3. merge
#    # --------------------------------------------------
#    df = merge_time_series(
#        df_ts,
#        df_inmet_clean,
#        left_time="data_hora",
#        right_time="datetime",
#        direction="backward",
#        tolerance="1h"
#    )
#    df["hora"] = df["data_hora"].dt.floor("H")
#    
#    cols = [
#        "temperatura_do_ar_bulbo_seco_horaria_c",
#        "umidade_relativa_do_ar_horaria",
#        "ofegacao_hora",
#        "ruminacao_hora",
#        "atividade_hora",
#        "ocio_hora",
#    ]
#    
#    df[cols] = df[cols].astype("float64")

    print("[INFO] Padronizando colunas...")
    df = standardize_columns(df)

    print("[INFO] Limpando dados...")
    df = convert_and_clean(df)

    print("[INFO] Calculando Umidade Especifica...")
    df["umidade_especifica"] = calculate_specific_humidity(df["temperatura"], df["umidade"])

    print("[INFO] Calculando THI e excesso térmico...")
    df = add_thi_and_heat_excess(df, thi_threshold=thi_threshold)

    return df


def run_manual_mode_(
    df: pd.DataFrame,
    window: int,
    min_duration: int,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[INFO] Modo manual: usando janela fixa de {window}h")

    df_window = add_heat_load(df, window)
    df_comfort = define_comfort(df_window, window)
    df_periods = extract_comfort_periods(df_comfort, min_duration=min_duration)

    plot_psychrometric(df_periods, output_dir)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_periods.to_csv(output_dir / "dados_conforto_psicrometrico.csv", index=False)

    return df_window, df_periods


def run_manual_mode(
    df: pd.DataFrame,
    window: int,
    min_duration: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    """
    Executa o modo manual com uma janela fixa, ajusta modelos OLS e MixedLM
    e exibe os resultados no final.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    window : int
        Janela fixa em horas para cálculo de carga térmica.
    min_duration : int
        Parâmetro mantido por compatibilidade da assinatura. Não é usado
        diretamente neste trecho.
    output_dir : str | Path
        Diretório de saída. Mantido para integração com o pipeline.

    Returns
    -------
    dict[str, Any]
        Dicionário contendo tabelas, modelos e resumos objetivos.
    """
    print(f"[INFO] Modo manual: usando janela fixa de {window}h")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df_window = add_heat_load(df, window)
    heat_col = f"heat_load_{window}h"

    required = [
        "animal_id",
        "ofegacao",
        heat_col,
        "temperatura",
        "umidade_especifica",
        "atividade",
        "ruminacao",
        "ocio",
    ]

    model_df = prepare_model_data(
        df_window,
        required,
        numeric_columns=[
            "ofegacao",
            "temperatura",
            "umidade_especifica",
            "ruminacao",
            "ocio"

        ],
    )

    print("\n[DEBUG] amostra das variáveis de modelagem:")
    print(model_df[["temperatura", "umidade_especifica", "ruminacao", "ocio"]].head())
    
    vif_table = calculate_vif(
        model_df,
        features=[
            "temperatura",
            "umidade_especifica",
            "ruminacao",
            "ocio"
        ],
    )

    ols_formula = (
        f"ofegacao ~ temperatura + umidade_especifica "
        f"+ ocio"
    )
    ols_model, ols_info = fit_ols_model(model_df, ols_formula)

    mixed_formula = (
        f"ofegacao ~ temperatura + umidade_especifica "
        f"+ ocio"
    )
    mixed_model, mixed_info = fit_mixed_model(
        model_df,
        formula=mixed_formula,
        group_col="animal_id",
    )

    print("\n" + "=" * 80)
    print("VIF DOS PREDITORES")
    print("=" * 80)
    print(vif_table.to_string(index=False))

    print("\n" + "=" * 80)
    print("RESUMO OBJETIVO - OLS")
    print("=" * 80)
    print(ols_info)

    print("\n" + "=" * 80)
    print("RESUMO COMPLETO - OLS")
    print("=" * 80)
    print(ols_model.summary())

    print("\n" + "=" * 80)
    print("RESUMO OBJETIVO - MIXED MODEL")
    print("=" * 80)
    print(mixed_info)

    print("\n" + "=" * 80)
    print("RESUMO COMPLETO - MIXED MODEL")
    print("=" * 80)
    print(mixed_model.summary())

    vif_table.to_csv(output_path / f"vif_window_{window}h.csv", index=False)

    with open(output_path / f"ols_summary_window_{window}h.txt", "w", encoding="utf-8") as f:
        f.write(str(ols_model.summary()))

    with open(output_path / f"mixed_summary_window_{window}h.txt", "w", encoding="utf-8") as f:
        f.write(str(mixed_model.summary()))

    return {
        "model_df": model_df,
        "vif_table": vif_table,
        "ols_model": ols_model,
        "ols_info": ols_info,
        "mixed_model": mixed_model,
        "mixed_info": mixed_info,
    }

def run_auto_mode(
    df: pd.DataFrame,
    windows: list[int],
    criterion: str,
    min_duration: int,
    output_dir: str | Path,
) -> tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[INFO] Modo automático: procurando melhor janela...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results = run_window_analysis(df, windows)
    df_results.to_csv(output_dir / "resultados_janelas.csv", index=False)

    #plot_window_results(df_results, output_dir)
    plot_window_results_academic(df_results, output_dir)
    
    best_window = choose_best_window(df_results, criterion=criterion)
    print(f"[INFO] Melhor janela escolhida: {best_window}h (critério: {criterion})")

    with open(output_dir / "best_window.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_window": best_window,
                "criterion": criterion,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    df_window = add_heat_load(df, best_window)
    df_comfort = define_comfort(df_window, best_window)
    df_periods = extract_comfort_periods(df_comfort, min_duration=min_duration)

    plot_psychrometric(df_periods, output_dir)

    df_periods.to_csv(output_dir / "dados_conforto_psicrometrico.csv", index=False)

    return best_window, df_results, df_window, df_periods


# ==========================================================
# CLI
# ==========================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Análise de carga térmica e extração de períodos de conforto."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Caminho para o arquivo parquet."
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "manual"],
        default="manual",
        help="Modo de execução: auto escolhe a melhor janela; manual usa janela fixa."
    )

    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Janela usada no modo manual."
    )

    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=DEFAULT_WINDOWS,
        help="Lista de janelas testadas no modo auto. Ex.: --windows 3 6 9 12 15 18 21 24"
    )

    parser.add_argument(
        "--criterion",
        choices=["mean_corr", "median_corr"],
        default="mean_corr",
        help="Critério para escolher a melhor janela no modo auto."
    )

    parser.add_argument(
        "--thi-threshold",
        type=float,
        default=DEFAULT_THI_THRESHOLD,
        help="Limiar de THI para cálculo do excesso térmico."
    )

    parser.add_argument(
        "--min-duration",
        type=int,
        default=DEFAULT_MIN_DURATION,
        help="Duração mínima do bloco de conforto em número de registros consecutivos."
    )

    parser.add_argument(
        "--output-dir",
        default="outputs_conforto",
        help="Diretório de saída."
    )

    #--------------------------------------------
    # Profilling flags
    #--------------------------------------------

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Ativa profiling com cProfile."
    )
    
    parser.add_argument(
        "--profile-file",
        default="outputs_conforto/profile.prof",
        help="Arquivo de saída do profiling."
    )
    
    parser.add_argument(
        "--profile-sort",
        default="cumulative",
        choices=["cumulative", "time", "calls"],
        help="Critério de ordenação do profiling."
    )
    
    parser.add_argument(
        "--profile-lines",
        type=int,
        default=30,
        help="Quantidade de linhas mostradas no resumo do profiling."
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    df = load_and_prepare_dataset(
        dataset_path=args.dataset,
        thi_threshold=args.thi_threshold,
    )

    if args.mode == "manual":
        manual_result = run_manual_mode(
            df=df,
            window=args.window,
            min_duration=args.min_duration,
            output_dir=args.output_dir,
        )
    
        model_df = manual_result["model_df"]
        vif_table = manual_result["vif_table"]
        ols_info = manual_result["ols_info"]
        mixed_info = manual_result["mixed_info"]
    
        print(f"[INFO] Registros usados na modelagem: {len(model_df):,}")
        print(f"[INFO] Número de preditores avaliados no VIF: {len(vif_table):,}")
        print(f"[INFO] OLS ajustado com {ols_info.n_obs:,} observações")
        print(f"[INFO] MixedLM ajustado com {mixed_info.n_obs:,} observações")
    elif args.mode == "auto":
        best_window, df_results, _, df_periods = run_auto_mode(
            df=df,
            windows=args.windows,
            criterion=args.criterion,
            min_duration=args.min_duration,
            output_dir=args.output_dir,
        )
        print(f"[INFO] Melhor janela: {best_window}h")
        print("[INFO] Resultados das janelas:")
        print(df_results)
        print(f"[INFO] Registros de conforto extraídos: {len(df_periods):,}")

    print("[INFO] Processo finalizado com sucesso.")

    
if __name__ == "__main__":
    main()
