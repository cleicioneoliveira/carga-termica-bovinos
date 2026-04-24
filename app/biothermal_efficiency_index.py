#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
biothermal_efficiency_index.py

Pipeline para análise de carga térmica acumulada, resposta de ofegação e
índices derivados de eficiência termorreguladora em bovinos leiteiros.

Objetivos principais
--------------------
1. Carregar e padronizar a base de dados ambiental e comportamental.
2. Calcular THI, excesso térmico e CTA acumulada em janela móvel de 15 horas.
3. Estimar o IOR empírico:
      IOR = ofegacao / CTA
4. Construir a tendência empírica do IOR por faixas de CTA.
5. Usar essa tendência para estimar a ofegação esperada e o score de desconforto.
6. Ajustar um modelo polinomial de saturação fisiológica da ofegação em função da CTA.
7. Gerar tabelas, figuras e base final consolidada.

Saídas
------
Os produtos são gravados no diretório OUTPUT_DIR:
- Figuras em PNG
- Tabelas em CSV
- Base final em Parquet
- Resumo de métricas em CSV

Observações
-----------
- A coluna categórica gerada por pd.cut() não é salva diretamente no parquet,
  pois pode causar incompatibilidade com pyarrow em alguns ambientes.
- Por isso, colunas categóricas/intervalares são convertidas para string
  ou removidas antes da exportação final.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# ==========================================================
# 1. CONFIGURAÇÕES GERAIS
# ==========================================================
THI_THRESHOLD = 72
WINDOW_H = 15
CTA_MIN = 2.0
EPS = 0.1

OUTPUT_DIR = Path("resultados_dissertacao")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATH_DATA = "/media/extra/wrk/DADOS_CLEO/outputs/dataset_final_1293.parquet"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


# ==========================================================
# 2. FUNÇÕES AUXILIARES
# ==========================================================
def calculate_thi(temp: pd.Series | np.ndarray, rh: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """
    Calcula o Temperature Humidity Index (THI).

    Parâmetros
    ----------
    temp : array-like
        Temperatura do ar em graus Celsius.
    rh : array-like
        Umidade relativa em porcentagem.

    Retorna
    -------
    array-like
        Valor do THI.
    """
    return (1.8 * temp + 32) - (0.55 - 0.0055 * rh) * (1.8 * temp - 26)


def save_figure(filename: str, dpi: int = 300) -> None:
    """
    Salva a figura atual no diretório de saída e encerra o canvas.

    Parâmetros
    ----------
    filename : str
        Nome do arquivo da figura.
    dpi : int, optional
        Resolução da figura, por padrão 300.
    """
    filepath = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Figura salva em: {filepath}")


def export_table(df: pd.DataFrame, filename: str) -> None:
    """
    Exporta um DataFrame para CSV no diretório de saída.

    Parâmetros
    ----------
    df : pd.DataFrame
        Tabela a ser exportada.
    filename : str
        Nome do arquivo CSV.
    """
    filepath = OUTPUT_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"[INFO] Tabela salva em: {filepath}")


def prepare_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara o DataFrame para exportação segura em parquet.

    Estratégias:
    - remove coluna intervalar 'faixa_cta', se existir;
    - converte colunas categóricas para string.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser preparado.

    Retorna
    -------
    pd.DataFrame
        Cópia do DataFrame pronta para exportação.
    """
    df_to_save = df.drop(columns=["faixa_cta"], errors="ignore").copy()

    for col in df_to_save.columns:
        if isinstance(df_to_save[col].dtype, pd.CategoricalDtype):
            df_to_save[col] = df_to_save[col].astype(str)

    return df_to_save


# ==========================================================
# 3. PRÉ PROCESSAMENTO
# ==========================================================
def pipeline_pre_processamento(path_parquet: str | Path) -> pd.DataFrame:
    """
    Carrega a base, padroniza nomes de colunas, calcula THI, excesso térmico
    e CTA acumulada por animal.

    Parâmetros
    ----------
    path_parquet : str or Path
        Caminho do arquivo parquet de entrada.

    Retorna
    -------
    pd.DataFrame
        Base filtrada para CTA > CTA_MIN.
    """
    print("[1/6] Carregando e padronizando dados...")
    df = pd.read_parquet(path_parquet).copy()

    mapping = {
        "temperatura_compost_1": "temp",
        "humidade_compost_1": "ur",
        "ofegacao_hora": "ofegacao",
    }
    df = df.rename(columns=mapping)

    required_cols = ["animal_id", "data_hora", "temp", "ur", "ofegacao"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    df["data_hora"] = pd.to_datetime(df["data_hora"])
    df = (
        df.dropna(subset=["temp", "ur", "ofegacao"])
        .sort_values(["animal_id", "data_hora"])
        .copy()
    )

    df["thi"] = calculate_thi(df["temp"], df["ur"])
    df["excess"] = np.maximum(0, df["thi"] - THI_THRESHOLD)

    print(f"[2/6] Calculando CTA acumulada em {WINDOW_H} h...")
    df["cta"] = (
        df.groupby("animal_id", observed=False)["excess"]
        .transform(lambda x: x.rolling(window=WINDOW_H, min_periods=1).sum())
    )

    df_filtrado = df[df["cta"] > CTA_MIN].copy()
    print(f"[INFO] Registros após filtro CTA > {CTA_MIN}: {len(df_filtrado)}")

    return df_filtrado


# ==========================================================
# 4. ANÁLISE EMPÍRICA DO IOR
# ==========================================================
def calcular_ior_e_tendencia(df: pd.DataFrame, bins: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    """
    Calcula o IOR empírico e sua tendência média por faixas de CTA.

    IOR = ofegacao / CTA

    Parâmetros
    ----------
    df : pd.DataFrame
        Base já filtrada para CTA > CTA_MIN.
    bins : int, optional
        Número de faixas de CTA para análise da tendência.

    Retorna
    -------
    tuple[pd.DataFrame, pd.Series]
        DataFrame com colunas adicionais e série com tendência média do IOR.
    """
    print("[3/6] Calculando IOR e tendência empírica...")
    df = df.copy()

    df["ior"] = df["ofegacao"] / df["cta"].replace(0, np.nan)
    df["ior"] = df["ior"].replace([np.inf, -np.inf], np.nan).fillna(0)

    print("--- Estatísticas do IOR (CTA > 2.0) ---")
    print(df["ior"].describe())

    df["faixa_cta"] = pd.cut(df["cta"], bins=bins)
    analise_tendencia = (
        df.groupby("faixa_cta", observed=True)["ior"]
        .mean()
        .dropna()
    )

    df["faixa_cta_label"] = df["faixa_cta"].astype(str)

    tabela_tendencia = pd.DataFrame(
        {
            "faixa_cta": [str(interval) for interval in analise_tendencia.index],
            "cta_mid": [interval.mid for interval in analise_tendencia.index],
            "ior_medio": analise_tendencia.values,
        }
    )
    export_table(tabela_tendencia, "tabela_01_tendencia_ior_empirico.csv")

    return df, analise_tendencia


def plot_tendencia_ior(analise_tendencia: pd.Series) -> None:
    """
    Plota a tendência média do IOR em função da CTA.

    Parâmetros
    ----------
    analise_tendencia : pd.Series
        Série indexada por intervalos de CTA contendo o IOR médio.
    """
    intervalos = analise_tendencia.index
    x_mid = [interval.mid for interval in intervalos]
    x_left = [interval.left for interval in intervalos]
    x_right = [interval.right for interval in intervalos]
    y = analise_tendencia.values

    plt.figure(figsize=(8, 5))
    plt.plot(x_mid, y, marker="o", linewidth=2)

    for xl, xr, yy in zip(x_left, x_right, y):
        plt.hlines(yy, xl, xr, alpha=0.4)

    plt.xlabel(f"CTA acumulada em {WINDOW_H} h")
    plt.ylabel("IOR médio = ofegação / CTA")
    plt.title("Tendência do IOR em função da CTA")
    plt.grid(True, alpha=0.3)

    save_figure("figura_01_tendencia_ior_por_cta.png")


def plot_tendencia_com_dados(df: pd.DataFrame, analise_tendencia: pd.Series) -> None:
    """
    Plota os valores individuais do IOR e a tendência média por faixa.

    Parâmetros
    ----------
    df : pd.DataFrame
        Base com colunas 'cta' e 'ior'.
    analise_tendencia : pd.Series
        Série com média do IOR por faixa de CTA.
    """
    x_mid = [interval.mid for interval in analise_tendencia.index]
    y_mean = analise_tendencia.values

    plt.figure(figsize=(9, 5))
    plt.scatter(df["cta"], df["ior"], alpha=0.15, s=10, label="Dados individuais")
    plt.plot(x_mid, y_mean, marker="o", color='red', linewidth=2, label="Média por faixa")

    plt.xlabel(f"CTA acumulada em {WINDOW_H} h")
    plt.ylabel("IOR = ofegação / CTA")
    plt.title("IOR individual e tendência média por faixa de CTA")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    save_figure("figura_02_tendencia_ior_com_dados_brutos.png")


def plot_ruptura_ior_vs_cta(df: pd.DataFrame) -> None:
    """
    Plota uma curva LOWESS aproximando a ruptura/decadência do IOR com o aumento da CTA.

    Parâmetros
    ----------
    df : pd.DataFrame
        Base com colunas 'cta_15h' e 'ior'.
    """
    plt.figure(figsize=(12, 6))
    amostra = df.sample(min(len(df), 10000), random_state=42)

    sns.regplot(
        data=amostra,
        x="cta",
        y="ior",
        lowess=True,
        scatter_kws={"alpha": 0.1},
        line_kws={"color": "red"},
    )

    plt.title("Evolução da eficiência de resfriamento (IOR) vs carga acumulada")
    plt.xlabel(f"Carga térmica acumulada (últimas {WINDOW_H} h)")
    plt.ylabel("IOR (minutos de ofegação / CTA)")
    plt.grid(True, alpha=0.3)

    save_figure("figura_03_ruptura_ior_vs_cta_lowess.png")


# ==========================================================
# 5. CLASSIFICAÇÃO OBJETIVA VIA TENDÊNCIA EMPÍRICA
# ==========================================================
def criar_funcao_ior_esperado(analise_tendencia: pd.Series):
    """
    Cria uma função interpolada de IOR esperado em função da CTA.

    Parâmetros
    ----------
    analise_tendencia : pd.Series
        Série com média do IOR por faixas de CTA.

    Retorna
    -------
    callable
        Função ior_esperado(cta).
    """
    x = np.array([interval.mid for interval in analise_tendencia.index], dtype=float)
    y = np.array(analise_tendencia.values, dtype=float)

    def ior_esperado(cta: float) -> float:
        return np.interp(cta, x, y, left=y[0], right=y[-1])

    return ior_esperado


def classificar_desconforto(row: pd.Series, func_ior_esperado, eps: float = EPS) -> float:
    """
    Calcula o score de desconforto para uma linha do DataFrame.

    Score = ofegacao_observada / ofegacao_esperada

    onde:
        ofegacao_esperada = CTA * IOR_esperado(CTA)

    Parâmetros
    ----------
    row : pd.Series
        Linha da base.
    func_ior_esperado : callable
        Função interpolada do IOR esperado.
    eps : float, optional
        Pequena constante para evitar divisão por zero.

    Retorna
    -------
    float
        Score de desconforto.
    """
    cta = row["cta"]
    ofeg = row["ofegacao"]

    ior_ref = func_ior_esperado(cta)
    esperado = cta * ior_ref
    score = ofeg / (esperado + eps)

    return score


def calcular_score_desconforto(df: pd.DataFrame, analise_tendencia: pd.Series, eps: float = EPS) -> pd.DataFrame:
    """
    Calcula de forma vetorizada:
    - IOR esperado
    - ofegação esperada
    - score de desconforto

    Parâmetros
    ----------
    df : pd.DataFrame
        Base com CTA e ofegação.
    analise_tendencia : pd.Series
        Série com média do IOR por faixas de CTA.
    eps : float, optional
        Pequena constante para evitar divisão por zero.

    Retorna
    -------
    pd.DataFrame
        Base com colunas adicionais.
    """
    df = df.copy()

    x = np.array([interval.mid for interval in analise_tendencia.index], dtype=float)
    y = np.array(analise_tendencia.values, dtype=float)

    cta = df["cta"].to_numpy(dtype=float)
    ofeg = df["ofegacao"].to_numpy(dtype=float)

    ior_ref = np.interp(cta, x, y, left=y[0], right=y[-1])
    esperado = cta * ior_ref
    score = ofeg / (esperado + eps)

    df["ior_esperado"] = ior_ref
    df["ofegacao_esperada_tendencia"] = esperado
    df["score_desconforto"] = score

    return df


def rotular_score_desconforto(score: float) -> str:
    """
    Rotula o score de desconforto em classes interpretativas.

    Parâmetros
    ----------
    score : float
        Score calculado.

    Retorna
    -------
    str
        Classe interpretativa.
    """
    if score < 0.8:
        return "abaixo_do_esperado"
    if score <= 1.2:
        return "dentro_do_esperado"
    return "acima_do_esperado"


# ==========================================================
# 6. MODELAGEM FISIOLÓGICA DA RESPOSTA ESPERADA
# ==========================================================
def modelar_eficiencia_biologica(df: pd.DataFrame) -> tuple[pd.DataFrame, LinearRegression, PolynomialFeatures]:
    """
    Ajusta um modelo polinomial de grau 2 para representar a saturação
    da resposta de ofegação em função da CTA.

    Parâmetros
    ----------
    df : pd.DataFrame
        Base com colunas 'cta' e 'ofegacao'.

    Retorna
    -------
    tuple[pd.DataFrame, LinearRegression, PolynomialFeatures]
        Base enriquecida, modelo ajustado e transformador polinomial.
    """
    print("[4/6] Modelando resposta esperada por regressão polinomial...")
    df = df.copy()

    X = df[["cta"]].values
    y = df["ofegacao"].values

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    modelo = LinearRegression()
    modelo.fit(X_poly, y)

    df["ofegacao_pred"] = modelo.predict(X_poly)
    df.loc[df["ofegacao_pred"] < 0, "ofegacao_pred"] = 0

    df["ior_corrigido"] = df["ofegacao"] / (df["ofegacao_pred"] + 1e-6)

    return df, modelo, poly


# ==========================================================
# 7. TABELAS E FIGURAS PRINCIPAIS
# ==========================================================
def gerar_visualizacoes(df: pd.DataFrame, modelo: LinearRegression, poly: PolynomialFeatures) -> None:
    """
    Gera tabelas e gráficos principais da análise.

    Parâmetros
    ----------
    df : pd.DataFrame
        Base final enriquecida.
    modelo : LinearRegression
        Modelo fisiológico ajustado.
    poly : PolynomialFeatures
        Transformador polinomial.
    """
    print("[5/6] Gerando tabelas e gráficos principais...")

    df_tmp = df.copy()
    df_tmp["faixa_cta"] = pd.cut(df_tmp["cta"], bins=10)

    tabela_tendencia = (
        df_tmp.groupby("faixa_cta", observed=True)
        .agg(
            ofegacao_media=("ofegacao", "mean"),
            ior_corrigido_medio=("ior_corrigido", "mean"),
            n_obs=("animal_id", "count"),
        )
        .reset_index()
    )
    tabela_tendencia["faixa_cta"] = tabela_tendencia["faixa_cta"].astype(str)
    export_table(tabela_tendencia, "tabela_02_tendencia_ior_corrigido.csv")

    plt.figure(figsize=(10, 6))
    df_sample = df.sample(n=min(len(df), 20000), random_state=42).sort_values("cta")
    y_pred_sample = modelo.predict(poly.transform(df_sample[["cta"]].values))
    y_pred_sample = np.maximum(y_pred_sample, 0)

    plt.scatter(
        df_sample["cta"],
        df_sample["ofegacao"],
        alpha=0.1,
        color="gray",
        label="Observações individuais",
    )
    plt.plot(
        df_sample["cta"],
        y_pred_sample,
        color="red",
        linewidth=3,
        label="Ofegação esperada",
    )

    plt.title("Fadiga térmica: saturação da resposta de ofegação")
    plt.xlabel(f"Carga térmica acumulada (últimas {WINDOW_H} h)")
    plt.ylabel("Minutos de ofegação por hora")
    plt.legend(loc="upper left")

    save_figure("figura_04_modelo_saturacao_ofegacao_vs_cta.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["ior_corrigido"], bins=50, kde=True)
    plt.axvline(1.0, color="red", linestyle="--", label="Eficiência média (1.0)")
    plt.title("Distribuição do IOR corrigido")
    plt.xlabel("IOR corrigido")
    plt.ylabel("Frequência")
    plt.xlim(0, 3)
    plt.legend(loc="upper right")

    save_figure("figura_05_distribuicao_ior_corrigido.png")

def gerar_figuras_dissertacao(df: pd.DataFrame) -> None:
    """
    Gera figuras síntese voltadas à interpretação da dissertação.

    Parâmetros
    ----------
    df : pd.DataFrame
        Base enriquecida com IOR corrigido.
    """
    print("[6/6] Gerando figuras finais da dissertação...")

    df_box = df.copy()
    df_box["faixas_resumo"] = pd.cut(
        df_box["cta"],
        bins=[0, 22.4, 83.6, 165.2, 210],
        labels=["Conforto", "Alerta", "Crítico", "Fadiga"],
        include_lowest=True,
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="faixas_resumo",
        y="ior_corrigido",
        data=df_box,
        showfliers=False,
    )
    plt.axhline(1.0, color="black", linestyle=":", alpha=0.5)
    plt.title("Eficiência termorreguladora por estágio de carga acumulada")
    plt.xlabel("Estágio de estresse térmico baseado em CTA")
    plt.ylabel("IOR corrigido")

    save_figure("figura_06_boxplot_ior_corrigido_por_estagio_cta.png")

    t_range = np.linspace(20, 45, 200)
    ur_range = np.linspace(20, 100, 200)
    T, UR = np.meshgrid(t_range, ur_range)
    Z_THI = calculate_thi(T, UR)
    excesso = np.maximum(0, Z_THI - THI_THRESHOLD)

    plt.figure(figsize=(10, 8))
    levels = [0, 1.5, 5.5, 11.0, 25]
    colors = ["#a1d99b", "#fdbb84", "#fc8d59", "#d7301f"]

    plt.contourf(T, UR, excesso, levels=levels, colors=colors, alpha=0.8)
    contours = plt.contour(
        T,
        UR,
        Z_THI,
        levels=[72, 75, 78, 82],
        colors="black",
        linestyles="--",
        alpha=0.4,
    )
    plt.clabel(contours, inline=True, fontsize=9, fmt="THI %.0f")

    plt.title("Diagrama psicrométrico de risco baseado em saturação fisiológica")
    plt.xlabel("Temperatura (°C)")
    plt.ylabel("Umidade relativa (%)")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#a1d99b", label="Zona verde: recuperação / eficiência alta"),
        Patch(facecolor="#fdbb84", label="Alerta: início do acúmulo"),
        Patch(facecolor="#fc8d59", label="Crítico: risco de saturação em 12-15 h"),
        Patch(facecolor="#d7301f", label="Fadiga térmica: colapso da resposta"),
    ]
    plt.legend(handles=legend_elements, loc="upper right", title="Status biológico")

    save_figure("figura_07_diagrama_psicrometrico_risco_termico.png")


# ==========================================================
# 8. EXECUÇÃO PRINCIPAL
# ==========================================================
def main() -> None:
    """
    Executa a pipeline completa:
    - pré-processamento
    - cálculo do IOR e tendência
    - score de desconforto
    - modelagem fisiológica
    - geração de tabelas e figuras
    - exportação da base final
    """
    df = pipeline_pre_processamento(PATH_DATA)

    df, analise_tendencia = calcular_ior_e_tendencia(df)
    plot_tendencia_ior(analise_tendencia)
    plot_tendencia_com_dados(df, analise_tendencia)
    plot_ruptura_ior_vs_cta(df)

    func_ior_esperado = criar_funcao_ior_esperado(analise_tendencia)
    df["score_desconforto_apply"] = df.apply(
        lambda row: classificar_desconforto(row, func_ior_esperado),
        axis=1,
    )

    df = calcular_score_desconforto(df, analise_tendencia)
    df["classe_score_desconforto"] = df["score_desconforto"].apply(rotular_score_desconforto)

    df, modelo_fisiologico, poly = modelar_eficiencia_biologica(df)

    gerar_visualizacoes(df, modelo_fisiologico, poly)
    gerar_figuras_dissertacao(df)

    df_to_save = prepare_dataframe_for_parquet(df)
    parquet_path = OUTPUT_DIR / "base_final_analise_termica.parquet"
    df_to_save.to_parquet(parquet_path, index=False)
    print(f"[INFO] Base final salva em: {parquet_path}")

    resumo = pd.DataFrame(
        {
            "n_registros": [len(df)],
            "cta_media": [df["cta"].mean()],
            "ior_medio": [df["ior"].mean()],
            "ior_corrigido_medio": [df["ior_corrigido"].mean()],
            "score_desconforto_medio": [df["score_desconforto"].mean()],
        }
    )
    export_table(resumo, "tabela_03_resumo_metricas.csv")

    print(f"\n[SUCESSO] Análise finalizada. Arquivos em: {OUTPUT_DIR}")
    print(f"[INFO] Média geral do IOR: {df['ior'].mean():.4f}")
    print(f"[INFO] Média geral do IOR corrigido: {df['ior_corrigido'].mean():.4f}")
    print(f"[INFO] Média do score de desconforto: {df['score_desconforto'].mean():.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERRO] Falha na execução: {e}")
        raise
