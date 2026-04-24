from __future__ import annotations

import pandas as pd
import numpy as np


# ==========================================================
# CONFIG
# ==========================================================
THI_THRESHOLD = 72
WINDOW = 15
MIN_DURATION = 3  # horas mínimas de conforto


# ==========================================================
# PADRONIZAÇÃO
# ==========================================================
def standardize_columns(df):
    mapping = {
        "temperatura_compost_1": "temperatura",
        "humidade_compost_1": "umidade",
        "ofegacao_hora": "ofegacao",
    }
    return df.rename(columns=mapping)

def convert_and_clean(df):
    # Conversão e limpeza
    df["temperatura"] = pd.to_numeric(df["temperatura"], errors="coerce")
    df["umidade"] = pd.to_numeric(df["umidade"], errors="coerce")
    df = df.dropna(subset=["temperatura", "umidade", "ofegacao"])
    
    return df.sort_values(["animal_id", "data_hora"]).reset_index(drop=True)

# ==========================================================
# THI
# ==========================================================
def calculate_thi(temp, rh):
    """Calcula o Temperature-Humidity Index (THI)"""
    return (1.8 * temp + 32) - (0.55 - 0.0055 * rh) * (1.8 * temp - 26)


# ==========================================================
# CARGA TERMICA 15h
# ==========================================================
def calculate_heat_load(df):

    df = df.copy()

    print("[INFO] Calculando THI...")
    df["thi"] = calculate_thi(df["temperatura"], df["umidade"])
    df["heat_excess"] = np.maximum(0, df["thi"] - THI_THRESHOLD)

    df["heat_load_15h"] = (
        df.groupby("animal_id", observed=False)["heat_excess"]
        .transform(lambda x: x.rolling(WINDOW, min_periods=1).sum())
    )

    return df


# ==========================================================
# CRITÉRIO DE CONFORTO
# ==========================================================
def define_comfort(df):

    df = df.copy()

    # limiar por animal (baixo calor)
    df["heat_p25"] = (
        df.groupby("animal_id", observed=False)["heat_load_15h"]
        .transform(lambda x: x.dropna().quantile(0.25) if len(x.dropna()) > 0 else np.nan)
    )

    # limiar de baixa ofegação
    df["pant_p25"] = (
        df.groupby("animal_id", observed=False)["ofegacao"]
        .transform(lambda x: x.dropna().quantile(0.25) if len(x.dropna()) > 0 else np.nan)
    )

    df["comfort_flag"] = (
        (df["heat_load_15h"] <= df["heat_p25"]) &
        (df["ofegacao"] <= df["pant_p25"])
    )

    return df


# ==========================================================
# DETECÇÃO DE PERÍODOS CONTÍNUOS
# ==========================================================
def extract_comfort_periods(df):

    periods = []

    for animal_id, g in df.groupby("animal_id", observed=False):

        g = g.sort_values("data_hora").copy()

        change = (
            (g["comfort_flag"] != g["comfort_flag"].shift())
            .fillna(True)
            .astype(int)
        )

        g["block"] = change.cumsum()

        for _, block in g.groupby("block"):

            flag = block["comfort_flag"].iloc[0]
            
            if pd.isna(flag) or (flag is False):
                continue
            duration = len(block)

            if duration >= MIN_DURATION:
                periods.append(block)

    return pd.concat(periods, ignore_index=True)

# ==========================================================
# MAIN
# ==========================================================
def main():

    dataset = "/media/extra/wrk/DADOS_CLEO/outputs/dataset_final_1293.parquet"

    print("[INFO] Carregando dataset...")
    df = pd.read_parquet(dataset)

    print("[INFO] Padronizando...")
    df = standardize_columns(df)

    print("[INFO] Calculando carga térmica...")
    df = calculate_heat_load(df)

    print("[INFO] Definindo conforto...")
    df = define_comfort(df)

    print("[INFO] Extraindo períodos de conforto...")
    df_comfort = extract_comfort_periods(df)

    print(f"[INFO] Registros de conforto: {len(df_comfort):,}")

    # salvar para psicrométrico
    df_comfort.to_csv("dados_conforto_psicrometrico.csv", index=False)

    print("[INFO] Dataset salvo para análise psicrométrica.")


if __name__ == "__main__":
    main()
