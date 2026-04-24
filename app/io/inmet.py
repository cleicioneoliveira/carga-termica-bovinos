from __future__ import annotations

import pandas as pd


def read_inmet_csv(path: str) -> pd.DataFrame:
    """
    Read INMET-style CSV with metadata header.

    Automatically detects header, handles encoding and decimal format,
    and standardizes column names.

    Parameters
    ----------
    path : str
        Path to INMET CSV file.

    Returns
    -------
    pd.DataFrame
        Parsed dataset with a `datetime` column.
    """

    # --------------------------------------------------
    # Detect header
    # --------------------------------------------------
    with open(path, "r", encoding="latin1") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "Data" in line and "Hora" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Header row not found in file: {path}")

    # --------------------------------------------------
    # Read CSV
    # --------------------------------------------------
    df = pd.read_csv(
        path,
        sep=";",
        skiprows=header_idx,
        decimal=",",
        encoding="latin1",
    )

    # --------------------------------------------------
    # Normalize columns
    # --------------------------------------------------
    df.columns = (
        df.columns
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^unnamed")]

    # --------------------------------------------------
    # Build datetime (ROBUSTO)
    # --------------------------------------------------
    if {"data", "hora_utc"}.issubset(df.columns):

        hora = (
            df["hora_utc"]
            .astype(str)
            .str.strip()
            .str.replace(r"utc", "", case=False, regex=True)
            .str.strip()
            .str.zfill(4)
        )

        df["datetime"] = pd.to_datetime(
            df["data"].astype(str).str.strip() + " " + hora,
            format="%Y/%m/%d %H%M",
            errors="coerce",
        )

    return df

