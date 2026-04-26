from __future__ import annotations

from enum import StrEnum


class Column(StrEnum):
    """Canonical column names used internally by the thermal-comfort pipeline."""

    # Core keys
    ANIMAL_ID = "brinco"
    DATA_HORA = "data_hora"

    # Final integrated health status
    STATUS_SAUDE = "status_saude"
    STATUS_VIGENTE = "status_vigente"

    # Operational thermal/behavioral variables
    TEMPERATURA = "temperatura"
    UMIDADE = "umidade"
    OFEGACAO = "ofegacao"
    THI = "thi"
    HEAT_EXCESS = "heat_excess"

    # Comfort-threshold variables
    HEAT_P25 = "heat_p25"
    PANT_P25 = "pant_p25"
    COMFORT_FLAG = "comfort_flag"
    BLOCK = "block"
    BLOCK_DURATION_H = "block_duration_h"


class SourceColumn(StrEnum):
    """Known source columns produced by upstream repositories or raw files."""

    # Compost/environment-correction outputs
    TEMPERATURA_COMPOST_1 = "temperatura_compost_1"
    HUMIDADE_COMPOST_1 = "humidade_compost_1"
    THI_COMPOST_1 = "thi_compost1"
    TEMPERATURA_COMPOST_2 = "temperatura_compost_2"
    HUMIDADE_COMPOST_2 = "humidade_compost_2"
    THI_COMPOST_2 = "thi_compost2"

    # Backward-compatible aliases used by older code
    TEMPERATURA_COMPOST = "temperatura_compost_1"
    HUMIDADE_COMPOST = "humidade_compost_1"

    # Behavioral monitoring outputs
    OFEGACAO_HORA = "ofegacao_hora"
    RUMINACAO_HORA = "ruminacao_hora"
    ATIVIDADE_HORA = "atividade_hora"
    OCIO_HORA = "ocio_hora"

    # INMET/raw environmental alternatives
    TEMPERATURA_INMET = "temperatura_do_ar_bulbo_seco_horaria_c"
    UMIDADE_INMET = "umidade_relativa_do_ar_horaria"

    # Generic aliases sometimes found in intermediate files
    ANIMAL_ID = "animal_id"
    TIMESTAMP = "timestamp"
    STATUS_VIGENTE = "status_vigente"


class FinalDatasetColumn(StrEnum):
    """Recommended stable columns for the integrated dataset."""

    ANIMAL_ID = "brinco"
    DATA_HORA = "data_hora"
    STATUS_SAUDE = "status_saude"
    OFEGACAO_HORA = "ofegacao_hora"
    TEMPERATURA_COMPOST_1 = "temperatura_compost_1"
    HUMIDADE_COMPOST_1 = "humidade_compost_1"
    THI_COMPOST_1 = "thi_compost1"
    TEMPERATURA_COMPOST_2 = "temperatura_compost_2"
    HUMIDADE_COMPOST_2 = "humidade_compost_2"
    THI_COMPOST_2 = "thi_compost2"


# Source-to-operational mapping used by the thermal analysis.
STANDARDIZATION_MAP: dict[str, str] = {
    SourceColumn.TEMPERATURA_COMPOST_1: Column.TEMPERATURA,
    SourceColumn.HUMIDADE_COMPOST_1: Column.UMIDADE,
    SourceColumn.OFEGACAO_HORA: Column.OFEGACAO,
    SourceColumn.STATUS_VIGENTE: Column.STATUS_SAUDE,
}


# Broader alias map for optional pre-standardization of integrated datasets.
COLUMN_ALIASES: dict[str, str] = {
    SourceColumn.ANIMAL_ID: Column.ANIMAL_ID,
    SourceColumn.TIMESTAMP: Column.DATA_HORA,
    SourceColumn.STATUS_VIGENTE: Column.STATUS_SAUDE,
    "status": Column.STATUS_SAUDE,
    "status_atual": Column.STATUS_SAUDE,
    "id_animal": Column.ANIMAL_ID,
}


REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    Column.ANIMAL_ID,
    Column.DATA_HORA,
    Column.TEMPERATURA,
    Column.UMIDADE,
    Column.OFEGACAO,
)


RECOMMENDED_FINAL_COLUMNS: tuple[str, ...] = tuple(item.value for item in FinalDatasetColumn)
