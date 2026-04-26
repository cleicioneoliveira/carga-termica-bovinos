from __future__ import annotations

from enum import StrEnum


class Column(StrEnum):
    """Canonical column names used internally by the thermal pipeline."""

    ANIMAL_ID = "brinco"
    DATA_HORA = "data_hora"
    STATUS_SAUDE = "status_saude"

    TEMPERATURA = "temperatura"
    UMIDADE = "umidade"
    OFEGACAO = "ofegacao"
    THI = "thi"

    HEAT_EXCESS = "heat_excess"
    HEAT_P25 = "heat_p25"
    PANT_P25 = "pant_p25"
    COMFORT_FLAG = "comfort_flag"
    BLOCK = "block"
    BLOCK_DURATION_H = "block_duration_h"


class SourceColumn(StrEnum):
    """Known source column names produced by upstream repositories."""

    # Common keys
    ANIMAL_ID = "brinco"
    DATA_HORA = "data_hora"

    # Health timeline / merged dataset
    STATUS_SAUDE = "status_saude"
    STATUS_VIGENTE = "status_vigente"

    # Monitoring behavior
    OFEGACAO_HORA = "ofegacao_hora"

    # Corrected compost environment, barn/compost 1
    TEMPERATURA_COMPOST_1 = "temperatura_compost_1"
    HUMIDADE_COMPOST_1 = "humidade_compost_1"
    UMIDADE_COMPOST_1 = "umidade_compost_1"
    THI_COMPOST_1 = "thi_compost1"

    # Corrected compost environment, barn/compost 2
    TEMPERATURA_COMPOST_2 = "temperatura_compost_2"
    HUMIDADE_COMPOST_2 = "humidade_compost_2"
    UMIDADE_COMPOST_2 = "umidade_compost_2"
    THI_COMPOST_2 = "thi_compost2"

    # INMET-like external source columns
    TEMPERATURA_INMET = "temperatura_do_ar_bulbo_seco_horaria_c"
    UMIDADE_INMET = "umidade_relativa_do_ar_horaria"


class OutputColumn(StrEnum):
    """Columns expected in the final merged dataset consumed by this package."""

    ANIMAL_ID = Column.ANIMAL_ID
    DATA_HORA = Column.DATA_HORA
    STATUS_SAUDE = Column.STATUS_SAUDE
    OFEGACAO_HORA = SourceColumn.OFEGACAO_HORA
    TEMPERATURA_COMPOST_1 = SourceColumn.TEMPERATURA_COMPOST_1
    HUMIDADE_COMPOST_1 = SourceColumn.HUMIDADE_COMPOST_1
    THI_COMPOST_1 = SourceColumn.THI_COMPOST_1
    TEMPERATURA_COMPOST_2 = SourceColumn.TEMPERATURA_COMPOST_2
    HUMIDADE_COMPOST_2 = SourceColumn.HUMIDADE_COMPOST_2
    THI_COMPOST_2 = SourceColumn.THI_COMPOST_2


STANDARDIZATION_MAP: dict[str, str] = {
    # Main source currently used for the thermal analysis.
    SourceColumn.TEMPERATURA_COMPOST_1: Column.TEMPERATURA,
    SourceColumn.HUMIDADE_COMPOST_1: Column.UMIDADE,
    SourceColumn.UMIDADE_COMPOST_1: Column.UMIDADE,
    SourceColumn.OFEGACAO_HORA: Column.OFEGACAO,

    # Health status compatibility.
    SourceColumn.STATUS_VIGENTE: Column.STATUS_SAUDE,

    # External meteorological sources.
    SourceColumn.TEMPERATURA_INMET: Column.TEMPERATURA,
    SourceColumn.UMIDADE_INMET: Column.UMIDADE,
}


REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    Column.ANIMAL_ID,
    Column.DATA_HORA,
    Column.TEMPERATURA,
    Column.UMIDADE,
    Column.OFEGACAO,
)


FINAL_DATASET_REQUIRED_COLUMNS: tuple[str, ...] = (
    OutputColumn.ANIMAL_ID,
    OutputColumn.DATA_HORA,
    OutputColumn.OFEGACAO_HORA,
    OutputColumn.TEMPERATURA_COMPOST_1,
    OutputColumn.HUMIDADE_COMPOST_1,
    OutputColumn.THI_COMPOST_1,
)


ENVIRONMENT_COLUMNS: tuple[str, ...] = (
    OutputColumn.TEMPERATURA_COMPOST_1,
    OutputColumn.HUMIDADE_COMPOST_1,
    OutputColumn.THI_COMPOST_1,
    OutputColumn.TEMPERATURA_COMPOST_2,
    OutputColumn.HUMIDADE_COMPOST_2,
    OutputColumn.THI_COMPOST_2,
)
