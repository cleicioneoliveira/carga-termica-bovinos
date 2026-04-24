from __future__ import annotations

from enum import StrEnum


class Column(StrEnum):
    """Nomes canônicos de colunas usados pelo pipeline."""

    ANIMAL_ID = "brinco"
    DATA_HORA = "data_hora"
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
    """Nomes de colunas vindos das fontes originais conhecidas."""

    TEMPERATURA_COMPOST = "temperatura_compost_1"
    HUMIDADE_COMPOST = "humidade_compost_1"
    OFEGACAO_HORA = "ofegacao_hora"

    TEMPERATURA_INMET = "temperatura_do_ar_bulbo_seco_horaria_c"
    UMIDADE_INMET = "umidade_relativa_do_ar_horaria"


STANDARDIZATION_MAP: dict[str, str] = {
    SourceColumn.TEMPERATURA_COMPOST: Column.TEMPERATURA,
    SourceColumn.HUMIDADE_COMPOST: Column.UMIDADE,
    SourceColumn.OFEGACAO_HORA: Column.OFEGACAO,
}


REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    Column.ANIMAL_ID,
    Column.DATA_HORA,
    Column.TEMPERATURA,
    Column.UMIDADE,
    Column.OFEGACAO,
)
