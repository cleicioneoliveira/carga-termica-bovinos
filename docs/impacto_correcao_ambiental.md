# Impacto da correção ambiental

Este diagnóstico responde à pergunta:

> O que mudou quando os dados corrigidos de temperatura e umidade foram incorporados ao monitoramento?

A análise é feita comparando o dataset original e o dataset corrigido/integrado por chave:

```text
brinco + data_hora
```

O objetivo é separar efeitos de:

```text
mudança dos valores ambientais
```

versus

```text
mudança de cobertura/composição da base
```

---

## Script

```text
scripts/analyze_environment_correction_impact.py
```

---

## Comando básico

```bash
python scripts/analyze_environment_correction_impact.py \
  --original "/media/extra/wrk/CONFORTO/dataset/raw/monitoramento_1293_completo(in).csv" \
  --corrected /media/extra/wrk/CONFORTO/dataset/processado/monitoramento_saude_unificado.parquet \
  --output-dir outputs_impacto_correcao_ambiental
```

---

## O que o script calcula

Para cada registro pareado por `brinco + data_hora`, o script compara:

```text
temperatura_compost_1 original vs corrigida
humidade_compost_1 original vs corrigida
THI recalculado original vs corrigido
heat_excess original vs corrigido
```

O THI é recalculado nas duas bases a partir de temperatura e umidade usando o mesmo limiar de excesso térmico configurado, por padrão:

```text
THI threshold = 72
```

---

## Arquivos gerados

```text
outputs_impacto_correcao_ambiental/
├── coverage_and_validity_summary.json
├── environment_delta_summary.csv
├── validity_transition_summary.csv
├── thi_threshold_transition_summary.csv
├── environment_delta_by_animal.csv
├── environment_delta_by_month.csv
├── environment_delta_by_hour.csv
├── temporal_continuity_by_animal.csv
└── matched_environment_impact.parquet
```

---

## Interpretação dos principais arquivos

### `coverage_and_validity_summary.json`

Mostra:

- linhas totais no original;
- linhas totais no corrigido;
- linhas pareadas;
- animais em cada base;
- período coberto em cada base;
- linhas válidas para análise térmica no original e no corrigido.

Esse arquivo ajuda a saber se a correção ambiental também alterou a quantidade de dados utilizáveis.

### `environment_delta_summary.csv`

Resume as diferenças numéricas entre original e corrigido:

- média original;
- média corrigida;
- diferença média;
- diferença mediana;
- diferença absoluta média;
- percentil 95 da diferença absoluta;
- máximo da diferença absoluta;
- fração de registros alterados.

Variáveis avaliadas:

```text
temperatura_compost_1
humidade_compost_1
thi_recalculated
heat_excess
```

### `validity_transition_summary.csv`

Mostra se a linha era válida para análise térmica antes e depois da correção:

```text
valid_in_both
valid_only_original
valid_only_corrected
invalid_in_both
```

Esse arquivo é importante porque uma linha pode ter passado a entrar na análise apenas depois da correção ambiental.

### `thi_threshold_transition_summary.csv`

Mostra se a correção mudou a classificação térmica em relação ao limiar THI = 72:

```text
below_to_below
below_to_above
above_to_below
above_to_above
```

Interpretação:

```text
below_to_above
```

significa que a correção fez o registro passar a contribuir para o excesso térmico.

```text
above_to_below
```

significa que a correção removeu aquele registro da zona de excesso térmico.

### `environment_delta_by_animal.csv`

Mostra quais animais foram mais afetados pela correção ambiental.

### `environment_delta_by_month.csv`

Mostra em quais meses a correção teve maior impacto.

### `environment_delta_by_hour.csv`

Mostra em quais horários do dia a correção teve maior impacto.

### `temporal_continuity_by_animal.csv`

Compara a continuidade temporal dos registros válidos por animal no original e no corrigido. Esse arquivo ajuda a investigar se a janela ótima mudou por alteração na continuidade das séries.

---

## Leitura esperada no caso atual

Pelos testes já feitos:

```text
original pareado  -> 15h
corrigido pareado -> 15h
corrigido exclusivo -> 8h
corrigido completo -> 18h
```

Portanto, a hipótese mais provável é que a janela de 18h surja da interação entre:

```text
valores ambientais corrigidos
+
registros adicionais incorporados
+
continuidade temporal alterada por animal
```

Este script ajuda a quantificar essa mudança.
