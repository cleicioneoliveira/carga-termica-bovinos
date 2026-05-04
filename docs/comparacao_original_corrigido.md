# Comparação entre dataset original e dataset corrigido

Este documento descreve como investigar se a mudança da janela ótima de carga térmica de 15h para 18h decorre dos valores ambientais corrigidos ou da mudança de composição/cobertura do dataset.

## Objetivo

Comparar dois arquivos:

- dataset original;
- dataset corrigido/integrado.

A comparação é feita por chave:

```text
brinco + data_hora
```

O script gera relatórios de cobertura e datasets pareados para rerodar o pipeline térmico em condições comparáveis.

---

## Script

```text
scripts/compare_original_corrected_dataset.py
```

---

## Exemplo de uso

```bash
python scripts/compare_original_corrected_dataset.py \
  --original "dataset/raw/monitoramento_1293_completo(in).csv" \
  --corrected dataset/processado/monitoramento_saude_unificado.parquet \
  --output-dir outputs_comparacao_original_corrigido
```

Também é possível usar CSV ou Parquet em qualquer uma das entradas.

---

## Arquivos gerados

```text
outputs_comparacao_original_corrigido/
├── coverage_summary.json
├── value_difference_summary.csv
├── original_matched_for_pipeline.parquet
├── corrected_matched_for_pipeline.parquet
├── matched_comparison_wide.parquet
├── original_only.parquet
└── corrected_only.parquet
```

---

## Interpretação dos arquivos

### `coverage_summary.json`

Mostra:

- número de linhas do original;
- número de linhas do corrigido;
- número de chaves únicas;
- número de chaves sobrepostas;
- fração de sobreposição;
- número de animais em cada base;
- intervalo temporal de cada base.

Esse arquivo ajuda a detectar se a diferença vem de cobertura, animais ou horários diferentes.

### `value_difference_summary.csv`

Compara, nos registros pareados, variáveis como:

```text
temperatura_compost_1
humidade_compost_1
thi_compost1
temperatura_compost_2
humidade_compost_2
thi_compost2
ofegacao_hora
```

Para cada variável, calcula:

- número de pares válidos;
- número e fração de valores alterados;
- média original;
- média corrigida;
- diferença média;
- diferença mediana;
- desvio padrão da diferença;
- mínimo e máximo da diferença;
- diferença absoluta média;
- percentil 95 da diferença absoluta.

### `original_matched_for_pipeline.parquet`

Contém apenas os registros do dataset original que também existem no corrigido.

Use para testar:

```bash
python -m app.run_pipeline --config app/config.yaml \
  --dataset outputs_comparacao_original_corrigido/original_matched_for_pipeline.parquet
```

### `corrected_matched_for_pipeline.parquet`

Contém apenas os registros do dataset corrigido que também existem no original.

Use para testar:

```bash
python -m app.run_pipeline --config app/config.yaml \
  --dataset outputs_comparacao_original_corrigido/corrected_matched_for_pipeline.parquet
```

---

## Como interpretar o resultado

### Caso 1

```text
original completo = 15h
original pareado  = 18h
corrigido pareado = 18h
corrigido completo = 18h
```

A mudança provavelmente vem da composição/cobertura dos dados.

### Caso 2

```text
original completo = 15h
original pareado  = 15h
corrigido pareado = 18h
corrigido completo = 18h
```

A mudança provavelmente vem dos valores ambientais corrigidos.

### Caso 3

```text
original pareado e corrigido pareado ficam próximos
```

A diferença pode estar em registros exclusivos de uma das bases ou no processo de integração.

---

## Observação metodológica

Este diagnóstico não substitui a análise científica principal. Ele serve para separar o efeito de:

```text
valor ambiental corrigido
```

versus

```text
mudança de cobertura/composição da base
```
