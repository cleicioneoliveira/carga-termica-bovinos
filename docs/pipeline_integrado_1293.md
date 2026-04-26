# Pipeline integrado da fazenda 1293

Este documento descreve o fluxo integrado de processamento usado para gerar o dataset final consumido pelo pacote `carga-termica-bovinos`.

O fluxo é composto por quatro repositórios independentes, executados em sequência:

1. `environment_correction`
2. `status_timeline_reconstructor`
3. `merge_monitoramento_saude`
4. `carga-termica-bovinos`

A lógica geral é:

```text
heat_stress_report_f1293.csv
        +
monitoramento_full.csv
        ↓
monitoramento_corrigido.csv

saude_1293.xlsx
        ↓
saude_timeline_final.parquet

monitoramento_corrigido.csv
        +
saude_timeline_final.parquet
        ↓
monitoramento_saude_unificado.parquet

monitoramento_saude_unificado.parquet
        ↓
análise de carga térmica e conforto psicrométrico
```

---

## 1. Correção das variáveis ambientais

Repositório: `environment_correction`

Objetivo: corrigir temperatura, umidade e THI do monitoramento usando o arquivo `heat_stress_report` como fonte ambiental de referência.

Comando operacional:

```bash
python -m environment_correction.environment_correction \
  --heat dataset/raw/heat_stress_report_f1293.csv \
  --monitoramento dataset/raw/monitoramento_full.csv \
  --output-monitoramento dataset/processado/monitoramento_corrigido.csv \
  --output-audit dataset/processado/reports/device_lag_audit.csv \
  --output-pairs dataset/processado/reports/device_pair_candidates.csv \
  --output-summary dataset/processado/reports/correction_summary.json \
  --output-quality dataset/processado/reports/quality_summary.csv \
  --output-inconsistencies dataset/processado/reports/environment_inconsistencies.csv \
  --output-coverage dataset/processado/reports/correction_coverage.csv \
  --lag-min -6 \
  --lag-max 6 \
  --min-overlap-hours 72 \
  --lag-mode shared \
  --humidity-unit auto \
  --aggregation mean \
  --min-score-margin 0.05 \
  --log-level INFO
```

Produto principal:

```text
dataset/processado/monitoramento_corrigido.csv
```

Produtos de auditoria:

```text
dataset/processado/reports/device_lag_audit.csv
dataset/processado/reports/device_pair_candidates.csv
dataset/processado/reports/correction_summary.json
dataset/processado/reports/quality_summary.csv
dataset/processado/reports/environment_inconsistencies.csv
dataset/processado/reports/correction_coverage.csv
```

O README do `environment_correction` informa que o utilitário preserva as demais colunas do monitoramento e substitui apenas as variáveis ambientais finais.

---

## 2. Reconstrução da timeline de saúde

Repositório: `status_timeline_reconstructor`

Objetivo: transformar eventos esparsos de mudança de status em uma linha do tempo regular, com um status vigente por animal e horário.

Comando operacional:

```bash
python -m status_timeline_reconstructor.status_timeline_reconstructor \
  --input dataset/raw/saude_1293.xlsx \
  --output-dir dataset/processado \
  --id-col brinco \
  --datetime-col data_mudanca_status \
  --status-col status_saude \
  --previous-status-col status_saude_anterior \
  --next-status-col prox_status_saude \
  --output-datetime-col data_hora \
  --analysis-start "2025-01-01 00:00:00" \
  --analysis-end "2025-12-31 23:00:00" \
  --freq h \
  --valid-status Desafio \
  --valid-status Observacao \
  --valid-status Grave \
  --valid-status Normal \
  --status-alias "observação=Observacao" \
  --status-alias "observacao=Observacao" \
  --status-alias "obs=Observacao" \
  --conflict-policy flag \
  --normalize-columns \
  --prefix saude \
  --log-level INFO
```

Produto principal esperado:

```text
dataset/processado/saude_timeline_final.parquet
```

Esse nome ocorre porque o prefixo `saude` é aplicado antes de `timeline_final.parquet`.

Produtos auxiliares esperados:

```text
dataset/processado/reconstruction/saude_timeline_final.csv
dataset/processado/reconstruction/saude_validated_events.csv
dataset/processado/reconstruction/saude_episodes.csv
dataset/processado/reconstruction/saude_issues.csv
dataset/processado/reports/saude_reconstruction_summary.json
dataset/processado/reports/saude_reconstruction_summary.md
```

A regra metodológica central é: o status registrado passa a valer no timestamp do evento e permanece vigente até imediatamente antes da próxima mudança registrada para o mesmo indivíduo.

---

## 3. Merge entre monitoramento e saúde

Repositório: `merge_monitoramento_saude`

Objetivo: combinar o monitoramento corrigido com a timeline de saúde reconstruída.

Comando operacional:

```bash
python -m merge_monitoramento_saude.cli \
  --monitoramento dataset/processado/monitoramento_corrigido.csv \
  --saude dataset/processado/saude_timeline_final.parquet \
  --output-dir dataset/processado/
```

Produto esperado para a etapa seguinte:

```text
dataset/processado/monitoramento_saude_unificado.parquet
```

Esse arquivo deve conter, no mínimo:

```text
brinco
data_hora
status_saude ou status_vigente
ruminacao_hora
atividade_hora
ocio_hora
ofegacao_hora
temperatura_compost_1
humidade_compost_1
thi_compost1
temperatura_compost_2
humidade_compost_2
thi_compost2
```

Ponto de atenção: o repositório `merge_monitoramento_saude` ainda precisa de README operacional completo. Ele é o elo mais frágil da cadeia do ponto de vista de documentação.

---

## 4. Análise de carga térmica e zonas psicrométricas

Repositório: `carga-termica-bovinos`

Objetivo: usar o dataset unificado para calcular carga térmica acumulada, identificar períodos de conforto e gerar zonas psicrométricas empíricas.

Comando oficial:

```bash
python -m app.run_pipeline --config app/config.yaml
```

Para exibir os gráficos:

```bash
python -m app.run_pipeline --config app/config.yaml --show-plots
```

O arquivo `app/config.yaml` deve apontar para:

```yaml
dataset_path: "/media/extra/wrk/CONFORTO/dataset/processado/monitoramento_saude_unificado.parquet"
```

Produtos principais:

```text
outputs_conforto/resultados_janelas.csv
outputs_conforto/best_window.json
outputs_conforto/dados_conforto_psicrometrico.csv
outputs_conforto/temporal_scale_academic.png
outputs_conforto/fig_psychrometric_comfort.png
outputs_conforto/fig_comfort_polygon.png
```

---

## Contratos entre etapas

| Etapa | Entrada principal | Saída principal | Consumida por |
|---|---|---|---|
| environment_correction | `heat_stress_report_f1293.csv` + `monitoramento_full.csv` | `monitoramento_corrigido.csv` | merge_monitoramento_saude |
| status_timeline_reconstructor | `saude_1293.xlsx` | `saude_timeline_final.parquet` | merge_monitoramento_saude |
| merge_monitoramento_saude | `monitoramento_corrigido.csv` + `saude_timeline_final.parquet` | `monitoramento_saude_unificado.parquet` | carga-termica-bovinos |
| carga-termica-bovinos | `monitoramento_saude_unificado.parquet` | figuras e tabelas de conforto | artigo/dissertação |

---

## Pontos críticos de compatibilidade

### 1. Chave temporal

Todas as etapas devem convergir para:

```text
brinco + data_hora
```

A timeline de saúde já deve ser emitida com:

```bash
--output-datetime-col data_hora
```

### 2. Frequência temporal

O monitoramento e a timeline de saúde devem estar na mesma frequência. No fluxo atual, a reconstrução usa:

```bash
--freq h
```

Portanto, o merge deve operar em base horária.

### 3. Nome do status final

O downstream deve padronizar para uma coluna estável. A recomendação é usar:

```text
status_saude
```

Se o merge receber `status_vigente`, ele deve renomear para `status_saude` antes de gerar o dataset final.

### 4. Variáveis ambientais

O dataset final deve preservar as colunas corrigidas:

```text
temperatura_compost_1
humidade_compost_1
thi_compost1
temperatura_compost_2
humidade_compost_2
thi_compost2
```

### 5. Caminho do dataset final

O arquivo final esperado pelo `carga-termica-bovinos` é:

```text
dataset/processado/monitoramento_saude_unificado.parquet
```

---

## Recomendação de evolução

O fluxo já funciona, mas a arquitetura ideal seria criar um pequeno pacote orquestrador, por exemplo:

```text
farm1293_pipeline/
```

Esse pacote poderia apenas coordenar os quatro repositórios, sem misturar as responsabilidades internas deles.

Enquanto isso, o script `scripts/run_pipeline_1293.sh` documenta e automatiza a ordem correta das etapas.
