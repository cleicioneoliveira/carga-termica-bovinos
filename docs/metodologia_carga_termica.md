# Metodologia: carga térmica acumulada e conforto térmico empírico

## 1. Objetivo

Este documento descreve a metodologia implementada no pipeline de análise de carga térmica em bovinos leiteiros. O objetivo é identificar períodos empiricamente associados ao conforto térmico e projetar esses períodos no espaço psicrométrico para derivar zonas de conforto baseadas em dados observacionais.

A abordagem parte do princípio de que conforto térmico não deve ser tratado apenas como uma condição instantânea definida por limites fixos de temperatura e umidade. No contexto de bovinos leiteiros em ambiente produtivo, a resposta fisiológica e comportamental tende a refletir o histórico térmico recente, especialmente quando o animal é submetido a estresse térmico persistente.

## 2. Fontes de dados

A análise envolve duas escalas de dados diferentes.

A primeira fonte é o `heat_stress_report`, utilizado como fonte ambiental de maior resolução temporal. Esse arquivo contém medições sub-horárias, normalmente em intervalos de 5 minutos, por dispositivo ambiental. O contrato mínimo de colunas segue o mesmo padrão utilizado no repositório `environment_correction`:

```text
timestamp
dispositivo
temperature
humidity
```

Também são aceitos aliases comuns, como `data`, `data_hora`, `datetime`, `device`, `sensor`, `temperatura` e `umidade`.

A segunda fonte é o dataset horário por animal, contendo as variáveis comportamentais e fisiológicas indiretas, como ofegação, ruminação, atividade e ócio. Essas variáveis são disponibilizadas como acumulados horários por animal.

Assim, a metodologia correta é calcular a carga térmica acumulada a partir do `heat_stress_report` em 5 minutos, agregar a CTA para a escala horária e, somente depois, unir essa informação ambiental horária à base animal horária.

## 3. Etapa ambiental: geração da CTA horária a partir do heat_stress_report

O script `scripts/build_hourly_cta_from_heat_stress.py` foi criado para gerar uma tabela ambiental horária pronta para uso na dissertação.

Ele executa as seguintes etapas:

1. lê o `heat_stress_report` em CSV ou Parquet;
2. normaliza os nomes das colunas;
3. valida as colunas `timestamp`, `dispositivo`, `temperature` e `humidity`;
4. converte data/hora, temperatura e umidade para tipos adequados;
5. calcula o Índice de Temperatura e Umidade (ITU);
6. calcula o excesso térmico acima de 72;
7. calcula a CTA ponderada pelo intervalo temporal de cada registro;
8. agrega os resultados para a escala horária por dispositivo;
9. salva uma tabela ambiental horária em CSV ou Parquet.

Comando recomendado:

```bash
python scripts/build_hourly_cta_from_heat_stress.py \
  --input /caminho/para/heat_stress_report_f1293.csv \
  --output outputs_conforto/cta_horaria_from_heat_stress.parquet \
  --output-csv outputs_conforto/cta_horaria_from_heat_stress.csv \
  --windows 6 9 12 15 18 24 \
  --threshold 72 \
  --input-frequency-minutes 5 \
  --humidity-unit auto
```

A saída gerada possui colunas no seguinte formato:

```text
data_hora
dispositivo
temperatura
umidade
itu
heat_excess
cta_6h
cta_9h
cta_12h
cta_15h
cta_18h
cta_24h
```

Essa saída ainda não contém ofegação ou identificação de animal. Ela representa a etapa ambiental da análise.

## 4. Cálculo do índice térmico instantâneo

O primeiro passo térmico é calcular o Índice de Temperatura e Umidade. A partir dele, calcula-se o excesso térmico em relação ao limiar crítico definido na configuração.

```text
heat_excess_t = max(ITU_t - ITU_threshold, 0)
```

No arquivo `app/config.yaml`, o limiar padrão é:

```yaml
thi_threshold: 72
```

Apesar do nome histórico `thi_threshold` no código, o texto da dissertação deve usar a terminologia em português: Índice de Temperatura e Umidade (ITU).

Esse valor representa o ponto a partir do qual o ambiente passa a contribuir para a carga térmica acumulada. Valores de ITU abaixo do limiar não aumentam a carga acumulada.

## 5. Carga térmica acumulada

A carga térmica acumulada pode ser calculada de duas formas, conforme a resolução temporal dos dados ambientais.

### 5.1 Modo legado, para dados horários

O modo legado preserva o comportamento original do pipeline e assume dados horários regulares. Nesse caso, uma janela de 15 representa os últimos 15 registros, equivalentes a 15 horas.

```text
heat_load_t(w) = sum heat_excess over the last w hourly records
```

Configuração correspondente:

```yaml
thermal_time_resolution:
  input_frequency_minutes: 60
  weighted_by_time: false
```

### 5.2 Modo ponderado no tempo, para dados de 5 minutos

Quando os dados ambientais estão em intervalos de 5 minutos, a carga térmica não deve ser calculada como soma simples dos registros, pois isso inflaria artificialmente a carga em relação à escala horária. Nesse caso, cada registro precisa ser ponderado pelo intervalo temporal que representa.

Para dados de 5 minutos:

```text
delta_t_hours = 5 / 60
```

A carga térmica acumulada passa a ser:

```text
heat_load_t(w) = sum heat_excess_i * delta_t_hours
```

em que a soma é feita sobre todos os registros contidos na janela temporal de interesse. Assim, uma janela de 15 horas em dados de 5 minutos usa 180 registros, e uma janela de 18 horas usa 216 registros.

Configuração correspondente:

```yaml
thermal_time_resolution:
  input_frequency_minutes: 5
  weighted_by_time: true
```

## 6. Etapa animal: integração com ofegação horária

Depois de gerar a tabela `cta_horaria_from_heat_stress`, ela deve ser unida à base animal horária, que contém pelo menos:

```text
brinco
data_hora
ofegacao_hora
```

A união deve respeitar o mapeamento entre `dispositivo` e o compost ou galpão correspondente, conforme definido pela auditoria do `environment_correction`. A tabela ambiental horária fornece a CTA por dispositivo; a tabela animal horária fornece a resposta por animal.

O resultado esperado para a análise CTA versus ofegação deve ter uma estrutura semelhante a:

```text
brinco
data_hora
ofegacao
dispositivo
temperatura
umidade
itu
heat_excess
cta_6h
cta_9h
cta_12h
cta_15h
cta_18h
cta_24h
```

Esse dataset integrado é a base correta para comparar janelas de CTA e definir qual escala temporal apresenta maior associação com a ofegação.

## 7. Modos de execução do pipeline integrado

O pipeline principal permite dois modos.

### Modo manual

Usa uma janela fixa definida em:

```yaml
thermal_mode: "manual"
thermal_window: 15
```

No modo ponderado, `thermal_window` continua sendo interpretado em horas. Assim, `thermal_window: 15` significa 15 horas, independentemente de o dado ser horário ou de 5 minutos.

### Modo automático

Testa várias janelas candidatas e escolhe aquela que maximiza o critério configurado:

```yaml
thermal_mode: "auto"
thermal_windows: [1, 2, ..., 24]
thermal_criterion: "mean_corr"
```

Para cada janela, o pipeline calcula a correlação entre carga térmica acumulada e ofegação por animal. Em seguida, agrega as correlações individuais por média e mediana, além de registrar o número de animais com correlação positiva ou negativa.

## 8. Escolha da melhor janela temporal

No modo automático, a melhor janela é definida pelo maior valor do critério escolhido. Os critérios atualmente aceitos são:

- `mean_corr`: maior correlação média entre carga térmica e ofegação;
- `median_corr`: maior correlação mediana entre carga térmica e ofegação.

A janela escolhida é salva em:

```text
outputs_conforto/best_window.json
```

E a tabela completa das janelas é salva em:

```text
outputs_conforto/resultados_janelas.csv
```

## 9. Definição dos períodos de conforto

Após definir a janela de carga térmica, o pipeline identifica períodos de conforto combinando baixa carga térmica acumulada e baixa ofegação. A lógica operacional considera percentis individuais por animal, permitindo que cada animal tenha seu próprio limiar relativo.

O período somente é aceito como conforto quando a condição persiste por uma duração mínima configurada:

```yaml
min_duration: 3
```

No modo legado horário, esse valor representa três registros consecutivos. No modo ponderado para dados de 5 minutos, esse valor é interpretado como horas e convertido internamente para o número correspondente de registros. Assim, `min_duration: 3` equivale a 36 registros consecutivos quando `input_frequency_minutes: 5`.

## 10. Projeção no espaço psicrométrico

Os registros classificados como conforto são projetados no espaço psicrométrico, usando temperatura de bulbo seco no eixo x e razão de umidade no eixo y.

Essa transformação permite interpretar o conforto em termos termodinâmicos mais consistentes do que temperatura e umidade relativa brutas.

## 11. Campo de densidade

A distribuição dos pontos de conforto no espaço psicrométrico é convertida em um histograma bidimensional normalizado. Esse campo representa a densidade empírica das condições ambientais associadas ao conforto.

Parâmetros principais:

```yaml
density:
  bins: 40
  min_density: 0.001
  use_filter: false
  percentile: 65
```

## 12. Zonas de conforto

As zonas são definidas por percentis da densidade empírica:

```yaml
zones:
  core_percentile: 85
  transition_percentile: 60
  limit_percentile: 30
```

A interpretação adotada é:

- `core`: região de maior suporte empírico;
- `transition`: região intermediária;
- `limit`: região periférica de tolerância.

Essas zonas não devem ser interpretadas como limites fisiológicos universais, mas como regiões derivadas dos dados e da metodologia aplicada.

## 13. Extração geométrica

Os pontos das zonas são convertidos em polígonos por meio de método geométrico configurável:

```yaml
geometry:
  method: "alpha"
  alpha: 1.2
```

As opções disponíveis são:

- `alpha`: alpha shape, mais flexível e capaz de representar concavidades;
- `convex`: envoltória convexa, útil como referência simples.

## 14. Suavização visual

A suavização é aplicada apenas para melhorar a apresentação gráfica:

```yaml
smoothing:
  enabled: true
  sigma: 2
```

A suavização não deve ser usada como substituto da geometria bruta para análise quantitativa.

## 15. Saídas principais

A etapa ambiental gera:

- `cta_horaria_from_heat_stress.parquet`;
- `cta_horaria_from_heat_stress.csv`.

O pipeline integrado gera, por padrão, os seguintes arquivos em `outputs_conforto/`:

- `resultados_janelas.csv`;
- `best_window.json`;
- `dados_conforto_psicrometrico.csv`;
- `temporal_scale_academic.png`;
- `temporal_scale_academic.pdf`;
- `fig_psychrometric_comfort.png`;
- `fig_psychrometric_comfort.pdf`;
- `fig_comfort_polygon.png`.

## 16. Interpretação científica

A metodologia é empírica e orientada por dados. Ela não propõe uma zona universal definitiva de conforto térmico para bovinos. Em vez disso, fornece um procedimento reprodutível para identificar, em um conjunto de dados específico, as regiões ambientais associadas a baixa carga térmica acumulada e baixa resposta de ofegação.

A interpretação correta é:

```text
densidade = suporte empírico dos dados
polígono = representação geométrica derivada
zona = região estatística, não limite fisiológico absoluto
```

## 17. Comandos oficiais

Para gerar a CTA horária a partir do `heat_stress_report`:

```bash
python scripts/build_hourly_cta_from_heat_stress.py \
  --input /caminho/para/heat_stress_report_f1293.csv \
  --output outputs_conforto/cta_horaria_from_heat_stress.parquet \
  --output-csv outputs_conforto/cta_horaria_from_heat_stress.csv \
  --windows 6 9 12 15 18 24 \
  --threshold 72 \
  --input-frequency-minutes 5 \
  --humidity-unit auto
```

Para dados horários já integrados:

```bash
python -m app.run_pipeline --config app/config.yaml
```

Para dados ambientais já integrados em 5 minutos, com CTA ponderada no tempo:

```bash
python -m app.run_pipeline \
  --config app/config.yaml \
  --dataset /caminho/para/dataset_5min.parquet \
  --thermal-mode auto \
  --input-frequency-minutes 5 \
  --weighted-by-time
```

Para exibir todos os gráficos durante a execução:

```bash
python -m app.run_pipeline --config app/config.yaml --show-plots
```
