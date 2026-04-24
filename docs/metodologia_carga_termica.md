# Metodologia: carga térmica acumulada e conforto térmico empírico

## 1. Objetivo

Este documento descreve a metodologia implementada no pipeline de análise de carga térmica em bovinos leiteiros. O objetivo é identificar períodos empiricamente associados ao conforto térmico e projetar esses períodos no espaço psicrométrico para derivar zonas de conforto baseadas em dados observacionais.

A abordagem parte do princípio de que conforto térmico não deve ser tratado apenas como uma condição instantânea definida por limites fixos de temperatura e umidade. No contexto de bovinos leiteiros em ambiente produtivo, a resposta fisiológica e comportamental tende a refletir o histórico térmico recente, especialmente quando o animal é submetido a estresse térmico persistente.

## 2. Dados de entrada

O pipeline espera um dataset unificado contendo, no mínimo, informações de identificação do animal, data e hora da observação, temperatura do ar, umidade relativa e tempo ou intensidade de ofegação.

Durante a preparação, as colunas são padronizadas para nomes internos, convertidas para tipos numéricos ou temporais e ordenadas por animal e data/hora. Registros sem os campos essenciais são removidos.

## 3. Cálculo do índice térmico instantâneo

O primeiro passo térmico é calcular o índice de temperatura e umidade, tratado internamente como ITU/THI. A partir dele, calcula-se o excesso térmico em relação a um limiar crítico definido na configuração.

```text
heat_excess_t = max(THI_t - THI_threshold, 0)
```

No arquivo `app/config.yaml`, o limiar padrão é:

```yaml
thi_threshold: 72
```

Esse valor representa o ponto a partir do qual o ambiente passa a contribuir para a carga térmica acumulada. Valores de THI abaixo do limiar não aumentam a carga acumulada.

## 4. Carga térmica acumulada

A carga térmica acumulada é calculada como a soma móvel do excesso térmico dentro de uma janela temporal definida em horas ou registros, assumindo dados horários regulares.

```text
heat_load_t(w) = sum heat_excess over the last w records
```

O pipeline permite dois modos.

### Modo manual

Usa uma janela fixa definida em:

```yaml
thermal_mode: "manual"
thermal_window: 15
```

### Modo automático

Testa várias janelas candidatas e escolhe aquela que maximiza o critério configurado:

```yaml
thermal_mode: "auto"
thermal_windows: [1, 2, ..., 24]
thermal_criterion: "mean_corr"
```

Para cada janela, o pipeline calcula a correlação entre carga térmica acumulada e ofegação por animal. Em seguida, agrega as correlações individuais por média e mediana, além de registrar o número de animais com correlação positiva ou negativa.

## 5. Escolha da melhor janela temporal

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

## 6. Definição dos períodos de conforto

Após definir a janela de carga térmica, o pipeline identifica períodos de conforto combinando baixa carga térmica acumulada e baixa ofegação. A lógica operacional considera percentis individuais por animal, permitindo que cada animal tenha seu próprio limiar relativo.

O período somente é aceito como conforto quando a condição persiste por uma duração mínima configurada:

```yaml
min_duration: 3
```

Em dados horários, esse valor representa três registros consecutivos.

## 7. Projeção no espaço psicrométrico

Os registros classificados como conforto são projetados no espaço psicrométrico, usando temperatura de bulbo seco no eixo x e razão de umidade no eixo y.

Essa transformação permite interpretar o conforto em termos termodinâmicos mais consistentes do que temperatura e umidade relativa brutas.

## 8. Campo de densidade

A distribuição dos pontos de conforto no espaço psicrométrico é convertida em um histograma bidimensional normalizado. Esse campo representa a densidade empírica das condições ambientais associadas ao conforto.

Parâmetros principais:

```yaml
density:
  bins: 40
  min_density: 0.001
  use_filter: false
  percentile: 65
```

## 9. Zonas de conforto

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

## 10. Extração geométrica

Os pontos das zonas são convertidos em polígonos por meio de método geométrico configurável:

```yaml
geometry:
  method: "alpha"
  alpha: 1.2
```

As opções disponíveis são:

- `alpha`: alpha shape, mais flexível e capaz de representar concavidades;
- `convex`: envoltória convexa, útil como referência simples.

## 11. Suavização visual

A suavização é aplicada apenas para melhorar a apresentação gráfica:

```yaml
smoothing:
  enabled: true
  sigma: 2
```

A suavização não deve ser usada como substituto da geometria bruta para análise quantitativa.

## 12. Saídas principais

O pipeline gera, por padrão, os seguintes arquivos em `outputs_conforto/`:

- `resultados_janelas.csv`;
- `best_window.json`;
- `dados_conforto_psicrometrico.csv`;
- `temporal_scale_academic.png`;
- `temporal_scale_academic.pdf`;
- `fig_psychrometric_comfort.png`;
- `fig_psychrometric_comfort.pdf`;
- `fig_comfort_polygon.png`.

## 13. Interpretação científica

A metodologia é empírica e orientada por dados. Ela não propõe uma zona universal definitiva de conforto térmico para bovinos. Em vez disso, fornece um procedimento reprodutível para identificar, em um conjunto de dados específico, as regiões ambientais associadas a baixa carga térmica acumulada e baixa resposta de ofegação.

A interpretação correta é:

```text
densidade = suporte empírico dos dados
polígono = representação geométrica derivada
zona = região estatística, não limite fisiológico absoluto
```

## 14. Comando oficial

```bash
python -m app.run_pipeline --config app/config.yaml
```

Para exibir todos os gráficos durante a execução:

```bash
python -m app.run_pipeline --config app/config.yaml --show-plots
```
