# 🐄 Carga Térmica em Bovinos

### Análise empírica de conforto térmico em espaço psicrométrico

Este projeto implementa um pipeline científico para estimar zonas empíricas de conforto térmico em bovinos leiteiros a partir de dados observacionais, carga térmica acumulada e resposta comportamental/fisiológica.

A abordagem evita tratar conforto térmico como uma condição binária fixa. Em vez disso, identifica regiões de maior suporte empírico no espaço psicrométrico, usando temperatura do ar, umidade, THI acumulado, períodos contínuos de conforto e densidade estatística.

---

## Ideia central

O conforto térmico emerge como uma distribuição de probabilidade no espaço ambiental.

O modelo identifica três regiões:

- **Core**: conforto ótimo, associado à maior densidade de observações.
- **Transition**: zona intermediária de conforto aceitável.
- **Limit**: zona de tolerância, com menor suporte estatístico.

---

## Pipeline científico

```text
raw dataset
↓
padronização e limpeza
↓
cálculo de THI
↓
cálculo de excesso térmico
↓
análise da janela temporal de carga térmica
↓
definição de conforto
↓
extração de períodos contínuos
↓
transformação psicrométrica (T, W)
↓
campo de densidade 2D
↓
segmentação por percentis
↓
extração geométrica (alpha-shape ou convex hull)
↓
suavização visual
↓
figura final
```

---

## Instalação

```bash
git clone https://github.com/cleicioneoliveira/carga-termica-bovinos.git
cd carga-termica-bovinos
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

O pipeline lê arquivos Parquet; por isso `pyarrow` está listado como dependência. As figuras intermediárias usam `seaborn`.

---

## Execução

O entrypoint oficial é:

```bash
python -m app.run_pipeline --config app/config.yaml
```

Também é possível usar o wrapper de compatibilidade:

```bash
python -m app.run --config app/config.yaml
```

Para rodar com profiling:

```bash
python -m app.run_pipeline --config app/config.yaml --profile
```

Por padrão, o pipeline roda em modo adequado para terminal, batch ou HPC: salva as figuras e fecha as janelas gráficas automaticamente. Para exibir as figuras durante a execução:

```bash
python -m app.run_pipeline --config app/config.yaml --show-plots
```

Para depurar mensagens internas do renderizador psicrométrico:

```bash
python -m app.run_pipeline --config app/config.yaml --verbose-chart
```

---

## Configuração

A configuração oficial do pipeline fica em:

```text
app/config.yaml
```

Esse arquivo é a fonte única de verdade para parâmetros ajustáveis. O módulo `app/config.py` apenas carrega, valida e fornece compatibilidade para código antigo.

Principais blocos:

- `dataset_path`: caminho do dataset unificado.
- `thermal_mode`: modo `manual` ou `auto`.
- `thi_threshold`: limiar usado para excesso térmico.
- `thermal_windows`: janelas testadas no modo automático.
- `thermal_criterion`: critério de escolha da melhor janela.
- `thermal_output_dir`: diretório de saída dos produtos gerados.
- `show_plots`: controla se as figuras são exibidas na tela.
- `suppress_chart_stdout`: controla a supressão de mensagens internas do gráfico psicrométrico.
- `log_level`: controla o nível de logs no terminal.
- `density`: resolução e filtragem da densidade psicrométrica.
- `geometry`: método geométrico (`alpha` ou `convex`).
- `smoothing`: suavização visual dos polígonos.
- `zones`: percentis usados nas zonas core, transition e limit.

Exemplos de sobrescrita pela linha de comando:

```bash
python -m app.run_pipeline --config app/config.yaml --thermal-mode manual --thermal-window 15
```

```bash
python -m app.run_pipeline --config app/config.yaml --dataset /path/to/dataset.parquet --no-smooth
```

```bash
python -m app.run_pipeline --config app/config.yaml --log-level DEBUG
```

---

## Saídas

As saídas são salvas por padrão em:

```text
outputs_conforto/
```

Esse diretório é tratado como saída gerada e não deve ser versionado.

Exemplos de produtos:

- `resultados_janelas.csv`
- `best_window.json`
- `dados_conforto_psicrometrico.csv`
- `fig_comfort_polygon.png`
- arquivos de profiling quando `--profile` é usado

---

## Interpretação científica

A densidade representa dados observados. Os polígonos representam uma interpretação geométrica derivada desses dados.

Portanto:

```text
densidade ≠ modelo biológico absoluto
polígono ≠ verdade fisiológica universal
```

A zona estimada deve ser interpretada como uma região empírica, dependente do conjunto de dados, do critério de conforto adotado e dos parâmetros de análise temporal.

---

## Estrutura principal

```text
app/
├── config.yaml                 # configuração oficial do pipeline
├── config.py                   # carregador/validador da configuração
├── run_pipeline.py             # entrypoint principal
├── run.py                      # wrapper de compatibilidade
├── pipeline/
│   ├── density.py
│   ├── geometry.py
│   ├── smoothing.py
│   ├── zones.py
│   └── thermal_comfort/        # lógica de carga térmica e conforto
├── plot/
│   └── plot_psychro.py
├── io/
├── time/
└── util/
```

---

## Licença

LGPL v3

---

## Autores

- João Gerd Zell de Mattos (INPE / CPTEC)
- Cleicione Moura de Oliveira (UFAC / PPGESPA)
- Rafael Augusto Satrapa (UFAC / PPGESPA)
