# Auditoria de timestamps e disponibilidade ambiental

Este diagnóstico responde a uma pergunta específica:

> Quando os dados corrigidos passaram a ter temperatura e umidade, esses valores foram preenchidos em timestamps que já existiam para o animal ou foram criados novos timestamps?

A análise é feita por:

```text
brinco + data_hora
```

E também por:

```text
brinco + data
```

para distinguir novos horários dentro de dias já existentes de dias totalmente novos.

---

## Script

```text
scripts/audit_timestamp_environment_availability.py
```

---

## Comando

```bash
python scripts/audit_timestamp_environment_availability.py \
  --original "/media/extra/wrk/CONFORTO/dataset/raw/monitoramento_1293_completo(in).csv" \
  --corrected /media/extra/wrk/CONFORTO/dataset/processado/monitoramento_saude_unificado.parquet \
  --output-dir outputs_auditoria_timestamps_ambiente
```

---

## Arquivos gerados

```text
outputs_auditoria_timestamps_ambiente/
├── timestamp_environment_audit_summary.json
├── matched_environment_availability_transitions.csv
├── timestamp_environment_by_animal.csv
├── timestamp_environment_by_animal_day.csv
├── corrected_only_timestamp_context.parquet
├── original_only_timestamp_context.parquet
└── matched_timestamp_environment_flags.parquet
```

---

## O que cada arquivo responde

### `timestamp_environment_audit_summary.json`

Resumo geral com:

- total de registros no original;
- total de registros no corrigido;
- timestamps pareados;
- timestamps exclusivos do original;
- timestamps exclusivos do corrigido;
- quantos registros tinham T/RH completos no original;
- quantos registros passaram a ter T/RH completos no corrigido;
- quantos timestamps já existiam no original, tinham comportamento, mas não tinham T/RH, e passaram a ter T/RH no corrigido.

### `matched_environment_availability_transitions.csv`

Mostra, apenas para timestamps existentes nas duas bases, as transições:

```text
env_present_in_both
env_missing_original_present_corrected
env_present_original_missing_corrected
env_missing_in_both
original_record_had_behavior_but_no_env_and_corrected_added_env
```

O caso mais importante para a tua pergunta é:

```text
original_record_had_behavior_but_no_env_and_corrected_added_env
```

Ele significa:

> O timestamp do animal já existia no original, havia pelo menos algum dado comportamental, mas temperatura/umidade estavam ausentes. Depois da correção, esse mesmo timestamp passou a ter T/RH.

### `timestamp_environment_by_animal.csv`

Resumo por animal mostrando:

- timestamps originais;
- timestamps corrigidos;
- timestamps pareados;
- timestamps exclusivos do corrigido;
- timestamps corrigidos em dias já existentes no original;
- timestamps corrigidos em dias novos;
- ganho de T/RH completos por animal.

### `timestamp_environment_by_animal_day.csv`

Resumo por animal e dia. Ajuda a ver se os dados novos foram adicionados dentro de dias que já existiam ou em dias completamente novos.

### `corrected_only_timestamp_context.parquet`

Detalha cada timestamp exclusivo do corrigido e informa:

- se o mesmo animal-dia existia no original;
- qual era o timestamp original mais próximo para o mesmo animal;
- distância em horas para esse timestamp mais próximo.

Isso ajuda a saber se o corrigido inseriu horários próximos a observações originais ou blocos temporais inteiros.

---

## Interpretação esperada

### Caso A

```text
original_record_had_behavior_but_no_env_and_corrected_added_env alto
```

Interpretação:

> A correção ambiental preencheu T/RH em registros que já existiam para o animal. Isso é metodologicamente mais defensável, porque o timestamp não foi criado; apenas a variável ambiental ausente foi incorporada.

### Caso B

```text
corrected_only_same_original_day alto
corrected_only_new_original_day baixo
```

Interpretação:

> Muitos timestamps foram adicionados, mas dentro de dias que já existiam para o animal. Isso sugere preenchimento/regularização horária dentro de períodos já monitorados.

### Caso C

```text
corrected_only_new_original_day alto
```

Interpretação:

> O corrigido trouxe dias inteiros que não existiam no original para aquele animal. Isso precisa de mais cautela, porque altera fortemente a cobertura temporal da série.

---

## Relação com a janela de CTA

Como a carga térmica acumulada depende de sequência temporal, preencher T/RH em timestamps existentes ou criar novos timestamps pode mudar:

- continuidade das séries por animal;
- número de lacunas;
- persistência do excesso térmico;
- correlação entre carga térmica e ofegação;
- janela ótima estimada.
