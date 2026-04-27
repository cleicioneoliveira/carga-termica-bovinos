# Contrato canônico de colunas do pipeline 1293

Este documento define a nomenclatura padronizada de colunas usada na cadeia completa de processamento:

1. `environment_correction`
2. `status_timeline_reconstructor`
3. `merge_monitoramento_saude`
4. `carga-termica-bovinos`

O objetivo é evitar que cada etapa use nomes diferentes para a mesma informação.

---

## 1. Princípio geral

Cada pacote pode aceitar nomes de entrada vindos de fontes originais, mas deve converter internamente para nomes canônicos antes de processar e deve exportar nomes padronizados para a próxima etapa.

A chave principal da cadeia é sempre:

```text
brinco + data_hora
```

---

## 2. Colunas canônicas principais

| Conceito | Nome canônico | Observação |
|---|---|---|
| Identificador do animal | `brinco` | Chave individual |
| Data/hora da observação | `data_hora` | Chave temporal horária |
| Status de saúde final | `status_saude` | Nome recomendado no dataset integrado |
| Status vigente reconstruído | `status_vigente` | Nome interno/saída do reconstrutor; pode ser renomeado para `status_saude` no merge |
| Temperatura operacional | `temperatura` | Nome interno usado pela análise térmica |
| Umidade operacional | `umidade` | Nome interno usado pela análise térmica |
| Ofegação operacional | `ofegacao` | Nome interno usado pela análise térmica |
| THI/ITU operacional | `thi` | Calculado internamente |
| Excesso térmico | `heat_excess` | Calculado internamente |

---

## 3. Colunas ambientais corrigidas

O `environment_correction` deve preservar e corrigir as seguintes colunas no monitoramento:

```text
temperatura_compost_1
humidade_compost_1
thi_compost1
temperatura_compost_2
humidade_compost_2
thi_compost2
```

Para a análise térmica, o pacote `carga-termica-bovinos` mapeia por padrão:

```text
temperatura_compost_1 -> temperatura
humidade_compost_1    -> umidade
thi_compost1          -> thi_compost1 preservado como fonte
```

A análise térmica calcula novamente `thi` internamente a partir de `temperatura` e `umidade`.

---

## 4. Colunas comportamentais

| Fonte | Canônico operacional |
|---|---|
| `ofegacao_hora` | `ofegacao` |
| `ruminacao_hora` | preservada no dataset final |
| `atividade_hora` | preservada no dataset final |
| `ocio_hora` | preservada no dataset final |

---

## 5. Colunas de saúde

O `status_timeline_reconstructor` recebe eventos de saúde com:

```text
brinco
data_mudanca_status
status_saude
status_saude_anterior
prox_status_saude
```

Ele reconstrói uma timeline regular contendo, entre outras:

```text
brinco
data_hora
status_vigente
status_inicio_vigencia
status_fim_vigencia_inferido
proxima_mudanca
episode_number
```

No dataset integrado final, recomenda-se exportar:

```text
status_saude
```

como alias final de `status_vigente`.

---

## 6. Contrato mínimo do dataset final integrado

O arquivo final consumido por `carga-termica-bovinos` deve conter pelo menos:

```text
brinco
data_hora
status_saude
ofegacao_hora
temperatura_compost_1
humidade_compost_1
thi_compost1
```

Colunas recomendadas adicionais:

```text
ruminacao_hora
atividade_hora
ocio_hora
ruminacao_acumulado
atividade_acumulado
ocio_acumulado
ofegacao_acumulado
temperatura_compost_2
humidade_compost_2
thi_compost2
```

---

## 7. Regras de padronização

### 7.1 Normalização de nomes brutos

Quando possível, nomes vindos de planilhas devem ser normalizados para:

- minúsculas;
- sem acentos;
- espaços substituídos por `_`;
- `/`, `-` e `.` substituídos/removidos conforme necessário.

Exemplo:

```text
Data Mudança Status -> data_mudanca_status
Observação          -> observacao
```

### 7.2 Alias explícitos

Todo pacote deve manter um mapa explícito de aliases conhecidos. Exemplo:

```python
{
    "animal_id": "brinco",
    "timestamp": "data_hora",
    "status_vigente": "status_saude",
    "ofegacao_hora": "ofegacao",
}
```

### 7.3 Exportação estável

A saída de cada etapa deve ter nomes previsíveis e documentados. O próximo pacote não deve depender de nomes acidentais ou temporários.

---

## 8. Responsabilidade por pacote

### environment_correction

Responsável por padronizar nomes ambientais do monitoramento e garantir que as seis colunas ambientais corrigidas sejam preservadas.

### status_timeline_reconstructor

Responsável por padronizar colunas de eventos de saúde e emitir `brinco`, `data_hora` e `status_vigente`.

### merge_monitoramento_saude

Responsável por unir monitoramento e saúde usando `brinco + data_hora` e exportar `status_saude` como coluna final padronizada.

### carga-termica-bovinos

Responsável por mapear o dataset final para nomes internos simples:

```text
temperatura_compost_1 -> temperatura
humidade_compost_1    -> umidade
ofegacao_hora         -> ofegacao
```

---

## 9. Recomendação futura

A solução ideal é criar um pequeno pacote compartilhado, por exemplo:

```text
farm1293_schema
```

Esse pacote conteria apenas enums, aliases e validadores de colunas, sendo usado pelos quatro repositórios.

Enquanto isso, cada repositório deve conter seu próprio `columns.py` alinhado com este contrato.
