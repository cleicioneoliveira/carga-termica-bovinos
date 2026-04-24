import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Simulação de Dados (Substitua pelo carregamento do seu CSV/Excel)
# Supondo que 'df' tenha as colunas: 'timestamp', 'id_individuo', 'itu', 'minutos_ofegacao'
def gerar_dados_exemplo():
    horas = 100
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=horas, freq='H'),
        'itu': np.random.uniform(65, 85, horas),
        'minutos_ofegacao': np.random.uniform(0, 60, horas)
    })
    return df

df = gerar_dados_exemplo()

# 2. Definição do Limite de Conforto (Gatilho de Excesso)
ITU_CONFORTO = 72
df['itu_excesso'] = (df['itu'] - ITU_CONFORTO).clip(lower=0)

# 3. Testando Diferentes Janelas de Acúmulo (Lag Analysis)
janelas_para_testar = [1, 2, 3, 4, 5, 6, 8, 12] # em horas
resultados_correlacao = {}

for j in janelas_para_testar:
    # Criamos a soma móvel (rolling sum) "olhando para trás"
    # O parâmetro 'window' define quantas linhas (horas) somar
    col_nome = f'carga_acumulada_{j}h'
    df[col_nome] = df['itu_excesso'].rolling(window=j).sum()
    
    # Calculamos a correlação de Pearson entre o acúmulo e a ofegação atual
    # Dropna é necessário porque as primeiras linhas da janela são NaN
    corr = df[col_nome].corr(df['minutos_ofegacao'])
    resultados_correlacao[j] = corr

# 4. Identificando a Melhor Janela
melhor_janela = max(resultados_correlacao, key=resultados_correlacao.get)

print(f"--- Resultados da Análise ---")
for j, r in resultados_correlacao.items():
    print(f"Janela de {j}h: Correlação r = {r:.4f}")

print(f"\n>>> A melhor janela de acúmulo identificada foi de {melhor_janela} horas.")

# 5. Visualização Gráfica
plt.figure(figsize=(10, 5))
plt.bar(resultados_correlacao.keys(), resultados_correlacao.values(), color='skyblue')
plt.xlabel('Tamanho da Janela de Acúmulo (Horas)')
plt.ylabel('Coeficiente de Correlação (r)')
plt.title('Identificação da Janela Ótima de Carga Térmica')
plt.axvline(x=melhor_janela, color='red', linestyle='--', label='Melhor Janela')
plt.legend()
plt.show()
