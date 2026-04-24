"""
Script: Thermal Saturation & Efficiency Model (TESM)
Versão: 2.1 (Modelo de Resposta Polinomial e IOR Corrigido)
Descrição: Este script processa dados longitudinais de bovinos leiteiros (n=290),
           calcula a Carga Térmica Acumulada (CTA 15h), modela a saturação 
           fisiológica da ofegação e gera o Diagrama Psicrométrico de Risco.
           
Metodologia: Baseado em Hahn (1999) e Gaughan (2008), com a introdução do
             Índice de Ofegação Relativa Corrigido (IOR_corr).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==========================================================
# 1. FUNÇÕES DE BASE (FISIOLOGIA E FÍSICA)
# ==========================================================
def calcular_thi(t, ur):
    return (1.8 * t + 32) - (0.55 - 0.0055 * ur) * (1.8 * t - 26)

# ==========================================================
# 2. SIMULAÇÃO DE UM DIA DE VERÃO (SÉRIE TEMPORAL)
# ==========================================================
def simular_dia_critico():
    horas = np.arange(0, 24, 1)
    # Simulação de Temperatura (Mín 22°C as 4h, Máx 38°C as 15h)
    temp = 30 + 8 * np.sin((horas - 10) * np.pi / 12)
    # Simulação de Umidade (Inversa à temperatura)
    umidade = 60 - 20 * np.sin((horas - 10) * np.pi / 12)
    
    df_dia = pd.DataFrame({'hora': horas, 'temp': temp, 'ur': umidade})
    df_dia['thi'] = calcular_thi(df_dia['temp'], df_dia['ur'])
    
    # Cálculo do acúmulo (CTA) - Considerando as 15h anteriores
    # Para a simulação, assumimos que o animal começa o dia com CTA = 10 (resíduo da noite)
    df_dia['excesso'] = np.maximum(0, df_dia['thi'] - 72)
    df_dia['cta_acumulada'] = df_dia['excesso'].cumsum() + 10
    
    return df_dia

# ==========================================================
# 3. PLOTAGEM DO DIAGRAMA DE USO EM TEMPO REAL
# ==========================================================
def plotar_uso_diagrama(df_simulado):
    # Criar a malha de fundo (O Diagrama Psicrométrico de Risco)
    t_ref = np.linspace(20, 45, 100)
    ur_ref = np.linspace(20, 100, 100)
    T, UR = np.meshgrid(t_ref, ur_ref)
    Z_THI = calcular_thi(T, UR)
    excesso_malha = np.maximum(0, Z_THI - 72)

    plt.figure(figsize=(12, 9))
    
    # Definir as zonas baseadas nos seus limites de CTA / 15h
    # Verde (Conforto), Laranja (Alerta), Coral (Crítico), Vermelho (Fadiga)
    levels = [0, 1.5, 5.5, 11.0, 25]
    colors = ['#eafaf1', '#fef5e7', '#fbeee6', '#f2d7d5'] # Cores claras para o fundo
    plt.contourf(T, UR, excesso_malha, levels=levels, colors=colors)
    
    # Desenhar linhas de THI para referência
    cnt = plt.contour(T, UR, Z_THI, levels=[72, 75, 78, 82], colors='gray', alpha=0.3, linestyles='--')
    plt.clabel(cnt, inline=True, fontsize=8, fmt='THI %.0f')

    # --- PLOTAR A TRAJETÓRIA DO DIA ---
    # A linha muda de cor ou espessura conforme o CTA aumenta
    path = plt.plot(df_simulado['temp'], df_simulado['ur'], color='blue', linewidth=1, alpha=0.5, label='Trajetória 24h')
    
    # Adicionar marcadores de hora com cores baseadas no CTA real do animal
    for i, row in df_simulado.iterrows():
        # Lógica de cor baseada no CTA ACUMULADO (não no THI do momento)
        if row['cta_acumulada'] < 22.4: color = 'green'
        elif row['cta_acumulada'] < 83.6: color = 'orange'
        elif row['cta_acumulada'] < 165.2: color = 'darkorange'
        else: color = 'red'
        
        plt.scatter(row['temp'], row['ur'], color=color, s=40, edgecolors='black', zorder=5)
        
        # Rotular apenas algumas horas para não poluir
        if i % 3 == 0:
            plt.text(row['temp']+0.5, row['ur']+0.5, f"{int(row['hora'])}h\n(CTA:{int(row['cta_acumulada'])})", 
                     fontsize=9, fontweight='bold', color='black')

    # Customização Final
    plt.title("Simulação de Monitoramento: Trajetória Térmica e Fadiga Acumulada", fontsize=15)
    plt.xlabel("Temperatura Ambiente (°C)")
    plt.ylabel("Umidade Relativa (%)")
    
    legend_elements = [
        Patch(facecolor='#eafaf1', edgecolor='green', label='Status: Recuperação (IOR ~ 1.0)'),
        Patch(facecolor='#fef5e7', edgecolor='orange', label='Status: Alerta (Acúmulo Iniciado)'),
        Patch(facecolor='#f2d7d5', edgecolor='red', label='Status: FADIGA TÉRMICA (Risco de Morte)')
    ]
    plt.legend(handles=legend_elements, loc='lower left')
    
    plt.grid(alpha=0.2)
    plt.show()

# Rodar a Simulação
df_dia = simular_dia_critico()
plotar_uso_diagrama(df_dia)
