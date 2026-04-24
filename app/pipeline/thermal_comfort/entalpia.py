import numpy as np

def calcular_entalpia(temp_c, ur_pct, p_local=1013.25):
    """
    Calcula a entalpia do ar (Carga Térmica) em kJ/kg.

    Parameters
    ----------
    temp_c : float
        Temperatura do ar em Celsius.
    ur_pct : float
        Umidade relativa em porcentagem (0-100).
    p_local : float, optional
        Pressão atmosférica local em hPa (padrão: 1013.25).

    Returns
    -------
    h : float
        Entalpia do ar em kJ/kg de ar seco.
    """
    # Pressão de saturação de vapor (Tetens)
    e_sat = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    # Pressão de vapor real
    e = (ur_pct / 100.0) * e_sat
    # Razão de mistura (w)
    w = 0.622 * (e / (p_local - e))
    
    # Entalpia
    h = 1.006 * temp_c + w * (2501 + 1.805 * temp_c)
    return h
