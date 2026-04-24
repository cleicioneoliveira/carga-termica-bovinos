import numpy as np

def calculate_itu(temp_c, umidade_rel):
    """
    Calcula o Índice de Temperatura e Umidade (ITU).

    Parameters
    ----------
    temp_c : float or ndarray
        Temperatura do ar em graus Celsius.
    umidade_rel : float or ndarray
        Umidade relativa do ar em porcentagem (0-100).

    Returns
    -------
    itu : float or ndarray
        O valor do ITU calculado.
    """
    tf = (1.8 * temp_c) + 32
    ur_dec = umidade_rel / 100
    itu = tf - (0.55 - 0.55 * ur_dec) * (tf - 58)
    
    return itu
