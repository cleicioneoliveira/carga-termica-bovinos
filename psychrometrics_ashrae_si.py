"""
psychrometrics_ashrae_si.py

Funções psicrométricas em unidades SI, baseadas nas equações do ASHRAE
(implementação compatível com a formulação usada pelo PsychroLib).

Convenções:
- Temperatura: °C
- Pressão: Pa
- Entalpia: J / kg_ar_seco
- Razão de umidade (humidity ratio): kg_vapor / kg_ar_seco
- Umidade específica (specific humidity): kg_vapor / kg_ar_umido

Observação:
As equações de pressão de saturação são definidas em dois ramos, abaixo e
acima do ponto triplo da água, como prática robusta para evitar problemas
numéricos perto de 0 °C.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


MIN_HUMIDITY_RATIO: float = 1e-7
ZERO_CELSIUS_AS_KELVIN: float = 273.15
TRIPLE_POINT_WATER_C: float = 0.01
R_DA_SI: float = 287.042  # J / (kg_da * K)
EPSILON: float = 0.621945  # razão entre massas molares do vapor d'água e do ar seco


class PsychrometricError(ValueError):
    """Erro de validação para entradas psicrométricas inválidas."""


@dataclass(frozen=True, slots=True)
class PsychrometricState:
    """
    Estado psicrométrico básico.

    Attributes:
        dry_bulb_c: Temperatura de bulbo seco em °C.
        pressure_pa: Pressão atmosférica em Pa.
        humidity_ratio: Razão de umidade em kg_vapor / kg_ar_seco.
    """

    dry_bulb_c: float
    pressure_pa: float
    humidity_ratio: float

    @property
    def specific_humidity(self) -> float:
        """Retorna a umidade específica q em kg_vapor / kg_ar_umido."""
        return specific_humidity_from_humidity_ratio(self.humidity_ratio)

    @property
    def enthalpy_j_per_kg_da(self) -> float:
        """Retorna a entalpia do ar úmido em J/kg de ar seco."""
        return enthalpy_moist_air(self.dry_bulb_c, self.humidity_ratio)

    @property
    def relative_humidity(self) -> float:
        """Retorna a umidade relativa em base 0..1."""
        return relative_humidity_from_t_w(
            dry_bulb_c=self.dry_bulb_c,
            humidity_ratio=self.humidity_ratio,
            pressure_pa=self.pressure_pa,
        )


def _validate_temperature_c(dry_bulb_c: float) -> None:
    """
    Valida temperatura para a faixa típica da equação ASHRAE em SI.

    Args:
        dry_bulb_c: Temperatura em °C.
    """
    if dry_bulb_c < -100.0 or dry_bulb_c > 200.0:
        raise PsychrometricError(
            "Temperatura fora da faixa suportada para esta implementação "
            "(-100 °C a 200 °C)."
        )


def _validate_pressure_pa(pressure_pa: float) -> None:
    """
    Valida pressão atmosférica.

    Args:
        pressure_pa: Pressão em Pa.
    """
    if pressure_pa <= 0.0:
        raise PsychrometricError("A pressão deve ser maior que zero.")


def _validate_rh(relative_humidity: float) -> None:
    """
    Valida umidade relativa.

    Args:
        relative_humidity: Umidade relativa na faixa [0, 1].
    """
    if not (0.0 <= relative_humidity <= 1.0):
        raise PsychrometricError("A umidade relativa deve estar entre 0 e 1.")


def _validate_vapor_pressure_pa(vapor_pressure_pa: float) -> None:
    """
    Valida pressão parcial de vapor.

    Args:
        vapor_pressure_pa: Pressão parcial do vapor em Pa.
    """
    if vapor_pressure_pa < 0.0:
        raise PsychrometricError("A pressão parcial de vapor não pode ser negativa.")


def _validate_humidity_ratio(humidity_ratio: float) -> None:
    """
    Valida razão de umidade.

    Args:
        humidity_ratio: Razão de umidade em kg_vapor / kg_ar_seco.
    """
    if humidity_ratio < 0.0:
        raise PsychrometricError("A razão de umidade não pode ser negativa.")


def _kelvin_from_celsius(temperature_c: float) -> float:
    """
    Converte °C para K.

    Args:
        temperature_c: Temperatura em °C.

    Returns:
        Temperatura em K.
    """
    return temperature_c + ZERO_CELSIUS_AS_KELVIN


def es(dry_bulb_c: float) -> float:
    """
    Retorna a pressão de saturação do vapor d'água em Pa.

    Esta é a função normalmente denotada por es(T) ou p_ws(T).

    Args:
        dry_bulb_c: Temperatura de bulbo seco em °C.

    Returns:
        Pressão de saturação em Pa.
    """
    _validate_temperature_c(dry_bulb_c)

    temperature_k = _kelvin_from_celsius(dry_bulb_c)

    if dry_bulb_c <= TRIPLE_POINT_WATER_C:
        ln_pws = (
            -5.6745359e3 / temperature_k
            + 6.3925247
            - 9.677843e-3 * temperature_k
            + 6.2215701e-7 * temperature_k**2
            + 2.0747825e-9 * temperature_k**3
            - 9.484024e-13 * temperature_k**4
            + 4.1635019 * math.log(temperature_k)
        )
    else:
        ln_pws = (
            -5.8002206e3 / temperature_k
            + 1.3914993
            - 4.8640239e-2 * temperature_k
            + 4.1764768e-5 * temperature_k**2
            - 1.4452093e-8 * temperature_k**3
            + 6.5459673 * math.log(temperature_k)
        )

    return math.exp(ln_pws)


def vapor_pressure_from_rh(dry_bulb_c: float, relative_humidity: float) -> float:
    """
    Calcula a pressão parcial de vapor a partir de temperatura e UR.

    Args:
        dry_bulb_c: Temperatura de bulbo seco em °C.
        relative_humidity: Umidade relativa na faixa [0, 1].

    Returns:
        Pressão parcial de vapor em Pa.
    """
    _validate_rh(relative_humidity)
    return relative_humidity * es(dry_bulb_c)


def humidity_ratio_from_vapor_pressure(
    vapor_pressure_pa: float,
    pressure_pa: float,
) -> float:
    """
    Retorna a razão de umidade w a partir da pressão parcial de vapor.

    Fórmula:
        w = EPSILON * pv / (p - pv)

    Args:
        vapor_pressure_pa: Pressão parcial do vapor em Pa.
        pressure_pa: Pressão total em Pa.

    Returns:
        Razão de umidade w em kg_vapor / kg_ar_seco.
    """
    _validate_vapor_pressure_pa(vapor_pressure_pa)
    _validate_pressure_pa(pressure_pa)

    if vapor_pressure_pa >= pressure_pa:
        raise PsychrometricError(
            "A pressão parcial de vapor deve ser menor que a pressão total."
        )

    humidity_ratio = EPSILON * vapor_pressure_pa / (pressure_pa - vapor_pressure_pa)
    return max(humidity_ratio, MIN_HUMIDITY_RATIO)


def vapor_pressure_from_humidity_ratio(
    humidity_ratio: float,
    pressure_pa: float,
) -> float:
    """
    Retorna a pressão parcial de vapor a partir da razão de umidade.

    Fórmula invertida:
        pv = p * w / (EPSILON + w)

    Args:
        humidity_ratio: Razão de umidade w em kg_vapor / kg_ar_seco.
        pressure_pa: Pressão total em Pa.

    Returns:
        Pressão parcial de vapor em Pa.
    """
    _validate_humidity_ratio(humidity_ratio)
    _validate_pressure_pa(pressure_pa)

    bounded_w = max(humidity_ratio, MIN_HUMIDITY_RATIO)
    return pressure_pa * bounded_w / (EPSILON + bounded_w)


def humidity_ratio_from_t_rh(
    dry_bulb_c: float,
    relative_humidity: float,
    pressure_pa: float = 101_325.0,
) -> float:
    """
    Retorna a razão de umidade a partir de temperatura, UR e pressão.

    Args:
        dry_bulb_c: Temperatura de bulbo seco em °C.
        relative_humidity: Umidade relativa na faixa [0, 1].
        pressure_pa: Pressão atmosférica em Pa.

    Returns:
        Razão de umidade w em kg_vapor / kg_ar_seco.
    """
    pv = vapor_pressure_from_rh(dry_bulb_c, relative_humidity)
    return humidity_ratio_from_vapor_pressure(pv, pressure_pa)


def specific_humidity_from_humidity_ratio(humidity_ratio: float) -> float:
    """
    Converte razão de umidade w em umidade específica q.

    Fórmula:
        q = w / (1 + w)

    Args:
        humidity_ratio: Razão de umidade w em kg_vapor / kg_ar_seco.

    Returns:
        Umidade específica q em kg_vapor / kg_ar_umido.
    """
    _validate_humidity_ratio(humidity_ratio)
    bounded_w = max(humidity_ratio, MIN_HUMIDITY_RATIO)
    return bounded_w / (1.0 + bounded_w)


def humidity_ratio_from_specific_humidity(specific_humidity: float) -> float:
    """
    Converte umidade específica q em razão de umidade w.

    Fórmula:
        w = q / (1 - q)

    Args:
        specific_humidity: Umidade específica na faixa [0, 1).

    Returns:
        Razão de umidade w em kg_vapor / kg_ar_seco.
    """
    if not (0.0 <= specific_humidity < 1.0):
        raise PsychrometricError("A umidade específica deve estar no intervalo [0, 1).")

    humidity_ratio = specific_humidity / (1.0 - specific_humidity)
    return max(humidity_ratio, MIN_HUMIDITY_RATIO)


def saturation_humidity_ratio(
    dry_bulb_c: float,
    pressure_pa: float = 101_325.0,
) -> float:
    """
    Retorna a razão de umidade de saturação na temperatura e pressão dadas.

    Args:
        dry_bulb_c: Temperatura em °C.
        pressure_pa: Pressão atmosférica em Pa.

    Returns:
        Razão de umidade de saturação em kg_vapor / kg_ar_seco.
    """
    p_ws = es(dry_bulb_c)
    if p_ws >= pressure_pa:
        raise PsychrometricError(
            "A pressão de saturação excede ou iguala a pressão total."
        )

    w_sat = EPSILON * p_ws / (pressure_pa - p_ws)
    return max(w_sat, MIN_HUMIDITY_RATIO)


def relative_humidity_from_t_w(
    dry_bulb_c: float,
    humidity_ratio: float,
    pressure_pa: float = 101_325.0,
) -> float:
    """
    Retorna a umidade relativa a partir de temperatura, razão de umidade e pressão.

    Args:
        dry_bulb_c: Temperatura em °C.
        humidity_ratio: Razão de umidade em kg_vapor / kg_ar_seco.
        pressure_pa: Pressão atmosférica em Pa.

    Returns:
        Umidade relativa na faixa [0, 1].
    """
    pv = vapor_pressure_from_humidity_ratio(humidity_ratio, pressure_pa)
    p_ws = es(dry_bulb_c)

    rh = pv / p_ws
    return max(0.0, min(1.0, rh))


def enthalpy_moist_air(dry_bulb_c: float, humidity_ratio: float) -> float:
    """
    Retorna a entalpia do ar úmido em J/kg de ar seco.

    Fórmula SI:
        h = (1.006*T + w*(2501 + 1.86*T)) * 1000

    com T em °C.

    Args:
        dry_bulb_c: Temperatura de bulbo seco em °C.
        humidity_ratio: Razão de umidade em kg_vapor / kg_ar_seco.

    Returns:
        Entalpia em J/kg de ar seco.
    """
    _validate_humidity_ratio(humidity_ratio)
    bounded_w = max(humidity_ratio, MIN_HUMIDITY_RATIO)
    return (1.006 * dry_bulb_c + bounded_w * (2501.0 + 1.86 * dry_bulb_c)) * 1000.0


def dry_air_enthalpy(dry_bulb_c: float) -> float:
    """
    Retorna a entalpia do ar seco em J/kg.

    Fórmula SI:
        h_da = 1.006 * T * 1000

    Args:
        dry_bulb_c: Temperatura em °C.

    Returns:
        Entalpia do ar seco em J/kg.
    """
    return 1.006 * dry_bulb_c * 1000.0


def dew_point_from_vapor_pressure(vapor_pressure_pa: float) -> float:
    """
    Calcula a temperatura de ponto de orvalho em °C a partir da pressão parcial de vapor.

    Usa bisseção sobre a função de saturação es(T), o que mantém a implementação
    consistente com a própria formulação ASHRAE adotada neste módulo.

    Args:
        vapor_pressure_pa: Pressão parcial de vapor em Pa.

    Returns:
        Ponto de orvalho em °C.
    """
    _validate_vapor_pressure_pa(vapor_pressure_pa)

    if vapor_pressure_pa == 0.0:
        raise PsychrometricError(
            "Ponto de orvalho indefinido para pressão parcial de vapor igual a zero."
        )

    low = -100.0
    high = 200.0

    for _ in range(100):
        mid = 0.5 * (low + high)
        p_mid = es(mid)

        if abs(p_mid - vapor_pressure_pa) <= 1e-6 * max(vapor_pressure_pa, 1.0):
            return mid

        if p_mid > vapor_pressure_pa:
            high = mid
        else:
            low = mid

    return 0.5 * (low + high)


def dew_point_from_t_rh(dry_bulb_c: float, relative_humidity: float) -> float:
    """
    Calcula o ponto de orvalho em °C a partir de temperatura e umidade relativa.

    Args:
        dry_bulb_c: Temperatura de bulbo seco em °C.
        relative_humidity: Umidade relativa na faixa [0, 1].

    Returns:
        Ponto de orvalho em °C.
    """
    pv = vapor_pressure_from_rh(dry_bulb_c, relative_humidity)
    return dew_point_from_vapor_pressure(pv)


def moist_air_density(
    dry_bulb_c: float,
    humidity_ratio: float,
    pressure_pa: float = 101_325.0,
) -> float:
    """
    Retorna a densidade do ar úmido em kg/m³.

    Fórmula SI:
        v = R_da * T * (1 + 1.607858*w) / p
        rho = (1 + w) / v

    onde:
        v é o volume específico por kg de ar seco [m³/kg_da].

    Args:
        dry_bulb_c: Temperatura em °C.
        humidity_ratio: Razão de umidade em kg_vapor / kg_ar_seco.
        pressure_pa: Pressão atmosférica em Pa.

    Returns:
        Densidade do ar úmido em kg/m³.
    """
    _validate_humidity_ratio(humidity_ratio)
    _validate_pressure_pa(pressure_pa)

    temperature_k = _kelvin_from_celsius(dry_bulb_c)
    bounded_w = max(humidity_ratio, MIN_HUMIDITY_RATIO)

    specific_volume = (
        R_DA_SI * temperature_k * (1.0 + 1.607858 * bounded_w) / pressure_pa
    )
    return (1.0 + bounded_w) / specific_volume


def standard_atmosphere_pressure(altitude_m: float) -> float:
    """
    Retorna a pressão atmosférica padrão em função da altitude.

    Fórmula padrão usada em psicrometria HVAC:
        p = 101325 * (1 - 2.25577e-5 * z) ** 5.2559

    Args:
        altitude_m: Altitude em metros.

    Returns:
        Pressão atmosférica estimada em Pa.
    """
    if altitude_m < -500.0 or altitude_m > 20_000.0:
        raise PsychrometricError(
            "Altitude fora da faixa suportada para esta aproximação."
        )

    return 101_325.0 * (1.0 - 2.25577e-5 * altitude_m) ** 5.2559


def state_from_t_rh(
    dry_bulb_c: float,
    relative_humidity: float,
    pressure_pa: float = 101_325.0,
) -> PsychrometricState:
    """
    Monta um estado psicrométrico a partir de temperatura, UR e pressão.

    Args:
        dry_bulb_c: Temperatura de bulbo seco em °C.
        relative_humidity: Umidade relativa na faixa [0, 1].
        pressure_pa: Pressão atmosférica em Pa.

    Returns:
        Instância de PsychrometricState.
    """
    humidity_ratio = humidity_ratio_from_t_rh(
        dry_bulb_c=dry_bulb_c,
        relative_humidity=relative_humidity,
        pressure_pa=pressure_pa,
    )
    return PsychrometricState(
        dry_bulb_c=dry_bulb_c,
        pressure_pa=pressure_pa,
        humidity_ratio=humidity_ratio,
    )


if __name__ == "__main__":
    pressure = 101_325.0
    temperature = 25.0
    relative_humidity = 0.60

    state = state_from_t_rh(
        dry_bulb_c=temperature,
        relative_humidity=relative_humidity,
        pressure_pa=pressure,
    )

    print("=== Exemplo psicrométrico ===")
    print(f"Tbs                 : {temperature:.2f} °C")
    print(f"UR                  : {relative_humidity * 100:.2f} %")
    print(f"Pressão             : {pressure:.2f} Pa")
    print(f"es(T)               : {es(temperature):.2f} Pa")
    print(f"w                   : {state.humidity_ratio:.8f} kg/kg_ar_seco")
    print(f"q                   : {state.specific_humidity:.8f} kg/kg_ar_umido")
    print(f"h                   : {state.enthalpy_j_per_kg_da:.2f} J/kg_ar_seco")
    print(f"Tpo                 : {dew_point_from_t_rh(temperature, relative_humidity):.2f} °C")
    print(f"rho ar úmido        : {moist_air_density(temperature, state.humidity_ratio, pressure):.5f} kg/m³")
