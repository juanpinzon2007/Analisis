import math


def T_lambda(lmbda: float) -> float:
    """
    Función del ejercicio:
    T(λ) = 2.5 + 0.8λ^2 - 3.2λ + ln(λ + 1)
    """
    if lmbda <= -1:
        raise ValueError("Dominio inválido: λ debe ser > -1 para ln(λ+1).")
    return 2.5 + 0.8 * (lmbda ** 2) - 3.2 * lmbda + math.log(lmbda + 1)
def E_workers(x: float) -> float:
    """
    E(x) = x^3 - 6x^2 + 11x - 6.5
    x = número óptimo de workers activos
    """
    return (x ** 3) - 6 * (x ** 2) + 11 * x - 6.5

import math

def g_db(x: float) -> float:
    """
    g(x) = 0.5 cos(x) + 1.5
    """
    return 0.5 * math.cos(x) + 1.5


def g_db_deriv(x: float) -> float:
    """
    g'(x) = -0.5 sin(x)
    """
    return -0.5 * math.sin(x)
def T_threads(n: float) -> float:
    """
    T(n) = n^3 - 8n^2 + 20n - 16
    """
    return (n ** 3) - 8 * (n ** 2) + 20 * n - 16


def T_threads_deriv(n: float) -> float:
    """
    T'(n) = 3n^2 - 16n + 20
    """
    return 3 * (n ** 2) - 16 * n + 20
import math

def P_scaling(x: float) -> float:
    """
    P(x) = x*e^(-x/2) - 0.3
    """
    return x * math.exp(-x / 2.0) - 0.3


def P_scaling_deriv(x: float) -> float:
    """
    Derivada:
    d/dx [x e^{-x/2}] = e^{-x/2}(1 - x/2)
    """
    return math.exp(-x / 2.0) * (1.0 - x / 2.0)