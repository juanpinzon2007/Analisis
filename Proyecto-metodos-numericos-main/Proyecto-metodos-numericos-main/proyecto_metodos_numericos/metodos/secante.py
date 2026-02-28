from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Dict, Any


@dataclass
class SecanteResultado:
    raiz: float
    convergio: bool
    iteraciones: int
    evaluaciones_f: int
    historial: List[Dict[str, Any]]
    errores_abs: List[float]
    tiempo_seg: float
    motivo_parada: str


def secante(
        f: Callable[[float], float],
        x0: float,
        x1: float,
        tol: float = 1e-9,
        max_iter: int = 100,
        limite_divergencia: float = 1e9,
) -> SecanteResultado:
    """
    Método de la secante:
      x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / ( f(x_n) - f(x_{n-1}) )

    Tabla requerida:
      n, x_{n-1}, x_n, f(x_{n-1}), f(x_n), x_{n+1}, error

    Validación:
      f(x_n) - f(x_{n-1}) != 0
    """
    if tol <= 0:
        raise ValueError("La tolerancia debe ser > 0.")
    if max_iter <= 0:
        raise ValueError("max_iter debe ser > 0.")
    if x0 == x1:
        raise ValueError("x0 y x1 deben ser diferentes.")

    t0 = time.perf_counter()

    historial: List[Dict[str, Any]] = []
    errores_abs: List[float] = []
    evals_f = 0

    x_prev = float(x0)
    x_curr = float(x1)

    f_prev = f(x_prev); evals_f += 1
    f_curr = f(x_curr); evals_f += 1

    convergio = False
    motivo = "Max iter alcanzado"
    x_next = x_curr

    for n in range(1, max_iter + 1):
        denom = (f_curr - f_prev)
        if denom == 0:
            motivo = "División por cero: f(x_n) - f(x_{n-1}) = 0"
            break

        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        err_abs = abs(x_next - x_curr)

        historial.append(
            {
                "n": n,
                "x_prev": x_prev,
                "x_curr": x_curr,
                "f_prev": f_prev,
                "f_curr": f_curr,
                "x_next": x_next,
                "err_abs": err_abs,
            }
        )
        errores_abs.append(err_abs)

        # criterio de parada
        if err_abs <= tol:
            convergio = True
            motivo = "|x_{n+1} - x_n| <= tol"
            x_curr = x_next
            break

        # avanzar
        x_prev, f_prev = x_curr, f_curr
        x_curr = x_next
        if abs(x_curr) > limite_divergencia:
            convergio = False
            motivo = f"Divergencia: |x_n| > {limite_divergencia}"
            break

        f_curr = f(x_curr); evals_f += 1

    t1 = time.perf_counter()

    return SecanteResultado(
        raiz=float(x_curr),
        convergio=convergio,
        iteraciones=len(historial),
        evaluaciones_f=evals_f,
        historial=historial,
        errores_abs=errores_abs,
        tiempo_seg=(t1 - t0),
        motivo_parada=motivo,
    )