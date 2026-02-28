from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Dict, Any


@dataclass
class NewtonResultado:
    raiz: float
    convergio: bool
    iteraciones: int

    evaluaciones_f: int
    evaluaciones_df: int
    evaluaciones_total: int

    historial: List[Dict[str, Any]]
    errores_abs: List[float]
    errores_rel: List[float]
    tiempo_seg: float
    motivo_parada: str


def newton_raphson(
        f: Callable[[float], float],
        df: Callable[[float], float],
        x0: float,
        tol: float = 1e-10,
        max_iter: int = 100,
        limite_divergencia: float = 1e6,
) -> NewtonResultado:
    """
    Newton-Raphson:
        x_{n+1} = x_n - f(x_n)/f'(x_n)

    Historial (por iteración):
      n, x_n, f(x_n), f'(x_n), x_{n+1}, error_abs, error_rel

    Contadores:
      - evaluaciones_f: # de llamadas a f
      - evaluaciones_df: # de llamadas a df
      - evaluaciones_total: f + df
    """
    if tol <= 0:
        raise ValueError("La tolerancia debe ser > 0.")
    if max_iter <= 0:
        raise ValueError("max_iter debe ser > 0.")

    t0 = time.perf_counter()

    historial: List[Dict[str, Any]] = []
    errores_abs: List[float] = []
    errores_rel: List[float] = []

    evals_f = 0
    evals_df = 0

    x_n = float(x0)
    convergio = False
    motivo = "Max iteraciones alcanzado"

    for n in range(1, max_iter + 1):
        fx = f(x_n)
        evals_f += 1

        dfx = df(x_n)
        evals_df += 1

        if dfx == 0:
            motivo = "Derivada cero: f'(x_n)=0. Newton no puede continuar."
            break

        x_next = x_n - (fx / dfx)

        err_abs = abs(x_next - x_n)
        err_rel = err_abs / abs(x_next) if x_next != 0 else float("inf")

        historial.append(
            {
                "n": n,
                "x_n": x_n,
                "fx": fx,
                "dfx": dfx,
                "x_next": x_next,
                "err_abs": err_abs,
                "err_rel": err_rel,
            }
        )
        errores_abs.append(err_abs)
        errores_rel.append(err_rel)

        # Criterios de parada
        if abs(fx) <= tol:
            convergio = True
            motivo = "|f(x_n)| <= tol"
            x_n = x_next
            break

        if err_abs <= tol:
            convergio = True
            motivo = "|x_{n+1} - x_n| <= tol"
            x_n = x_next
            break

        x_n = x_next

        if abs(x_n) > limite_divergencia:
            convergio = False
            motivo = f"Divergencia detectada: |x_n| > {limite_divergencia}"
            break

    t1 = time.perf_counter()

    return NewtonResultado(
        raiz=x_n,
        convergio=convergio,
        iteraciones=len(historial),
        evaluaciones_f=evals_f,
        evaluaciones_df=evals_df,
        evaluaciones_total=(evals_f + evals_df),
        historial=historial,
        errores_abs=errores_abs,
        errores_rel=errores_rel,
        tiempo_seg=(t1 - t0),
        motivo_parada=motivo,
    )