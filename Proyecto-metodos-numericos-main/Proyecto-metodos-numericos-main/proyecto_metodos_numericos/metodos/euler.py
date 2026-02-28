from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional


@dataclass
class ODEResultado:
    metodo: str
    convergio: bool
    pasos: int
    historial: List[Dict[str, Any]]
    tiempo_seg: float
    motivo: str


def euler(
        f: Callable[[float, float], float],
        t0: float,
        y0: float,
        tf: float,
        h: float,
        limite_divergencia: float = 1e9,
) -> ODEResultado:
    """
    Euler explícito:
        y_{n+1} = y_n + h*f(t_n, y_n)
    """
    if h <= 0:
        raise ValueError("h debe ser > 0.")
    if tf <= t0:
        raise ValueError("tf debe ser mayor que t0.")

    t_start = time.perf_counter()

    n_steps = int((tf - t0) / h)
    if n_steps <= 0:
        raise ValueError("Con ese h no hay pasos (tf-t0)/h <= 0.")

    historial: List[Dict[str, Any]] = []
    t = float(t0)
    y = float(y0)

    historial.append({"n": 0, "t": t, "y": y})

    convergio = True
    motivo = "OK"

    for n in range(1, n_steps + 1):
        y_next = y + h * f(t, y)
        t_next = t + h

        if abs(y_next) > limite_divergencia:
            convergio = False
            motivo = f"Divergencia: |y| > {limite_divergencia}"
            y = y_next
            t = t_next
            historial.append({"n": n, "t": t, "y": y})
            break

        y, t = y_next, t_next
        historial.append({"n": n, "t": t, "y": y})

    t_end = time.perf_counter()
    return ODEResultado(
        metodo="Euler",
        convergio=convergio,
        pasos=len(historial) - 1,
        historial=historial,
        tiempo_seg=t_end - t_start,
        motivo=motivo,
    )