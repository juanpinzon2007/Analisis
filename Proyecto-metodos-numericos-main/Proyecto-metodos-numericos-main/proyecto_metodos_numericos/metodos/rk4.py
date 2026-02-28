from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Dict, Any


@dataclass
class ODEResultado:
    metodo: str
    convergio: bool
    pasos: int
    historial: List[Dict[str, Any]]
    tiempo_seg: float
    motivo: str


def rk4(
        f: Callable[[float, float], float],
        t0: float,
        y0: float,
        tf: float,
        h: float,
        limite_divergencia: float = 1e9,
) -> ODEResultado:
    """
    Runge-Kutta de 4to orden:
        k1 = f(t, y)
        k2 = f(t + h/2, y + h*k1/2)
        k3 = f(t + h/2, y + h*k2/2)
        k4 = f(t + h,   y + h*k3)
        y_{n+1} = y_n + h*(k1 + 2k2 + 2k3 + k4)/6
    """
    if h <= 0:
        raise ValueError("h debe ser > 0.")
    if tf <= t0:
        raise ValueError("tf debe ser mayor que t0.")

    t_start = time.perf_counter()

    n_steps = int((tf - t0) / h)
    if n_steps <= 0:
        raise ValueError("Con ese h no hay pasos (tf-t0)/h <= 0.")

    historial = []
    t = float(t0)
    y = float(y0)

    historial.append({"n": 0, "t": t, "y": y})

    convergio = True
    motivo = "OK"

    for n in range(1, n_steps + 1):
        k1 = f(t, y)
        k2 = f(t + h / 2.0, y + (h * k1) / 2.0)
        k3 = f(t + h / 2.0, y + (h * k2) / 2.0)
        k4 = f(t + h, y + h * k3)

        y_next = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
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
        metodo="RK4",
        convergio=convergio,
        pasos=len(historial) - 1,
        historial=historial,
        tiempo_seg=t_end - t_start,
        motivo=motivo,
    )