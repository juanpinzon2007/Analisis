from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional


@dataclass
class PuntoFijoResultado:
    x_final: float
    convergio: bool
    iteraciones: int
    historial: List[Dict[str, Any]]
    errores_abs: List[float]
    errores_rel: List[float]
    tiempo_seg: float
    motivo_parada: str


def punto_fijo(
        g: Callable[[float], float],
        x0: float,
        tol: float = 1e-8,
        max_iter: int = 100,
        limite_divergencia: float = 1e6,
) -> PuntoFijoResultado:
    """
    Iteración de Punto Fijo:
      x_{n+1} = g(x_n)

    Tabla solicitada:
      n, x_n, g(x_n), |x_n - g(x_n)|, error relativo

    Criterios de parada:
      - |x_n - g(x_n)| <= tol
      - error relativo <= tol (si aplica)
      - max_iter
      - divergencia: |x_n| > limite_divergencia
    """
    if tol <= 0:
        raise ValueError("La tolerancia debe ser > 0.")
    if max_iter <= 0:
        raise ValueError("max_iter debe ser > 0.")

    t0 = time.perf_counter()

    historial: List[Dict[str, Any]] = []
    errores_abs: List[float] = []
    errores_rel: List[float] = []

    x_n = float(x0)
    convergio = False
    motivo = "Max iteraciones alcanzado"

    for n in range(1, max_iter + 1):
        gx = g(x_n)
        err_abs = abs(x_n - gx)
        err_rel = err_abs / abs(gx) if gx != 0 else float("inf")

        historial.append(
            {
                "n": n,
                "x_n": x_n,
                "gxn": gx,
                "abs_diff": err_abs,   # |x_n - g(x_n)|
                "err_rel": err_rel,
            }
        )
        errores_abs.append(err_abs)
        errores_rel.append(err_rel)

        # Parada por tolerancia
        if err_abs <= tol:
            convergio = True
            motivo = "Convergió por |x_n - g(x_n)| <= tol"
            x_n = gx
            break

        # Actualizar
        x_n = gx

        # Divergencia
        if abs(x_n) > limite_divergencia:
            convergio = False
            motivo = f"Divergencia detectada: |x_n| > {limite_divergencia}"
            break

    t1 = time.perf_counter()

    return PuntoFijoResultado(
        x_final=x_n,
        convergio=convergio,
        iteraciones=len(historial),
        historial=historial,
        errores_abs=errores_abs,
        errores_rel=errores_rel,
        tiempo_seg=(t1 - t0),
        motivo_parada=motivo,
    )