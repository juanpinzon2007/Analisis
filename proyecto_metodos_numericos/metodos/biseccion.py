from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional

from utils.validaciones import validar_parametros_biseccion, validar_cambio_signo


@dataclass
class BiseccionResultado:
    raiz: float
    convergio: bool
    iteraciones: int
    historial: List[Dict[str, Any]]
    errores_abs: List[float]
    errores_rel: List[float]
    tiempo_seg: float


def biseccion(
        f: Callable[[float], float],
        a: float,
        b: float,
        tol: float = 1e-6,
        max_iter: int = 100,
) -> BiseccionResultado:
    """
    Método de bisección para hallar una raíz de f(x)=0 en [a,b].
    Guarda historial para tabla y listas de errores para gráficas.
    """
    validar_parametros_biseccion(a, b, tol, max_iter)

    t0 = time.perf_counter()

    fa = f(a)
    fb = f(b)
    validar_cambio_signo(fa, fb)

    historial: List[Dict[str, Any]] = []
    errores_abs: List[float] = []
    errores_rel: List[float] = []

    c_prev: Optional[float] = None
    convergio = False
    c = (a + b) / 2.0

    for n in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c)

        if c_prev is None:
            err_abs = float("inf")
            err_rel = float("inf")
        else:
            err_abs = abs(c - c_prev)
            err_rel = err_abs / abs(c) if c != 0 else float("inf")

        # Guardar fila (tabla)
        historial.append(
            {
                "n": n,
                "a": a,
                "b": b,
                "c": c,
                "fc": fc,
                "err_abs": err_abs,
                "err_rel": err_rel,
            }
        )
        errores_abs.append(err_abs)
        errores_rel.append(err_rel)

        # Criterios de parada
        if abs(fc) <= tol:
            convergio = True
            break
        if c_prev is not None and err_abs <= tol:
            convergio = True
            break

        # Actualizar intervalo
        # Si fc == 0 ya habría parado por abs(fc) <= tol (salvo tol muy chica)
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        c_prev = c

    t1 = time.perf_counter()
    return BiseccionResultado(
        raiz=c,
        convergio=convergio,
        iteraciones=len(historial),
        historial=historial,
        errores_abs=errores_abs,
        errores_rel=errores_rel,
        tiempo_seg=(t1 - t0),
    )
