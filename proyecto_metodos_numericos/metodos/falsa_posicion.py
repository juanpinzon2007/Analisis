from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional

from utils.validaciones import validar_parametros_biseccion, validar_cambio_signo


@dataclass
class FalsaPosicionResultado:
    raiz: float
    convergio: bool
    iteraciones: int
    historial: List[Dict[str, Any]]
    errores_abs: List[float]
    errores_rel: List[float]
    tiempo_seg: float


def falsa_posicion(
        f: Callable[[float], float],
        a: float,
        b: float,
        tol: float = 1e-7,
        max_iter: int = 100,
) -> FalsaPosicionResultado:
    """
    Método de Falsa Posición (Regla Falsa) para f(x)=0 en [a,b].

    Fórmula:
        c = b - f(b)*(b-a)/(f(b)-f(a))

    Requisitos:
    - Debe haber cambio de signo f(a)*f(b) < 0
    - Manejar división por cero si f(b)-f(a)=0
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
    c = a

    for n in range(1, max_iter + 1):
        denom = (fb - fa)
        if denom == 0:
            raise ZeroDivisionError("División por cero: f(b) - f(a) = 0 en Falsa Posición.")

        c = b - fb * (b - a) / denom
        fc = f(c)

        if c_prev is None:
            err_abs = float("inf")
            err_rel = float("inf")
        else:
            err_abs = abs(c - c_prev)
            err_rel = err_abs / abs(c) if c != 0 else float("inf")

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

        if abs(fc) <= tol:
            convergio = True
            break
        if c_prev is not None and err_abs <= tol:
            convergio = True
            break

        # Actualizar intervalo conservando cambio de signo
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        c_prev = c

    t1 = time.perf_counter()
    return FalsaPosicionResultado(
        raiz=c,
        convergio=convergio,
        iteraciones=len(historial),
        historial=historial,
        errores_abs=errores_abs,
        errores_rel=errores_rel,
        tiempo_seg=(t1 - t0),
    )