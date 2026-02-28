def validar_parametros_biseccion(a: float, b: float, tol: float, max_iter: int) -> None:
    if tol <= 0:
        raise ValueError("La tolerancia debe ser mayor que 0.")
    if max_iter <= 0:
        raise ValueError("El número máximo de iteraciones debe ser mayor que 0.")
    if a >= b:
        raise ValueError("El intervalo es inválido: a debe ser menor que b.")


def validar_cambio_signo(fa: float, fb: float) -> None:
    if fa == 0:
        # raíz exacta en a
        return
    if fb == 0:
        # raíz exacta en b
        return
    if fa * fb > 0:
        raise ValueError("No hay cambio de signo en [a, b]. Bisección requiere f(a)*f(b) < 0.")
