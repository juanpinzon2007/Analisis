import unittest

from funciones.definiciones import T_lambda
from metodos.biseccion import biseccion


class TestBiseccion(unittest.TestCase):
    def test_debe_converger_en_intervalo_guia(self):
        res = biseccion(T_lambda, 0.5, 2.5, tol=1e-6, max_iter=100)
        self.assertTrue(res.iteraciones <= 100)
        self.assertTrue(res.convergio)

        # Verificar que la raíz encontrada hace T(λ) cerca de 0
        self.assertLessEqual(abs(T_lambda(res.raiz)), 1e-6)

    def test_error_si_no_hay_cambio_signo(self):
        # Elegimos un intervalo donde es muy probable que no cambie signo (puede ajustarse si fuese necesario)
        with self.assertRaises(ValueError):
            biseccion(T_lambda, 0.5, 0.6, tol=1e-6, max_iter=50)


if __name__ == "__main__":
    unittest.main()
