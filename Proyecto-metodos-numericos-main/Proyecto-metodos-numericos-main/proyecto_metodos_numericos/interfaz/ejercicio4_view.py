import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from funciones.definiciones import T_threads, T_threads_deriv
from metodos.newton_raphson import newton_raphson


class Ejercicio4View(ttk.Frame):
    """
    Ejercicio 4 - Newton-Raphson
    - Tabla: n, x_n, f(x_n), f'(x_n), error abs, error rel
    - Gráfica: f(x) + tangentes en cada iteración
    - Gráfica: error en escala log
    - Probar diferentes x0: 1.0, 2.0, 3.0, 5.0
    """

    X0_PREDETERMINADOS = [1.0, 2.0, 3.0, 5.0]

    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self._build_ui()
        self._plot_inicial()

    # ------------------ UI ------------------

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        right = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="nsew")
        right.grid(row=0, column=1, sticky="nsew")

        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)

        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)

        # ---- Parámetros
        input_frame = ttk.LabelFrame(left, text="Ejercicio 4 - Newton-Raphson: T(n)=0", padding=10)
        input_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)

        self.var_x0 = tk.StringVar(value="3.0")
        self.var_tol = tk.StringVar(value="1e-10")
        self.var_iter = tk.StringVar(value="100")

        ttk.Label(input_frame, text="n0 (x0):").grid(row=0, column=0, sticky="w")
        self.cmb_x0 = ttk.Combobox(
            input_frame,
            textvariable=self.var_x0,
            values=[str(x) for x in self.X0_PREDETERMINADOS],
            width=12,
            state="readonly",
        )
        self.cmb_x0.grid(row=0, column=1, sticky="w", padx=(5, 0))

        ttk.Label(input_frame, text="Tolerancia:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(input_frame, textvariable=self.var_tol, width=15).grid(row=1, column=1, sticky="w", padx=(5, 0), pady=(8, 0))

        ttk.Label(input_frame, text="Max iter:").grid(row=1, column=2, sticky="w", padx=(15, 0), pady=(8, 0))
        ttk.Entry(input_frame, textvariable=self.var_iter, width=15).grid(row=1, column=3, sticky="w", padx=(5, 0), pady=(8, 0))

        btns = ttk.Frame(input_frame)
        btns.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        ttk.Button(btns, text="Calcular x0 seleccionado", command=self.on_calcular).pack(side="left")
        ttk.Button(btns, text="Comparar x0 (1,2,3,5)", command=self.on_comparar).pack(side="left", padx=(10, 0))
        ttk.Button(btns, text="Limpiar", command=self.on_limpiar).pack(side="left", padx=(10, 0))

        # ---- Resumen comparación
        resumen_frame = ttk.LabelFrame(left, text="Resumen comparativo (x0)", padding=10)
        resumen_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        resumen_frame.columnconfigure(0, weight=1)

        self.resumen = ttk.Treeview(
            resumen_frame,
            columns=("x0", "estado", "raiz", "iter", "err_final", "tiempo"),
            show="headings",
            height=4,
        )
        for col, w in [("x0", 80), ("estado", 100), ("raiz", 140), ("iter", 80), ("err_final", 150), ("tiempo", 110)]:
            self.resumen.heading(col, text=col)
            self.resumen.column(col, width=w, anchor="center")
        self.resumen.grid(row=0, column=0, sticky="ew")

        # ---- Tabla iteraciones
        table_frame = ttk.LabelFrame(left, text="Tabla de iteraciones (Newton)", padding=10)
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        cols = ("n", "x_n", "f(x_n)", "f'(x_n)", "err_abs", "err_rel")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=16)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor="center", width=125)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # ---- Gráficas
        charts_frame = ttk.LabelFrame(right, text="Gráficas", padding=10)
        charts_frame.grid(row=0, column=0, sticky="nsew")
        charts_frame.rowconfigure(0, weight=1)
        charts_frame.columnconfigure(0, weight=1)

        # ✅ figura más alta + redraw pro
        self.fig = Figure(figsize=(6, 7.5), dpi=100)
        self.ax_func = self.fig.add_subplot(2, 1, 1)
        self.ax_err = self.fig.add_subplot(2, 1, 2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=charts_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # ---- Análisis
        result_frame = ttk.LabelFrame(right, text="Análisis", padding=10)
        result_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        result_frame.columnconfigure(0, weight=1)

        self.var_analysis = tk.StringVar(value="Sin calcular.")
        ttk.Label(result_frame, textvariable=self.var_analysis, justify="left").grid(row=0, column=0, sticky="w")

    # ------------------ Matplotlib layout pro ------------------

    def _redraw(self):
        self.fig.tight_layout(pad=2.0)
        self.fig.subplots_adjust(hspace=0.60)
        self.canvas.draw()

    # ------------------ Acciones ------------------

    def on_limpiar(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for item in self.resumen.get_children():
            self.resumen.delete(item)

        self.ax_func.clear()
        self.ax_err.clear()
        self.var_analysis.set("Sin calcular.")
        self._plot_inicial()

    def on_calcular(self):
        try:
            x0 = float(self.var_x0.get().strip())
            tol = float(self.var_tol.get().strip())
            max_iter = int(self.var_iter.get().strip())

            # limpiar tabla
            for item in self.tree.get_children():
                self.tree.delete(item)

            res = newton_raphson(T_threads, T_threads_deriv, x0=x0, tol=tol, max_iter=max_iter)

            for row in res.historial:
                self.tree.insert(
                    "",
                    "end",
                    values=(
                        row["n"],
                        f"{row['x_n']:.12f}",
                        f"{row['fx']:.3e}",
                        f"{row['dfx']:.3e}",
                        f"{row['err_abs']:.3e}",
                        f"{row['err_rel']:.3e}",
                    ),
                )

            self._plot_func_y_tangentes(res)
            self._plot_error(res)

            estado = "Convergió" if res.convergio else "No convergió"
            err_final = res.errores_abs[-1] if res.errores_abs else float("nan")

            self.var_analysis.set(
                f"Estado: {estado}\n"
                f"x0: {x0}\n"
                f"Raíz aproximada n*: {res.raiz:.12f}\n"
                f"Iteraciones: {res.iteraciones}\n"
                f"Error final: {err_final:.3e}\n"
                f"Motivo: {res.motivo_parada}\n"
                f"Tiempo: {res.tiempo_seg:.6f} s\n\n"
                f"Interpretación: n representa el número óptimo de threads donde T(n)=0 según el modelo."
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_comparar(self):
        try:
            tol = float(self.var_tol.get().strip())
            max_iter = int(self.var_iter.get().strip())

            for item in self.resumen.get_children():
                self.resumen.delete(item)

            resultados = {}
            for x0 in self.X0_PREDETERMINADOS:
                res = newton_raphson(T_threads, T_threads_deriv, x0=x0, tol=tol, max_iter=max_iter)
                resultados[x0] = res

                estado = "OK" if res.convergio else "FAIL"
                err_final = res.errores_abs[-1] if res.errores_abs else float("nan")
                self.resumen.insert(
                    "",
                    "end",
                    values=(f"{x0:.1f}", estado, f"{res.raiz:.10f}", res.iteraciones, f"{err_final:.3e}", f"{res.tiempo_seg:.5f}"),
                )

            # conclusión rápida: el que menos iteraciones tiene
            mejor_x0 = min(resultados.keys(), key=lambda k: resultados[k].iteraciones)
            self.var_analysis.set(
                "Comparación completada.\n"
                + "\n".join([f"x0={x0}: iter={resultados[x0].iteraciones}, convergio={resultados[x0].convergio}" for x0 in self.X0_PREDETERMINADOS])
                + f"\n\nConcluye más rápido (menos iteraciones): x0={mejor_x0}."
            )

            # opcional: graficar error del "mejor" (para tener algo visible al comparar)
            self._plot_func_y_tangentes(resultados[mejor_x0])
            self._plot_error(resultados[mejor_x0])

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ------------------ Gráficas ------------------

    def _plot_inicial(self):
        # Función sola como vista inicial
        self.ax_func.clear()
        self.ax_err.clear()

        x = np.linspace(0.0, 6.0, 600)
        y = np.array([T_threads(float(xx)) for xx in x])

        self.ax_func.set_title("T(n) (vista inicial)", fontsize=10)
        self.ax_func.axhline(0.0, linewidth=1.0)
        self.ax_func.plot(x, y, label="T(n)")
        self.ax_func.set_xlabel("n (threads)")
        self.ax_func.set_ylabel("T(n)")
        self.ax_func.grid(True)
        self.ax_func.legend(loc="best")

        self.ax_err.set_title("Error (se mostrará al calcular)", fontsize=10)
        self.ax_err.set_xlabel("Iteración")
        self.ax_err.set_ylabel("Error abs (log)")
        self.ax_err.set_yscale("log")
        self.ax_err.grid(True)

        self._redraw()

    def _plot_func_y_tangentes(self, res):
        self.ax_func.clear()

        x = np.linspace(0.0, 6.0, 600)
        y = np.array([T_threads(float(xx)) for xx in x])

        self.ax_func.set_title("T(n) + tangentes por iteración", fontsize=10)
        self.ax_func.axhline(0.0, linewidth=1.0)
        self.ax_func.plot(x, y, label="T(n)")
        self.ax_func.set_xlabel("n (threads)")
        self.ax_func.set_ylabel("T(n)")
        self.ax_func.grid(True)

        # Tangentes: y = f(x_n) + f'(x_n)(x - x_n)
        # Para que no se vea infinito, dibujamos cada tangente en una ventana corta alrededor de x_n
        for row in res.historial:
            xn = row["x_n"]
            fx = row["fx"]
            dfx = row["dfx"]

            xt = np.linspace(xn - 0.8, xn + 0.8, 50)
            yt = fx + dfx * (xt - xn)
            self.ax_func.plot(xt, yt)

            # Punto de iteración
            self.ax_func.scatter([xn], [fx])

        self.ax_func.legend(loc="best")
        self._redraw()

    def _plot_error(self, res):
        self.ax_err.clear()
        self.ax_err.set_title("Convergencia del error absoluto (log)", fontsize=10)
        self.ax_err.set_xlabel("Iteración")
        self.ax_err.set_ylabel("|x_{n+1}-x_n| (log)")
        self.ax_err.set_yscale("log")
        self.ax_err.grid(True)

        it = list(range(1, res.iteraciones + 1))
        errs = np.array(res.errores_abs, dtype=float)
        errs = np.where(np.isfinite(errs), errs, np.nan)

        self.ax_err.plot(it, errs, label="Error abs")
        self.ax_err.legend(loc="best")
        self._redraw()