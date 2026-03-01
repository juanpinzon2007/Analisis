import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from funciones.definiciones import g_db, g_db_deriv
from metodos.punto_fijo import punto_fijo


class Ejercicio3View(ttk.Frame):
    """
    Ejercicio 3: Punto Fijo
    - Verificar condición de convergencia |g'(x)| < 1
    - Tabla iterativa
    - y=x y y=g(x) con cobweb plot
    - Comparación de convergencia para varios x0
    """

    X0_PREDETERMINADOS = [0.5, 1.0, 1.5, 2.0]

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

        # --- Parámetros
        input_frame = ttk.LabelFrame(left, text="Ejercicio 3 - Punto Fijo: x = 0.5 cos(x) + 1.5", padding=10)
        input_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)

        self.var_x0 = tk.StringVar(value="1.0")
        self.var_tol = tk.StringVar(value="1e-8")
        self.var_iter = tk.StringVar(value="100")
        self.var_lim = tk.StringVar(value="1e6")

        ttk.Label(input_frame, text="x0 (seleccionado):").grid(row=0, column=0, sticky="w")
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

        ttk.Label(input_frame, text="Límite divergencia |x|:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(input_frame, textvariable=self.var_lim, width=15).grid(row=2, column=1, sticky="w", padx=(5, 0), pady=(8, 0))

        btns = ttk.Frame(input_frame)
        btns.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        ttk.Button(btns, text="Calcular x0 seleccionado", command=self.on_calcular_seleccionado).pack(side="left")
        ttk.Button(btns, text="Comparar x0 (0.5,1.0,1.5,2.0)", command=self.on_comparar).pack(side="left", padx=(10, 0))
        ttk.Button(btns, text="Limpiar", command=self.on_limpiar).pack(side="left", padx=(10, 0))

        # --- Condición de convergencia
        conv_frame = ttk.LabelFrame(left, text="Condición de convergencia", padding=10)
        conv_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        conv_frame.columnconfigure(0, weight=1)
        self.var_conv = tk.StringVar(value="Sin calcular.")
        ttk.Label(conv_frame, textvariable=self.var_conv, justify="left").grid(row=0, column=0, sticky="w")

        # --- Tabla iteraciones
        table_frame = ttk.LabelFrame(left, text="Tabla de iteraciones (x0 seleccionado)", padding=10)
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        cols = ("n", "x_n", "g(x_n)", "|x_n-g(x_n)|", "err_rel")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=16)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor="center", width=140)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # --- Gráficas
        charts_frame = ttk.LabelFrame(right, text="Gráficas", padding=10)
        charts_frame.grid(row=0, column=0, sticky="nsew")
        charts_frame.rowconfigure(0, weight=1)
        charts_frame.columnconfigure(0, weight=1)

        # ✅ Figura más alta para que respire mejor
        self.fig = Figure(figsize=(6, 7.5), dpi=100)
        self.ax_cobweb = self.fig.add_subplot(2, 1, 1)
        self.ax_comp = self.fig.add_subplot(2, 1, 2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=charts_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # --- Resultado / análisis
        result_frame = ttk.LabelFrame(right, text="Análisis", padding=10)
        result_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        result_frame.columnconfigure(0, weight=1)

        self.var_analysis = tk.StringVar(value="Sin calcular.")
        ttk.Label(result_frame, textvariable=self.var_analysis, justify="left").grid(row=0, column=0, sticky="w")

    # ------------------ Render / Layout de Matplotlib ------------------

    def _redraw(self):
        """
        ✅ Ajuste profesional de espacios para que los subplots no se monten.
        """
        self.fig.tight_layout(pad=2.0)
        self.fig.subplots_adjust(hspace=0.60)
        self.canvas.draw()

    # ------------------ Botones ------------------

    def on_limpiar(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.ax_cobweb.clear()
        self.ax_comp.clear()

        self.var_analysis.set("Sin calcular.")
        self.var_conv.set("Sin calcular.")

        self._plot_inicial()

    def on_calcular_seleccionado(self):
        try:
            x0 = float(self.var_x0.get().strip())
            tol = float(self.var_tol.get().strip())
            max_iter = int(self.var_iter.get().strip())
            lim = float(self.var_lim.get().strip())

            # Convergencia
            self._mostrar_condicion_convergencia(x0)

            # Cálculo
            res = punto_fijo(g_db, x0=x0, tol=tol, max_iter=max_iter, limite_divergencia=lim)

            # Tabla
            for item in self.tree.get_children():
                self.tree.delete(item)

            for row in res.historial:
                self.tree.insert(
                    "",
                    "end",
                    values=(
                        row["n"],
                        f"{row['x_n']:.12f}",
                        f"{row['gxn']:.12f}",
                        f"{row['abs_diff']:.3e}",
                        f"{row['err_rel']:.3e}",
                    ),
                )

            # Gráficas
            self._plot_cobweb(res, x0=x0)
            self._plot_comparacion({x0: res})

            estado = "Convergió" if res.convergio else "No convergió"
            self.var_analysis.set(
                f"Estado: {estado}\n"
                f"x0: {x0}\n"
                f"x* (aprox): {res.x_final:.12f}\n"
                f"Iteraciones: {res.iteraciones}\n"
                f"Motivo: {res.motivo_parada}\n"
                f"Tiempo: {res.tiempo_seg:.6f} s\n\n"
                f"Análisis: El valor inicial puede acelerar o retardar la convergencia\n"
                f"dependiendo de qué tan cerca esté del punto fijo y del tamaño de |g'(x)|."
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_comparar(self):
        try:
            tol = float(self.var_tol.get().strip())
            max_iter = int(self.var_iter.get().strip())
            lim = float(self.var_lim.get().strip())

            resultados = {}
            for x0 in self.X0_PREDETERMINADOS:
                resultados[x0] = punto_fijo(g_db, x0=x0, tol=tol, max_iter=max_iter, limite_divergencia=lim)

            self._plot_comparacion(resultados)

            resumen = []
            for x0, res in resultados.items():
                estado = "OK" if res.convergio else "FAIL"
                resumen.append(f"x0={x0}: {estado}, iter={res.iteraciones}, x*≈{res.x_final:.8f}")

            self.var_analysis.set(
                "Comparación de convergencia por x0:\n"
                + "\n".join(resumen)
                + "\n\nConclusión: Un x0 más cercano al punto fijo (y con |g'(x)| pequeño cerca de la trayectoria)\n"
                  "tiende a converger en menos iteraciones."
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ------------------ Gráficas ------------------

    def _plot_inicial(self):
        self.ax_cobweb.clear()
        self.ax_comp.clear()

        x = np.linspace(0.0, 3.0, 600)
        y = np.array([g_db(float(xx)) for xx in x])

        self.ax_cobweb.set_title("y=x y y=g(x) (vista inicial)", fontsize=10)
        self.ax_cobweb.plot(x, x, label="y = x")
        self.ax_cobweb.plot(x, y, label="y = g(x)")
        self.ax_cobweb.set_xlabel("x")
        self.ax_cobweb.set_ylabel("y")
        self.ax_cobweb.grid(True)
        self.ax_cobweb.legend(loc="best")

        self.ax_comp.set_title("Comparación de convergencia (se mostrará al comparar)", fontsize=10)
        self.ax_comp.set_xlabel("Iteración")
        self.ax_comp.set_ylabel("Error abs (log)")
        self.ax_comp.set_yscale("log")
        self.ax_comp.grid(True)

        self._redraw()

    def _plot_cobweb(self, res, x0: float):
        self.ax_cobweb.clear()

        x = np.linspace(0.0, 3.0, 600)
        y = np.array([g_db(float(xx)) for xx in x])

        self.ax_cobweb.set_title(f"Cobweb plot (x0={x0})", fontsize=10)
        self.ax_cobweb.plot(x, x, label="y = x")
        self.ax_cobweb.plot(x, y, label="y = g(x)")
        self.ax_cobweb.grid(True)
        self.ax_cobweb.set_xlabel("x")
        self.ax_cobweb.set_ylabel("y")

        xs = [row["x_n"] for row in res.historial]
        gxs = [row["gxn"] for row in res.historial]

        for xn, gxn in zip(xs, gxs):
            self.ax_cobweb.plot([xn, xn], [xn, gxn])       # vertical
            self.ax_cobweb.plot([xn, gxn], [gxn, gxn])     # horizontal

        self.ax_cobweb.legend(loc="best")

    def _plot_comparacion(self, resultados: dict):
        self.ax_comp.clear()
        self.ax_comp.set_title("Comparación de convergencia por x0 (error absoluto)", fontsize=10)
        self.ax_comp.set_xlabel("Iteración")
        self.ax_comp.set_ylabel("|x_n - g(x_n)| (log)")
        self.ax_comp.set_yscale("log")
        self.ax_comp.grid(True)

        for x0, res in resultados.items():
            it = list(range(1, res.iteraciones + 1))
            errs = np.array(res.errores_abs, dtype=float)
            errs = np.where(np.isfinite(errs), errs, np.nan)
            self.ax_comp.plot(it, errs, label=f"x0={x0}")

        self.ax_comp.legend(loc="best")
        self._redraw()

    # ------------------ Convergencia ------------------

    def _mostrar_condicion_convergencia(self, x0: float):
        """
        La guía pide verificar |g'(x)| < 1.
        Aquí hacemos un chequeo en una ventana alrededor de x0.
        """
        a = max(0.0, x0 - 1.0)
        b = min(3.0, x0 + 1.0)

        xs = np.linspace(a, b, 400)
        vals = np.abs([g_db_deriv(float(xx)) for xx in xs])
        vmax = float(np.max(vals))

        ok = vmax < 1.0
        self.var_conv.set(
            f"Chequeo en x∈[{a:.2f},{b:.2f}]: max |g'(x)| ≈ {vmax:.6f}\n"
            f"Condición |g'(x)| < 1: {'CUMPLE (convergencia probable)' if ok else 'NO CUMPLE (riesgo de divergencia)'}\n"
            f"Nota: g'(x) = -0.5 sin(x) ⇒ |g'(x)| ≤ 0.5 (en todo x), por tanto converge."
        )