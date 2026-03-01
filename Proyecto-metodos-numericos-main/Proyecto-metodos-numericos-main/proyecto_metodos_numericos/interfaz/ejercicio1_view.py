import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from funciones.definiciones import T_lambda
from metodos.biseccion import biseccion


def make_scrollable_treeview(parent, columns, col_widths=None, height=12):
    """
    Crea un Treeview con scroll vertical y horizontal.
    Devuelve: (tree, container_frame)
    """
    container = ttk.Frame(parent)
    container.rowconfigure(0, weight=1)
    container.columnconfigure(0, weight=1)

    tree = ttk.Treeview(container, columns=columns, show="headings", height=height)

    for i, col in enumerate(columns):
        tree.heading(col, text=col)
        w = col_widths[i] if (col_widths and i < len(col_widths)) else 120
        # stretch=False fuerza a usar scroll horizontal cuando no cabe
        tree.column(col, width=w, anchor="center", stretch=False)

    vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")

    return tree, container


class Ejercicio1View(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self._build_ui()
        self._plot_inicial()

    # ------------------ UI ------------------

    def _build_ui(self):
        # Layout principal: izquierda (controles+tabla) / derecha (gráficas+resultado)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        right = ttk.Frame(self, padding=10)

        left.grid(row=0, column=0, sticky="nsew")
        right.grid(row=0, column=1, sticky="nsew")

        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)
        right.columnconfigure(0, weight=1)

        # --- Panel de entradas
        input_frame = ttk.LabelFrame(left, text="Ejercicio 1 - Bisección (T(λ)=0)", padding=10)
        input_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.var_a = tk.StringVar(value="0.5")
        self.var_b = tk.StringVar(value="2.5")
        self.var_tol = tk.StringVar(value="1e-6")
        self.var_iter = tk.StringVar(value="100")

        ttk.Label(input_frame, text="a:").grid(row=0, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.var_a, width=15).grid(row=0, column=1, sticky="w", padx=(5, 0))

        ttk.Label(input_frame, text="b:").grid(row=0, column=2, sticky="w", padx=(15, 0))
        ttk.Entry(input_frame, textvariable=self.var_b, width=15).grid(row=0, column=3, sticky="w", padx=(5, 0))

        ttk.Label(input_frame, text="Tolerancia:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(input_frame, textvariable=self.var_tol, width=15).grid(row=1, column=1, sticky="w", padx=(5, 0), pady=(8, 0))

        ttk.Label(input_frame, text="Max iter:").grid(row=1, column=2, sticky="w", padx=(15, 0), pady=(8, 0))
        ttk.Entry(input_frame, textvariable=self.var_iter, width=15).grid(row=1, column=3, sticky="w", padx=(5, 0), pady=(8, 0))

        btns = ttk.Frame(input_frame)
        btns.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        ttk.Button(btns, text="Calcular", command=self.on_calcular).pack(side="left")
        ttk.Button(btns, text="Limpiar", command=self.on_limpiar).pack(side="left", padx=(10, 0))

        # --- Tabla
        table_frame = ttk.LabelFrame(left, text="Tabla de iteraciones", padding=10)
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        cols = ("n", "a", "b", "c", "f(c)", "err_abs", "err_rel")
        widths = [70, 140, 140, 140, 170, 170, 170]

        self.tree, tree_container = make_scrollable_treeview(
            table_frame,
            columns=cols,
            col_widths=widths,
            height=18
        )
        tree_container.grid(row=0, column=0, sticky="nsew")

        # --- Gráficas
        charts_frame = ttk.LabelFrame(right, text="Gráficas", padding=10)
        charts_frame.grid(row=0, column=0, sticky="nsew")
        charts_frame.rowconfigure(0, weight=1)
        charts_frame.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(6, 7.5), dpi=100)
        self.ax_func = self.fig.add_subplot(2, 1, 1)
        self.ax_err = self.fig.add_subplot(2, 1, 2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=charts_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # --- Resultados
        result_frame = ttk.LabelFrame(right, text="Resultado final", padding=10)
        result_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        result_frame.columnconfigure(0, weight=1)

        self.var_res = tk.StringVar(value="Sin calcular.")
        ttk.Label(result_frame, textvariable=self.var_res, justify="left").grid(row=0, column=0, sticky="w")

    # ------------------ Matplotlib layout ------------------

    def _redraw(self):
        self.fig.tight_layout(pad=2.0)
        self.fig.subplots_adjust(hspace=0.60)
        self.canvas.draw()

    # ------------------ Acciones ------------------

    def on_limpiar(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.ax_func.clear()
        self.ax_err.clear()
        self.var_res.set("Sin calcular.")
        self._plot_inicial()

    def on_calcular(self):
        try:
            a = float(self.var_a.get().strip())
            b = float(self.var_b.get().strip())
            tol = float(self.var_tol.get().strip())
            max_iter = int(self.var_iter.get().strip())

            res = biseccion(T_lambda, a, b, tol=tol, max_iter=max_iter)

            # tabla
            for item in self.tree.get_children():
                self.tree.delete(item)

            for row in res.historial:
                self.tree.insert(
                    "",
                    "end",
                    values=(
                        row["n"],
                        f"{row['a']:.10f}",
                        f"{row['b']:.10f}",
                        f"{row['c']:.10f}",
                        f"{row['fc']:.10e}",
                        f"{row['err_abs']:.10e}" if np.isfinite(row["err_abs"]) else "inf",
                        f"{row['err_rel']:.10e}" if np.isfinite(row["err_rel"]) else "inf",
                    ),
                )

            # gráficas
            self._plot_func_and_iters(a, b, res)
            self._plot_error(res)

            estado = "Convergió" if res.convergio else "No convergió"
            err_final = res.errores_abs[-1] if res.errores_abs else float("nan")
            motivo = getattr(res, "motivo_parada", "")

            self.var_res.set(
                f"Estado: {estado}\n"
                f"Raíz aproximada (λ): {res.raiz:.10f}\n"
                f"Iteraciones: {res.iteraciones}\n"
                f"Error absoluto final: {err_final:.10e}\n"
                f"Tiempo: {res.tiempo_seg:.6f} s\n"
                + (f"Motivo: {motivo}\n" if motivo else "")
            )

        except Exception as e:
            # ejemplo: No hay cambio de signo, etc.
            messagebox.showerror("Error", str(e))

    # ------------------ Gráficas ------------------

    def _plot_inicial(self):
        try:
            a = float(self.var_a.get().strip())
            b = float(self.var_b.get().strip())
        except Exception:
            a, b = 0.5, 2.5

        self.ax_func.clear()
        self.ax_err.clear()

        x = np.linspace(a, b, 500)
        y = np.array([T_lambda(float(xx)) for xx in x])

        self.ax_func.set_title("T(λ) (vista inicial)", fontsize=10)
        self.ax_func.axhline(0.0, linewidth=1.0)
        self.ax_func.plot(x, y, label="T(λ)")
        self.ax_func.set_xlabel("λ")
        self.ax_func.set_ylabel("T(λ)")
        self.ax_func.grid(True)
        self.ax_func.legend(loc="best")

        self.ax_err.set_title("Error absoluto (se mostrará al calcular)", fontsize=10)
        self.ax_err.set_xlabel("Iteración")
        self.ax_err.set_ylabel("Error absoluto (log)")
        self.ax_err.set_yscale("log")
        self.ax_err.grid(True)

        self._redraw()

    def _plot_func_and_iters(self, a: float, b: float, res):
        self.ax_func.clear()

        x = np.linspace(a, b, 500)
        y = np.array([T_lambda(float(xx)) for xx in x])

        self.ax_func.set_title("T(λ) con aproximaciones sucesivas", fontsize=10)
        self.ax_func.axhline(0.0, linewidth=1.0)
        self.ax_func.plot(x, y, label="T(λ)")

        cs = [r["c"] for r in res.historial]
        fcs = [r["fc"] for r in res.historial]

        # puntos de aproximación
        if cs:
            self.ax_func.scatter(cs, fcs, label="Aproximaciones (c)")

            # raíz final destacada
            c_final = cs[-1]
            fc_final = fcs[-1]
            self.ax_func.scatter([c_final], [fc_final], marker="*", s=140, label="Raíz final")

        self.ax_func.set_xlabel("λ")
        self.ax_func.set_ylabel("T(λ)")
        self.ax_func.legend(loc="best")
        self.ax_func.grid(True)

        self._redraw()

    def _plot_error(self, res):
        self.ax_err.clear()
        self.ax_err.set_title("Convergencia del error absoluto (log)", fontsize=10)

        iters = list(range(1, res.iteraciones + 1))
        errs = np.array(res.errores_abs, dtype=float)
        errs = np.where(np.isfinite(errs), errs, np.nan)

        self.ax_err.plot(iters, errs, label="|c_n - c_{n-1}|")
        self.ax_err.set_yscale("log")
        self.ax_err.set_xlabel("Iteración")
        self.ax_err.set_ylabel("Error absoluto (log)")
        self.ax_err.grid(True)
        self.ax_err.legend(loc="best")

        self._redraw()