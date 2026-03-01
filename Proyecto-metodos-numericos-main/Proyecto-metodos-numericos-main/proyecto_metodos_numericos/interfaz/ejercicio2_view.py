import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from funciones.definiciones import E_workers
from metodos.biseccion import biseccion
from metodos.falsa_posicion import falsa_posicion


def make_scrollable_treeview(parent, columns, col_widths=None, height=12):
    """
    Treeview con scroll vertical + horizontal.
    Devuelve: (tree, container_frame)
    """
    container = ttk.Frame(parent)
    container.rowconfigure(0, weight=1)
    container.columnconfigure(0, weight=1)

    tree = ttk.Treeview(container, columns=columns, show="headings", height=height)

    for i, col in enumerate(columns):
        tree.heading(col, text=col)
        w = col_widths[i] if (col_widths and i < len(col_widths)) else 120
        tree.column(col, width=w, anchor="center", stretch=False)

    vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")

    return tree, container


class Ejercicio2View(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)

        self.ultimo_bis = None
        self.ultimo_fp = None

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

        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)

        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)
        right.columnconfigure(0, weight=1)

        # ---- Parámetros
        input_frame = ttk.LabelFrame(left, text="Ejercicio 2 - Falsa Posición vs Bisección", padding=10)
        input_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.var_a = tk.StringVar(value="2.0")
        self.var_b = tk.StringVar(value="4.0")
        self.var_tol = tk.StringVar(value="1e-7")
        self.var_iter = tk.StringVar(value="100")

        ttk.Label(input_frame, text="a:").grid(row=0, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.var_a, width=12).grid(row=0, column=1, sticky="w", padx=(5, 0))

        ttk.Label(input_frame, text="b:").grid(row=0, column=2, sticky="w", padx=(15, 0))
        ttk.Entry(input_frame, textvariable=self.var_b, width=12).grid(row=0, column=3, sticky="w", padx=(5, 0))

        ttk.Label(input_frame, text="Tolerancia:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(input_frame, textvariable=self.var_tol, width=12).grid(row=1, column=1, sticky="w", padx=(5, 0), pady=(8, 0))

        ttk.Label(input_frame, text="Max iter:").grid(row=1, column=2, sticky="w", padx=(15, 0), pady=(8, 0))
        ttk.Entry(input_frame, textvariable=self.var_iter, width=12).grid(row=1, column=3, sticky="w", padx=(5, 0), pady=(8, 0))

        # ---- Modo
        mode_frame = ttk.Frame(input_frame)
        mode_frame.grid(row=2, column=0, columnspan=4, sticky="w", pady=(10, 0))
        self.var_mode = tk.StringVar(value="comparar")

        ttk.Label(mode_frame, text="Modo:").pack(side="left")
        ttk.Radiobutton(mode_frame, text="Comparar ambos", variable=self.var_mode, value="comparar").pack(
            side="left", padx=8
        )
        ttk.Radiobutton(mode_frame, text="Solo Bisección", variable=self.var_mode, value="biseccion").pack(
            side="left", padx=8
        )
        ttk.Radiobutton(mode_frame, text="Solo Falsa Posición", variable=self.var_mode, value="falsa").pack(
            side="left", padx=8
        )

        # ---- Botones
        btns = ttk.Frame(input_frame)
        btns.grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Button(btns, text="Calcular", command=self.on_calcular).pack(side="left")
        ttk.Button(btns, text="Limpiar", command=self.on_limpiar).pack(side="left", padx=(10, 0))

        # ---- Resumen comparativo (tabla)
        resumen_frame = ttk.LabelFrame(left, text="Resumen comparativo", padding=10)
        resumen_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        resumen_frame.columnconfigure(0, weight=1)

        cols_res = ("metodo", "raiz", "iter", "err_final", "tiempo")
        widths_res = [150, 170, 90, 170, 140]
        self.resumen, resumen_container = make_scrollable_treeview(
            resumen_frame, columns=cols_res, col_widths=widths_res, height=3
        )
        resumen_container.grid(row=0, column=0, sticky="ew")

        # ---- Pestañas de tablas
        notebook_frame = ttk.LabelFrame(left, text="Tablas de iteraciones", padding=10)
        notebook_frame.grid(row=2, column=0, sticky="nsew")
        notebook_frame.rowconfigure(0, weight=1)
        notebook_frame.columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.tab_bis = ttk.Frame(self.notebook)
        self.tab_fp = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_bis, text="Bisección")
        self.notebook.add(self.tab_fp, text="Falsa Posición")

        self.tree_bis = self._crear_tabla_iter(self.tab_bis)
        self.tree_fp = self._crear_tabla_iter(self.tab_fp)

        # ---- Gráficas
        charts_frame = ttk.LabelFrame(right, text="Gráficas", padding=10)
        charts_frame.grid(row=0, column=0, sticky="nsew")
        charts_frame.rowconfigure(0, weight=1)
        charts_frame.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(6, 7.5), dpi=100)
        self.ax_func = self.fig.add_subplot(2, 1, 1)
        self.ax_err = self.fig.add_subplot(2, 1, 2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=charts_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # ---- Panel final
        result_frame = ttk.LabelFrame(right, text="Resultados finales", padding=10)
        result_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        result_frame.columnconfigure(0, weight=1)

        self.var_analysis = tk.StringVar(value="Sin calcular.")
        ttk.Label(result_frame, textvariable=self.var_analysis, justify="left").grid(row=0, column=0, sticky="w")

    # ------------------ Helpers UI ------------------

    def _redraw(self):
        self.fig.tight_layout(pad=2.0)
        self.fig.subplots_adjust(hspace=0.60)
        self.canvas.draw()

    def _crear_tabla_iter(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        cols = ("n", "a", "b", "c", "f(c)", "err_abs", "err_rel")
        widths = [70, 150, 150, 150, 170, 170, 170]

        tree, container = make_scrollable_treeview(parent, columns=cols, col_widths=widths, height=14)
        container.grid(row=0, column=0, sticky="nsew")
        return tree

    # ------------------ Acciones ------------------

    def on_limpiar(self):
        for t in (self.tree_bis, self.tree_fp, self.resumen):
            for item in t.get_children():
                t.delete(item)

        self.ultimo_bis = None
        self.ultimo_fp = None

        self.ax_func.clear()
        self.ax_err.clear()
        self.var_analysis.set("Sin calcular.")
        self._plot_inicial()

    def on_calcular(self):
        try:
            a = float(self.var_a.get().strip())
            b = float(self.var_b.get().strip())
            tol = float(self.var_tol.get().strip())
            max_iter = int(self.var_iter.get().strip())
            mode = self.var_mode.get().strip()

            # limpiar tablas
            for t in (self.tree_bis, self.tree_fp, self.resumen):
                for item in t.get_children():
                    t.delete(item)

            self.ultimo_bis = None
            self.ultimo_fp = None

            res_bis = None
            res_fp = None

            if mode in ("comparar", "biseccion"):
                res_bis = biseccion(E_workers, a, b, tol=tol, max_iter=max_iter)
                self.ultimo_bis = res_bis
                self._llenar_tabla(self.tree_bis, res_bis)

            if mode in ("comparar", "falsa"):
                res_fp = falsa_posicion(E_workers, a, b, tol=tol, max_iter=max_iter)
                self.ultimo_fp = res_fp
                self._llenar_tabla(self.tree_fp, res_fp)

            if res_bis is not None:
                self._add_resumen("Bisección", res_bis)
            if res_fp is not None:
                self._add_resumen("Falsa Posición", res_fp)

            self._plot_func(a, b)
            self._plot_convergencia(res_bis, res_fp)

            self.var_analysis.set(self._generar_panel_final(res_bis, res_fp, mode))

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _llenar_tabla(self, tree, res):
        for row in res.historial:
            tree.insert(
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

    def _add_resumen(self, metodo, res):
        err_final = res.errores_abs[-1] if getattr(res, "errores_abs", None) else float("nan")
        self.resumen.insert(
            "",
            "end",
            values=(
                metodo,
                f"{res.raiz:.10f}",
                res.iteraciones,
                f"{err_final:.10e}",
                f"{res.tiempo_seg:.6f} s",
            ),
        )

    # ------------------ Gráficas ------------------

    def _plot_inicial(self):
        self._plot_func(2.0, 4.0)
        self.ax_err.clear()
        self.ax_err.set_title("Convergencia del error absoluto (log)", fontsize=10)
        self.ax_err.set_xlabel("Iteración")
        self.ax_err.set_ylabel("Error abs (log)")
        self.ax_err.set_yscale("log")
        self.ax_err.grid(True)
        self._redraw()

    def _plot_func(self, a, b):
        self.ax_func.clear()

        x = np.linspace(a, b, 600)
        y = np.array([E_workers(float(xx)) for xx in x])

        self.ax_func.set_title("E(x) en el intervalo + raíz final", fontsize=10)
        self.ax_func.axhline(0.0, linewidth=1.0)
        self.ax_func.plot(x, y, label="E(x)")
        self.ax_func.set_xlabel("x")
        self.ax_func.set_ylabel("E(x)")
        self.ax_func.grid(True)

        # ⭐ raíces finales
        if self.ultimo_bis is not None:
            xb = float(self.ultimo_bis.raiz)
            self.ax_func.scatter([xb], [E_workers(xb)], marker="*", s=140, label="Raíz final (Bisección)")

        if self.ultimo_fp is not None:
            xf = float(self.ultimo_fp.raiz)
            self.ax_func.scatter([xf], [E_workers(xf)], marker="*", s=140, label="Raíz final (Falsa Posición)")

        self.ax_func.legend(loc="best")

    def _plot_convergencia(self, res_bis, res_fp):
        self.ax_err.clear()
        self.ax_err.set_title("Convergencia del error absoluto (log)", fontsize=10)
        self.ax_err.set_yscale("log")
        self.ax_err.set_xlabel("Iteración")
        self.ax_err.set_ylabel("Error abs (log)")
        self.ax_err.grid(True)

        if res_bis is not None and getattr(res_bis, "errores_abs", None):
            it_b = list(range(1, res_bis.iteraciones + 1))
            errs_b = np.array(res_bis.errores_abs, dtype=float)
            errs_b = np.where(np.isfinite(errs_b), errs_b, np.nan)
            self.ax_err.plot(it_b, errs_b, label="Bisección")

        if res_fp is not None and getattr(res_fp, "errores_abs", None):
            it_f = list(range(1, res_fp.iteraciones + 1))
            errs_f = np.array(res_fp.errores_abs, dtype=float)
            errs_f = np.where(np.isfinite(errs_f), errs_f, np.nan)
            self.ax_err.plot(it_f, errs_f, label="Falsa Posición")

        self.ax_err.legend(loc="best")
        self._redraw()

    # ------------------ Panel final ------------------

    def _generar_panel_final(self, res_bis, res_fp, mode):
        def _pack(nombre, res):
            if res is None:
                return ""
            estado = "Convergió" if res.convergio else "No convergió"
            err_final = res.errores_abs[-1] if getattr(res, "errores_abs", None) else float("nan")
            motivo = getattr(res, "motivo_parada", "")
            txt = (
                f"{nombre}:\n"
                f"- Estado: {estado}\n"
                f"- Raíz: {res.raiz:.10f}\n"
                f"- Iteraciones: {res.iteraciones}\n"
                f"- Error final: {err_final:.10e}\n"
                f"- Tiempo: {res.tiempo_seg:.6f} s\n"
            )
            if motivo:
                txt += f"- Motivo: {motivo}\n"
            return txt + "\n"

        if mode == "biseccion":
            return _pack("Bisección", res_bis)

        if mode == "falsa":
            return _pack("Falsa Posición", res_fp)

        txt = "Comparación:\n\n" + _pack("Bisección", res_bis) + _pack("Falsa Posición", res_fp)

        if res_bis is not None and res_fp is not None:
            mejor = "Falsa Posición" if res_fp.iteraciones < res_bis.iteraciones else "Bisección"
            txt += (
                "Conclusión:\n"
                f"- Menos iteraciones: {mejor}\n"
                "- Nota: Bisección reduce el intervalo de forma garantizada; Falsa Posición usa interpolación lineal.\n"
            )
        return txt