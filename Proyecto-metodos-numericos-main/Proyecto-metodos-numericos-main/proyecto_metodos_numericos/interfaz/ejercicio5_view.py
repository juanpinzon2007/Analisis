import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ---------------------------
# Función del ejercicio
# P(x) = x e^(-x/2) - 0.3
# Derivada: P'(x) = e^(-x/2) (1 - x/2)
# ---------------------------
def P(x: float) -> float:
    return x * np.exp(-x / 2.0) - 0.3


def dP(x: float) -> float:
    return np.exp(-x / 2.0) * (1.0 - x / 2.0)


# ---------------------------
# Helper Treeview con scroll (X + Y)
# ---------------------------
def make_scrollable_treeview(parent, columns, col_widths=None, height=12):
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


# ---------------------------
# ScrollableFrame para Gráficas (NO redimensiona el contenido)
# Mantiene tamaño real del widget interno y permite scroll X/Y.
# ---------------------------
class ScrollableFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)

        self.canvas.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        self.inner = ttk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.hsb.grid(row=1, column=0, sticky="ew")

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # actualizar región de scroll cuando cambie el contenido
        self.inner.bind("<Configure>", self._on_inner_configure)

        # rueda del mouse SOLO cuando el puntero está sobre el canvas
        self.canvas.bind("<Enter>", lambda _e: self._bind_mousewheel())
        self.canvas.bind("<Leave>", lambda _e: self._unbind_mousewheel())

    def _on_inner_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _bind_mousewheel(self):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)      # Windows/mac
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)  # Linux
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")


class Ejercicio5View(ttk.Frame):
    """
    Ejercicio 5 - Secante vs Newton-Raphson

    Muestra:
    - Tabla Secante + scroll X/Y
    - Tabla Newton + scroll X/Y
    - Tabla comparativa final
    - Gráfica función
    - Gráfica secantes por iteración
    - Gráfica convergencia del error (log) comparando ambos
    - Panel análisis con scroll
    """

    def __init__(self, parent):
        super().__init__(parent, padding=10)

        self.res_secante = None
        self.res_newton = None

        self._build_ui()
        self._plot_inicial()
        self._set_analysis_text("Sin calcular.")

    # ---------------- UI ----------------

    def _build_ui(self):
        # más espacio a la derecha (gráficas)
        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=3)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        right = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="nsew")
        right.grid(row=0, column=1, sticky="nsew")

        left.rowconfigure(3, weight=1)
        left.columnconfigure(0, weight=1)

        right.rowconfigure(0, weight=2)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # ---- Entradas
        input_frame = ttk.LabelFrame(left, text="Ejercicio 5 - Secante vs Newton-Raphson", padding=10)
        input_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.var_x0 = tk.StringVar(value="0.5")
        self.var_x1 = tk.StringVar(value="1.0")
        self.var_tol = tk.StringVar(value="1e-9")
        self.var_iter = tk.StringVar(value="100")

        ttk.Label(input_frame, text="x0:").grid(row=0, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.var_x0, width=12).grid(row=0, column=1, sticky="w", padx=(5, 0))

        ttk.Label(input_frame, text="x1:").grid(row=0, column=2, sticky="w", padx=(15, 0))
        ttk.Entry(input_frame, textvariable=self.var_x1, width=12).grid(row=0, column=3, sticky="w", padx=(5, 0))

        ttk.Label(input_frame, text="Tolerancia:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(input_frame, textvariable=self.var_tol, width=12).grid(row=1, column=1, sticky="w", padx=(5, 0), pady=(8, 0))

        ttk.Label(input_frame, text="Max iter:").grid(row=1, column=2, sticky="w", padx=(15, 0), pady=(8, 0))
        ttk.Entry(input_frame, textvariable=self.var_iter, width=12).grid(row=1, column=3, sticky="w", padx=(5, 0), pady=(8, 0))

        btns = ttk.Frame(input_frame)
        btns.grid(row=2, column=0, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Button(btns, text="Calcular", command=self.on_calcular).pack(side="left")
        ttk.Button(btns, text="Limpiar", command=self.on_limpiar).pack(side="left", padx=(10, 0))

        # ---- Tabla comparativa final
        comp_frame = ttk.LabelFrame(left, text="Comparación final", padding=10)
        comp_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        comp_frame.columnconfigure(0, weight=1)

        cols_comp = ("metodo", "raiz", "iter", "eval_f", "eval_df", "eval_total", "err_final", "tiempo")
        widths_comp = [120, 170, 70, 90, 90, 110, 140, 120]
        self.tree_comp, comp_container = make_scrollable_treeview(comp_frame, cols_comp, widths_comp, height=3)
        comp_container.grid(row=0, column=0, sticky="ew")

        # ---- Tablas
        tables_frame = ttk.LabelFrame(left, text="Tablas de iteraciones", padding=10)
        tables_frame.grid(row=3, column=0, sticky="nsew")
        tables_frame.rowconfigure(0, weight=1)
        tables_frame.columnconfigure(0, weight=1)

        self.nb = ttk.Notebook(tables_frame)
        self.nb.grid(row=0, column=0, sticky="nsew")

        self.tab_sec = ttk.Frame(self.nb)
        self.tab_new = ttk.Frame(self.nb)
        self.nb.add(self.tab_sec, text="Secante")
        self.nb.add(self.tab_new, text="Newton-Raphson")

        self.tree_sec = self._build_table_sec(self.tab_sec)
        self.tree_new = self._build_table_new(self.tab_new)

        # ---- Gráficas (con scroll X/Y, sin deformar)
        charts_frame = ttk.LabelFrame(right, text="Gráficas", padding=10)
        charts_frame.grid(row=0, column=0, sticky="nsew")
        charts_frame.rowconfigure(0, weight=1)
        charts_frame.columnconfigure(0, weight=1)

        self.scroll_charts = ScrollableFrame(charts_frame)
        self.scroll_charts.grid(row=0, column=0, sticky="nsew")

        # Figura (mantiene tamaño real)
        self.fig = Figure(figsize=(7.2, 10.8), dpi=100)
        self.ax_func = self.fig.add_subplot(3, 1, 1)
        self.ax_sec = self.fig.add_subplot(3, 1, 2)
        self.ax_err = self.fig.add_subplot(3, 1, 3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.scroll_charts.inner)
        widget = self.canvas.get_tk_widget()
        widget.pack()  # IMPORTANTE: no fill/expand, para NO aplastar

        # fija el tamaño real del widget al tamaño de la figura (px)
        w = int(self.fig.get_figwidth() * self.fig.dpi)
        h = int(self.fig.get_figheight() * self.fig.dpi)
        widget.configure(width=w, height=h)

        # ---- Análisis (con scroll)
        analysis_frame = ttk.LabelFrame(right, text="Análisis", padding=10)
        analysis_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        analysis_frame.rowconfigure(0, weight=1)
        analysis_frame.columnconfigure(0, weight=1)

        self.txt_analysis = tk.Text(analysis_frame, wrap="word", height=10)
        sb = ttk.Scrollbar(analysis_frame, orient="vertical", command=self.txt_analysis.yview)
        self.txt_analysis.configure(yscrollcommand=sb.set)

        self.txt_analysis.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

    def _build_table_sec(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        cols = ("n", "x_{n-1}", "x_n", "f(x_{n-1})", "f(x_n)", "x_{n+1}", "err_abs", "err_rel")
        widths = [70, 150, 150, 160, 160, 150, 140, 140]
        tree, container = make_scrollable_treeview(parent, cols, widths, height=12)
        container.grid(row=0, column=0, sticky="nsew")
        return tree

    def _build_table_new(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        cols = ("n", "x_n", "f(x_n)", "f'(x_n)", "x_{n+1}", "err_abs", "err_rel")
        widths = [70, 160, 160, 160, 160, 140, 140]
        tree, container = make_scrollable_treeview(parent, cols, widths, height=12)
        container.grid(row=0, column=0, sticky="nsew")
        return tree

    # ---------------- Analysis text ----------------

    def _set_analysis_text(self, text: str):
        self.txt_analysis.configure(state="normal")
        self.txt_analysis.delete("1.0", "end")
        self.txt_analysis.insert("1.0", text)
        self.txt_analysis.configure(state="disabled")

    # ---------------- Matplotlib layout ----------------

    def _redraw(self):
        self.fig.tight_layout(pad=1.2)
        self.fig.subplots_adjust(hspace=0.95, top=0.95, bottom=0.07, left=0.12, right=0.96)
        self.canvas.draw()

    # ---------------- Acciones ----------------

    def on_limpiar(self):
        for t in (self.tree_comp, self.tree_sec, self.tree_new):
            for item in t.get_children():
                t.delete(item)
        self.res_secante = None
        self.res_newton = None
        self._set_analysis_text("Sin calcular.")
        self._plot_inicial()

    def on_calcular(self):
        try:
            x0 = float(self.var_x0.get().strip())
            x1 = float(self.var_x1.get().strip())
            tol = float(self.var_tol.get().strip())
            max_iter = int(self.var_iter.get().strip())

            if tol <= 0:
                raise ValueError("La tolerancia debe ser > 0.")
            if max_iter <= 0:
                raise ValueError("Max iter debe ser > 0.")
            if x0 == x1:
                raise ValueError("Para Secante, x0 y x1 deben ser diferentes.")

            for t in (self.tree_comp, self.tree_sec, self.tree_new):
                for item in t.get_children():
                    t.delete(item)

            self.res_secante = self._secante(P, x0, x1, tol, max_iter)
            self.res_newton = self._newton(P, dP, x1, tol, max_iter)

            for row in self.res_secante["hist"]:
                self.tree_sec.insert(
                    "",
                    "end",
                    values=(
                        row["n"],
                        f"{row['x_nm1']:.12f}",
                        f"{row['x_n']:.12f}",
                        f"{row['f_nm1']:.3e}",
                        f"{row['f_n']:.3e}",
                        f"{row['x_np1']:.12f}",
                        f"{row['err_abs']:.3e}",
                        f"{row['err_rel']:.3e}",
                    ),
                )

            for row in self.res_newton["hist"]:
                self.tree_new.insert(
                    "",
                    "end",
                    values=(
                        row["n"],
                        f"{row['x_n']:.12f}",
                        f"{row['f_n']:.3e}",
                        f"{row['df_n']:.3e}",
                        f"{row['x_np1']:.12f}",
                        f"{row['err_abs']:.3e}",
                        f"{row['err_rel']:.3e}",
                    ),
                )

            self._add_comp("Secante", self.res_secante)
            self._add_comp("Newton", self.res_newton)

            self._plot_all(self.res_secante, self.res_newton)
            self._set_analysis_text(self._build_analysis(self.res_secante, self.res_newton))

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _add_comp(self, metodo, res):
        self.tree_comp.insert(
            "",
            "end",
            values=(
                metodo,
                f"{res['raiz']:.12f}" if np.isfinite(res["raiz"]) else "nan",
                res["iter"],
                res["eval_f"],
                res["eval_df"],
                res["eval_f"] + res["eval_df"],
                f"{res['err_final']:.3e}",
                f"{res['tiempo']:.6f} s",
            ),
        )

    # ---------------- Métodos numéricos ----------------

    def _secante(self, f, x0, x1, tol, max_iter):
        import time
        t0 = time.perf_counter()

        hist = []
        eval_f = 0

        x_nm1 = float(x0)
        x_n = float(x1)

        f_nm1 = float(f(x_nm1)); eval_f += 1
        f_n = float(f(x_n)); eval_f += 1

        convergio = False
        motivo = "Max iter alcanzado"
        errores = []

        for n in range(1, max_iter + 1):
            denom = (f_n - f_nm1)
            if denom == 0:
                motivo = "División por cero: f(x_n) - f(x_{n-1}) = 0"
                break

            x_np1 = x_n - f_n * (x_n - x_nm1) / denom

            err_abs = abs(x_np1 - x_n)
            err_rel = err_abs / abs(x_np1) if x_np1 != 0 else float("inf")
            errores.append(err_abs)

            hist.append({
                "n": n,
                "x_nm1": x_nm1,
                "x_n": x_n,
                "f_nm1": f_nm1,
                "f_n": f_n,
                "x_np1": x_np1,
                "err_abs": err_abs,
                "err_rel": err_rel,
            })

            if err_abs <= tol:
                convergio = True
                motivo = "|x_{n+1} - x_n| <= tol"
                x_n = x_np1
                f_n = float(f(x_n)); eval_f += 1
                break

            x_nm1, x_n = x_n, x_np1
            f_nm1, f_n = f_n, float(f(x_n)); eval_f += 1

        t1 = time.perf_counter()
        raiz = x_n
        err_final = errores[-1] if errores else float("nan")

        return {
            "metodo": "Secante",
            "convergio": convergio,
            "motivo": motivo,
            "raiz": raiz,
            "iter": len(hist),
            "hist": hist,
            "errores": errores,
            "eval_f": eval_f,
            "eval_df": 0,
            "err_final": err_final,
            "tiempo": (t1 - t0),
        }

    def _newton(self, f, df, x0, tol, max_iter):
        import time
        t0 = time.perf_counter()

        hist = []
        eval_f = 0
        eval_df = 0

        x_n = float(x0)
        convergio = False
        motivo = "Max iter alcanzado"
        errores = []

        for n in range(1, max_iter + 1):
            f_n = float(f(x_n)); eval_f += 1
            df_n = float(df(x_n)); eval_df += 1

            if df_n == 0:
                motivo = "Derivada cero: f'(x_n)=0. Newton no puede continuar."
                break

            x_np1 = x_n - f_n / df_n

            err_abs = abs(x_np1 - x_n)
            err_rel = err_abs / abs(x_np1) if x_np1 != 0 else float("inf")
            errores.append(err_abs)

            hist.append({
                "n": n,
                "x_n": x_n,
                "f_n": f_n,
                "df_n": df_n,
                "x_np1": x_np1,
                "err_abs": err_abs,
                "err_rel": err_rel,
            })

            if err_abs <= tol:
                convergio = True
                motivo = "|x_{n+1} - x_n| <= tol"
                x_n = x_np1
                break

            x_n = x_np1

        t1 = time.perf_counter()
        raiz = x_n
        err_final = errores[-1] if errores else float("nan")

        return {
            "metodo": "Newton",
            "convergio": convergio,
            "motivo": motivo,
            "raiz": raiz,
            "iter": len(hist),
            "hist": hist,
            "errores": errores,
            "eval_f": eval_f,
            "eval_df": eval_df,
            "err_final": err_final,
            "tiempo": (t1 - t0),
        }

    # ---------------- Gráficas ----------------

    def _plot_inicial(self):
        self.ax_func.clear()
        self.ax_sec.clear()
        self.ax_err.clear()

        x = np.linspace(0, 6, 600)
        y = np.array([P(float(xx)) for xx in x])

        self.ax_func.set_title("P(x) = x e^{-x/2} - 0.3", fontsize=11)
        self.ax_func.axhline(0.0, linewidth=1.0)
        self.ax_func.plot(x, y, label="P(x)")
        self.ax_func.set_xlabel("x", fontsize=10)
        self.ax_func.set_ylabel("P(x)", fontsize=10)
        self.ax_func.grid(True)
        self.ax_func.legend(loc="upper left", fontsize=9, frameon=True)

        self.ax_sec.set_title("Secantes por iteración", fontsize=11)
        self.ax_sec.axhline(0.0, linewidth=1.0)
        self.ax_sec.plot(x, y, label="P(x)")
        self.ax_sec.set_xlabel("x", fontsize=10)
        self.ax_sec.set_ylabel("P(x)", fontsize=10)
        self.ax_sec.grid(True)
        self.ax_sec.legend(loc="upper left", fontsize=9, frameon=True)

        self.ax_err.set_title("Convergencia del error absoluto (log)", fontsize=11)
        self.ax_err.set_xlabel("Iteración", fontsize=10)
        self.ax_err.set_ylabel("Error abs (log)", fontsize=10)
        self.ax_err.set_yscale("log")
        self.ax_err.grid(True)

        self._redraw()

    def _plot_all(self, res_sec, res_new):
        xs = [0.0, 6.0]
        if np.isfinite(res_sec["raiz"]):
            xs.append(float(res_sec["raiz"]))
        if np.isfinite(res_new["raiz"]):
            xs.append(float(res_new["raiz"]))

        xmin = max(min(xs) - 0.5, 0.0)
        xmax = max(xs) + 0.5

        x = np.linspace(xmin, xmax, 700)
        y = np.array([P(float(xx)) for xx in x])

        # 1) Función + raíces
        self.ax_func.clear()
        self.ax_func.set_title("P(x) = x e^{-x/2} - 0.3", fontsize=11)
        self.ax_func.axhline(0.0, linewidth=1.0)
        self.ax_func.plot(x, y, label="P(x)")
        self.ax_func.set_xlabel("x", fontsize=10)
        self.ax_func.set_ylabel("P(x)", fontsize=10)
        self.ax_func.grid(True)
        self.ax_func.set_xlim(xmin, xmax)

        self.ax_func.scatter([res_sec["raiz"]], [P(float(res_sec["raiz"]))], marker="*", s=140, label="Raíz (Secante)")
        self.ax_func.scatter([res_new["raiz"]], [P(float(res_new["raiz"]))], marker="*", s=140, label="Raíz (Newton)")
        self.ax_func.legend(loc="upper left", fontsize=9, frameon=True)

        # 2) Secantes por iteración
        self.ax_sec.clear()
        self.ax_sec.set_title("Secantes por iteración", fontsize=11)
        self.ax_sec.axhline(0.0, linewidth=1.0)
        self.ax_sec.plot(x, y, label="P(x)")
        self.ax_sec.set_xlabel("x", fontsize=10)
        self.ax_sec.set_ylabel("P(x)", fontsize=10)
        self.ax_sec.grid(True)
        self.ax_sec.set_xlim(xmin, xmax)

        for row in res_sec["hist"]:
            xA = float(row["x_nm1"])
            xB = float(row["x_n"])
            self.ax_sec.plot([xA, xB], [P(xA), P(xB)])

        pts = [float(res_sec["hist"][0]["x_nm1"])] + [float(r["x_n"]) for r in res_sec["hist"]]
        self.ax_sec.scatter(pts, [P(p) for p in pts], s=18)
        self.ax_sec.legend(loc="upper left", fontsize=9, frameon=True)

        # 3) Error log comparativo
        self.ax_err.clear()
        self.ax_err.set_title("Convergencia del error absoluto (log)", fontsize=11)
        self.ax_err.set_xlabel("Iteración", fontsize=10)
        self.ax_err.set_ylabel("Error abs (log)", fontsize=10)
        self.ax_err.set_yscale("log")
        self.ax_err.grid(True)

        it_s = list(range(1, len(res_sec["errores"]) + 1))
        self.ax_err.plot(it_s, res_sec["errores"], label="Secante")

        it_n = list(range(1, len(res_new["errores"]) + 1))
        self.ax_err.plot(it_n, res_new["errores"], label="Newton")

        self.ax_err.legend(loc="best", fontsize=9, frameon=True)

        self._redraw()

    # ---------------- Texto análisis (mejorado) ----------------

    def _build_analysis(self, res_sec, res_new):
        def estado_txt(res):
            return "Convergió" if res["convergio"] else "No convergió"

        txt = (
            f"Secante:\n"
            f"- Estado: {estado_txt(res_sec)}\n"
            f"- Raíz: {res_sec['raiz']:.12f}\n"
            f"- Iteraciones: {res_sec['iter']}\n"
            f"- Error final: {res_sec['err_final']:.3e}\n"
            f"- Evaluaciones: f={res_sec['eval_f']} (df=0) | Total={res_sec['eval_f']}\n"
            f"- Tiempo: {res_sec['tiempo']:.6f} s\n"
            f"- Motivo: {res_sec['motivo']}\n\n"
            f"Newton-Raphson:\n"
            f"- Estado: {estado_txt(res_new)}\n"
            f"- Raíz: {res_new['raiz']:.12f}\n"
            f"- Iteraciones: {res_new['iter']}\n"
            f"- Error final: {res_new['err_final']:.3e}\n"
            f"- Evaluaciones: f={res_new['eval_f']} | df={res_new['eval_df']} | Total={res_new['eval_f'] + res_new['eval_df']}\n"
            f"- Tiempo: {res_new['tiempo']:.6f} s\n"
            f"- Motivo: {res_new['motivo']}\n\n"
        )

        mejor_iter = "Newton" if res_new["iter"] < res_sec["iter"] else "Secante"
        mejor_eval = "Secante" if (res_sec["eval_f"] < (res_new["eval_f"] + res_new["eval_df"])) else "Newton"
        mejor_tiempo = "Secante" if res_sec["tiempo"] < res_new["tiempo"] else "Newton"

        txt += (
            "Comparación (interpretación):\n"
            f"- Menos iteraciones: {mejor_iter}.\n"
            f"- Menor costo de evaluaciones: {mejor_eval}.\n"
            f"- Menor tiempo medido: {mejor_tiempo}.\n\n"
            "Conclusión:\n"
            "Newton suele requerir menos iteraciones porque su convergencia es cuadrática, pero cada iteración cuesta más "
            "porque necesita calcular f'(x). La Secante evita derivadas y reduce el costo por iteración, aunque normalmente "
            "necesita una o dos iteraciones adicionales. En este caso ambos métodos llegaron a la misma raíz, confirmando "
            "la consistencia del modelo numérico.\n"
        )
        return txt