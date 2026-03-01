import tkinter as tk
from tkinter import ttk

from interfaz.ejercicio1_view import Ejercicio1View
from interfaz.ejercicio2_view import Ejercicio2View
from interfaz.ejercicio3_view import Ejercicio3View
from interfaz.ejercicio4_view import Ejercicio4View
from interfaz.ejercicio5_view import Ejercicio5View

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Proyecto Métodos Numéricos")
        self.geometry("1300x760")
        self.minsize(1200, 700)

        # Layout: menú (izq) + contenido (der)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.sidebar = ttk.Frame(self, padding=10)
        self.sidebar.grid(row=0, column=0, sticky="ns")

        self.content = ttk.Frame(self, padding=10)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.rowconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)

        # Título / Menú
        ttk.Label(self.sidebar, text="Navegación", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 10))

        ttk.Button(self.sidebar, text="Ejercicio 1: Bisección", command=self.show_ejercicio1).pack(fill="x", pady=5)

        # Botones placeholders para ejercicios futuros
        ttk.Button(self.sidebar, text="Ejercicio 2: Falsa Posición", command=self.show_ejercicio2).pack(fill="x", pady=5)
        ttk.Button(self.sidebar, text="Ejercicio 3: Punto Fijo", command=self.show_ejercicio3).pack(fill="x", pady=5)
        ttk.Button(self.sidebar, text="Ejercicio 4: Newton-Raphson", command=self.show_ejercicio4).pack(fill="x", pady=5)
        ttk.Button(self.sidebar, text="Ejercicio 5: Secante", command=self.show_ejercicio5).pack(fill="x", pady=5)

        ttk.Separator(self.sidebar).pack(fill="x", pady=10)
        ttk.Button(self.sidebar, text="Salir", command=self.destroy).pack(fill="x")

        self.current_view = None
        self.show_ejercicio1()

    def _set_view(self, view_cls):
        # borrar vista actual
        if self.current_view is not None:
            self.current_view.destroy()
        # crear nueva vista
        self.current_view = view_cls(self.content)
        self.current_view.grid(row=0, column=0, sticky="nsew")

    def show_ejercicio1(self):
        self._set_view(Ejercicio1View)

    def show_placeholder(self):
        self._set_view(PlaceholderView)
    def show_ejercicio2(self):
     self._set_view(Ejercicio2View)
    def show_ejercicio3(self):
     self._set_view(Ejercicio3View)
    def show_ejercicio4(self):
     self._set_view(Ejercicio4View)
    def show_ejercicio5(self):
        self._set_view(Ejercicio5View)

class PlaceholderView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=20)
        ttk.Label(self, text="Este ejercicio aún no está implementado.", font=("Segoe UI", 14)).pack(anchor="w")
        ttk.Label(self, text="Cuando terminemos el Ejercicio 1, aquí conectamos el siguiente módulo.").pack(anchor="w", pady=(10, 0))