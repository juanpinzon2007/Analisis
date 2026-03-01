from tkinter import ttk


def make_scrollable_treeview(parent, columns, headings=None, col_widths=None, height=12):
    """
    Crea un ttk.Treeview con scrollbar vertical y horizontal.
    Retorna: (tree, container_frame)
    """
    container = ttk.Frame(parent)
    container.rowconfigure(0, weight=1)
    container.columnconfigure(0, weight=1)

    tree = ttk.Treeview(container, columns=columns, show="headings", height=height)

    # Headings
    if headings is None:
        headings = columns

    for i, col in enumerate(columns):
        tree.heading(col, text=headings[i] if i < len(headings) else col)
        w = col_widths[i] if (col_widths and i < len(col_widths)) else 120
        tree.column(col, width=w, anchor="center", stretch=False)  # stretch=False obliga a usar scroll horizontal

    # Scrollbars
    vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    # Layout
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")

    return tree, container