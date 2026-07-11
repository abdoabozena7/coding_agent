#!/usr/bin/env python3
"""
A simple GUI calculator using Tkinter.

Features
--------
- Basic arithmetic: +, -, *, /
- Clear (C) and backspace (←) buttons
- Keyboard support for digits and operators
- Handles division‑by‑zero and other errors gracefully
"""

import tkinter as tk
from tkinter import ttk

class Calculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calculator")
        self.resizable(False, False)
        self._make_widgets()
        self.bind_events()

        self.expression = ""

    # ------------------------------------------------------------------ UI
    def _make_widgets(self):
        # Display entry (read‑only)
        self.display = ttk.Entry(self, font=("Helvetica", 20), justify="right")
        self.display.grid(row=0, column=0, columnspan=4, sticky="nsew", padx=5, pady=5)
        self.display.configure(state="readonly")

        # Button layout
        # Define button layout (text, row, column, command, optional grid options)
        buttons = [
            ("C", 1, 0, self._clear),
            ("←", 1, 1, self._backspace),
            ("%", 1, 2, lambda _: self._append("%")),
            ("/", 1, 3, lambda _: self._append("/")),
            ("7", 2, 0, lambda _: self._append("7")),
            ("8", 2, 1, lambda _: self._append("8")),
            ("9", 2, 2, lambda _: self._append("9")),
            ("*", 2, 3, lambda _: self._append("*")),
            ("4", 3, 0, lambda _: self._append("4")),
            ("5", 3, 1, lambda _: self._append("5")),
            ("6", 3, 2, lambda _: self._append("6")),
            ("-", 3, 3, lambda _: self._append("-")),
            ("1", 4, 0, lambda _: self._append("1")),
            ("2", 4, 1, lambda _: self._append("2")),
            ("3", 4, 2, lambda _: self._append("3")),
            ("+", 4, 3, lambda _: self._append("+")),
            ("0", 5, 0, lambda _: self._append("0")),
            (".", 5, 1, lambda _: self._append(".")),
            ("=", 5, 2, self._evaluate, {"columnspan": 2}),
        ]

        # Create buttons using tk.Button (which accepts font and width directly)
        for (text, r, c, cmd, extra) in [(*b, {}) if len(b) == 4 else b for b in buttons]:
            btn = tk.Button(self, text=text, command=cmd, font=("Helvetica", 16), width=4)
            btn.grid(row=r, column=c, **extra, sticky="nsew", padx=2, pady=2)

        # Make the grid cells expand evenly
        for i in range(6):
            self.rowconfigure(i, weight=1)
        for i in range(4):
            self.columnconfigure(i, weight=1)

    # -------------------------------------------------------------- events
    def bind_events(self):
        # Allow typing directly into the display
        self.bind("<Key>", self._on_key)

    def _on_key(self, event):
        """Handle keyboard input."""
        char = event.char
        if char.isdigit() or char in "+-*/.%":
            self._append(char)
        elif event.keysym == "Return":
            self._evaluate()
        elif event.keysym == "BackSpace":
            self._backspace()
        elif event.keysym == "Escape":
            self._clear()

    # -------------------------------------------------------------- logic
    def _set_display(self, text):
        self.display.configure(state="normal")
        self.display.delete(0, tk.END)
        self.display.insert(0, text)
        self.display.configure(state="readonly")

    def _append(self, value: str):
        self.expression += value
        self._set_display(self.expression)

    def _clear(self, _=None):
        self.expression = ""
        self._set_display("")

    def _backspace(self, _=None):
        self.expression = self.expression[:-1]
        self._set_display(self.expression)

    def _evaluate(self, _=None):
        try:
            # Replace the percent operator with Python's modulo
            expr = self.expression.replace("%", "/100")
            result = eval(expr, {"__builtins__": {}}, {})
            # Trim trailing .0 for integer results
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            self.expression = str(result)
        except Exception:
            self.expression = "Error"
        finally:
            self._set_display(self.expression)


if __name__ == "__main__":
    Calculator().mainloop()
