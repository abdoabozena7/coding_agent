import tkinter as tk

class Calculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Calculator')
        self.geometry('300x400')
        self.resizable(False, False)
        self._add_widgets()
        self.expression = ''

    def _add_widgets(self):
        self.display = tk.Entry(self, font=('Arial', 20), borderwidth=2, relief='ridge', justify='right')
        self.display.grid(row=0, column=0, columnspan=4, sticky='nsew', padx=5, pady=5)
        btn_cfg = {'font': ('Arial', 18), 'width': 4, 'height': 2, 'borderwidth':1, 'relief':'raised'}
        buttons = [
            ('7', 1, 0), ('8', 1, 1), ('9', 1, 2), ('/', 1, 3),
            ('4', 2, 0), ('5', 2, 1), ('6', 2, 2), ('*', 2, 3),
            ('1', 3, 0), ('2', 3, 1), ('3', 3, 2), ('-', 3, 3),
            ('0', 4, 0), ('.', 4, 1), ('=', 4, 2), ('+', 4, 3),
            ('C', 5, 0),
        ]
        for (text, r, c) in buttons:
            action = lambda char=text: self._on_button_click(char)
            tk.Button(self, text=text, **btn_cfg, command=action).grid(row=r, column=c, padx=2, pady=2)
        # make rows/columns expand equally
        for i in range(6):
            self.rowconfigure(i, weight=1)
        for i in range(4):
            self.columnconfigure(i, weight=1)

    def _on_button_click(self, char):
        if char == 'C':
            self.expression = ''
            self.display.delete(0, tk.END)
        elif char == '=':
            try:
                result = str(eval(self.expression))
                self.display.delete(0, tk.END)
                self.display.insert(tk.END, result)
                self.expression = result
            except Exception:
                self.display.delete(0, tk.END)
                self.display.insert(tk.END, 'Error')
                self.expression = ''
        else:
            self.expression += str(char)
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, self.expression)

if __name__ == '__main__':
    app = Calculator()
    app.mainloop()
