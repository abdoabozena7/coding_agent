import tkinter as tk

class Calculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Calculator')
        self.geometry('300x400')
        self.resizable(False, False)
        self.expression = ''
        self.create_widgets()

    def create_widgets(self):
        self.entry = tk.Entry(self, font=('Arial', 20), bd=5, relief=tk.RIDGE, justify='right')
        self.entry.grid(row=0, column=0, columnspan=4, sticky='nsew', padx=5, pady=5)
        # Buttons layout
        btn_texts = [
            '7', '8', '9', '/',
            '4', '5', '6', '*',
            '1', '2', '3', '-',
            '0', '.', '=', '+'
        ]
        for i, txt in enumerate(btn_texts):
            row = 1 + i // 4
            col = i % 4
            btn = tk.Button(self, text=txt, font=('Arial', 18), command=lambda t=txt: self.on_button_click(t))
            btn.grid(row=row, column=col, sticky='nsew', padx=2, pady=2)
        # Configure grid weight
        for i in range(5):
            self.grid_rowconfigure(i, weight=1)
        for i in range(4):
            self.grid_columnconfigure(i, weight=1)

    def on_button_click(self, char):
        if char == '=':
            try:
                # Evaluate the expression safely
                result = eval(self.entry.get())
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, str(result))
            except Exception:
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, 'Error')
        else:
            self.entry.insert(tk.END, char)

if __name__ == '__main__':
    calc = Calculator()
    calc.mainloop()
