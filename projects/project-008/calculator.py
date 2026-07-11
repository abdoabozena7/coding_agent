import tkinter as tk
import ast
import operator as op

# supported operators for safe eval
allowed_operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.Mod: op.mod,
}

def safe_eval(expr):
    """Safely evaluate a mathematical expression using ast"""
    def _eval(node):
        if isinstance(node, ast.Num):  # <number>
            return node.n
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](_eval(node.operand))
        raise ValueError('Unsupported expression')
    return _eval(ast.parse(expr, mode='eval').body)

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
            'MC','MR','M+','M-',
            'C', '(', ')', '←',
            '7', '8', '9', '/',
            '4', '5', '6', '*',
            '1', '2', '3', '-',
            '0', '.', '^', '+',
            '='
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
        if char == 'C':
            # Clear the entry
            self.entry.delete(0, tk.END)
        elif char == '←':
            # Backspace: delete last character
            current = self.entry.get()
            self.entry.delete(0, tk.END)
            self.entry.insert(tk.END, current[:-1])
        elif char == '^':
            # Insert exponent operator as '**' for Python evaluation
            self.entry.insert(tk.END, '**')
        elif char == '=':
            try:
                # Evaluate the expression safely using our safe_eval function
                expr = self.entry.get()
                result = safe_eval(expr)
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, str(result))
            except Exception:
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, 'Error')
        else:
            # For all other buttons, just insert their character
            self.entry.insert(tk.END, char)

if __name__ == '__main__':
    calc = Calculator()
    calc.mainloop()
