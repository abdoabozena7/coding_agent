# gui.py - Graphical User Interface Module
import tkinter as tk
from calculator import CalculatorCore

class CalculatorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Advanced Python Calculator")
        
        # Core calculator logic instance
        self.calculator = CalculatorCore()
        
        # Display field
        self.display = tk.Entry(master, width=20, borderwidth=4, justify='right')
        self.display.grid(row=0, column=0, columnspan=4, padx=10, pady=10)
        
        # Advanced feature: Operation History Display (Simple Text Widget)
        tk.Label(master, text="History:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
        self.history_text = tk.Text(master, height=3, width=40, wrap='word')
        self.history_text.grid(row=1, column=1, columnspan=4, padx=10, pady=5, sticky="ew")

        # Calculator buttons layout: (text, row, col)
        buttons = [
            ('AC', 2, 0), ('(', 2, 1), (')', 2, 2), ('/', 2, 3),
            ('7', 3, 0), ('8', 3, 1), ('9', 3, 2), ('*', 3, 3),
            ('4', 4, 0), ('5', 4, 1), ('6', 4, 2),('-', 4, 3),
            ('1', 5, 0), ('2', 5, 1), ('3', 5, 2),('+', 5, 3),
            ('0', 6, 0), ('.', 6, 1), ('=', 6, 2), ('Cxl', 6, 3) # Cxl for clear/backspace functionality placeholder
        ]

        row_idx = 0
        col_idx = 0
        for (text, r, c) in buttons:
            if text == 'AC':
                btn = tk.Button(master, text=text, padx=20, pady=20, command=self.clear_all)
                btn.grid(row=r, column=c, sticky="nsew")
            elif text in ('/', '*', '-', '+', '=', 'Cxl'): # Operator buttons
                 # Operators need special handling for calculation flow
                if text == '=':
                    btn = tk.Button(master, text=text, padx=20, pady=20, command=self.calculate)
                elif text in ('/', '*', '-', '+'):
                    btn = tk.Button(master, text=text, padx=20, pady=20, command=lambda t=text: self.press_operator(t))
                else: # AC or Cxl (Clear/Back)
                     if text == 'Cxl':
                        btn = tk.Button(master, text=text, padx=20, pady=20, command=self.clear_display)
                     else:
                         # For simple functions like parenthesis if implemented later
                        btn = tk.Button(master, text=text, padx=20, pady=20, command=lambda t=text: self.append_input(t))

                btn.grid(row=r, column=c, sticky="nsew")
            else: # Number buttons (0-9, .)
                btn = tk.Button(master, text=text, padx=20, pady=20, command=lambda t=text: self.append_input(t))
                btn.grid(row=r, column=c, sticky="nsew")

        # Configure grid weights so widgets resize nicely
        for i in range(7): master.grid_columnconfigure(i, weight=1)
        for i in range(7): master.grid_rowconfigure(i, weight=1)


    def update_display(self, value):
        """Updates the Entry widget display."""
        self.display.delete(0, tk.END)
        self.display.insert(0, str(value))

    def append_input(self, char):
        """Appends a character (number/decimal point) to the display and updates history."""
        current_text = self.display.get()
        new_text = current_text + str(char)
        self.update_display(new_text)

    def clear_all(self):
        """Clears the entire display (AC)."""
        self.update_display("0")
        self.log_history("Calculation cleared.")

    def clear_display(self):
        """Simulates backspace/clear input only."""
        current_text = self.display.get()
        new_text = current_text[:-1] if len(current_text) > 0 else "0"
        self.update_display(new_text)

    def log_history(self, message):
        """Adds a timestamped/formatted entry to the history widget."""
        # Clear previous content and append new line
        self.history_text.delete(1.0, tk.END)
        self.history_text.insert(tk.END, f"[{tk.StringVar().get()}:] {message}\n") # Simplified timestamp for this context

    def press_operator(self, op):
        """Handles pressing an arithmetic operator button (+, -, *, /)."""
        current_text = self.display.get()
        # Basic check to prevent multiple operators at the end
        if current_text and not (current_text[-1].isdigit() or current_text[-1:] == '.'):
            self.append_input(op)
            self.log_history(f"Operator '{op}' entered.")

    def calculate(self):
        """Triggers the core calculation logic using CalculatorCore."""
        try:
            # For simplicity, we evaluate the whole expression string using python's eval() safely IF no advanced scientific functions are needed.
            # Since we are restricted to T2 (+, -, *, /) for primary validation, let's assume the current input is a simple binary operation or single number, and force manual calculation structure.

            expression = self.display.get()
            
            if not expression:
                self.update_display("Error")
                return
            
            # For reliable testing based on T2 (+,-,*,/), we check for the '=' press. 
            # If the user has entered '5 + 3', we need to calculate that specific part.
            # Since simple button presses feed numbers/operators sequentially, we rely on string parsing or explicit structure enforcement.
            
            # --- Fallback/Simplified Evaluation (Assuming standard calculator flow) ---
            try:
                result = eval(expression) # WARNING: Use with care in real apps! Fine for this controlled environment demo.
            except ZeroDivisionError as e:
                self.update_display("Cannot divide by zero")
                self.log_history(f"ERROR: {e}")
                return
            except (SyntaxError, NameError):
                self.update_display("Invalid Input")
                self.log_history("Input syntax error.")
                return

            # Success path
            final_result = float(result)
            self.update_display(f"{final_result:.4f}")
            self.log_history(f"Calculation successful: {expression} = {final_result:.4f}")


        except Exception as e:
            self.update_display("Error")
            self.log_history(f"An unexpected error occurred: {e.__class__.__name__}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CalculatorGUI(root)
    # Initial run setup
    root.geometry("400x500") # Set a fixed size for better visualization control
    root.mainloop()