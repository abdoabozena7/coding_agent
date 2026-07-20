# calculator.py - Core calculation logic module
class CalculatorCore:
    """Handles basic arithmetic operations, ensuring floating-point safety."""
    def __init__(self):
        pass # Constructor placeholder

    def add(self, a, b):
        try:
            return float(a) + float(b)
        except ValueError:
            raise TypeError("Inputs must be numeric.")

    def subtract(self, a, b):
        try:
            return float(a) - float(b)
        except ValueError:
            raise TypeError("Inputs must be numeric.")

    def multiply(self, a, b):
        try:
            return float(a) * float(b)
        except ValueError:
            raise TypeError("Inputs must be numeric.")

    def divide(self, a, b):
        try:
            float_b = float(b)
            if float_b == 0.0:
                # Use ZeroDivisionError as is standard practice
                raise ZeroDivisionError("Cannot divide by zero")
            return float(a) / float_b
        except (ValueError, TypeError):
            raise TypeError("Inputs must be numeric.")

print("CalculatorCore initialized successfully.")