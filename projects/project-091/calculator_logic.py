class CalculatorError(Exception):
    """Custom exception for calculator related errors."""
    pass

class AdvancedCalculator:
    """
    A robust calculator engine supporting standard arithmetic operations 
    for both integer and float inputs, with built-in error handling.
    """
    def __init__(self):
        # No internal state needed for stateless calculations
        pass

    @staticmethod
    def _validate_input(value: str) -> float | None:
        """Validates if the string can be converted to a float."""
        try:
            return float(value.replace(' ', '')) # Allow spaces just in case, though unnecessary for typical calculator input flow
        except ValueError:
            return None

    def calculate(self, num1_str: str, num2_str: str) -> tuple[float | str, bool]:
        """
        Performs the calculation based on the provided operator.

        Args:
            num1_str: String representation of the first number.
            num2_str: String representation of the second number.
            operator: The arithmetic operator (+, -, *, /).

        Returns:
            A tuple (result, success_flag): 
            - result is the float outcome if successful, or an error message string.
            - success_flag is True if calculation was successful, False otherwise.
        """
        # 1. Input Validation & Conversion
        num1 = self._validate_input(num1_str)
        num2 = self._validate_input(num2_str)

        if num1 is None:
            return "Error: Invalid number format for the first input.", False
        if num2 is None:
            return "Error: Invalid number format for the second input.", False

        try:
            # 2. Perform Calculation based on operator
            if operator == '+':
                result = num1 + num2
            elif operator == '-':
                result = num1 - num2
            elif operator == '*':
                result = num1 * num2
            elif operator == '/':
                # 3. Division by Zero Check
                if num2 == 0:
                    return "Error: Division by Zero", False
                result = num1 / num2
            else:
                return f"Error: Unknown operator '{operator}'.", False

            return result, True

        except Exception as e:
            # Catch any unexpected errors during calculation
            return f"An unexpected error occurred: {e}", False

if __name__ == '__main__':
    # Simple self-test to ensure basic functionality before unit testing setup
    calc = AdvancedCalculator()
    
    print("--- Basic Tests ---")
    
    # Addition (Int)
    res, success = calc.calculate("5", "3")
    print(f"5 + 3: Result={res}, Success={success}") # Expected: 8.0, True

    # Subtraction (Float)
    res, success = calc.calculate("10.5", "2.1")
    print(f"10.5 - 2.1: Result={res}, Success={success}") # Expected: 8.4, True
    
    # Multiplication (Int)
    res, success = calc.calculate("-4", "6")
    print(f"-4 * 6: Result={res}, Success={success}") # Expected: -24.0, True

    # Division (Float result)
    res, success = calc.calculate("10", "3")
    print(f"10 / 3: Result={res}, Success={success}") # Expected: 3.33..., True

    print("\n--- Edge Case Tests ---")

    # Invalid Input Test
    res, success = calc.calculate("abc", "5")
    print(f"'abc' + 5: Result={res}, Success={success}") # Expected: Error message, False

    # Division by Zero Test
    res, success = calc.calculate("10", "0")
    print(f"10 / 0: Result={res}, Success={success}") # Expected: Error: Division by Zero, False