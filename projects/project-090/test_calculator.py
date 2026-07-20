import unittest
from calculator import CalculatorCore

class TestCalculatorCore(unittest.TestCase):
    """Tests basic arithmetic operations and edge case handling."""

    def setUp(self):
        # Initialize fresh core object for each test
        self.calc = CalculatorCore()

    def test_addition_success(self):
        # Test successful addition (2 + 3)
        result = self.calc.add(2, 3)
        self.assertAlmostEqual(result, 5.0)

    def test_subtraction_success(self):
        self.assertAlmostEqual(self.calc.subtract(10.5, 4), 6.5)

    def test_multiplication_success(self):
        self.assertAlmostEqual(self.calc.multiply(5, 2.5), 12.5)

    def test_division_success(self):
        self.assertAlmostEqual(self.calc.divide(10, 4), 2.5)

    def test_division_by_zero(self):
        # Test handling of division by zero (must raise ZeroDivisionError/TypeError)
        with self.assertRaises((ZeroDivisionError, TypeError)):
            self.calc.divide(5, 0)

    def test_non_numeric_input(self):
        # Test graceful handling of invalid input types
        with self.assertRaises(TypeError):
            self.calc.add("a", 5)
        
        with self.assertRaises(TypeError):
            self.calc.divide("x", "y")

if __name__ == '__main__':
    unittest.main()