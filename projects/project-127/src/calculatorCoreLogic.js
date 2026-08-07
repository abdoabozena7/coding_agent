/**
 * Calculator Core Logic Engine.
 * Manages state, handles number input accumulation, and executes arithmetic operations.
 */
class Calculator {
    constructor() {
        this.currentValue = ''; // Stores displayed/current accumulated number string
        this.previousOperand = null; 
        this.operation = null;  // e.g., '+', '-', '*', '/' 
    }

    clear() {
        this.currentValue = '';
        this.previousOperand = null;
        this.operation = null;
    }

    appendNumber(number) {
        if (this.currentValue === '0' && number !== '.') {
            this.currentValue = String(number);
        } else if (this.currentValue === '' && number !== '.') {
            this.currentValue = String(number);
        } else {
            this.currentValue += String(number);
        }
    }

    appendDecimal() {
        if (!this.currentValue.includes('.')) {
            this.currentValue += '.';
        }
    }

    setOperation(nextOperation) {
        const inputValue = parseFloat(this.currentValue);

        // If an operation was already pending and we are providing a new number, 
        // it implies continuing the calculation (e.g., pressing + then 5, then *).
        if (typeof this.operation === 'string' && !isNaN(inputValue)) {
            this.calculate(parseFloat(nextOperation)); // Calculate intermediate result before changing operation
        } else if (inputValue !== null) {
             // This handles the initial input for a new operation chain.
            this.previousOperand = inputValue;
            this.operation = nextOperation;
        } else {
            return this; // No valid change, return self for chaining
        }
    }

    calculate(nextNumber) {
        if (this.operation === null || this.previousOperand === null) { 
            // Nothing to calculate if no prior operation/operand exists.
            return this.currentValue = '';
        }

        const result = this.performCalculation(
            parseFloat(this.previousOperand),
            nextNumber,
            this.operation
        );

        // Update state for the result and reset previous operand if needed, but keep operation history context.
        this.currentValue = String(Math.round(result * 100000) / 100000); // Limit precision
        this.previousOperand = null; // Calculation completed for this step
        this.operation = ''; // Clear operation after solving, or keep if building a chain (more complex logic needed here).
        return this;
    }

    performCalculation(operand1, operand2, op) {
        switch(op) {
            case '+': return operand1 + operand2;\n            case '-': return operand1 - operand2;\n            case '*': return operand1 * operand2;\n            case '/': 
                if (operand2 === 0) throw new Error(\"Cannot divide by zero\");
                return operand1 / operand2;\n            default: return operand2;\n        }\n    }
}