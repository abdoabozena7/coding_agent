/**
 * @file CoreCalculatorLogic.js
 * @description Handles the core calculation logic, independent of any visual framework (Three.js).
 */

class CalculatorCoreLogic {
    constructor() {
        this.currentValue = 0;
        this.pendingOperation = null; // '+', '-', '*', '/';
        this.firstOperand = null;
        console.log("Calculator Core Logic Initialized.");
    }

    /**
     * Updates the current value based on a button press (number or decimal).
     * @param {string} char - The character pressed ('0'-'9', '.').
     */
    handleInput(char) {
        if (!isNaN(parseInt(char)) || char === '.') {
            const number = isNaN(parseFloat(char)) ? char : parseFloat(char);

            if (this.currentValue === 0 && char !== '.') {
                this.currentValue = number;
            } else if (char === '.' && this.currentValue === 0) {
                 // Do nothing, current value remains 0 until first digit is entered
            } else if (char === '.') {
                if (!String(this.currentValue).includes('.')) {
                    this.currentValue += char;
                }
            } else {
                this.currentValue = (this.currentValue * 10) + parseFloat(char);
            }
        }
    }

    /**
     * Executes a pending operation with the current operand.
     * @param {string} operation - The operator pressed ('+', '-', '*', '/').
     */
    handleOperation(operation) {
        if (this.firstOperand === null || this.pendingOperation === null) {
            // If no first operand or operation is set, treat the new operation as setting the pending op
            this.pendingOperation = operation;
            this.firstOperand = this.currentValue;
            return this.currentValue; // No change yet
        }

        let result;
        const a = this.firstOperand;
        const b = this.currentValue;

        switch (operation) {
            case '+':
                result = a + b;
                break;
            case '-':
                result = a - b;
                break;
            case '*':
                result = a * b;
                break;
            case '/':
                if (b === 0) {
                    console.error("Division by zero.");
                    return NaN; // Indicate error state
                }
                result = a / b;
                break;
            default:
                result = this.currentValue;
        }

        this.currentValue = result;
        this.firstOperand = null; // Operation is now complete, reset first operand context
        this.pendingOperation = null;
        return result;
    }

    /**
     * Handles the equals button (=). Calculates the final result.
     * @returns {number|string} The calculated value or an error message.
     */
    handleEquals() {
        if (this.firstOperand === null || this.pendingOperation === null) {
            return Math.floor(this.currentValue * 10) / 10; // Round if no op was pending
        }

        let result;
        const a = this.firstOperand;
        const b = this.currentValue;

        switch (this.pendingOperation) {
            case '+':
                result = a + b;
                break;
            case '-':
                result = a - b;
                break;
            case '*':
                result = a * b;
                break;
            case '/':
                if (b === 0) {
                    return "Error: Div by Zero";
                }
                result = a / b;
                break;
        }

        this.currentValue = result;
        this.firstOperand = null;
        this.pendingOperation = null;

        // Limit precision for display consistency
        return Math.floor(Math.abs(result) * 10) / 10;
    }


    /**
     * Resets all calculator state variables.
     */
    clear() {
        this.currentValue = 0;
        this.pendingOperation = null;
        this.firstOperand = null;
        console.log("Calculator State Cleared.");
    }

    /**
     * Get the current displayed value, formatted as a string.
     */
    getCurrentDisplayValue() {
        // Use toFixed(1) for clean display if it's an integer or half-integer result (typical calculator format).
        const roundedValue = Math.floor(this.currentValue * 10) / 10;

        if (isNaN(roundedValue)) return "Error";
        
        // Standard handling: If it's an integer or simple float, return number string representation
        return String(roundedValue);
    }
}

export const coreLogic = new CalculatorCoreLogic();