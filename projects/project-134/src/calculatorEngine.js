
/**
 * Calculator Engine Module
 * Handles the core mathematical logic.
 */
export class CalculatorEngine {
    constructor() {
        // State holds: current display value, pending operation (if any), and history of inputs.
        this.state = {
            displayValue: '0',
            pendingOperation: null, // e.g., '+', '-', '*', '/'
            firstOperand: null,      // The first number before an operator was pressed
            history: []             // Array to store the sequence of operations/results
        };
    }

    /**
     * Clears all state variables.
     */
    clear() {
        this.state = { displayValue: '0', pendingOperation: null, firstOperand: null, history: [] };
        console.log('Calculator cleared.');
    }

    /**
     * Appends a number digit to the current display value.
     * @param {string} digit - The digit (0-9) pressed.
     */
    appendNumber(digit) {
        if (this.state.displayValue === '0' && digit !== '.') {
            this.state.displayValue = digit;
        } else if (digit === '.' && this.state.displayValue.includes('.')) {
            // Ignore if decimal already present
            return; 
        } else {
            this.state.displayValue += digit;
        }
    }

    /**
     * Sets the pending operation after a number is displayed and an operator button is pressed.
     * @param {string} operation - The arithmetic operator ('+', '-', '*', '/'').
     */
    setOperation(operation) {
        const currentValue = parseFloat(this.state.displayValue);
        if (isNaN(currentValue)) return; // Should not happen if controls are followed

        // If an operation was already pending, calculate the intermediate result first.
        if (this.state.pendingOperation && this.state.firstOperand !== null) {
            const intermediateResult = this._performCalculation(
                parseFloat(this.state.firstOperand),
                parseFloat(this.state.displayValue), // The number currently displayed
                this.state.pendingOperation
            );
            if (intermediateResult === null) { 
                this.clear(); return; } // Error state handled by clearing
        }

        // Set up for the new calculation chain
        this.state.firstOperand = currentValue.toString();
        this.state.pendingOperation = operation;
        this.state.history.push(`${this.state.firstOperand} ${operation}`);
    }

    /**
     * Calculates the final result when '=' is pressed.
     * @returns {number|null} The final computed result, or null on error.
     */
    calculateResult() {
        if (!this.state.pendingOperation || this.state.firstOperand === null) {
            // If no operation was set, just return the current number as a string/float.
            const result = parseFloat(this.state.displayValue); 
            this.state.history = []; // Clear history if only number input is pressed before '='
            return isNaN(result) ? null : result;
        }
        
        const secondOperand = parseFloat(this.state.displayValue);
        if (isNaN(secondOperand)) { 
             console.error("Cannot calculate: Second operand is not a valid number.");
            return null; 
        }

        // Perform the final calculation
        const result = this._performCalculation(parseFloat(this.state.firstOperand), secondOperand, this.state.pendingOperation);
        
        if (result === null) {
             console.error("Final Calculation Error.");
            return null;
        }
        
        // Update state after successful calculation
        this.state.displayValue = result.toString();
        this.state.history.push(`${this.state.firstOperand} ${this.state.pendingOperation} ${secondOperand} = ${result}`);
        // Reset first operand for chain calculations, but keep the result as the new starting point.
        this.state.firstOperand = result.toString(); 
        this.state.pendingOperation = null; // Calculation complete
        return result;
    }

    /**
     * Handles percentage calculation (e.g., 200 %).
     * @param {number} value - The number to calculate the percentage of.
     */
    calculatePercentage(value) {
        // For simplicity matching standard calculators, this treats it as division by 100 relative to current display.
        const result = parseFloat(this.state.displayValue) / 100; 
        this.state.displayValue = result.toString();
        this.state.history.push(`${parseFloat(this.state.displayValue)} %`); 
        return result;
    }

    /**
     * Helper method to perform the actual arithmetic.
     * @param {number} op1 - The first operand.
     * @param {number} op2 - The second operand.
     * @param {string} operation - The operator symbol.
     * @returns {number|null} The result or null if error occurs.
     */
    _performCalculation(op1, op2, operation) {
        try {
            switch (operation) {
                case '+': return op1 + op2;
                case '-': return op1 - op2;
                case '*': return op1 * op2;
                case '/': 
                    if (op2 === 0) throw new Error("Division by Zero");
                    return op1 / op2;
            }
        } catch(e) {
            console.error("Calculation failed:", e);
            return null;
        }
    }

    /**
     * Provides a snapshot of the current immutable state.
     */
    getState() {
        // Return a deep clone to prevent external mutation of internal state
        return JSON.parse(JSON.stringify({ ...this.state }));
    }
}
