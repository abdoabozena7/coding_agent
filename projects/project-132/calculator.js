/**
 * Calculator Logic Engine (State Management & Pure Math)
 * This module handles all computational logic and state tracking, 
 * completely independent of the DOM structure. It is designed to be a single source of truth
 * for the calculator's current state (display value, first operand, pending operation).
 */

class CalculatorEngine {
    constructor() {
        // State Variables
        this._currentValue = '0'; // What is currently being typed or displayed.
        this._firstOperand = null; // The first number in the calculation (e.g., 5 in 5+3)
        this._operator = null;    // The pending operation ('+', '-', '*', '/')
        this._awaitingSecondOperand = false; // True if we just entered an operator or =
    }

    /**
     * Retrieves the current display string value.
     * @returns {string} The number to be displayed on screen.
     */
    getCurrentDisplay() {
        return this._currentValue === 'Error' ? 'Error' : this._currentValue;
    }

    /**
     * Checks if the engine is in its initial, default state (e.g., after AC).
     * @returns {boolean} True if no significant operations have occurred.
     */
    isInitialState() {
        return this._firstOperand === null && this._operator === null;
    }

    /**
     * Resets the entire calculator state (equivalent to pressing AC).
     */
    clear() {
        this._currentValue = '0';
        this._firstOperand = null;
        this._operator = null;
        this._awaitingSecondOperand = false;
    }

    /**
     * Appends a digit or decimal point to the current display value.
     * Handles initial state and preventing multiple decimals.
     * @param {string} input - The character ('0'-'9', '.').
     */
    appendDigit(input) {
        if (this._currentValue === 'Error') {
            this.clear(); // Clear error first
        }

        // If the engine is awaiting a second operand, start a new number sequence
        if (this._awaitingSecondOperand && !['.', '+', '-', '*', '/'].includes(input)) {
             this._currentValue = input === '0' ? '0' : this._currentValue + input;
             this._awaitingSecondOperand = false; // We started typing again
             return;
        }

        // If current is '0' and input is not '.', replace it. Otherwise, append.
        if (this._currentValue === '0' && input !== '.') {
            this._currentValue = input;
        } else if (input === '.' && this._currentValue.includes('.')) {
            // Ignore multiple decimals
            return;
        } else {
            this._currentValue += input;
        }

        // If appending digits, we are definitely not awaiting a second operand anymore.
        this._awaitingSecondOperand = false;
    }

    /**
     * Sets the operation and potentially saves the current number as the first operand.
     * @param {string} operator - The mathematical operator ('+', '-', '*', '/').
     */
    setOperation(operator) {
        const inputValue = parseFloat(this._currentValue);

        // If this is the very first operation, store the input value
        if (this.isInitialState() && !isNaN(inputValue)) {
            this._firstOperand = inputValue;
        } 
        // If an operator was already set and we are providing a new number, calculate the result instantly
        else if (this._firstOperand !== null && this._operator !== null) {
             this.calculateResult(); // Execute pending calculation using the newly entered value
             this._firstOperand = parseFloat(this.getCurrentDisplay()); // New start for next chain
        }

        // Store state variables
        this._operator = operator;
        this._awaitingSecondOperand = true; 
    }

    /**
     * Calculates and sets the final result based on the pending operation.
     * This function must be called when '=' is pressed or an operator is used after a calculation.
     */
    calculateResult() {
        if (this._firstOperand === null || this._operator === null) {
            return; // Not enough information to calculate
        }

        const secondOperand = parseFloat(this._currentValue);
        let result = 0;

        // Perform calculation based on the stored operator
        switch (this._operator) {
            case '+':
                result = this._firstOperand + secondOperand;
                break;
            case '-':
                result = this._firstOperand - secondOperand;
                break;
            case '*':
                result = this._firstOperand * secondOperand;
                break;
            case '/':
                if (secondOperand === 0) {
                    this.currentValue = 'Error'; // Handle division by zero
                    this._firstOperand = null;
                    this._operator = null;
                    return;
                }
                result = this._firstOperand / secondOperand;
                break;
        }

        // Format result: Limit to 10 decimal places and ensure it is a string.
        const formattedResult = parseFloat(result.toFixed(10)).toString();
        this.currentValue = formattedResult === '0' ? '0' : formattedResult;
        
        // Reset for chaining: The result becomes the new first operand/start value
        this._firstOperand = parseFloat(formattedResult);
        this._operator = null; // Operation completed

        // We do NOT set awaitingSecondOperand to true, as a user might immediately press an operator again.
    }
}

/**
 * Global instance of the calculator engine, accessible by the UIController in index.html.
 */
const calculatorLogic = new CalculatorEngine();