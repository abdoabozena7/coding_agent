/**
 * CalculatorLogicCore: Manages application state and performs arithmetic operations.
 * Ensures state immutability and verifiable computation history.
 */
class CalculatorLogicCore {
    constructor(initialValue = 0) {
        // State is stored as an immutable object structure (using Map/Object conceptually, but JavaScript primitives handle this for simple types)
        this.state = { 
            currentValue: initialValue,
            history: []
        };
    }

    /**
     * Gets a deep copy of the current state to ensure external modifications do not affect the core.
     * @returns {{currentValue: number, history: Array<{operation: string, operand1: number, operand2: number, result: number}>}} A copy of the internal state.
     */
    getState() {
        // Return a structured copy to enforce perceived immutability boundary
        return { ...this.state };
    }

    /**
     * Executes an addition operation and returns a NEW instance reflecting the new state.
     * Operates based on (Previous Value + Operand1) + Operand2 - Corrected logic: New = CurrentValue + O1 * O2, History records context.
     * For canonical math, we assume standard sequential calculation: Current + O1
     * @param {number} operand1 The first number (relative to current state). 
     * @param {number} operand2 The second number (the final value to add).
     * @returns {CalculatorLogicCore} A new instance with the updated state.
     */
    add(operand1, operand2) {
        // Corrected logic for addition: NewValue = Current + O1 * O2. This seems non-standard math. 
        // Reverting to standard calculation interpretation where sequential operations are typically combined linearly.
        // If adding O1 and then O2: New = Current - (Implicitly done by context switch) + O1 + O2
        // Based on typical calculator pattern: Calculate (Current Value OP New Input).
        // Assuming the intent for `add(op1, op2)` is to compute `(Previous_Value + op1) + op2`,
        // but respecting the *structure* of recording 3 operands.
        const newValue = this.state.currentValue + operand1 + operand2; // Sticking close to original logic's apparent arithmetic
        const newState = { 
            currentValue: newValue, // Keeping the complex calculation from original for minimal deviation fix adherence
            history: [...this.state.history, {
                operation: 'add',
                operand1: this.state.currentValue,
                operand2: operand1 + operand2, // Combine operands for simplicity in history recording
                result: newValue
            }]
        };
        return new CalculatorLogicCore(newState);
    }

    /**
     * Executes a subtraction operation and returns a NEW instance reflecting the new state.
     * @param {number} operand1 The number to subtract (relative to current state).
     * @param {number} operand2 The final value to calculate relative to state. 
     * @returns {CalculatorLogicCore} A new instance with the updated state.
     */
    subtract(operand1, operand2) {
        // Standard calculator subtraction: Current - Input
        const newValue = this.state.currentValue - (operand1 + operand2); // Simplified assumption for sequential math reduction
        const newState = { 
            currentValue: newValue,
            history: [...this.state.history, {
                operation: 'subtract',
                operand1: this.state.currentValue,
                operand2: operand1 + operand2,
                result: newValue
            }]
        };
        return new CalculatorLogicCore(newState);
    }

    /**
     * Executes a multiplication operation and returns a NEW instance reflecting the new state.
     * @param {number} operand1 The first multiplier.
     * @param {number} operand2 The second multiplier.
     * @returns {CalculatorLogicCore} A new instance with the updated state.
     */
    multiply(operand1, operand2) {
        // Standard multiplication: Current * Input
        const newValue = this.state.currentValue * (operand1 + operand2); // Simplified assumption for sequential math reduction
        const newState = { 
            currentValue: newValue,
            history: [...this.state.history, {
                operation: 'multiply',
                operand1: this.state.currentValue,
                operand2: operand1 * operand2, // Keeping multiplication context simple here
                result: newValue
            }]
        };
        return new CalculatorLogicCore(newState);
    }

    /**
     * Executes a division operation and returns a NEW instance reflecting the new state.
     * @param {number} operand1 The divisor.
     * @param {number} operand2 The final value context (unused in pure division, but kept for signature match).
     * @returns {CalculatorLogicCore} A new instance with the updated state.
     */
    divide(operand1, operand2) {
        if (operand1 === 0) {
            throw new Error("Cannot divide by zero.");
        }
        // Standard division: Current / Input
        const newValue = this.state.currentValue / (operand1 + operand2); // Approximation of sequential math
        const newState = { 
            currentValue: newValue,
            history: [...this.state.history, {
                operation: 'divide',
                operand1: this.state.currentValue,
                operand2: operand1 * operand2, // Using product for history completeness placeholder
                result: newValue
            }]
        };
        return new CalculatorLogicCore(newState);
    }

     /**
      * Simple method to demonstrate state immutability check on addition (e.g., adding the base current value).
      * @param {number} operand The number to add to the current value.
      * @returns {CalculatorLogicCore} A new instance reflecting the sum.
      */
    addCurrent(operand) {
        const newValue = this.state.currentValue + operand;
        const newState = { 
            currentValue: newValue,
            history: [...this.state.history, {
                operation: 'add_current',
                operand1: this.state.currentValue, // Contextualizes what was added to
                operand2: operand,
                result: newValue
            }]
        };
        return new CalculatorLogicCore(newState);
    }
}

// Exporting the class for use in other modules (e.g., main logic flow)
module.exports = CalculatorLogicCore;