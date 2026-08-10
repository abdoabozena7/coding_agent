// src/inputController.js

/**
 * Handles user input from various sources and relays events to the main application logic.
 */
class InputController {
    constructor(calculatorEngine, visualizer) {
        this.engine = calculatorEngine;
        this.visualizer = visualizer;
        console.log("InputController initialized: Ready to manage user interactions.");
    }

    /**
     * Processes a raw input event (e.g., button click, keyboard entry).
     * @param {string} inputData The data received from the UI.
     */
    handleInput(inputData) {
        console.log(`[InputController] Received input: ${inputData}`);
        // TODO: Implement parsing and state management based on inputData format.
        if (this.isValidOperation(inputData)) {
            this.engine.processInput(inputData);
        }
    }

    /**
     * Updates the 3D visualization based on the current operational state.
     * @param {any} result The calculated result or intermediate state to visualize.
     */
    updateVisualization(result) {
        console.log("[InputController] Notifying visualizer of update...");
        this.visualizer.renderState(result);
    }

    /**
     * Checks if the input data represents a valid actionable operation.
     * @param {string} data The input string.
     * @returns {boolean} True if valid, false otherwise.
     */
    isValidOperation(data) {
        // Mock validation logic
        return !!data && typeof data === 'string' && !isNaN(parseFloat(data));
    }
}

export default InputController;