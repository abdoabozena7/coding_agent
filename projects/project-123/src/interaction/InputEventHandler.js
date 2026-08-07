/**
 * @file InputEventHandler.js
 * @description Handles the mapping of virtual button presses into structured event objects 
 *              that the CalculatorCoreLogic can consume.
 */

import { coreLogic } from '../calculator/CoreCalculatorLogic';

export class InputEventHandler {
    constructor(rendererInstance) {
        this.renderer = rendererInstance;
        console.log("InputEventHandler Initialized.");
    }

    /**
     * Processes a single button event, determining if it is a number, operator, or control function.
     * @param {string} buttonId - The ID of the button pressed (e.g., '7', '+', '=').
     * @returns {object|null} An event object for the core logic to handle, or null if irrelevant.
     */
    handleInput(buttonId) {
        let type = null;
        let value = null;

        if (!buttonId || buttonId === 'null') {
            return null; 
        }

        // 1. Number/Decimal Input
        if (!isNaN(parseInt(buttonId)) || buttonId === '.') {
            type = 'input';
            value = buttonId;
        }
        // 2. Operators
        else if (['+', '-', '*', '/'].includes(buttonId)) {
            type = 'operator';
            value = buttonId;
        }
        // 3. Control Functions
        else if (buttonId === '=') {
            type = 'calculate';
            value = null;
        }
        else if (buttonId === 'AC') {
            type = 'clear';
            value = null;
        }
        else if (['+/-', '%'].includes(buttonId)) {
            // These are complex, specialized functions usually handled by the core logic after input.
             type = 'control';
             value = buttonId;
        }


        // --- Execution flow based on determined type ---\n       let resultValue = null;\n        switch (type) {\n            case 'input':\n                coreLogic.handleInput(value);\n                resultValue = coreLogic.getCurrentDisplayValue();\n                break;\n            case 'operator':\n                const opResult = coreLogic.handleOperation(value);\n                resultValue = coreLogic.getCurrentDisplayValue();\n                if (isNaN(opResult)) resultValue = \"Error\"; // Handle division by zero etc.\n                break;\n            case 'calculate':\n                const calcResult = coreLogic.handleEquals();\n                // Rounding/Formatting needed for display\n                resultValue = isNaN(calcResult) ? \"Error\" : String(Math.floor(Math.abs(calcResult) * 10) / 10);\n                break;\n            case 'clear':\n                coreLogic.clear();\n                resultValue = \"0\";\n                break;\n             case 'control':\n                 // For control buttons, we assume the core logic handles the state change internally\n                 if (buttonId === '+/-') { \n                     // Logic would handle sign flip...\n                 } else if (buttonId === '%') {\n                     // Logic would handle percent calculation...\n                 }\n                 resultValue = coreLogic.getCurrentDisplayValue(); // Re-read value after internal state change simulation\n                 break;\n\n            default:\n                resultValue = null;\n        }\n\n        // Visual feedback update (calling the renderer to show current state)\n        if(this.renderer) {\n            this.renderer.updateDisplay(resultValue || coreLogic.getCurrentDisplayValue());\n        }\n\n        return { type: type, value: value }; // Returning structured event payload is better practice\n    }
}