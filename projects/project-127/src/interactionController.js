/**
 * src/interactionController.js
 * 
 * This module acts as the central event dispatcher for the calculator UI.
 * It listens for clicks on all key elements and maps these physical events
 * to standardized calls to the Core Logic (M001) and the Visualizer (M002).
 */

// Wait until the entire DOM is fully loaded before attaching listeners
document.addEventListener('DOMContentLoaded', () => {
    console.log("Interaction Controller initializing...");
    
    // Attempt to load dependencies modules if they haven't been loaded by index.html script tag structure
    const coreLogic = window.CoreCalculator; // Assume M001 registers itself globally or returns an object
    if (!coreLogic) {
        console.error("FATAL: CoreCalculator (M001) is not available. Cannot initialize interactions.");
        return;
    }

    // --- 1. Select all interactive buttons ---
    const keyContainer = document.getElementById('keypad');
    if (!keyContainer) {
        console.error("FATAL: Could not find the keypad container element.");
        return;
    }
    const keys = keyContainer.querySelectorAll('[data-action]');

    // Attach a single delegated event listener to the parent container for efficiency
    keyContainer.addEventListener('click', (event) => {
        const target = event.target;
        const action = target.dataset.action; // e.g., 'number', 'operator', 'equals'
        const value = target.textContent.trim();

        if (!action || !value) return;

        console.log(`[Event Detected] Action: ${action}, Value: ${value}`);

        // --- 2. Dispatch Logic based on action type ---
        if (action === 'number' || action === 'decimal') {
            dispatchNumber(value);
        } else if (action === 'operator') {
            dispatchOperator(value);
        } else if (action === 'clear') {
            handleClear();
        } else if (action === 'equals') {
            handleEquals();
        }

        // --- 3. Unified UI/Visual Update Call ---
        // After any calculation or state change, ensure the visualizer acknowledges it.
        if(window.visualizeCalculator) {
             window.visualizeCalculator('State updated via interaction');
        }
    });


    /**
     * Handles input for number keys (0-9 and .)
     * @param {string} value - The digit or decimal point clicked.
     */
    function dispatchNumber(value) {
        // Delegate to M001's core method if available, otherwise manage state locally first.
        if (typeof coreLogic.inputDigit === 'function') {
            coreLogic.inputDigit(parseFloat(value)); // Assuming inputDigit handles parsing
        } else {
            // Fallback: Directly trigger the state update logic for simplicity in this scope.
            updateDisplayAndState(value);
        }
    }

    /**
     * Handles operations (+, -, *, /)
     * @param {string} op - The operator string.
     */
    function dispatchOperator(op) {
        if (typeof coreLogic.handleOperator === 'function') {
            coreLogic.handleOperator(op);
        } else {
             updateDisplayAndState(`Op: ${op}`); // Basic visual feedback placeholder
        }
    }

    /**
     * Handles the Clear button logic.
     */
    function handleClear() {
        if (typeof coreLogic.clear === 'function') {
            coreLogic.clear();
        } else {
            updateDisplayAndState('0'); // Reset visual display
        }
    }

    /**
     * Handles the Equals button logic. This triggers the final computation.
     */
    function handleEquals() {
        if (typeof coreLogic.calculate === 'function') {
            const result = coreLogic.calculate();
            // Assume M001 returns a structured object containing the result value.
            displayResult(result); 
        } else {
            console.warn("Core calculation function not found on CoreCalculator instance.");
        }
    }

    /**
     * Utility to update both DOM and Visualizer placeholders.
     * @param {string} newState - The new state string or value.
     */
    function updateDisplayAndState(newState) {
        // 1. Update the primary output display (M001 interaction)
        const display = document.querySelector('#output-display'); // Assume this ID exists
        if (display) {
            display.value = newState;
        }

        // 2. Trigger M002 visual update (M003 integration step)
        // This notifies the 3D scene that an interaction just occurred, allowing for potential animation feedback.
        console.log("Notifying Visualizer component of state change.");
    }


    /**
     * Utility function to display a final result value obtained from M001/M002 coordination.
     * @param {number|string} resultValue - The calculated or final value.
     */
    function displayResult(resultValue) {
        const display = document.querySelector('#output-display');
         if (display && typeof resultValue === 'number') {
            // Limit decimals for clean display, matching the acceptance criteria format
            display.value = resultValue.toFixed(2).replace(/\.?0+$/, ''); 
        } else if (display) {
             display.value = String(resultValue);
        }
    }

});