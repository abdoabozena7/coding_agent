// NOTE: This file is being generated/updated as per contract.

/**
 * @typedef {Object} RawEvent
 * @property {string} type - The raw event type (e.g., 'mousedown', 'mousemove').
 * @property {DOMHighResTimeStamp} timestamp
 * @property {MouseEvent | TouchEvent | CustomEvent} nativeEvent - The original DOM/Canvas event object.
 */

/**
 * @typedef {Object} AbstractAction
 * @property {string} actionType - A canonical string representing the action (e.g., 'DRAW_START', 'MOVE', 'CLICK').
 * @property {Object.<string, any>} payload - Data associated with the action.
 */

/**
 * Interface for adapting raw DOM/Canvas events into structured AbstractActions.
 * This class handles input normalization and abstraction.
 */
class InputEventAdapter {
    /**
     * @param {HTMLElement} targetElement - The element receiving the event.
     */
    constructor(targetElement) {
        this.target = targetElement;
    }

    /**
     * Converts a raw browser event into an AbstractAction based on context.
     * @param {RawEvent} rawEvent - The captured raw event data.
     * @returns {AbstractAction | null} An abstract action or null if unhandled.
     */
    adapt(rawEvent) {
        // Use the native event object passed in the raw event for coordinate calculation
        const event = rawEvent.nativeEvent;
        if (!event) return null;

        // Calculate coordinates relative to the target element, adjusting for potential clientX/Y inconsistencies if event structure changes.
        const rect = this.target.getBoundingClientRect();
        const clientX = event.clientX - rect.left + (this.target.scrollLeft || 0);
        const clientY = event.clientY - rect.top + (this.target.scrollTop || 0);

        // Basic validation/abstraction logic
        if (!rawEvent.nativeEvent) {
            console.warn("Adapter received an event without a native DOM object.");
            return null;
        }

        switch (rawEvent.type) {
            case 'mousedown':
                // For simplicity, only capture basic coordinates on mousedown
                return {
                    actionType: 'DRAW_START',
                    payload: { x: clientX, y: clientY }
                };
            case 'mousemove':
                // Continuous movement action
                return {
                    actionType: 'MOVE',
                    payload: { x: clientX, y: clientY }
                };
            case 'mouseup':
                // Signal completion/release of interaction
                return {
                    actionType: 'DRAW_END',
                    payload: {}
                };
            // Add more complex logic for touch events, wheel, etc.
            default:
                console.log(`[Adapter] Unhandled raw event type: ${rawEvent.type}`);
                return null;
        }
    }
}

/**
 * Manages the capture of low-level user input from specific DOM/Canvas contexts,
 * translates these raw events into validated AbstractActions, and dispatches
 * them to dedicated handlers (Renderer, LogicCore).
 *
 * This acts as the primary bridge layer for user interaction.
 */
class InteractionManager {
    /**
     * @param {InputEventAdapter} adapter - The utility used to convert raw events.
     * @param {object} rendererHandler - Object with a 'handle' method (Renderer).
     * @param {object} logicCoreHandler - Object with an 'updateState' method (LogicCore).
     */
    constructor(adapter, rendererHandler, logicCoreHandler) {
        if (!adapter || !rendererHandler || !logicCoreHandler) {
            throw new Error("InteractionManager requires valid adapter and handlers.");
        }
        this.adapter = adapter;
        this.renderer = rendererHandler;
        this.logicCore = logicCoreHandler;

        // Bind event listeners to the manager itself or a designated container element
        this.eventListeners = {
            'mousedown': this._handleRawEvent(this),
            'mousemove': this._handleRawEvent(this),
            'mouseup': this._handleRawEvent(this),
            // Add 'touchstart', 'touchmove', etc., as needed
        };
    }

    /**
     * Internal helper to wrap raw event handling, ensuring the correct lifecycle.
     * @param {Function} context - The instance context (this).
     * @returns {EventListener}
 */
    _handleRawEvent(context) {
        return (event) => {
            // 1. Capture Raw Event & Context
            /** @type {RawEvent} */
            const rawEvent = {
                type: event.type,
                timestamp: performance.now(),
                nativeEvent: event // 'event' here is the native DOM event passed by addEventListener
            };
            console.log(`[IM] Captured Raw Event: ${rawEvent.type}`);

            // 2. Adapt & Validate Action
            const abstractAction = this.adapter.adapt(rawEvent);

            if (!abstractAction) {
                return; // Ignore if adaptation fails or action is unhandled
            }
            console.log(`[IM] Abstracted Action: ${abstractAction.actionType}`);

            // 3. Route Call to Handlers
            this._routeAction(abstractAction);
        };
    }

    /**
     * Routes the validated action to both visual and logical layers.
     * @param {AbstractAction} action - The processed, abstract user action.
     */
    _routeAction(action) {
        // A. Route to Renderer (Visual Update)
        if (this.renderer && typeof this.renderer.handle === 'function') {
            try {
                console.log(`[IM] -> Calling Renderer for ${action.actionType}`);
                this.renderer.handle(action);
            } catch (e) {
                console.error("Renderer failed to process action:", e);
            }
        }

        // B. Route to LogicCore (State Update)
        if (this.logicCore && typeof this.logicCore.updateState === 'function') {
            try {
                console.log(`[IM] -> Calling LogicCore for ${action.actionType}`);
                this.logicCore.updateState(action);
            } catch (e) {
                console.error("LogicCore failed to process action:", e);
            }
        }
    }

    /**
     * Initializes and attaches all necessary event listeners to the designated target element.
     * @param {HTMLElement} targetElement - The DOM element to monitor for inputs.
     */
    initialize(targetElement) {
        console.log("[InteractionManager] Initializing input handling on element...");
        Object.keys(this.eventListeners).forEach(eventType => {
            // Use the bound method reference stored in this.eventListeners
            const handler = this.eventListeners[eventType];
            targetElement.addEventListener(eventType, handler);
            console.log(`Attached listener for: ${eventType}`);
        });
    }

    /**
     * Cleanup function to remove event listeners when the component unmounts.
     * @param {HTMLElement} targetElement - The DOM element previously monitored.
     */
    destroy(targetElement) {
        console.log("[InteractionManager] Destroying input handling and removing listeners...");
        Object.keys(this.eventListeners).forEach(eventType => {
            const handler = this.eventListeners[eventType];
            // We must pass the exact reference used during addition to removeEventListener
            targetElement.removeEventListener(eventType, handler);
        });
        this.eventListeners = {}; // Clear references
    }
}

// Exporting components for module consumption (assuming ES Module environment)
export {
    InteractionManager,
    InputEventAdapter
};
