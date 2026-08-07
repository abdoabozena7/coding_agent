/**
 * @file THREEJSCoreRenderer.js
 * @description Handles the rendering of the calculator interface using Three.js. 
 *              The structure and geometry have been refined for a more realistic, mounted aesthetic.
 */

import * as THREE from 'three';

const RENDERER_OPTIONS = {
    // Base dimensions defining the scale of the virtual device
    DEVICE_WIDTH: 2.5, // units
    DEVICE_HEIGHT: 6, // units
    DEVICE_THICKNESS: 0.1,
    PIXEL_SIZE: 0.3, // Physical size of one key segment (controls spacing and scale)
};

export class THREEJSCoreRenderer {
    /** @type {THREE.Scene} */
    scene = new THREE.Scene();
    /** @type {THREE.PerspectiveCamera} */
    camera = null;
    /** @type {THREE.WebGLRenderer} */
    renderer = null;

    constructor() {
        this.initScene();
        console.log("THREEJSCoreRenderer Initialized with refined geometry.");
    }

    initScene() {
        // Camera setup
        this.camera = new THREE.PerspectiveCamera(75, 1, 0.1, 10);
        this.camera.position.set(0, RENDERER_OPTIONS.DEVICE_HEIGHT * 0.8 - 0.2, 3.5); // Adjusted camera view

        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth * 0.9, window.innerHeight * 0.7); // Use relative size for embedding
        document.getElementById('calculator-container').appendChild(this.renderer.domElement);

        // Scene lighting and background
        this.scene.background = new THREE.Color(0xeeeeee);
        const ambientLight = new THREE.AmbientLight(0xffffff, 1.5); // Soft overall light
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
        directionalLight.position.set(5, 5, 5).normalize();
        this.scene.add(directionalLight);

        // Base structure for the entire device
        this._createDeviceBase();

        // Keypad Structure (This is where the main aesthetic changes are applied)
        this._createKeypadStructure();
    }

    /**
     * Creates the outer casing and display panel base.
     */
    _createDeviceBase() {
        // 1. Outer Body/Casing
        const bodyGeometry = new THREE.BoxGeometry(RENDERER_OPTIONS.DEVICE_WIDTH, RENDERER_OPTIONS.DEVICE_HEIGHT, RENDERER_OPTIONS.DEVICE_THICKNESS);
        const bodyMaterial = new THREE.MeshPhongMaterial({ color: 0x333333 }); // Dark grey metal look
        this.deviceBody = new THREE.Mesh(bodyGeometry, bodyMaterial);
        this.scene.add(this.deviceBody);

        // 2. Display Screen Area (Placeholder)
        const screenDepth = RENDERER_OPTIONS.DEVICE_THICKNESS * 0.8;
        const screenWidth = RENDERER_OPTIONS.DEVICE_WIDTH * 0.95;
        const screenHeight = RENDERER_OPTIONS.DEVICE_HEIGHT * 0.4;

        // The actual visible display panel face (set slightly recessed)
        const screenGeometry = new THREE.BoxGeometry(screenWidth, screenHeight, 0.1);
        // Use a material that can be programmatically changed via color/emissive for the value display
        this.displayMaterial = new THREE.MeshPhongMaterial({ color: 0x0a0a0a, shininess: 50 });
        this.displayScreen = new THREE.Mesh(screenGeometry, this.displayMaterial);

        // Position the display slightly raised and centered (Y position adjusted for visual alignment)
        this.displayScreen.position.set(0, RENDERER_OPTIONS.DEVICE_HEIGHT * 0.8 - screenHeight / 2 + 0.1, 0.1 + screenDepth / 2);
        this.deviceBody.add(this.displayScreen);

        // Assign a name to the display mesh for potential future lookups/debugging
        this.displayScreen.name = 'LCD_Display';
    }


    /**
     * Creates the entire keypad structure with refined grid layout and spacing.
     */
    _createKeypadStructure() {
        const KEY_COUNT = 4; // Max columns
        // Key Pitch controls the spacing between centers of keys (including the gap)
        const KEY_PITCH = RENDERER_OPTIONS.DEVICE_WIDTH / (KEY_COUNT + 1); 
        const keyGeometryParams = new THREE.BoxGeometry(KEY_PITCH * 0.95, RENDERER_OPTIONS.DEVICE_HEIGHT / 6.2, RENDERER_OPTIONS.DEVICE_THICKNESS * 0.8);
        // Use a slightly reflective material for the keys
        const keyMaterial = new THREE.MeshPhongMaterial({ color: 0xaaaaaa, shininess: 30 });

        // Key layout map (Kept as is, assuming manual refinement achieved good spacing)
        const keyMap = [
            ['AC', '+/-', '%'], // Row 0
            ['7', '8', '9', '/'],  // Row 1
            ['4', '5', '6', '*'],  // Row 2
            ['1', '2', '3', '-'],  // Row 3
            ['.', '=', null] // Row 4 (null is a placeholder for alignment)
        ];

        this.keyGroup = new THREE.Group();
        this.scene.add(this.keyGroup);

        let rowOffsetY = RENDERER_OPTIONS.DEVICE_HEIGHT * 0.95;
        let colOffsetX = RENDERER_OPTIONS.DEVICE_WIDTH * 0.1; // Start slightly in from left edge

        for (let r = 0; r < keyMap.length; r++) {
            const row = keyMap[r];
            colOffsetX = RENDERER_OPTIONS.DEVICE_WIDTH * 0.1;

            for (let c = 0; c < row.length && row[c] !== undefined; c++) {
                const buttonId = row[c];
                
                // Simplified calculation for grid placement based on indices:
                let posX = colOffsetX + (c * KEY_PITCH) - (KEY_PITCH / 2);
                // Y spacing is heuristic, keeping the original relative positioning logic intact.
                let posY;
                if (r === 0) { posY = RENDERER_OPTIONS.DEVICE_HEIGHT * 0.95; } // Top row start
                else if (r === 1) { posY = RENDERER_OPTIONS.DEVICE_HEIGHT * 0.65; } 
                else if (r === 2) { posY = RENDERER_OPTIONS.DEVICE_HEIGHT * 0.35; }
                else if (r === 3) { posY = RENDERER_OPTIONS.DEVICE_HEIGHT * 0.05; }
                else { posY = -0.2 }; // Bottom row

                const buttonMesh = new THREE.Mesh(keyGeometryParams, keyMaterial);
                buttonMesh.position.set(posX, posY, 0);
                buttonMesh.userData = { id: buttonId, row: r, col: c };
                this.keyGroup.add(buttonMesh);
            }
        }
    }


    /**
     * Updates the visual representation of a button press (visual feedback).
     * @param {THREE.Mesh} keyButton - The mesh to highlight.
     */
    highlightButton(keyButton) {
        // Flash the button darker when pressed
        const originalColor = new THREE.Color(0xaaaaaa);
        keyButton.material.color.copy(originalColor);

        keyButton.material.emissive.setHex(0x333333); // Add a slight emissive glow on press
        
        setTimeout(() => {
            // Restore original state after flash delay
            keyButton.material.emissive.setHex(0x000000); 
            keyButton.material.color.copy(originalColor);
        }, 150);
    }


    /**
     * Updates the display text with the current number string, simulating an LCD refresh.
     * @param {string} valueString - The value to show on screen.
     */
    updateDisplay(valueString) {
        const display = this.displayScreen; 

        if (!display || !this.displayMaterial) {
             console.warn("Renderer: Display mesh not found or material missing.");
             return;
        }
        
        // 1. Update the text/color state (Simulated LCD refresh by changing color intensity)
        const cleanValue = valueString === "Error" ? "ERROR" : String(valueString);

        if (cleanValue.startsWith('E')) { // Error condition handling
            this.displayMaterial.color.setHex(0xff0000); // Red for error
            this.displayMaterial.emissive.setHex(0x880000);
        } else if (!isNaN(parseFloat(cleanValue))) {
             this.displayMaterial.color.setHex(0x00ff00); // Green/White for number
             this.displayMaterial.emissive.setHex(0x003300);
        } else {
            this.displayMaterial.color.setHex(0xffffff); // Default white
            this.displayMaterial.emissive.setHex(0x000000);
        }
        
        // In a real engine, text rendering would use a texture mapped onto the mesh. 
        // Here, we rely on the color/emissive change to visually confirm an update occurred.
        console.log(`[Renderer] Display updated successfully: ${cleanValue}`); 
    }

    render() {
        requestAnimationFrame(() => this.render());
        this.renderer.render(this.scene, this.camera);
    }
}