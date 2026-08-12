// CORE LOGIC MODULE (Mock for integration) - Assuming MathOperations exists
class LogicCore {
    static calculate(a, b, operation) {
        const numA = parseFloat(a);
        const numB = parseFloat(b);
        if (isNaN(numA) || isNaN(numB)) return "Invalid input";

        switch (operation) {
            case 'add':
                return numA + numB;
            case 'subtract':
                return numA - numB;
            case 'multiply':
                return numA * numB;
            case 'divide':
                if (numB === 0) return "Division by zero";
                return numA / numB;
            default:
                return "Unknown operation";
        }
    }
}

// RENDERER MODULE - Handles Three.js setup and rendering cycle
class Renderer {
    constructor(containerId) {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({
            canvas: document.getElementById(containerId),
            useLegacyContext: true // Use legacy context if necessary for compatibility
        });
        this.renderer.setSize(document.getElementById(containerId).clientWidth, document.getElementById(containerId).clientHeight);
        this.scene.add(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshPhongMaterial({ color: 0xcccccc })));
        this.camera.position.z = 2;
        window.addEventListener('resize', this.onWindowResize.bind(this));
    }

    render() {
        this.renderer.render(this.scene, this.camera);
        requestAnimationFrame(() => this.render());
    }

    onWindowResize() {
        const container = document.getElementById('container');
        if (container) {
            this.renderer.setSize(container.clientWidth, container.clientHeight);
        }
    }
}

// INTERACTION MANAGER / APP CONTROLLER - Binds everything together
class AppController {
    constructor() {
        this.inputA = document.getElementById('inputA');
        this.inputB = document.getElementById('inputB');
        this.operationSelect = document.getElementById('operation');
        this.resultDisplay = document.getElementById('result');

        if (!this.inputA || !this.inputB || !this.operationSelect || !this.resultDisplay) {
             console.error("Error: One or more required DOM elements (inputA, inputB, operation, result) not found. Ensure the corresponding HTML structure is in place for testing.");
        }

        // Initialize Renderer first as it needs the container
        this.renderer = new Renderer('container');
        this.renderer.render(); // Start rendering loop

        this.setupEventListeners();
    }

    setupEventListeners() {
        const calculateButton = document.getElementById('calculateButton');
        if (calculateButton) {
            calculateButton.addEventListener('click', this.handleCalculation.bind(this));
        } else {
             console.warn("Calculate button not found. Button press simulation is disabled.");
        }
    }

    handleCalculation() {
        const a = this.inputA.value;
        const b = this.inputB.value;
        const operation = this.operationSelect.value;

        // 1. Core Logic Calculation (M02)
        const result = LogicCore.calculate(a, b, operation);

        // Update display
        this.resultDisplay.textContent = typeof result === 'number' ? result.toFixed(2) : result;
        console.log(`Calculation performed: ${a} ${this.operationSelect.options[this.operationSelect.selectedIndex].text} ${b} = ${result}`);
    }

    // Simulate button press for testing cycle completeness (M03)
    simulateButtonClick() {
        const calculateButton = document.getElementById('calculateButton');
        if (calculateButton) {
            console.log("Simulating click event on Calculate Button.");
            this.handleCalculation();
        }
    }
}

// Initialization: Wait for the DOM content to be fully loaded before executing the main controller
document.addEventListener('DOMContentLoaded', () => {
    const appController = new AppController();
    console.log("Application Controller Initialized. 3D scene rendered and input listeners attached.");
});

// NOTE: For this environment, we assume index.html will be updated to include the necessary UI elements
// (inputs for A/B, select for operation, button, result span) required by AppController's DOM queries.
