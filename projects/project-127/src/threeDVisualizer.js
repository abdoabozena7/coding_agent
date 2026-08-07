import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.159.0/build/three.module.js';

// --- Global Setup Variables ---
let scene, camera, renderer;
const CALCULATOR_SIZE = { width: 8, height: 2, depth: 4 }; // Defining basic dimensions

function init() {
    // 1. SCENE SETUP
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0); // Light grey background

    // 2. CAMERA SETUP
    const aspect = window.innerWidth / window.innerHeight;
    camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
    camera.position.set(0, 3, 8); // Position camera to view the calculator nicely
    camera.lookAt(0, 1, 0);

    // 3. RENDERER SETUP
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // 4. LIGHTING
    // Ambient Light (soft general light)
    const ambientLight = new THREE.AmbientLight(0x606060, 2); // soft white light
    scene.add(ambientLight);

    // Directional Light (simulating sun or main source)
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
    directionalLight.position.set(5, 10, 7.5);
    scene.add(directionalLight);

    // 5. CALCULATOR MODEL (The main shell)
    const geometry = new THREE.BoxGeometry(CALCULATOR_SIZE.width, CALCULATOR_SIZE.height, CALCULATOR_SIZE.depth);
    // Use a material that reacts to light for better visual confirmation
    const material = new THREE.MeshPhongMaterial({ color: 0xcccccc }); 
    const calculatorShell = new THREE.Mesh(geometry, material);
    scene.add(calculatorShell);

    // Add a floor plane for grounding the object (optional but good practice)
    const floorGeometry = new THREE.PlaneGeometry(20, 20);
    const floorMaterial = new THREE.MeshPhongMaterial({ color: 0xaaaaaa, side: THREE.DoubleSide });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = Math.PI / 2;
    floor.position.y = -0.1;
    scene.add(floor);

    // 6. EVENT LISTENERS AND INITIAL RENDER CALL
    window.addEventListener('resize', onWindowResize, false);
    animate();
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

let timeElapsed = 0; // Time tracker for animation
function animate() {
    requestAnimationFrame(animate);
    
    // Update elapsed time (more reliable than Date object calls)
    timeElapsed += 0.016; 

    // Find the calculator shell mesh and animate it
    const calculatorShell = scene.children.find(c => c instanceof THREE.Mesh && c.geometry.type === 'BoxGeometry');
    if (calculatorShell) {
        // Simple continuous rotation proving animation is working
        calculatorShell.rotation.y += 0.005;
        calculatorShell.rotation.x = Math.sin(timeElapsed * 0.5) * 0.1; // Adding subtle tilt based on time
    }

    renderer.render(scene, camera);
}

// Start the visualization only after all resources are loaded/available globally
init();