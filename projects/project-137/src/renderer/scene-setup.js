// src/renderer/scene-setup.js

import * as THREE from 'three';

/**
 * Sets up the Three.js scene, camera, lighting, and renders a static representation of the calculator keypad.
 */
export function setupScene() {
    // 1. Scene Setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xcccccc);

    // 2. Camera Setup (PerspectiveCamera is standard for UI/world views)
    const camera = new THREE.PerspectiveCamera(
        75, 
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );
    camera.position.set(0, 2, 5);
    camera.lookAt(0, 0, 0);

    // 3. Renderer Setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight * 0.8); // Make it slightly smaller than the viewport height to leave room for other UI elements if needed
    // Assuming an element with id='container' exists in the main HTML file.
    const container = document.getElementById('container');
    if (!container) {
        console.error("Could not find target DOM element with id 'container'. Renderer initialization failed.");
        return { scene, renderer, camera }; // Return empty/partial setup if DOM is missing
    }
    renderer.domElement.id = "three-d-calculator";
    container.appendChild(renderer.domElement);

    // Handle window resizing
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight * 0.8);
    });

    // 4. Lighting Setup
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.5); // Soft white light
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
    directionalLight.position.set(5, 10, 7.5);
    scene.add(directionalLight);

    // 5. Create Keypad Geometry (Static Representation)
    const keypadGroup = new THREE.Group();
    const keyMaterial = new THREE.MeshLambertMaterial({ color: 0xaaaaaa }); // Slate gray for buttons
    const backgroundMaterial = new THREE.MeshLambertMaterial({ color: 0x333333 }); // Dark surface
    const keyDepth = 0.1; 
    const keySizeX = 1.5;
    const keySizeY = 1.5;

    // A. Backing surface/housing for the keypad
    // Simplified dimensions based on structure: 4 wide, 3 high.
    const housingWidth = 4 * keySizeX + 2; 
    const housingHeight = 3 * keySizeY + 1.5; 
    const housingGeometry = new THREE.BoxGeometry(housingWidth, housingHeight, keyDepth);
    const housingMesh = new THREE.Mesh(housingGeometry, backgroundMaterial);
    housingMesh.position.set(0, 0, -0.5); // Positioned slightly in front of the origin
    keypadGroup.add(housingMesh);

    // Define Key positions manually for a clear static grid approximation (3 keys wide, ~4 rows high)
    const keyPositions = [
        { x: -1.5, y: 1.0, z: 0 }, // Top Left (AC/Memory spot area)
        { x: 0, y: 1.0, z: 0 },  // Top Middle
        { x: 1.5, y: 1.0, z: 0 }, // Top Right (Operator area)
        
        { x: -1.5, y: 0.0, z: 0 }, // Row 2 Left
        { x: 0, y: 0.0, z: 0 },  // Row 2 Center
        { x: 1.5, y: 0.0, z: 0 },  // Row 2 Right
        
        { x: -1.5, y: -1.0, z: 0 }, // Row 3 Left
        { x: 0, y: -1.0, z: 0 },  // Row 3 Center
        { x: 1.5, y: -1.0, z: 0 }  // Row 3 Right (Operator area)
    ];

    const keyGeometry = new THREE.BoxGeometry(keySizeX * 0.9, keySizeY * 0.9, keyDepth);

    keyPositions.forEach((pos, index) => {
        // Assign unique identifiers/data properties that InteractionManager can read later if needed.
        const keyMesh = new THREE.Mesh(keyGeometry, keyMaterial);
        keyMesh.userData = { type: "button", id: `btn-${index}` };
        // Center the mesh on the calculated position (adjusting for slightly smaller geometry)
        keyMesh.position.set(pos.x, pos.y + (keySizeY * 0.45), pos.z);
        keypadGroup.add(keyMesh);
    });


    // Add the group to the scene
    scene.add(keypadGroup);

    // 6. Animation Loop & Rendering Logic
    const animate = () => {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
    };

    // Initial render call
    animate();

    return { scene, renderer, camera };
}
