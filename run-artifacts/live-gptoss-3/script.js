// script.js - Three.js 3D Calculator
// This is a minimal implementation that creates a dark scene with buttons
// and evaluates arithmetic expressions with proper operator precedence.

let scene, camera, renderer, raycaster, mouse;
let buttons = [];
let expression = '';
const displayDiv = document.getElementById('display');

init();
animate();

function init() {
  const canvas = document.getElementById('calcCanvas');
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x111111);

  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 5, 10);
  camera.lookAt(0, 0, 0);

  const ambient = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambient);
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(5, 10, 7);
  scene.add(dirLight);

  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  // create button layout
  const layout = [
    ['7', '8', '9', '/'],
    ['4', '5', '6', '*'],
    ['1', '2', '3', '-'],
    ['C', '0', '=', '+']
  ];
  const size = 1;
  const gap = 0.2;
  const startX = -((layout[0].length * (size + gap)) / 2) + size / 2;
  const startY = ((layout.length * (size + gap)) / 2) - size / 2;

  const geometry = new THREE.BoxGeometry(size, size, 0.5);
  const material = new THREE.MeshStandardMaterial({ color: 0x3333ff });
  const materialPressed = new THREE.MeshStandardMaterial({ color: 0x5555ff });

  layout.forEach((row, rowIdx) => {
    row.forEach((label, colIdx) => {
      const mesh = new THREE.Mesh(geometry, material.clone());
      mesh.position.x = startX + colIdx * (size + gap);
      mesh.position.y = startY - rowIdx * (size + gap);
      mesh.position.z = 0;
      mesh.userData = { label, originalY: mesh.position.y, material, materialPressed };
      scene.add(mesh);
      buttons.push(mesh);
    });
  });

  // event listeners
  window.addEventListener('resize', onWindowResize);
  canvas.addEventListener('pointerdown', onPointerDown);
}

function onWindowResize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

function onPointerDown(event) {
  // calculate mouse position in normalized device coordinates (-1 to +1)
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(buttons);
  if (intersects.length > 0) {
    const btn = intersects[0].object;
    pressButton(btn);
    handleInput(btn.userData.label);
  }
}

function pressButton(btn) {
  // simple press animation: lower Y and change color briefly
  const { originalY, material, materialPressed } = btn.userData;
  btn.position.y = originalY - 0.1;
  btn.material = materialPressed;
  setTimeout(() => {
    btn.position.y = originalY;
    btn.material = material;
  }, 150);
}

function handleInput(label) {
  if (label === 'C') {
    expression = '';
    updateDisplay('0');
    return;
  }
  if (label === '=') {
    try {
      // Evaluate using Function to respect operator precedence
      const result = Function('return ' + expression)();
      updateDisplay(String(result));
      expression = String(result);
    } catch (e) {
      updateDisplay('Error');
      expression = '';
    }
    return;
  }
  // Append digit or operator
  expression += label;
  updateDisplay(expression);
}

function updateDisplay(text) {
  displayDiv.textContent = text;
}

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
