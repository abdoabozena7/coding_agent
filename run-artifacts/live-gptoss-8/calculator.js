// calculator.js
// Minimal Three.js setup and calculator logic

function initThree() {
  const canvas = document.getElementById('three-canvas');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.z = 5;

  const geometry = new THREE.TorusKnotGeometry(1, 0.3, 100, 16);
  const material = new THREE.MeshNormalMaterial();
  const torus = new THREE.Mesh(geometry, material);
  scene.add(torus);

  const light = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(light);

  function animate() {
    requestAnimationFrame(animate);
    torus.rotation.x += 0.01;
    torus.rotation.y += 0.01;
    renderer.render(scene, camera);
  }
  animate();
}

// Calculator UI overlay
function createCalculatorUI() {
  const container = document.createElement('div');
  container.id = 'calc-container';
  container.innerHTML = `
    <input id="calc-input" type="text" placeholder="Expression" style="width:150px;"/>
    <button id="calc-equals">=</button>
    <div id="calc-result" style="margin-top:5px; color:#0f0; font-weight:bold;"></div>
  `;
  document.body.appendChild(container);

  document.getElementById('calc-equals').addEventListener('click', () => {
    const expr = document.getElementById('calc-input').value;
    const result = evaluateExpression(expr);
    document.getElementById('calc-result').textContent = result;
  });
}

function evaluateExpression(expr) {
  try {
    // Very simple and unsafe eval for demonstration; in production use a proper parser.
    const func = new Function('return ' + expr);
    const result = func();
    return String(result);
  } catch (e) {
    return 'Error';
  }
}

// Export for Node testing
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { evaluateExpression };
}

// Initialize when running in browser
if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', () => {
    if (typeof THREE !== 'undefined') {
      initThree();
    }
    createCalculatorUI();
  });
}
