import React, { useState } from 'react';

function App() {
  const [num1, setNum1] = useState('');
  const [num2, setNum2] = useState('');
  const [result, setResult] = useState('');

  const performOperation = (operation) => {
    const a = parseFloat(num1);
    const b = parseFloat(num2);
    if (!isNaN(a) && !isNaN(b)) {
      switch (operation) {
        case 'add':
          setResult(a + b);
          break;
        case 'subtract':
          setResult(a - b);
          break;
        case 'multiply':
          setResult(a * b);
          break;
        case 'divide':
          if (b !== 0) {
            setResult(a / b);
          } else {
            setResult('Error: Division by zero');
          }
          break;
        default:
          setResult('Invalid operation');
      }
    } else {
      setResult('Please enter valid numbers');
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Calculator</h1>
      <input
        type='number'
        value={num1}
        onChange={(e) => setNum1(e.target.value)}
        placeholder='Number 1'
      />
      <input
        type='number'
        value={num2}
        onChange={(e) => setNum2(e.target.value)}
        placeholder='Number 2'
      />
      <button onClick={() => performOperation('add')}>Add</button>
      <button onClick={() => performOperation('subtract')}>Subtract</button>
      <button onClick={() => performOperation('multiply')}>Multiply</button>
      <button onClick={() => performOperation('divide')}>Divide</button>
      <p>Result: {result}</p>
    </div>
  );
}

export default App;