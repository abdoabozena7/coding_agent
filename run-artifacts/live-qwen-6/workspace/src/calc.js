// CalculatorComponent handles user input and performs calculations.

import React, { useState } from 'react';

const CalculatorComponent = () => {
  const [input, setInput] = useState('');
  const [result, setResult] = useState(null);

  const calculate = (operation) => {
    let num1 = parseFloat(input.split(',')[0]);
    let num2 = parseFloat(input.split(',')[1]);
    if (!isNaN(num1) && !isNaN(num2)) {
      switch(operation) {
        case 'add': setResult(num1 + num2); break;
        case 'subtract': setResult(num1 - num2); break;
        case 'multiply': setResult(num1 * num2); break;
        case 'divide': setResult(num1 / num2); break;
        default: setResult(null);
      }
    } else {
      setResult(null);
    }
  };

  return (
    <div>
      <input type="text" value={input} onChange={(e) => setInput(e.target.value)} placeholder="Enter numbers separated by comma" />
      <button onClick={() => calculate('add')}>Add</button>
      <button onClick={() => calculate('subtract')}>Subtract</button>
      <button onClick={() => calculate('multiply')}>Multiply</button>
      <button onClick={() => calculate('divide')}>Divide</button>
      {result !== null && <p>Result: {result}</p>}
    </div>
  );
};

export default CalculatorComponent;