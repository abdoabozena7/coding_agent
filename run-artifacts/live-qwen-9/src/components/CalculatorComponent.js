// CalculatorComponent.js
import React, { useState } from 'react';

const CalculatorComponent = ({ operation }) => {
  const [input, setInput] = useState('');
  const [result, setResult] = useState(null);

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleCalculate = () => {
    try {
      const [num1, num2] = input.split(',').map(Number);
      switch(operation) {
        case '+':
          setResult(num1 + num2);
          break;
        case '-':
          setResult(num1 - num2);
          break;
        case '*':
          setResult(num1 * num2);
          break;
        case '/':
          setResult(num1 / num2);
          break;
        default:
          setResult('Error');
      }
    } catch (error) {
      setResult('Error');
    }
  };

  return (
    <div>
      <input type='text' value={input} onChange={handleInputChange} />
      <button onClick={handleCalculate}>Calculate</button>
      {result !== null && <p>Result: {result}</p>}
    </div>
  );
};

export default CalculatorComponent;
