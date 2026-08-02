import React from 'react';

function CalculatorLogic(props) {
  const performOperation = (operation, a, b) => {
    if (operation === 'add') return a + b;
    else if (operation === 'subtract') return a - b;
    else if (operation === 'multiply') return a * b;
    else if (operation === 'divide' && b !== 0) return a / b;
    return '';
  };

  const handleAdd = () => {
    const num1 = parseFloat(props.num1);
    const num2 = parseFloat(props.num2);
    props.onResult(performOperation('add', num1, num2));
  };

  const handleSubtract = () => {
    const num1 = parseFloat(props.num1);
    const num2 = parseFloat(props.num2);
    props.onResult(performOperation('subtract', num1, num2));
  };

  const handleMultiply = () => {
    const num1 = parseFloat(props.num1);
    const num2 = parseFloat(props.num2);
    props.onResult(performOperation('multiply', num1, num2));
  };

  const handleDivide = () => {
    const num1 = parseFloat(props.num1);
    const num2 = parseFloat(props.num2);
    props.onResult(performOperation('divide', num1, num2));
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Calculator</h1>
      <input
        type='number'
        id='num1'
        placeholder='Number 1'
        value={props.num1}
        onChange={(e) => props.onNumChange('num1', e.target.value)}
      />
      <input
        type='number'
        id='num2'
        placeholder='Number 2'
        value={props.num2}
        onChange={(e) => props.onNumChange('num2', e.target.value)}
      />
      <button onClick={handleAdd}>Add</button>
      <button onClick={handleSubtract}>Subtract</button>
      <button onClick={handleMultiply}>Multiply</button>
      <button onClick={handleDivide}>Divide</button>
      <p>Result: {props.result}</p>
    </div>
  );
}

export default CalculatorLogic;