import React from 'react';

function CalculatorUI(props) {
  return (
    <div style={{ padding: '20px' }}>
      <h1>Calculator</h1>
      <input
        type='number'
        id='num1'
        placeholder='Number 1'
        onChange={(e) => props.onNumChange('num1', e.target.value)}
      />
      <input
        type='number'
        id='num2'
        placeholder='Number 2'
        onChange={(e) => props.onNumChange('num2', e.target.value)}
      />
      <button onClick={() => props.onOperation('add')}>Add</button>
      <button onClick={() => props.onOperation('subtract')}>Subtract</button>
      <button onClick={() => props.onOperation('multiply')}>Multiply</button>
      <button onClick={() => props.onOperation('divide')}>Divide</button>
      <p>Result: {props.result}</p>
    </div>
  );
}

export default CalculatorUI;