// This is the updated DisplayComponent implementation with polished design and integration with CalculatorAPI
import React, { useState } from 'react';
import { useCalculatorAPI } from '../api/CalculatorAPI';

const DisplayComponent = ({ initialText }) => {
  const [displayText, setDisplayText] = useState(initialText);
  const { calculate } = useCalculatorAPI();

  function handleInput(event) {
    setInput(event.target.value);
    setDisplayText(calculate(event.target.value));
  }

  return (
    <div className="display-container">
      <h1>Updated Dynamic Display Component</h1>
      <input 
        type="text" 
        value={displayText} 
        onChange={handleInput} 
        className="display-input"
      />
      <p className="display-text">{calculate(displayText)}</p>
    </div>
  );
};

export default DisplayComponent;
