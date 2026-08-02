// src/App.js
import React from 'react';
import CalculatorComponent from './CalculatorComponent';
import DisplayComponent from './DisplayComponent';

const App = () => {
  return (
    <div className="App">
      <CalculatorComponent />
      <DisplayComponent />
    </div>
  );
};

export default App;
