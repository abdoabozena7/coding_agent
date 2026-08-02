// Service Layer implementation
import { createStore } from 'redux';
import calculatorReducer from './reducers/calculatorReducer';

const store = createStore(calculatorReducer);

export const performCalculation = (operation, a, b) => {
  return new Promise((resolve, reject) => {
    switch (operation) {
      case 'add':
        resolve(a + b);
        break;
      case 'subtract':
        resolve(a - b);
        break;
      case 'multiply':
        resolve(a * b);
        break;
      case 'divide':
        if (b !== 0) {
          resolve(a / b);
        } else {
          reject('Division by zero error');
        }
        break;
      default:
        reject('Invalid operation');
    }
  });
};