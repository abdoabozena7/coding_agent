// Calculator reducer implementation
import { ADD, SUBTRACT, MULTIPLY, DIVIDE } from '../actions';

const initialState = {
  result: 0,
};

const calculatorReducer = (state = initialState, action) => {
  switch (action.type) {
    case ADD:
      return { ...state, result: state.result + action.payload };
    case SUBTRACT:
      return { ...state, result: state.result - action.payload };
    case MULTIPLY:
      return { ...state, result: state.result * action.payload };
    case DIVIDE:
      if (action.payload !== 0) {
        return { ...state, result: state.result / action.payload };
      } else {
        return { ...state, error: 'Division by zero' };
      }
    default:
      return state;
  }
};

export default calculatorReducer;