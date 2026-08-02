// Calculator actions implementation
export const ADD = 'ADD';
export const SUBTRACT = 'SUBTRACT';
export const MULTIPLY = 'MULTIPLY';
export const DIVIDE = 'DIVIDE';

export const add = (value) => ({ type: ADD, payload: value });
export const subtract = (value) => ({ type: SUBTRACT, payload: value });
export const multiply = (value) => ({ type: MULTIPLY, payload: value });
export const divide = (value) => ({ type: DIVIDE, payload: value });