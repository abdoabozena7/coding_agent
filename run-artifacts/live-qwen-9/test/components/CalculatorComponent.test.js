// CalculatorComponent.test.js
import React from 'react';
import { render, fireEvent, screen } from '@testing-library/react';
import CalculatorComponent from '../CalculatorComponent';

describe('CalculatorComponent', () => {
  test('renders input and button', () => {
    render(<CalculatorComponent />);
    expect(screen.getByPlaceholderText(/input/i)).toBeInTheDocument();
    expect(screen.getByText(/calculate/i)).toBeInTheDocument();
  });

  test('updates result on calculate', () => {
    render(<CalculatorComponent />);
    fireEvent.change(screen.getByPlaceholderText(/input/i), { target: { value: '[2, 3]' } });
    fireEvent.click(screen.getByText(/calculate/i));
    expect(screen.getByText(/result: 5/i)).toBeInTheDocument();
  });

  test('handles errors', () => {
    render(<CalculatorComponent />);
    fireEvent.change(screen.getByPlaceholderText(/input/i), { target: { value: '[2, 0]' } });
    fireEvent.click(screen.getByText(/calculate/i));
    expect(screen.getByText(/result: Error/i)).toBeInTheDocument();
  });
});