// test/App.test.js
import React from 'react';
import { render, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom/extend-expect';
import App from '../src/App';
import DisplayComponent from '../src/DisplayComponent';


describe('App', () => {
  it('renders correctly', () => {
    const { getByText, getByRole } = render(<App />);
    expect(getByRole('heading', { name: 'Polished Calculator' })).toBeInTheDocument();
    expect(getByText('Calculate')).toBeInTheDocument();
    expect(getByText('0')).toBeInTheDocument();
  });

  it('updates display on button click', () => {
    const { getByText, getByRole } = render(<App />);
    fireEvent.click(getByText('1'));
    expect(getByText('1')).toBeInTheDocument();
  });

  it('displays result from CalculatorComponent', () => {
    const mockResult = '5';
    jest.spyOn(CalculatorComponent, 'useCalculator').mockReturnValue(mockResult);
    const { getByText } = render(<App />);
    expect(getByText(mockResult)).toBeInTheDocument();
  });
});
