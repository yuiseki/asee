// @vitest-environment jsdom

import { render, screen } from '@testing-library/react';

import { App } from './App';

describe('App', () => {
  it('renders the viewer heading and migration focus areas', () => {
    render(<App />);

    expect(screen.getByRole('heading', { name: 'ASEE Viewer' })).toBeInTheDocument();
    expect(screen.getByText('Python backend')).toBeInTheDocument();
    expect(screen.getByText('Electron viewer')).toBeInTheDocument();
    expect(screen.getByText('Adapter boundary')).toBeInTheDocument();
  });
});
