import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';

if (typeof HTMLCanvasElement !== 'undefined') {
  Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
    configurable: true,
    value: vi.fn(() => ({
      clearRect: vi.fn(),
      fillText: vi.fn(),
      strokeRect: vi.fn(),
    })),
  });
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
