// @vitest-environment jsdom

import { render, screen, waitFor } from '@testing-library/react';

import { App } from './App';

describe('App', () => {
  it('renders camera tiles and backend status from the preload bridge', async () => {
    window.aseeViewerApi = {
      fetchSnapshot: vi.fn(async () => ({
        baseUrl: 'http://127.0.0.1:8765',
        cameras: [0, 2],
        status: { running: true },
        overlayText: {
          caption: '観測中',
          prediction: '静穏',
        },
        biometricStatus: {
          running: true,
          ownerEmbeddingLoaded: true,
          ownerPresent: true,
          ownerCount: 1,
          subjectCount: 1,
          peopleCount: 2,
          ownerSeenAgoMs: 120,
          updatedAt: 1234.5,
        },
      })),
      getConfig: vi.fn(() => ({
        title: 'ASEE Viewer',
        backendBaseUrl: 'http://127.0.0.1:8765',
        pollIntervalMs: 2000,
        autoDemo: false,
      })),
    };

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('観測中')).toBeInTheDocument();
    });

    expect(screen.getByText('OWNER PRESENT')).toBeInTheDocument();
    expect(screen.getByText('CAM 0')).toBeInTheDocument();
    expect(screen.getByText('CAM 2')).toBeInTheDocument();
    expect(screen.getByText('2 PEOPLE DETECTED')).toBeInTheDocument();
  });

  it('shows an offline state when the backend bridge rejects', async () => {
    window.aseeViewerApi = {
      fetchSnapshot: vi.fn(async () => {
        throw new Error('connect ECONNREFUSED');
      }),
      getConfig: vi.fn(() => ({
        title: 'ASEE Viewer',
        backendBaseUrl: 'http://127.0.0.1:8765',
        pollIntervalMs: 2000,
        autoDemo: false,
      })),
    };

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('BACKEND OFFLINE')).toBeInTheDocument();
    });

    expect(screen.getByText('connect ECONNREFUSED')).toBeInTheDocument();
  });
});
