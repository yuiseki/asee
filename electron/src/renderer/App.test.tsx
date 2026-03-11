// @vitest-environment jsdom

import { act, render, screen, waitFor } from '@testing-library/react';
import React from 'react';

import { App } from './App';
import * as webrtcClient from './webrtc-client';

describe('App', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('renders camera tiles and backend status from the preload bridge', async () => {
    window.aseeViewerApi = {
      fetchSnapshot: vi.fn(async () => ({
        baseUrl: 'http://127.0.0.1:8765',
        cameras: [0, 2],
        status: { running: true, transport: 'mjpeg' as const },
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

  it('polls at the configured interval without spawning duplicate timers', async () => {
    vi.useFakeTimers();
    const fetchSnapshot = vi.fn(async () => ({
      baseUrl: 'http://127.0.0.1:8765',
      cameras: [0, 2, 4, 6],
      status: { running: true, transport: 'mjpeg' as const },
      overlayText: {
        caption: 'OBSERVING',
        prediction: 'CALM',
      },
      biometricStatus: {
        running: true,
        ownerEmbeddingLoaded: true,
        ownerPresent: true,
        ownerCount: 1,
        subjectCount: 0,
        peopleCount: 1,
        ownerSeenAgoMs: 120,
        updatedAt: 1234.5,
      },
    }));

    window.aseeViewerApi = {
      fetchSnapshot,
      getConfig: vi.fn(() => ({
        title: 'ASEE Viewer',
        backendBaseUrl: 'http://127.0.0.1:8765',
        pollIntervalMs: 5000,
        autoDemo: false,
      })),
    };

    render(
      <React.StrictMode>
        <App />
      </React.StrictMode>,
    );

    await act(async () => {
      await Promise.resolve();
    });

    expect(fetchSnapshot).toHaveBeenCalled();

    const initialCallCount = fetchSnapshot.mock.calls.length;

    await act(async () => {
      await vi.advanceTimersByTimeAsync(4900);
    });

    expect(fetchSnapshot).toHaveBeenCalledTimes(initialCallCount);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });

    expect(fetchSnapshot).toHaveBeenCalledTimes(initialCallCount + 1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });

    expect(fetchSnapshot).toHaveBeenCalledTimes(initialCallCount + 2);
  });

  it('renders WebRTC video tiles when backend transport is webrtc', async () => {
    vi.spyOn(webrtcClient, 'connectWebRtcFeeds').mockResolvedValue({
      close: vi.fn(),
    });

    window.aseeViewerApi = {
      fetchSnapshot: vi.fn(async () => ({
        baseUrl: 'http://127.0.0.1:8765',
        cameras: [0, 2],
        status: { running: true, transport: 'webrtc' as const },
        overlayText: {
          caption: '観測中',
          prediction: '静穏',
        },
        biometricStatus: {
          running: true,
          ownerEmbeddingLoaded: true,
          ownerPresent: true,
          ownerCount: 1,
          subjectCount: 0,
          peopleCount: 1,
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
      expect(screen.getByTestId('webrtc-feed-0')).toBeInTheDocument();
    });

    expect(screen.getByTestId('webrtc-feed-0').tagName).toBe('VIDEO');
    expect(screen.getByTestId('webrtc-feed-2').tagName).toBe('VIDEO');
    expect(webrtcClient.connectWebRtcFeeds).toHaveBeenCalledOnce();
  });
});
