// @vitest-environment node

import { buildDemoViewerSnapshot, fetchViewerSnapshot, normalizeBaseUrl } from './backend-client';

describe('normalizeBaseUrl', () => {
  it('removes a trailing slash from the backend URL', () => {
    expect(normalizeBaseUrl('http://127.0.0.1:8765/')).toBe('http://127.0.0.1:8765');
  });
});

describe('fetchViewerSnapshot', () => {
  it('assembles cameras, overlay text, status, and biometric status', async () => {
    const fetchMock = vi.fn(async (input: string | URL | Request) => {
      const url = String(input);
      if (url.endsWith('/cameras')) {
        return new Response(JSON.stringify({ cameras: [0, 2] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/overlay_text')) {
        return new Response(JSON.stringify({ caption: '観測中', prediction: '静穏' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/status')) {
        return new Response(JSON.stringify({ running: true }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      return new Response(
        JSON.stringify({
          running: true,
          ownerEmbeddingLoaded: true,
          ownerPresent: true,
          ownerCount: 1,
          subjectCount: 1,
          peopleCount: 2,
          ownerSeenAgoMs: 320,
          updatedAt: 1234.5,
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        },
      );
    });

    const snapshot = await fetchViewerSnapshot({
      baseUrl: 'http://127.0.0.1:8765/',
      fetchImpl: fetchMock,
    });

    expect(snapshot.baseUrl).toBe('http://127.0.0.1:8765');
    expect(snapshot.cameras).toEqual([0, 2]);
    expect(snapshot.overlayText.caption).toBe('観測中');
    expect(snapshot.status.running).toBe(true);
    expect(snapshot.biometricStatus.ownerPresent).toBe(true);
  });
});

describe('buildDemoViewerSnapshot', () => {
  it('returns a self-contained demo state', () => {
    const snapshot = buildDemoViewerSnapshot('http://127.0.0.1:8765');

    expect(snapshot.cameras).toEqual([0, 2, 4, 6]);
    expect(snapshot.overlayText.caption).toContain('OBSERVING');
    expect(snapshot.biometricStatus.ownerPresent).toBe(true);
  });
});
