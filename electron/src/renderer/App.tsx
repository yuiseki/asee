import { startTransition, useEffect, useEffectEvent, useMemo, useState } from 'react';

import type { ViewerApi, ViewerConfig, ViewerSnapshot } from '@/shared/viewer-api';

const DEFAULT_CONFIG: ViewerConfig = {
  title: 'ASEE Viewer',
  backendBaseUrl: 'http://127.0.0.1:8765',
  pollIntervalMs: 2000,
  autoDemo: false,
};

function getViewerApi(): ViewerApi | null {
  return window.aseeViewerApi ?? null;
}

function formatClock(now: Date): string {
  const pad = (value: number) => String(value).padStart(2, '0');
  return `${now.getFullYear()}.${pad(now.getMonth() + 1)}.${pad(now.getDate())} ${pad(
    now.getHours(),
  )}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
}

function formatOwnerPresence(snapshot: ViewerSnapshot): string {
  return snapshot.biometricStatus.ownerPresent ? 'OWNER PRESENT' : 'OWNER ABSENT';
}

function formatPeopleSummary(snapshot: ViewerSnapshot): string {
  return `${snapshot.biometricStatus.peopleCount} PEOPLE DETECTED`;
}

function buildCameraFeeds(snapshot: ViewerSnapshot): Array<{ cameraId: number | null; src: string; label: string }> {
  if (snapshot.cameras.length <= 1) {
    return [
      {
        cameraId: snapshot.cameras[0] ?? null,
        src:
          snapshot.cameras.length === 1
            ? `${snapshot.baseUrl}/stream/${snapshot.cameras[0]}`
            : `${snapshot.baseUrl}/stream`,
        label: snapshot.cameras.length === 1 ? `CAM ${snapshot.cameras[0]}` : 'LIVE FEED',
      },
    ];
  }

  return snapshot.cameras.map((cameraId) => ({
    cameraId,
    src: `${snapshot.baseUrl}/stream/${cameraId}`,
    label: `CAM ${cameraId}`,
  }));
}

export function App() {
  const api = getViewerApi();
  const [config] = useState<ViewerConfig>(() => api?.getConfig() ?? DEFAULT_CONFIG);
  const [snapshot, setSnapshot] = useState<ViewerSnapshot | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [now, setNow] = useState(() => new Date());

  const refreshSnapshot = useEffectEvent(async () => {
    if (!api) {
      startTransition(() => {
        setErrorMessage('ASEE preload bridge is unavailable');
      });
      return;
    }

    try {
      const nextSnapshot = await api.fetchSnapshot();
      startTransition(() => {
        setSnapshot(nextSnapshot);
        setErrorMessage(null);
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      startTransition(() => {
        setErrorMessage(message);
      });
    }
  });

  useEffect(() => {
    void refreshSnapshot();
    const timerId = window.setInterval(() => {
      void refreshSnapshot();
    }, config.pollIntervalMs);
    return () => {
      window.clearInterval(timerId);
    };
  }, [config.pollIntervalMs]);

  useEffect(() => {
    const timerId = window.setInterval(() => {
      setNow(new Date());
    }, 1000);
    return () => {
      window.clearInterval(timerId);
    };
  }, []);

  const feeds = useMemo(() => (snapshot ? buildCameraFeeds(snapshot) : []), [snapshot]);

  return (
    <main className="viewer-shell">
      <div className="camera-stage">
        {snapshot && feeds.length > 1 ? (
          <section className="tile-grid" aria-label="Camera grid">
            {feeds.map((feed) => (
              <article className="cam-tile" key={feed.label}>
                <img alt={feed.label} src={feed.src} />
                <span className="tile-label">{feed.label}</span>
              </article>
            ))}
          </section>
        ) : snapshot && feeds.length === 1 ? (
          <img className="single-feed" alt={feeds[0].label} src={feeds[0].src} />
        ) : (
          <section className="placeholder-feed">
            <p>CONNECTING TO BACKEND</p>
          </section>
        )}
      </div>

      <div className="corner" aria-hidden="true" />
      <div className="clock">{formatClock(now)}</div>

      <section className="panel panel-env">
        <p className="panel-label">ENV</p>
        <p className="panel-body">{snapshot?.overlayText.caption || 'SCANNING'}</p>
        <p className="panel-subtle">{snapshot?.overlayText.prediction || config.backendBaseUrl}</p>
      </section>

      <section className="panel panel-status">
        <p className="panel-label">BIOMETRIC</p>
        {snapshot ? (
          <>
            <p className="status-headline">{formatOwnerPresence(snapshot)}</p>
            <p className="panel-body">{formatPeopleSummary(snapshot)}</p>
            <p className="panel-subtle">
              {snapshot.status.running ? 'BACKEND RUNNING' : 'BACKEND STOPPED'}
            </p>
          </>
        ) : errorMessage ? (
          <>
            <p className="status-headline">BACKEND OFFLINE</p>
            <p className="panel-body">{errorMessage}</p>
          </>
        ) : (
          <>
            <p className="status-headline">WAITING</p>
            <p className="panel-body">Polling {config.backendBaseUrl}</p>
          </>
        )}
      </section>
    </main>
  );
}
