import {
  startTransition,
  useEffect,
  useEffectEvent,
  useMemo,
  useRef,
  useState,
} from 'react';

import type { ViewerApi, ViewerConfig, ViewerSnapshot } from '@/shared/viewer-api';

import { connectWebRtcFeeds, type OverlayFrameMessage } from './webrtc-client';
import { drawOverlayFrameToCanvas } from './overlay-canvas';

const DEFAULT_CONFIG: ViewerConfig = {
  title: 'ASEE Viewer',
  backendBaseUrl: 'http://127.0.0.1:8765',
  pollIntervalMs: 2000,
  autoDemo: false,
};

type CameraFeed = {
  cameraId: number | null;
  label: string;
  mjpegSrc: string;
};

const EMPTY_CAMERA_IDS: number[] = [];

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

function buildCameraFeeds(snapshot: ViewerSnapshot): CameraFeed[] {
  if (snapshot.cameras.length <= 1) {
    return [
      {
        cameraId: snapshot.cameras[0] ?? null,
        mjpegSrc:
          snapshot.cameras.length === 1
            ? `${snapshot.baseUrl}/stream/${snapshot.cameras[0]}`
            : `${snapshot.baseUrl}/stream`,
        label: snapshot.cameras.length === 1 ? `CAM ${snapshot.cameras[0]}` : 'LIVE FEED',
      },
    ];
  }

  return snapshot.cameras.map((cameraId) => ({
    cameraId,
    mjpegSrc: `${snapshot.baseUrl}/stream/${cameraId}`,
    label: `CAM ${cameraId}`,
  }));
}

function WebRtcFeed({
  feed,
  stream,
  overlayFrame,
  single,
}: {
  feed: CameraFeed;
  stream: MediaStream | null;
  overlayFrame: OverlayFrameMessage | null;
  single: boolean;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const video = videoRef.current as (HTMLVideoElement & { srcObject?: MediaStream | null }) | null;
    if (video == null) {
      return;
    }
    video.srcObject = stream;
  }, [stream]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (canvas == null) {
      return;
    }
    const context = canvas.getContext('2d');
    if (context == null) {
      return;
    }

    const width = canvas.clientWidth || 640;
    const height = canvas.clientHeight || 360;
    canvas.width = width;
    canvas.height = height;
    context.clearRect(0, 0, width, height);

    if (overlayFrame == null) {
      return;
    }

    drawOverlayFrameToCanvas({
      context,
      canvasWidth: width,
      canvasHeight: height,
      frame: overlayFrame,
      sourceWidth: overlayFrame.frame_width ?? video?.videoWidth ?? width,
      sourceHeight: overlayFrame.frame_height ?? video?.videoHeight ?? height,
    });
  }, [overlayFrame]);

  return (
    <article className={single ? 'cam-tile cam-tile-single' : 'cam-tile'} key={feed.label}>
      <video
        autoPlay
        className={single ? 'single-feed' : 'tile-feed'}
        data-testid={`webrtc-feed-${feed.cameraId ?? 'live'}`}
        muted
        playsInline
        ref={videoRef}
      />
      <canvas className="feed-overlay" ref={canvasRef} />
      <span className="tile-label">{feed.label}</span>
    </article>
  );
}

export function App() {
  const api = getViewerApi();
  const [config] = useState<ViewerConfig>(() => api?.getConfig() ?? DEFAULT_CONFIG);
  const [snapshot, setSnapshot] = useState<ViewerSnapshot | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [now, setNow] = useState(() => new Date());
  const [webrtcStreams, setWebrtcStreams] = useState<Record<number, MediaStream>>({});
  const [overlayFrames, setOverlayFrames] = useState<Record<number, OverlayFrameMessage>>({});

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
  const isWebRtc = snapshot?.status.transport === 'webrtc';
  const cameraKey = feeds.map((feed) => feed.cameraId ?? 'live').join(',');
  const baseUrl = snapshot?.baseUrl ?? '';
  const webRtcCameraIds = useMemo(
    () => (snapshot?.cameras ? [...snapshot.cameras] : EMPTY_CAMERA_IDS),
    [cameraKey],
  );

  useEffect(() => {
    if (!isWebRtc || baseUrl === '') {
      startTransition(() => {
        setWebrtcStreams({});
        setOverlayFrames({});
      });
      return;
    }

    let cancelled = false;
    let session: { close: () => void } | null = null;

    void connectWebRtcFeeds({
      baseUrl,
      cameraIds: webRtcCameraIds,
      onStream: (cameraId, stream) => {
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setWebrtcStreams((previous) => ({ ...previous, [cameraId]: stream }));
        });
      },
      onOverlayFrame: (frame) => {
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setOverlayFrames((previous) => ({ ...previous, [frame.camera_id]: frame }));
        });
      },
    })
      .then((nextSession) => {
        if (cancelled) {
          nextSession.close();
          return;
        }
        session = nextSession;
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        const message = error instanceof Error ? error.message : String(error);
        startTransition(() => {
          setErrorMessage(message);
        });
      });

    return () => {
      cancelled = true;
      session?.close();
    };
  }, [baseUrl, cameraKey, isWebRtc, webRtcCameraIds]);

  return (
    <main className="viewer-shell">
      <div className="camera-stage">
        {snapshot && feeds.length > 1 ? (
          <section className="tile-grid" aria-label="Camera grid">
            {feeds.map((feed) =>
              isWebRtc ? (
                <WebRtcFeed
                  feed={feed}
                  key={feed.label}
                  overlayFrame={overlayFrames[feed.cameraId ?? 0] ?? null}
                  single={false}
                  stream={webrtcStreams[feed.cameraId ?? 0] ?? null}
                />
              ) : (
                <article className="cam-tile" key={feed.label}>
                  <img alt={feed.label} src={feed.mjpegSrc} />
                  <span className="tile-label">{feed.label}</span>
                </article>
              ),
            )}
          </section>
        ) : snapshot && feeds.length === 1 ? (
          isWebRtc ? (
            <WebRtcFeed
              feed={feeds[0]}
              overlayFrame={overlayFrames[feeds[0].cameraId ?? 0] ?? null}
              single
              stream={webrtcStreams[feeds[0].cameraId ?? 0] ?? null}
            />
          ) : (
            <img className="single-feed" alt={feeds[0].label} src={feeds[0].mjpegSrc} />
          )
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
              {snapshot.status.running ? 'BACKEND RUNNING' : 'BACKEND STOPPED'} /{' '}
              {snapshot.status.transport.toUpperCase()}
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
