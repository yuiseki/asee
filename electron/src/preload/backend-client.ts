import type {
  BiometricStatus,
  OverlayText,
  ViewerSnapshot,
  ViewerStatus,
} from '../shared/viewer-api';

type FetchLike = typeof fetch;

type JsonResponse = {
  json: () => Promise<unknown>;
  ok: boolean;
  status: number;
};

const DEFAULT_BIOMETRIC_STATUS: BiometricStatus = {
  running: false,
  ownerEmbeddingLoaded: false,
  ownerPresent: false,
  ownerCount: 0,
  subjectCount: 0,
  peopleCount: 0,
  ownerSeenAgoMs: null,
  updatedAt: 0,
};

export function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, '');
}

async function fetchJson<T>(fetchImpl: FetchLike, url: string): Promise<T> {
  const response = (await fetchImpl(url)) as JsonResponse;
  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function fetchViewerSnapshot({
  baseUrl,
  fetchImpl,
}: {
  baseUrl: string;
  fetchImpl: FetchLike;
}): Promise<ViewerSnapshot> {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl);

  const [cameraResponse, overlayResponse, statusResponse, biometricResponse] = await Promise.all([
    fetchJson<{ cameras?: number[] }>(fetchImpl, `${normalizedBaseUrl}/cameras`),
    fetchJson<Partial<OverlayText>>(fetchImpl, `${normalizedBaseUrl}/overlay_text`),
    fetchJson<Partial<ViewerStatus>>(fetchImpl, `${normalizedBaseUrl}/status`),
    fetchJson<Partial<BiometricStatus>>(fetchImpl, `${normalizedBaseUrl}/biometric_status`),
  ]);

  return {
    baseUrl: normalizedBaseUrl,
    cameras: Array.isArray(cameraResponse.cameras) ? cameraResponse.cameras : [],
    status: {
      running: statusResponse.running ?? false,
      transport: statusResponse.transport === 'webrtc' ? 'webrtc' : 'mjpeg',
    },
    overlayText: {
      caption: overlayResponse.caption ?? '',
      prediction: overlayResponse.prediction ?? '',
    },
    biometricStatus: {
      ...DEFAULT_BIOMETRIC_STATUS,
      ...biometricResponse,
      running: biometricResponse.running ?? statusResponse.running ?? false,
    },
  };
}

export function buildDemoViewerSnapshot(baseUrl: string): ViewerSnapshot {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl);
  return {
    baseUrl: normalizedBaseUrl,
    cameras: [0, 2, 4, 6],
    status: { running: true, transport: 'mjpeg' },
    overlayText: {
      caption: 'OBSERVING OWNER PRESENCE',
      prediction: 'ROOM CALM / FOUR CAMERA GRID',
    },
    biometricStatus: {
      running: true,
      ownerEmbeddingLoaded: true,
      ownerPresent: true,
      ownerCount: 1,
      subjectCount: 0,
      peopleCount: 1,
      ownerSeenAgoMs: 120,
      updatedAt: Date.now() / 1000,
    },
  };
}
