export type OverlayFaceMessage = {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  confidence: number;
};

export type OverlayFrameMessage = {
  seq: number;
  ts_ms: number;
  camera_id: number;
  frame_width?: number;
  frame_height?: number;
  faces: OverlayFaceMessage[];
  caption?: string;
  prediction?: string;
};

type FetchLike = typeof fetch;
type LowLatencyReceiver = RTCRtpReceiver & {
  playoutDelayHint?: number;
  jitterBufferTarget?: number;
};

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, '');
}

function buildTrackStream(event: RTCTrackEvent): MediaStream | null {
  if (event.track != null && typeof MediaStream !== 'undefined') {
    return new MediaStream([event.track]);
  }
  return event.streams[0] ?? null;
}

function tuneReceiversForLowLatency(peerConnection: RTCPeerConnection): void {
  const candidate = peerConnection as RTCPeerConnection & {
    getReceivers?: () => RTCRtpReceiver[];
  };
  const receivers = candidate.getReceivers?.() ?? [];
  for (const receiver of receivers) {
    const typedReceiver = receiver as LowLatencyReceiver;
    if (typedReceiver.track?.kind !== 'video') {
      continue;
    }
    try {
      typedReceiver.playoutDelayHint = 0;
    } catch {}
    try {
      typedReceiver.jitterBufferTarget = 0;
    } catch {}
  }
}

export async function connectWebRtcFeeds({
  baseUrl,
  cameraIds,
  fetchImpl = fetch,
  peerConnectionFactory = () => new RTCPeerConnection(),
  onStream,
  onOverlayFrame,
}: {
  baseUrl: string;
  cameraIds: number[];
  fetchImpl?: FetchLike;
  peerConnectionFactory?: () => RTCPeerConnection;
  onStream: (cameraId: number, stream: MediaStream) => void;
  onOverlayFrame: (frame: OverlayFrameMessage) => void;
}): Promise<{ close: () => void }> {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl);
  const peerConnection = peerConnectionFactory();
  const resolvedCameraIds = cameraIds.length > 0 ? cameraIds : [0];
  let nextTrackIndex = 0;

  for (let index = 0; index < resolvedCameraIds.length; index += 1) {
    peerConnection.addTransceiver('video', { direction: 'recvonly' });
  }

  const overlayChannel = peerConnection.createDataChannel('overlay');
  overlayChannel.addEventListener('message', (messageEvent) => {
    onOverlayFrame(JSON.parse(String(messageEvent.data)) as OverlayFrameMessage);
  });

  peerConnection.ontrack = (event) => {
    const stream = buildTrackStream(event);
    if (!stream) {
      return;
    }
    const cameraId =
      resolvedCameraIds[Math.min(nextTrackIndex, resolvedCameraIds.length - 1)] ?? resolvedCameraIds[0];
    nextTrackIndex += 1;
    onStream(cameraId, stream);
  };

  const offer = await peerConnection.createOffer();
  await peerConnection.setLocalDescription(offer);

  const response = await fetchImpl(`${normalizedBaseUrl}/offer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      type: offer.type,
      sdp: offer.sdp,
    }),
  });
  if (!response.ok) {
    throw new Error(`${normalizedBaseUrl}/offer returned ${response.status}`);
  }

  const answer = (await response.json()) as { type: RTCSdpType; sdp: string };
  await peerConnection.setRemoteDescription(answer);
  tuneReceiversForLowLatency(peerConnection);

  return {
    close: () => {
      peerConnection.close();
    },
  };
}
