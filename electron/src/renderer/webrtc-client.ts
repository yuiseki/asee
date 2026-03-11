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
  faces: OverlayFaceMessage[];
  caption?: string;
  prediction?: string;
};

type FetchLike = typeof fetch;

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, '');
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

  peerConnection.ontrack = (event) => {
    const stream = event.streams[0];
    if (!stream) {
      return;
    }
    const cameraId =
      resolvedCameraIds[Math.min(nextTrackIndex, resolvedCameraIds.length - 1)] ?? resolvedCameraIds[0];
    nextTrackIndex += 1;
    onStream(cameraId, stream);
  };

  peerConnection.ondatachannel = (event) => {
    if (event.channel.label !== 'overlay') {
      return;
    }
    event.channel.addEventListener('message', (messageEvent) => {
      onOverlayFrame(JSON.parse(String(messageEvent.data)) as OverlayFrameMessage);
    });
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

  return {
    close: () => {
      peerConnection.close();
    },
  };
}
