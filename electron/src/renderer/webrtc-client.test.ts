// @vitest-environment node

import { connectWebRtcFeeds, type OverlayFrameMessage } from './webrtc-client';

class FakeRtcPeerConnection {
  public addTransceiver = vi.fn();
  public createOffer = vi.fn(async () => ({ type: 'offer', sdp: 'offer-sdp' }));
  public setLocalDescription = vi.fn(async (desc: RTCSessionDescriptionInit) => {
    this.localDescription = desc;
  });
  public setRemoteDescription = vi.fn(async (_desc: RTCSessionDescriptionInit) => {
    return undefined;
  });
  public close = vi.fn();
  public localDescription: RTCSessionDescriptionInit | null = null;
  public ontrack: ((event: RTCTrackEvent) => void) | null = null;
  public ondatachannel: ((event: RTCDataChannelEvent) => void) | null = null;
}

describe('connectWebRtcFeeds', () => {
  it('negotiates recvonly transceivers and routes tracks / overlay frames', async () => {
    const pc = new FakeRtcPeerConnection();
    const streamsByCamera = new Map<number, MediaStream>();
    const overlays: OverlayFrameMessage[] = [];
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ type: 'answer', sdp: 'answer-sdp' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );

    const session = await connectWebRtcFeeds({
      baseUrl: 'http://127.0.0.1:8765',
      cameraIds: [0, 2],
      fetchImpl: fetchMock,
      peerConnectionFactory: () => pc as unknown as RTCPeerConnection,
      onStream(cameraId, stream) {
        streamsByCamera.set(cameraId, stream);
      },
      onOverlayFrame(frame) {
        overlays.push(frame);
      },
    });

    expect(pc.addTransceiver).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenCalledWith(
      'http://127.0.0.1:8765/offer',
      expect.objectContaining({
        method: 'POST',
      }),
    );

    const firstStream = {} as MediaStream;
    const secondStream = {} as MediaStream;
    pc.ontrack?.({ streams: [firstStream] } as unknown as RTCTrackEvent);
    pc.ontrack?.({ streams: [secondStream] } as unknown as RTCTrackEvent);

    expect(streamsByCamera.get(0)).toBe(firstStream);
    expect(streamsByCamera.get(2)).toBe(secondStream);

    const messageHandlers: Array<(event: MessageEvent<string>) => void> = [];
    pc.ondatachannel?.({
      channel: {
        label: 'overlay',
        addEventListener(_name: 'message', listener: (event: MessageEvent<string>) => void) {
          messageHandlers.push(listener);
        },
      },
    } as unknown as RTCDataChannelEvent);
    messageHandlers[0]?.(
      new MessageEvent('message', {
        data: JSON.stringify({
          seq: 1,
          ts_ms: 2,
          camera_id: 2,
          faces: [],
          caption: 'OBSERVING',
        }),
      }),
    );

    expect(overlays).toEqual([
      {
        seq: 1,
        ts_ms: 2,
        camera_id: 2,
        faces: [],
        caption: 'OBSERVING',
      },
    ]);

    session.close();
    expect(pc.close).toHaveBeenCalledOnce();
  });
});
