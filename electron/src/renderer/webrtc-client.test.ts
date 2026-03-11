// @vitest-environment node

import { connectWebRtcFeeds, type OverlayFrameMessage } from './webrtc-client';

class FakeRtcPeerConnection {
  public addTransceiver = vi.fn();
  public createDataChannel = vi.fn(
    (label: string) =>
      ({
        label,
        addEventListener: vi.fn(),
      }) as unknown as RTCDataChannel,
  );
  public createOffer = vi.fn(async () => ({ type: 'offer', sdp: 'offer-sdp' }));
  public setLocalDescription = vi.fn(async (desc: RTCSessionDescriptionInit) => {
    this.localDescription = desc;
  });
  public setRemoteDescription = vi.fn(async (_desc: RTCSessionDescriptionInit) => {
    return undefined;
  });
  public close = vi.fn();
  public getReceivers = vi.fn(
    () =>
      [
        { track: { kind: 'video' }, playoutDelayHint: undefined, jitterBufferTarget: undefined },
        { track: { kind: 'audio' } },
      ] as unknown as RTCRtpReceiver[],
  );
  public localDescription: RTCSessionDescriptionInit | null = null;
  public ontrack: ((event: RTCTrackEvent) => void) | null = null;
  public ondatachannel: ((event: RTCDataChannelEvent) => void) | null = null;
}

describe('connectWebRtcFeeds', () => {
  it('negotiates recvonly transceivers, creates overlay data channel, and routes tracks / overlay frames', async () => {
    const pc = new FakeRtcPeerConnection();
    const streamsByCamera = new Map<number, MediaStream>();
    const overlays: OverlayFrameMessage[] = [];
    const messageHandlers: Array<(event: MessageEvent<string>) => void> = [];
    pc.createDataChannel.mockReturnValue({
      label: 'overlay',
      addEventListener(_name: 'message', listener: (event: MessageEvent<string>) => void) {
        messageHandlers.push(listener);
      },
    } as unknown as RTCDataChannel);
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
    expect(pc.createDataChannel).toHaveBeenCalledWith('overlay');
    expect(fetchMock).toHaveBeenCalledWith(
      'http://127.0.0.1:8765/offer',
      expect.objectContaining({
        method: 'POST',
      }),
    );
    expect(pc.getReceivers).toHaveBeenCalled();
    const receivers = pc.getReceivers.mock.results[0]?.value as Array<{
      track?: { kind?: string };
      playoutDelayHint?: number;
      jitterBufferTarget?: number;
    }>;
    expect(receivers[0]?.playoutDelayHint).toBe(0);
    expect(receivers[0]?.jitterBufferTarget).toBe(0);

    class FakeMediaStream {
      public tracks: MediaStreamTrack[];

      constructor(tracks: MediaStreamTrack[] = []) {
        this.tracks = tracks;
      }
    }

    vi.stubGlobal('MediaStream', FakeMediaStream);

    const firstTrack = { id: 'track-0' } as MediaStreamTrack;
    const secondTrack = { id: 'track-2' } as MediaStreamTrack;
    const sharedStream = {} as MediaStream;
    pc.ontrack?.({ track: firstTrack, streams: [sharedStream] } as unknown as RTCTrackEvent);
    pc.ontrack?.({ track: secondTrack, streams: [sharedStream] } as unknown as RTCTrackEvent);

    expect((streamsByCamera.get(0) as unknown as FakeMediaStream).tracks).toEqual([firstTrack]);
    expect((streamsByCamera.get(2) as unknown as FakeMediaStream).tracks).toEqual([secondTrack]);

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
    vi.unstubAllGlobals();
  });
});
