// @vitest-environment node

import {
  computeCoverTransform,
  drawOverlayFrameToCanvas,
  getFaceOverlayStyle,
} from './overlay-canvas';

describe('computeCoverTransform', () => {
  it('projects source coordinates through object-fit cover scaling', () => {
    expect(
      computeCoverTransform({
        sourceWidth: 1280,
        sourceHeight: 720,
        targetWidth: 640,
        targetHeight: 360,
      }),
    ).toEqual({
      scale: 0.5,
      offsetX: 0,
      offsetY: 0,
    });

    expect(
      computeCoverTransform({
        sourceWidth: 1280,
        sourceHeight: 720,
        targetWidth: 320,
        targetHeight: 320,
      }),
    ).toEqual({
      scale: 320 / 720,
      offsetX: (320 - 1280 * (320 / 720)) / 2,
      offsetY: 0,
    });
  });
});

describe('getFaceOverlayStyle', () => {
  it('matches the Python HUD palette for owner and subject', () => {
    expect(getFaceOverlayStyle('OWNER')).toMatchObject({
      boxColor: 'rgb(200, 40, 200)',
      cornerColor: 'rgb(200, 40, 200)',
      labelColor: 'rgb(200, 40, 200)',
      cornerThickness: 5,
    });
    expect(getFaceOverlayStyle('SUBJECT')).toMatchObject({
      boxColor: 'rgb(255, 255, 255)',
      cornerColor: 'rgb(255, 200, 0)',
      labelColor: 'rgb(255, 200, 0)',
      cornerThickness: 3,
    });
  });
});

describe('drawOverlayFrameToCanvas', () => {
  it('scales face rectangles using cover geometry', () => {
    const ops: Array<Record<string, unknown>> = [];
    const context = {
      clearRect: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      setLineDash: vi.fn(),
      strokeRect: vi.fn((x: number, y: number, w: number, h: number) => {
        ops.push({ kind: 'strokeRect', x, y, w, h });
      }),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      strokeText: vi.fn(),
      fillText: vi.fn(),
      measureText: vi.fn(() => ({ width: 60 })),
      strokeStyle: '',
      fillStyle: '',
      lineWidth: 0,
      lineJoin: '',
      miterLimit: 0,
      font: '',
    } as unknown as CanvasRenderingContext2D;

    drawOverlayFrameToCanvas({
      context,
      canvasWidth: 320,
      canvasHeight: 320,
      sourceWidth: 1280,
      sourceHeight: 720,
      frame: {
        seq: 1,
        ts_ms: 2,
        camera_id: 0,
        frame_width: 1280,
        frame_height: 720,
        faces: [{ x: 100, y: 50, w: 200, h: 100, label: 'SUBJECT', confidence: 0.8 }],
      },
    });

    expect(ops[0]).toMatchObject({
      kind: 'strokeRect',
      x: expect.closeTo(-80, 2),
      y: expect.closeTo(22.222, 2),
      w: expect.closeTo(88.888, 2),
      h: expect.closeTo(44.444, 2),
    });
  });
});
