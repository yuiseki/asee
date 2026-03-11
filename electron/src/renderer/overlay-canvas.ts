import type { OverlayFaceMessage, OverlayFrameMessage } from './webrtc-client';

export type CoverTransform = {
  scale: number;
  offsetX: number;
  offsetY: number;
};

type FaceOverlayStyle = {
  boxColor: string;
  cornerColor: string;
  labelColor: string;
  cornerThickness: number;
};

export function computeCoverTransform({
  sourceWidth,
  sourceHeight,
  targetWidth,
  targetHeight,
}: {
  sourceWidth: number;
  sourceHeight: number;
  targetWidth: number;
  targetHeight: number;
}): CoverTransform {
  if (sourceWidth <= 0 || sourceHeight <= 0 || targetWidth <= 0 || targetHeight <= 0) {
    return { scale: 1, offsetX: 0, offsetY: 0 };
  }
  const scale = Math.max(targetWidth / sourceWidth, targetHeight / sourceHeight);
  const displayWidth = sourceWidth * scale;
  const displayHeight = sourceHeight * scale;
  return {
    scale,
    offsetX: (targetWidth - displayWidth) / 2,
    offsetY: (targetHeight - displayHeight) / 2,
  };
}

export function getFaceOverlayStyle(label: string): FaceOverlayStyle {
  const isOwner = label === 'OWNER';
  return {
    boxColor: isOwner ? 'rgb(200, 40, 200)' : 'rgb(255, 255, 255)',
    cornerColor: isOwner ? 'rgb(200, 40, 200)' : 'rgb(255, 200, 0)',
    labelColor: isOwner ? 'rgb(200, 40, 200)' : 'rgb(255, 200, 0)',
    cornerThickness: isOwner ? 5 : 3,
  };
}

function drawDashedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  color: string,
): void {
  context.save();
  context.strokeStyle = color;
  context.lineWidth = 2;
  context.setLineDash([10, 6]);
  context.strokeRect(x, y, width, height);
  context.restore();
}

function drawCornerGuides(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  {
    color,
    thickness,
  }: {
    color: string;
    thickness: number;
  },
): void {
  const arm = Math.max(16, width / 6);
  const centerX = x + width / 2;
  const centerY = y + height / 2;
  const guideLength = Math.max(8, width / 12);
  const guideThickness = Math.max(1, thickness - 2);

  context.save();
  context.strokeStyle = color;
  context.lineWidth = thickness;
  context.beginPath();
  context.moveTo(x, y);
  context.lineTo(x + arm, y);
  context.moveTo(x, y);
  context.lineTo(x, y + arm);
  context.moveTo(x + width, y);
  context.lineTo(x + width - arm, y);
  context.moveTo(x + width, y);
  context.lineTo(x + width, y + arm);
  context.moveTo(x, y + height);
  context.lineTo(x + arm, y + height);
  context.moveTo(x, y + height);
  context.lineTo(x, y + height - arm);
  context.moveTo(x + width, y + height);
  context.lineTo(x + width - arm, y + height);
  context.moveTo(x + width, y + height);
  context.lineTo(x + width, y + height - arm);
  context.stroke();

  context.lineWidth = guideThickness;
  context.beginPath();
  context.moveTo(centerX, y);
  context.lineTo(centerX, y + guideLength);
  context.moveTo(centerX, y + height);
  context.lineTo(centerX, y + height - guideLength);
  context.moveTo(x, centerY);
  context.lineTo(x + guideLength, centerY);
  context.moveTo(x + width, centerY);
  context.lineTo(x + width - guideLength, centerY);
  context.stroke();
  context.restore();
}

function drawOutlinedText(
  context: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  {
    color,
    font,
  }: {
    color: string;
    font: string;
  },
): void {
  context.save();
  context.font = font;
  context.lineJoin = 'round';
  context.miterLimit = 2;
  context.strokeStyle = 'rgb(0, 0, 0)';
  context.lineWidth = 4;
  context.strokeText(text, x, y);
  context.fillStyle = color;
  context.fillText(text, x, y);
  context.restore();
}

function measureTextWidth(
  context: CanvasRenderingContext2D,
  text: string,
  font: string,
): number {
  context.save();
  context.font = font;
  const width = context.measureText(text).width;
  context.restore();
  return width;
}

function projectFaceRect(
  face: OverlayFaceMessage,
  transform: CoverTransform,
): { x: number; y: number; width: number; height: number } {
  return {
    x: transform.offsetX + face.x * transform.scale,
    y: transform.offsetY + face.y * transform.scale,
    width: face.w * transform.scale,
    height: face.h * transform.scale,
  };
}

export function drawOverlayFrameToCanvas({
  context,
  canvasWidth,
  canvasHeight,
  frame,
  sourceWidth,
  sourceHeight,
}: {
  context: CanvasRenderingContext2D;
  canvasWidth: number;
  canvasHeight: number;
  frame: OverlayFrameMessage;
  sourceWidth: number;
  sourceHeight: number;
}): void {
  const transform = computeCoverTransform({
    sourceWidth,
    sourceHeight,
    targetWidth: canvasWidth,
    targetHeight: canvasHeight,
  });

  context.clearRect(0, 0, canvasWidth, canvasHeight);

  for (const face of frame.faces) {
    const style = getFaceOverlayStyle(face.label);
    const projected = projectFaceRect(face, transform);
    drawDashedRect(context, projected.x, projected.y, projected.width, projected.height, style.boxColor);
    drawCornerGuides(context, projected.x, projected.y, projected.width, projected.height, {
      color: style.cornerColor,
      thickness: style.cornerThickness,
    });

    const labelY = Math.max(projected.y - 8, 18);
    const labelFont = '26px "IBM Plex Mono", "Noto Sans Mono CJK JP", monospace';
    drawOutlinedText(context, face.label, projected.x, labelY, {
      color: style.labelColor,
      font: labelFont,
    });

    if (face.confidence < 1) {
      const labelWidth = measureTextWidth(context, `${face.label} `, labelFont);
      drawOutlinedText(context, `${Math.round(face.confidence * 100)}%`, projected.x + labelWidth, labelY, {
        color: 'rgb(255, 255, 255)',
        font: labelFont,
      });
    }
  }
}
