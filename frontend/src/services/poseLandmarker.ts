import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';

const WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm';
const POSE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

export type Landmark = {
  x: number;
  y: number;
  z: number;
  visibility?: number;
  presence?: number;
};

export type PoseDetector = {
  detectForVideo: (video: HTMLVideoElement, timestamp: number) => { landmarks?: Landmark[][] };
  close: () => void;
};

export async function createPoseDetector(): Promise<PoseDetector> {
  // A fresh detector per session means Stop fully releases the camera-side worker/resources.
  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
  const detector = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: POSE_MODEL_URL },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  return detector as unknown as PoseDetector;
}

export const POSE_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
  [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [27, 29], [29, 31],
  [27, 31], [24, 26], [26, 28], [28, 30], [30, 32], [28, 32],
];

export function drawPose(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  landmarks?: Landmark[],
  highlighted = false,
): void {
  const context = canvas.getContext('2d');
  if (!context) return;

  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  if (!landmarks?.length || video.readyState < 2) return;

  const sourceWidth = video.videoWidth || width;
  const sourceHeight = video.videoHeight || height;
  const coverScale = Math.max(width / sourceWidth, height / sourceHeight);
  const offsetX = (width - sourceWidth * coverScale) / 2;
  const offsetY = (height - sourceHeight * coverScale) / 2;
  const point = (index: number) => ({
    x: landmarks[index].x * sourceWidth * coverScale + offsetX,
    y: landmarks[index].y * sourceHeight * coverScale + offsetY,
    visibility: landmarks[index].visibility ?? landmarks[index].presence ?? 0,
  });
  const lineColour = highlighted ? 'rgba(241, 182, 111, 0.95)' : 'rgba(193, 232, 211, 0.9)';

  context.lineCap = 'round';
  context.lineJoin = 'round';
  context.lineWidth = 2.1;
  for (const [from, to] of POSE_CONNECTIONS) {
    const a = point(from);
    const b = point(to);
    if (a.visibility < 0.25 || b.visibility < 0.25) continue;
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.strokeStyle = lineColour;
    context.stroke();
  }

  for (let index = 0; index < landmarks.length; index += 1) {
    const p = point(index);
    if (p.visibility < 0.25) continue;
    context.beginPath();
    context.arc(p.x, p.y, index === 11 || index === 12 || index === 23 || index === 24 ? 3.2 : 2.1, 0, Math.PI * 2);
    context.fillStyle = lineColour;
    context.fill();
  }
}
