export type PoseLandmark = {
  x: number;
  y: number;
  z: number;
  visibility?: number;
  presence?: number;
};

/**
 * Convert the 33 MediaPipe landmarks to the supplied classifier contract:
 * hip-centred x/y/z for every landmark, followed by its visibility value.
 * This yields exactly 132 float features in landmark-index order.
 */
export function landmarksToFeatures(landmarks: PoseLandmark[]): number[] {
  if (landmarks.length !== 33) {
    throw new Error(`Expected 33 pose landmarks; received ${landmarks.length}.`);
  }
  for (const [index, point] of landmarks.entries()) {
    if (![point.x, point.y, point.z].every(Number.isFinite)) {
      throw new Error(`Landmark ${index} contains a non-finite coordinate.`);
    }
  }
  const leftHip = landmarks[23];
  const rightHip = landmarks[24];
  const hipX = (leftHip.x + rightHip.x) / 2;
  const hipY = (leftHip.y + rightHip.y) / 2;
  const hipZ = (leftHip.z + rightHip.z) / 2;

  const features: number[] = [];
  for (const point of landmarks) {
    features.push(
      point.x - hipX,
      point.y - hipY,
      point.z - hipZ,
      finiteOrZero(point.visibility ?? 0),
    );
  }
  if (features.length !== 132) throw new Error('Internal feature contract mismatch.');
  return features;
}

function finiteOrZero(value: number): number {
  return Number.isFinite(value) ? value : 0;
}
