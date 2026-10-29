import { describe, expect, it } from 'vitest';
import { landmarksToFeatures, type PoseLandmark } from '../src/utils/landmarks';

function makeLandmarks(): PoseLandmark[] {
  return Array.from({ length: 33 }, (_, index) => ({
    x: index / 100,
    y: index / 200,
    z: index / 300,
    visibility: 0.9,
  }));
}

describe('landmarksToFeatures', () => {
  it('emits 132 features in x, y, z, visibility order', () => {
    const features = landmarksToFeatures(makeLandmarks());
    expect(features).toHaveLength(132);
    expect(features[0]).toBeCloseTo(-0.235, 5);
    expect(features[1]).toBeCloseTo(-0.1175, 5);
    expect(features[2]).toBeCloseTo(-0.0783333333, 5);
    expect(features[3]).toBeCloseTo(0.9, 5);
  });

  it('centres x/y/z around the average of both hip landmarks', () => {
    const landmarks = makeLandmarks();
    const features = landmarksToFeatures(landmarks);
    expect(features[23 * 4]).toBeCloseTo(-0.005);
    expect(features[24 * 4]).toBeCloseTo(0.005);
    expect(features[23 * 4 + 1]).toBeCloseTo(-0.0025);
    expect(features[24 * 4 + 1]).toBeCloseTo(0.0025);
  });

  it('falls back to visibility zero if neither visibility nor presence exists', () => {
    const landmarks = makeLandmarks();
    delete landmarks[0].visibility;
    landmarks[0].presence = 0.99;
    expect(landmarksToFeatures(landmarks)[3]).toBe(0);
  });

  it('rejects a landmark list that is not 33 points', () => {
    expect(() => landmarksToFeatures(makeLandmarks().slice(1))).toThrow(/Expected 33/);
  });
});
