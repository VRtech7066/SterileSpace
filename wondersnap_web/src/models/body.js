// Skeleton and full human body. Units 0.5 cm (175 cm tall = 350), y up from the soles, the body faces +z.
// The body comes apart system by system: skin, organs, nervous system, arteries, veins and skeleton.
import { line, polyline, paramSurface, ellipsoid, orient, ring, tri, TAU } from '../lib/sampling.js';
import { add, mul, sub, norm, linspace } from '../lib/vec.js';
import { spline, vessel, pathTube, blob, bone } from '../lib/shapes.js';
import { G } from './engines.js';

const SPINE = spline([[0, 302, -4], [0, 282, -6], [0, 250, -11], [0, 222, -8], [0, 205, -5], [0, 190, -9]], 8);

/** Bone groups keyed by region; side s = +1 is the body's left (+x). */
function skeletonParts() {
  const P = { skull: [], spine: [], ribs: [], shoulders: [], arms: { 1: [], [-1]: [] }, hands: { 1: [], [-1]: [] }, pelvis: [], legs: { 1: [], [-1]: [] }, feet: { 1: [], [-1]: [] } };
  P.skull.push(blob([0, 324, -1], [18, 22, 20], { density: 0.8 }), blob([0, 306, 9], [13, 11, 11], { density: 0.8 }),
    vessel([[-11, 300, -2], [-10, 293, 8], [0, 290, 14], [10, 293, 8], [11, 300, -2]], 2.4, 2.4, 1.4));   // cranium, face, jaw
  for (const s of [-1, 1]) P.skull.push(orient(ring([0, 0, 0], 4.5, { density: 5 }), [0, 0, 1], [s * 7, 314, 16]));
  const idx = linspace(0, SPINE.length - 2, 24).map(Math.round);
  for (const i of idx) {                                                  // 24 vertebrae + spinous processes
    const p = SPINE[i], d = norm(sub(SPINE[i + 1], p));
    P.spine.push(orient(paramSurface((u, h) => [5 * Math.cos(u), h * 4, 4.5 * Math.sin(u)], 0, TAU, 0, 1, { density: 1, gu: 12, gv: 2 }), d, p));
    P.spine.push(line(p, add(p, [0, -2, -8]), 0, 3));
  }
  P.spine.push(blob([0, 181, -9], [9, 10, 5], { density: 1 }));
  for (let k = 0; k < 12; k++) {                                          // 12 pairs of ribs
    const y = 282 - k * 7.4, w = 17 + 15 * Math.sin((Math.PI * (k + 2)) / 15), end = k >= 10 ? 0.2 : Math.PI / 2;
    for (const s of [-1, 1]) {
      const pts = linspace(-Math.PI / 2, end, 14).map((f) => [s * (3 + w * Math.cos(f)), y - 9 * ((f + Math.PI / 2) / Math.PI), 2 + 14 * Math.sin(f)]);
      P.ribs.push(pathTube(pts, 1.6, 1.4));
      if (k < 10) P.ribs.push(line(pts[pts.length - 1], [s * 3, 262 - k * 3, 16], 0, 2));   // costal cartilage to the sternum
    }
  }
  P.ribs.push(blob([0, 258, 16], [5, 27, 2.5], { density: 1.2 }));        // sternum
  for (const s of [-1, 1]) {
    P.shoulders.push(vessel([[s * 4, 288, 12], [s * 22, 291, 10], [s * 38, 289, 0]], 2.2, 2.2, 1.5));   // clavicle
    P.shoulders.push(tri([s * 20, 283, -13], [s * 42, 283, -8], [s * 30, 245, -13], 0.8), polyline([[s * 20, 283, -13], [s * 42, 283, -8], [s * 30, 245, -13]], 0, 3, true));   // scapula
    const A = P.arms[s], Hh = P.hands[s];
    A.push(bone([s * 40, 283, 0], [s * 47, 225, -2], 3.2, 1.1), bone([s * 47, 225, -2], [s * 52, 172, 6], 2.1, 1.1), bone([s * 45, 224, -5], [s * 50, 171, 1], 2.1, 1.1));
    Hh.push(ellipsoid([s * 52, 167, 4], [5, 4, 3], 1.3));
    for (let f = 0; f < 5; f++) {
      const b = [s * (48.5 + f * 1.9), 163, 1 + f * 1.4], dir = f === 0 ? norm([s * -0.4, -0.6, 0.7]) : norm([s * 0.05, -1, 0.08]);
      const k1 = add(b, mul(dir, 8)), k2 = add(k1, mul(dir, f === 0 ? 5 : 6)), k3 = add(k2, mul(dir, 4));
      Hh.push(polyline([b, k1, k2, k3], 0, 5));
    }
    P.legs[s].push(bone([s * 16, 181, 2], [s * 19, 100, 0], 3.8, 1.1), ellipsoid([s * 19, 102, 6], [3, 3.5, 1.6], 1.4),
      bone([s * 19, 97, 0], [s * 20, 22, -1], 3.1, 1.1), bone([s * 23, 95, -2], [s * 23, 24, -2], 1.5, 1.1));
    P.feet[s].push(ellipsoid([s * 20, 17, -2], [4, 4, 4], 1.4), ellipsoid([s * 20, 8, -7], [4, 4, 5], 1.2));
    for (let t = 0; t < 5; t++) P.feet[s].push(polyline([[s * (16 + t * 2), 12, 0], [s * (15 + t * 2.4), 5, 14], [s * (15 + t * 2.5), 2, 22]], 0, 5));
  }
  P.pelvis.push(paramSurface((u, t) => [(14 + 14 * t) * Math.cos(u) * 1.25, 172 + 22 * t, (9 + 6 * t) * Math.sin(u)], 0, TAU, 0, 1, { density: 0.8, gu: 36, gv: 8 }),
    orient(ring([0, 0, 0], 5, { density: 4 }), [1, 0, 0], [16, 180, 2]), orient(ring([0, 0, 0], 5, { density: 4 }), [1, 0, 0], [-16, 180, 2]));
  return P;
}

// ============================================================ SKELETON
function skeleton() {
  const P = skeletonParts(), BONE = [0.95, 0.93, 0.85];
  return [
    G('Skull', BONE, [0, 70, 20], 0, P.skull, { info: '22 bones that protect the brain' }),
    G('Rib cage & sternum', [0.8, 0.9, 1.0], [0, 15, 70], 0, P.ribs, { info: '12 pairs of ribs shield heart & lungs' }),
    G('Spine (vertebral column)', [1.0, 0.8, 0.5], [0, 0, -70], 0.1, P.spine, { info: '33 vertebrae guard the spinal cord' }),
    G('Shoulder girdle', [0.7, 1.0, 0.8], [0, 35, -45], 0.05, P.shoulders, { info: 'collarbones and shoulder blades' }),
    G('Arm bones', [0.6, 0.85, 1.0], [70, 10, 0], 0.05, P.arms[1], { info: 'humerus, radius and ulna' }),
    G('Arm bones (L)', [0.6, 0.85, 1.0], [-70, 10, 0], 0.05, P.arms[-1], { showLabel: false }),
    G('Hand bones', [1.0, 0.7, 0.9], [110, -20, 30], 0.1, P.hands[1], { info: '27 bones in each hand' }),
    G('Hand bones (L)', [1.0, 0.7, 0.9], [-110, -20, 30], 0.1, P.hands[-1], { showLabel: false }),
    G('Pelvis', [1.0, 0.9, 0.6], [0, -15, 50], 0.1, P.pelvis, { info: 'carries the upper body onto the legs' }),
    G('Leg bones', [0.75, 0.75, 1.0], [35, -45, 0], 0.1, P.legs[1], { info: 'the femur is the longest bone' }),
    G('Leg bones (L)', [0.75, 0.75, 1.0], [-35, -45, 0], 0.1, P.legs[-1], { showLabel: false }),
    G('Foot bones', [1.0, 0.85, 0.5], [45, -80, 40], 0.15, P.feet[1], { info: '26 bones in each foot' }),
    G('Foot bones (L)', [1.0, 0.85, 0.5], [-45, -80, 40], 0.15, P.feet[-1], { showLabel: false }),
  ];
}

export const BODY = [
  { name: 'Skeleton', groups: skeleton, color: [0.95, 0.93, 0.85], tilt: 4, viewYaw: 0.5, fact: '206 bones · the femur is the longest' },
];
