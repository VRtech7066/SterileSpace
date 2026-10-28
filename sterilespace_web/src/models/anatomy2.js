// More explodable anatomy: lungs (they breathe), eye, ear, tooth, skull. Every labelled part says what it does.
import { line, polyline, tube, paramSurface, ellipsoid, revolve, orient, torus, ring, disc, quad, boxSurface, boxEdges, TAU } from '../lib/sampling.js';
import { add, sub, mul, norm, cross, linspace } from '../lib/vec.js';
import { makeRng } from '../lib/rng.js';
import { vessel, pathTube, dots } from '../lib/shapes.js';
import { G } from './engines.js';

const H = Math.PI / 2;
/** Rotate direction d by angle a around a random axis perpendicular to it. */
function turn(d, a, rng) {
  const ax = norm(cross(d, [rng.random() - 0.5, rng.random() - 0.5, rng.random() - 0.5]));
  return norm(add(mul(d, Math.cos(a)), mul(cross(ax, d), Math.sin(a))));
}
/** Recursive branching tree of tubes (bronchi, vessels); tips collected for alveoli. */
export function tree(out, tips, p, d, L, r, depth, rng, spread = 0.5, density = 1.3) {
  const q = add(p, mul(d, L));
  out.push(tube(p, q, r, r * 0.8, density));
  if (depth === 0) { tips.push(q); return; }
  for (const s of [-1, 1]) tree(out, tips, q, turn(d, spread * (0.8 + 0.4 * rng.random()) * s, rng), L * 0.74, r * 0.72, depth - 1, rng, spread, density);
}

// ============================================================ LUNGS  (x = patient's left, y up, z front; ~0.5 mm units)
function lungs() {
  const lung = (s) => (u, v) => {
    const cv = Math.cos(v), sv = Math.sin(v), taper = 1 - 0.38 * Math.max(0, sv);
    let x = 68 * cv * Math.cos(u) * taper;
    const z = 60 * cv * Math.sin(u) * taper;
    if (x * s < 0) x *= 0.42;                                          // flat medial side facing the heart
    return [s * 92 + x, 20 + 150 * sv * (sv < 0 ? 0.62 : 1), z];
  };
  const obl = (z) => 5 - 1.1 * z;                                      // oblique fissure
  const notch = (x, y, z) => x < 78 && z > 18 && y < 15 && y > -70;    // cardiac notch (left lung)
  const lobe = (s, keep) => [paramSurface(lung(s), 0, TAU, -H, H, { density: 0.85, gu: 44, gv: 30, keep })];
  const RU = lobe(-1, (x, y, z) => y > obl(z) && !(y < 35 && z > -20));
  const RM = lobe(-1, (x, y, z) => y > obl(z) && y < 35 && z > -20);
  const RL = lobe(-1, (x, y, z) => y <= obl(z));
  const LU = lobe(1, (x, y, z) => y > obl(z) && !notch(x, y, z));
  const LL = lobe(1, (x, y, z) => y <= obl(z) && !notch(x, y, z));
  const clampInLung = (s, p, margin = 0.82) => {
    const py = Math.max(-65, Math.min(148, p[1]));
    const dy = py - 20;
    const sv = dy < 0 ? dy / (150 * 0.62) : dy / 150;
    const v = Math.asin(Math.max(-0.95, Math.min(0.95, sv)));
    const cv = Math.cos(v);
    const taper = 1 - 0.38 * Math.max(0, sv);
    const rx = 68 * cv * taper * margin;
    const rz = 60 * cv * taper * margin;

    let dx = p[0] - s * 92;
    let dz = p[2];
    const isMedial = dx * s < 0;
    const effRx = isMedial ? rx * 0.42 : rx;
    const dist = Math.hypot(dx / Math.max(effRx, 1e-3), dz / Math.max(rz, 1e-3));
    if (dist > 1.0) {
      dx = (dx / dist) * 0.98;
      dz = (dz / dist) * 0.98;
    }
    return [s * 92 + dx, py, dz];
  };

  const lungTree = (out, tips, p, d, L, r, depth, rng, s) => {
    const q = clampInLung(s, add(p, mul(d, L)), 0.80);
    out.push(tube(p, q, r, r * 0.8, 1.3));
    if (depth === 0) { tips.push(q); return; }
    for (const side of [-1, 1]) {
      lungTree(out, tips, q, turn(d, 0.42 * (0.8 + 0.4 * rng.random()) * side, rng), L * 0.72, r * 0.72, depth - 1, rng, s);
    }
  };

  const trachea = [tube([0, 235, 8], [0, 112, 4], 14, 13, 1.1), ellipsoid([0, 246, 10], [20, 14, 16], 1.2)];
  for (let y = 120; y <= 228; y += 11) trachea.push(orient(ring([0, 0, 0], 15, { density: 5 }), [0, 1, 0], [0, y, 6]));   // cartilage rings
  const rng = makeRng(5), bronchi = [], tips = [];
  for (const s of [-1, 1]) {
    const hilum = clampInLung(s, [s * 65, 68, 0], 0.75);
    bronchi.push(tube([0, 112, 4], hilum, 10, 8, 1.3));
    lungTree(bronchi, tips, hilum, norm([s * 0.35, 0.85, 0]), 26, 5.5, 4, rng, s);
    lungTree(bronchi, tips, hilum, norm([s * 0.4, -0.8, 0]), 34, 6.5, 4, rng, s);
  }
  const alveoli = [dots(tips.flatMap((p) => [0, 1, 2, 3].map(() => clampInLung(p[0] > 0 ? 1 : -1, add(p, [rng.normal() * 2.5, rng.normal() * 2.5, rng.normal() * 2.5]), 0.85))), 3.0, 1.1)];
  const art = [], vein = [], dump = [];
  for (const s of [-1, 1]) {
    const hilum = clampInLung(s, [s * 65, 68, 0], 0.75);
    art.push(vessel([[0, 128, 30], [s * 30, 100, 20], hilum], 7, 6, 1.2));
    lungTree(art, dump, hilum, norm([s * 0.45, -0.7, 0.1]), 32, 4, 3, rng, s);
    vein.push(vessel([[0, 68, -14], [s * 22, 66, -13], hilum], 6, 5.5, 1.2));
    lungTree(vein, dump, hilum, norm([s * 0.45, -0.6, -0.1]), 30, 4, 3, rng, s);
  }
  const diaphragm = [paramSurface((r, u) => [175 * r * Math.cos(u), -118 + (52 + 10 * Math.cos(2 * u)) * (1 - r * r), 110 * r * Math.sin(u)], 0, 1, 0, TAU, { density: 0.45, gu: 16, gv: 40 })];
  const L1 = [0.55, 0.8, 1.0], L2 = [0.45, 0.65, 1.0], L3 = [0.7, 0.55, 1.0];
  return [
    G('Right upper lobe', L1, [-160, 90, 0], 0, RU, { info: 'right lung has 3 lobes: air in, oxygen out' }),
    G('Right middle lobe', L3, [-170, 0, 110], 0, RM, { info: 'smallest lobe, beside the heart' }),
    G('Right lower lobe', L2, [-160, -90, -60], 0, RL, { info: 'largest lobe: most of every breath' }),
    G('Left upper lobe', L1, [160, 90, 0], 0, LU, { info: 'only 2 lobes on the left: room for the heart' }),
    G('Left lower lobe', L2, [160, -90, -60], 0, LL, { info: 'the cardiac notch cradles the heart' }),
    G('Trachea (windpipe)', [0.95, 0.95, 1.0], [0, 170, 0], 0.05, trachea, { info: 'kept open by C-shaped cartilage rings' }),
    G('Bronchial tree', [1.0, 0.85, 0.4], [0, 20, 150], 0.25, bronchi, { info: 'airways branch ~23 times to the air sacs' }),
    G('Alveoli (air sacs)', [1.0, 0.55, 0.7], [0, -20, 230], 0.35, alveoli, { info: '~480 million sacs where oxygen enters blood' }),
    G('Pulmonary arteries', [0.35, 0.5, 1.0], [0, 40, -150], 0.3, art, { info: 'bring used, oxygen-poor blood in' }),
    G('Pulmonary veins', [1.0, 0.3, 0.3], [0, -40, -190], 0.3, vein, { info: 'carry fresh oxygen-rich blood out' }),
    G('Diaphragm', [1.0, 0.5, 0.45], [0, -180, 0], 0.05, diaphragm, { info: 'dome muscle: pulls down to breathe in' }),
  ];
}

// ============================================================ EAR  (schematic cross-section: sound travels along +x)
function ear() {
  const pinna = [paramSurface((a, w) => [-255 + 62 * w * Math.cos(a), 25 + 118 * w * Math.sin(a), -18 * w * w], -0.35 * Math.PI, 1.3 * Math.PI, 0.35, 1, { density: 0.6, gu: 40, gv: 10 }),
    polyline(linspace(-0.35 * Math.PI, 1.3 * Math.PI, 50).map((a) => [-255 + 64 * Math.cos(a), 25 + 121 * Math.sin(a), -19]), 0, 5),
    ellipsoid([-240, -85, -6], [24, 26, 8], 0.8), ellipsoid([-218, -5, -4], [18, 22, 8], 0.8)];              // lobule + concha
  const canal = [vessel([[-210, -5, 0], [-150, 5, 0], [-80, -4, 0], [-8, 0, 0]], 22, 17, 0.7)];
  const drum = [orient(revolve([0, 9], [26, 1], { density: 1.6 }), [1, 0, 0], [-6, 0, 0]), orient(ring([0, 0, 0], 26, { density: 6 }), [1, 0, 0], [-6, 0, 0])];
  const ossicles = [vessel([[0, 16, 0], [12, 32, 0], [20, 44, 2]], 4.5, 5.5, 2), ellipsoid([22, 47, 2], [8, 9, 7], 2),   // malleus
    ellipsoid([34, 45, 2], [9, 8, 7], 2), vessel([[36, 38, 2], [40, 20, 2], [46, 8, 2]], 3.5, 3, 2),               // incus
    polyline([[46, 8, 2], [56, 12, 6], [64, 6, 5]], 0, 6), polyline([[46, 8, 2], [56, 3, -3], [64, 6, -5]], 0, 6),   // stapes
    orient(disc([0, 0, 0], 0, 7, [0, 1, 0], 3), [1, 0, 0], [64, 6, 0])];
  const middle = [ellipsoid([32, 18, 0], [40, 44, 28], 0.14)];
  const eust = [vessel([[30, -18, 0], [55, -70, 20], [80, -150, 40]], 7, 6, 1.2)];
  const spiral = linspace(0, 1, 90).map((t) => { const a = t * 2.75 * TAU, r = 40 * (1 - t) + 8; return [105 + r * Math.cos(a), -20 + r * Math.sin(a), 22 * t]; });
  const cochlea = [...pathTube(spiral, 9, 1.1), polyline(spiral, 0, 5)];
  const canals = [torus([95, 74, -10], 26, 3.5, [1, 0, 0], 2.2), torus([80, 62, 8], 26, 3.5, [0, 1, 0], 2.2), torus([110, 62, 6], 26, 3.5, [0, 0, 1], 2.2), ellipsoid([88, 25, 0], [22, 20, 16], 0.9)];
  const nerve = [vessel([[118, -10, 0], [170, 15, -5], [260, 30, -15]], 9, 8, 1.3)];
  return [
    G('Outer ear (pinna)', [1.0, 0.72, 0.6], [-150, 0, 0], 0, pinna, { info: 'funnels sound into the ear canal' }),
    G('Ear canal', [1.0, 0.6, 0.5], [-80, -40, 0], 0.05, canal, { info: '2.5 cm tube that guides sound in' }),
    G('Eardrum', [1.0, 0.45, 0.6], [-20, 70, 0], 0.1, drum, { info: 'thin membrane that vibrates with sound' }),
    G('Ossicles (hammer, anvil, stirrup)', [1.0, 0.95, 0.7], [10, 130, 60], 0.2, ossicles, { info: 'tiniest bones: amplify vibration ~20x' }),
    G('Middle ear', [0.6, 0.7, 1.0], [0, -90, 0], 0.15, middle, { info: 'air-filled chamber behind the eardrum' }),
    G('Eustachian tube', [0.8, 0.6, 1.0], [0, -130, 70], 0.2, eust, { info: 'equalises pressure: the pop when you fly' }),
    G('Cochlea', [0.6, 1.0, 0.4], [90, -70, 0], 0.2, cochlea, { info: '15,000 hair cells turn sound into signals' }),
    G('Semicircular canals', [0.3, 0.9, 1.0], [90, 120, 0], 0.2, canals, { info: 'balance: sense how your head turns' }),
    G('Auditory nerve', [1.0, 0.85, 0.3], [150, 0, 0], 0.25, nerve, { info: 'sends sound and balance to the brain' }),
  ];
}

export const ANATOMY2 = [
  { name: 'Lungs', groups: lungs, color: [0.55, 0.75, 1.0], tilt: 8, viewYaw: 0.5, breath: { period: 4.5, amp: 0.06 }, fact: '~480 million alveoli · 70 m² of gas-exchange surface' },
  { name: 'Human Ear', groups: ear, color: [1.0, 0.75, 0.55], tilt: 5, viewYaw: 0.35, fact: 'hears 20-20,000 Hz · smallest bones in the body' },
];
