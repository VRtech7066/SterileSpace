// Leaf cross-section (~0.1 mm units: 240 wide, 160 deep, y up). Layers explode vertically (Y axis).
// Top -> bottom: cuticle, upper epidermis, palisade mesophyll, vascular bundle (sheath / phloem / xylem, running
// along x), spongy mesophyll with air gaps, lower epidermis with a stoma and its two guard cells.
import { boxSurface, box, revolve, cylinder, ring, orient, tube, quad, paramSurface, ellipsoid, TAU } from '../lib/sampling.js';
import { makeRng } from '../lib/rng.js';
import { G } from './engines.js';

const X = [1, 0, 0];
const W = 240, D = 160, VEIN_Y = 10, VEIN_R = 32;

const C_CUTICLE = [0.7, 0.9, 0.3], C_PALISADE = [0.1, 0.5, 0.1], C_VASC = [0.8, 0.8, 0.4], C_PHLOEM = [0.88, 0.86, 0.5];
const C_SPONGY = [0.3, 0.7, 0.2], C_LOWER = [0.6, 0.8, 0.3], C_GUARD = [0.2, 0.6, 0.2], C_UPPER = [0.65, 0.88, 0.3];

/** Rows x cols of slightly irregular, interlocking epidermis cells covering the leaf footprint. skip(cx, cz) drops a cell. */
function epidermis(y, h, density, seed, skip = () => false) {
  const rng = makeRng(seed), out = [];
  const nx = 6, nz = 4, px = W / nx, pz = D / nz;
  for (let i = 0; i < nx; i++) for (let k = 0; k < nz; k++) {
    const cx = -W / 2 + px * (i + 0.5), cz = -D / 2 + pz * (k + 0.5);
    if (skip(cx, cz)) continue;
    out.push(box([cx, y, cz], [px - 3 - rng.random() * 3, h, pz - 3 - rng.random() * 3], density, 2));
  }
  return out;
}

function leafCrossSectionProc() {
  // ---- cuticle: one thin waxy sheet slightly wider than the leaf
  const cuticle = [boxSurface([0, 122, 0], [W + 10, 3, D + 10], 0.55, null, true, true)];

  // ---- upper epidermis: block of flat interlocking cells
  const upper = epidermis(105, 18, 0.7, 11);

  // ---- palisade: tightly packed upright cylinders in a staggered (hex) grid
  const palisade = [], pitch = 18;
  for (let row = 0, z = -D / 2 + 9; z < D / 2 - 5; z += pitch * 0.87, row++) {
    for (let x = -W / 2 + 9 + (row % 2 ? pitch / 2 : 0); x < W / 2 - 5; x += pitch) {
      palisade.push(cylinder([x, 42, z], 8, 50, { density: 0.55, cap: true, capBottom: true }));
    }
  }

  // ---- vascular bundle along x: outer sheath, phloem (outer half) and xylem (inner half) tube bundles
  const x0 = -W / 2, x1 = W / 2;
  const alongX = (parts, y = VEIN_Y) => orient(parts, X, [x0, y, 0]);
  const sheath = [
    alongX(revolve([0, W], [VEIN_R, VEIN_R], { density: 0.8 })),
    alongX(revolve([0, W], [VEIN_R - 3, VEIN_R - 3], { density: 0.5 })),
    orient(ring([0, 0, 0], VEIN_R, { density: 6 }), X, [x0, VEIN_Y, 0]), orient(ring([0, 0, 0], VEIN_R, { density: 6 }), X, [x1, VEIN_Y, 0]),
  ];
  const xylem = [], phloem = [];
  const bundle = (list, rows, r, cy, spread) => {            // hex-ish cluster of parallel tubes centred at (cy, 0)
    rows.forEach((n, ri) => {
      for (let j = 0; j < n; j++) {
        const y = cy + (ri - (rows.length - 1) / 2) * spread, z = (j - (n - 1) / 2) * spread;
        list.push(tube([x0 - 6, y, z], [x1 + 6, y, z], r, r, 1.5));
      }
    });
  };
  bundle(xylem, [3, 4, 3], 4.2, VEIN_Y + 8, 9.5);            // big vessels
  bundle(phloem, [4, 3], 3.2, VEIN_Y - 14, 7.5);             // smaller sieve tubes below

  // ---- spongy mesophyll: loose spheres around the vein, leaving air spaces
  const rng = makeRng(23), spheres = [], placed = [];
  for (let tries = 0; tries < 900 && placed.length < 85; tries++) {
    const r = 6 + rng.random() * 6;
    const x = (rng.random() - 0.5) * (W - 24), y = -44 + rng.random() * 70, z = (rng.random() - 0.5) * (D - 24);
    if ((y - VEIN_Y) ** 2 + z * z < (VEIN_R + r + 2) ** 2) continue;                       // keep clear of the vein
    if (placed.some((p) => Math.hypot(p[0] - x, p[1] - y, p[2] - z) < p[3] + r - 3)) continue;   // loose, slight overlap only
    placed.push([x, y, z, r]);
    spheres.push(ellipsoid([x, y, z], [r, r * 0.92, r], 1.3));
  }

  // ---- lower epidermis with a stoma: one cell removed, replaced by two guard cells (flattened curved rings)
  const STOMA = { x: 60, z: 15 };
  const lower = epidermis(-82, 18, 0.7, 37, (cx, cz) => Math.abs(cx - STOMA.x) < 1 && Math.abs(cz - STOMA.z) < 1);
  const guard = (sign) => paramSurface((u, v) => {          // half of an elliptic ring around the pore (bean-shaped guard cell)
    const a = u * Math.PI * sign, R = 13, t = 4.2;
    const rr = R + t * Math.cos(v);
    return [STOMA.x + rr * Math.cos(a) * 1.25 * (sign > 0 ? 1 : 1), -82 + t * 0.6 * Math.sin(v), STOMA.z + (rr * Math.sin(a)) * 0.8 + sign * 2];
  }, 0, 1, 0, TAU, { density: 20, gu: 24, gv: 12 });
  const guardCells = [guard(1), guard(-1)];

  return [
    G('Cuticle', C_CUTICLE, [0, 270, 0], 0.7, cuticle, { info: 'waxy waterproof film: stops the leaf drying out' }),
    G('Upper epidermis', C_UPPER, [0, 205, 0], 0.55, upper, { info: 'transparent skin cells that let light through' }),
    G('Palisade mesophyll', C_PALISADE, [0, 140, 0], 0.4, palisade, { info: 'tall packed cells full of chloroplasts: most photosynthesis' }),
    G('Bundle sheath (vein)', C_VASC, [0, 85, 0], 0.3, sheath, { info: 'wraps the vein and links it to the leaf cells' }),
    G('Phloem (food carrying)', C_PHLOEM, [0, 52, 0], 0.25, phloem, { info: 'tubes carrying sugar from the leaf to the plant' }),
    G('Xylem (water carrying)', C_VASC, [0, 22, 0], 0.2, xylem, { info: 'vessels bringing water and minerals up into the leaf' }),
    G('Spongy mesophyll', C_SPONGY, [0, -40, 0], 0.1, spheres, { info: 'loose round cells with air spaces for gas exchange' }),
    G('Lower epidermis', C_LOWER, [0, -125, 0], 0.5, lower, { info: 'protective skin on the underside, pierced by stomata' }),
    G('Stoma guard cells', C_GUARD, [0, -175, 0], 0.7, guardCells, { info: 'pair of cells that open and close the pore (stoma)' }),
  ];
}

export const LEAF = [
  { name: 'Leaf Cross Section', groups: leafCrossSectionProc, color: [0.4, 0.8, 0.25], tilt: 20, viewYaw: 0.5, fact: 'cuticle · epidermis · palisade · vein · spongy · stoma' },
];
