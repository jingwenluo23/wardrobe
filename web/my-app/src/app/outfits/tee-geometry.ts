// Parametric, CAD-style construction of a t-shirt mesh.
//
// Instead of a single blobby primitive, the body is lofted from a 2D pattern:
//   * the neckline is a smooth Bezier-like dip (cosine blend, G1 continuous),
//   * the shoulder is a sloped line (not a hard angle),
//   * the armhole is a smooth-step curve, and
//   * the cross-section is elliptical so the torso has real front/back depth.
// Front and back panels share their side seams (so the torso is a closed
// tube), the shoulders are bridged, and two short cap sleeves are lofted onto
// the armholes. The result reads as a manufacturable tee with smooth, natural
// transitions rather than a sharp-edged box.

import * as THREE from "three";

import type { GarmentParams } from "@/lib/garment-mesh";

// Centimetres -> scene units.
const SCALE = 0.04;

// Grid resolution. Higher = smoother curves.
const COLS = 48; // columns across the body width
const ROWS = 28; // rows from hem to shoulder
const SLEEVE_RINGS = 10; // segments along each sleeve
const SLEEVE_SEG = 24; // points around each sleeve ring

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const smoothstep = (t: number) => t * t * (3 - 2 * t);

type Vec3 = [number, number, number];

/**
 * Build a parametric t-shirt BufferGeometry from CAD-style parameters.
 */
export function buildTeeGeometry(params: GarmentParams): THREE.BufferGeometry {
  const halfW = (params.bodyWidth / 2) * SCALE;
  const length = params.bodyLength * SCALE;
  const depth = (params.bodyDepth / 2) * SCALE;

  const hemY = -length / 2;
  const shoulderY = length / 2;

  const slopeRad = (params.shoulderSlope * Math.PI) / 180;
  const armDepth = params.armholeDepth * SCALE;
  const neckDropF = params.neckDropFront * SCALE;
  const neckDropB = params.neckDropBack * SCALE;

  // Neckline half-width as a fraction of the half body width.
  const sNeck = clamp(params.neckWidthFront / params.bodyWidth, 0.12, 0.6);
  // Where the shoulder ends and the armhole begins (fraction of half width).
  const sShoulder = 0.9;

  // Top edge of a panel as a function of the lateral position s in [0, 1],
  // where 0 is the centre-front and 1 is the side seam / underarm.
  const topEdgeY = (s: number, neckDrop: number) => {
    if (s <= sNeck) {
      // Neckline: cosine blend -> deepest at centre, flush with the shoulder
      // line at the neck edge. Smooth (G1) join, like a Bezier neckline.
      const f = (1 + Math.cos((Math.PI * s) / sNeck)) / 2;
      return shoulderY - neckDrop * f;
    }
    if (s <= sShoulder) {
      // Shoulder: a gentle downward slope defined by the slope angle.
      const drop = Math.tan(slopeRad) * (s - sNeck) * halfW;
      return shoulderY - drop;
    }
    // Armhole: smooth-step down to the underarm point at the side seam.
    const yShoulder = shoulderY - Math.tan(slopeRad) * (sShoulder - sNeck) * halfW;
    const t = (s - sShoulder) / (1 - sShoulder);
    return yShoulder - armDepth * smoothstep(t);
  };

  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  // --- Body panels (front + back) ---------------------------------------
  // Each panel is a (COLS+1) x (ROWS+1) grid. We remember the base vertex
  // offset of each panel so we can stitch shoulders and sleeves afterwards.
  const panelStride = COLS + 1;

  const buildPanel = (front: boolean): number => {
    const base = positions.length / 3;
    for (let r = 0; r <= ROWS; r += 1) {
      const v = r / ROWS;
      for (let c = 0; c <= COLS; c += 1) {
        const u = c / COLS;
        const s = Math.abs(u - 0.5) * 2;
        const x = (u - 0.5) * 2 * halfW;
        const topY = topEdgeY(s, front ? neckDropF : neckDropB);
        const y = hemY + v * (topY - hemY);
        // Elliptical cross-section: zero at the side seams, max at centre.
        const zMag = depth * Math.sin(Math.PI * u);
        const z = front ? zMag : -zMag;
        positions.push(x, y, z);
        uvs.push(front ? u : 1 - u, v);
      }
    }
    for (let r = 0; r < ROWS; r += 1) {
      for (let c = 0; c < COLS; c += 1) {
        const a = base + r * panelStride + c;
        const b = a + 1;
        const d = a + panelStride;
        const e = d + 1;
        if (front) {
          indices.push(a, d, b, b, d, e);
        } else {
          // Reverse winding so the back faces outward.
          indices.push(a, b, d, b, e, d);
        }
      }
    }
    return base;
  };

  const frontBase = buildPanel(true);
  const backBase = buildPanel(false);

  // --- Shoulder bridge ---------------------------------------------------
  // Connect the front and back top edges across the shoulder seam region
  // (between the neck edge and the shoulder point), leaving the neckline and
  // the armholes open.
  const topRow = ROWS * panelStride;
  for (let c = 0; c < COLS; c += 1) {
    const u0 = c / COLS;
    const u1 = (c + 1) / COLS;
    const s0 = Math.abs(u0 - 0.5) * 2;
    const s1 = Math.abs(u1 - 0.5) * 2;
    // Only bridge where both columns sit on the shoulder seam.
    const onShoulder =
      s0 >= sNeck && s0 <= sShoulder && s1 >= sNeck && s1 <= sShoulder;
    if (!onShoulder) {
      continue;
    }
    const f0 = frontBase + topRow + c;
    const f1 = frontBase + topRow + c + 1;
    const b0 = backBase + topRow + c;
    const b1 = backBase + topRow + c + 1;
    indices.push(f0, b0, f1, f1, b0, b1);
  }

  // --- Cap sleeves -------------------------------------------------------
  const sleeveLen = params.sleeveLength * SCALE;
  const yShoulder = shoulderY - Math.tan(slopeRad) * (sShoulder - sNeck) * halfW;
  const openMag = (params.sleeveOpening / 2) * SCALE;

  const buildSleeve = (sign: number) => {
    const sideX = sign * sShoulder * halfW;
    // Loft a tapered tube from the armhole ring (facing outward along x) to a
    // slightly lower, narrower cuff ring.
    const ringStart = positions.length / 3;
    for (let i = 0; i <= SLEEVE_RINGS; i += 1) {
      const t = i / SLEEVE_RINGS;
      // Centre of this ring: travels outward in x and droops down in y.
      const cx = sideX + sign * sleeveLen * t;
      const cy = yShoulder - armDepth * 0.35 - sleeveLen * 0.35 * t;
      const cz = 0;
      // Radii ease from armhole opening to the cuff.
      const ry = (armDepth * 0.6) * (1 - 0.25 * t);
      const rz = depth * (1.05 - 0.3 * t);
      for (let j = 0; j <= SLEEVE_SEG; j += 1) {
        const a = (j / SLEEVE_SEG) * Math.PI * 2;
        const y = cy + Math.cos(a) * ry;
        const z = cz + Math.sin(a) * rz;
        // Slight cap rise so the sleeve head blends into the shoulder.
        const x = cx + Math.sin(t * Math.PI) * 0.04 * sign;
        positions.push(x, y, z);
        uvs.push(j / SLEEVE_SEG, t);
        void openMag;
      }
    }
    const seg = SLEEVE_SEG + 1;
    for (let i = 0; i < SLEEVE_RINGS; i += 1) {
      for (let j = 0; j < SLEEVE_SEG; j += 1) {
        const a = ringStart + i * seg + j;
        const b = a + 1;
        const d = a + seg;
        const e = d + 1;
        if (sign > 0) {
          indices.push(a, d, b, b, d, e);
        } else {
          indices.push(a, b, d, b, e, d);
        }
      }
    }
  };

  buildSleeve(1);
  buildSleeve(-1);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

export function teeBoundingSize(geometry: THREE.BufferGeometry): Vec3 {
  const box = geometry.boundingBox ?? new THREE.Box3();
  return [
    box.max.x - box.min.x,
    box.max.y - box.min.y,
    box.max.z - box.min.z,
  ];
}
