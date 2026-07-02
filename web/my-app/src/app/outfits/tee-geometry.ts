// Parametric, CAD-style construction of a t-shirt mesh.
//
// Structure:
//  1. Body tube: closed elliptical tube, hem → shaped top edge
//  2. Shoulder surface: 5-row smooth cap crossing front→back over each shoulder
//     (replaces the former flat single-row bridge; eliminates the
//     "hourglass / double-cone" artifact visible from 3/4 angles)
//  3. Armhole closing patches: zipper-fills the open armhole polygons
//  4. Collar band: thin lip rising from the neckline edge on front and back
//  5. Cap sleeves: compact elliptical rings lofted outward from shoulder tip

import * as THREE from "three";

import type { GarmentParams } from "@/lib/garment-mesh";

const SCALE = 0.04;
const TUBE_SEGS = 48;
const TUBE_ROWS = 28;
// Rows crossing front→back over each shoulder column (1 = flat, 5 = smooth).
const SHOULDER_CROSS_ROWS = 5;
// Collar band height in scene units (~0.8 cm at SCALE=0.04).
const COLLAR_H = 0.032;
const SLEEVE_RINGS = 10;
const SLEEVE_SEGS = 24;

type Vec3 = [number, number, number];

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const smoothstep = (t: number) => t * t * (3 - 2 * t);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

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

  const sNeck = clamp(params.neckWidthFront / params.bodyWidth, 0.12, 0.6);
  const sShoulder = 0.9;

  // Height of the shaped top edge at lateral fraction s (0=centre, 1=side seam).
  const topEdgeY = (s: number, neckDrop: number): number => {
    if (s <= sNeck) {
      const f = (1 + Math.cos((Math.PI * s) / sNeck)) / 2;
      return shoulderY - neckDrop * f;
    }
    if (s <= sShoulder) {
      return shoulderY - Math.tan(slopeRad) * (s - sNeck) * halfW;
    }
    const ysh = shoulderY - Math.tan(slopeRad) * (sShoulder - sNeck) * halfW;
    const t = (s - sShoulder) / (1 - sShoulder);
    return ysh - armDepth * smoothstep(t);
  };

  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  // ---- 1. Body tube -------------------------------------------------------
  // theta=0: front centre (+z), theta=π/2: right side (+x), theta=π: back (−z)
  // x = halfW·sin(θ),  z = depth·cos(θ)   CCW winding → outward normals

  const tubeStride = TUBE_SEGS + 1;

  for (let row = 0; row <= TUBE_ROWS; row++) {
    const v = row / TUBE_ROWS;
    for (let seg = 0; seg <= TUBE_SEGS; seg++) {
      const theta = (seg / TUBE_SEGS) * Math.PI * 2;
      const sinT = Math.sin(theta);
      const cosT = Math.cos(theta);
      const x = halfW * sinT;
      const z = depth * cosT;
      const nd = cosT >= 0 ? neckDropF : neckDropB;
      const topY = topEdgeY(Math.abs(sinT), nd);
      positions.push(x, hemY + v * (topY - hemY), z);
      uvs.push((sinT + 1) * 0.5, v);
    }
  }

  for (let row = 0; row < TUBE_ROWS; row++) {
    for (let seg = 0; seg < TUBE_SEGS; seg++) {
      const a = row * tubeStride + seg;
      const b = a + 1;
      const c = a + tubeStride;
      const d = c + 1;
      indices.push(a, b, c, b, d, c);
    }
  }

  // Shared geometry constants
  const halfSegs = TUBE_SEGS / 2;           // back-centre column (theta=π)
  const topBase = TUBE_ROWS * tubeStride;   // vertex offset of the top row
  const thetaN = Math.asin(clamp(sNeck, 0, 1));
  const thetaS = Math.asin(clamp(sShoulder, 0, 1));
  const segN = Math.max(1, Math.round(TUBE_SEGS * thetaN / (Math.PI * 2)));
  const segS = Math.min(halfSegs - 1, Math.round(TUBE_SEGS * thetaS / (Math.PI * 2)));

  // Stable index into the tube top row (handles seg=TUBE_SEGS wrap)
  const topIdx = (seg: number): number =>
    topBase + ((seg % (TUBE_SEGS + 1) + TUBE_SEGS + 1) % (TUBE_SEGS + 1));

  // ---- 2. Shoulder surface (multi-row smooth cap) -------------------------
  // Each shoulder column k (segN ≤ k ≤ segS) gets (SHOULDER_CROSS_ROWS−1)
  // intermediate vertices that linearly bridge from the front top-row vertex
  // to the back top-row vertex.  The z interpolation (front +depth·cos →
  // back −depth·cos) gives the shoulder 3-D roundness instead of a flat seam.

  const interRows = SHOULDER_CROSS_ROWS - 1; // new verts added per column

  const rShldBase = positions.length / 3;
  for (let k = segN; k <= segS; k++) {
    const tF = (k / TUBE_SEGS) * Math.PI * 2;
    const s = Math.abs(Math.sin(tF));
    const xF = halfW * Math.sin(tF);
    const yF = topEdgeY(s, neckDropF);
    const zF = depth * Math.cos(tF);         // positive (front side)
    const yB = topEdgeY(s, neckDropB);       // |sin(π−tF)| = sin(tF) = same s
    // cos(π−tF) = −cos(tF) so back z = −zF
    for (let r = 1; r < SHOULDER_CROSS_ROWS; r++) {
      const u = r / SHOULDER_CROSS_ROWS;
      positions.push(xF, lerp(yF, yB, u), lerp(zF, -zF, u));
      uvs.push((xF / halfW + 1) * 0.5, lerp(0.85, 1.0, u));
    }
  }

  const lShldBase = positions.length / 3;
  for (let k = segN; k <= segS; k++) {
    const tFL = ((TUBE_SEGS - k) / TUBE_SEGS) * Math.PI * 2;
    const s = Math.abs(Math.sin(tFL));
    const xFL = halfW * Math.sin(tFL);       // negative (left side)
    const yFL = topEdgeY(s, neckDropF);
    const zFL = depth * Math.cos(tFL);       // positive (front side for left shoulder)
    const yBL = topEdgeY(s, neckDropB);
    for (let r = 1; r < SHOULDER_CROSS_ROWS; r++) {
      const u = r / SHOULDER_CROSS_ROWS;
      positions.push(xFL, lerp(yFL, yBL, u), lerp(zFL, -zFL, u));
      uvs.push((xFL / halfW + 1) * 0.5, lerp(0.85, 1.0, u));
    }
  }

  // Vertex lookup: column k, cross-row r (0=front tube row, ROWS=back tube row)
  const rShldV = (k: number, r: number): number => {
    if (r === 0) return topIdx(k);
    if (r === SHOULDER_CROSS_ROWS) return topIdx(halfSegs - k);
    return rShldBase + (k - segN) * interRows + (r - 1);
  };
  const lShldV = (k: number, r: number): number => {
    if (r === 0) return topIdx(TUBE_SEGS - k);
    if (r === SHOULDER_CROSS_ROWS) return topIdx(halfSegs + k);
    return lShldBase + (k - segN) * interRows + (r - 1);
  };

  // Right shoulder quads — CCW from above gives upward (+y) normals
  for (let k = segN; k < segS; k++) {
    for (let r = 0; r < SHOULDER_CROSS_ROWS; r++) {
      const a = rShldV(k, r),     b = rShldV(k + 1, r);
      const c = rShldV(k, r + 1), d = rShldV(k + 1, r + 1);
      indices.push(a, b, c, b, d, c);
    }
  }
  // Left shoulder quads — reversed winding for mirrored geometry
  for (let k = segN; k < segS; k++) {
    for (let r = 0; r < SHOULDER_CROSS_ROWS; r++) {
      const a = lShldV(k, r),     b = lShldV(k + 1, r);
      const c = lShldV(k, r + 1), d = lShldV(k + 1, r + 1);
      indices.push(a, c, b, c, d, b);
    }
  }

  // ---- 3. Armhole closing patches ----------------------------------------
  // Zipper-triangulates the U-shaped armhole polygon on each side, reusing
  // existing top-row vertices so no new positions are needed.
  const zipFill = (verts: number[]) => {
    let lo = 0, hi = verts.length - 1;
    while (hi > lo + 1) {
      indices.push(verts[lo], verts[lo + 1], verts[hi]);
      lo++;
      if (hi > lo + 1) {
        indices.push(verts[lo], verts[hi - 1], verts[hi]);
        hi--;
      }
    }
  };

  {
    const v: number[] = [];
    for (let s = segS; s <= halfSegs - segS; s++) v.push(topBase + s);
    zipFill(v);
  }
  {
    const v: number[] = [];
    for (let s = halfSegs + segS; s <= TUBE_SEGS - segS; s++) v.push(topBase + s);
    zipFill(v);
  }

  // ---- 4. Collar band -----------------------------------------------------
  // A thin strip (COLLAR_H tall) above the neckline segs (|sin|≤sNeck) on
  // front and back arcs.  Makes the neckline look finished and closes the
  // top-down silhouette gap that the shoulder surface leaves open.

  const collarVMap = new Map<number, number>();

  const collarVert = (seg: number): number => {
    if (collarVMap.has(seg)) return collarVMap.get(seg)!;
    const theta = (seg / TUBE_SEGS) * Math.PI * 2;
    const sinT = Math.sin(theta);
    const cosT = Math.cos(theta);
    const nd = cosT >= 0 ? neckDropF : neckDropB;
    const y = topEdgeY(Math.abs(sinT), nd) + COLLAR_H;
    const vi = positions.length / 3;
    positions.push(halfW * sinT, y, depth * cosT);
    uvs.push((sinT + 1) * 0.5, 1.0);
    collarVMap.set(seg, vi);
    return vi;
  };

  // Emit collar quads along an ordered sequence of segs.
  // flipWinding=true: normals point toward front (+z) for the front collar.
  // flipWinding=false: normals point toward back (−z) for the back collar.
  const addCollarStrip = (segs: number[], flipWinding: boolean) => {
    for (let i = 0; i < segs.length - 1; i++) {
      const a = topIdx(segs[i]);
      const b = topIdx(segs[i + 1]);
      const c = collarVert(segs[i]);
      const d = collarVert(segs[i + 1]);
      if (flipWinding) {
        indices.push(a, d, b, a, c, d);
      } else {
        indices.push(a, b, d, a, d, c);
      }
    }
  };

  // Front neckline collar: right neck (segN) → front centre (0) → left neck
  const frontCollarSegs: number[] = [];
  for (let k = segN; k >= 0; k--) frontCollarSegs.push(k);
  for (let k = TUBE_SEGS - 1; k >= TUBE_SEGS - segN; k--) frontCollarSegs.push(k);
  addCollarStrip(frontCollarSegs, true);

  // Back neckline collar: right-back → back centre → left-back
  const backCollarSegs: number[] = [];
  for (let k = halfSegs - segN; k <= halfSegs + segN; k++) backCollarSegs.push(k);
  addCollarStrip(backCollarSegs, false);

  // ---- 5. Cap sleeves -----------------------------------------------------
  const yShoulderPt =
    shoulderY - Math.tan(slopeRad) * (sShoulder - sNeck) * halfW;
  const sleeveLen = params.sleeveLength * SCALE;

  const sleeveCY = yShoulderPt - armDepth * 0.28;
  const ry0 = armDepth * 0.22;   // compact cross-section height
  const rz0 = depth * 0.80;      // cross-section depth

  const buildSleeve = (sign: number) => {
    const rootX = sign * sShoulder * halfW;
    const ringBase = positions.length / 3;

    for (let i = 0; i <= SLEEVE_RINGS; i++) {
      const t = i / SLEEVE_RINGS;
      const cx = rootX + sign * sleeveLen * t;
      const cy = sleeveCY - sleeveLen * 0.30 * t;
      const ry = ry0 * (1 - 0.35 * t);
      const rz = rz0 * (1 - 0.25 * t);
      for (let j = 0; j <= SLEEVE_SEGS; j++) {
        const a = (j / SLEEVE_SEGS) * Math.PI * 2;
        positions.push(cx, cy + Math.cos(a) * ry, Math.sin(a) * rz);
        uvs.push(j / SLEEVE_SEGS, t);
      }
    }

    const segStride = SLEEVE_SEGS + 1;
    for (let i = 0; i < SLEEVE_RINGS; i++) {
      for (let j = 0; j < SLEEVE_SEGS; j++) {
        const a = ringBase + i * segStride + j;
        const b = a + 1;
        const c = a + segStride;
        const d = c + 1;
        if (sign > 0) {
          indices.push(a, b, c, b, d, c);
        } else {
          indices.push(a, c, b, c, d, b);
        }
      }
    }
  };

  buildSleeve(1);
  buildSleeve(-1);

  // ---- Assemble -----------------------------------------------------------
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
