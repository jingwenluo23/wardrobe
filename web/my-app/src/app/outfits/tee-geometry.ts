// Parametric, CAD-style construction of a t-shirt mesh.
//
// The body follows a real 2D pattern (see the garment CAD reference):
//   * front/back panels with a smooth Bezier-like neckline dip,
//   * a sloped shoulder line ending at the shoulder point,
//   * an armhole CURVE carved into the panel's side edge — the panel width
//     shrinks from the underarm up to the shoulder point, exactly like the
//     concave armhole on a flat pattern,
//   * an elliptical cross-section that opens up around the armhole so the
//     front and back edges separate to form a real arm opening.
// Each sleeve is then lofted FROM the armhole boundary curve itself (its root
// ring reuses the armhole edge vertices), tapering to a circular cuff. That
// keeps a smooth, seam-true transition from shoulder to sleeve with no
// floating cylinders.

import * as THREE from "three";

import type { GarmentParams } from "@/lib/garment-mesh";

// Centimetres -> scene units.
const SCALE = 0.04;

// Grid resolution. Higher = smoother curves.
const COLS = 64; // columns across the body width
const ROWS = 44; // rows from hem to shoulder
const SLEEVE_RINGS = 12; // rings along each sleeve

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const smoothstep = (t: number) => {
  const c = clamp(t, 0, 1);
  return c * c * (3 - 2 * c);
};

/**
 * Build a parametric t-shirt BufferGeometry from CAD-style parameters.
 */
export function buildTeeGeometry(params: GarmentParams): THREE.BufferGeometry {
  const halfW = (params.bodyWidth / 2) * SCALE;
  const length = params.bodyLength * SCALE;
  const depth = (params.bodyDepth / 2) * SCALE;

  const hemY = -length / 2;
  const neckShoulderY = length / 2; // highest point: shoulder line at the neck

  const slopeRad = (params.shoulderSlope * Math.PI) / 180;
  const armDepth = params.armholeDepth * SCALE;
  const neckDropF = params.neckDropFront * SCALE;
  const neckDropB = params.neckDropBack * SCALE;
  const neckHalf = clamp(
    (params.neckWidthFront / 2) * SCALE,
    halfW * 0.15,
    halfW * 0.6,
  );

  // Shoulder point: end of the shoulder seam, start of the armhole curve.
  const shoulderX = Math.max(neckHalf + halfW * 0.12, halfW * 0.8);
  const shoulderPtY =
    neckShoulderY - Math.tan(slopeRad) * (shoulderX - neckHalf);
  // Underarm point: bottom of the armhole curve, on the side seam.
  const underarmY = shoulderPtY - armDepth;

  // Panel half-width as a function of height. Below the underarm this is the
  // (slightly tapered) side seam; above it, the concave armhole curve pulls
  // the edge inward until it reaches the shoulder point.
  const widthAt = (y: number) => {
    if (y <= underarmY) {
      // Gentle A-line: a touch wider at the hem than at the chest.
      const t = (y - hemY) / (underarmY - hemY);
      return halfW * (1.03 - 0.03 * smoothstep(t));
    }
    const t = clamp((y - underarmY) / (shoulderPtY - underarmY), 0, 1);
    return halfW - (halfW - shoulderX) * smoothstep(t);
  };

  // Front/back separation along the armhole edge: zero at the underarm and
  // at the shoulder point, widest in the middle, so the opening reads as a
  // smooth oval when seen from the side.
  const maxOpen = depth * 0.85;
  const armholeGap = (y: number) => {
    if (y <= underarmY || y >= shoulderPtY) {
      return 0;
    }
    const t = (y - underarmY) / (shoulderPtY - underarmY);
    return maxOpen * Math.sin(Math.PI * t);
  };

  // Top edge of a panel as a function of lateral position s in [0, 1]
  // (0 = centre front, 1 = shoulder point).
  const neckFrac = neckHalf / shoulderX;
  const topEdgeY = (s: number, neckDrop: number) => {
    if (s <= neckFrac) {
      // Neckline: cosine blend -> deepest at centre, meets the shoulder line
      // tangentially at the neck edge (G1 continuous, like a Bezier).
      const f = (1 + Math.cos((Math.PI * s) / neckFrac)) / 2;
      return neckShoulderY - neckDrop * f;
    }
    // Shoulder: straight sloped seam from neck edge to shoulder point.
    return neckShoulderY - Math.tan(slopeRad) * (s * shoulderX - neckHalf);
  };

  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  const pushVertex = (x: number, y: number, z: number, u: number, v: number) => {
    positions.push(x, y, z);
    uvs.push(u, v);
    return positions.length / 3 - 1;
  };

  // --- Body panels (front + back) ---------------------------------------
  const panelStride = COLS + 1;

  const buildPanel = (front: boolean): number => {
    const base = positions.length / 3;
    const neckDrop = front ? neckDropF : neckDropB;
    for (let r = 0; r <= ROWS; r += 1) {
      const v = r / ROWS;
      for (let c = 0; c <= COLS; c += 1) {
        const u = c / COLS;
        const s = Math.abs(u - 0.5) * 2;
        const topY = topEdgeY(s, neckDrop);
        const y = hemY + v * (topY - hemY);
        const w = widthAt(y);
        const x = Math.sign(u - 0.5 || 1) * s * w;
        // Elliptical cross-section, blended into the armhole gap near the
        // side edge so the front/back separate around the arm opening.
        const zEllipse = depth * Math.sqrt(Math.max(0, 1 - s * s));
        const mix = smoothstep((s - 0.78) / 0.22);
        const gap = armholeGap(y);
        const z = (front ? 1 : -1) * (zEllipse + (gap - zEllipse) * mix);
        pushVertex(x, y, z, front ? u : 1 - u, v);
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
          indices.push(a, b, d, b, e, d);
        }
      }
    }
    return base;
  };

  const frontBase = buildPanel(true);
  const backBase = buildPanel(false);

  // --- Shoulder seam -----------------------------------------------------
  // Bridge the front and back top edges between the neck edge and the
  // shoulder point, leaving the neckline open.
  const topRow = ROWS * panelStride;
  for (let c = 0; c < COLS; c += 1) {
    const s0 = Math.abs(c / COLS - 0.5) * 2;
    const s1 = Math.abs((c + 1) / COLS - 0.5) * 2;
    if (Math.min(s0, s1) < neckFrac) {
      continue;
    }
    const f0 = frontBase + topRow + c;
    const f1 = frontBase + topRow + c + 1;
    const b0 = backBase + topRow + c;
    const b1 = backBase + topRow + c + 1;
    indices.push(f0, b0, f1, f1, b0, b1);
  }

  // --- Sleeves lofted from the armhole boundary --------------------------
  // The armhole boundary is the panels' side-edge column above the underarm:
  // front edge going up, back edge coming down — a closed oval loop. The
  // sleeve's root ring reuses those exact positions, so the sleeve grows out
  // of the armhole with no gap or overlap.
  const sleeveLen = params.sleeveLength * SCALE;
  const cuffRadius = Math.max((params.sleeveOpening / 2) * SCALE, 0.05);
  const droopRad = slopeRad + (24 * Math.PI) / 180;

  const buildSleeve = (side: 1 | -1) => {
    const sideCol = side > 0 ? COLS : 0;

    // Collect armhole edge rows (side column, from underarm to shoulder).
    const edgeRows: number[] = [];
    for (let r = 0; r <= ROWS; r += 1) {
      const idx = (frontBase + r * panelStride + sideCol) * 3;
      const y = positions[idx + 1];
      if (y >= underarmY - 1e-6) {
        edgeRows.push(r);
      }
    }
    if (edgeRows.length < 2) {
      return;
    }

    const readVec = (base: number, r: number) => {
      const idx = (base + r * panelStride + sideCol) * 3;
      return new THREE.Vector3(
        positions[idx],
        positions[idx + 1],
        positions[idx + 2],
      );
    };

    // Root loop: front edge bottom->top, then back edge top->bottom.
    const loop: THREE.Vector3[] = [];
    for (const r of edgeRows) {
      loop.push(readVec(frontBase, r));
    }
    for (let i = edgeRows.length - 1; i >= 0; i -= 1) {
      loop.push(readVec(backBase, edgeRows[i]));
    }

    const loopCount = loop.length;
    const centroid = loop
      .reduce((acc, p) => acc.add(p), new THREE.Vector3())
      .multiplyScalar(1 / loopCount);

    // Sleeve axis: outward and drooping down, in the XY plane.
    const axis = new THREE.Vector3(
      side * Math.cos(droopRad),
      -Math.sin(droopRad),
      0,
    );
    // Local frame perpendicular to the axis, for the circular cuff.
    const e2 = new THREE.Vector3(0, 0, 1);
    const e1 = new THREE.Vector3().crossVectors(e2, axis).normalize();

    const cuffCenter = centroid.clone().addScaledVector(axis, sleeveLen);

    // Map each loop point to an angle around the axis so the cuff circle
    // keeps the same vertex ordering (no twist along the sleeve).
    const cuffPoints = loop.map((p) => {
      const offset = p.clone().sub(centroid);
      const a = Math.atan2(offset.dot(e2), offset.dot(e1));
      return cuffCenter
        .clone()
        .addScaledVector(e1, Math.cos(a) * cuffRadius)
        .addScaledVector(e2, Math.sin(a) * cuffRadius);
    });

    // Loft rings from the armhole loop to the cuff.
    const ringStart = positions.length / 3;
    for (let i = 0; i <= SLEEVE_RINGS; i += 1) {
      const t = i / SLEEVE_RINGS;
      const blend = smoothstep(t);
      for (let j = 0; j < loopCount; j += 1) {
        const p = loop[j].clone().lerp(cuffPoints[j], blend);
        pushVertex(p.x, p.y, p.z, j / loopCount, t);
      }
    }
    for (let i = 0; i < SLEEVE_RINGS; i += 1) {
      for (let j = 0; j < loopCount; j += 1) {
        const jn = (j + 1) % loopCount;
        const a = ringStart + i * loopCount + j;
        const b = ringStart + i * loopCount + jn;
        const d = ringStart + (i + 1) * loopCount + j;
        const e = ringStart + (i + 1) * loopCount + jn;
        if (side > 0) {
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
