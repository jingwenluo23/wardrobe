// Parametric, CAD-style garment construction from composable blocks.
//
// The torso follows a real 2D pattern (front/back panels with a Bezier-like
// neckline, sloped shoulder seam, concave armhole carved into the side edge,
// elliptical cross-section that opens around the armhole). Trim blocks are
// attached to the boundary loops the torso exposes:
//   - sleeves loft from the armhole edge loop (straight or tapered, with an
//     optional ribbed cuff),
//   - the neck opening gets a ribbed crew band or an attached hood,
//   - the hem optionally gets a ribbed sweatshirt band.
// Which blocks run — and with what proportions — comes from a GarmentFeatures
// preset, so every clothing type in the registry is data, not new geometry.

import * as THREE from "three";

import {
  defaultTeeFeatures,
  type GarmentFeatures,
  type GarmentParams,
} from "@/lib/garment-mesh";

// Centimetres -> scene units.
const SCALE = 0.04;

// Grid resolution. Higher = smoother curves.
const COLS = 64; // columns across the body width
const ROWS = 44; // rows from hem to shoulder
const SLEEVE_RINGS = 12; // rings along each sleeve
const HOOD_RINGS = 8; // rings from neckline to hood apex

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const smoothstep = (t: number) => {
  const c = clamp(t, 0, 1);
  return c * c * (3 - 2 * c);
};

/**
 * Build a garment BufferGeometry from CAD-style params + feature toggles.
 * Dispatches to the archetype block; with default tee features this
 * reproduces the original t-shirt exactly.
 */
export function buildGarmentGeometry(
  params: GarmentParams,
  features: GarmentFeatures = defaultTeeFeatures,
): THREE.BufferGeometry {
  if (features.archetype === "bottoms") {
    return buildBottomsGeometry(params, features);
  }
  return buildTopGeometry(params, features);
}

function buildTopGeometry(
  params: GarmentParams,
  features: GarmentFeatures,
): THREE.BufferGeometry {
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

  // Vertical depth taper: the torso keeps full depth up to the chest, then
  // front and back draw together toward the shoulder seam, so the shoulder
  // reads as a narrow rounded ridge from above instead of a flat deck.
  const depthTaper = (y: number) => {
    if (y <= underarmY) {
      return 1;
    }
    const t = (y - underarmY) / (neckShoulderY - underarmY);
    return 1 - 0.68 * smoothstep(t);
  };

  // Front/back separation along the armhole edge: zero at the underarm and
  // widest in the middle, so the opening reads as a smooth oval from the
  // side. Kept shallow (a real armhole is much flatter front-to-back than
  // the torso), and it does NOT close fully at the shoulder point — a real
  // sleeve cap stays rounded there instead of pinching to a crease.
  const maxOpen = depth * 0.5;
  const armholeGap = (y: number) => {
    if (y <= underarmY) {
      return 0;
    }
    const t = Math.min(1, (y - underarmY) / (shoulderPtY - underarmY));
    const dome = Math.pow(Math.sin(Math.PI * t), 0.8);
    const capRound = smoothstep(Math.min(1, t * 3));
    return maxOpen * (0.72 * dome + 0.28 * capRound);
  };

  // Top edge of a panel as a function of lateral position s in [0, 1]
  // (0 = centre front, 1 = shoulder point).
  const neckFrac = neckHalf / shoulderX;
  const topEdgeY = (s: number, neckDrop: number) => {
    if (s <= neckFrac) {
      // Crew neckline: superellipse blend -> flat-bottomed U at the centre
      // that turns up smoothly into the shoulder line.
      const t = s / neckFrac;
      const f = Math.pow(1 - Math.pow(t, 2.2), 0.8);
      return neckShoulderY - neckDrop * f;
    }
    // Shoulder: sloped seam from neck edge to shoulder point with a gentle
    // convex roll so the shoulder reads as soft fabric, not a straight bar.
    // The outermost stretch dips down in a fillet so the shoulder tip rounds
    // over into the sleeve cap instead of ending in a pinch.
    const t = (s - neckFrac) / (1 - neckFrac);
    const roll = 0.45 * SCALE * Math.sin(Math.PI * t);
    const fillet = 0.7 * SCALE * smoothstep((s - 0.88) / 0.12);
    return (
      neckShoulderY -
      Math.tan(slopeRad) * (s * shoulderX - neckHalf) +
      roll -
      fillet
    );
  };

  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  const pushVertex = (x: number, y: number, z: number, u: number, v: number) => {
    positions.push(x, y, z);
    uvs.push(u, v);
    return positions.length / 3 - 1;
  };

  const readVertex = (index: number) =>
    new THREE.Vector3(
      positions[index * 3],
      positions[index * 3 + 1],
      positions[index * 3 + 2],
    );

  // --- Block: body panels (front + back) --------------------------------
  const panelStride = COLS + 1;

  const buildPanel = (front: boolean): number => {
    const base = positions.length / 3;
    const neckDrop = front ? neckDropF : neckDropB;
    // The front panel sits slightly fuller than the back, like a real torso.
    const panelDepth = depth * (front ? 1.05 : 0.88);
    const foldPhase = front ? 0.4 : 2.1;
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
        const zEllipse =
          panelDepth * depthTaper(y) * Math.sqrt(Math.max(0, 1 - s * s));
        const mix = smoothstep((s - 0.78) / 0.22);
        const gap = armholeGap(y);
        // Soft vertical drape folds: strongest toward the hem, fading to
        // nothing at the shoulders and at the side seams.
        const fold =
          depth *
          0.055 *
          (0.55 * Math.sin(u * Math.PI * 8 + foldPhase) +
            0.45 * Math.sin(u * Math.PI * 3.2 + 1.3)) *
          (zEllipse / panelDepth) *
          smoothstep((0.9 - v) / 0.9);
        const z = (front ? 1 : -1) * (zEllipse + (gap - zEllipse) * mix + fold);
        // Texture V follows absolute height, not the per-column row index,
        // so prints stay straight instead of bending along the neckline.
        const vTex = (y - hemY) / (neckShoulderY - hemY);
        pushVertex(x, y, z, front ? u : 1 - u, vTex);
      }
    }
    for (let r = 0; r < ROWS; r += 1) {
      for (let c = 0; c < COLS; c += 1) {
        const a = base + r * panelStride + c;
        const b = a + 1;
        const d = a + panelStride;
        const e = d + 1;
        // Wound so face normals point outward (+z for front, -z for back).
        if (front) {
          indices.push(a, b, d, b, e, d);
        } else {
          indices.push(a, d, b, b, d, e);
        }
      }
    }
    return base;
  };

  const frontBase = buildPanel(true);
  const frontIndexEnd = indices.length;
  const backBase = buildPanel(false);
  const backIndexEnd = indices.length;

  const topRow = ROWS * panelStride;

  // --- Block: shoulder seam ----------------------------------------------
  // Bridge the front and back top edges between the neck edge and the
  // shoulder point through a slightly raised midline, so the seam is a soft
  // rounded roll rather than a flat crease. The neckline stays open.
  {
    const seamRise = 0.75 * SCALE;
    const shoulderMid: Record<number, number> = {};
    for (let c = 0; c <= COLS; c += 1) {
      const s = Math.abs(c / COLS - 0.5) * 2;
      // Start one column inside the neck arc so the bridge overlaps the
      // neck trim; otherwise the straddling quad belongs to neither and
      // leaves a slit at the neckline.
      if (s < neckFrac - 2.5 / COLS) {
        continue;
      }
      // The rounded roll fades out toward the shoulder tip, where the sleeve
      // cap takes over — a full-height ridge there sticks up as a loose fin.
      const rise = seamRise * (1 - smoothstep((s - 0.78) / 0.22));
      const fi = (frontBase + topRow + c) * 3;
      const bi = (backBase + topRow + c) * 3;
      shoulderMid[c] = pushVertex(
        (positions[fi] + positions[bi]) / 2,
        (positions[fi + 1] + positions[bi + 1]) / 2 + rise,
        (positions[fi + 2] + positions[bi + 2]) / 2,
        c / COLS,
        1,
      );
    }
    for (let c = 0; c < COLS; c += 1) {
      const m0 = shoulderMid[c];
      const m1 = shoulderMid[c + 1];
      if (m0 === undefined || m1 === undefined) {
        continue;
      }
      const f0 = frontBase + topRow + c;
      const f1 = frontBase + topRow + c + 1;
      const b0 = backBase + topRow + c;
      const b1 = backBase + topRow + c + 1;
      indices.push(f0, f1, m0, f1, m1, m0);
      indices.push(m0, m1, b0, m1, b1, b0);
    }
  }

  // Neck opening rim, shared by the neckband and hood blocks.
  // Closed loop: front neckline left->right, back neckline right->left.
  const neckLoop: THREE.Vector3[] = [];
  {
    const neckCols: number[] = [];
    for (let c = 0; c <= COLS; c += 1) {
      if (Math.abs(c / COLS - 0.5) * 2 <= neckFrac + 1e-9) {
        neckCols.push(c);
      }
    }
    for (const c of neckCols) {
      neckLoop.push(readVertex(frontBase + topRow + c));
    }
    for (let i = neckCols.length - 1; i >= 0; i -= 1) {
      neckLoop.push(readVertex(backBase + topRow + neckCols[i]));
    }
  }

  // --- Block: ribbed neckband ---------------------------------------------
  // A folded rib collar following the neck opening: outer ring on the raw
  // neckline edge, rising and turning slightly inward, like a sewn-on band.
  const buildNeckband = () => {
    const loopCount = neckLoop.length;
    if (loopCount < 4) {
      return;
    }
    const centroid = neckLoop
      .reduce((acc, p) => acc.clone().add(p), new THREE.Vector3())
      .multiplyScalar(1 / loopCount);

    const bandH = 1.25 * SCALE; // band height (cm)
    const bandIn = 1.1 * SCALE; // how far the band turns inward (cm)
    const rings: number[][] = [[], [], []];
    for (let j = 0; j < loopCount; j += 1) {
      const p = neckLoop[j];
      const inward = new THREE.Vector3(centroid.x - p.x, 0, centroid.z - p.z);
      if (inward.lengthSq() > 1e-10) {
        inward.normalize();
      }
      const mid = p
        .clone()
        .addScaledVector(inward, bandIn * 0.4)
        .add(new THREE.Vector3(0, bandH * 0.9, 0));
      const inner = p
        .clone()
        .addScaledVector(inward, bandIn)
        .add(new THREE.Vector3(0, bandH * 0.35, 0));
      rings[0].push(pushVertex(p.x, p.y, p.z, j / loopCount, 0));
      rings[1].push(pushVertex(mid.x, mid.y, mid.z, j / loopCount, 0.5));
      rings[2].push(pushVertex(inner.x, inner.y, inner.z, j / loopCount, 1));
    }
    for (let ring = 0; ring < 2; ring += 1) {
      for (let j = 0; j < loopCount; j += 1) {
        const jn = (j + 1) % loopCount;
        const a = rings[ring][j];
        const b = rings[ring][jn];
        const d = rings[ring + 1][j];
        const e = rings[ring + 1][jn];
        // Wound so the band's outer face points up and outward.
        indices.push(a, b, d, b, e, d);
      }
    }
  };

  // --- Block: hood ----------------------------------------------------------
  // An open hood lofted from the back arc of the neck rim: rings rise and
  // lean back while shrinking toward an apex, leaving the front arc of the
  // neckline as the face opening.
  const buildHood = () => {
    if (neckLoop.length < 6) {
      return;
    }
    const centroid = neckLoop
      .reduce((acc, p) => acc.clone().add(p), new THREE.Vector3())
      .multiplyScalar(1 / neckLoop.length);

    // Base arc: rim points whose angle around the neck centre is within
    // ~120 deg of the centre-back, ordered left-front -> back -> right-front.
    const arc = neckLoop
      .map((p) => ({ p, theta: Math.atan2(p.x - centroid.x, -(p.z - centroid.z)) }))
      .filter(({ theta }) => Math.abs(theta) <= 2.1)
      .sort((a, b) => a.theta - b.theta)
      .map(({ p }) => p);
    if (arc.length < 4) {
      return;
    }

    // Draped cowl: the hood rises from the neckline, folds back over the
    // shoulders and tapers closed low on the back — the natural resting
    // pose for an unworn hood.
    const arcCenter = arc
      .reduce((acc, p) => acc.clone().add(p), new THREE.Vector3())
      .multiplyScalar(1 / arc.length);

    const rings: number[][] = [];
    for (let i = 0; i <= HOOD_RINGS; i += 1) {
      const t = i / HOOD_RINGS;
      const lift =
        10 * Math.sin(Math.PI * Math.min(1, t * 1.15)) -
        9 * smoothstep((t - 0.5) / 0.5);
      const back = 13 * smoothstep(t);
      // The cowl spreads wider as it drapes (a resting hood is wider than
      // the neck opening) and pinches closed at its bottom edge.
      const widen = 1 + 0.5 * smoothstep(t);
      const pinch = t < 0.8 ? 1 : 1 - 0.7 * smoothstep((t - 0.8) / 0.2);
      const ringCenter = new THREE.Vector3(
        arcCenter.x,
        arcCenter.y + lift * SCALE,
        arcCenter.z - back * SCALE,
      );
      const ring: number[] = [];
      for (let j = 0; j < arc.length; j += 1) {
        const radial = arc[j].clone().sub(arcCenter);
        radial.y = 0;
        const p = new THREE.Vector3(
          ringCenter.x + radial.x * widen * pinch,
          ringCenter.y,
          ringCenter.z + radial.z * pinch,
        );
        ring.push(pushVertex(p.x, p.y, p.z, j / (arc.length - 1), t));
      }
      rings.push(ring);
    }
    // Open strip (no wrap): the gap between the arc ends is the face opening.
    for (let i = 0; i < HOOD_RINGS; i += 1) {
      for (let j = 0; j < arc.length - 1; j += 1) {
        const a = rings[i][j];
        const b = rings[i][j + 1];
        const d = rings[i + 1][j];
        const e = rings[i + 1][j + 1];
        indices.push(a, b, d, b, e, d);
      }
    }
  };

  // --- Block: ribbed hem band ----------------------------------------------
  // A snug sweatshirt band around the bottom opening: follows the hem edge,
  // dropping down and pulling slightly inward.
  const buildHemBand = () => {
    const bandH = 4.5 * SCALE;
    const pinch = 0.93;
    // Closed loop around the hem: front row left->right, back row right->left.
    const loop: THREE.Vector3[] = [];
    for (let c = 0; c <= COLS; c += 1) {
      loop.push(readVertex(frontBase + c));
    }
    for (let c = COLS; c >= 0; c -= 1) {
      loop.push(readVertex(backBase + c));
    }
    const loopCount = loop.length;
    const ringA: number[] = [];
    const ringB: number[] = [];
    for (let j = 0; j < loopCount; j += 1) {
      const p = loop[j];
      ringA.push(pushVertex(p.x, p.y, p.z, j / loopCount, 0));
      ringB.push(
        pushVertex(p.x * pinch, p.y - bandH, p.z * pinch, j / loopCount, 1),
      );
    }
    for (let j = 0; j < loopCount; j += 1) {
      const jn = (j + 1) % loopCount;
      // Wound so the band's outer face points outward/down.
      indices.push(ringA[j], ringB[j], ringA[jn], ringA[jn], ringB[j], ringB[jn]);
    }
  };

  // --- Block: sleeves lofted from the armhole boundary ---------------------
  // The armhole boundary is the panels' side-edge column above the underarm:
  // front edge going up, back edge coming down — a closed oval loop. The
  // sleeve's root ring reuses those exact positions, so the sleeve grows out
  // of the armhole with no gap or overlap.
  const sleeveLen = params.sleeveLength * SCALE;
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

    // Root loop: front edge bottom->top, then back edge top->bottom.
    const loop: THREE.Vector3[] = [];
    for (const r of edgeRows) {
      loop.push(readVertex(frontBase + r * panelStride + sideCol));
    }
    for (let i = edgeRows.length - 1; i >= 0; i -= 1) {
      loop.push(readVertex(backBase + edgeRows[i] * panelStride + sideCol));
    }

    const loopCount = loop.length;
    const centroid = loop
      .reduce((acc, p) => acc.clone().add(p), new THREE.Vector3())
      .multiplyScalar(1 / loopCount);

    // Sleeve axis: outward and drooping down, in the XY plane. Long sleeves
    // hang closer to vertical than a cap sleeve does.
    const lengthT = smoothstep((params.sleeveLength - 21) / 35);
    const droop = droopRad + lengthT * (26 * Math.PI) / 180;
    const axis = new THREE.Vector3(
      side * Math.cos(droop),
      -Math.sin(droop),
      0,
    );
    // Local frame perpendicular to the axis, for ring construction.
    const e2 = new THREE.Vector3(0, 0, 1);
    const e1 = new THREE.Vector3().crossVectors(e2, axis).normalize();

    // Every ring keeps the root ring's shape; the pointed top of the armhole
    // oval is softened along the way so the sleeve cap rounds off, and the
    // ring scale eases toward the feature's taper at the opening.
    const rootOffsets = loop.map((p) => p.clone().sub(centroid));
    const avgRootRadius =
      rootOffsets.reduce(
        (acc, offset) => acc + Math.hypot(offset.dot(e1), offset.dot(e2)),
        0,
      ) / loopCount;
    const softOffsets = rootOffsets.map((offset) => {
      const r = Math.hypot(offset.dot(e1), offset.dot(e2));
      const a = Math.atan2(offset.dot(e2), offset.dot(e1));
      // Pull extreme radii 35% toward the mean: same overall width, softer
      // corners.
      const rs = r + (avgRootRadius - r) * 0.35;
      return new THREE.Vector3()
        .addScaledVector(e1, Math.cos(a) * rs)
        .addScaledVector(e2, Math.sin(a) * rs);
    });

    const taper = clamp(features.sleeveTaper, 0.35, 1);

    const emitRing = (center: THREE.Vector3, scale: number, soften: number, v: number) => {
      const ring: number[] = [];
      for (let j = 0; j < loopCount; j += 1) {
        const offset = rootOffsets[j]
          .clone()
          .lerp(softOffsets[j], soften)
          .multiplyScalar(scale);
        const p = center.clone().add(offset);
        ring.push(pushVertex(p.x, p.y, p.z, j / loopCount, v));
      }
      return ring;
    };

    const stitchRings = (ringA: number[], ringB: number[]) => {
      for (let j = 0; j < loopCount; j += 1) {
        const jn = (j + 1) % loopCount;
        const a = ringA[j];
        const b = ringA[jn];
        const d = ringB[j];
        const e = ringB[jn];
        if (side > 0) {
          indices.push(a, d, b, b, d, e);
        } else {
          indices.push(a, b, d, b, e, d);
        }
      }
    };

    let previousRing: number[] | null = null;
    for (let i = 0; i <= SLEEVE_RINGS; i += 1) {
      const t = i / SLEEVE_RINGS;
      const soften = smoothstep(Math.min(1, t / 0.5));
      const scale = 1 + (taper - 1) * smoothstep(t);
      const ringCenter = centroid.clone().addScaledVector(axis, sleeveLen * t);
      const ring = emitRing(ringCenter, scale, soften, t);
      if (previousRing) {
        stitchRings(previousRing, ring);
      }
      previousRing = ring;
    }

    // Optional ribbed cuff: a short, snugger band past the sleeve end.
    if (features.cuff === "ribbed" && previousRing) {
      const cuffLen = 3 * SCALE;
      const cuffScale = taper * 0.8;
      const endCenter = centroid.clone().addScaledVector(axis, sleeveLen);
      const rib1 = emitRing(
        endCenter.clone().addScaledVector(axis, cuffLen * 0.15),
        cuffScale,
        1,
        1,
      );
      stitchRings(previousRing, rib1);
      const rib2 = emitRing(
        endCenter.clone().addScaledVector(axis, cuffLen),
        cuffScale,
        1,
        1,
      );
      stitchRings(rib1, rib2);
    }
  };

  // --- Assemble from features ----------------------------------------------
  if (features.neckFinish === "hood") {
    buildHood();
  } else {
    buildNeckband();
  }
  if (features.hemBand) {
    buildHemBand();
  }
  buildSleeve(1);
  buildSleeve(-1);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  // Material groups: 0 = front panel (front photo), 1 = back panel (back
  // photo), 2 = everything else (sleeves, seams, trims -> plain fabric).
  geometry.addGroup(0, frontIndexEnd, 0);
  geometry.addGroup(frontIndexEnd, backIndexEnd - frontIndexEnd, 1);
  geometry.addGroup(backIndexEnd, indices.length - backIndexEnd, 2);
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

// ---------------------------------------------------------------------------
// Bottoms archetype: trousers / shorts built from two mirrored leg lofts.
//
// Each leg is a tube of elliptical half-arcs (front + back emitted as
// separate strips so the panels get their own texture groups). Above the
// crotch the legs' inner edges are lifted to the garment depth so the fabric
// is continuous across the hip block; below the crotch the lift drops to
// zero and the legs separate. A waistband ring caps the top, and features
// add ribbed jogger cuffs and cargo pockets.
// ---------------------------------------------------------------------------

const LEG_SEGS = 22; // arc segments per panel half
const LEG_ROWS = 44; // rows from waist to hem

function buildBottomsGeometry(
  params: GarmentParams,
  features: GarmentFeatures,
): THREE.BufferGeometry {
  const waistHalf = ((params.waistWidth ?? 42) / 2) * SCALE;
  const hipHalf = ((params.hipWidth ?? 51) / 2) * SCALE;
  const rise = (params.rise ?? 28) * SCALE;
  const inseam = (params.inseam ?? 78) * SCALE;
  const thighHalf = ((params.thighWidth ?? 34) / 2) * SCALE;
  const openHalf = ((params.legOpening ?? 22) / 2) * SCALE;
  const depth = (params.bodyDepth / 2) * SCALE;

  const totalH = rise + inseam;
  const waistY = totalH / 2;
  const crotchY = waistY - rise;
  const hemY = -totalH / 2;

  // Per-leg radius (half-width of one leg) as a function of height. The
  // silhouette stays continuous (waist -> hip -> opening); the thigh
  // measurement acts as a mid-leg fullness factor (baggy vs slim).
  const thighFullness = clamp(thighHalf / (hipHalf / 2), 0.85, 1.3);
  const legRadius = (y: number) => {
    if (y >= crotchY) {
      const t = (waistY - y) / rise;
      return waistHalf / 2 + (hipHalf / 2 - waistHalf / 2) * smoothstep(t);
    }
    const t = (crotchY - y) / inseam;
    const base = hipHalf / 2 + (openHalf - hipHalf / 2) * Math.pow(t, 0.85);
    const fullness =
      1 + (thighFullness - 1) * Math.sin(Math.PI * Math.min(1, t * 1.6)) * 0.4;
    return base * fullness;
  };
  const crotchRadius = legRadius(crotchY);

  // Front/back depth of a leg's cross-section.
  const legDepth = (y: number) => {
    if (y >= crotchY) {
      return depth;
    }
    const t = (crotchY - y) / inseam;
    const hemDepth = Math.min(depth, openHalf * 0.95);
    return depth + (hemDepth - depth) * smoothstep(t * 1.1);
  };

  // Above the crotch the inner edges lift to full garment depth so the
  // front/back fabric runs continuously across both legs; at the crotch it
  // falls to zero and the legs separate.
  const innerLift = (y: number) => {
    if (y <= crotchY) {
      return 0;
    }
    return depth * 0.85 * smoothstep((y - crotchY) / (rise * 0.7));
  };

  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  const pushVertex = (x: number, y: number, z: number, u: number, v: number) => {
    positions.push(x, y, z);
    uvs.push(u, v);
    return positions.length / 3 - 1;
  };

  // Texture space: u spans the garment's full width, v runs hem -> waist.
  const texHalfW = hipHalf * 1.25;
  const uvFor = (x: number, y: number, front: boolean) => {
    const u = 0.5 + (front ? x : -x) / (2 * texHalfW);
    const v = (y - hemY) / totalH;
    return [clamp(u, 0, 1), v] as const;
  };

  // One panel strip: half-arcs of one leg, front or back.
  const buildLegPanel = (side: 1 | -1, front: boolean) => {
    const rows: number[][] = [];
    for (let r = 0; r <= LEG_ROWS; r += 1) {
      const y = waistY - (totalH * r) / LEG_ROWS;
      const rx = legRadius(y);
      const rz = legDepth(y) * (front ? 1 : 0.92);
      const lift = innerLift(y);
      // Leg centre: inner edges meet at x=0 above the crotch, then the
      // centre freezes so the legs part symmetrically below it.
      const cx = side * (y >= crotchY ? rx : crotchRadius);
      const ring: number[] = [];
      for (let j = 0; j <= LEG_SEGS; j += 1) {
        const phi = (j / LEG_SEGS) * Math.PI;
        const x = cx - side * Math.cos(phi) * rx;
        let z = Math.sin(phi) * rz;
        // Blend the inner quarter of the arc up to the lift height so the
        // hip front/back is continuous across the two legs.
        const innerW = smoothstep((0.3 - phi / Math.PI) / 0.3);
        z = z + (lift - z) * innerW * (lift > 0 ? 1 : 0);
        if (!front) {
          z = -z;
        }
        const [u, v] = uvFor(x, y, front);
        ring.push(pushVertex(x, y, z, u, v));
      }
      rows.push(ring);
    }
    for (let r = 0; r < LEG_ROWS; r += 1) {
      for (let j = 0; j < LEG_SEGS; j += 1) {
        const a = rows[r][j];
        const b = rows[r][j + 1];
        const d = rows[r + 1][j];
        const e = rows[r + 1][j + 1];
        // Outward winding flips with panel side and leg side.
        const flip = (front ? 1 : -1) * side;
        if (flip > 0) {
          indices.push(a, d, b, b, d, e);
        } else {
          indices.push(a, b, d, b, e, d);
        }
      }
    }
  };

  // Front panels of both legs first, then back panels, so the texture
  // groups stay contiguous.
  buildLegPanel(-1, true);
  buildLegPanel(1, true);
  const frontIndexEnd = indices.length;
  buildLegPanel(-1, false);
  buildLegPanel(1, false);
  const backIndexEnd = indices.length;

  // --- Block: waistband ----------------------------------------------------
  // An elliptical band hugging the top opening, folding slightly inward.
  {
    const bandH = 3.5 * SCALE;
    const segs = 48;
    const ringAt = (yOffset: number, scale: number, v: number) => {
      const ring: number[] = [];
      for (let j = 0; j <= segs; j += 1) {
        const beta = (j / segs) * Math.PI * 2;
        const x = Math.cos(beta) * waistHalf * scale;
        const z = Math.sin(beta) * depth * 0.93 * scale;
        ring.push(pushVertex(x, waistY + yOffset, z, j / segs, v));
      }
      return ring;
    };
    const r0 = ringAt(-0.8 * SCALE, 1.02, 0);
    const r1 = ringAt(bandH, 0.99, 0.5);
    const r2 = ringAt(bandH * 0.75, 0.9, 1);
    const stitch = (ra: number[], rb: number[]) => {
      for (let j = 0; j < segs; j += 1) {
        indices.push(ra[j], rb[j], ra[j + 1], ra[j + 1], rb[j], rb[j + 1]);
      }
    };
    stitch(r0, r1);
    stitch(r1, r2);
  }

  // --- Block: ribbed jogger cuffs -------------------------------------------
  if (features.cuff === "ribbed") {
    for (const side of [-1, 1] as const) {
      const cuffH = 4 * SCALE;
      const rx = legRadius(hemY) * 0.66;
      const rz = legDepth(hemY) * 0.66;
      const cx = side * crotchRadius;
      const segs = 28;
      const ringAt = (yOffset: number, scale: number, v: number) => {
        const ring: number[] = [];
        for (let j = 0; j <= segs; j += 1) {
          const beta = (j / segs) * Math.PI * 2;
          ring.push(
            pushVertex(
              cx + Math.cos(beta) * rx * scale,
              hemY + yOffset,
              Math.sin(beta) * rz * scale,
              j / segs,
              v,
            ),
          );
        }
        return ring;
      };
      const r0 = ringAt(1.2 * SCALE, 1.25, 0);
      const r1 = ringAt(-cuffH * 0.4, 1, 0.5);
      const r2 = ringAt(-cuffH, 0.96, 1);
      for (const [ra, rb] of [
        [r0, r1],
        [r1, r2],
      ] as const) {
        for (let j = 0; j < segs; j += 1) {
          indices.push(ra[j], rb[j], ra[j + 1], ra[j + 1], rb[j], rb[j + 1]);
        }
      }
    }
  }

  // --- Block: cargo pockets --------------------------------------------------
  if (features.cargoPockets) {
    for (const side of [-1, 1] as const) {
      const yTop = crotchY - 6 * SCALE;
      const yBottom = yTop - 17 * SCALE;
      const rxMid = legRadius((yTop + yBottom) / 2);
      const xFace = side * (side * (side * crotchRadius) + rxMid + 1.4 * SCALE);
      const xRoot = side * (crotchRadius + rxMid * 0.98);
      const zHalf = 6 * SCALE;
      // Outer face corners + root corners.
      const f = [
        pushVertex(xFace, yTop, zHalf, 0, 0.05),
        pushVertex(xFace, yTop, -zHalf, 0.05, 0.05),
        pushVertex(xFace, yBottom, -zHalf, 0.05, 0),
        pushVertex(xFace, yBottom, zHalf, 0, 0),
      ];
      const r = [
        pushVertex(xRoot, yTop + 0.8 * SCALE, zHalf, 0, 0.05),
        pushVertex(xRoot, yTop + 0.8 * SCALE, -zHalf, 0.05, 0.05),
        pushVertex(xRoot, yBottom - 0.4 * SCALE, -zHalf, 0.05, 0),
        pushVertex(xRoot, yBottom - 0.4 * SCALE, zHalf, 0, 0),
      ];
      const quad = (a: number, b: number, c: number, d: number) => {
        if (side > 0) {
          indices.push(a, b, c, a, c, d);
        } else {
          indices.push(a, c, b, a, d, c);
        }
      };
      quad(f[0], f[1], f[2], f[3]); // outer face
      quad(r[0], r[1], f[1], f[0]); // top
      quad(f[3], f[2], r[2], r[3]); // bottom
      quad(r[0], f[0], f[3], r[3]); // front side
      quad(f[1], r[1], r[2], f[2]); // back side
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.addGroup(0, frontIndexEnd, 0);
  geometry.addGroup(frontIndexEnd, backIndexEnd - frontIndexEnd, 1);
  geometry.addGroup(backIndexEnd, indices.length - backIndexEnd, 2);
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}
