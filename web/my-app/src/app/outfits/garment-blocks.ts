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
const HOOD_RINGS = 14; // rings from neckline to hood end

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const smoothstep = (t: number) => {
  const c = clamp(t, 0, 1);
  return c * c * (3 - 2 * c);
};

/**
 * Average vertex normals across all vertices that share a position, so seams
 * between separately-built pieces (e.g. the sleeve root ring welded onto the
 * torso's armhole edge) shade as one continuous surface. `computeVertexNormals`
 * treats those coincident-but-distinct vertices independently, which leaves a
 * hard normal crease — a visible "bump" — right at the join even though the
 * geometry is watertight. Welding by position (UVs untouched, so texture
 * groups are unaffected) removes it.
 */
function weldNormalsByPosition(geometry: THREE.BufferGeometry) {
  const position = geometry.getAttribute("position");
  const normal = geometry.getAttribute("normal");
  if (!position || !normal) {
    return;
  }
  const key = (i: number) =>
    Math.round(position.getX(i) * 1e4) +
    "," +
    Math.round(position.getY(i) * 1e4) +
    "," +
    Math.round(position.getZ(i) * 1e4);

  const buckets = new Map<string, THREE.Vector3>();
  const tmp = new THREE.Vector3();
  for (let i = 0; i < position.count; i += 1) {
    const k = key(i);
    let acc = buckets.get(k);
    if (!acc) {
      acc = new THREE.Vector3();
      buckets.set(k, acc);
    }
    acc.add(tmp.set(normal.getX(i), normal.getY(i), normal.getZ(i)));
  }
  for (let i = 0; i < position.count; i += 1) {
    const acc = buckets.get(key(i));
    if (!acc || acc.lengthSq() < 1e-12) {
      continue;
    }
    tmp.copy(acc).normalize();
    normal.setXYZ(i, tmp.x, tmp.y, tmp.z);
  }
  normal.needsUpdate = true;
}

/**
 * Build a garment BufferGeometry from CAD-style params + feature toggles.
 * Dispatches to the archetype block; with default tee features this
 * reproduces the original t-shirt exactly.
 */
export function buildGarmentGeometry(
  params: GarmentParams,
  features: GarmentFeatures = defaultTeeFeatures,
): THREE.BufferGeometry {
  let geometry: THREE.BufferGeometry;
  if (features.archetype === "bottoms") {
    geometry = buildBottomsGeometry(params, features);
    geometry.center();
  } else if (features.archetype === "skirt") {
    geometry = buildSkirtGeometry(params, features);
    geometry.center();
  } else {
    // buildTopGeometry already recenters on the body (excluding the hood).
    geometry = buildTopGeometry(params, features);
  }
  weldNormalsByPosition(geometry);
  return geometry;
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
  // shoulderWidthFactor < 0.8 narrows the straps (tanks, undershirts).
  const shoulderX = Math.max(
    neckHalf + halfW * 0.08,
    halfW * (params.shoulderWidthFactor ?? 0.8),
  );
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
    // Keep a real front/back gap at the top: the shoulder arch spans that gap,
    // so leaving it wider gives the arch a bigger radius and the shoulder a
    // full rounded top instead of a pinched flat ridge.
    return 1 - 0.62 * smoothstep(t);
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
  const vNeck = features.neckShape === "v";
  const topEdgeY = (s: number, neckDrop: number) => {
    if (s <= neckFrac) {
      const t = s / neckFrac;
      if (vNeck) {
        // V-neck: a near-linear taper from a rounded point at the centre
        // up to the shoulder line (cardigans, V-neck jerseys/sweaters).
        const f = Math.pow(1 - t, 1.15);
        return neckShoulderY - neckDrop * f;
      }
      // Crew neckline: superellipse blend -> flat-bottomed U at the centre
      // that turns up smoothly into the shoulder line.
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

  // Heavier fabrics get visibly chunkier trims (bands, cuffs, collars).
  const trimScale =
    features.fabric === "fleece" ? 1.6 : features.fabric === "knit" ? 1.35 : 1;

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
    // Bridge each column's front and back top edge with a rounded ARCH (a
    // half-ellipse of ARCN segments) instead of a single raised midline, so
    // the shoulder reads as a smooth rounded tube from above, not a flat deck
    // or a peaked crease.
    const ARCN = 5;
    // Low ellipse, not a semicircle: the shoulder rounds over without bulging
    // above the hood base — the seam crowns gently at collar height.
    const domeScale = 0.45;
    const cols: number[] = [];
    const arches: Record<number, number[]> = {};
    for (let c = 0; c <= COLS; c += 1) {
      const s = Math.abs(c / COLS - 0.5) * 2;
      if (s < neckFrac - 2.5 / COLS) {
        continue; // neckline stays open
      }
      // Fade the dome height out toward the shoulder tip where the sleeve cap
      // takes over, so it doesn't stick up as a fin.
      const fade = 1 - smoothstep((s - 0.78) / 0.22);
      const fi = (frontBase + topRow + c) * 3;
      const bi = (backBase + topRow + c) * 3;
      const fx = positions[fi];
      const fy = positions[fi + 1];
      const fz = positions[fi + 2];
      const bx = positions[bi];
      const by = positions[bi + 1];
      const bz = positions[bi + 2];
      const half = Math.abs(fz - bz) / 2; // radius of the arch
      const arch: number[] = [];
      for (let p = 1; p < ARCN; p += 1) {
        const tp = p / ARCN;
        const rise = half * domeScale * fade * Math.sin(Math.PI * tp);
        arch.push(
          pushVertex(
            fx + (bx - fx) * tp,
            fy + (by - fy) * tp + rise,
            fz + (bz - fz) * tp,
            c / COLS,
            1,
          ),
        );
      }
      arches[c] = arch;
      cols.push(c);
    }
    for (let ci = 0; ci < cols.length - 1; ci += 1) {
      const c = cols[ci];
      if (cols[ci + 1] !== c + 1) {
        continue; // only bridge adjacent columns
      }
      const a0 = arches[c];
      const a1 = arches[c + 1];
      const f0 = frontBase + topRow + c;
      const f1 = frontBase + topRow + c + 1;
      const b0 = backBase + topRow + c;
      const b1 = backBase + topRow + c + 1;
      // front edge -> first arch point
      indices.push(f0, f1, a0[0], f1, a1[0], a0[0]);
      // arch interior
      for (let p = 0; p < a0.length - 1; p += 1) {
        indices.push(a0[p], a1[p], a0[p + 1], a1[p], a1[p + 1], a0[p + 1]);
      }
      // last arch point -> back edge
      const last = a0.length - 1;
      indices.push(a0[last], a1[last], b0, a1[last], b1, b0);
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

    const bandH = 1.25 * SCALE * trimScale; // band height (cm)
    const bandIn = 1.1 * SCALE * trimScale; // how far the band turns inward (cm)
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

    // Standing hood: lofted rings rise tall over the collar and lean back
    // toward the spine, holding their volume through the body and rounding
    // closed at the crown — the structured hood in the reference flats,
    // not a flat cowl slumped on the back.
    const arcCenter = arc
      .reduce((acc, p) => acc.clone().add(p), new THREE.Vector3())
      .multiplyScalar(1 / arc.length);

    // Measure the neckline opening so the hood base matches it.
    const N = arc.length;
    let nHalfX = 0;
    let nHalfZ = 0;
    for (const p of arc) {
      nHalfX = Math.max(nHalfX, Math.abs(p.x - arcCenter.x));
      nHalfZ = Math.max(nHalfZ, Math.abs(p.z - arcCenter.z));
    }

    // Dropped hood = a soft bag hanging down the upper back. The mouth (the
    // face opening) sits at the top by the neckline, gaping UP so you look
    // straight into the cavity; the bag body hangs down and back to a closed,
    // rounded bottom that rests on the back panel. It is a closed elliptical
    // tube swept down a drooping spine — open only at the mouth — so it has
    // real pouch volume AND a genuine head opening, and it lies down under its
    // own weight instead of standing upright.
    //   - Principle: the hood always falls onto the back, never bolt upright.
    //   - Mouth ring (t=0) is left uncapped = the opening; the bottom converges
    //     to a closed tip.
    const cx = arcCenter.x;
    // Bigger, proportionate hood: mouth is wider than the neck opening and the
    // pouch is deep, so it reads as a full hood, not a small pocket.
    const rxBase = nHalfX * 1.4; // mouth half-width
    const rzBase = nHalfX * 0.55; // shallow front-to-back so it lies flat on back
    // Spine: the hood is sewn at the BACK neckline, then falls back and down
    // under gravity so it hangs BEHIND the back panel — never pushed through
    // it. The hanging portion sits at a fixed depth so its front face clears
    // the back panel by a small air gap (like a real dropped hood resting on,
    // but not fused into, the back).
    const gap = 3 * SCALE; // air gap between the hood's front face and the back
    const mouthY = arcCenter.y + 2 * SCALE; // just above the neckline (bridge rises to it)
    const tipY = arcCenter.y - 19 * SCALE;
    // Track the back panel's z profile per height: the hood hangs one gap
    // behind the LOCAL back surface at every level, so it follows the back
    // closely (connected, small even gap) and can never clip through. Above
    // the panel's top edge the profile is flat, so the mouth stays level and
    // the opening faces up.
    const yBins = 32;
    const backBin = new Array<number>(yBins).fill(Infinity);
    const pyLo = tipY;
    const pyHi = mouthY;
    for (let r = 0; r <= ROWS; r += 1) {
      for (let c = 0; c <= COLS; c += 1) {
        const bi = (backBase + r * panelStride + c) * 3;
        const bin = Math.floor(((positions[bi + 1] - pyLo) / (pyHi - pyLo)) * yBins);
        if (bin < 0 || bin >= yBins) {
          continue;
        }
        if (positions[bi + 2] < backBin[bin]) {
          backBin[bin] = positions[bi + 2];
        }
      }
    }
    let lastBin = -depth;
    for (let i = 0; i < yBins; i += 1) {
      if (!Number.isFinite(backBin[i])) {
        backBin[i] = lastBin;
      } else {
        lastBin = backBin[i];
      }
    }
    const backAtY = (y: number) => {
      let b = Math.round(((y - pyLo) / (pyHi - pyLo)) * yBins);
      b = Math.max(0, Math.min(yBins - 1, b));
      return backBin[b];
    };
    const spine = (u: number) => {
      const y = mouthY + (tipY - mouthY) * u;
      return { y, z: backAtY(y) - gap - rzBase };
    };

    const rings: number[][] = [];
    for (let i = 0; i <= HOOD_RINGS; i += 1) {
      const t = i / HOOD_RINGS;
      const c = spine(t);
      const cn = spine(Math.min(1, t + 1e-3));
      // Tangent down the spine; ring sits in the plane perpendicular to it.
      const ty = cn.y - c.y;
      const tz = cn.z - c.z;
      const tl = Math.hypot(ty, tz) || 1;
      const uy = ty / tl;
      const uz = tz / tl;
      // In-plane "depth" axis (perp to tangent, in the y-z plane); width axis
      // is world x. Together they orient the ellipse to face along the spine.
      const dy = -uz;
      const dz = uy;
      // Radius: full at the mouth, rounding to a closed tip at the bottom.
      const rad = Math.sqrt(Math.max(0, 1 - Math.pow(t, 2.2)));
      const rx = rxBase * rad;
      const rz = rzBase * rad;
      const ring: number[] = [];
      for (let j = 0; j < N; j += 1) {
        const a = (2 * Math.PI * j) / N; // closed loop
        const wx = rx * Math.cos(a);
        const depthOff = rz * Math.sin(a);
        let px = cx + wx;
        let py = c.y + depthOff * dy;
        let pz = c.z + depthOff * dz;
        // Extend the hood's OWN front edge down onto the neckline (no extra
        // collar piece). The front half of each upper ring maps 1:1 onto the
        // neckline arc (left end -> left shoulder ... right end -> right
        // shoulder) and is pulled toward it in x, y AND z together, so the
        // fabric forms one smooth curtain — mapping by nearest-x snapped many
        // vertices onto the same arc end and produced crossing spikes that
        // poked through the back.
        const frontness = Math.max(0, -Math.sin(a));
        const pull =
          smoothstep(frontness) * (1 - smoothstep(Math.min(1, t / 0.35)));
        if (pull > 0) {
          const f = (a - Math.PI) / Math.PI; // 0..1 across the front half
          const k = Math.min(N - 1, Math.max(0, Math.round(f * (N - 1))));
          const anchor = arc[k];
          px += (anchor.x - px) * pull;
          py += (anchor.y - py) * pull;
          pz += (anchor.z - pz) * pull;
        }
        ring.push(pushVertex(px, py, pz, j / N, t));
      }
      rings.push(ring);
    }
    // Stitch the closed tube (j wraps), leaving the mouth (t=0) open. Wound so
    // the printed outer fabric faces outward, seam allowance on the inside.
    for (let i = 0; i < HOOD_RINGS; i += 1) {
      for (let j = 0; j < N; j += 1) {
        const jn = (j + 1) % N;
        const a = rings[i][j];
        const b = rings[i][jn];
        const d = rings[i + 1][j];
        const e = rings[i + 1][jn];
        indices.push(a, d, b, b, d, e);
      }
    }

    // Folded-over front: the hood's opening edge continues past each shoulder
    // and drapes down the CHEST, the two edges crossing in a soft V below the
    // front neckline (right laid over left) — the dropped-hood look in the
    // reference flat. Each flap is a tapered band starting exactly at the
    // hood rim's end (same fabric, welded by position) and hugging the front
    // panel surface.
    {
      // Front panel surface z per height so the flaps lie on the chest.
      const fBins = 24;
      const fLo = arcCenter.y - 16 * SCALE;
      const fHi = arcCenter.y + 4 * SCALE;
      const frontBin = new Array<number>(fBins).fill(-Infinity);
      for (let r = 0; r <= ROWS; r += 1) {
        for (let c2 = 0; c2 <= COLS; c2 += 1) {
          const fi = (frontBase + r * panelStride + c2) * 3;
          const bin = Math.floor(
            ((positions[fi + 1] - fLo) / (fHi - fLo)) * fBins,
          );
          if (bin < 0 || bin >= fBins) {
            continue;
          }
          if (positions[fi + 2] > frontBin[bin]) {
            frontBin[bin] = positions[fi + 2];
          }
        }
      }
      let lastF = depth;
      for (let i2 = fBins - 1; i2 >= 0; i2 -= 1) {
        if (!Number.isFinite(frontBin[i2])) {
          frontBin[i2] = lastF;
        } else {
          lastF = frontBin[i2];
        }
      }
      const frontAtY = (y: number) => {
        let b = Math.round(((y - fLo) / (fHi - fLo)) * fBins);
        b = Math.max(0, Math.min(fBins - 1, b));
        return frontBin[b];
      };
      // Lowest point of the FRONT neckline curve (the centre dip).
      let fcY = Infinity;
      for (const p of neckLoop) {
        if (p.z > centroid.z && p.y < fcY) {
          fcY = p.y;
        }
      }
      const M = 12;
      for (const s of [-1, 1] as const) {
        const start = s < 0 ? arc[0] : arc[N - 1];
        // The right flap sits a little further off the chest so it reads as
        // laid OVER the left where they cross.
        const lift = s < 0 ? 0.8 * SCALE : 2.0 * SCALE;
        const tip = { x: -s * 2.5 * SCALE, y: fcY - 8 * SCALE };
        const ctrl = {
          x: start.x + s * 2 * SCALE,
          y: (start.y + tip.y) / 2 + 2.5 * SCALE,
        };
        const top: number[] = [];
        const bot: number[] = [];
        for (let m = 0; m <= M; m += 1) {
          const u = m / M;
          const iu = 1 - u;
          const bx = iu * iu * start.x + 2 * iu * u * ctrl.x + u * u * tip.x;
          const by = iu * iu * start.y + 2 * iu * u * ctrl.y + u * u * tip.y;
          const w = (5.5 - 3 * u) * SCALE; // band tapers toward the tip
          // Blend off the arc end onto the chest surface over the first bit.
          const settle = smoothstep(Math.min(1, u * 4));
          const zT = start.z + (frontAtY(by) + lift - start.z) * settle;
          const zB = frontAtY(by - w) + lift;
          top.push(pushVertex(bx, by, zT, u, 0.02));
          bot.push(pushVertex(bx, by - w, zB, u, 0.12));
        }
        for (let m = 0; m < M; m += 1) {
          const a = top[m];
          const b2 = bot[m];
          const d = top[m + 1];
          const e = bot[m + 1];
          // Wound so the printed face points out (+z) on both sides.
          if (s < 0) {
            indices.push(a, b2, d, d, b2, e);
          } else {
            indices.push(a, d, b2, d, e, b2);
          }
        }
      }
    }
  };

  // --- Block: ribbed hem band ----------------------------------------------
  // A snug sweatshirt band around the bottom opening: follows the hem edge,
  // dropping down and pulling slightly inward.
  const buildHemBand = () => {
    const bandH = 4.5 * SCALE * trimScale;
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

  // --- Block: patch pockets ------------------------------------------------
  // Two flat raised pockets low on the front (cardigans, camp shirts).
  const buildPatchPockets = () => {
    const pocketW = 13 * SCALE;
    const pocketH = 13 * SCALE;
    const cy = hemY + length * 0.24;
    const raise = 0.45 * SCALE;
    for (const side of [-1, 1] as const) {
      const cx = side * halfW * 0.42;
      const surfZ = depth * 1.05 * 0.72 + raise;
      const x0 = cx - pocketW / 2;
      const x1 = cx + pocketW / 2;
      const yTop = cy + pocketH / 2;
      const yBot = cy - pocketH / 2;
      // Rounded bottom corners via a small chamfer.
      const ch = 2 * SCALE;
      const face = [
        pushVertex(x0, yTop, surfZ, 0, 1),
        pushVertex(x1, yTop, surfZ, 1, 1),
        pushVertex(x1, yBot + ch, surfZ, 1, 0.1),
        pushVertex(x1 - ch, yBot, surfZ, 0.9, 0),
        pushVertex(x0 + ch, yBot, surfZ, 0.1, 0),
        pushVertex(x0, yBot + ch, surfZ, 0, 0.1),
      ];
      // Fan the front face.
      for (let i = 1; i < face.length - 1; i += 1) {
        indices.push(face[0], face[i], face[i + 1]);
      }
      // Thin sidewalls back to the body surface.
      const bodyZ = surfZ - raise;
      const back = face.map((_, i) => {
        const idx = face[i] * 3;
        return pushVertex(
          positions[idx],
          positions[idx + 1],
          bodyZ,
          0,
          0,
        );
      });
      for (let i = 0; i < face.length; i += 1) {
        const j = (i + 1) % face.length;
        indices.push(face[i], back[i], face[j], face[j], back[i], back[j]);
      }
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

    // Sleeve hang: starts nearly along the shoulder line and bends toward
    // vertical along the length, like fabric falling under gravity. A
    // straight steep axis pinched the armpit right at the root.
    const lengthT = smoothstep((params.sleeveLength - 21) / 35);
    const droopStart = slopeRad + (14 * Math.PI) / 180;
    const droopEnd = droopRad + (lengthT * 26 * Math.PI) / 180;
    const droopAt = (t: number) =>
      droopStart + (droopEnd - droopStart) * smoothstep(Math.min(1, t * 1.4));
    // Average axis for the ring frame.
    const droopMid = droopAt(0.5);
    const axis = new THREE.Vector3(
      side * Math.cos(droopMid),
      -Math.sin(droopMid),
      0,
    );
    // Local frame perpendicular to the axis, for ring construction.
    const e2 = new THREE.Vector3(0, 0, 1);
    const e1 = new THREE.Vector3().crossVectors(e2, axis).normalize();

    // Every ring keeps the root ring's shape; the pointed top of the armhole
    // oval is softened along the way so the sleeve cap rounds off, and the
    // ring scale eases toward the feature's taper at the opening.
    // Soft ring: a smooth, evenly spaced near-circle in the plane
    // perpendicular to the sleeve axis. The armhole loop's points are
    // unevenly spaced with corner remnants where the front and back edges
    // meet; carrying those angles down the tube left a crease along the
    // front and back of the sleeve. Unwrapping the angles and blending them
    // to a uniform sweep (and radii strongly toward the mean, using each
    // offset's FULL 3D length so steep axes don't collapse the ring) gives
    // clean cross-sections by mid-sleeve.
    const rootOffsets = loop.map((p) => p.clone().sub(centroid));
    const avgRootRadius =
      rootOffsets.reduce((acc, offset) => acc + offset.length(), 0) /
      loopCount;
    const rawAngles = rootOffsets.map((offset) =>
      Math.atan2(offset.dot(e2), offset.dot(e1)),
    );
    const unwrapped: number[] = [rawAngles[0]];
    for (let j = 1; j < loopCount; j += 1) {
      let delta = rawAngles[j] - rawAngles[j - 1];
      while (delta > Math.PI) delta -= Math.PI * 2;
      while (delta < -Math.PI) delta += Math.PI * 2;
      unwrapped.push(unwrapped[j - 1] + delta);
    }
    const sweep =
      Math.sign(unwrapped[loopCount - 1] - unwrapped[0] || 1) * Math.PI * 2;
    const softOffsets = rootOffsets.map((offset, j) => {
      const uniform = unwrapped[0] + (sweep * j) / loopCount;
      const a = unwrapped[j] + (uniform - unwrapped[j]) * 0.85;
      const r = offset.length();
      const rs = r + (avgRootRadius - r) * 0.7;
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

    // Ring centres follow the curved hang; shaping starts a little away
    // from the root so the armpit joins the torso without a pinch.
    let previousRing: number[] | null = null;
    const ringCenter = centroid.clone();
    const step = sleeveLen / SLEEVE_RINGS;
    for (let i = 0; i <= SLEEVE_RINGS; i += 1) {
      const t = i / SLEEVE_RINGS;
      if (i > 0) {
        const d = droopAt(t);
        ringCenter.add(
          new THREE.Vector3(side * Math.cos(d) * step, -Math.sin(d) * step, 0),
        );
      }
      const soften = smoothstep(Math.max(0, t - 0.12) / 0.5);
      const scale = 1 + (taper - 1) * smoothstep(Math.max(0, t - 0.15) / 0.85);
      const ring = emitRing(ringCenter.clone(), scale, soften, t);
      if (previousRing) {
        stitchRings(previousRing, ring);
      }
      previousRing = ring;
    }

    // Optional ribbed cuff: a short, snugger band past the sleeve end.
    if (features.cuff === "ribbed" && previousRing) {
      const cuffLen = 3 * SCALE * trimScale;
      const cuffScale = taper * 0.8;
      const endCenter = ringCenter.clone();
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

  // --- Block: turtleneck ------------------------------------------------------
  // A tall knit tube rising from the neck opening with a fold-over hint.
  const buildTurtleneck = () => {
    const loopCount = neckLoop.length;
    if (loopCount < 4) {
      return;
    }
    const rings: number[][] = [];
    const stages: Array<[number, number]> = [
      [0, 1], // rim
      [5, 0.97],
      [8.5, 0.99],
      [10, 1.04], // folded-over lip
    ];
    for (let k = 0; k < stages.length; k += 1) {
      const [rise, scale] = stages[k];
      const ring: number[] = [];
      for (let j = 0; j < loopCount; j += 1) {
        const q = neckLoop[j];
        ring.push(
          pushVertex(
            q.x * scale,
            q.y + rise * SCALE,
            q.z * scale,
            j / loopCount,
            k / (stages.length - 1),
          ),
        );
      }
      rings.push(ring);
    }
    for (let k = 0; k < rings.length - 1; k += 1) {
      for (let j = 0; j < loopCount; j += 1) {
        const jn = (j + 1) % loopCount;
        indices.push(
          rings[k][j],
          rings[k][jn],
          rings[k + 1][j],
          rings[k][jn],
          rings[k + 1][jn],
          rings[k + 1][j],
        );
      }
    }
  };

  // --- Block: folded collar (polo / shirt) -----------------------------------
  // A stand rising from the neckline, folding outward and down over itself.
  const buildFoldedCollar = (
    standH: number,
    foldScale: number,
    foldDrop: number,
  ) => {
    const loopCount = neckLoop.length;
    if (loopCount < 4) {
      return;
    }
    const stages: Array<[number, number]> = [
      [0, 1], // rim
      [standH, 0.96], // stand
      [standH - foldDrop * 0.35, foldScale * 0.55 + 0.45], // roll
      [standH - foldDrop, foldScale], // fold-over edge
    ];
    const rings: number[][] = [];
    for (let k = 0; k < stages.length; k += 1) {
      const [rise, scale] = stages[k];
      const ring: number[] = [];
      for (let j = 0; j < loopCount; j += 1) {
        const q = neckLoop[j];
        ring.push(
          pushVertex(
            q.x * scale,
            q.y + rise * SCALE,
            q.z * scale,
            j / loopCount,
            k / (stages.length - 1),
          ),
        );
      }
      rings.push(ring);
    }
    for (let k = 0; k < rings.length - 1; k += 1) {
      for (let j = 0; j < loopCount; j += 1) {
        const jn = (j + 1) % loopCount;
        indices.push(
          rings[k][j],
          rings[k][jn],
          rings[k + 1][j],
          rings[k][jn],
          rings[k + 1][jn],
          rings[k + 1][j],
        );
      }
    }
  };

  // --- Block: placket ---------------------------------------------------------
  // A raised button strip down the centre front: polo half-placket or a full
  // button front (shirts, cardigans), with small button bumps.
  const buildPlacket = (half: boolean) => {
    const stripHalf = 1.6 * SCALE;
    const raise = 0.7 * SCALE;
    const yTop = topEdgeY(0, neckDropF) + 0.4 * SCALE;
    const yBottom = half ? yTop - 16 * SCALE : hemY + 0.5 * SCALE;
    const rows = 14;
    const frontZ = (y: number) =>
      depth * 1.05 * depthTaper(y) + 0.15 * SCALE;
    const cols: number[][] = [[], [], [], []];
    for (let r = 0; r <= rows; r += 1) {
      const y = yTop + ((yBottom - yTop) * r) / rows;
      const zSurf = frontZ(y);
      const v = r / rows;
      cols[0].push(pushVertex(-stripHalf, y, zSurf, 0, v));
      cols[1].push(pushVertex(-stripHalf, y, zSurf + raise, 0.05, v));
      cols[2].push(pushVertex(stripHalf, y, zSurf + raise, 0.1, v));
      cols[3].push(pushVertex(stripHalf, y, zSurf, 0.15, v));
    }
    for (let c = 0; c < 3; c += 1) {
      for (let r = 0; r < rows; r += 1) {
        const a = cols[c][r];
        const b = cols[c + 1][r];
        const d = cols[c][r + 1];
        const e = cols[c + 1][r + 1];
        indices.push(a, b, d, b, e, d);
      }
    }
    // Button bumps down the strip.
    const buttonCount = Math.max(2, Math.floor((yTop - yBottom) / (9 * SCALE)));
    const bh = 0.55 * SCALE;
    for (let i = 0; i < buttonCount; i += 1) {
      const y = yTop - (i + 0.75) * ((yTop - yBottom) / buttonCount);
      const z = frontZ(y) + raise + 0.2 * SCALE;
      const a = pushVertex(-bh, y + bh, z, 0, 0);
      const b = pushVertex(bh, y + bh, z, 0.02, 0);
      const c = pushVertex(bh, y - bh, z, 0.02, 0.02);
      const d = pushVertex(-bh, y - bh, z, 0, 0.02);
      indices.push(a, b, c, a, c, d);
    }
  };

  // --- Assemble from features ----------------------------------------------
  // The hood's index range is tracked so it can carry its own material
  // (group 3): plain fabric plus a subtle stitched-seam bump texture.
  let hoodIndexStart = -1;
  let hoodIndexEnd = -1;
  let hoodVertStart = -1;
  let hoodVertEnd = -1;
  if (features.neckFinish === "hood") {
    hoodIndexStart = indices.length;
    hoodVertStart = positions.length / 3;
    buildHood();
    hoodVertEnd = positions.length / 3;
    hoodIndexEnd = indices.length;
  } else if (features.neckFinish === "turtleneck") {
    buildTurtleneck();
  } else if (features.neckFinish === "polo-collar") {
    buildFoldedCollar(2.6, 1.32, 1.8);
  } else if (features.neckFinish === "shirt-collar") {
    buildFoldedCollar(3.2, 1.5, 2.6);
  } else {
    buildNeckband();
  }
  if (features.placket === "half") {
    buildPlacket(true);
  } else if (features.placket === "full") {
    buildPlacket(false);
  }
  if (features.hemBand) {
    buildHemBand();
  }
  if (features.patchPockets) {
    buildPatchPockets();
  }
  // --- Block: dress skirt ----------------------------------------------
  // Lofts a skirt from the bodice hem: straight for bodycon, flared for
  // A-line, with a soft ripple gathering toward the hem.
  if (features.skirtLength && features.skirtLength > 0) {
    const skirtL = features.skirtLength * SCALE;
    const flare = clamp(features.skirtFlare ?? 0.4, 0, 1);
    const rings = 14;
    const loop: THREE.Vector3[] = [];
    for (let c = 0; c <= COLS; c += 1) {
      loop.push(readVertex(frontBase + c));
    }
    for (let c = COLS; c >= 0; c -= 1) {
      loop.push(readVertex(backBase + c));
    }
    const loopCount = loop.length;
    let previous: number[] | null = null;
    for (let i = 0; i <= rings; i += 1) {
      const t = i / rings;
      const spread = 1 + flare * 0.9 * t;
      const ripple = 0.035 * flare * smoothstep(t);
      const ring: number[] = [];
      for (let j = 0; j < loopCount; j += 1) {
        const q = loop[j];
        const wave = 1 + ripple * Math.sin((j / loopCount) * Math.PI * 10);
        ring.push(
          pushVertex(
            q.x * spread * wave,
            q.y - skirtL * t,
            q.z * spread * wave,
            j / loopCount,
            t,
          ),
        );
      }
      if (previous) {
        for (let j = 0; j < loopCount; j += 1) {
          const jn = (j + 1) % loopCount;
          indices.push(
            previous[j],
            ring[j],
            previous[jn],
            previous[jn],
            ring[j],
            ring[jn],
          );
        }
      }
      previous = ring;
    }
  }
  let sleeveIndexStart = -1;
  let sleeveIndexEnd = -1;
  if (features.sleeves !== false) {
    sleeveIndexStart = indices.length;
    buildSleeve(1);
    buildSleeve(-1);
    sleeveIndexEnd = indices.length;
  }

  // Recenter on the BODY (front/back panels, sleeves, trims) — deliberately
  // excluding the hood. The hood hangs down and back; if it were included the
  // bounding-box centre would drift off the torso and the viewer's orbit pivot
  // would circle a point behind/under the garment. Centring on the body keeps
  // the spin axis through the middle of the garment regardless of the hood.
  {
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    const count = positions.length / 3;
    for (let v = 0; v < count; v += 1) {
      if (v >= hoodVertStart && v < hoodVertEnd) {
        continue; // skip hood vertices
      }
      const x = positions[v * 3];
      const y = positions[v * 3 + 1];
      const z = positions[v * 3 + 2];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
    }
    const cX = (minX + maxX) / 2;
    const cY = (minY + maxY) / 2;
    const cZ = (minZ + maxZ) / 2;
    for (let v = 0; v < count; v += 1) {
      positions[v * 3] -= cX;
      positions[v * 3 + 1] -= cY;
      positions[v * 3 + 2] -= cZ;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  // Material groups: 0 = front panel (front photo), 1 = back panel (back
  // photo), 2 = seams/trims (plain fabric), 3 = hood (extracted hood texture
  // + stitched-seam bump), 4 = sleeves (extracted sleeve texture).
  geometry.addGroup(0, frontIndexEnd, 0);
  geometry.addGroup(frontIndexEnd, backIndexEnd - frontIndexEnd, 1);
  const special: Array<[number, number, number]> = [];
  if (hoodIndexStart >= 0 && hoodIndexEnd > hoodIndexStart) {
    special.push([hoodIndexStart, hoodIndexEnd, 3]);
  }
  if (sleeveIndexStart >= 0 && sleeveIndexEnd > sleeveIndexStart) {
    special.push([sleeveIndexStart, sleeveIndexEnd, 4]);
  }
  special.sort((a, b) => a[0] - b[0]);
  let cursor = backIndexEnd;
  for (const [start, end, material] of special) {
    if (start > cursor) {
      geometry.addGroup(cursor, start - cursor, 2);
    }
    geometry.addGroup(start, end - start, material);
    cursor = end;
  }
  if (indices.length > cursor) {
    geometry.addGroup(cursor, indices.length - cursor, 2);
  }
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
  const thighFullness = clamp(thighHalf / (hipHalf / 2), 0.85, 1.5);
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
  // An elliptical band hugging the top opening. "flat" is a crisp tailored
  // band with belt loops (jeans, chinos, dress trousers); "elastic" is a
  // thicker gathered band with a hanging drawcord (joggers, sports shorts).
  {
    const waistStyle = features.waistband ?? "flat";
    const bandH = (waistStyle === "elastic" ? 4.4 : 3.2) * SCALE;
    const segs = 48;
    const ringAt = (yOffset: number, scale: number, v: number) => {
      const ring: number[] = [];
      for (let j = 0; j <= segs; j += 1) {
        const beta = (j / segs) * Math.PI * 2;
        // Elastic bands gather: a subtle vertical ripple around the ring.
        const gather =
          waistStyle === "elastic" ? 1 + 0.012 * Math.sin(beta * 14) : 1;
        const x = Math.cos(beta) * waistHalf * scale * gather;
        const z = Math.sin(beta) * depth * 0.93 * scale * gather;
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

    if (waistStyle === "flat") {
      // Belt loops: slim raised tabs spaced around the band.
      const loopH = 2.7 * SCALE;
      const loopW = 0.95 * SCALE;
      const lift = 0.55 * SCALE;
      for (const deg of [58, 122, 180, 238, 302]) {
        const beta = (deg * Math.PI) / 180;
        const nx = Math.cos(beta);
        const nz = Math.sin(beta);
        const cx = nx * waistHalf * 1.01;
        const cz = nz * depth * 0.94;
        // Tangent along the band for the tab's width.
        const tx = -nz * loopW;
        const tz = nx * loopW * (depth / waistHalf);
        const yTop = waistY + bandH * 0.9;
        const yBottom = waistY - 0.4 * SCALE + (loopH - loopH);
        const ox = nx * lift;
        const oz = nz * lift * (depth / waistHalf);
        const a = pushVertex(cx - tx / 2 + ox, yTop, cz - tz / 2 + oz, 0, 0);
        const b = pushVertex(cx + tx / 2 + ox, yTop, cz + tz / 2 + oz, 0.02, 0);
        const c2 = pushVertex(
          cx + tx / 2 + ox,
          yBottom - loopH,
          cz + tz / 2 + oz,
          0.02,
          0.02,
        );
        const d = pushVertex(
          cx - tx / 2 + ox,
          yBottom - loopH,
          cz - tz / 2 + oz,
          0,
          0.02,
        );
        indices.push(a, b, c2, a, c2, d, a, c2, b, a, d, c2);
      }
    } else {
      // Drawcord: two short cords hanging at the centre front.
      const cordL = 7 * SCALE;
      const cordW = 0.45 * SCALE;
      const zFront = depth * 0.93 + 0.35 * SCALE;
      for (const side of [-1, 1] as const) {
        const x0 = side * 1.1 * SCALE;
        const x1 = x0 + side * cordW;
        const yTop = waistY + bandH * 0.35;
        const a = pushVertex(x0, yTop, zFront, 0, 0);
        const b = pushVertex(x1, yTop, zFront, 0.01, 0);
        const c2 = pushVertex(
          x1 + side * 0.6 * SCALE,
          yTop - cordL,
          zFront - 0.2 * SCALE,
          0.01,
          0.02,
        );
        const d = pushVertex(
          x0 + side * 0.6 * SCALE,
          yTop - cordL,
          zFront - 0.2 * SCALE,
          0,
          0.02,
        );
        indices.push(a, b, c2, a, c2, d, a, c2, b, a, d, c2);
      }
    }
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


// ---------------------------------------------------------------------------
// Skirt archetype: a single waist-to-hem loft (mini / midi / maxi / pencil).
// Front and back half-arc panels keep the photo texture groups; the
// silhouette runs waist -> hip -> hem, straight or flared, with a soft
// ripple gathering toward a flared hem.
// ---------------------------------------------------------------------------

const SKIRT_SEGS = 26;
const SKIRT_ROWS = 34;

function buildSkirtGeometry(
  params: GarmentParams,
  features: GarmentFeatures,
): THREE.BufferGeometry {
  const waistHalf = ((params.waistWidth ?? 38) / 2) * SCALE;
  const hipHalf = ((params.hipWidth ?? 50) / 2) * SCALE;
  const hemHalf = ((params.legOpening ?? 50) / 2) * SCALE;
  const hipDrop = (params.rise ?? 18) * SCALE;
  const length = (params.inseam ?? 60) * SCALE;
  const depth = (params.bodyDepth / 2) * SCALE;

  const waistY = length / 2;
  const hemY = -length / 2;
  const hipY = waistY - hipDrop;

  const radiusAt = (y: number) => {
    if (y >= hipY) {
      const t = (waistY - y) / hipDrop;
      return waistHalf + (hipHalf - waistHalf) * smoothstep(t);
    }
    const t = (hipY - y) / (hipY - hemY);
    return hipHalf + (hemHalf - hipHalf) * Math.pow(t, 0.9);
  };
  const flareRatio = Math.max(0, hemHalf / hipHalf - 1);

  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  const pushVertex = (x: number, y: number, z: number, u: number, v: number) => {
    positions.push(x, y, z);
    uvs.push(u, v);
    return positions.length / 3 - 1;
  };

  const buildPanel = (front: boolean) => {
    const rows: number[][] = [];
    for (let r = 0; r <= SKIRT_ROWS; r += 1) {
      const t = r / SKIRT_ROWS;
      const y = waistY - length * t;
      const rx = radiusAt(y);
      const rz = depth * Math.min(1.6, rx / hipHalf) * (front ? 1 : 0.92);
      const ripple = 0.03 * Math.min(1, flareRatio) * smoothstep(t);
      const ring: number[] = [];
      for (let j = 0; j <= SKIRT_SEGS; j += 1) {
        const phi = (j / SKIRT_SEGS) * Math.PI;
        const wave = 1 + ripple * Math.sin(phi * 9 + (front ? 0 : 1.4));
        const x = -Math.cos(phi) * rx * wave;
        const z = (front ? 1 : -1) * Math.sin(phi) * rz * wave;
        const u = front ? j / SKIRT_SEGS : 1 - j / SKIRT_SEGS;
        ring.push(pushVertex(x, y, z, u, 1 - t));
      }
      rows.push(ring);
    }
    for (let r = 0; r < SKIRT_ROWS; r += 1) {
      for (let j = 0; j < SKIRT_SEGS; j += 1) {
        const a = rows[r][j];
        const b = rows[r][j + 1];
        const d = rows[r + 1][j];
        const e = rows[r + 1][j + 1];
        if (front) {
          indices.push(a, d, b, b, d, e);
        } else {
          indices.push(a, b, d, b, e, d);
        }
      }
    }
  };

  buildPanel(true);
  const frontIndexEnd = indices.length;
  buildPanel(false);
  const backIndexEnd = indices.length;

  // Waistband: slim closed band around the top edge.
  {
    const bandH = 2.6 * SCALE;
    const segs = 44;
    const ringAt = (yOffset: number, scale: number, v: number) => {
      const ring: number[] = [];
      for (let j = 0; j <= segs; j += 1) {
        const beta = (j / segs) * Math.PI * 2;
        ring.push(
          pushVertex(
            Math.cos(beta) * waistHalf * scale,
            waistY + yOffset,
            Math.sin(beta) * depth * 0.95 * scale,
            j / segs,
            v,
          ),
        );
      }
      return ring;
    };
    const r0 = ringAt(-0.6 * SCALE, 1.02, 0);
    const r1 = ringAt(bandH, 0.99, 0.5);
    const r2 = ringAt(bandH * 0.7, 0.9, 1);
    for (const [ra, rb] of [
      [r0, r1],
      [r1, r2],
    ] as const) {
      for (let j = 0; j < segs; j += 1) {
        indices.push(ra[j], rb[j], ra[j + 1], ra[j + 1], rb[j], rb[j + 1]);
      }
    }
  }

  void features;

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
