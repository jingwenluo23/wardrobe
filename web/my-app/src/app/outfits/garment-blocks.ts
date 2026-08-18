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
const SLEEVE_RINGS = 18; // rings along each sleeve

// --- Sleeve kinds -----------------------------------------------------------
// A cap sleeve and a long sleeve behave like different garments: one stands out
// from the shoulder, is short and taut and barely falls; the other reaches the
// wrist, carries real weight and drapes. They used to share a single formula
// with a smoothstep on sleeve length blending between them, so every adjustment
// aimed at one silently moved the other and neither was ever stated outright.
// Each kind now declares its own numbers.
//
// The construction (lofting rings off the armhole) is deliberately still
// shared: it is the part that must stay correct for both, and duplicating it
// would let the two drift apart.
type SleeveKind = "cap" | "long";

type SleeveProfile = {
  /** Degrees below the shoulder line where the sleeve leaves the armhole.
   *  Deliberately shallow for every kind: the armhole ring faces sideways, so
   *  a steep root extrudes the tube almost within its own plane and collapses
   *  it — that is what reads as a pinched upper arm. */
  rootOffsetDeg: number;
  /** Degrees below the shoulder line at the cuff. Past 90 the sleeve swings
   *  back toward the body, which is how a long sleeve gets its cuff to hang
   *  beside the hip rather than out to the side. */
  endOffsetDeg: number;
  /** Depth of the long gravity folds down the sleeve. */
  drape: number;
  /** Depth of the finer tension wrinkles. */
  tension: number;
  /** Fabric stacking just above the cuff band. */
  stack: number;
  /** Ease across the sleeve cap — the domed head that sets into the armhole
   *  (step 2/3 of the construction sheet). */
  capEase: number;
  /** Width at the cuff edge relative to the bicep. The pattern pieces taper
   *  gently from the cap down to the hem allowance. */
  wristTaper: number;
  /** Rib cuff folded in half lengthwise, so the band is doubled and its lower
   *  edge is a fold rather than a raw opening (steps 5-6). */
  foldedRibCuff: boolean;
  /** Build the sleeve as a tube hanging from the SHOULDER rather than one
   *  extruded out of the armhole and then turned down. See buildSleeve. */
  hangFromShoulder: boolean;
  /** Sleeve radius at the bicep, as a fraction of the body's half width.
   *  Sizing the tube from the armhole's area instead gave a sleeve about a
   *  fifth of the thickness a real one has, because the armhole opening is a
   *  narrow slit between the front and back panels rather than a full
   *  cross-section of the arm. */
  bicepRadiusFactor: number;
};

const SLEEVE_PROFILES: Record<SleeveKind, SleeveProfile> = {
  // Short cap sleeve: flares gently outward off the shoulder and stops. Too
  // short and too taut to develop meaningful folds, so it stays nearly smooth.
  cap: {
    rootOffsetDeg: 18,
    endOffsetDeg: 24,
    drape: 0.012,
    tension: 0.008,
    stack: 0,
    capEase: 0,
    wristTaper: 1,
    foldedRibCuff: false,
    // A cap sleeve genuinely does project out of the armhole — that is its
    // whole shape — so it keeps the extruded construction.
    hangFromShoulder: false,
    bicepRadiusFactor: 0.3,
  },
  // Long sleeve: leaves the armhole shallow, then falls past vertical so the
  // cuff comes to rest beside the body. Long enough to hang under its own
  // weight, and it stacks where the cuff band stops it.
  long: {
    // Steeper than a true shoulder-line start: the raglan seam already carries
    // the sleeve outward, so the tube itself can turn down sooner and bring the
    // cuff in beside the body.
    rootOffsetDeg: 20,
    // Root is shallow because the first part of a raglan sleeve IS the
    // shoulder — it travels outward across the top of the body before the arm
    // starts. It then falls to vertical at the cuff.
    // A little past vertical, so the cuff settles in toward the hip without
    // meeting the torso.
    endOffsetDeg: 69,
    // Shallow: the reference sleeve is a smooth, clean cone. Deep folds make
    // the outline wobble, which reads as the sleeve flaring in and out.
    drape: 0.018,
    tension: 0.010,
    stack: 0.012,
    capEase: 0,
    // Narrows steadily from armhole to wrist, as the reference does. A
    // constant-width tube keeps its full armhole girth all the way down and
    // then steps abruptly into a much narrower cuff, and that combination is
    // what reads as the sleeve flaring out.
    wristTaper: 0.84,
    foldedRibCuff: true,
    hangFromShoulder: true,
    // ~0.6 x half width across, so the sleeve reads about 0.3 of the body
    // width — where a relaxed hoodie's bicep sits.
    bicepRadiusFactor: 0.25,
  },
};

/** Sleeves past mid-forearm hang and drape; shorter ones behave as caps. */
const LONG_SLEEVE_CM = 40;
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

  // Accumulate unnormalised face normals into POSITION buckets. Summing the
  // already-normalised vertex normals gives every separately-built piece the
  // same vote regardless of its adjacent triangle area, which leaves a dark
  // groove at the torso/sleeve join. Face cross-products are area-weighted and
  // reproduce the result of a genuinely shared topological vertex while the
  // duplicate vertices—and therefore their independent UVs—remain intact.
  const buckets = new Map<string, THREE.Vector3>();
  const index = geometry.getIndex();
  const pa = new THREE.Vector3();
  const pb = new THREE.Vector3();
  const pc = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const faceNormal = new THREE.Vector3();
  const addFace = (ia: number, ib: number, ic: number) => {
    pa.fromBufferAttribute(position, ia);
    pb.fromBufferAttribute(position, ib);
    pc.fromBufferAttribute(position, ic);
    ab.subVectors(pb, pa);
    ac.subVectors(pc, pa);
    faceNormal.crossVectors(ab, ac);
    if (faceNormal.lengthSq() < 1e-16) {
      return;
    }
    for (const i of [ia, ib, ic]) {
      const k = key(i);
      let acc = buckets.get(k);
      if (!acc) {
        acc = new THREE.Vector3();
        buckets.set(k, acc);
      }
      acc.add(faceNormal);
    }
  };
  if (index) {
    for (let i = 0; i < index.count; i += 3) {
      addFace(index.getX(i), index.getX(i + 1), index.getX(i + 2));
    }
  } else {
    for (let i = 0; i < position.count; i += 3) {
      addFace(i, i + 1, i + 2);
    }
  }
  const tmp = new THREE.Vector3();
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
 * Fair shading across recorded sleeve-cap rows without moving vertices or UVs.
 * This distributes the remaining normal rotation from the welded torso seam
 * through the upper arm, so lighting does not reveal a rigid attachment line.
 */
function fairSleeveAttachmentNormals(geometry: THREE.BufferGeometry) {
  const rowSets = geometry.userData.sleeveNormalRows as
    | number[][][]
    | undefined;
  const normal = geometry.getAttribute("normal");
  if (!rowSets?.length || !normal) {
    return;
  }
  const start = new THREE.Vector3();
  const end = new THREE.Vector3();
  const current = new THREE.Vector3();
  const target = new THREE.Vector3();
  for (const rows of rowSets) {
    if (rows.length < 3) {
      continue;
    }
    const last = rows.length - 1;
    const around = Math.min(...rows.map((row) => row.length));
    for (let j = 0; j < around; j += 1) {
      start.fromBufferAttribute(normal, rows[0][j]).normalize();
      end.fromBufferAttribute(normal, rows[last][j]).normalize();
      if (start.dot(end) < 0) {
        end.negate();
      }
      for (let k = 1; k < last; k += 1) {
        const t = smoothstep(k / last);
        target.copy(start).lerp(end, t).normalize();
        current.fromBufferAttribute(normal, rows[k][j]).normalize();
        if (current.dot(target) < 0) {
          current.negate();
        }
        current.lerp(target, 0.78).normalize();
        normal.setXYZ(rows[k][j], current.x, current.y, current.z);
      }
    }
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
  fairSleeveAttachmentNormals(geometry);
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
  // Hard cap below halfW: widthAt() interpolates from halfW at the underarm to
  // shoulderX at the shoulder, so a shoulder wider than the body inverts that
  // term and the panel flares OUTWARD above the underarm — a bulged shoulder
  // with a broken armhole instead of a concave curve.
  //
  // Drop-shoulder padding: for a sleeved garment the shoulder point is carried
  // half-way out toward the body's full width. That is what a relaxed hoodie
  // does — the shoulder seam sits out on the arm and the armhole stands nearly
  // vertical — and it is the move that lets the sleeve hang STEEPLY without
  // driving into the torso, so the cuff can sit close to the body. The padding
  // works by approaching halfW, never exceeding it: past halfW the widthAt()
  // term inverts and the panel bulges outward instead of curving in.
  //
  // RAGLAN for long sleeves: the shoulder belongs to the SLEEVE, not the body.
  //
  // While the torso carries a shoulder cap, a sleeve can only be attached to
  // the side of it, and the join always reads as two parts meeting — the notch
  // at the top of the arm that no amount of positioning removed. In a raglan
  // the body ends at a diagonal seam running from the neckline down to the
  // underarm and the sleeve piece carries the whole shoulder, so the outline
  // sweeps unbroken from the neck, over the shoulder, down to the cuff.
  //
  // Narrowing the panel to a point near the neck turns its top edge into
  // exactly that seam, and the sleeve lofted from that edge inherits the
  // shoulder rather than having one bolted on.
  const raglan =
    features.sleeves !== false && params.sleeveLength >= LONG_SLEEVE_CM;
  const shoulderBase = Math.max(
    neckHalf + halfW * 0.08,
    halfW * (params.shoulderWidthFactor ?? 0.8),
  );
  const dropShoulder = features.sleeves === false || raglan ? 0 : 0.5;
  const shoulderX = raglan
    ? clamp(Math.max(neckHalf * 1.2, halfW * 0.38), neckHalf * 1.1, halfW * 0.5)
    : clamp(
        shoulderBase + (halfW * 0.97 - shoulderBase) * dropShoulder,
        neckHalf + halfW * 0.06,
        halfW * 0.97,
      );
  const shoulderPtY =
    neckShoulderY - Math.tan(slopeRad) * (shoulderX - neckHalf);
  // Underarm point: bottom of the armhole curve, on the side seam.
  const underarmY = shoulderPtY - armDepth;

  // Panel half-width as a function of height. Below the underarm this is the
  // (slightly tapered) side seam; above it, the concave armhole curve pulls
  // the edge inward until it reaches the shoulder point.
  // Fitted silhouette: an hourglass — the waist nips in on a bell curve
  // centred on the natural waist (upper third of the torso), while the hip
  // (hem) and bust (chest) flare slightly WIDER, so it reads as a curvy
  // female block, not just a narrower straight tube.
  const waistPinch = features.waistPinch ?? 0;
  const pinchAt = (y: number) => {
    if (waistPinch <= 0 || y > underarmY) {
      return 1;
    }
    const t = (y - hemY) / (underarmY - hemY); // 0 = hem, 1 = underarm
    // Subtle shaping, per the reference flats: one wide, gentle waist curve
    // with a whisper of hip/bust fullness — aggressive bells turn the side
    // seam into an unnatural S-wave.
    const waist = Math.exp(-Math.pow((t - 0.6) / 0.3, 2)); // nip at the waist
    const hip = Math.exp(-Math.pow(t / 0.32, 2)); // slight ease at the hem
    const bust = Math.exp(-Math.pow((t - 0.85) / 0.2, 2)); // slight bust ease
    const dev = -waistPinch * waist + 0.18 * waistPinch * (0.6 * hip + bust);
    // Fade the whole deviation to 0 right at the underarm so the torso meets
    // the chest with no step/crease, and taper it in gently at the hem.
    const fade =
      smoothstep(Math.min(1, (1 - t) / 0.14)) * smoothstep(Math.min(1, t / 0.1 + 0.2));
    return 1 + dev * fade;
  };

  const widthAt = (y: number) => {
    if (y <= underarmY) {
      // Gentle A-line: a touch wider at the hem than at the chest.
      const t = (y - hemY) / (underarmY - hemY);
      const hemFactor = params.hemWidthFactor ?? 1.03;
      return halfW * (hemFactor + (1 - hemFactor) * smoothstep(t)) * pinchAt(y);
    }
    const t = clamp((y - underarmY) / (shoulderPtY - underarmY), 0, 1);
    // On a raglan the seam has a long way to travel — underarm to neckline —
    // and easing it symmetrically starts pulling the body in from just above
    // the underarm, which narrows the chest and makes the torso look squeezed.
    // Hold close to full width and turn in late instead, so the body keeps its
    // shape and the seam does its work up near the neck.
    const shape = raglan ? Math.pow(t, 2.1) : smoothstep(t);
    return halfW - (halfW - shoulderX) * shape;
  };

  // Vertical depth taper: the torso keeps full depth up to the chest, then
  // front and back draw together toward the shoulder seam, so the shoulder
  // reads as a narrow rounded ridge from above instead of a flat deck.
  const depthTaper = (y: number) => {
    if (y <= underarmY) {
      // The fitted waist pinches depth too (softer than width) so the
      // silhouette stays rounded, not flattened.
      return 1 - (1 - pinchAt(y)) * 0.6;
    }
    const t = (y - underarmY) / (neckShoulderY - underarmY);
    // Keep a real front/back gap at the top: the shoulder arch spans that gap,
    // so leaving it wider gives the arch a bigger radius and the shoulder a
    // full rounded top instead of a pinched flat ridge.
    return 1 - 0.62 * smoothstep(t);
  };

  // Front/back separation only at the armhole boundary.  This edge becomes
  // the sleeve seam, so it needs enough depth to form a stitchable loop but
  // must stay much flatter than the torso itself.  A deep oval here leaves
  // two visible lobes (front and back) beside the sleeve in side view.
  const maxOpen = depth * 0.28;
  const armholeGap = (y: number) => {
    if (y <= underarmY) {
      return 0;
    }
    const t = Math.min(1, (y - underarmY) / (shoulderPtY - underarmY));
    // Close naturally at both ends of the seam.  The tiny cap term prevents
    // coincident front/back shoulder vertices while remaining hidden inside
    // the sleeve cap; it does not change the torso away from this boundary.
    const dome = Math.pow(Math.sin(Math.PI * t), 1.15);
    const capRound = smoothstep(Math.min(1, t * 3));
    return maxOpen * (0.94 * dome + 0.06 * capRound);
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
        // A woven stripe follows the fabric grain, not the width of each mesh
        // row. Row-normalised U stretched the full photograph across the
        // narrowing upper panel, fanning vertical stripes sideways over the
        // shoulder. Project U from the vertex's real X position instead.
        const planarHalf = halfW * (params.hemWidthFactor ?? 1.03);
        const planarU = clamp(0.5 + x / (2 * planarHalf), 0, 1);
        const uTex = features.fabric === "woven" ? planarU : u;
        pushVertex(x, y, z, front ? uTex : 1 - uTex, vTex);
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

  // Round the neck opening in plan view. The raw panel top edges leave a
  // near-square hole: straight front/back edges meeting the side columns at
  // hard corners. Pull the top-edge depth toward the centreline as the
  // columns approach the neck sides (elliptical profile), and feather the
  // row below so the opening wall stays smooth. The neckLoop (and therefore
  // the collar, neckband and hood) is built from these vertices, so every
  // neck finish inherits the rounded opening.
  for (let c = 0; c <= COLS; c += 1) {
    const s = (Math.abs(c / COLS - 0.5) * 2) / neckFrac;
    if (s > 1) {
      continue;
    }
    const round = Math.sqrt(Math.max(0, 1 - s * s));
    const k = 0.22 + 0.78 * round;
    const fi = (frontBase + topRow + c) * 3;
    const bi = (backBase + topRow + c) * 3;
    positions[fi + 2] *= k;
    positions[bi + 2] *= k;
    const fi2 = (frontBase + (ROWS - 1) * panelStride + c) * 3;
    const bi2 = (backBase + (ROWS - 1) * panelStride + c) * 3;
    positions[fi2 + 2] *= (1 + k) / 2;
    positions[bi2 + 2] *= (1 + k) / 2;
  }

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

    // Base arc: rim points around the centre-back that the hood is sewn to.
    //
    // Kept to ~85 deg either side of centre-back, so the seam ends near the
    // shoulders. A wider arc reaches around to the front neckline, and because
    // the hood's front rim is welded onto this arc the extra span hangs across
    // the neck opening as a flat strip bridging the two front edges — the
    // "collar band" spanning the front. Ending at the shoulders leaves the
    // front neckline open, which is where a real hood's seam stops.
    const arc = neckLoop
      .map((p) => ({ p, theta: Math.atan2(p.x - centroid.x, -(p.z - centroid.z)) }))
      .filter(({ theta }) => Math.abs(theta) <= 1.5)
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
      // Radius: rounding to a closed tip at the bottom. The MOUTH stays close
      // to neck width (a real face opening) so the side walls sit at the
      // neckline instead of jutting past the shoulders as a collar fin; the
      // bag widens to full size as it descends.
      const rad = Math.sqrt(Math.max(0, 1 - Math.pow(t, 2.2)));
      const widen = 0.74 + 0.26 * smoothstep(Math.min(1, t / 0.4));
      const rx = rxBase * rad * widen;
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
        const sinA = Math.sin(a);
        const rowFade = 1 - smoothstep(Math.min(1, t / 0.32));
        // Weld the WHOLE front half of the upper rings onto the neckline arc
        // (mapped 1:1, left rim end -> left shoulder ... right rim end ->
        // right shoulder). With the front rim ON the seam there is no apron
        // shelf standing in front of the mouth — no "front collar".
        const f = Math.min(1, Math.max(0, (a - Math.PI) / Math.PI));
        // Interpolate along the arc (snapping to the nearest point serrates
        // the welded rim).
        const fIdx = f * (N - 1);
        const i0 = Math.floor(fIdx);
        const i1 = Math.min(N - 1, i0 + 1);
        const it = fIdx - i0;
        const anchor = {
          x: arc[i0].x + (arc[i1].x - arc[i0].x) * it,
          y: arc[i0].y + (arc[i1].y - arc[i0].y) * it,
          z: arc[i0].z + (arc[i1].z - arc[i0].z) * it,
        };
        if (sinA < 0 && rowFade > 0) {
          px += (anchor.x - px) * rowFade;
          py += (anchor.y - py) * rowFade;
          pz += (anchor.z - pz) * rowFade;
        } else if (sinA >= 0) {
          // Gather the narrow side bands of the back half onto the SHOULDER
          // ends so the hood's left/right edges connect to the shoulder line.
          // Anchored per side (right band -> right shoulder, left -> left);
          // anchoring across the centre is what tore the tube before.
          const sideW =
            (1 - smoothstep(Math.min(1, Math.abs(sinA) / 0.3))) * rowFade;
          if (sideW > 0) {
            const sAnchor =
              a < Math.PI / 2 || a > 1.5 * Math.PI ? arc[N - 1] : arc[0];
            px += (sAnchor.x - px) * sideW;
            py += (sAnchor.y - py) * sideW;
            pz += (sAnchor.z - pz) * sideW;
          }
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

    // The crossed "dropped-hood" front bands that used to drape over the chest
    // in an X below the neckline were removed: they read as an unwanted overlap
    // flap at the front of the neck. The hood tube above stands on its own.
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
  const sleeveNormalRows: number[][][] = [];

  /**
   * A sleeve that hangs from the shoulder. Its upper centreline eases from an
   * outward shoulder direction into the final hanging axis; the rest remains
   * straight. This avoids concentrating the entire shoulder-to-arm direction
   * change into a single ring.
   */
  const buildHangingSleeve = (
    side: 1 | -1,
    loop: THREE.Vector3[],
    centroid: THREE.Vector3,
    profile: SleeveProfile,
    seamTangents: THREE.Vector3[],
  ) => {
    const loopCount = loop.length;
    // Woven stripes are cut on the garment grain, so they stay vertical in
    // the front view even while the shoulder surface turns from the neckline
    // into the hanging sleeve. Ring-local UVs made the stripes follow that
    // turn and fan outward like a raglan print. Project the repeating fabric
    // swatch in world X/Y instead; one tile covers roughly 13.5 cm, matching
    // the stripe density in the extracted front panel.
    const wovenTile = 13.5 * SCALE;
    const sleeveUv = (p: THREE.Vector3, u: number, v: number) =>
      features.fabric === "woven"
        ? ([
            0.5 + p.x / wovenTile,
            clamp(0.5 + (0.7 * p.y) / length, 0.05, 0.95),
          ] as const)
        : ([u, v] as const);
    // One constant hanging direction: down, angled slightly outward.
    const hang = slopeRad + (profile.endOffsetDeg * Math.PI) / 180;
    const axis = new THREE.Vector3(
      side * Math.cos(hang),
      -Math.sin(hang),
      0,
    ).normalize();
    const e2 = new THREE.Vector3(0, 0, 1);
    const e1 = new THREE.Vector3().crossVectors(e2, axis).normalize();

    // Size the tube to the armhole's area, so it covers the opening without
    // the ballooning a mean-radius circle would give on a tall narrow oval.
    const offsets = loop.map((p) => p.clone().sub(centroid));
    let twice = 0;
    for (let j = 0; j < loopCount; j += 1) {
      const a = offsets[j];
      const b = offsets[(j + 1) % loopCount];
      twice += a.dot(e1) * b.dot(e2) - b.dot(e1) * a.dot(e2);
    }
    // Size the tube from the arm, not from the armhole opening. The opening is
    // a narrow slit between the front and back panels, so its area yields a
    // sleeve roughly a fifth of a real one's thickness; measured against the
    // render, sleeve-to-torso width came out at 0.05 where a hoodie sits near
    // 0.3. The armhole area is kept only as a floor, so a very roomy armhole
    // still gets a sleeve wide enough to cover it.
    const areaRadius = Math.sqrt(Math.abs(twice) / 2 / Math.PI);
    // A raglan opening includes the long seam from neckline to underarm, so
    // its projected area is much larger than an arm cross-section. Using it as
    // the sleeve radius creates a broad slab in side view. Long hanging
    // sleeves are sized from the body/arm proportion; the measured area is
    // retained only for cap sleeves whose opening is a conventional armhole.
    const rEff = profile.hangFromShoulder
      ? halfW * profile.bicepRadiusFactor
      : Math.max(areaRadius, halfW * profile.bicepRadiusFactor);
    if (!(rEff > 1e-6)) {
      return;
    }

    // Place the tube beside the torso, not through it.
    //
    // Set the lateral position explicitly rather than offsetting along e1: at a
    // near-vertical hang the axis has almost no sideways component, so e1
    // collapses to +x for BOTH sleeves and offsetting along it pushes the left
    // sleeve inward, burying it in the body. Seat the tube's inner wall on the
    // torso's side seam instead, which is where a sleeve hangs when the arm is
    // down, and lift it slightly so it starts up at the shoulder.
    // Start at the SHOULDER, not part-way down the armhole.
    //
    // Taking the armhole's centroid put the tube's first ring halfway down the
    // opening, so the sleeve began below the shoulder line and read as a slab
    // hung off the side of the body with the shoulder missing above it. The
    // sleeve runs from the shoulder seam all the way to the cuff, so begin it
    // at the TOP of the armhole — the shoulder end — and let the seam close
    // the opening beneath it.
    let shoulderTopY = -Infinity;
    for (const p of loop) {
      shoulderTopY = Math.max(shoulderTopY, p.y);
    }
    const start = centroid.clone();
    start.x = side * (halfW + rEff * 0.35);
    // A round upper arm reaches the shoulder at its TOP, not at its centre.
    // Lowering the first-ring centre by one radius removes the horizontal
    // cylinder top that made the shoulder look square.
    start.y = shoulderTopY - rEff * 1.42;
    let signed = 0;
    for (let j = 0; j < loopCount; j += 1) {
      const a = offsets[j];
      const b = offsets[(j + 1) % loopCount];
      signed += a.dot(e1) * b.dot(e2) - b.dot(e1) * a.dot(e2);
    }
    // e1/e2 form the opposite handedness to the armhole loop's projected
    // basis here. Reverse the angular walk so the front half of the raglan
    // seam maps to the front half of the sleeve instead of twisting to back.
    const dir = signed >= 0 ? -1 : 1;
    // Match the top of the circular tube to the top of the raglan loop. The
    // former phase came from the first (underarm) point, which put the tube's
    // top near the end of the loop while the shoulder top sits halfway around
    // it. Lofting between those phases twisted the cap and removed a large
    // wedge from the visible shoulder.
    const topIndices: number[] = [];
    for (let j = 0; j < loopCount; j += 1) {
      if (Math.abs(loop[j].y - shoulderTopY) < 1e-5) {
        topIndices.push(j);
      }
    }
    const topPhaseIndex =
      topIndices.reduce((sum, j) => sum + j, 0) /
      Math.max(topIndices.length, 1);
    const topTheta = side > 0 ? 0 : Math.PI;
    const a0 =
      topTheta - dir * ((2 * Math.PI * topPhaseIndex) / loopCount);

    const taper = clamp(features.sleeveTaper, 0.35, 1);
    // A hanging sleeve is oval rather than a round cylinder in side view.
    // Preserve its front-view arm width while reducing only front-to-back
    // depth to match the flatter shoulder and sleeve profile of real fleece.
    const sleeveDepthScale = profile.hangFromShoulder ? 0.72 : 1;
    const bell = (t: number, at: number, w: number) =>
      Math.exp(-Math.pow((t - at) / w, 2));
    const girth = (t: number) => 1 + (profile.wristTaper - 1) * clamp(t, 0, 1);
    const foldAt = (t: number, theta: number) => {
      const slack = 0.35 + 0.65 * smoothstep(clamp((t - 0.2) / 0.5, 0, 1));
      const drape = profile.drape * slack * Math.sin(3 * theta + 1.7 * t);
      const tension =
        profile.tension * slack * Math.sin(5 * theta - 3.1 * t + 0.9);
      const stack = profile.stack * bell(t, 0.9, 0.09) * Math.sin(4 * theta);
      return drape + tension + stack;
    };

    // Ease the first part of the arm outward before settling into the hanging
    // axis. The previous straight centreline began nearly vertical at t=0,
    // leaving a flat cap shelf followed by a hard corner. This C1 shoulder
    // sweep distributes that turn across several rings and then reaches zero
    // extra curvature, so the lower sleeve remains straight.
    const shoulderBendEnd = 0.28;
    const shoulderSweep = rEff * 0.3;
    const outward = new THREE.Vector3(side, 0, 0);
    const sleeveCenterAt = (t: number) => {
      const s = clamp(t / shoulderBendEnd, 0, 1);
      const sweep = shoulderSweep * (2 * s - s * s);
      return start
        .clone()
        .addScaledVector(axis, sleeveLen * t)
        .addScaledVector(outward, sweep);
    };
    const sleeveAxisAt = (t: number) => {
      const s = clamp(t / shoulderBendEnd, 0, 1);
      const sweepRate =
        t < shoulderBendEnd
          ? (shoulderSweep * 2 * (1 - s)) / shoulderBendEnd
          : 0;
      return axis
        .clone()
        .multiplyScalar(sleeveLen)
        .addScaledVector(outward, sweepRate)
        .normalize();
    };

    const sleevePointAt = (
      t: number,
      scale: number,
      withFolds: boolean,
      j: number,
    ) => {
      const center = sleeveCenterAt(t);
      const localAxis = sleeveAxisAt(t);
      const localE1 = new THREE.Vector3()
        .crossVectors(e2, localAxis)
        .normalize();
      const theta = a0 + dir * ((2 * Math.PI * j) / loopCount);
      const r = rEff * scale * (withFolds ? 1 + foldAt(t, theta) : 1);
      const p = center
        .clone()
        .addScaledVector(localE1, Math.cos(theta) * r)
        .addScaledVector(e2, Math.sin(theta) * r * sleeveDepthScale);
      // The cap itself now carries the transition from the torso-depth raglan
      // loop to this arm-sized cross-section. Keeping loop[j].z through the
      // upper 22% of the sleeve made the whole shoulder as deep as the torso,
      // which produced the bulky side profile and intersecting cap faces.
      return p;
    };

    const ringAt = (t: number, scale: number, withFolds: boolean) => {
      const ring: number[] = [];
      for (let j = 0; j < loopCount; j += 1) {
        const p = sleevePointAt(t, scale, withFolds, j);
        const [u, v] = sleeveUv(p, j / loopCount, t);
        ring.push(pushVertex(p.x, p.y, p.z, u, v));
      }
      return ring;
    };

    const stitch = (ringA: number[], ringB: number[]) => {
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

    // Shoulder seam: weld the torso's armhole edge into a short curved cap,
    // then enter the hanging tube with the same tangent as its axis. A direct
    // one-strip stitch made the raglan panel meet the round tube as a sharp
    // folded triangle even though their normals were averaged.
    const seam = loop.map((p, j) => {
      const [u, v] = sleeveUv(p, j / loopCount, 0);
      return pushVertex(p.x, p.y, p.z, u, v);
    });
    const normalRows: number[][] = [seam];
    // First continue the torso's final grid step exactly. This strip is
    // coplanar with the panel at the boundary, so the attachment cannot fold
    // inward before the curved cap begins.
    const leadPositions = loop.map((p, j) => {
      const leadPoint = p.clone().addScaledVector(seamTangents[j], 1.05);
      // Continue the panel mainly across X/Y. Its curved side-edge tangent can
      // point strongly forward/backward; carrying all of that Z component
      // into the sleeve makes the cap briefly deeper than the torso seam.
      leadPoint.z = p.z;
      return leadPoint;
    });
    const lead = leadPositions.map((p, j) => {
      const [u, v] = sleeveUv(p, j / loopCount, 0.005);
      return pushVertex(p.x, p.y, p.z, u, v);
    });
    stitch(seam, lead);
    normalRows.push(lead);
    let previous = lead;
    // Dense sampling is important at the seam itself: with only a few rings,
    // the first chord cuts across the Hermite curve and its face normal can be
    // almost perpendicular to the torso normal even though the mathematical
    // tangent is continuous.
    const capRings = 12;
    for (let k = 1; k <= capRings; k += 1) {
      const u = k / capRings;
      const u2 = u * u;
      const u3 = u2 * u;
      const h00 = 2 * u3 - 3 * u2 + 1;
      const h10 = u3 - 2 * u2 + u;
      const h01 = -2 * u3 + 3 * u2;
      const h11 = u3 - u2;
      const ring: number[] = [];
      for (let j = 0; j < loopCount; j += 1) {
        const p0 = leadPositions[j];
        const p1 = sleevePointAt(0, 1, false, j);
        // Bound both Hermite handles to the actual seam-to-tube span. The old
        // radius-sized handles could cross past that span, producing a bump at
        // the raglan seam followed by a flat shelf and an almost 90-degree
        // drop into the sleeve. Short, span-relative handles keep the cap
        // monotone while still matching the torso and sleeve directions.
        const span = p0.distanceTo(p1);
        const seamTangent = seamTangents[j]
          .clone()
          .normalize()
          .multiplyScalar(Math.min(rEff * 0.62, span * 0.32));
        const tubeTangent = sleeveAxisAt(0)
          .clone()
          .multiplyScalar(Math.min(rEff * 0.72, span * 0.34));
        const p = p0
          .clone()
          .multiplyScalar(h00)
          .addScaledVector(seamTangent, h10)
          .addScaledVector(p1, h01)
          .addScaledVector(tubeTangent, h11);
        // Keep front-to-back depth monotone. Independent Hermite Z handles
        // previously collapsed the middle of the cap almost flat and then
        // expanded it again at the arm, creating a bulbous circular shoulder
        // in side view. X/Y retain the tangent-matched Hermite shoulder curve.
        const depthU = clamp(u / 0.3, 0, 1);
        p.z = p0.z + (p1.z - p0.z) * smoothstep(depthU);
        const [texU, texV] = sleeveUv(p, j / loopCount, u * 0.08);
        ring.push(pushVertex(p.x, p.y, p.z, texU, texV));
      }
      stitch(previous, ring);
      previous = ring;
      normalRows.push(ring);
    }

    for (let i = 1; i <= SLEEVE_RINGS; i += 1) {
      const t = i / SLEEVE_RINGS;
      const ring = ringAt(t, girth(t), true);
      stitch(previous, ring);
      previous = ring;
      if (i <= 4) {
        normalRows.push(ring);
      }
    }
    sleeveNormalRows.push(normalRows);

    // Rib cuff: cut narrower than the sleeve so the tube gathers into it, and
    // rolled around its lower edge because the band is folded double.
    if (features.cuff === "ribbed") {
      const cuffLen = 7 * SCALE * trimScale;
      const sleeveEndScale = girth(1);
      const cuffScale = 0.84 * sleeveEndScale;
      const at = (d: number, scale: number) => {
        const center = sleeveCenterAt(1).addScaledVector(axis, d);
        const ring: number[] = [];
        for (let j = 0; j < loopCount; j += 1) {
          const theta = a0 + dir * ((2 * Math.PI * j) / loopCount);
          const p = center
            .clone()
            .addScaledVector(e1, Math.cos(theta) * rEff * scale)
            .addScaledVector(
              e2,
              Math.sin(theta) * rEff * scale * sleeveDepthScale,
            );
          ring.push(pushVertex(p.x, p.y, p.z, j / loopCount, 1));
        }
        return ring;
      };
      // Keep the cuff on the sleeve's curved centreline and remove the small
      // terminal folds before narrowing the rib. This prevents both the
      // lateral offset and the abrupt radius step at the join.
      const cuffJoin = at(cuffLen * 0.04, sleeveEndScale);
      stitch(previous, cuffJoin);
      const band1 = at(cuffLen * 0.22, cuffScale);
      stitch(cuffJoin, band1);
      const band2 = at(cuffLen, cuffScale);
      stitch(band1, band2);
      if (profile.foldedRibCuff) {
        stitch(
          band2,
          at(cuffLen + 0.25 * SCALE * trimScale, cuffScale * 0.96),
        );
      }
    }

    // Barrel cuff (woven button shirts): the sleeve gathers into a crisp,
    // slightly snugger straight band at the wrist, finished with two buttons
    // on the band and a gauntlet button up the sleeve placket.
    //
    // buildSleeve has carried this since the button-shirt presets landed, but
    // it only builds cap sleeves — every LONG sleeve is built here, and this
    // function had a ribbed branch and nothing else. So a long-sleeve woven
    // shirt came out with a raw sleeve end and no buttons anywhere on it.
    if (features.cuff === "barrel") {
      const cuffLen = 6 * SCALE;
      const sleeveEndScale = girth(1);
      // A real barrel cuff is only slightly snugger than the sleeve it gathers
      // out of; a drastic step reads as two mismatched tubes.
      const cuffScale = 0.92 * sleeveEndScale;
      const ringAtDistance = (d: number, scale: number) => {
        const center = sleeveCenterAt(1).addScaledVector(axis, d);
        const ring: number[] = [];
        for (let j = 0; j < loopCount; j += 1) {
          const theta = a0 + dir * ((2 * Math.PI * j) / loopCount);
          const p = center
            .clone()
            .addScaledVector(e1, Math.cos(theta) * rEff * scale)
            .addScaledVector(
              e2,
              Math.sin(theta) * rEff * scale * sleeveDepthScale,
            );
          ring.push(pushVertex(p.x, p.y, p.z, j / loopCount, 1));
        }
        return ring;
      };
      const gather = ringAtDistance(cuffLen * 0.06, sleeveEndScale);
      stitch(previous, gather);
      const band1 = ringAtDistance(cuffLen * 0.22, cuffScale);
      stitch(gather, band1);
      const band2 = ringAtDistance(cuffLen, cuffScale);
      stitch(band1, band2);

      // Buttons: shallow round discs standing just proud of the cuff's front
      // face. A shirt button is round, and at the zoom someone inspects a cuff
      // at, the flat square quad used elsewhere reads as a square. The rim is
      // dropped slightly behind the centre so the disc catches a gradient and
      // reads as a domed button rather than a flat sticker. Double-sided, so
      // it survives being seen from behind on a thin sleeve.
      const br = 0.62 * SCALE; // ~1.2cm across, a shirt button
      const BUTTON_SEGMENTS = 12;
      // A cuff fastens on the OUTER side of the wrist, not across the front of
      // it — the placket runs up the little-finger side of the forearm and the
      // buttons sit on that edge, which is where they appear on a flat lay.
      // Sitting them at the tube's front-most point put them in the middle of
      // the cuff facing the viewer, which is not where a shirt buttons.
      //
      // e1 points along +x for BOTH sleeves (it is built from the hang axis and
      // world Z), so multiplying by `side` is what turns it into "away from the
      // body". Tilt a little toward the front from there, or the disc is edge
      // on to the camera and disappears.
      // 0 puts the button exactly on the silhouette edge, seen edge on, where
      // it reads as a nick in the outline rather than a button; pi/2 puts it
      // back in the middle of the cuff, which is not how a shirt fastens.
      // Measured on the built mesh at 0.7 the disc's outer rim landed 0.6cm
      // from an edge 11.7cm away — still grazing it. 0.9 sits the button about
      // 60% of the way out: plainly off to the side, and turned far enough
      // toward the viewer to catch light and read as round.
      const buttonTilt = 0.9; // radians toward the front, ~52 degrees
      const buttonTheta = side > 0 ? buttonTilt : Math.PI - buttonTilt;
      const cosT = Math.cos(buttonTheta);
      const sinT = Math.sin(buttonTheta);
      // Outward normal of the elliptical cross-section (semi-axes r and
      // r*sleeveDepthScale) at that angle: (b·cos, a·sin), which reduces to
      // this once the common r drops out.
      const outward = new THREE.Vector3()
        .addScaledVector(e1, sleeveDepthScale * cosT)
        .addScaledVector(e2, sinT)
        .normalize();
      // Around the cuff, perpendicular to both the sleeve and that normal.
      const around = new THREE.Vector3()
        .crossVectors(outward, axis)
        .normalize();
      const mkButton = (dist: number, scale: number) => {
        const r = rEff * scale;
        const centre = sleeveCenterAt(1)
          .addScaledVector(axis, dist)
          .addScaledVector(e1, cosT * r)
          .addScaledVector(e2, sinT * r * sleeveDepthScale)
          .addScaledVector(outward, 0.32 * SCALE);
        // One UV for the whole disc, so the fabric's stripes cannot run across
        // the button face.
        const hub = pushVertex(centre.x, centre.y, centre.z, 0.5, 0.5);
        const rim: number[] = [];
        for (let k = 0; k < BUTTON_SEGMENTS; k += 1) {
          const phi = (2 * Math.PI * k) / BUTTON_SEGMENTS;
          const p = centre
            .clone()
            .addScaledVector(around, Math.cos(phi) * br)
            .addScaledVector(axis, Math.sin(phi) * br)
            .addScaledVector(outward, -0.08 * SCALE);
          rim.push(pushVertex(p.x, p.y, p.z, 0.5, 0.5));
        }
        for (let k = 0; k < BUTTON_SEGMENTS; k += 1) {
          const a3 = rim[k];
          const b3 = rim[(k + 1) % BUTTON_SEGMENTS];
          indices.push(hub, a3, b3);
          indices.push(hub, b3, a3);
        }
      };
      mkButton(cuffLen * 0.42, cuffScale); // cuff button
      mkButton(cuffLen * 0.82, cuffScale); // second cuff button
      mkButton(-3.5 * SCALE, sleeveEndScale); // gauntlet, up the placket
    }
  };

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
    const seamTangents: THREE.Vector3[] = [];
    const innerCol = sideCol - side;
    const addSeamPoint = (base: number, row: number) => {
      const boundary = readVertex(base + row * panelStride + sideCol);
      const interior = readVertex(base + row * panelStride + innerCol);
      const tangent = boundary.clone().sub(interior);
      if (tangent.lengthSq() < 1e-10) {
        tangent.set(side * SCALE, 0, 0);
      }
      loop.push(boundary);
      seamTangents.push(tangent);
    };
    for (const r of edgeRows) {
      addSeamPoint(frontBase, r);
    }
    for (let i = edgeRows.length - 1; i >= 0; i -= 1) {
      addSeamPoint(backBase, edgeRows[i]);
    }

    const loopCount = loop.length;
    const centroid = loop
      .reduce((acc, p) => acc.clone().add(p), new THREE.Vector3())
      .multiplyScalar(1 / loopCount);

    // Sleeve hang, taken from this sleeve's kind rather than blended from its
    // length. Two findings are baked into those profiles and should not be
    // re-litigated by tweaking angles here:
    //
    //  - The ROOT angle is what pinches the upper arm, not the curvature. Root
    //    54 and 70 deg both pinched, and the 70 deg version had the lowest
    //    curvature of any attempt; root 34 deg never pinched. The armhole ring
    //    faces sideways, so a steep root extrudes the tube within its own plane
    //    and collapses it.
    //  - Closeness is therefore bought at the far END, by carrying a long
    //    sleeve past vertical so it swings back toward the body.
    //
    // The turn is spread linearly between the two, giving constant curvature
    // and so no local kink anywhere along the sleeve to read as an elbow.
    const sleeveKind: SleeveKind =
      params.sleeveLength >= LONG_SLEEVE_CM ? "long" : "cap";
    const sleeveProfile = SLEEVE_PROFILES[sleeveKind];

    // --- Torso + sleeve construction (long sleeves) -------------------------
    //
    // The torso is built as a tank: body, shoulders, and an armhole opening,
    // nothing more. The sleeve is then a separate tube hung from that opening.
    //
    // The older construction extruded the sleeve OUT of the armhole — the tube
    // left sideways, following the armhole's own plane, and had to turn roughly
    // ninety degrees to get pointing down. That turn is the source of most of
    // the sleeve trouble: a crease wherever the bend concentrated, and a
    // collapsed tube whenever the bend was moved up to the shoulder to avoid
    // creasing lower down.
    //
    // Here the tube never turns. Its cross-section is built perpendicular to
    // the hanging direction from the very first ring, matched to the armhole's
    // area so it is the same thickness as the opening it covers, and the
    // armhole edge is stitched onto it as the shoulder seam. So the geometry
    // is torso + sleeve, joined at a seam, rather than torso + armhole +
    // sleeve with a hinge in the middle.
    if (sleeveProfile.hangFromShoulder) {
      buildHangingSleeve(side, loop, centroid, sleeveProfile, seamTangents);
      return;
    }

    const droopStart =
      slopeRad + (sleeveProfile.rootOffsetDeg * Math.PI) / 180;
    const droopEnd = slopeRad + (sleeveProfile.endOffsetDeg * Math.PI) / 180;
    const droopAt = (t: number) =>
      droopStart + (droopEnd - droopStart) * clamp(t, 0, 1);
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
    // Radius the rounded ring settles on, matched to the armhole's AREA rather
    // than its mean radius.
    //
    // The armhole is a tall, narrow oval. A circle built from the mean radius
    // of such an oval encloses noticeably more area than the oval itself, so
    // the tube swells as it rounds off — the sleeve came out slim at the
    // shoulder and fat below the elbow, the opposite of the reference. Taking
    // the radius of the equal-area circle keeps the sleeve the same thickness
    // as the armhole it grows out of, so the taper alone decides its shape.
    let loopArea = 0;
    for (let j = 0; j < loopCount; j += 1) {
      const a = rootOffsets[j];
      const b = rootOffsets[(j + 1) % loopCount];
      loopArea += a.dot(e1) * b.dot(e2) - b.dot(e1) * a.dot(e2);
    }
    const avgRootRadius = Math.sqrt(Math.abs(loopArea) / 2 / Math.PI);
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

    // --- Sleeve anatomy ----------------------------------------------------
    // Follows the pattern pieces on the construction sheet: a domed cap that
    // eases into the armhole, then a gentle taper from the cap down to the hem
    // allowance. No bicep or elbow bulge — the pattern's side seams are close
    // to straight, and the sleeve reads as one even tube apart from that taper.
    const bell = (t: number, at: number, width: number) =>
      Math.exp(-Math.pow((t - at) / width, 2));
    const armGirth = (t: number) => {
      // LINEAR from armhole to cuff. Easing the taper in bows the outline —
      // it stays wide through the upper arm and then falls away — whereas the
      // reference's sleeve edge is a straight line converging on the cuff.
      const toWrist = 1 + (sleeveProfile.wristTaper - 1) * clamp(t, 0, 1);
      return toWrist * (1 + sleeveProfile.capEase * bell(t, 0.1, 0.16));
    };
    // Normalised so the root is exactly 1: the first ring is welded to the
    // armhole loop, and any swell there opens a gap along the armhole seam
    // instead of shaping the sleeve.
    const armRoot = armGirth(0);
    const armRadius = (t: number) => armGirth(t) / armRoot;

    // Folds from gravity and fabric tension: shallow creases running down the
    // sleeve, faint where the cap is stretched over the shoulder and deeper
    // through the slack lower arm, plus fabric stacking just above the cuff
    // where the sleeve is stopped by the band. The modulation is zero-mean
    // around the ring, so girth — and therefore the cuff junction and the
    // texture's scale — are unchanged; it only ripples the surface.
    const foldAt = (t: number, theta: number) => {
      const slack = 0.35 + 0.65 * smoothstep(clamp((t - 0.2) / 0.5, 0, 1));
      const drape =
        sleeveProfile.drape * slack * Math.sin(3 * theta + 1.7 * t);
      const tension =
        sleeveProfile.tension * slack * Math.sin(5 * theta - 3.1 * t + 0.9);
      const stack =
        sleeveProfile.stack * bell(t, 0.9, 0.09) * Math.sin(4 * theta + 0.6);
      return drape + tension + stack;
    };

    // Rings are built in one frame taken from the sleeve's MID direction, so a
    // sleeve that turns along its length ends up with rings that are no longer
    // square to it — at the cuff, where the local direction differs from the
    // mid by half the total turn, the opening reads as cut on a slant. Rotate
    // each ring about Z by the difference between its local direction and the
    // mid, which puts every cross-section perpendicular to the tube again.
    //
    // Weighted by `soften`, which is 0 at the root: the first ring must stay
    // exactly on the armhole loop it is welded to, and the correction fades in
    // as the ring becomes a free circle further down the sleeve.
    const emitRing = (
      center: THREE.Vector3,
      scale: number,
      soften: number,
      v: number,
      droop: number,
      folds = 1,
    ) => {
      const theta = -side * (droop - droopMid) * soften;
      const cs = Math.cos(theta);
      const sn = Math.sin(theta);
      const ring: number[] = [];
      for (let j = 0; j < loopCount; j += 1) {
        // Folds fade in with `soften`, so the root ring stays exactly on the
        // armhole loop it is welded to.
        const around = (2 * Math.PI * j) / loopCount;
        const ripple = 1 + folds * soften * foldAt(v, around);
        const offset = rootOffsets[j]
          .clone()
          .lerp(softOffsets[j], soften)
          .multiplyScalar(scale * ripple);
        const p = new THREE.Vector3(
          center.x + offset.x * cs - offset.y * sn,
          center.y + offset.x * sn + offset.y * cs,
          center.z + offset.z,
        );
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
      const ring = emitRing(
        ringCenter.clone(),
        armRadius(t),
        soften,
        t,
        droopAt(t),
      );
      if (previousRing) {
        stitchRings(previousRing, ring);
      }
      previousRing = ring;
    }

    // Cuffs continue along the sleeve's END tangent — the sleeve droops as it
    // descends, so extending along the average axis would kink the cuff off
    // at an angle from the wrist.
    const dEnd = droopAt(1);
    const endAxis = new THREE.Vector3(
      side * Math.cos(dEnd),
      -Math.sin(dEnd),
      0,
    );

    // Optional ribbed cuff: a short, snugger band past the sleeve end.
    if (features.cuff === "ribbed" && previousRing) {
      const cuffLen = 3 * SCALE * trimScale;
      // The cuff is cut narrower than the sleeve and stretched to fit, so the
      // sleeve gathers into it. sleeveTaper says how snug this garment's is.
      const cuffScale = (0.62 + 0.28 * taper) * armRadius(1);
      const endCenter = ringCenter.clone();
      const rib1 = emitRing(
        endCenter.clone().addScaledVector(endAxis, cuffLen * 0.15),
        cuffScale,
        1,
        1,
        dEnd,
        0,
      );
      stitchRings(previousRing, rib1);
      const rib2 = emitRing(
        endCenter.clone().addScaledVector(endAxis, cuffLen),
        cuffScale,
        1,
        1,
        dEnd,
        0,
      );
      stitchRings(rib1, rib2);
      if (sleeveProfile.foldedRibCuff) {
        // Rib knit folded in half lengthwise: the band's lower edge is a fold,
        // not a raw opening. Roll the surface tightly around that edge and
        // stop — an extra return ring up the inside stacked visible concentric
        // steps at the wrist, which read as a telescoping tube rather than a
        // soft doubled cuff.
        const roll = emitRing(
          endCenter
            .clone()
            .addScaledVector(endAxis, cuffLen + 0.25 * SCALE * trimScale),
          cuffScale * 0.9,
          1,
          1,
          dEnd,
          0,
        );
        stitchRings(rib2, roll);
      }
    }

    // Barrel cuff (woven button shirts): the sleeve gathers into a crisp,
    // slightly snugger straight band at the wrist, finished with buttons —
    // two on the band and one gauntlet button up the sleeve placket.
    if (features.cuff === "barrel" && previousRing) {
      const cuffLen = 6 * SCALE;
      // Nearly the sleeve-end width: a real cuff is only slightly snugger
      // than the sleeve it gathers out of — a drastic step reads as two
      // mismatched tubes.
      // Relative to the full sleeve width (the sleeve itself is straight).
      const cuffScale = Math.min(0.96, (0.62 + 0.28 * taper) * 1.12);
      const endCenter = ringCenter.clone();
      // Sharp gather where the blousy sleeve pleats into a clean, smooth
      // band — no ridge line across the cuff.
      const gather = emitRing(
        endCenter.clone().addScaledVector(endAxis, 0.4 * SCALE),
        cuffScale,
        1,
        1,
        dEnd,
        0,
      );
      stitchRings(previousRing, gather);
      // Straight crisp band to the wrist edge.
      const band = emitRing(
        endCenter.clone().addScaledVector(endAxis, cuffLen),
        cuffScale * 0.98,
        1,
        1,
        dEnd,
        0,
      );
      stitchRings(gather, band);
      // Buttons sit on the front face of the cuff: anchor at the band's
      // front-most vertex, spaced along the sleeve axis.
      let anchor = gather[0];
      let bestZ = -Infinity;
      for (const vi of gather) {
        const z = positions[vi * 3 + 2];
        if (z > bestZ) {
          bestZ = z;
          anchor = vi;
        }
      }
      const ax2 = positions[anchor * 3];
      const ay2 = positions[anchor * 3 + 1];
      const az2 = positions[anchor * 3 + 2];
      const along = endAxis.clone().normalize();
      const perp = new THREE.Vector3()
        .crossVectors(along, new THREE.Vector3(0, 0, 1))
        .normalize();
      const bh = 0.8 * SCALE;
      const mkButton = (dist: number) => {
        const cx2 = ax2 + along.x * dist;
        const cy2 = ay2 + along.y * dist;
        const cz2 = az2 + 0.3 * SCALE;
        const a2 = pushVertex(
          cx2 - perp.x * bh - along.x * bh,
          cy2 - perp.y * bh - along.y * bh,
          cz2,
          0,
          0,
        );
        const b2 = pushVertex(
          cx2 + perp.x * bh - along.x * bh,
          cy2 + perp.y * bh - along.y * bh,
          cz2,
          0.02,
          0,
        );
        const c2 = pushVertex(
          cx2 + perp.x * bh + along.x * bh,
          cy2 + perp.y * bh + along.y * bh,
          cz2,
          0.02,
          0.02,
        );
        const d2 = pushVertex(
          cx2 - perp.x * bh + along.x * bh,
          cy2 - perp.y * bh + along.y * bh,
          cz2,
          0,
          0.02,
        );
        indices.push(a2, b2, c2, a2, c2, d2);
        indices.push(a2, c2, b2, a2, d2, c2);
      };
      mkButton(1.6 * SCALE); // cuff button
      mkButton(3.9 * SCALE); // second cuff button
      mkButton(-3.5 * SCALE); // gauntlet button up the sleeve placket
    }
  };

  // --- Block: turtleneck ------------------------------------------------------
  // A tall knit tube rising from the neck opening with a fold-over hint.
  const buildTurtleneck = () => {
    const loopCount = neckLoop.length;
    if (loopCount < 4) {
      return;
    }
    // Loop centre and extents, to round the tube toward an ellipse — the raw
    // neck opening is squarish, and a turtleneck must read as a smooth round
    // funnel, not a boxy chimney.
    let ccx = 0;
    let ccz = 0;
    for (const q of neckLoop) {
      ccx += q.x;
      ccz += q.z;
    }
    ccx /= loopCount;
    ccz /= loopCount;
    let ex = 0;
    let ez = 0;
    for (const q of neckLoop) {
      ex = Math.max(ex, Math.abs(q.x - ccx));
      ez = Math.max(ez, Math.abs(q.z - ccz));
    }
    // The neckline slopes up toward the shoulders; a roll neck's top edge is
    // LEVEL, so the upper rings blend to a flat height (the neckline's high
    // point) instead of inheriting the slope as raised side "ears".
    let ccy = -Infinity;
    for (const q of neckLoop) {
      ccy = Math.max(ccy, q.y);
    }
    const rings: number[][] = [];
    // [rise, scale, roundness, level]: welded to the real neckline at the
    // rim, fully elliptical and level by mid-height. The wall eases in as it
    // rises, then the top FOLDS OVER — the crest rounds outward and the last
    // ring comes back DOWN outside the wall, the doubled-over roll of a real
    // roll neck instead of a stiff open-ended cylinder.
    // A roll neck is a tall knit tube FOLDED IN HALF: the inner wall rises to
    // the crest, then the fabric folds over and comes back DOWN the outside,
    // so a doubled band sits folded around the neck (visible fold at the top,
    // free hem hanging down the outside).
    const stages: Array<[number, number, number, number]> = [
      [0, 1, 0, 0], // rim, welded to the neckline seam
      [1.5, 0.97, 0.85, 0.55], // eases round and part-way level
      [4, 0.95, 1, 1], // fully round and level — inner wall
      [6.5, 0.95, 1, 1], // upper inner wall
      [7.6, 1.0, 1, 1], // rounded crest where it folds over
      [5, 1.07, 1, 1], // outer layer folds back down outside the wall
      [2.4, 1.07, 1, 1], // free hem of the fold hangs down the neck
    ];
    // The loop only samples the front and back necklines — the SIDES of the
    // opening have no points, so lofting per loop column stretches one giant
    // quad across each side (a hard vertical crease). Resample the tube at
    // dense angles: the loop's own angles (exact weld at the rim) merged
    // with a uniform sweep (smooth walls everywhere).
    const angs: number[] = [];
    {
      let prev = -Infinity;
      for (let j = 0; j < loopCount; j += 1) {
        const q = neckLoop[j];
        let a2 = Math.atan2(q.x - ccx, q.z - ccz);
        if (j > 0 && a2 < prev - Math.PI) {
          a2 += 2 * Math.PI; // unwrap across the +/-PI cut
        }
        if (a2 < prev) {
          a2 = prev + 1e-3; // enforce monotonic ordering
        }
        angs.push(a2);
        prev = a2;
      }
    }
    const a0 = angs[0];
    const samples: number[] = [...angs];
    const UNIFORM = 64;
    for (let i = 0; i < UNIFORM; i += 1) {
      const t = a0 + (2 * Math.PI * i) / UNIFORM;
      // Skip samples that nearly coincide with a loop angle.
      if (angs.every((v) => Math.abs(v - t) > 0.02)) {
        samples.push(t);
      }
    }
    samples.sort((p, q2) => p - q2);
    // Rim position at any angle: piecewise-linear along the loop polyline.
    const rimAt = (theta: number) => {
      let j = 0;
      while (j < loopCount - 1 && angs[j + 1] < theta) {
        j += 1;
      }
      const jn = (j + 1) % loopCount;
      const aA = angs[j];
      const aB = j + 1 < loopCount ? angs[j + 1] : a0 + 2 * Math.PI;
      const f = aB > aA ? Math.min(1, Math.max(0, (theta - aA) / (aB - aA))) : 0;
      const A = neckLoop[j];
      const B = neckLoop[jn];
      return {
        x: A.x + (B.x - A.x) * f,
        y: A.y + (B.y - A.y) * f,
        z: A.z + (B.z - A.z) * f,
      };
    };
    const segs = samples.length;
    for (let k = 0; k < stages.length; k += 1) {
      const [rise, scale, round, level] = stages[k];
      const ring: number[] = [];
      for (let i = 0; i < segs; i += 1) {
        const theta = samples[i];
        const q = rimAt(theta);
        const rxp = ccx + Math.sin(theta) * ex;
        const rzp = ccz + Math.cos(theta) * ez;
        const bx = q.x + (rxp - q.x) * round;
        const bz = q.z + (rzp - q.z) * round;
        const by = q.y + (ccy - q.y) * level + rise * SCALE;
        ring.push(
          pushVertex(
            ccx + (bx - ccx) * scale,
            by,
            ccz + (bz - ccz) * scale,
            (theta - a0) / (2 * Math.PI),
            k / (stages.length - 1),
          ),
        );
      }
      rings.push(ring);
    }
    for (let k = 0; k < rings.length - 1; k += 1) {
      for (let i = 0; i < segs; i += 1) {
        const inx = (i + 1) % segs;
        indices.push(
          rings[k][i],
          rings[k][inx],
          rings[k + 1][i],
          rings[k][inx],
          rings[k + 1][inx],
          rings[k + 1][i],
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
    // Loop centre and extents, for rounding the collar toward an ellipse —
    // the raw neck opening is nearly rectangular, and scaling it directly
    // makes the folded collar read as a square box.
    let ccx = 0;
    let ccz = 0;
    for (const q of neckLoop) {
      ccx += q.x;
      ccz += q.z;
    }
    ccx /= loopCount;
    ccz /= loopCount;
    let ex = 0;
    let ez = 0;
    for (const q of neckLoop) {
      ex = Math.max(ex, Math.abs(q.x - ccx));
      ez = Math.max(ez, Math.abs(q.z - ccz));
    }
    // [rise, scale, roundness]: the stand keeps the neckline shape, the
    // fold-over is fully rounded so the collar drapes as a soft oval.
    const stages: Array<[number, number, number]> = [
      [0, 1, 0], // rim
      [standH, 0.97, 0.3], // stand
      [standH - foldDrop * 0.35, foldScale * 0.55 + 0.45, 0.75], // roll
      [standH - foldDrop, foldScale, 1], // fold-over edge
    ];
    const rings: number[][] = [];
    const thetas: number[] = [];
    for (let j = 0; j < loopCount; j += 1) {
      const q = neckLoop[j];
      // Angle from the front centre (+z), for the front opening.
      thetas.push(Math.atan2(q.x - ccx, q.z - ccz));
    }
    for (let k = 0; k < stages.length; k += 1) {
      const [rise, scale, round] = stages[k];
      const ring: number[] = [];
      for (let j = 0; j < loopCount; j += 1) {
        const q = neckLoop[j];
        const ang = thetas[j];
        // Blend the raw loop point toward its ellipse projection.
        const rxp = ccx + Math.sin(ang) * ex;
        const rzp = ccz + Math.cos(ang) * ez;
        const bx = q.x + (rxp - q.x) * round;
        const bz = q.z + (rzp - q.z) * round;
        ring.push(
          pushVertex(
            ccx + (bx - ccx) * scale,
            q.y + rise * SCALE,
            ccz + (bz - ccz) * scale,
            j / loopCount,
            k / (stages.length - 1),
          ),
        );
      }
      rings.push(ring);
    }
    // Stitch, leaving the collar OPEN at the front centre (a real shirt
    // collar's ends meet the placket with a notch — it is not a continuous
    // wall across the front).
    const frontGap = 0.38; // half-angle of the front opening (rad)
    for (let k = 0; k < rings.length - 1; k += 1) {
      for (let j = 0; j < loopCount; j += 1) {
        const jn = (j + 1) % loopCount;
        if (Math.abs(thetas[j]) < frontGap && Math.abs(thetas[jn]) < frontGap) {
          continue;
        }
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

    // Collar points: per classic shirt construction, the fold-over extends
    // past the stand's front ends into two triangular tips that angle down
    // toward the placket and lie on the chest.
    const frontRowEnd = Math.floor(loopCount / 2); // front row = first half
    let jL = -1;
    let jR = -1;
    for (let j = 0; j < frontRowEnd; j += 1) {
      if (thetas[j] <= -frontGap) {
        jL = j; // last column left of the gap
      }
    }
    for (let j = frontRowEnd - 1; j >= 0; j -= 1) {
      if (thetas[j] >= frontGap) {
        jR = j; // first column right of the gap
      }
    }
    const mkPoint = (jEnd: number, sx: number) => {
      if (jEnd < 0) {
        return;
      }
      const base = neckLoop[jEnd];
      const tipY = base.y - 3.4 * SCALE;
      const tip = {
        x: sx * 2.2 * SCALE,
        y: tipY,
        z: depth * 1.05 * depthTaper(tipY) + 0.35 * SCALE,
      };
      const EXT = 3;
      const colIdx: number[][] = [];
      for (let m = 0; m <= EXT; m += 1) {
        const w = m / EXT;
        const col: number[] = [];
        for (let k = 1; k < stages.length; k += 1) {
          if (m === 0) {
            col.push(rings[k][jEnd]);
          } else {
            const bi = rings[k][jEnd] * 3;
            col.push(
              pushVertex(
                positions[bi] + (tip.x - positions[bi]) * w,
                positions[bi + 1] + (tip.y - positions[bi + 1]) * w,
                positions[bi + 2] + (tip.z - positions[bi + 2]) * w,
                0.5 + sx * 0.5 * w,
                (k - 1) / 2,
              ),
            );
          }
        }
        colIdx.push(col);
      }
      for (let m = 0; m < EXT; m += 1) {
        for (let k = 0; k < 2; k += 1) {
          const a = colIdx[m][k];
          const b = colIdx[m + 1][k];
          const d = colIdx[m][k + 1];
          const e = colIdx[m + 1][k + 1];
          if (sx > 0) {
            indices.push(a, b, d, b, e, d);
          } else {
            indices.push(a, d, b, b, d, e);
          }
        }
      }
    };
    mkPoint(jL, -1);
    mkPoint(jR, 1);
  };

  // --- Block: placket ---------------------------------------------------------
  // A raised button strip down the centre front: polo half-placket or a full
  // button front (shirts, cardigans), with small button bumps.
  const buildPlacket = (half: boolean) => {
    // A soft, low strip: narrow, barely raised, with chamfered edges so it
    // reads as a sewn button band pressed into the front, not a box glued on.
    const stripHalf = 1.25 * SCALE;
    const raise = 0.3 * SCALE;
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
      cols[1].push(pushVertex(-stripHalf * 0.55, y, zSurf + raise, 0.05, v));
      cols[2].push(pushVertex(stripHalf * 0.55, y, zSurf + raise, 0.1, v));
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
    // Lower stand, flatter fold: a relaxed casual-shirt collar.
    buildFoldedCollar(2.8, 1.42, 3.0);
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
  geometry.userData.sleeveNormalRows = sleeveNormalRows;
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
