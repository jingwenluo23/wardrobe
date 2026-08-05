// ML-based garment extraction.
//
// Runs SegFormer clothes segmentation (Xenova/segformer_b2_clothes, ONNX)
// in-process via @huggingface/transformers. The model labels every pixel
// with a clothing class (Upper-clothes, Pants, Dress, ...), so the garment
// is isolated from the wearer and the background properly instead of by
// colour heuristics.
//
// The model (~65MB quantised) is downloaded from the Hugging Face Hub on
// first use and cached on disk, so the first upload after a fresh install
// takes noticeably longer. If the model cannot be loaded (offline, blocked
// network), callers fall back to the colour-heuristic extractor.

import sharp from "sharp";
import type { GarmentShapeEstimate } from "./garment-mesh";

type SegmenterOutput = Array<{
  label: string;
  mask: { data: Uint8Array | Uint8ClampedArray; width: number; height: number };
}>;

type Segmenter = (image: unknown) => Promise<SegmenterOutput>;

// ATR label set used by segformer_b2_clothes.
const CLOTHING_LABELS = new Set([
  "Upper-clothes",
  "Skirt",
  "Pants",
  "Dress",
  "Belt",
  "Left-shoe",
  "Right-shoe",
  "Scarf",
]);

/**
 * Class filter for the chosen category. `primary` is the strict match for
 * the garment the user picked; `fallback` is only consulted when the
 * primary classes cover almost nothing (e.g. SegFormer labels a longline
 * hoodie as Dress). Keeping the sets narrow stops other worn items (pants,
 * an undershirt labelled Dress, a scarf) from bleeding into the texture.
 */
function labelsForCategory(category: string): {
  primary: Set<string>;
  fallback: Set<string>;
} {
  const normalized = category.toLowerCase();
  if (normalized.startsWith("top") || normalized.startsWith("outer")) {
    return { primary: new Set(["Upper-clothes"]), fallback: new Set(["Dress"]) };
  }
  if (normalized.startsWith("bottom")) {
    return { primary: new Set(["Pants", "Skirt"]), fallback: new Set(["Dress"]) };
  }
  if (normalized.startsWith("dress")) {
    return { primary: new Set(["Dress"]), fallback: new Set(["Upper-clothes"]) };
  }
  if (normalized.startsWith("shoe") || normalized.startsWith("sock")) {
    return {
      primary: new Set(["Left-shoe", "Right-shoe"]),
      fallback: new Set<string>(),
    };
  }
  return { primary: CLOTHING_LABELS, fallback: new Set<string>() };
}

// Lazy singleton so the model loads once per server process.
//
// A FAILED load is deliberately not cached. The first load downloads the
// weights, so a network hiccup fails it — and remembering that failure poisons
// every later draft in the process, silently demoting all of them to the
// colour heuristic (visible as extraction quality dropping to 82% and the
// background and undershirt leaking into the texture) until the server is
// restarted. Clearing the slot lets the next draft try again; a successful
// load is still cached and shared.
const globalForSegmenter = globalThis as unknown as {
  __wardrobeSegmenter?: Promise<Segmenter | null>;
};

function getSegmenter(): Promise<Segmenter | null> {
  if (!globalForSegmenter.__wardrobeSegmenter) {
    const attempt = (async () => {
      // The first load pulls the weights over the network, so it is the step
      // most likely to fail on a flaky or rate-limited connection. Try a few
      // times with a short backoff before falling back — a single hiccup
      // should not cost this draft its segmentation.
      const { pipeline } = await import("@huggingface/transformers");
      for (let tryIndex = 0; tryIndex < 3; tryIndex += 1) {
        try {
          const segmenter = await pipeline(
            "image-segmentation",
            "Xenova/segformer_b2_clothes",
            { dtype: "q8" },
          );
          return segmenter as unknown as Segmenter;
        } catch (error) {
          const last = tryIndex === 2;
          console.warn(
            "[garment-segmentation] model load attempt " +
              (tryIndex + 1) +
              "/3 failed" +
              (last ? ", using colour heuristic" : ", retrying") +
              ": " +
              (error instanceof Error ? error.message : String(error)),
          );
          if (last) {
            return null;
          }
          await new Promise((resolve) =>
            setTimeout(resolve, 800 * (tryIndex + 1)),
          );
        }
      }
      return null;
    })().then((segmenter) => {
      if (!segmenter && globalForSegmenter.__wardrobeSegmenter === attempt) {
        globalForSegmenter.__wardrobeSegmenter = undefined;
      }
      return segmenter;
    });
    globalForSegmenter.__wardrobeSegmenter = attempt;
  }
  return globalForSegmenter.__wardrobeSegmenter;
}

export type ExtractionResult = {
  color: string;
  textureUrl: string;
  /** Plain fabric swatch (no print) for sleeves/collar, from the photo. */
  fabricTextureUrl?: string;
  /** Extracted sleeve texture (data URL), warped from the photo's sleeve. */
  sleeveTextureUrl?: string;
  /** Extracted hood texture (data URL), from the region above the shoulders. */
  hoodTextureUrl?: string;
  /** Relative silhouette measurements recovered from the cleaned mask. */
  shape?: GarmentShapeEstimate;
};

function estimateShape(
  mask: Uint8Array,
  width: number,
  height: number,
  torso: { minX: number; maxX: number; minY: number; maxY: number },
): GarmentShapeEstimate {
  const bodyW = Math.max(1, torso.maxX - torso.minX + 1);
  const bodyH = Math.max(1, torso.maxY - torso.minY + 1);
  let allMinX = width;
  let allMaxX = 0;
  let area = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (!mask[y * width + x]) continue;
      area += 1;
      allMinX = Math.min(allMinX, x);
      allMaxX = Math.max(allMaxX, x);
    }
  }
  const rowWidth = (fraction: number) => {
    const center = torso.minY + fraction * (bodyH - 1);
    const radius = Math.max(1, Math.round(bodyH * 0.025));
    const samples: number[] = [];
    for (
      let y = Math.max(torso.minY, Math.round(center) - radius);
      y <= Math.min(torso.maxY, Math.round(center) + radius);
      y += 1
    ) {
      let lo = width;
      let hi = -1;
      for (let x = torso.minX; x <= torso.maxX; x += 1) {
        if (mask[y * width + x]) {
          lo = Math.min(lo, x);
          hi = Math.max(hi, x);
        }
      }
      if (hi >= lo) samples.push(hi - lo + 1);
    }
    samples.sort((a, b) => a - b);
    return (samples[Math.floor(samples.length / 2)] ?? bodyW) / bodyW;
  };

  // The neck is the central background run above the first solid chest row.
  let maxNeckWidth = 0;
  let neckBottom = torso.minY;
  const centerX = Math.round((torso.minX + torso.maxX) / 2);
  for (let y = torso.minY; y <= torso.minY + bodyH * 0.3; y += 1) {
    if (mask[y * width + centerX]) continue;
    let lo = centerX;
    let hi = centerX;
    while (lo > torso.minX && !mask[y * width + lo - 1]) lo -= 1;
    while (hi < torso.maxX && !mask[y * width + hi + 1]) hi += 1;
    maxNeckWidth = Math.max(maxNeckWidth, hi - lo + 1);
    neckBottom = y;
  }
  const solidity = area / Math.max(1, (allMaxX - allMinX + 1) * bodyH);
  const frameCoverage = area / (width * height);
  const confidence = Math.max(
    0.25,
    Math.min(0.98, 0.45 + solidity * 0.35 + Math.min(0.18, frameCoverage)),
  );
  return {
    bodyAspectRatio: bodyH / bodyW,
    spanRatio: (allMaxX - allMinX + 1) / bodyW,
    shoulderRatio: rowWidth(0.1),
    chestRatio: rowWidth(0.3),
    waistRatio: rowWidth(0.62),
    hemRatio: rowWidth(0.94),
    neckWidthRatio: maxNeckWidth / bodyW,
    neckDepthRatio: Math.max(0, neckBottom - torso.minY) / bodyH,
    confidence,
  };
}

/**
 * Decompose a warped tile into base fabric + print layer, then recompose it
 * as clean flat fabric with the print composited back at its coordinates —
 * instead of pasting the photo crop, whose baked-in shading, wrinkles and
 * gap-fill seams read as a sticker on the mesh.
 *
 * 1. Scan every pixel to find the BASE colour: the dominant quantised colour
 *    (the fabric), refined to the mean of its neighbourhood.
 * 2. Classify each pixel as base fabric vs print/decoration with a
 *    shading-invariant test: photo shading scales the fabric colour
 *    multiplicatively, so darker/lighter fabric has a small residual against
 *    the scaled base, while graphics differ in hue and keep a large one.
 * 3. Recompose: flat base everywhere (gaps included — they now match
 *    perfectly), print pixels keep their photo colours; isolated speckles
 *    (noise, shadow edges) are dropped.
 *
 * Pass `base` to reuse the torso's base for sleeve/hood tiles so the whole
 * garment shares one exact fabric colour.
 */
export function flattenTile(
  tile: Buffer,
  size: number,
  base?: [number, number, number],
  apply?: boolean,
): { base: [number, number, number]; applied: boolean } {
  const n = size * size;
  let br: number;
  let bg: number;
  let bb: number;
  if (base) {
    [br, bg, bb] = base;
  } else {
    // Bin by CHROMATICITY, not raw RGB: shading spreads the fabric across
    // many RGB bins while a solid logo concentrates in one, so a raw
    // histogram can crown the logo. All shades of the fabric share one
    // chroma bin, so the fabric always wins on area.
    const CH = 40;
    const chromaKey = (r: number, g: number, b: number) => {
      const sum = r + g + b;
      if (sum < 24) {
        return -1; // near-black: chroma unstable
      }
      const cu = Math.min(CH - 1, Math.floor((r / sum) * CH));
      const cv = Math.min(CH - 1, Math.floor((g / sum) * CH));
      return cu * CH + cv;
    };
    const counts = new Map<number, number>();
    for (let i = 0; i < n; i += 1) {
      const key = chromaKey(tile[i * 3], tile[i * 3 + 1], tile[i * 3 + 2]);
      if (key >= 0) {
        counts.set(key, (counts.get(key) ?? 0) + 1);
      }
    }
    let bestKey = -1;
    let bestCount = -1;
    for (const [k, c] of counts) {
      if (c > bestCount) {
        bestCount = c;
        bestKey = k;
      }
    }
    const ku = Math.floor(bestKey / CH);
    const kv = bestKey % CH;
    // Within the winning chroma, pick the dominant LUMINANCE band — this
    // separates white fabric from a black print that shares its chroma.
    const lumCounts = new Array<number>(16).fill(0);
    for (let i = 0; i < n; i += 1) {
      const r = tile[i * 3];
      const g = tile[i * 3 + 1];
      const b = tile[i * 3 + 2];
      const key = chromaKey(r, g, b);
      if (key < 0) {
        continue;
      }
      if (Math.abs(Math.floor(key / CH) - ku) <= 1 && Math.abs((key % CH) - kv) <= 1) {
        lumCounts[Math.min(15, Math.floor((r + g + b) / 48))] += 1;
      }
    }
    let bestLum = 0;
    for (let l = 1; l < 16; l += 1) {
      if (lumCounts[l] > lumCounts[bestLum]) {
        bestLum = l;
      }
    }
    let sr = 0;
    let sg = 0;
    let sb = 0;
    let sc = 0;
    for (let i = 0; i < n; i += 1) {
      const r = tile[i * 3];
      const g = tile[i * 3 + 1];
      const b = tile[i * 3 + 2];
      const key = chromaKey(r, g, b);
      if (key < 0) {
        continue;
      }
      const lum = Math.min(15, Math.floor((r + g + b) / 48));
      if (
        Math.abs(Math.floor(key / CH) - ku) <= 1 &&
        Math.abs((key % CH) - kv) <= 1 &&
        Math.abs(lum - bestLum) <= 1
      ) {
        sr += r;
        sg += g;
        sb += b;
        sc += 1;
      }
    }
    br = sc ? sr / sc : 128;
    bg = sc ? sg / sc : 128;
    bb = sc ? sb / sc : 128;
  }
  const baseNorm = br * br + bg * bg + bb * bb || 1;
  const print = new Uint8Array(n);
  let resSum = 0;
  let resCount = 0;
  // Coarse chromaticity histogram. Shading moves a pixel along the base
  // direction (luminance) but leaves its chroma bin fixed, so a solid fabric —
  // however wrinkled or shadowed — stays in one bin. Multi-colour weaves
  // (camo, plaid, floral) genuinely span several earthy bins; that spread is
  // invisible to the multiplicative-residual and luminance-energy tests, which
  // is why low-contrast photographic camo was being flattened to blank cloth.
  const CH2 = 12;
  const chromaHist = new Array<number>(CH2 * CH2).fill(0);
  let chromaTotal = 0;
  for (let i = 0; i < n; i += 1) {
    const r = tile[i * 3];
    const g = tile[i * 3 + 1];
    const b = tile[i * 3 + 2];
    const sum = r + g + b;
    if (sum >= 24) {
      const cu = Math.min(CH2 - 1, Math.floor((r / sum) * CH2));
      const cv = Math.min(CH2 - 1, Math.floor((g / sum) * CH2));
      chromaHist[cu * CH2 + cv] += 1;
      chromaTotal += 1;
    }
    let s = (r * br + g * bg + b * bb) / baseNorm;
    s = Math.min(1.9, Math.max(0.35, s));
    const dr = r - s * br;
    const dg = g - s * bg;
    const db = b - s * bb;
    const res2 = dr * dr + dg * dg + db * db;
    if (res2 > 34 * 34) {
      print[i] = 1;
    } else {
      // Residual of "fabric" pixels: near zero for solid cloth (sensor
      // noise), large for patterned fabric whose tones hover under the
      // print threshold (camo, plaid, heather). RMS weighting makes the
      // pattern deviations count more than flat noise.
      resSum += res2;
      resCount += 1;
    }
  }
  // Despeckle: drop only truly ISOLATED print pixels (noise, shadow edges).
  // A thin line — a pinstripe, a seam, fine text — has just two neighbours
  // along its run, so the threshold must stay at <2 or stripes are erased and
  // the garment renders blank.
  const cleaned = new Uint8Array(print);
  for (let y = 1; y < size - 1; y += 1) {
    for (let x = 1; x < size - 1; x += 1) {
      const i = y * size + x;
      if (!print[i]) {
        continue;
      }
      const nb =
        print[i - 1] +
        print[i + 1] +
        print[i - size] +
        print[i + size] +
        print[i - size - 1] +
        print[i - size + 1] +
        print[i + size - 1] +
        print[i + size + 1];
      if (nb < 2) {
        cleaned[i] = 0;
      }
    }
  }
  // All-over patterned fabric (camo, plaid, floral) shares chroma with the
  // base, so flattening erases the pattern and leaves only streaks. When a
  // large share of pixels reads as print — or when the fabric bin itself is
  // a minority — this is patterned fabric: keep the photo tile untouched.
  let printCount = 0;
  for (let i = 0; i < n; i += 1) {
    printCount += cleaned[i];
  }
  const rmsResidual = resCount > 0 ? Math.sqrt(resSum / resCount) : 0;
  // High-frequency luminance energy: a neutral stripe/plaid/heather on a
  // neutral fabric is invisible to the multiplicative residual test (grey =
  // white x small factor looks exactly like shading), so it would be
  // flattened to blank cloth. But its sharp edges make the local luminance
  // jump; smooth shading does not. Mean absolute adjacent-luma difference
  // separates the two — high energy => patterned, keep the photo tile.
  let energy = 0;
  let energyCount = 0;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size - 1; x += 1) {
      const i = y * size + x;
      const l0 = tile[i * 3] + tile[i * 3 + 1] + tile[i * 3 + 2];
      const l1 = tile[(i + 1) * 3] + tile[(i + 1) * 3 + 1] + tile[(i + 1) * 3 + 2];
      energy += Math.abs(l0 - l1) / 3;
      energyCount += 1;
    }
  }
  const hfEnergy = energyCount > 0 ? energy / energyCount : 0;
  // Count chroma bins that each hold a meaningful share of the fabric. One or
  // two bins => solid cloth (optionally with a logo); three or more => a
  // genuine multi-colour weave (bright plaid/floral) that flattening erases.
  let chromaSpread = 0;
  if (chromaTotal > 0) {
    const floor = chromaTotal * 0.06;
    for (const count of chromaHist) {
      if (count >= floor) {
        chromaSpread += 1;
      }
    }
  }
  // Mid-scale luminance structure. Low-saturation camo (earthy Realtree tones)
  // barely moves in chroma or between adjacent pixels — it hides from both the
  // chroma spread and the high-frequency energy test — but its tonal BLOBS
  // survive heavy downsampling, while a solid fabric's fold shading is a smooth
  // gradient that averages out. The coefficient of variation of coarse block
  // means separates the two: ~0.12 for shaded solids, ~0.35 for camo.
  const grid = 16;
  const cell = Math.max(1, Math.floor(size / grid));
  const blocks: number[] = [];
  for (let by = 0; by + cell <= size; by += cell) {
    for (let bx = 0; bx + cell <= size; bx += cell) {
      let sum = 0;
      for (let y = by; y < by + cell; y += 1) {
        for (let x = bx; x < bx + cell; x += 1) {
          const i = (y * size + x) * 3;
          sum += (tile[i] + tile[i + 1] + tile[i + 2]) / 3;
        }
      }
      blocks.push(sum / (cell * cell));
    }
  }
  let coarseCov = 0;
  if (blocks.length > 0) {
    const mean = blocks.reduce((a, b) => a + b, 0) / blocks.length;
    const variance =
      blocks.reduce((a, b) => a + (b - mean) ** 2, 0) / blocks.length;
    coarseCov = Math.sqrt(variance) / Math.max(1, mean);
  }
  const shouldApply =
    apply ??
    (printCount / n <= 0.3 &&
      rmsResidual <= 10 &&
      hfEnergy <= 9 &&
      chromaSpread < 3 &&
      coarseCov < 0.28);
  if (shouldApply) {
    const R = Math.round(br);
    const G = Math.round(bg);
    const B = Math.round(bb);
    for (let i = 0; i < n; i += 1) {
      if (!cleaned[i]) {
        tile[i * 3] = R;
        tile[i * 3 + 1] = G;
        tile[i * 3 + 2] = B;
      }
    }
  }
  return { base: [br, bg, bb], applied: shouldApply };
}

/**
 * Fill non-garment tile pixels with nearby fabric instead of a flat colour,
 * so painted regions (neck opening, seams) blend into the shirt's real
 * shading rather than reading as a ghost collar or patch.
 */
export function inpaintTile(
  tileRaw: Buffer,
  keep: Uint8Array,
  size: number,
  fallback: [number, number, number],
) {
  // Filling a gap by repeating one edge pixel produces streaks. Instead,
  // blend each gap between the nearest kept pixels above and below
  // (vertical pass), then between left and right (horizontal pass, also
  // covering fully-empty columns), and finally smooth the filled region so
  // patched areas read as soft fabric rather than smeared stripes.
  const filled = new Uint8Array(size * size);

  // Pass 0 — texture-preserving fill for LARGE holes.
  //
  // Interpolating across a big empty region stretches whatever pixel sits at
  // its edge over the whole span, which turns a patterned fabric (camo, plaid)
  // into long smeared streaks. Patterned cloth is statistically uniform, so a
  // far more faithful fill is real fabric copied from elsewhere in the tile.
  // Find the densest patch of surviving garment and mirror-tile it into every
  // pixel that is far from any kept pixel, then mark those pixels kept so the
  // interpolation below only handles what it is good at: thin seams and small
  // holes near real fabric. Pixels close to kept fabric are left alone so the
  // fill still blends locally instead of hard-edging.
  {
    const n = size * size;
    let keptCount = 0;
    for (let i = 0; i < n; i += 1) {
      keptCount += keep[i];
    }
    if (keptCount > n * 0.04 && keptCount < n) {
      // Chamfer distance to the nearest kept pixel.
      const INF = 1e9;
      const dist = new Float32Array(n);
      for (let i = 0; i < n; i += 1) {
        dist[i] = keep[i] ? 0 : INF;
      }
      for (let y = 0; y < size; y += 1) {
        for (let x = 0; x < size; x += 1) {
          const i = y * size + x;
          let d = dist[i];
          if (y > 0) d = Math.min(d, dist[i - size] + 1);
          if (x > 0) d = Math.min(d, dist[i - 1] + 1);
          if (y > 0 && x > 0) d = Math.min(d, dist[i - size - 1] + 1.414);
          if (y > 0 && x < size - 1) d = Math.min(d, dist[i - size + 1] + 1.414);
          dist[i] = d;
        }
      }
      for (let y = size - 1; y >= 0; y -= 1) {
        for (let x = size - 1; x >= 0; x -= 1) {
          const i = y * size + x;
          let d = dist[i];
          if (y < size - 1) d = Math.min(d, dist[i + size] + 1);
          if (x < size - 1) d = Math.min(d, dist[i + 1] + 1);
          if (y < size - 1 && x < size - 1)
            d = Math.min(d, dist[i + size + 1] + 1.414);
          if (y < size - 1 && x > 0) d = Math.min(d, dist[i + size - 1] + 1.414);
          dist[i] = d;
        }
      }
      // Densest square of kept fabric, via a summed-area table.
      const P = Math.max(16, Math.min(128, size >> 2));
      const stride = size + 1;
      const integral = new Int32Array(stride * stride);
      for (let y = 0; y < size; y += 1) {
        for (let x = 0; x < size; x += 1) {
          integral[(y + 1) * stride + (x + 1)] =
            keep[y * size + x] +
            integral[y * stride + (x + 1)] +
            integral[(y + 1) * stride + x] -
            integral[y * stride + x];
        }
      }
      let bestX = 0;
      let bestY = 0;
      let best = -1;
      for (let y = 0; y + P <= size; y += 4) {
        for (let x = 0; x + P <= size; x += 4) {
          const s =
            integral[(y + P) * stride + (x + P)] -
            integral[y * stride + (x + P)] -
            integral[(y + P) * stride + x] +
            integral[y * stride + x];
          if (s > best) {
            best = s;
            bestX = x;
            bestY = y;
          }
        }
      }
      if (best > P * P * 0.6) {
        // Snapshot the patch so sampling can never read pixels this pass has
        // already rewritten.
        const patch = Buffer.alloc(P * P * 3);
        for (let y = 0; y < P; y += 1) {
          for (let x = 0; x < P; x += 1) {
            const src = ((bestY + y) * size + (bestX + x)) * 3;
            const dst = (y * P + x) * 3;
            patch[dst] = tileRaw[src];
            patch[dst + 1] = tileRaw[src + 1];
            patch[dst + 2] = tileRaw[src + 2];
          }
        }
        // Each patch-sized block is wrapped (not mirrored) and shifted by a
        // deterministic per-block offset. Mirroring joins blocks seamlessly but
        // makes every pair a reflection, which on a large fill reads as an
        // obvious kaleidoscope butterfly down the middle of the garment. Camo
        // and other organic prints are irregular and high-frequency, so a
        // shifted wrap hides its joins far better than symmetry does.
        const blockShift = (bx: number, by: number) => {
          let h = (bx * 73856093) ^ (by * 19349663);
          h = (h ^ (h >>> 13)) >>> 0;
          return h % P;
        };
        for (let y = 0; y < size; y += 1) {
          for (let x = 0; x < size; x += 1) {
            const i = y * size + x;
            if (keep[i] || dist[i] <= 6) {
              continue;
            }
            const bx = Math.floor(x / P);
            const by = Math.floor(y / P);
            const su = (x + blockShift(bx, by)) % P;
            const sv = (y + blockShift(by, bx + 7)) % P;
            const s = (sv * P + su) * 3;
            tileRaw[i * 3] = patch[s];
            tileRaw[i * 3 + 1] = patch[s + 1];
            tileRaw[i * 3 + 2] = patch[s + 2];
            // Real fabric now: let the passes below treat it as source and
            // only feather the narrow band still separating it from the
            // original garment pixels.
            keep[i] = 1;
          }
        }
      }
    }
  }

  const lerpFill = (
    ti: number,
    ai: number,
    bi: number | null,
    w: number,
  ) => {
    for (let c = 0; c < 3; c += 1) {
      tileRaw[ti + c] =
        bi === null
          ? tileRaw[ai + c]
          : Math.round(tileRaw[ai + c] * (1 - w) + tileRaw[bi + c] * w);
    }
  };

  // Vertical pass: interpolate across each gap in the column.
  for (let x = 0; x < size; x += 1) {
    let prevY = -1;
    let y = 0;
    while (y < size) {
      if (keep[y * size + x]) {
        prevY = y;
        y += 1;
        continue;
      }
      let nextY = y;
      while (nextY < size && !keep[nextY * size + x]) {
        nextY += 1;
      }
      const hasNext = nextY < size;
      if (prevY === -1 && !hasNext) {
        // Whole column empty: leave for the horizontal pass.
        for (let fy = 0; fy < size; fy += 1) {
          filled[fy * size + x] = 2;
        }
        break;
      }
      for (let fy = y; fy < (hasNext ? nextY : size); fy += 1) {
        const ti = (fy * size + x) * 3;
        if (prevY === -1) {
          lerpFill(ti, (nextY * size + x) * 3, null, 0);
        } else if (!hasNext) {
          lerpFill(ti, (prevY * size + x) * 3, null, 0);
        } else {
          const w = (fy - prevY) / (nextY - prevY);
          lerpFill(ti, (prevY * size + x) * 3, (nextY * size + x) * 3, w);
        }
        filled[fy * size + x] = 1;
      }
      y = hasNext ? nextY : size;
    }
  }

  // Horizontal pass for columns that had no fabric at all: interpolate
  // between the nearest filled/kept columns on each side.
  for (let y = 0; y < size; y += 1) {
    let prevX = -1;
    let x = 0;
    while (x < size) {
      if (filled[y * size + x] !== 2) {
        prevX = x;
        x += 1;
        continue;
      }
      let nextX = x;
      while (nextX < size && filled[y * size + nextX] === 2) {
        nextX += 1;
      }
      const hasNext = nextX < size;
      for (let fx = x; fx < (hasNext ? nextX : size); fx += 1) {
        const ti = (y * size + fx) * 3;
        if (prevX === -1 && !hasNext) {
          tileRaw[ti] = fallback[0];
          tileRaw[ti + 1] = fallback[1];
          tileRaw[ti + 2] = fallback[2];
        } else if (prevX === -1) {
          lerpFill(ti, (y * size + nextX) * 3, null, 0);
        } else if (!hasNext) {
          lerpFill(ti, (y * size + prevX) * 3, null, 0);
        } else {
          const w = (fx - prevX) / (nextX - prevX);
          lerpFill(ti, (y * size + prevX) * 3, (y * size + nextX) * 3, w);
        }
        filled[y * size + fx] = 1;
      }
      x = hasNext ? nextX : size;
    }
  }

  // Smooth only the filled pixels (3x3 box, two rounds) so patches blend
  // into the surrounding fabric without blurring the real print.
  for (let round = 0; round < 2; round += 1) {
    const src = Buffer.from(tileRaw);
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        if (!filled[y * size + x]) {
          continue;
        }
        const sums = [0, 0, 0];
        let n = 0;
        for (let dy = -1; dy <= 1; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            const sy = y + dy;
            const sx = x + dx;
            if (sy < 0 || sy >= size || sx < 0 || sx >= size) {
              continue;
            }
            const si = (sy * size + sx) * 3;
            sums[0] += src[si];
            sums[1] += src[si + 1];
            sums[2] += src[si + 2];
            n += 1;
          }
        }
        const ti = (y * size + x) * 3;
        tileRaw[ti] = Math.round(sums[0] / n);
        tileRaw[ti + 1] = Math.round(sums[1] / n);
        tileRaw[ti + 2] = Math.round(sums[2] / n);
      }
    }
  }
}

/**
 * Pick the most uniform block of the tile (lowest colour variance = plain
 * fabric, no print) to use as the sleeve/collar swatch, and return its mean
 * colour as the garment base colour.
 */
export function pickFabricSwatch(
  tileRaw: Buffer,
  size: number,
): { swatch: Buffer; swatchSize: number; mean: [number, number, number] } {
  const block = 64;
  const stride = 32;
  let best = { score: Infinity, x: 0, y: 0 };
  for (let by = size * 0.1; by + block <= size * 0.9; by += stride) {
    for (let bx = size * 0.1; bx + block <= size * 0.9; bx += stride) {
      const sum = [0, 0, 0];
      const sumSq = [0, 0, 0];
      for (let y = 0; y < block; y += 4) {
        for (let x = 0; x < block; x += 4) {
          const i = ((Math.floor(by) + y) * size + Math.floor(bx) + x) * 3;
          for (let c = 0; c < 3; c += 1) {
            sum[c] += tileRaw[i + c];
            sumSq[c] += tileRaw[i + c] * tileRaw[i + c];
          }
        }
      }
      const n = (block / 4) * (block / 4);
      const variance =
        sumSq[0] / n - (sum[0] / n) ** 2 +
        (sumSq[1] / n - (sum[1] / n) ** 2) +
        (sumSq[2] / n - (sum[2] / n) ** 2);
      if (variance < best.score) {
        best = { score: variance, x: Math.floor(bx), y: Math.floor(by) };
      }
    }
  }

  const swatch = Buffer.alloc(block * block * 3);
  const sums: [number, number, number] = [0, 0, 0];
  for (let y = 0; y < block; y += 1) {
    for (let x = 0; x < block; x += 1) {
      const si = ((best.y + y) * size + best.x + x) * 3;
      const ti = (y * block + x) * 3;
      swatch[ti] = tileRaw[si];
      swatch[ti + 1] = tileRaw[si + 1];
      swatch[ti + 2] = tileRaw[si + 2];
      sums[0] += tileRaw[si];
      sums[1] += tileRaw[si + 1];
      sums[2] += tileRaw[si + 2];
    }
  }
  const n = block * block;
  return {
    swatch,
    swatchSize: block,
    mean: [
      Math.round(sums[0] / n),
      Math.round(sums[1] / n),
      Math.round(sums[2] / n),
    ],
  };
}

/**
 * Dense pose/perspective warp from photo space to panel UV space.
 *
 * The mesh panel's UVs run side seam -> side seam horizontally at every
 * height, and hem -> shoulder vertically. This function builds the same
 * parameterisation over the photo's garment mask:
 * - per-row left/right garment edges give the horizontal mapping, following
 *   the real silhouette (drape, pose tilt, perspective),
 * - shoulder and hem lines fitted through two probe columns give the
 *   vertical mapping, de-rotating a tilted garment.
 * Every output pixel is sampled through that mapping, so prints land where
 * they sit on the real shirt even when the photo is not straight-on.
 */
export function warpGarmentTile(input: {
  data: Buffer | Uint8Array;
  mask: Uint8Array;
  width: number;
  height: number;
  minX: number;
  maxX: number;
  tileSize: number;
  exclude?: (r: number, g: number, b: number) => boolean;
  /** Optional higher-resolution copy of the photo for sharper sampling. */
  hiData?: Buffer | Uint8Array;
  hiWidth?: number;
  hiHeight?: number;
}): { tileRaw: Buffer; keep: Uint8Array } {
  const { data, mask, width, height, minX, maxX, tileSize, exclude } = input;
  const hiData = input.hiData ?? data;
  const hiWidth = input.hiWidth ?? width;
  const hiHeight = input.hiHeight ?? height;
  const scaleX = hiWidth / width;
  const scaleY = hiHeight / height;

  // Per-row garment edges, restricted to torso columns so sleeves are not
  // mistaken for the side seams.
  const rowLeft = new Int32Array(height).fill(-1);
  const rowRight = new Int32Array(height).fill(-1);
  for (let y = 0; y < height; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      if (mask[y * width + x]) {
        if (rowLeft[y] === -1) {
          rowLeft[y] = x;
        }
        rowRight[y] = x;
      }
    }
  }
  // Smooth the edges (5-row moving average over valid rows) to remove
  // segmentation jitter.
  const smooth = (edges: Int32Array) => {
    const source = edges.slice();
    for (let y = 0; y < height; y += 1) {
      if (source[y] === -1) {
        continue;
      }
      let sum = 0;
      let count = 0;
      for (let dy = -2; dy <= 2; dy += 1) {
        const yy = y + dy;
        if (yy >= 0 && yy < height && source[yy] !== -1) {
          sum += source[yy];
          count += 1;
        }
      }
      edges[y] = Math.round(sum / count);
    }
  };
  smooth(rowLeft);
  smooth(rowRight);

  // Shoulder and hem lines through two probe columns (25% / 75% across the
  // torso). Scanning a few neighbouring columns guards against mask holes.
  const probe = (xTarget: number) => {
    let top = -1;
    let bottom = -1;
    for (let dx = 0; dx <= 4 && (top === -1 || bottom === -1); dx += 1) {
      for (const x of [xTarget - dx, xTarget + dx]) {
        if (x < minX || x > maxX) {
          continue;
        }
        for (let y = 0; y < height; y += 1) {
          if (mask[y * width + x]) {
            if (top === -1 || y < top) {
              top = y;
            }
            if (bottom === -1 || y > bottom) {
              bottom = y;
            }
            break;
          }
        }
        for (let y = height - 1; y >= 0; y -= 1) {
          if (mask[y * width + x]) {
            if (bottom === -1 || y > bottom) {
              bottom = y;
            }
            break;
          }
        }
      }
    }
    return { top, bottom };
  };
  const xA = Math.round(minX + (maxX - minX) * 0.25);
  const xB = Math.round(minX + (maxX - minX) * 0.75);
  const a = probe(xA);
  const b = probe(xB);

  const lineAt = (x: number, yA: number, yB: number) => {
    if (yA === -1 || yB === -1) {
      return yA !== -1 ? yA : yB;
    }
    if (xB === xA) {
      return yA;
    }
    return yA + ((x - xA) / (xB - xA)) * (yB - yA);
  };

  const tileRaw = Buffer.alloc(tileSize * tileSize * 3);
  const keep = new Uint8Array(tileSize * tileSize);

  for (let ty = 0; ty < tileSize; ty += 1) {
    const v = ty / (tileSize - 1);
    for (let tx = 0; tx < tileSize; tx += 1) {
      const u = tx / (tileSize - 1);

      // First estimate of the source column, refined once through the
      // row-edge mapping (the two are mutually dependent).
      let sx = minX + u * (maxX - minX);
      let sy = 0;
      for (let iter = 0; iter < 2; iter += 1) {
        const yTop = lineAt(sx, a.top, b.top);
        const yBottom = lineAt(sx, a.bottom, b.bottom);
        if (yTop === -1 || yBottom === -1 || yBottom - yTop < 4) {
          break;
        }
        sy = clampInt(Math.round(yTop + v * (yBottom - yTop)), 0, height - 1);
        const left = rowLeft[sy];
        const right = rowRight[sy];
        if (left !== -1 && right - left >= 4) {
          sx = left + u * (right - left);
        }
      }
      const sxi = clampInt(Math.round(sx), 0, width - 1);
      const si = sy * width + sxi;
      // Colour from the high-res copy, sampled bilinearly — nearest-neighbour
      // duplicates pixels into blocky streaks wherever the panel stretches
      // the photo. Mask/keep still comes from the inference-res grid.
      const fx = Math.min(Math.max(sx * scaleX, 0), hiWidth - 1);
      const fy = Math.min(Math.max(sy * scaleY, 0), hiHeight - 1);
      const x0 = Math.floor(fx);
      const y0 = Math.floor(fy);
      const x1 = Math.min(x0 + 1, hiWidth - 1);
      const y1 = Math.min(y0 + 1, hiHeight - 1);
      const wx = fx - x0;
      const wy = fy - y0;
      const i00 = (y0 * hiWidth + x0) * 3;
      const i10 = (y0 * hiWidth + x1) * 3;
      const i01 = (y1 * hiWidth + x0) * 3;
      const i11 = (y1 * hiWidth + x1) * 3;
      const ti = (ty * tileSize + tx) * 3;
      let r = 0;
      let g = 0;
      let bch = 0;
      for (let c = 0; c < 3; c += 1) {
        const top = hiData[i00 + c] * (1 - wx) + hiData[i10 + c] * wx;
        const bottom = hiData[i01 + c] * (1 - wx) + hiData[i11 + c] * wx;
        const value = Math.round(top * (1 - wy) + bottom * wy);
        if (c === 0) r = value;
        else if (c === 1) g = value;
        else bch = value;
      }
      tileRaw[ti] = r;
      tileRaw[ti + 1] = g;
      tileRaw[ti + 2] = bch;
      if (mask[si] && !(exclude && exclude(r, g, bch))) {
        keep[ty * tileSize + tx] = 1;
      }
    }
  }

  return { tileRaw, keep };
}

function clampInt(value: number, min: number, max: number) {
  return value < min ? min : value > max ? max : value;
}

export function rgbToHex(r: number, g: number, b: number) {
  const toHex = (value: number) =>
    Math.max(0, Math.min(255, Math.round(value)))
      .toString(16)
      .padStart(2, "0");
  return "#" + toHex(r) + toHex(g) + toHex(b);
}

/** In-place binary erosion (4-neighbourhood), `iterations` pixels deep. */
function erodeMask(
  mask: Uint8Array,
  width: number,
  height: number,
  iterations: number,
) {
  for (let iter = 0; iter < iterations; iter += 1) {
    const source = mask.slice();
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const i = y * width + x;
        if (!source[i]) {
          continue;
        }
        const atEdge = x === 0 || x === width - 1 || y === 0 || y === height - 1;
        if (
          atEdge ||
          !source[i - 1] ||
          !source[i + 1] ||
          !source[i - width] ||
          !source[i + width]
        ) {
          mask[i] = 0;
        }
      }
    }
  }
}

/** Keep only the largest 4-connected component of the mask, in place. */
function keepLargestComponent(mask: Uint8Array, width: number, height: number) {
  const labels = new Int32Array(width * height).fill(-1);
  const sizes: number[] = [];
  const stack: number[] = [];
  for (let start = 0; start < mask.length; start += 1) {
    if (!mask[start] || labels[start] !== -1) {
      continue;
    }
    const label = sizes.length;
    let size = 0;
    stack.push(start);
    labels[start] = label;
    while (stack.length > 0) {
      const i = stack.pop() as number;
      size += 1;
      const x = i % width;
      const neighbors = [
        x > 0 ? i - 1 : -1,
        x < width - 1 ? i + 1 : -1,
        i - width,
        i + width,
      ];
      for (const n of neighbors) {
        if (n >= 0 && n < mask.length && mask[n] && labels[n] === -1) {
          labels[n] = label;
          stack.push(n);
        }
      }
    }
    sizes.push(size);
  }
  if (sizes.length <= 1) {
    return;
  }
  const largest = sizes.indexOf(Math.max(...sizes));
  for (let i = 0; i < mask.length; i += 1) {
    if (mask[i] && labels[i] !== largest) {
      mask[i] = 0;
    }
  }
}

/**
 * Drop layered foreign garments from the mask edge (an undershirt peeking
 * below the hem, a shirt collar above the neckline). The segmentation model
 * often gives them the SAME class as the chosen garment, so class filtering
 * cannot separate them — but they only ever appear along the mask boundary.
 * Build the garment's colour palette from the eroded core of the mask, then
 * remove boundary-band pixels whose colour is not in that palette.
 */
export function trimForeignLayers(
  mask: Uint8Array,
  data: Buffer | Uint8Array,
  width: number,
  height: number,
) {
  // The palette core is the eroded mask restricted to the CENTRAL band of
  // the garment's bounding box: a layered garment only ever shows at the
  // extremes (below the hem, above the collar), so the centre is pure
  // chosen-garment fabric even when the layer is thick enough to survive
  // erosion.
  let minX = width;
  let maxX = 0;
  let minY = height;
  let maxY = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (mask[y * width + x]) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxY - minY < 20 || maxX - minX < 20) {
    return;
  }
  const yLo = minY + (maxY - minY) * 0.15;
  const yHi = minY + (maxY - minY) * 0.78;
  const xLo = minX + (maxX - minX) * 0.1;
  const xHi = maxX - (maxX - minX) * 0.1;

  const core = mask.slice();
  erodeMask(core, width, height, 6);
  let coreCount = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x;
      if (core[i] && (y < yLo || y > yHi || x < xLo || x > xHi)) {
        core[i] = 0;
      }
      coreCount += core[i];
    }
  }
  let maskCount = 0;
  for (let i = 0; i < mask.length; i += 1) {
    maskCount += mask[i];
  }
  // Thin garments erode away; without a trustworthy core, don't trim.
  if (coreCount < maskCount * 0.15 || coreCount < 500) {
    return;
  }

  // Quantised palette (5 bits per channel) of the core fabric. Bins that
  // cover a meaningful share of the core are "garment colours"; prints are
  // multi-colour, so every recurring print colour lands in the palette.
  const bin = (i: number) =>
    ((data[i * 3] >> 3) << 10) | ((data[i * 3 + 1] >> 3) << 5) | (data[i * 3 + 2] >> 3);
  const counts = new Map<number, number>();
  for (let i = 0; i < core.length; i += 1) {
    if (core[i]) {
      const key = bin(i);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  const minCount = Math.max(8, coreCount * 0.002);
  const palette = new Set<number>();
  for (const [key, count] of counts) {
    if (count >= minCount) {
      palette.add(key);
    }
  }
  if (palette.size === 0) {
    return;
  }

  // A boundary pixel matches if its bin — or any bin within ±2 levels per
  // channel — is in the palette (tolerance ~±16 RGB levels; looser keeps
  // print shades near the hem from being over-trimmed into inpaint smears).
  const matches = (key: number) => {
    const r = (key >> 10) & 31;
    const g = (key >> 5) & 31;
    const b = key & 31;
    for (let dr = -2; dr <= 2; dr += 1) {
      for (let dg = -2; dg <= 2; dg += 1) {
        for (let db = -2; db <= 2; db += 1) {
          const rr = r + dr;
          const gg = g + dg;
          const bb = b + db;
          if (rr < 0 || rr > 31 || gg < 0 || gg > 31 || bb < 0 || bb > 31) {
            continue;
          }
          if (palette.has((rr << 10) | (gg << 5) | bb)) {
            return true;
          }
        }
      }
    }
    return false;
  };
  for (let i = 0; i < mask.length; i += 1) {
    if (mask[i] && !core[i] && !matches(bin(i))) {
      mask[i] = 0;
    }
  }
}

/**
 * RGB skin-tone test, the last guard against skin on the texture.
 * Tightened so warm print colours survive: real skin always has green above
 * blue (pink/magenta prints have blue above green) and only moderate
 * saturation (vivid reds/oranges in graphics exceed it).
 */
export function isSkinTone(r: number, g: number, b: number) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  return (
    r > 95 &&
    g > 40 &&
    b > 20 &&
    max - min > 15 &&
    Math.abs(r - g) > 15 &&
    r > g &&
    r > b &&
    g > b &&
    max - min < max * 0.62
  );
}

/**
 * Decide whether the skin-tone exclusion is safe for this garment.
 *
 * The skin test accepts any moderately saturated colour with r > g > b, which
 * is also the exact signature of earthy fabric: tan, khaki, brown, and the bark
 * tones of a camo print. The segmentation mask already isolates the clothing,
 * so genuine skin bleed inside it is small (a little at the neck and wrists).
 * When a large share of the masked garment reads as "skin" the test is clearly
 * matching the fabric itself, and applying it would punch most of the garment
 * out of the tile — the inpainter then stretches the few surviving pixels down
 * each column, which is what produced heavy vertical smearing on camo. In that
 * case drop the exclusion and keep the fabric.
 */
function skinExcluderFor(
  data: Uint8Array | Buffer,
  mask: Uint8Array,
  width: number,
  height: number,
): ((r: number, g: number, b: number) => boolean) | undefined {
  let masked = 0;
  let skin = 0;
  // Sparse scan: this only needs a ratio, not an exact count.
  for (let y = 0; y < height; y += 2) {
    for (let x = 0; x < width; x += 2) {
      const i = y * width + x;
      if (!mask[i]) {
        continue;
      }
      masked += 1;
      if (isSkinTone(data[i * 3], data[i * 3 + 1], data[i * 3 + 2])) {
        skin += 1;
      }
    }
  }
  if (masked === 0) {
    return isSkinTone;
  }
  return skin / masked > 0.25 ? undefined : isSkinTone;
}

const INFER_SIZE = 768;
const HI_SIZE = 1536;
const TILE = 512;

// Photographs of cloth come back more saturated than the garment reads in
// hand — studio lighting and JPEG both push colour. Pull the saturation back
// and lift the tone a touch so the print looks paler and more natural on the
// mesh, rather than a vivid version of itself.
const FABRIC_TONE = { saturation: 0.78, brightness: 1.03 };

/**
 * Chromaticity histogram of the masked pixels in a column range.
 *
 * Binned by chromaticity (r/sum, g/sum) rather than raw RGB so that the same
 * cloth photographed in shadow on the arm and in light on the chest lands in
 * the same bins. What survives is which COLOURS the fabric is made of, not how
 * brightly that patch happened to be lit.
 */
function chromaHistogram(
  data: Uint8Array | Buffer,
  mask: Uint8Array,
  width: number,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
): { hist: Float64Array; count: number } {
  const B = 8;
  const hist = new Float64Array(B * B);
  let count = 0;
  for (let y = y0; y <= y1; y += 1) {
    for (let x = x0; x <= x1; x += 1) {
      const i = y * width + x;
      if (!mask[i]) {
        continue;
      }
      const r = data[i * 3];
      const g = data[i * 3 + 1];
      const b = data[i * 3 + 2];
      const sum = r + g + b;
      if (sum < 24) {
        continue;
      }
      const cu = Math.min(B - 1, Math.floor((r / sum) * B * 1.5));
      const cv = Math.min(B - 1, Math.floor((g / sum) * B * 1.5));
      hist[cu * B + cv] += 1;
      count += 1;
    }
  }
  if (count > 0) {
    for (let i = 0; i < hist.length; i += 1) {
      hist[i] /= count;
    }
  }
  return { hist, count };
}

/** Histogram intersection: 1 = identical palettes, 0 = nothing in common. */
function histogramIntersection(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += Math.min(a[i], b[i]);
  }
  return sum;
}

/**
 * Extract the garment from a photo with the segmentation model.
 * Returns null when the model is unavailable or finds no garment,
 * so the caller can fall back to the colour heuristic.
 */
export async function segmentGarment(
  buffer: Buffer,
  category: string,
  options?: { hood?: boolean },
): Promise<ExtractionResult | null> {
  const segmenter = await getSegmenter();
  if (!segmenter) {
    return null;
  }

  try {
    const { data, info } = await sharp(buffer)
      .rotate()
      .resize(INFER_SIZE, INFER_SIZE, { fit: "inside" })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    const width = info.width;
    const height = info.height;

    // Higher-resolution copy for texture sampling: inference runs at 768,
    // but sampling colours from it blurs prints on large photos.
    const { data: hiData, info: hiInfo } = await sharp(buffer)
      .rotate()
      .resize(HI_SIZE, HI_SIZE, { fit: "inside", withoutEnlargement: true })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const { RawImage } = await import("@huggingface/transformers");
    const image = new RawImage(new Uint8ClampedArray(data), width, height, 3);
    const results = await segmenter(image);

    // Union of the masks matching the chosen category, strictly: only the
    // primary classes for that category; the fallback set is consulted only
    // when the primary classes cover almost nothing (e.g. a longline hoodie
    // that the model labelled Dress).
    const wanted = labelsForCategory(category);
    const collect = (labels: Set<string>) => {
      const mask = new Uint8Array(width * height);
      let maskCount = 0;
      for (const result of results) {
        if (!labels.has(result.label)) {
          continue;
        }
        const maskData = result.mask.data;
        for (let i = 0; i < mask.length; i += 1) {
          if (maskData[i] > 127 && !mask[i]) {
            mask[i] = 1;
            maskCount += 1;
          }
        }
      }
      return { mask, maskCount };
    };
    let { mask, maskCount } = collect(wanted.primary);
    if (maskCount < width * height * 0.02 && wanted.fallback.size > 0) {
      ({ mask, maskCount } = collect(wanted.fallback));
    }
    // Require a meaningful garment area (>2% of the frame).
    if (maskCount < width * height * 0.02) {
      return null;
    }

    // Clean the mask:
    // 1. Erode a few pixels so boundary bleed (arm skin hugging the side
    //    seam, background halos) cannot reach the texture.
    // 2. Keep only the largest connected component, dropping stray
    //    misclassified blobs (photo captions, hands near the hem).
    // 3. Trim layered foreign garments (undershirt below the hem, shirt
    //    collar at the neck) whose colours don't belong to the garment.
    erodeMask(mask, width, height, 2);
    keepLargestComponent(mask, width, height);
    trimForeignLayers(mask, data, width, height);
    keepLargestComponent(mask, width, height);

    // Torso-aware crop: the mesh's front/back panels only span the torso
    // (side seam to side seam), but the garment mask includes the sleeves.
    // Columns through the torso are tall; columns that only cross a sleeve
    // are short. Restricting the crop to shoulder-height columns keeps the
    // print's position and scale close to 1:1 with the real shirt.
    const colHeight = new Array<number>(width).fill(0);
    for (let x = 0; x < width; x += 1) {
      let count = 0;
      for (let y = 0; y < height; y += 1) {
        if (mask[y * width + x]) {
          count += 1;
        }
      }
      colHeight[x] = count;
    }
    const maxColHeight = Math.max(...colHeight);
    const isTorsoCol = (x: number) => colHeight[x] >= maxColHeight * 0.55;

    let minX = width;
    let maxX = 0;
    let minY = height;
    let maxY = 0;
    for (let x = 0; x < width; x += 1) {
      if (!isTorsoCol(x)) {
        continue;
      }
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      for (let y = 0; y < height; y += 1) {
        if (mask[y * width + x]) {
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (minX >= maxX || minY >= maxY) {
      return null;
    }
    const shape = estimateShape(mask, width, height, {
      minX,
      maxX,
      minY,
      maxY,
    });

    // Garment colour: median of the masked pixels.
    const rs: number[] = [];
    const gs: number[] = [];
    const bs: number[] = [];
    const stride = Math.max(1, Math.floor(Math.sqrt(maskCount / 4000)));
    for (let y = minY; y <= maxY; y += stride) {
      for (let x = minX; x <= maxX; x += stride) {
        const i = y * width + x;
        if (mask[i]) {
          rs.push(data[i * 3]);
          gs.push(data[i * 3 + 1]);
          bs.push(data[i * 3 + 2]);
        }
      }
    }
    const median = (arr: number[]) => {
      const sorted = [...arr].sort((a, b) => a - b);
      return sorted[Math.floor(sorted.length / 2)] ?? 128;
    };
    const gr = median(rs);
    const gg = median(gs);
    const gb = median(bs);

    // Skip the skin-tone exclusion on earthy fabrics it would misread as skin
    // (camo, tan, khaki). Decided once here and reused for every region below
    // so the whole garment is treated consistently.
    const skinExclude = skinExcluderFor(data, mask, width, height);

    // Build the texture tile through the dense pose/perspective warp, then
    // inpaint non-garment pixels (neck opening, seams, skin) from the
    // surrounding fabric so no ghost collar or flat patches appear.
    const { tileRaw, keep } = warpGarmentTile({
      data,
      mask,
      width,
      height,
      minX,
      maxX,
      tileSize: TILE,
      exclude: skinExclude,
      hiData,
      hiWidth: hiInfo.width,
      hiHeight: hiInfo.height,
    });

    // Fidelity guard: if the warp only covered part of the tile, the rest has
    // to be invented, and no amount of clever filling reproduces a print that
    // was never sampled — patched fabric shows up as blocks and smears that do
    // not match the photo. A plain rectified crop of the garment box is a far
    // better likeness in that case: every pixel is real photographed cloth, the
    // print stays continuous, and the only cost is a little background near the
    // edges. The warp still wins when it covers the tile, because it undoes the
    // body's perspective.
    {
      let covered = 0;
      for (let i = 0; i < TILE * TILE; i += 1) {
        covered += keep[i];
      }
      if (covered < TILE * TILE * 0.62) {
        const sx = hiInfo.width / width;
        const sy = hiInfo.height / height;
        const x0 = clampInt(Math.floor(minX * sx), 0, hiInfo.width - 1);
        const x1 = clampInt(Math.ceil((maxX + 1) * sx), x0 + 1, hiInfo.width);
        const y0 = clampInt(Math.floor(minY * sy), 0, hiInfo.height - 1);
        const y1 = clampInt(Math.ceil((maxY + 1) * sy), y0 + 1, hiInfo.height);
        const cw = x1 - x0;
        const ch = y1 - y0;
        for (let ty = 0; ty < TILE; ty += 1) {
          const srcY = y0 + Math.min(ch - 1, Math.floor((ty / TILE) * ch));
          for (let tx = 0; tx < TILE; tx += 1) {
            const srcX = x0 + Math.min(cw - 1, Math.floor((tx / TILE) * cw));
            const s = (srcY * hiInfo.width + srcX) * 3;
            const d = (ty * TILE + tx) * 3;
            tileRaw[d] = hiData[s];
            tileRaw[d + 1] = hiData[s + 1];
            tileRaw[d + 2] = hiData[s + 2];
            // Mark as real fabric ONLY where the segmentation says garment.
            // The crop is a plain rectangle, so it also spans background
            // between the arms, skin at the neck, and whatever else is worn
            // under the hem — treating all of it as cloth is what painted the
            // studio backdrop onto the sleeves and an undershirt band across
            // the chest. Everything outside the mask is left for the inpainter
            // to fill from surrounding garment, and the skin-tone exclusion
            // still applies where it is trusted.
            const mx = clampInt(Math.floor(srcX / sx), 0, width - 1);
            const my = clampInt(Math.floor(srcY / sy), 0, height - 1);
            const isGarment =
              mask[my * width + mx] === 1 &&
              !(
                skinExclude &&
                skinExclude(hiData[s], hiData[s + 1], hiData[s + 2])
              );
            keep[ty * TILE + tx] = isGarment ? 1 : 0;
          }
        }
      }
    }
    inpaintTile(tileRaw, keep, TILE, [gr, gg, gb]);

    // Decompose into base fabric + print layer and recompose as clean flat
    // fabric with the graphics composited back — the pasted-photo look
    // (baked shading, wrinkles, gap seams) is replaced by uniform fabric.
    // Auto-skips for all-over patterned fabric (camo, plaid), where
    // flattening would erase the pattern; those keep the photo tile.
    const flat = flattenTile(tileRaw, TILE);
    const baseColor = flat.base;

    // Plain fabric swatch for the sleeves/collar; the base colour IS the
    // fabric colour now, so trims match the panels exactly.
    const { swatch, swatchSize, mean } = pickFabricSwatch(tileRaw, TILE);
    // The swatch patch can land in a shadowed corner of the photo, which
    // renders trims (turtleneck collar, cuffs) visibly darker than the body.
    // Scale it channel-by-channel so its mean matches the garment's median.
    {
      const kx = Math.min(2, Math.max(0.5, gr / Math.max(1, mean[0])));
      const ky = Math.min(2, Math.max(0.5, gg / Math.max(1, mean[1])));
      const kz = Math.min(2, Math.max(0.5, gb / Math.max(1, mean[2])));
      for (let i = 0; i < swatch.length; i += 3) {
        swatch[i] = clampInt(Math.round(swatch[i] * kx), 0, 255);
        swatch[i + 1] = clampInt(Math.round(swatch[i + 1] * ky), 0, 255);
        swatch[i + 2] = clampInt(Math.round(swatch[i + 2] * kz), 0, 255);
      }
    }
    const fabricColor = rgbToHex(
      Math.round(baseColor[0]),
      Math.round(baseColor[1]),
      Math.round(baseColor[2]),
    );

    // Whole-garment extraction: the mesh maps more than the torso panels, so
    // pull dedicated tiles for the other regions through the same warp.
    const regionTile = async (rMinX: number, rMaxX: number, rMask: Uint8Array) => {
      const region = warpGarmentTile({
        data,
        mask: rMask,
        width,
        height,
        minX: rMinX,
        maxX: rMaxX,
        tileSize: TILE,
        exclude: skinExclude,
        hiData,
        hiWidth: hiInfo.width,
        hiHeight: hiInfo.height,
      });
      // A narrow region (a sleeve seen edge-on with the arm hanging down) warps
      // into a tile with almost no real garment in it. Inpainting that fills
      // the whole tile with the flat median colour, which is why printed
      // sleeves rendered as solid brown. Report it as unusable so the caller
      // falls back to the torso tile — the same printed cloth.
      let covered = 0;
      for (let i = 0; i < TILE * TILE; i += 1) {
        covered += region.keep[i];
      }
      if (covered < TILE * TILE * 0.25) {
        return undefined;
      }
      inpaintTile(region.tileRaw, region.keep, TILE, [gr, gg, gb]);
      // Share the torso's base AND its flatten/keep decision, so every
      // region of the garment is treated as the same fabric.
      flattenTile(region.tileRaw, TILE, baseColor, flat.applied);
      let pipeline = sharp(region.tileRaw, {
        raw: { width: TILE, height: TILE, channels: 3 },
      });
      if (!flat.applied) {
        // Photo tiles (patterned fabric) keep the gentle local contrast so
        // washed prints stay visible; flattened tiles must stay uniform.
        pipeline = pipeline.clahe({ width: 192, height: 192, maxSlope: 1.15 });
      }
      pipeline = pipeline.modulate(FABRIC_TONE);
      const jpg = await pipeline.jpeg({ quality: 86 }).toBuffer();
      return "data:image/jpeg;base64," + jpg.toString("base64");
    };

    // Sleeves — extracted separately ONLY when they are decorated.
    //
    // A sleeve is usually cut from the same bolt as the body, so a dedicated
    // tile just re-samples the same cloth from a narrower, more foreshortened
    // strip: a worse likeness of the same print. It earns its place only when
    // the sleeve carries something the body does not — a contrast panel, a
    // stripe down the arm, a printed logo. Otherwise return nothing and let
    // the body print dress the sleeve, which is exactly right for an all-over
    // print like camo.
    let sleeveTextureUrl: string | undefined;
    {
      let gMinX = width;
      let gMaxX = 0;
      for (let x = 0; x < width; x += 1) {
        if (colHeight[x] > 0) {
          if (x < gMinX) gMinX = x;
          if (x > gMaxX) gMaxX = x;
        }
      }
      const garmentW = gMaxX - gMinX + 1;
      const area = (x0: number, x1: number) => {
        let sum = 0;
        for (let x = x0; x <= x1; x += 1) {
          sum += colHeight[x] ?? 0;
        }
        return sum;
      };

      // Locate the sleeve columns. When the arms are held away from the body
      // they fall outside the torso crop. When they hang down alongside it
      // those columns are as tall as the torso and get absorbed into the crop,
      // so fall back to the outer margin of the garment, which is where a
      // hanging sleeve sits.
      const outsideLeft = minX - gMinX;
      const outsideRight = gMaxX - maxX;
      const detached =
        outsideLeft >= garmentW * 0.08 || outsideRight >= garmentW * 0.08;
      const margin = Math.max(4, Math.round(garmentW * 0.15));
      const leftBand = detached
        ? { x0: gMinX, x1: Math.max(gMinX, minX - 1) }
        : { x0: gMinX, x1: Math.min(gMaxX, gMinX + margin) };
      const rightBand = detached
        ? { x0: Math.min(gMaxX, maxX + 1), x1: gMaxX }
        : { x0: Math.max(gMinX, gMaxX - margin), x1: gMaxX };
      const best =
        area(leftBand.x0, leftBand.x1) >= area(rightBand.x0, rightBand.x1)
          ? leftBand
          : rightBand;

      // Compare the sleeve's palette with the CENTRE of the torso, below the
      // neckline so a collar cannot skew it. Chromaticity bins make this
      // robust to the arm simply being more shaded than the chest.
      const bodyX0 = Math.round(minX + (maxX - minX) * 0.3);
      const bodyX1 = Math.round(minX + (maxX - minX) * 0.7);
      const bodyY0 = Math.round(minY + (maxY - minY) * 0.25);
      const sleeve = chromaHistogram(
        data,
        mask,
        width,
        best.x0,
        best.x1,
        minY,
        maxY,
      );
      const body = chromaHistogram(
        data,
        mask,
        width,
        bodyX0,
        bodyX1,
        bodyY0,
        maxY,
      );
      const enoughToJudge =
        sleeve.count > 400 && body.count > 400 &&
        area(best.x0, best.x1) >= maskCount * 0.03;
      if (enoughToJudge) {
        const shared = histogramIntersection(sleeve.hist, body.hist);
        // Same cloth overlaps almost completely. A decorated sleeve brings
        // colours the body does not have, dropping the overlap sharply.
        if (shared < 0.7) {
          sleeveTextureUrl = await regionTile(best.x0, best.x1, mask);
        }
      }
    }

    // Hood (only when the chosen template has one): the mask region above
    // the shoulder line. The shoulder line is where rows first reach most of
    // the torso width scanning down from the top of the garment.
    let hoodTextureUrl: string | undefined;
    if (options?.hood) {
      const torsoW = maxX - minX + 1;
      let shoulderY = -1;
      for (let y = minY; y <= maxY; y += 1) {
        let rowCount = 0;
        for (let x = minX; x <= maxX; x += 1) {
          rowCount += mask[y * width + x];
        }
        if (rowCount >= torsoW * 0.72) {
          shoulderY = y;
          break;
        }
      }
      if (shoulderY > minY && shoulderY - minY >= (maxY - minY) * 0.06) {
        const hoodMask = mask.slice();
        for (let y = shoulderY; y < height; y += 1) {
          for (let x = 0; x < width; x += 1) {
            hoodMask[y * width + x] = 0;
          }
        }
        let hLo = width;
        let hHi = 0;
        for (let y = minY; y < shoulderY; y += 1) {
          for (let x = 0; x < width; x += 1) {
            if (hoodMask[y * width + x]) {
              if (x < hLo) hLo = x;
              if (x > hHi) hHi = x;
            }
          }
        }
        if (hHi - hLo >= 12) {
          hoodTextureUrl = await regionTile(hLo, hHi, hoodMask);
        }
      }
    }

    let mainPipeline = sharp(tileRaw, {
      raw: { width: TILE, height: TILE, channels: 3 },
    });
    if (!flat.applied) {
      // Photo tile (patterned fabric): gentle local contrast so washed
      // prints stay visible. Flattened tiles must stay uniform — CLAHE
      // would blotch the flat base.
      mainPipeline = mainPipeline.clahe({ width: 192, height: 192, maxSlope: 1.15 });
    }
    mainPipeline = mainPipeline.modulate(FABRIC_TONE);
    const [tile, fabricTile] = await Promise.all([
      mainPipeline.jpeg({ quality: 86 }).toBuffer(),
      sharp(swatch, {
        raw: { width: swatchSize, height: swatchSize, channels: 3 },
      })
        .resize(256, 256, { fit: "fill" })
        .blur(1.2)
        .jpeg({ quality: 75 })
        .toBuffer(),
    ]);

    const mainTextureUrl = "data:image/jpeg;base64," + tile.toString("base64");
    return {
      color: fabricColor,
      textureUrl: mainTextureUrl,
      fabricTextureUrl:
        "data:image/jpeg;base64," + fabricTile.toString("base64"),
      // When no dedicated sleeve/hood region could be isolated, reuse the torso
      // tile rather than letting the viewer fall back to the plain swatch. A
      // photo of a worn garment with the arms hanging down has sleeve columns
      // nearly as tall as the torso, so they get absorbed into the torso crop
      // and no separate region survives — which rendered printed sleeves as
      // flat colour. The sleeves are cut from the same printed cloth, so the
      // torso tile is a far better likeness than a solid fill.
      sleeveTextureUrl: sleeveTextureUrl ?? mainTextureUrl,
      hoodTextureUrl: hoodTextureUrl ?? mainTextureUrl,
      shape,
    };
  } catch (error) {
    console.warn(
      "[garment-segmentation] inference failed, using colour heuristic:",
      error instanceof Error ? error.message : error,
    );
    return null;
  }
}
