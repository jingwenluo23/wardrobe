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

function labelsForCategory(category: string): Set<string> {
  const normalized = category.toLowerCase();
  if (normalized.startsWith("top") || normalized.startsWith("outer")) {
    return new Set(["Upper-clothes", "Dress"]);
  }
  if (normalized.startsWith("bottom")) {
    return new Set(["Pants", "Skirt", "Dress"]);
  }
  if (normalized.startsWith("shoe") || normalized.startsWith("sock")) {
    return new Set(["Left-shoe", "Right-shoe"]);
  }
  return CLOTHING_LABELS;
}

// Lazy singleton so the model loads once per server process. A failed load
// is remembered as null and never retried mid-process (callers fall back).
const globalForSegmenter = globalThis as unknown as {
  __wardrobeSegmenter?: Promise<Segmenter | null>;
};

function getSegmenter(): Promise<Segmenter | null> {
  if (!globalForSegmenter.__wardrobeSegmenter) {
    globalForSegmenter.__wardrobeSegmenter = (async () => {
      try {
        const { pipeline } = await import("@huggingface/transformers");
        const segmenter = await pipeline(
          "image-segmentation",
          "Xenova/segformer_b2_clothes",
          { dtype: "q8" },
        );
        return segmenter as unknown as Segmenter;
      } catch (error) {
        console.warn(
          "[garment-segmentation] model unavailable, using colour heuristic:",
          error instanceof Error ? error.message : error,
        );
        return null;
      }
    })();
  }
  return globalForSegmenter.__wardrobeSegmenter;
}

export type ExtractionResult = {
  color: string;
  textureUrl: string;
  /** Plain fabric swatch (no print) for sleeves/collar, from the photo. */
  fabricTextureUrl?: string;
};

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

const INFER_SIZE = 768;
const HI_SIZE = 1536;
const TILE = 512;

/**
 * Extract the garment from a photo with the segmentation model.
 * Returns null when the model is unavailable or finds no garment,
 * so the caller can fall back to the colour heuristic.
 */
export async function segmentGarment(
  buffer: Buffer,
  category: string,
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

    // Union of all masks whose label matches the garment category.
    const wanted = labelsForCategory(category);
    const mask = new Uint8Array(width * height);
    let maskCount = 0;
    for (const result of results) {
      if (!wanted.has(result.label)) {
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
    // Require a meaningful garment area (>2% of the frame).
    if (maskCount < width * height * 0.02) {
      return null;
    }

    // Clean the mask:
    // 1. Erode a few pixels so boundary bleed (arm skin hugging the side
    //    seam, background halos) cannot reach the texture.
    // 2. Keep only the largest connected component, dropping stray
    //    misclassified blobs (photo captions, hands near the hem).
    erodeMask(mask, width, height, 2);
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
      exclude: isSkinTone,
      hiData,
      hiWidth: hiInfo.width,
      hiHeight: hiInfo.height,
    });
    inpaintTile(tileRaw, keep, TILE, [gr, gg, gb]);

    // Plain fabric swatch for the sleeves/collar, and the base colour from
    // it — so untextured parts match the torso's real shading, not a median.
    const { swatch, swatchSize, mean } = pickFabricSwatch(tileRaw, TILE);
    const fabricColor = rgbToHex(mean[0], mean[1], mean[2]);

    const [tile, fabricTile] = await Promise.all([
      sharp(tileRaw, { raw: { width: TILE, height: TILE, channels: 3 } })
        // Gentle local contrast so washed/tonal prints stay clearly visible
        // on the mesh instead of sinking into the fabric colour.
        .clahe({ width: 128, height: 128, maxSlope: 2 })
        .jpeg({ quality: 86 })
        .toBuffer(),
      sharp(swatch, {
        raw: { width: swatchSize, height: swatchSize, channels: 3 },
      })
        .resize(256, 256, { fit: "fill" })
        .blur(1.2)
        .jpeg({ quality: 75 })
        .toBuffer(),
    ]);

    return {
      color: fabricColor,
      textureUrl: "data:image/jpeg;base64," + tile.toString("base64"),
      fabricTextureUrl:
        "data:image/jpeg;base64," + fabricTile.toString("base64"),
    };
  } catch (error) {
    console.warn(
      "[garment-segmentation] inference failed, using colour heuristic:",
      error instanceof Error ? error.message : error,
    );
    return null;
  }
}
