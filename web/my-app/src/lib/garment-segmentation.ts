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
  // Vertical pass: fill from the nearest kept pixel above/below, then a
  // horizontal pass for columns with no fabric at all.
  for (let x = 0; x < size; x += 1) {
    let lastY = -1;
    for (let y = 0; y < size; y += 1) {
      if (keep[y * size + x]) {
        if (lastY === -1 && y > 0) {
          // Backfill the run above the first kept pixel.
          for (let fy = 0; fy < y; fy += 1) {
            const ti = (fy * size + x) * 3;
            const si = (y * size + x) * 3;
            tileRaw[ti] = tileRaw[si];
            tileRaw[ti + 1] = tileRaw[si + 1];
            tileRaw[ti + 2] = tileRaw[si + 2];
          }
        }
        lastY = y;
      } else if (lastY !== -1) {
        const ti = (y * size + x) * 3;
        const si = (lastY * size + x) * 3;
        tileRaw[ti] = tileRaw[si];
        tileRaw[ti + 1] = tileRaw[si + 1];
        tileRaw[ti + 2] = tileRaw[si + 2];
      }
    }
    if (lastY === -1) {
      // Whole column empty: mark for the horizontal pass via fallback.
      for (let y = 0; y < size; y += 1) {
        const ti = (y * size + x) * 3;
        tileRaw[ti] = fallback[0];
        tileRaw[ti + 1] = fallback[1];
        tileRaw[ti + 2] = fallback[2];
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

/** Classic RGB skin-tone test, the last guard against skin on the texture. */
function isSkinTone(r: number, g: number, b: number) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  return (
    r > 95 &&
    g > 40 &&
    b > 20 &&
    max - min > 15 &&
    Math.abs(r - g) > 15 &&
    r > g &&
    r > b
  );
}

const INFER_SIZE = 768;
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
    const boxW = maxX - minX + 1;
    const boxH = maxY - minY + 1;

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

    // Build the texture tile: sample the bounding box into a TILE x TILE
    // image, then inpaint non-garment pixels (neck opening, seams, skin)
    // from the surrounding fabric so no ghost collar or flat patches appear.
    const tileRaw = Buffer.alloc(TILE * TILE * 3);
    const keep = new Uint8Array(TILE * TILE);
    for (let ty = 0; ty < TILE; ty += 1) {
      const sy = minY + Math.min(boxH - 1, Math.floor((ty / TILE) * boxH));
      for (let tx = 0; tx < TILE; tx += 1) {
        const sx = minX + Math.min(boxW - 1, Math.floor((tx / TILE) * boxW));
        const si = sy * width + sx;
        const ti = (ty * TILE + tx) * 3;
        const r = data[si * 3];
        const g = data[si * 3 + 1];
        const b = data[si * 3 + 2];
        tileRaw[ti] = r;
        tileRaw[ti + 1] = g;
        tileRaw[ti + 2] = b;
        if (mask[si] && !isSkinTone(r, g, b)) {
          keep[ty * TILE + tx] = 1;
        }
      }
    }
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
        .jpeg({ quality: 80 })
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
