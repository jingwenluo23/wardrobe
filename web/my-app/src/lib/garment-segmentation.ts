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
};

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

    // Tight bounding box of the mask.
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
    const toHex = (value: number) =>
      Math.round(value).toString(16).padStart(2, "0");
    const color = "#" + toHex(gr) + toHex(gg) + toHex(gb);

    // Build the texture tile: sample the bounding box into a TILE x TILE
    // image; pixels outside the garment mask become plain fabric colour.
    const tileRaw = Buffer.alloc(TILE * TILE * 3);
    for (let ty = 0; ty < TILE; ty += 1) {
      const sy = minY + Math.min(boxH - 1, Math.floor((ty / TILE) * boxH));
      for (let tx = 0; tx < TILE; tx += 1) {
        const sx = minX + Math.min(boxW - 1, Math.floor((tx / TILE) * boxW));
        const si = sy * width + sx;
        const ti = (ty * TILE + tx) * 3;
        if (mask[si]) {
          tileRaw[ti] = data[si * 3];
          tileRaw[ti + 1] = data[si * 3 + 1];
          tileRaw[ti + 2] = data[si * 3 + 2];
        } else {
          tileRaw[ti] = gr;
          tileRaw[ti + 1] = gg;
          tileRaw[ti + 2] = gb;
        }
      }
    }

    const tile = await sharp(tileRaw, {
      raw: { width: TILE, height: TILE, channels: 3 },
    })
      .jpeg({ quality: 80 })
      .toBuffer();

    return {
      color,
      textureUrl: "data:image/jpeg;base64," + tile.toString("base64"),
    };
  } catch (error) {
    console.warn(
      "[garment-segmentation] inference failed, using colour heuristic:",
      error instanceof Error ? error.message : error,
    );
    return null;
  }
}
