// In-memory store for uploaded garment drafts and their (simulated) mesh
// reconstruction pipeline. State resets when the dev server restarts, which is
// fine for local previewing — there is no external database to wire up.

import sharp from "sharp";

import {
  inpaintTile,
  pickFabricSwatch,
  rgbToHex,
  segmentGarment,
} from "./garment-segmentation";
import {
  boundsFromParams,
  defaultTeeParams,
  type DraftMesh,
  type GarmentParams,
  type GarmentTemplate,
} from "./garment-mesh";

export type DraftPipelineStatus = "processing" | "ready" | "failed";
export type DraftStageStatus = "pending" | "active" | "done";
export type PhotoRole = "front" | "back" | "side";

export type DraftStage = {
  id: string;
  label: string;
  status: DraftStageStatus;
};

export type DraftPhoto = {
  url: string;
  role: PhotoRole;
};

export type ApiDraft = {
  id: string;
  name: string;
  category: string;
  color: string;
  status: DraftPipelineStatus;
  progress: number;
  currentStage: string;
  createdAt: string;
  photos: DraftPhoto[];
  stages: DraftStage[];
  mesh?: DraftMesh;
  error?: string;
};

type StoredDraft = {
  id: string;
  name: string;
  category: string;
  color: string;
  createdAtMs: number;
  photos: DraftPhoto[];
  params: GarmentParams;
  template: GarmentTemplate;
  segmentationConfidence: number;
  extractedTextureUrl?: string;
  extractedBackTextureUrl?: string;
  fabricTextureUrl?: string;
  /** False while the background extraction task is still running. */
  extractionReady: boolean;
};

const STAGE_LABELS = [
  "Extract garment region",
  "Build standard base mesh",
  "Fit neutral garment proportions",
  "Project garment texture",
] as const;

// Total time for the simulated pipeline to reach "ready".
const PIPELINE_MS = 3600;

// Use a module-level singleton that survives Next.js hot reloads in dev.
const globalForDrafts = globalThis as unknown as {
  __wardrobeDrafts?: Map<string, StoredDraft>;
};

const drafts: Map<string, StoredDraft> =
  globalForDrafts.__wardrobeDrafts ?? new Map();
globalForDrafts.__wardrobeDrafts = drafts;

function createId() {
  return "draft_" + Math.random().toString(36).slice(2, 10);
}

function templateForCategory(category: string): GarmentTemplate {
  const normalized = category.toLowerCase();
  if (normalized.startsWith("outer")) {
    return "outerwear-boxy";
  }
  if (normalized.startsWith("bottom")) {
    return "bottom-straight";
  }
  if (normalized.startsWith("shoe")) {
    return "shoe-low";
  }
  return "top-standard-tee";
}

function paramsForTemplate(template: GarmentTemplate): GarmentParams {
  switch (template) {
    case "outerwear-boxy":
      return {
        ...defaultTeeParams,
        bodyWidth: 62,
        bodyDepth: 24,
        sleeveLength: 30,
        armholeDepth: 26,
      };
    case "top-fitted":
      return {
        ...defaultTeeParams,
        bodyWidth: 48,
        bodyDepth: 16,
        neckWidthFront: 17,
      };
    default:
      return { ...defaultTeeParams };
  }
}

const MASK_SIZE = 96;

async function analyzePhoto(buffer: Buffer): Promise<{
  color: string;
  textureUrl: string;
  fabricTextureUrl?: string;
}> {
  // Colour-based garment extraction:
  // 1. Sample the garment colour from the chest area (centre-horizontal,
  //    mid-height — where the garment sits when worn or laid flat).
  // 2. Mask every pixel whose colour is close to that sample.
  // 3. Take a robust bounding box of the mask and crop the original photo
  //    to it, so the texture is the actual garment front/back, not the
  //    wearer's skin or the background.
  const meta = await sharp(buffer).rotate().metadata();
  const srcW = meta.width ?? 512;
  const srcH = meta.height ?? 512;

  const raw = await sharp(buffer)
    .rotate()
    .resize(MASK_SIZE, MASK_SIZE, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer();

  const pixelAt = (x: number, y: number) => {
    const i = (y * MASK_SIZE + x) * 3;
    return [raw[i], raw[i + 1], raw[i + 2]] as const;
  };

  // Reference colour: median of the chest patch (40-60% wide, 42-60% tall).
  const rs: number[] = [];
  const gs: number[] = [];
  const bs: number[] = [];
  for (let y = Math.floor(MASK_SIZE * 0.42); y < MASK_SIZE * 0.6; y += 1) {
    for (let x = Math.floor(MASK_SIZE * 0.4); x < MASK_SIZE * 0.6; x += 1) {
      const [r, g, b] = pixelAt(x, y);
      rs.push(r);
      gs.push(g);
      bs.push(b);
    }
  }
  const median = (arr: number[]) => {
    const sorted = [...arr].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)] ?? 128;
  };
  const gr = median(rs);
  const gg = median(gs);
  const gb = median(bs);

  // Mask pixels close to the garment colour. The crop is then restricted to
  // torso columns (tall mask columns) so the sleeves don't stretch the
  // texture: the mesh panels only span side seam to side seam, and matching
  // that region keeps print placement close to 1:1 with the real shirt.
  const colCount = new Array<number>(MASK_SIZE).fill(0);
  const ys: number[] = [];
  const maskGrid = new Uint8Array(MASK_SIZE * MASK_SIZE);
  for (let y = 0; y < MASK_SIZE; y += 1) {
    for (let x = 0; x < MASK_SIZE; x += 1) {
      const [r, g, b] = pixelAt(x, y);
      const dist = Math.abs(r - gr) + Math.abs(g - gg) + Math.abs(b - gb);
      if (dist < 110) {
        maskGrid[y * MASK_SIZE + x] = 1;
        colCount[x] += 1;
      }
    }
  }
  const maxColCount = Math.max(...colCount);
  let colLeft = -1;
  let colRight = -1;
  for (let x = 0; x < MASK_SIZE; x += 1) {
    if (colCount[x] >= maxColCount * 0.55) {
      if (colLeft === -1) {
        colLeft = x;
      }
      colRight = x;
    }
  }
  for (let y = 0; y < MASK_SIZE; y += 1) {
    for (let x = Math.max(0, colLeft); x <= colRight; x += 1) {
      if (maskGrid[y * MASK_SIZE + x]) {
        ys.push(y);
        break;
      }
    }
  }

  let region = { left: 0, top: 0, width: srcW, height: srcH };
  if (colLeft !== -1 && ys.length > 4) {
    const pct = (arr: number[], p: number) => {
      const sorted = [...arr].sort((a, b) => a - b);
      return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
    };
    const sx = srcW / MASK_SIZE;
    const sy = srcH / MASK_SIZE;
    const left = Math.max(0, Math.floor(colLeft * sx));
    const right = Math.min(srcW, Math.ceil((colRight + 1) * sx));
    const top = Math.max(0, Math.floor(pct(ys, 0.02) * sy));
    const bottom = Math.min(srcH, Math.ceil((pct(ys, 0.98) + 1) * sy));
    if (right - left > srcW * 0.12 && bottom - top > srcH * 0.12) {
      region = { left, top, width: right - left, height: bottom - top };
    }
  }

  // Crop to the garment box, then paint over anything that is not garment:
  // for each row, everything outside the leftmost..rightmost garment span
  // becomes plain fabric colour, and rows whose garment span is too narrow
  // (the head/neck poking above the collar, arms below the sleeves) are
  // filled entirely. Pixels between the span edges are kept, so prints and
  // graphics survive even though their colours differ from the base fabric.
  const TILE = 512;
  const tileRaw = await sharp(buffer)
    .rotate()
    .extract(region)
    .resize(TILE, TILE, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer();

  const spans: Array<{ left: number; right: number }> = [];
  for (let y = 0; y < TILE; y += 1) {
    let left = -1;
    let right = -1;
    for (let x = 0; x < TILE; x += 1) {
      const i = (y * TILE + x) * 3;
      const dist =
        Math.abs(tileRaw[i] - gr) +
        Math.abs(tileRaw[i + 1] - gg) +
        Math.abs(tileRaw[i + 2] - gb);
      if (dist < 110) {
        if (left === -1) {
          left = x;
        }
        right = x;
      }
    }
    spans.push({ left, right });
  }

  // The garment's top edge is its shoulders — the widest structure in the
  // photo. A head/neck poking above the collar (even hair whose colour is
  // close to the fabric) is far narrower, so everything above the first
  // shoulder-wide row gets painted out entirely.
  let shoulderRow = 0;
  while (
    shoulderRow < TILE &&
    spans[shoulderRow].right - spans[shoulderRow].left < TILE * 0.55
  ) {
    shoulderRow += 1;
  }

  // Classic RGB skin-tone test: catches faces/necks/hands that survive the
  // span rules (e.g. a chin overlapping the shoulder line).
  const isSkin = (r: number, g: number, b: number) => {
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
  };

  const keep = new Uint8Array(TILE * TILE);
  for (let y = 0; y < TILE; y += 1) {
    const { left, right } = spans[y];
    const paintAll =
      y < shoulderRow || left === -1 || right - left < TILE * 0.3;
    for (let x = 0; x < TILE; x += 1) {
      const i = (y * TILE + x) * 3;
      if (
        !paintAll &&
        x >= left &&
        x <= right &&
        !isSkin(tileRaw[i], tileRaw[i + 1], tileRaw[i + 2])
      ) {
        keep[y * TILE + x] = 1;
      }
    }
  }
  // Fill removed regions from the surrounding fabric instead of a flat
  // colour, so they blend into the shirt's real shading.
  inpaintTile(tileRaw, keep, TILE, [gr, gg, gb]);

  // Plain fabric swatch for the sleeves/collar; its mean becomes the base
  // colour so untextured parts match the torso's real shading.
  const { swatch, swatchSize, mean } = pickFabricSwatch(tileRaw, TILE);

  const [tile, fabricTile] = await Promise.all([
    sharp(tileRaw, { raw: { width: TILE, height: TILE, channels: 3 } })
      // Gentle local contrast so prints stay clearly visible on the mesh.
      .clahe({ width: 128, height: 128, maxSlope: 2 })
      .jpeg({ quality: 78 })
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
    color: rgbToHex(mean[0], mean[1], mean[2]),
    textureUrl: "data:image/jpeg;base64," + tile.toString("base64"),
    fabricTextureUrl:
      "data:image/jpeg;base64," + fabricTile.toString("base64"),
  };
}

async function fileToDataUrl(file: File): Promise<{
  buffer: Buffer;
  dataUrl: string;
}> {
  const buffer = Buffer.from(await file.arrayBuffer());
  const mime = file.type || "image/jpeg";
  const dataUrl = "data:" + mime + ";base64," + buffer.toString("base64");
  return { buffer, dataUrl };
}

export async function createDraft(input: {
  name: string;
  category: string;
  frontPhoto: File;
  backPhoto: File;
  sidePhoto?: File | null;
}): Promise<ApiDraft> {
  const id = createId();
  const template = templateForCategory(input.category);
  const params = paramsForTemplate(template);

  const front = await fileToDataUrl(input.frontPhoto);
  const back = await fileToDataUrl(input.backPhoto);
  const side = input.sidePhoto ? await fileToDataUrl(input.sidePhoto) : null;

  const photos: DraftPhoto[] = [
    { url: front.dataUrl, role: "front" },
    { url: back.dataUrl, role: "back" },
  ];
  if (side) {
    photos.push({ url: side.dataUrl, role: "side" });
  }

  // Provisional colour so the card renders immediately; the extraction task
  // replaces it with the true garment colour.
  const stats = await sharp(front.buffer).rotate().stats();
  const toHex = (value: number) =>
    Math.round(value).toString(16).padStart(2, "0");
  const provisionalColor =
    "#" +
    toHex(stats.channels[0]?.mean ?? 150) +
    toHex(stats.channels[1]?.mean ?? 150) +
    toHex(stats.channels[2]?.mean ?? 150);

  const stored: StoredDraft = {
    id,
    name: input.name,
    category: input.category,
    color: provisionalColor,
    createdAtMs: Date.now(),
    photos,
    params,
    template,
    segmentationConfidence: 0.8,
    extractionReady: false,
  };

  drafts.set(id, stored);

  // Extraction runs in the background so the upload request returns right
  // away; the outfit card's progress bar covers the wait and the mesh flips
  // to ready once both the timer and the extraction are done.
  void (async () => {
    try {
      // Preferred path: ML clothes segmentation (SegFormer). Falls back to
      // the colour-heuristic extractor when the model is unavailable.
      const frontSeg = await segmentGarment(front.buffer, input.category);
      const backSeg = await segmentGarment(back.buffer, input.category);
      const usedModel = Boolean(frontSeg);
      const frontResult = frontSeg ?? (await analyzePhoto(front.buffer));
      const backResult = backSeg ?? (await analyzePhoto(back.buffer));

      stored.color = frontResult.color;
      stored.extractedTextureUrl = frontResult.textureUrl;
      stored.extractedBackTextureUrl = backResult.textureUrl;
      stored.fabricTextureUrl = frontResult.fabricTextureUrl;
      // Model-based extraction is far more reliable than the heuristic;
      // extra reference views nudge confidence up either way.
      stored.segmentationConfidence = Math.min(
        0.99,
        (usedModel ? 0.93 : 0.78) + photos.length * 0.02,
      );
    } catch (error) {
      console.warn(
        "[draft-store] extraction failed, keeping plain colour:",
        error instanceof Error ? error.message : error,
      );
    } finally {
      stored.extractionReady = true;
    }
  })();

  return serializeDraft(stored);
}

function serializeDraft(stored: StoredDraft): ApiDraft {
  const elapsed = Date.now() - stored.createdAtMs;
  const ratio = Math.min(1, elapsed / PIPELINE_MS);
  // The mesh is ready when the staged timer has run its course AND the
  // background extraction has finished (the model may still be downloading
  // on its very first run). Progress holds at 90% while extraction runs.
  const isReady = ratio >= 1 && stored.extractionReady;
  const progress = isReady
    ? 100
    : Math.min(stored.extractionReady ? 99 : 90, Math.round(ratio * 100));

  const totalStages = STAGE_LABELS.length;
  const activeIndex = isReady
    ? totalStages
    : Math.min(totalStages - 1, Math.floor(ratio * totalStages));

  const stages: DraftStage[] = STAGE_LABELS.map((label, index) => ({
    id: "stage-" + index,
    label,
    status:
      index < activeIndex ? "done" : index === activeIndex ? "active" : "pending",
  }));

  const status: DraftPipelineStatus = isReady ? "ready" : "processing";
  const currentStage = isReady
    ? "Mesh ready"
    : STAGE_LABELS[activeIndex] ?? STAGE_LABELS[0];

  const mesh: DraftMesh | undefined = isReady
    ? {
        assetUrl: "mem://" + stored.id,
        generatedAt: stored.createdAtMs + PIPELINE_MS,
        template: stored.template,
        params: stored.params,
        segmentation: { confidence: stored.segmentationConfidence },
        extractedTextureUrl: stored.extractedTextureUrl,
        extractedBackTextureUrl: stored.extractedBackTextureUrl,
        fabricTextureUrl: stored.fabricTextureUrl,
        color: stored.color,
        bounds: boundsFromParams(stored.params),
      }
    : undefined;

  return {
    id: stored.id,
    name: stored.name,
    category: stored.category,
    color: stored.color,
    status,
    progress,
    currentStage,
    createdAt: new Date(stored.createdAtMs).toISOString(),
    photos: stored.photos,
    stages,
    mesh,
  };
}

export function listDrafts(): ApiDraft[] {
  return Array.from(drafts.values())
    .sort((a, b) => b.createdAtMs - a.createdAtMs)
    .map(serializeDraft);
}

export function deleteDraft(id: string): boolean {
  return drafts.delete(id);
}
