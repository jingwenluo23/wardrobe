// In-memory store for uploaded garment drafts and their (simulated) mesh
// reconstruction pipeline. State resets when the dev server restarts, which is
// fine for local previewing — there is no external database to wire up.

import sharp from "sharp";

import { segmentGarment } from "./garment-segmentation";
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
  const toHex = (value: number) =>
    Math.round(value).toString(16).padStart(2, "0");
  const color = "#" + toHex(gr) + toHex(gg) + toHex(gb);

  // Mask pixels close to the garment colour, then take a robust bounding
  // box (3rd-97th percentile) so stray background matches don't blow it up.
  const xs: number[] = [];
  const ys: number[] = [];
  for (let y = 0; y < MASK_SIZE; y += 1) {
    for (let x = 0; x < MASK_SIZE; x += 1) {
      const [r, g, b] = pixelAt(x, y);
      const dist = Math.abs(r - gr) + Math.abs(g - gg) + Math.abs(b - gb);
      if (dist < 110) {
        xs.push(x);
        ys.push(y);
      }
    }
  }

  let region = { left: 0, top: 0, width: srcW, height: srcH };
  if (xs.length > MASK_SIZE) {
    const pct = (arr: number[], p: number) => {
      const sorted = [...arr].sort((a, b) => a - b);
      return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
    };
    const sx = srcW / MASK_SIZE;
    const sy = srcH / MASK_SIZE;
    const left = Math.max(0, Math.floor(pct(xs, 0.03) * sx));
    const right = Math.min(srcW, Math.ceil((pct(xs, 0.97) + 1) * sx));
    const top = Math.max(0, Math.floor(pct(ys, 0.03) * sy));
    const bottom = Math.min(srcH, Math.ceil((pct(ys, 0.97) + 1) * sy));
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

  for (let y = 0; y < TILE; y += 1) {
    const { left, right } = spans[y];
    const paintAll =
      y < shoulderRow || left === -1 || right - left < TILE * 0.3;
    for (let x = 0; x < TILE; x += 1) {
      const i = (y * TILE + x) * 3;
      if (
        paintAll ||
        x < left ||
        x > right ||
        isSkin(tileRaw[i], tileRaw[i + 1], tileRaw[i + 2])
      ) {
        tileRaw[i] = gr;
        tileRaw[i + 1] = gg;
        tileRaw[i + 2] = gb;
      }
    }
  }

  const tile = await sharp(tileRaw, {
    raw: { width: TILE, height: TILE, channels: 3 },
  })
    .jpeg({ quality: 78 })
    .toBuffer();
  const textureUrl = "data:image/jpeg;base64," + tile.toString("base64");

  return { color, textureUrl };
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

  // Preferred path: ML clothes segmentation (SegFormer). Falls back to the
  // colour-heuristic extractor when the model is unavailable.
  const frontSeg = await segmentGarment(front.buffer, input.category);
  const backSeg = await segmentGarment(back.buffer, input.category);
  const usedModel = Boolean(frontSeg);
  const { color, textureUrl } = frontSeg ?? (await analyzePhoto(front.buffer));
  const backTextureUrl = (backSeg ?? (await analyzePhoto(back.buffer)))
    .textureUrl;

  const photos: DraftPhoto[] = [
    { url: front.dataUrl, role: "front" },
    { url: back.dataUrl, role: "back" },
  ];
  if (side) {
    photos.push({ url: side.dataUrl, role: "side" });
  }

  // Model-based extraction is far more reliable than the colour heuristic;
  // extra reference views nudge confidence up either way.
  const segmentationConfidence = Math.min(
    0.99,
    (usedModel ? 0.93 : 0.78) + photos.length * 0.02,
  );

  const stored: StoredDraft = {
    id,
    name: input.name,
    category: input.category,
    color,
    createdAtMs: Date.now(),
    photos,
    params,
    template,
    segmentationConfidence,
    extractedTextureUrl: textureUrl,
    extractedBackTextureUrl: backTextureUrl,
  };

  drafts.set(id, stored);
  return serializeDraft(stored);
}

function serializeDraft(stored: StoredDraft): ApiDraft {
  const elapsed = Date.now() - stored.createdAtMs;
  const ratio = Math.min(1, elapsed / PIPELINE_MS);
  const progress = Math.round(ratio * 100);
  const isReady = ratio >= 1;

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
