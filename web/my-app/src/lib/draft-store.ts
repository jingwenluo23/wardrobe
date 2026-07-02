// In-memory store for uploaded garment drafts and their (simulated) mesh
// reconstruction pipeline. State resets when the dev server restarts, which is
// fine for local previewing — there is no external database to wire up.

import sharp from "sharp";

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

async function analyzePhoto(buffer: Buffer): Promise<{
  color: string;
  textureUrl: string;
}> {
  // Dominant colour via sharp's stats, and a small isolated texture tile
  // (a centre crop) projected onto the mesh. No faces are reproduced.
  const image = sharp(buffer).rotate();
  const stats = await image.stats();
  const [r, g, b] = stats.channels;
  const toHex = (value: number) =>
    Math.round(value).toString(16).padStart(2, "0");
  const color =
    "#" + toHex(r?.mean ?? 180) + toHex(g?.mean ?? 180) + toHex(b?.mean ?? 180);

  // Crop the chest band of the photo (centre-horizontal, upper-middle
  // vertically) so the texture picks up fabric rather than the wearer's
  // face or the background.
  const meta = await sharp(buffer).rotate().metadata();
  const srcW = meta.width ?? 512;
  const srcH = meta.height ?? 512;
  const cropW = Math.max(16, Math.round(srcW * 0.36));
  const cropH = Math.max(16, Math.round(srcH * 0.26));
  const tile = await sharp(buffer)
    .rotate()
    .extract({
      left: Math.round((srcW - cropW) / 2),
      top: Math.round(srcH * 0.38),
      width: cropW,
      height: Math.min(cropH, srcH - Math.round(srcH * 0.38)),
    })
    .resize(256, 256, { fit: "cover" })
    .jpeg({ quality: 72 })
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

  const { color, textureUrl } = await analyzePhoto(front.buffer);

  const photos: DraftPhoto[] = [
    { url: front.dataUrl, role: "front" },
    { url: back.dataUrl, role: "back" },
  ];
  if (side) {
    photos.push({ url: side.dataUrl, role: "side" });
  }

  // More reference views -> higher extraction confidence.
  const segmentationConfidence = Math.min(0.99, 0.78 + photos.length * 0.05);

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
