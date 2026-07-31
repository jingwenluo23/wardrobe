// SQLite-backed store for uploaded garment drafts and their (simulated)
// mesh reconstruction pipeline. Drafts persist across dev-server restarts in
// data/wardrobe.db (gitignored).

import fs from "node:fs";
import path from "node:path";

import Database from "better-sqlite3";
import sharp from "sharp";

import {
  inpaintTile,
  isSkinTone,
  pickFabricSwatch,
  rgbToHex,
  segmentGarment,
} from "./garment-segmentation";
import {
  boundsFromParams,
  type DraftMesh,
  type GarmentFeatures,
  type GarmentParams,
} from "./garment-mesh";
import { fitGarmentToShape } from "./garment-fit";
import {
  GARMENT_TEMPLATE_VERSION,
  getTemplate,
  resolveTemplate,
  templateLabel,
} from "./garment-templates";
import {
  blendSideIntoPanel,
  synthesizeSideTexture,
} from "./side-texture";

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
  features: GarmentFeatures;
  templateId: string;
  templateVersion: number;
  segmentationConfidence: number;
  extractedTextureUrl?: string;
  extractedBackTextureUrl?: string;
  fabricTextureUrl?: string;
  sleeveTextureUrl?: string;
  hoodTextureUrl?: string;
  extractedSideTextureUrl?: string;
  sideTextureMode?: "photo" | "synthesized";
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

// SQLite handle as a module-level singleton that survives Next.js hot
// reloads in dev.
const globalForDrafts = globalThis as unknown as {
  __wardrobeDb?: Database.Database;
};

function getDb(): Database.Database {
  if (!globalForDrafts.__wardrobeDb) {
    const dataDir = path.join(process.cwd(), "data");
    fs.mkdirSync(dataDir, { recursive: true });
    const db = new Database(path.join(dataDir, "wardrobe.db"));
    db.pragma("journal_mode = WAL");
    db.exec(
      "CREATE TABLE IF NOT EXISTS drafts (" +
        "id TEXT PRIMARY KEY," +
        "created_at INTEGER NOT NULL," +
        "name TEXT NOT NULL," +
        "category TEXT NOT NULL," +
        "template_id TEXT NOT NULL," +
        "template_version INTEGER NOT NULL," +
        "color TEXT NOT NULL," +
        "confidence REAL NOT NULL," +
        "extraction_ready INTEGER NOT NULL," +
        "params_json TEXT NOT NULL," +
        "features_json TEXT NOT NULL," +
        "photos_json TEXT NOT NULL," +
        "front_texture TEXT," +
        "back_texture TEXT," +
        "fabric_texture TEXT," +
        "sleeve_texture TEXT," +
        "hood_texture TEXT," +
        "side_texture TEXT," +
        "side_texture_mode TEXT" +
        ")",
    );
    // Older databases predate the regional texture columns.
    for (const column of [
      "sleeve_texture",
      "hood_texture",
      "side_texture",
      "side_texture_mode",
    ]) {
      try {
        db.exec("ALTER TABLE drafts ADD COLUMN " + column + " TEXT");
      } catch {
        // Column already exists.
      }
    }
    // Recover drafts whose background extraction was lost to a server
    // restart: surface them as ready with whatever they have.
    db.prepare(
      "UPDATE drafts SET extraction_ready = 1 " +
        "WHERE extraction_ready = 0 AND created_at < ?",
    ).run(Date.now() - 60_000);
    globalForDrafts.__wardrobeDb = db;
  }
  return globalForDrafts.__wardrobeDb;
}

type DraftRow = {
  id: string;
  created_at: number;
  name: string;
  category: string;
  template_id: string;
  template_version: number;
  color: string;
  confidence: number;
  extraction_ready: number;
  params_json: string;
  features_json: string;
  photos_json: string;
  front_texture: string | null;
  back_texture: string | null;
  fabric_texture: string | null;
  sleeve_texture: string | null;
  hood_texture: string | null;
  side_texture: string | null;
  side_texture_mode: string | null;
};

function rowToStored(row: DraftRow): StoredDraft {
  return {
    id: row.id,
    name: row.name,
    category: row.category,
    color: row.color,
    createdAtMs: row.created_at,
    photos: JSON.parse(row.photos_json) as DraftPhoto[],
    params: JSON.parse(row.params_json) as GarmentParams,
    features: JSON.parse(row.features_json) as GarmentFeatures,
    templateId: row.template_id,
    templateVersion: row.template_version,
    segmentationConfidence: row.confidence,
    extractedTextureUrl: row.front_texture ?? undefined,
    extractedBackTextureUrl: row.back_texture ?? undefined,
    fabricTextureUrl: row.fabric_texture ?? undefined,
    sleeveTextureUrl: row.sleeve_texture ?? undefined,
    hoodTextureUrl: row.hood_texture ?? undefined,
    extractedSideTextureUrl: row.side_texture ?? undefined,
    sideTextureMode:
      row.side_texture_mode === "photo" ||
      row.side_texture_mode === "synthesized"
        ? row.side_texture_mode
        : undefined,
    extractionReady: row.extraction_ready === 1,
  };
}

function createId() {
  return "draft_" + Math.random().toString(36).slice(2, 10);
}

const MASK_SIZE = 96;

/**
 * Last-resort garment texture: a plain central chest crop of the photo. It
 * keeps the head (top) and most of the background (sides) out of frame, so for
 * a worn or laid-flat top the crop is almost entirely fabric. Unlike the
 * segmentation and colour-heuristic extractors it does no masking that can come
 * up empty — a valid photo always yields a texture — so it guarantees the mesh
 * is never left blank when the smarter steps fail or time out.
 */
async function chestCropTexture(buffer: Buffer): Promise<string | undefined> {
  try {
    const meta = await sharp(buffer).rotate().metadata();
    const w = meta.width ?? 0;
    const h = meta.height ?? 0;
    if (w < 8 || h < 8) {
      return undefined;
    }
    const left = Math.round(w * 0.18);
    const top = Math.round(h * 0.38);
    const width = Math.max(1, Math.min(w - left, Math.round(w * 0.64)));
    const height = Math.max(1, Math.min(h - top, Math.round(h * 0.5)));
    const jpg = await sharp(buffer)
      .rotate()
      .extract({ left, top, width, height })
      .resize(512, 512, { fit: "fill" })
      .jpeg({ quality: 82 })
      .toBuffer();
    return "data:image/jpeg;base64," + jpg.toString("base64");
  } catch {
    return undefined;
  }
}

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
        !isSkinTone(tileRaw[i], tileRaw[i + 1], tileRaw[i + 2])
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
  templateId?: string | null;
  frontPhoto: File;
  backPhoto: File;
  sidePhoto?: File | null;
}): Promise<ApiDraft> {
  const id = createId();
  const template = resolveTemplate(input.templateId, input.category);
  const params = { ...template.params };
  const features = { ...template.features };

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

  // Seed the panels with a guaranteed chest crop synchronously, and persist it
  // in the initial row. The background extraction below only ever upgrades
  // these — so even if that task never completes or fails in this deployment,
  // the mesh still shows the garment's real fabric instead of a blank base.
  const [seedFront, seedBack] = await Promise.all([
    chestCropTexture(front.buffer),
    chestCropTexture(back.buffer),
  ]);
  const seededFront = seedFront ?? seedBack;
  const seededBack = seedBack ?? seedFront;

  const stored: StoredDraft = {
    id,
    name: input.name,
    category: input.category,
    color: provisionalColor,
    createdAtMs: Date.now(),
    photos,
    params,
    features,
    templateId: template.id,
    templateVersion: GARMENT_TEMPLATE_VERSION,
    segmentationConfidence: 0.8,
    extractedTextureUrl: seededFront,
    extractedBackTextureUrl: seededBack,
    extractionReady: false,
  };

  getDb()
    .prepare(
      "INSERT INTO drafts (id, created_at, name, category, template_id, " +
        "template_version, color, confidence, extraction_ready, params_json, " +
        "features_json, photos_json, front_texture, back_texture) " +
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
    )
    .run(
      stored.id,
      stored.createdAtMs,
      stored.name,
      stored.category,
      stored.templateId,
      stored.templateVersion,
      stored.color,
      stored.segmentationConfidence,
      0,
      JSON.stringify(stored.params),
      JSON.stringify(stored.features),
      JSON.stringify(stored.photos),
      stored.extractedTextureUrl ?? null,
      stored.extractedBackTextureUrl ?? null,
    );

  // Extraction runs in the background so the upload request returns right
  // away; the outfit card's progress bar covers the wait and the mesh flips
  // to ready once both the timer and the extraction are done.
  void (async () => {
    let update: {
      color?: string;
      front?: string;
      back?: string;
      fabric?: string;
      sleeve?: string;
      hood?: string;
      side?: string;
      sideMode?: "photo" | "synthesized";
      confidence?: number;
      params?: GarmentParams;
      features?: GarmentFeatures;
    } = {};
    try {
      // Guaranteed baseline: a plain central chest crop of each photo, applied
      // BEFORE any of the segmentation/synthesis work runs. If that work throws
      // or hits the timeout below, this crop survives so the garment texture
      // still appears instead of a blank mesh; every step after it only ever
      // upgrades these panels to the cleaner isolated version.
      const [fallbackFront, fallbackBack] = await Promise.all([
        chestCropTexture(front.buffer),
        chestCropTexture(back.buffer),
      ]);
      if (fallbackFront || fallbackBack) {
        update = {
          color: stored.color,
          front: fallbackFront,
          back: fallbackBack ?? fallbackFront,
          confidence: 0.5,
        };
      }

      // Preferred path: ML clothes segmentation (SegFormer). Falls back to
      // the colour-heuristic extractor when the model is unavailable. The
      // whole thing races a timeout so a hung model download or wedged
      // decode can never leave the draft "Building..." forever.
      const wantsHood = features.neckFinish === "hood";
      const segOptions = { hood: wantsHood };
      const extract = async () => {
        const frontSeg = await segmentGarment(front.buffer, input.category, segOptions);
        const backSeg = await segmentGarment(back.buffer, input.category, segOptions);
        const sideSeg = side
          ? await segmentGarment(side.buffer, input.category, segOptions)
          : null;
        const usedModel = Boolean(frontSeg);
        const frontResult = frontSeg ?? (await analyzePhoto(front.buffer));
        const backResult = backSeg ?? (await analyzePhoto(back.buffer));
        const sideResult = side
          ? sideSeg ?? (await analyzePhoto(side.buffer))
          : null;
        return {
          frontSeg,
          backSeg,
          usedModel,
          frontResult,
          backResult,
          sideResult,
        };
      };
      let timer: ReturnType<typeof setTimeout> | undefined;
      const timeout = new Promise<never>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error("extraction timed out")),
          EXTRACTION_DEADLINE_MS,
        );
      });
      const {
        frontSeg,
        backSeg,
        usedModel,
        frontResult,
        backResult,
        sideResult,
      } =
        await Promise.race([extract(), timeout]).finally(() => {
          if (timer) {
            clearTimeout(timer);
          }
        });

      // Baseline: the raw extracted (or heuristic) panel textures. These are
      // committed first so that a failure in any of the optional enhancement
      // steps below (side synthesis, edge blend, shape fit) can never discard
      // the real garment texture and leave a blank mesh — the enhancement
      // steps only ever UPGRADE this baseline.
      update = {
        color: frontResult.color,
        front: frontResult.textureUrl,
        back: backResult.textureUrl,
        fabric: frontResult.fabricTextureUrl,
        sleeve: frontSeg?.sleeveTextureUrl ?? backSeg?.sleeveTextureUrl,
        // The hood is best seen from behind; prefer the back photo's region.
        hood: backSeg?.hoodTextureUrl ?? frontSeg?.hoodTextureUrl,
        params,
        features,
        // Model-based extraction is far more reliable than the heuristic;
        // extra reference views nudge confidence up either way.
        confidence: Math.min(
          0.99,
          (usedModel ? 0.84 : 0.78) + photos.length * 0.02,
        ),
      };

      // Best-effort: bake a synthesized/photographed side band into the panel
      // edges and refine the silhouette from the segmentation. Isolated so a
      // throw here degrades to the baseline panels rather than a blank mesh.
      try {
        const sideTextureUrl =
          sideResult?.textureUrl ??
          (await synthesizeSideTexture({
            frontTextureUrl: frontResult.textureUrl,
            backTextureUrl: backResult.textureUrl,
            fabricTextureUrl: frontResult.fabricTextureUrl,
          }));
        const [projectedFront, projectedBack] = await Promise.all([
          blendSideIntoPanel({
            panelTextureUrl: frontResult.textureUrl,
            sideTextureUrl,
            surface: "front",
          }),
          blendSideIntoPanel({
            panelTextureUrl: backResult.textureUrl,
            sideTextureUrl,
            surface: "back",
          }),
        ]);
        const fitted = fitGarmentToShape(
          params,
          features,
          frontSeg?.shape,
          backSeg?.shape,
        );
        update.front = projectedFront;
        update.back = projectedBack;
        update.side = sideTextureUrl;
        update.sideMode = sideResult ? "photo" : "synthesized";
        update.params = fitted.params;
        update.features = fitted.features;
        update.confidence = Math.min(
          0.99,
          (usedModel ? 0.84 + fitted.confidence * 0.1 : 0.78) +
            photos.length * 0.02,
        );
      } catch (error) {
        console.warn(
          "[draft-store] side/fit enhancement skipped, keeping panels:",
          error instanceof Error ? error.message : error,
        );
      }
    } catch (error) {
      console.warn(
        "[draft-store] extraction failed, keeping plain colour:",
        error instanceof Error ? error.message : error,
      );
    } finally {
      try {
        getDb()
          .prepare(
            "UPDATE drafts SET extraction_ready = 1, " +
              "color = COALESCE(?, color), " +
              "confidence = COALESCE(?, confidence), " +
              // COALESCE so a failed/partial extraction (null) keeps the
              // synchronous chest-crop seed instead of blanking the panels.
              "front_texture = COALESCE(?, front_texture), " +
              "back_texture = COALESCE(?, back_texture), " +
              "fabric_texture = ?, " +
              "sleeve_texture = ?, hood_texture = ?, side_texture = ?, " +
              "side_texture_mode = ?, " +
              "params_json = COALESCE(?, params_json), " +
              "features_json = COALESCE(?, features_json) " +
              "WHERE id = ?",
          )
          .run(
            update.color ?? null,
            update.confidence ?? null,
            update.front ?? null,
            update.back ?? null,
            update.fabric ?? null,
            update.sleeve ?? null,
            update.hood ?? null,
            update.side ?? null,
            update.sideMode ?? null,
            update.params ? JSON.stringify(update.params) : null,
            update.features ? JSON.stringify(update.features) : null,
            id,
          );
      } catch (error) {
        console.warn(
          "[draft-store] failed to persist extraction:",
          error instanceof Error ? error.message : error,
        );
      }
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

  // Drafts snapshot their template's params/features at creation. When the
  // registry has moved on (new collars, cuffs, ...), refresh stale drafts
  // from the current template so improvements reach existing wardrobe items.
  let params = stored.params;
  let features = stored.features;
  if (stored.templateVersion !== GARMENT_TEMPLATE_VERSION) {
    const current = getTemplate(stored.templateId);
    if (current) {
      params = current.params;
      features = current.features;
    }
  }

  const mesh: DraftMesh | undefined = isReady
    ? {
        assetUrl: "db://" + stored.id,
        generatedAt: stored.createdAtMs + PIPELINE_MS,
        template: stored.templateId,
        templateLabel: templateLabel(stored.templateId),
        templateVersion: stored.templateVersion,
        params,
        features,
        segmentation: { confidence: stored.segmentationConfidence },
        extractedTextureUrl: stored.extractedTextureUrl,
        extractedBackTextureUrl: stored.extractedBackTextureUrl,
        fabricTextureUrl: stored.fabricTextureUrl,
        sleeveTextureUrl: stored.sleeveTextureUrl,
        hoodTextureUrl: stored.hoodTextureUrl,
        extractedSideTextureUrl: stored.extractedSideTextureUrl,
        sideTextureMode: stored.sideTextureMode,
        color: stored.color,
        bounds: boundsFromParams(params, features),
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

// If the background extraction crashes or wedges (a hung model download, a
// killed worker), the draft would show "Building the mesh..." forever. Any
// read older than this deadline is force-marked ready with whatever it has.
const EXTRACTION_DEADLINE_MS = 180_000;

function recoverStuckDrafts() {
  try {
    getDb()
      .prepare(
        "UPDATE drafts SET extraction_ready = 1 " +
          "WHERE extraction_ready = 0 AND created_at < ?",
      )
      .run(Date.now() - EXTRACTION_DEADLINE_MS);
  } catch {
    // Recovery is best-effort; reads continue regardless.
  }
}

export function listDrafts(): ApiDraft[] {
  recoverStuckDrafts();
  const rows = getDb()
    .prepare("SELECT * FROM drafts ORDER BY created_at DESC")
    .all() as DraftRow[];
  return rows.map((row) => serializeDraft(rowToStored(row)));
}

export function deleteDraft(id: string): boolean {
  const result = getDb().prepare("DELETE FROM drafts WHERE id = ?").run(id);
  return result.changes > 0;
}
