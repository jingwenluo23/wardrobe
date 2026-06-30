import { randomUUID } from "crypto";
import { existsSync, readFileSync } from "fs";
import { mkdir, readFile, rm, writeFile } from "fs/promises";
import path from "path";
import {
  buildDraftMeshAssetForTemplate,
  type DraftMesh,
  generateDraftMesh,
  readDraftMeshAsset,
  readDraftMeshTexture,
  STANDARD_TEE_MESH_VERSION,
} from "@/lib/garment-mesh";

export type DraftPipelineStatus = "processing" | "ready" | "failed";
export type DraftStageStatus = "pending" | "active" | "done";

export type DraftStage = {
  id: string;
  label: string;
  status: DraftStageStatus;
};

export type DraftPhoto = {
  fileName: string;
  originalName: string;
  type: string;
  size: number;
  url: string;
  role?: "front" | "back" | "side";
};

export type DraftJob = {
  id: string;
  name: string;
  category: string;
  color: string;
  status: DraftPipelineStatus;
  progress: number;
  currentStage: string;
  createdAt: string;
  updatedAt: string;
  photos: DraftPhoto[];
  stages: DraftStage[];
  mesh?: DraftMesh;
  error?: string;
};

type SqliteStatement<Row = Record<string, unknown>> = {
  all(...params: unknown[]): Row[];
  get(...params: unknown[]): Row | undefined;
  run(...params: unknown[]): unknown;
};

type SqliteDatabase = {
  exec(sql: string): void;
  prepare<Row = Record<string, unknown>>(sql: string): SqliteStatement<Row>;
};

type SqliteModule = {
  DatabaseSync: new (location: string) => SqliteDatabase;
};

type DraftRow = {
  id: string;
  name: string;
  category: string;
  color: string;
  status: DraftPipelineStatus;
  progress: number;
  current_stage: string;
  created_at: string;
  updated_at: string;
  mesh_json: string | null;
  error: string | null;
};

type DraftPhotoRow = {
  file_name: string;
  original_name: string;
  type: string;
  size: number;
  role: string | null;
};

const dataRoot = path.join(process.cwd(), ".wardrobe-data");
const uploadsRoot = path.join(dataRoot, "uploads");
const meshRoot = path.join(dataRoot, "meshes");
const draftsPath = path.join(dataRoot, "drafts.json");
const databasePath = path.join(dataRoot, "wardrobe.sqlite");
const loadNodeSqlite = Function(
  'return import("node:sqlite")',
) as () => Promise<SqliteModule>;

let database: SqliteDatabase | null = null;
let databaseReady = false;
let DatabaseSyncCtor: SqliteModule["DatabaseSync"] | null = null;

const pipelineStages = [
  { id: "segment", label: "Extract garment region" },
  { id: "template", label: "Build standard base mesh" },
  { id: "mesh", label: "Fit neutral garment proportions" },
  { id: "texture", label: "Project garment texture" },
];

const categoryMap: Record<string, string> = {
  Tops: "Top",
  Bottoms: "Bottom",
};

function isUploadFile(value: FormDataEntryValue | null): value is File {
  return (
    typeof value === "object" &&
    value !== null &&
    "arrayBuffer" in value &&
    typeof value.arrayBuffer === "function" &&
    "name" in value
  );
}

function normalizeCategory(category: string) {
  return categoryMap[category] ?? category;
}

function safeFileName(index: number, file: File) {
  const extension = path.extname(file.name).toLowerCase().replace(/[^a-z0-9.]/g, "");
  return String(index + 1).padStart(2, "0") + "-" + randomUUID() + (extension || ".jpg");
}

function photoUrl(id: string, fileName: string) {
  return "/api/drafts/" + id + "/photos/" + encodeURIComponent(fileName);
}

type DraftPhotoRole = "front" | "back" | "side";

function roleForPosition(position: number): DraftPhotoRole | undefined {
  if (position === 0) {
    return "front";
  }
  if (position === 1) {
    return "back";
  }
  if (position === 2) {
    return "side";
  }
  return undefined;
}

function normalizePhotoRole(value: string | null | undefined) {
  if (value === "front" || value === "back" || value === "side") {
    return value;
  }
  return undefined;
}

function stagesForIndex(activeIndex: number, isReady: boolean): DraftStage[] {
  return pipelineStages.map((stage, index) => ({
    ...stage,
    status: (isReady || index < activeIndex ? "done" : index === activeIndex ? "active" : "pending") as DraftStageStatus,
  }));
}

function parseStoredMesh(value: string | null) {
  if (!value) {
    return undefined;
  }

  try {
    const parsed = JSON.parse(value) as Partial<DraftMesh>;
    if (
      parsed &&
      parsed.kind === "garment-mesh" &&
      parsed.source === "template-fit" &&
      typeof parsed.assetUrl === "string"
    ) {
      return parsed as DraftMesh;
    }
  } catch {
    return undefined;
  }

  return undefined;
}

async function advanceJob(job: DraftJob): Promise<DraftJob> {
  if (job.status !== "processing") {
    return job;
  }

  const elapsed = Date.now() - new Date(job.createdAt).getTime();
  let progress = 18;
  let activeIndex = 0;

  if (elapsed >= 900) {
    progress = 43;
    activeIndex = 1;
  }
  if (elapsed >= 1800) {
    progress = 68;
    activeIndex = 2;
  }
  if (elapsed >= 2900) {
    progress = 88;
    activeIndex = 3;
  }
  if (elapsed >= 4200) {
    try {
      const mesh =
        job.mesh ??
        (await generateDraftMesh({
          draftId: job.id,
          category: job.category,
          photos: job.photos,
          uploadsRoot,
          meshRoot,
        }));

      return {
        ...job,
        status: "ready",
        progress: 100,
        currentStage: "3D mesh ready",
        stages: stagesForIndex(pipelineStages.length, true),
        mesh,
        updatedAt: new Date().toISOString(),
      };
    } catch (error) {
      return {
        ...job,
        status: "failed",
        currentStage: "Mesh generation failed",
        updatedAt: new Date().toISOString(),
        error:
          error instanceof Error
            ? error.message
            : "Unable to generate the garment mesh.",
      };
    }
  }

  return {
    ...job,
    progress,
    currentStage: pipelineStages[activeIndex].label,
    stages: stagesForIndex(activeIndex, false),
    updatedAt: new Date().toISOString(),
  };
}

async function ensureStore() {
  await mkdir(uploadsRoot, { recursive: true });
  await mkdir(meshRoot, { recursive: true });
  await initializeDatabase();
}

async function initializeDatabase() {
  if (!DatabaseSyncCtor) {
    const sqlite = await loadNodeSqlite();
    DatabaseSyncCtor = sqlite.DatabaseSync;
  }

  if (!database) {
    database = new DatabaseSyncCtor(databasePath);
  }

  if (databaseReady) {
    return database;
  }

  database.exec(`
    CREATE TABLE IF NOT EXISTS drafts (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      category TEXT NOT NULL,
      color TEXT NOT NULL,
      status TEXT NOT NULL,
      progress INTEGER NOT NULL,
      current_stage TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      mesh_json TEXT,
      error TEXT
    );

    CREATE TABLE IF NOT EXISTS draft_photos (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      draft_id TEXT NOT NULL,
      file_name TEXT NOT NULL,
      original_name TEXT NOT NULL,
      type TEXT NOT NULL,
      size INTEGER NOT NULL,
      position INTEGER NOT NULL,
      FOREIGN KEY(draft_id) REFERENCES drafts(id) ON DELETE CASCADE
    );
  `);

  try {
    database.exec("ALTER TABLE draft_photos ADD COLUMN role TEXT");
  } catch {}

  migrateJsonDrafts(database);
  databaseReady = true;
  return database;
}

function migrateJsonDrafts(db: SqliteDatabase) {
  if (!existsSync(draftsPath)) {
    return;
  }

  const existingDraft = db
    .prepare<{ count: number }>("SELECT COUNT(*) AS count FROM drafts")
    .get();

  if ((existingDraft?.count ?? 0) > 0) {
    return;
  }

  const content = readFileSync(draftsPath, "utf8");
  const jobs = JSON.parse(content) as DraftJob[];

  const insertDraft = db.prepare(`
    INSERT INTO drafts (
      id, name, category, color, status, progress, current_stage,
      created_at, updated_at, mesh_json, error
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const insertPhoto = db.prepare(`
    INSERT INTO draft_photos (
      draft_id, file_name, original_name, type, size, position, role
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
  `);

  for (const job of jobs) {
    insertDraft.run(
      job.id,
      job.name,
      job.category,
      job.color,
      job.status,
      job.progress,
      job.currentStage,
      job.createdAt,
      job.updatedAt,
      job.mesh ? JSON.stringify(job.mesh) : null,
      job.error ?? null,
    );

    for (const [index, photo] of job.photos.entries()) {
      insertPhoto.run(
        job.id,
        photo.fileName,
        photo.originalName,
        photo.type,
        photo.size,
        index,
        photo.role ?? roleForPosition(index) ?? null,
      );
    }
  }
}

function mapDraftRow(row: DraftRow, photos: DraftPhoto[]): DraftJob {
  return {
    id: row.id,
    name: row.name,
    category: row.category,
    color: row.color,
    status: row.status,
    progress: row.progress,
    currentStage: row.current_stage,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    photos,
    stages: stagesForIndex(0, false),
    mesh: parseStoredMesh(row.mesh_json),
    error: row.error ?? undefined,
  };
}

async function readJobs() {
  await ensureStore();
  const db = initializeDatabase();
  const readyDb = await db;
  const draftRows = readyDb
    .prepare<DraftRow>("SELECT * FROM drafts ORDER BY created_at DESC")
    .all();
  const photoRows = readyDb
    .prepare<DraftPhotoRow & { draft_id: string }>(
      "SELECT draft_id, file_name, original_name, type, size, role FROM draft_photos ORDER BY draft_id, position ASC",
    )
    .all();

  const photosByDraftId = new Map<string, DraftPhoto[]>();
  for (const row of photoRows) {
    const photos = photosByDraftId.get(row.draft_id) ?? [];
    photos.push({
      fileName: row.file_name,
      originalName: row.original_name,
      type: row.type,
      size: row.size,
      url: photoUrl(row.draft_id, row.file_name),
      role: normalizePhotoRole(row.role) ?? roleForPosition(photos.length),
    });
    photosByDraftId.set(row.draft_id, photos);
  }

  return draftRows.map((row) => mapDraftRow(row, photosByDraftId.get(row.id) ?? []));
}

async function persistJob(job: DraftJob) {
  await ensureStore();
  const db = await initializeDatabase();

  db.prepare(`
    UPDATE drafts
    SET name = ?, category = ?, color = ?, status = ?, progress = ?,
        current_stage = ?, created_at = ?, updated_at = ?, mesh_json = ?, error = ?
    WHERE id = ?
  `).run(
    job.name,
    job.category,
    job.color,
    job.status,
    job.progress,
    job.currentStage,
    job.createdAt,
    job.updatedAt,
    job.mesh ? JSON.stringify(job.mesh) : null,
    job.error ?? null,
    job.id,
  );
}

async function tryBuildMesh(job: DraftJob) {
  return generateDraftMesh({
    draftId: job.id,
    category: job.category,
    photos: job.photos,
    uploadsRoot,
    meshRoot,
  });
}

async function shouldRefreshMeshAsset(job: DraftJob) {
  if (!job.mesh) {
    return true;
  }

  try {
    const storedAsset = await readDraftMeshAsset(meshRoot, job.id);
    const expectedAsset = buildDraftMeshAssetForTemplate(job.mesh.template);
    const expectedVersion = STANDARD_TEE_MESH_VERSION;

    return (
      storedAsset.version !== expectedVersion ||
      storedAsset.signature !== expectedAsset.signature
    );
  } catch {
    return true;
  }
}

async function readAdvancedJobs() {
  const jobs = await readJobs();
  const advancedJobs: DraftJob[] = [];

  for (const job of jobs) {
    const shouldRefreshTopMesh =
      job.category.toLowerCase().includes("top") &&
      job.mesh?.template !== "top-standard-tee";
    const shouldRefreshOutdatedAsset =
      job.status === "ready" && job.photos.length >= 2 && (await shouldRefreshMeshAsset(job));
    const shouldRetryFailedMesh =
      job.status === "failed" && !job.mesh && job.photos.length >= 2;
    let advancedJob: DraftJob;

    try {
      advancedJob =
        job.status === "ready" &&
        (!job.mesh || shouldRefreshTopMesh || shouldRefreshOutdatedAsset)
          ? {
              ...job,
              status: "ready",
              progress: 100,
              currentStage: "3D mesh ready",
              stages: stagesForIndex(pipelineStages.length, true),
              mesh: await tryBuildMesh(job),
              updatedAt: new Date().toISOString(),
              error: undefined,
            }
          : shouldRetryFailedMesh
            ? {
                ...job,
                status: "ready",
                progress: 100,
                currentStage: "3D mesh ready",
                stages: stagesForIndex(pipelineStages.length, true),
                mesh: await tryBuildMesh(job),
                updatedAt: new Date().toISOString(),
                error: undefined,
              }
          : await advanceJob(job);
    } catch (error) {
      if (job.status === "ready" && job.mesh) {
        advancedJob = {
          ...job,
          updatedAt: new Date().toISOString(),
          error:
            error instanceof Error
              ? error.message
              : "Unable to refresh the garment mesh.",
        };
      } else {
        advancedJob = {
          ...job,
          status: "failed",
          currentStage: "Mesh generation failed",
          updatedAt: new Date().toISOString(),
          error:
            error instanceof Error
              ? error.message
              : "Unable to generate the garment mesh.",
        };
      }
    }

    if (JSON.stringify(job) !== JSON.stringify(advancedJob)) {
      await persistJob(advancedJob);
    }
    advancedJobs.push(advancedJob);
  }

  return advancedJobs;
}

export async function listDrafts() {
  return readAdvancedJobs();
}

export async function getDraft(id: string) {
  const jobs = await readAdvancedJobs();
  return jobs.find((job) => job.id === id) ?? null;
}

export async function deleteDraft(id: string) {
  await ensureStore();
  const db = await initializeDatabase();
  const existingDraft = db
    .prepare<{ id: string }>("SELECT id FROM drafts WHERE id = ?")
    .get(id);

  if (!existingDraft) {
    return false;
  }

  db.prepare("DELETE FROM draft_photos WHERE draft_id = ?").run(id);
  db.prepare("DELETE FROM drafts WHERE id = ?").run(id);

  await Promise.all([
    rm(path.join(uploadsRoot, id), { recursive: true, force: true }),
    rm(path.join(meshRoot, id + ".json"), { force: true }),
    rm(path.join(meshRoot, id + "-texture.png"), { force: true }),
  ]);

  return true;
}

export async function createDraft(formData: FormData) {
  const name = String(formData.get("name") ?? "").trim();
  const category = normalizeCategory(String(formData.get("category") ?? "Tops"));
  const frontPhoto = formData.get("frontPhoto");
  const backPhoto = formData.get("backPhoto");
  const sidePhoto = formData.get("sidePhoto");
  const orderedEntries: Array<{ photo: File; role: DraftPhotoRole }> = [];

  if (isUploadFile(frontPhoto)) {
    orderedEntries.push({ photo: frontPhoto, role: "front" });
  }
  if (isUploadFile(backPhoto)) {
    orderedEntries.push({ photo: backPhoto, role: "back" });
  }
  if (isUploadFile(sidePhoto)) {
    orderedEntries.push({ photo: sidePhoto, role: "side" });
  }

  if (orderedEntries.length === 0) {
    const legacyPhotos = formData.getAll("photos").filter(isUploadFile).slice(0, 6);
    legacyPhotos.forEach((photo, index) => {
      const role = roleForPosition(index);
      if (role) {
        orderedEntries.push({ photo, role });
      }
    });
  }

  if (!name) {
    throw new Error("Item name is required.");
  }
  if (!orderedEntries.some((entry) => entry.role === "front")) {
    throw new Error("Front garment photo is required.");
  }
  if (!orderedEntries.some((entry) => entry.role === "back")) {
    throw new Error("Back garment photo is required.");
  }
  if (orderedEntries.some(({ photo }) => !photo.type.startsWith("image/"))) {
    throw new Error("Only image uploads are supported.");
  }

  const id = randomUUID();
  const uploadDir = path.join(uploadsRoot, id);
  await mkdir(uploadDir, { recursive: true });

  const storedPhotos: DraftPhoto[] = [];
  for (const [index, entry] of orderedEntries.entries()) {
    const { photo, role } = entry;
    const fileName = safeFileName(index, photo);
    await writeFile(path.join(uploadDir, fileName), Buffer.from(await photo.arrayBuffer()));
    storedPhotos.push({
      fileName,
      originalName: photo.name,
      type: photo.type || "application/octet-stream",
      size: photo.size,
      url: photoUrl(id, fileName),
      role,
    });
  }

  const now = new Date().toISOString();
  const job: DraftJob = {
    id,
    name,
    category,
    color: "From photos",
    status: "processing",
    progress: 18,
    currentStage: pipelineStages[0].label,
    createdAt: now,
    updatedAt: now,
    photos: storedPhotos,
    stages: stagesForIndex(0, false),
  };

  await ensureStore();
  const db = await initializeDatabase();

  db.prepare(`
    INSERT INTO drafts (
      id, name, category, color, status, progress, current_stage,
      created_at, updated_at, mesh_json, error
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    job.id,
    job.name,
    job.category,
    job.color,
    job.status,
    job.progress,
    job.currentStage,
    job.createdAt,
    job.updatedAt,
    null,
    null,
  );

  const insertPhoto = db.prepare(`
    INSERT INTO draft_photos (
      draft_id, file_name, original_name, type, size, position, role
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
  `);

  for (const [index, photo] of storedPhotos.entries()) {
    insertPhoto.run(
      job.id,
      photo.fileName,
      photo.originalName,
      photo.type,
      photo.size,
      index,
      photo.role ?? roleForPosition(index) ?? null,
    );
  }

  return job;
}

export async function getDraftPhoto(id: string, fileName: string) {
  const job = await getDraft(id);
  if (!job) {
    return null;
  }

  const photo = job.photos.find((candidate) => candidate.fileName === fileName);
  if (!photo) {
    return null;
  }

  const filePath = path.join(uploadsRoot, id, path.basename(fileName));
  return {
    content: await readFile(filePath),
    type: photo.type,
  };
}

export async function getDraftMeshAsset(id: string) {
  const job = await getDraft(id);
  if (!job?.mesh) {
    return null;
  }

  return readDraftMeshAsset(meshRoot, id);
}

export async function getDraftMeshTexture(id: string) {
  const job = await getDraft(id);
  if (!job?.mesh) {
    return null;
  }

  return readDraftMeshTexture(meshRoot, id);
}
