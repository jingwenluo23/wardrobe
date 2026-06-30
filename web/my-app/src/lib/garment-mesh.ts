import { readFile } from "fs/promises";
import path from "path";

export const STANDARD_TEE_MESH_VERSION = 72;
const STANDARD_TEE_TEMPLATE = "top-standard-tee";
const STANDARD_TEE_TEXTURE_MODE = "neutral-preview-v1";
const STANDARD_TEE_SIGNATURE =
  STANDARD_TEE_TEMPLATE +
  ":v" +
  STANDARD_TEE_MESH_VERSION +
  ":" +
  STANDARD_TEE_TEXTURE_MODE;

// Runtime uses one checked-in CAD asset. The generator scripts are offline tools
// only and are not part of request-time mesh creation.
const standardAssetPath = path.join(
  process.cwd(),
  "src/lib/assets/tshirt_cad.json",
);

type DraftMeshAssetSegment = {
  id: string;
  label: string;
  indexStart: number;
  indexCount: number;
};

export type DraftMeshAsset = {
  version: number;
  template: string;
  positions: number[];
  normals: number[];
  uvs: number[];
  indices: number[];
  segments?: DraftMeshAssetSegment[];
  bounds: {
    width: number;
    height: number;
    depth: number;
  };
  signature?: string;
  textureUrl?: string;
  extractedTextureUrl?: string;
  textureVersion?: string;
};

export type DraftMesh = {
  kind: "garment-mesh";
  source: "template-fit";
  template: string;
  assetUrl: string;
  generatedAt: string;
  bounds: DraftMeshAsset["bounds"];
  segmentation: {
    confidence: number;
  };
  textureUrl?: string;
  extractedTextureUrl?: string;
  textureVersion?: string;
};

type DraftPhoto = {
  fileName: string;
  originalName: string;
  type: string;
  size: number;
  url: string;
  role?: "front" | "back" | "side";
};

type GenerateDraftMeshOptions = {
  draftId: string;
  category: string;
  photos: DraftPhoto[];
  uploadsRoot: string;
  meshRoot: string;
};

async function readStandardAsset(): Promise<DraftMeshAsset> {
  const source = await readFile(standardAssetPath, "utf8");
  return JSON.parse(source) as DraftMeshAsset;
}

export function buildDraftMeshAssetForTemplate(template: string) {
  return {
    version: STANDARD_TEE_MESH_VERSION,
    signature:
      template === STANDARD_TEE_TEMPLATE
        ? STANDARD_TEE_SIGNATURE
        : template +
          ":v" +
          STANDARD_TEE_MESH_VERSION +
          ":" +
          STANDARD_TEE_TEXTURE_MODE,
  };
}

export async function buildStandardTShirtAsset() {
  return readStandardAsset();
}

export async function generateDraftMesh({
  draftId,
  meshRoot,
}: GenerateDraftMeshOptions): Promise<DraftMesh> {
  const baseAsset = await readStandardAsset();
  const generatedAt = new Date().toISOString();

  const asset: DraftMeshAsset = {
    ...baseAsset,
    version: STANDARD_TEE_MESH_VERSION,
    template: STANDARD_TEE_TEMPLATE,
    signature: STANDARD_TEE_SIGNATURE,
    extractedTextureUrl: undefined,
    textureUrl: undefined,
    textureVersion: undefined,
  };

  await readFile(standardAssetPath);
  await import("fs/promises").then(({ writeFile }) =>
    writeFile(
      path.join(meshRoot, draftId + ".json"),
      JSON.stringify(asset),
      "utf8",
    ),
  );

  return {
    kind: "garment-mesh",
    source: "template-fit",
    template: asset.template,
    assetUrl: "/api/drafts/" + draftId + "/mesh",
    generatedAt,
    bounds: asset.bounds,
    segmentation: {
      confidence: 0.96,
    },
    extractedTextureUrl: asset.extractedTextureUrl,
    textureUrl: asset.textureUrl,
    textureVersion: undefined,
  };
}

export async function readDraftMeshAsset(meshRoot: string, draftId: string) {
  const source = await readFile(path.join(meshRoot, draftId + ".json"), "utf8");
  return JSON.parse(source) as DraftMeshAsset;
}

export async function readDraftMeshTexture(meshRoot: string, draftId: string) {
  return readFile(path.join(meshRoot, draftId + "-texture.png"));
}
