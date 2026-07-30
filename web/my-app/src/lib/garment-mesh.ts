// Shared types and parametric definitions for the CAD-style garment mesh.
//
// The geometry itself is built on the client (it needs three.js), but the
// *parameters* that drive the parametric construction live here so the API
// layer and the viewer agree on a single source of truth.

/**
 * Template ids come from the registry in `garment-templates.ts`; kept as a
 * string so adding a template never requires touching this type.
 */
export type GarmentTemplate = string;

/**
 * Feature toggles layered on top of the numeric params. Each clothing type
 * in the registry is just a params + features preset — the geometry blocks
 * read these to decide which trims to build.
 */
export type GarmentFeatures = {
  /** Which geometry archetype builds this garment. Defaults to "top". */
  archetype?: "top" | "bottoms" | "skirt";
  /** Neck opening finish. */
  neckFinish: "band" | "hood" | "turtleneck" | "polo-collar" | "shirt-collar";
  /** Neckline shape: rounded crew (default) or a V. */
  neckShape?: "crew" | "v";
  /** Two flat patch pockets low on the front (cardigans, camp shirts). */
  patchPockets?: boolean;
  /** Ribbed sweatshirt-style band at the bottom hem. */
  hemBand: boolean;
  /** Sleeve/leg-end finish: raw edge, a snug ribbed cuff (joggers), or a
   *  buttoned barrel cuff (woven shirts). */
  cuff: "raw" | "ribbed" | "barrel";
  /** Sleeve width at the opening relative to the root (1 = straight tube). */
  sleeveTaper: number;
  /** Front placket strip with buttons: polo half-placket or full button front. */
  placket?: "none" | "half" | "full";
  /** Set false for tank tops / sleeveless jerseys (open armholes). */
  sleeves?: boolean;
  /** Bottoms: boxy utility pockets on the outer thighs. */
  cargoPockets?: boolean;
  /** Bottoms/skirts: flat tailored waistband with belt loops, or a thicker
   *  elastic band with a hanging drawcord. */
  waistband?: "flat" | "elastic";
  /** Tops: attach a lofted skirt below the bodice hem (dresses). Length in
   *  cm from the hem down. */
  skirtLength?: number;
  /** Dresses: hem width relative to the bodice hem (0 = straight/bodycon,
   *  0.6 = full A-line flare). */
  skirtFlare?: number;
  /** Fabric weight/structure: scales trims and adds surface texture.
   *  jersey = light knit (default), fleece = thick sweatshirt fabric,
   *  knit = chunky visible-rib sweater knit. */
  fabric?: "jersey" | "fleece" | "knit";
  /** Fitted silhouette: pinch the torso at the waist (0 = straight cut,
   *  ~0.12 = women's fitted shirt) so chest and hip read fuller. */
  waistPinch?: number;
};

export const defaultTeeFeatures: GarmentFeatures = {
  neckFinish: "band",
  hemBand: false,
  cuff: "raw",
  sleeveTaper: 1,
};

/**
 * Live-editable construction parameters for the parametric t-shirt.
 *
 * Values are in centimetres (matching a real pattern), except `shoulderSlope`
 * which is in degrees. The viewer normalises these into scene units.
 */
export type GarmentParams = {
  /** Full chest width of the body block. */
  bodyWidth: number;
  /** Hem-to-shoulder length of the body block. */
  bodyLength: number;
  /** Front/back depth of the torso (how rounded the cross-section is). */
  bodyDepth: number;
  /** Width of the front neckline opening. */
  neckWidthFront: number;
  /** How far the front neckline dips below the shoulder line. */
  neckDropFront: number;
  /** How far the back neckline dips below the shoulder line. */
  neckDropBack: number;
  /** Shoulder slope angle in degrees (defined by a curve, not a hard angle). */
  shoulderSlope: number;
  /** Vertical depth of the armhole curve below the shoulder point. */
  armholeDepth: number;
  /** Length of the (short) cap sleeve. */
  sleeveLength: number;
  /** Opening width of the sleeve hem relative to the armhole. */
  sleeveOpening: number;
  /** Hem width relative to the chest width (1 = straight side seams). */
  hemWidthFactor?: number;

  // --- Bottoms archetype (present when features.archetype === "bottoms") ---
  /** Flat waist width. */
  waistWidth?: number;
  /** Flat hip width (widest point). */
  hipWidth?: number;
  /** Waist-to-crotch length. */
  rise?: number;
  /** Crotch-to-hem length (long pants ~78, shorts 10-30). */
  inseam?: number;
  /** Flat single-leg width at the thigh. */
  thighWidth?: number;
  /** Flat single-leg width at the hem opening. */
  legOpening?: number;
  /** Shoulder width as a fraction of body half-width (default 0.8);
   *  small values give narrow tank/undershirt straps. */
  shoulderWidthFactor?: number;
};

/** View-normalized measurements recovered from a segmented garment mask. */
export type GarmentShapeEstimate = {
  bodyAspectRatio: number;
  spanRatio: number;
  shoulderRatio: number;
  chestRatio: number;
  waistRatio: number;
  hemRatio: number;
  neckWidthRatio: number;
  neckDepthRatio: number;
  confidence: number;
};

export const defaultTeeParams: GarmentParams = {
  bodyWidth: 55,
  bodyLength: 70,
  bodyDepth: 24,
  neckWidthFront: 18.5,
  neckDropFront: 8.5,
  neckDropBack: 2.5,
  shoulderSlope: 16,
  armholeDepth: 20,
  sleeveLength: 21,
  sleeveOpening: 17,
};

export type GarmentBounds = {
  width: number;
  height: number;
  depth: number;
};

export type DraftMesh = {
  /** Logical id for the generated asset (used for cache-busting the viewer). */
  assetUrl: string;
  generatedAt: number;
  template: GarmentTemplate;
  /** Human-readable template name for the viewer info panel. */
  templateLabel?: string;
  /** Registry version the draft was built with (for future migrations). */
  templateVersion?: number;
  /** Parametric controls that fully describe the garment geometry. */
  params: GarmentParams;
  /** Feature toggles (trims) layered on the params. */
  features?: GarmentFeatures;
  /** Garment extraction metadata derived from the uploaded photos. */
  segmentation: {
    confidence: number;
  };
  /** Optional isolated garment texture (data URL) for the front panel. */
  extractedTextureUrl?: string;
  /** Optional isolated garment texture (data URL) for the back panel. */
  extractedBackTextureUrl?: string;
  /** Plain fabric swatch (data URL) for sleeves/collar, from the photo. */
  fabricTextureUrl?: string;
  /** Extracted sleeve texture (data URL), warped from the photo's sleeve. */
  sleeveTextureUrl?: string;
  /** Extracted hood texture (data URL), from above the shoulder line. */
  hoodTextureUrl?: string;
  /** Optional photographed or front/back-synthesized side surface. */
  extractedSideTextureUrl?: string;
  /** Whether the side surface came from a photo or automatic synthesis. */
  sideTextureMode?: "photo" | "synthesized";
  /** Dominant garment colour as a hex string, sampled from the photos. */
  color: string;
  bounds: GarmentBounds;
};

/**
 * Approximate the on-screen bounding box of the parametric tee from its
 * construction parameters. Returned in centimetres.
 */
export function boundsFromParams(
  params: GarmentParams,
  features?: GarmentFeatures,
): GarmentBounds {
  if (features?.archetype === "bottoms") {
    const width = params.hipWidth ?? 50;
    const height = (params.rise ?? 28) + (params.inseam ?? 78) + 4;
    return {
      width: Math.round(width) / 10,
      height: Math.round(height) / 10,
      depth: Math.round(params.bodyDepth) / 10,
    };
  }
  if (features?.archetype === "skirt") {
    const width = Math.max(params.hipWidth ?? 50, params.legOpening ?? 50);
    const height = params.inseam ?? 60;
    return {
      width: Math.round(width) / 10,
      height: Math.round(height) / 10,
      depth: Math.round(params.bodyDepth) / 10,
    };
  }
  // Total width = body width + two cap sleeves projecting outward.
  const width = params.bodyWidth + params.sleeveLength * 1.1;
  const height = params.bodyLength;
  const depth = params.bodyDepth;
  return {
    width: Math.round(width) / 10,
    height: Math.round(height) / 10,
    depth: Math.round(depth) / 10,
  };
}
