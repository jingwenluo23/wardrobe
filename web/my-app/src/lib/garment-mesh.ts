// Shared types and parametric definitions for the CAD-style garment mesh.
//
// The geometry itself is built on the client (it needs three.js), but the
// *parameters* that drive the parametric construction live here so the API
// layer and the viewer agree on a single source of truth.

export type GarmentTemplate =
  | "top-standard-tee"
  | "top-fitted"
  | "outerwear-boxy"
  | "bottom-straight"
  | "shoe-low";

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
  /** Parametric controls that fully describe the garment geometry. */
  params: GarmentParams;
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
  /** Dominant garment colour as a hex string, sampled from the photos. */
  color: string;
  bounds: GarmentBounds;
};

/**
 * Approximate the on-screen bounding box of the parametric tee from its
 * construction parameters. Returned in centimetres.
 */
export function boundsFromParams(params: GarmentParams): GarmentBounds {
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
