// Declarative garment template registry.
//
// Every clothing type the app supports is a data preset here: numeric pattern
// params + feature toggles. The geometry blocks consume these; adding a new
// type of the same archetype means adding an object, not geometry code.
//
// This module is pure data (no sharp/three imports), safe for both the
// server store and the client upload form.

import {
  defaultTeeFeatures,
  defaultTeeParams,
  type GarmentFeatures,
  type GarmentParams,
} from "./garment-mesh";

/** Bump when geometry/param semantics change so stored drafts can migrate. */
export const GARMENT_TEMPLATE_VERSION = 2;

export type GarmentTemplateDef = {
  id: string;
  label: string;
  /** Wardrobe category this type belongs to (matches the category picker). */
  category: string;
  params: GarmentParams;
  features: GarmentFeatures;
};

const longSleeve: Partial<GarmentParams> = {
  sleeveLength: 56,
  armholeDepth: 21,
};

/**
 * Bottoms preset factory: every pants/shorts type is the same leg-loft
 * block with different measurements (cm) and trims.
 */
function bottomsPresets(): GarmentTemplateDef[] {
  const make = (
    id: string,
    label: string,
    dims: {
      inseam: number;
      thigh: number;
      open: number;
      waist?: number;
      hip?: number;
      rise?: number;
      depth?: number;
    },
    trims?: { cuff?: "ribbed"; cargo?: boolean },
  ): GarmentTemplateDef => ({
    id,
    label,
    category: "Bottoms",
    params: {
      ...defaultTeeParams,
      bodyDepth: dims.depth ?? 22,
      waistWidth: dims.waist ?? 41,
      hipWidth: dims.hip ?? 51,
      rise: dims.rise ?? 28,
      inseam: dims.inseam,
      thighWidth: dims.thigh,
      legOpening: dims.open,
    },
    features: {
      ...defaultTeeFeatures,
      archetype: "bottoms",
      cuff: trims?.cuff ?? "raw",
      cargoPockets: trims?.cargo ?? false,
    },
  });

  return [
    // Core everyday
    make("bottom-jeans", "Jeans", { inseam: 78, thigh: 33, open: 21 }),
    make("bottom-trousers", "Trousers / pants", { inseam: 80, thigh: 34, open: 23 }),
    make("bottom-chinos", "Chinos", { inseam: 78, thigh: 32, open: 19 }),
    make("bottom-cargo", "Cargo pants", { inseam: 78, thigh: 36, open: 24 }, { cargo: true }),
    make("bottom-joggers", "Joggers", { inseam: 74, thigh: 34, open: 13 }, { cuff: "ribbed" }),
    make("bottom-sweatpants", "Sweatpants", { inseam: 76, thigh: 38, open: 16 }, { cuff: "ribbed" }),
    // Formal
    make("bottom-dress-trousers", "Dress trousers", { inseam: 82, thigh: 34, open: 23 }),
    make("bottom-suit-pants", "Suit pants", { inseam: 82, thigh: 34, open: 22 }),
    make("bottom-tuxedo-pants", "Tuxedo pants", { inseam: 82, thigh: 34, open: 22 }),
    // Sports / active
    make("bottom-shorts", "Shorts (sports)", { inseam: 22, thigh: 34, open: 30 }),
    make("bottom-running-shorts", "Running shorts", { inseam: 10, thigh: 32, open: 32 }),
    make("bottom-training-shorts", "Training shorts", { inseam: 18, thigh: 34, open: 30 }),
    make("bottom-basketball-shorts", "Basketball shorts", { inseam: 28, thigh: 38, open: 34 }),
    make("bottom-compression-leggings", "Compression leggings", {
      inseam: 78,
      thigh: 26,
      open: 11,
      waist: 37,
      hip: 46,
      depth: 19,
    }),
    make("bottom-track-pants", "Track pants", { inseam: 78, thigh: 36, open: 18 }),
    // Casual / seasonal shorts
    make("bottom-denim-shorts", "Denim shorts", { inseam: 26, thigh: 34, open: 28 }),
    make("bottom-chino-shorts", "Chino shorts", { inseam: 24, thigh: 32, open: 26 }),
    make("bottom-cargo-shorts", "Cargo shorts", { inseam: 26, thigh: 36, open: 30 }, { cargo: true }),
    make("bottom-board-shorts", "Beach / swim shorts", { inseam: 30, thigh: 36, open: 32 }),
    // Other / special
    make("bottom-work-pants", "Work pants (utility)", { inseam: 80, thigh: 36, open: 24 }, { cargo: true }),
    make("bottom-harem-pants", "Harem pants", { inseam: 72, thigh: 44, open: 14, rise: 34, hip: 56 }),
    make("bottom-linen-pants", "Linen pants", { inseam: 80, thigh: 36, open: 24 }),
  ];
}

export const garmentTemplates: GarmentTemplateDef[] = [
  {
    id: "top-standard-tee",
    label: "T-shirt (short sleeve)",
    category: "Tops",
    params: { ...defaultTeeParams },
    features: { ...defaultTeeFeatures },
  },
  {
    id: "top-long-sleeve-tee",
    label: "Long-sleeve T-shirt",
    category: "Tops",
    params: { ...defaultTeeParams, ...longSleeve },
    features: { ...defaultTeeFeatures, sleeveTaper: 0.72 },
  },
  {
    id: "top-sweatshirt",
    label: "Sweatshirt (crewneck)",
    category: "Tops",
    params: {
      ...defaultTeeParams,
      ...longSleeve,
      bodyWidth: 58,
      bodyDepth: 26,
      neckWidthFront: 19,
      neckDropFront: 7,
    },
    features: {
      neckFinish: "band",
      hemBand: true,
      cuff: "ribbed",
      sleeveTaper: 0.6,
    },
  },
  {
    id: "top-hoodie",
    label: "Hoodie",
    category: "Tops",
    params: {
      ...defaultTeeParams,
      ...longSleeve,
      bodyWidth: 59,
      bodyDepth: 27,
      neckWidthFront: 21,
      neckDropFront: 6.5,
      neckDropBack: 3,
    },
    features: {
      neckFinish: "hood",
      hemBand: true,
      cuff: "ribbed",
      sleeveTaper: 0.6,
    },
  },
  // Legacy / other-category placeholders (same knit block until their own
  // archetypes land).
  {
    id: "outerwear-boxy",
    label: "Boxy outerwear",
    category: "Outerwear",
    params: {
      ...defaultTeeParams,
      bodyWidth: 62,
      bodyDepth: 24,
      sleeveLength: 56,
      armholeDepth: 26,
    },
    features: { ...defaultTeeFeatures, sleeveTaper: 0.75 },
  },
  // --- Bottoms archetype ---------------------------------------------------
  ...bottomsPresets(),
  {
    id: "shoe-low",
    label: "Low shoe",
    category: "Shoes",
    params: { ...defaultTeeParams },
    features: { ...defaultTeeFeatures },
  },
];

const byId = new Map(garmentTemplates.map((t) => [t.id, t]));

// Ids that shipped before the registry existed.
const LEGACY_ALIASES: Record<string, string> = {
  "bottom-straight": "bottom-jeans",
  "top-fitted": "top-standard-tee",
};

export function getTemplate(id: string | null | undefined) {
  if (!id) {
    return null;
  }
  return byId.get(id) ?? byId.get(LEGACY_ALIASES[id] ?? "") ?? null;
}

export function templatesForCategory(category: string): GarmentTemplateDef[] {
  const matches = garmentTemplates.filter((t) => t.category === category);
  return matches.length > 0 ? matches : [garmentTemplates[0]];
}

/** Resolve the template for an upload: explicit id first, else category default. */
export function resolveTemplate(
  templateId: string | null | undefined,
  category: string,
): GarmentTemplateDef {
  return getTemplate(templateId) ?? templatesForCategory(category)[0];
}

export function templateLabel(id: string): string {
  return byId.get(id)?.label ?? id;
}
