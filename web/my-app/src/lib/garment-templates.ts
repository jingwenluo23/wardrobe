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
export const GARMENT_TEMPLATE_VERSION = 8;

export type Gender = "male" | "female";

export type GarmentTemplateDef = {
  id: string;
  label: string;
  /** Wardrobe category this type belongs to (matches the category picker). */
  category: string;
  /** Who the type is offered to. Omit for unisex (both). */
  gender?: Gender;
  params: GarmentParams;
  features: GarmentFeatures;
};

const CATEGORY_ORDER = ["Tops", "Bottoms", "Dresses", "Outerwear", "Shoes"];

// Long-sleeve base: the woven shirt family (74 cm body, no hem band).
//
// Where the cuff lands is a relationship, not a constant. The raglan sleeve
// hangs from the shoulder line, which rises with bodyLength, while the hem
// falls with it — so one shared sleeveLength cannot serve bodies of different
// lengths. Measured on the built geometry, the cuff finishes roughly
//
//     bodyLength - sleeveLength - 13 cm     above the hem
//
// and a hem band lifts the hem about 5 cm further, so banded garments need a
// correspondingly shorter sleeve. 61 cm puts the shirt cuff 2.2 cm above its
// hem; every preset below that has a shorter body or a hem band carries its
// own sleeveLength for the same 2-3 cm finish. Left at a shared 61 the hoodie
// and sweatshirt cuffs hung 7.3 cm PAST the hem.
const longSleeve: Partial<GarmentParams> = {
  sleeveLength: 61,
  // Deep armhole. The sleeve is lofted from the armhole loop, so the armhole
  // IS the sleeve's girth — at 21 the tube came out noticeably skinny beside
  // the torso, where a relaxed long-sleeve top drops from a roomy armhole.
  armholeDepth: 28,
};

/**
 * Woven button-front shirts: one geometry preset (shirt collar + full
 * placket + barrel cuffs), the types differ by fit and — later — fabric.
 */
function wovenShirtPresets(): GarmentTemplateDef[] {
  const make = (
    id: string,
    label: string,
    fit?: Partial<GarmentParams>,
  ): GarmentTemplateDef => ({
    id,
    label,
    category: "Tops",
    gender: "male",
    params: {
      ...defaultTeeParams,
      ...longSleeve,
      bodyWidth: 54,
      bodyDepth: 23,
      bodyLength: 74,
      neckWidthFront: 15.5,
      neckDropFront: 6,
      neckDropBack: 2,
      ...fit,
    },
    features: {
      neckFinish: "shirt-collar",
      hemBand: false,
      cuff: "barrel",
      sleeveTaper: 0.6,
      placket: "full",
      // A button shirt is woven cloth, not jersey. The renderer and the
      // geometry both key the fabric-grain handling off this flag: panel UVs
      // are projected from the vertex's real X so a pinstripe stays vertical
      // over the narrowing upper panel instead of fanning out, and the sleeve
      // takes a repeating world-space projection instead of the torso
      // photograph wrapped round the tube. Without it every shirt rendered as
      // jersey and those paths never ran.
      fabric: "woven",
    },
  });
  // Women's cut: same construction (collar, placket, barrel cuffs), fitted
  // silhouette — pinched waist so chest and hip read fuller, slightly
  // narrower shoulders and shorter body.
  const makeFitted = (base: GarmentTemplateDef): GarmentTemplateDef => ({
    ...base,
    id: base.id + "-f",
    gender: "female",
    params: {
      ...base.params,
      bodyWidth: base.params.bodyWidth - 4,
      bodyLength: base.params.bodyLength - 4,
      // Shorter body, shorter sleeve — the cuff has to keep its place
      // relative to the hem.
      sleeveLength: base.params.sleeveLength - 4,
      shoulderWidthFactor: 0.76,
    },
    features: {
      ...base.features,
      waistPinch: 0.15,
    },
  });
  const straight = [
    make("top-dress-shirt", "Dress shirt", { bodyWidth: 52 }),
    make("top-casual-shirt", "Casual shirt"),
    make("top-flannel-shirt", "Flannel shirt", { bodyWidth: 56, bodyDepth: 24 }),
    make("top-denim-shirt", "Denim shirt"),
    make("top-oxford-shirt", "Oxford shirt", { bodyWidth: 53 }),
    make("top-button-down-shirt", "Button-down shirt", { bodyWidth: 53 }),
  ];
  return [...straight, ...straight.map(makeFitted)];
}

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
    trims?: {
      cuff?: "ribbed";
      cargo?: boolean;
      waistband?: "flat" | "elastic";
    },
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
      waistband: trims?.waistband ?? "flat",
    },
  });

  return [
    // Core everyday — tailored belt-loop waistbands
    make("bottom-jeans", "Jeans (straight)", { inseam: 78, thigh: 33, open: 21 }),
    make("bottom-flare-jeans", "Flare jeans", { inseam: 82, thigh: 30, open: 30 }),
    make("bottom-baggy-jeans", "Baggy jeans", { inseam: 76, thigh: 44, open: 30, hip: 56 }),
    make("bottom-trousers", "Trousers / pants", { inseam: 80, thigh: 34, open: 23 }),
    make("bottom-chinos", "Chinos", { inseam: 78, thigh: 32, open: 19 }),
    make("bottom-cargo", "Cargo pants", { inseam: 78, thigh: 36, open: 24 }, { cargo: true }),
    make("bottom-joggers", "Joggers", { inseam: 74, thigh: 34, open: 13 }, { cuff: "ribbed", waistband: "elastic" }),
    make("bottom-sweatpants", "Sweatpants", { inseam: 76, thigh: 38, open: 16 }, { cuff: "ribbed", waistband: "elastic" }),
    // Formal
    make("bottom-dress-trousers", "Dress trousers", { inseam: 82, thigh: 34, open: 23 }),
    make("bottom-suit-pants", "Suit pants", { inseam: 82, thigh: 34, open: 22 }),
    make("bottom-tuxedo-pants", "Tuxedo pants", { inseam: 82, thigh: 34, open: 22 }),
    // Sports / active — elastic drawcord waistbands
    make("bottom-shorts", "Shorts (sports)", { inseam: 22, thigh: 34, open: 30 }, { waistband: "elastic" }),
    make("bottom-running-shorts", "Running shorts", { inseam: 10, thigh: 32, open: 32 }, { waistband: "elastic" }),
    make("bottom-training-shorts", "Training shorts", { inseam: 18, thigh: 34, open: 30 }, { waistband: "elastic" }),
    make("bottom-basketball-shorts", "Basketball shorts", { inseam: 28, thigh: 38, open: 34 }, { waistband: "elastic" }),
    make("bottom-compression-leggings", "Compression leggings", {
      inseam: 78,
      thigh: 26,
      open: 11,
      waist: 37,
      hip: 46,
      depth: 19,
    }, { waistband: "elastic" }),
    make("bottom-track-pants", "Track pants", { inseam: 78, thigh: 36, open: 18 }, { cuff: "ribbed", waistband: "elastic" }),
    // Casual / seasonal shorts
    make("bottom-denim-shorts", "Denim shorts", { inseam: 26, thigh: 34, open: 28 }),
    make("bottom-chino-shorts", "Chino shorts", { inseam: 24, thigh: 32, open: 26 }),
    make("bottom-cargo-shorts", "Cargo shorts", { inseam: 26, thigh: 36, open: 30 }, { cargo: true }),
    make("bottom-board-shorts", "Beach / swim shorts", { inseam: 30, thigh: 36, open: 32 }, { waistband: "elastic" }),
    // Other / special
    make("bottom-work-pants", "Work pants (utility)", { inseam: 80, thigh: 36, open: 24 }, { cargo: true }),
    make("bottom-harem-pants", "Harem pants", { inseam: 72, thigh: 44, open: 14, rise: 34, hip: 56 }, { waistband: "elastic" }),
    make("bottom-linen-pants", "Linen pants", { inseam: 80, thigh: 36, open: 24 }, { waistband: "elastic" }),
    make("bottom-straight-trousers", "Straight-leg trousers", { inseam: 82, thigh: 33, open: 24 }),
    // Biker shorts (short leggings)
    make("bottom-biker-shorts", "Short leggings / biker shorts", {
      inseam: 22,
      thigh: 26,
      open: 24,
      waist: 37,
      hip: 46,
      depth: 19,
    }, { waistband: "elastic" }),
  ];
}

/**
 * Skirt archetype: a single waist-to-hem loft; types differ by length and
 * how much the hem flares past the hips.
 */
function skirtPresets(): GarmentTemplateDef[] {
  const make = (
    id: string,
    label: string,
    dims: { length: number; hem: number; hip?: number; waist?: number; rise?: number },
  ): GarmentTemplateDef => ({
    id,
    label,
    category: "Bottoms",
    gender: "female",
    params: {
      ...defaultTeeParams,
      bodyDepth: 20,
      waistWidth: dims.waist ?? 36,
      hipWidth: dims.hip ?? 50,
      rise: dims.rise ?? 18,
      inseam: dims.length,
      legOpening: dims.hem,
    },
    features: { ...defaultTeeFeatures, archetype: "skirt" },
  });
  return [
    make("skirt-mini", "Mini skirt", { length: 40, hem: 54 }),
    make("skirt-midi", "Midi skirt", { length: 66, hem: 74 }),
    make("skirt-maxi", "Maxi skirt", { length: 98, hem: 96 }),
    make("skirt-pencil", "Pencil skirt", { length: 62, hem: 44, hip: 50 }),
  ];
}

/**
 * Dresses: a bodice (top archetype) with a lofted skirt attached at the hem.
 * They differ by neckline, sleeves, skirt length and flare.
 */
function dressPresets(): GarmentTemplateDef[] {
  const base: GarmentParams = {
    ...defaultTeeParams,
    bodyLength: 62,
    bodyWidth: 46,
    bodyDepth: 20,
    neckWidthFront: 17,
  };
  return [
    {
      id: "dress-aline",
      label: "A-line dress",
      category: "Dresses",
      gender: "female",
      params: { ...base, neckDropFront: 9, shoulderWidthFactor: 0.62 },
      features: {
        ...defaultTeeFeatures,
        sleeves: false,
        skirtLength: 46,
        skirtFlare: 0.55,
      },
    },
    {
      id: "dress-bodycon",
      label: "Bodycon dress",
      category: "Dresses",
      gender: "female",
      params: { ...base, bodyWidth: 42, neckDropFront: 8, shoulderWidthFactor: 0.6 },
      features: {
        ...defaultTeeFeatures,
        sleeves: false,
        skirtLength: 40,
        skirtFlare: 0.05,
      },
    },
    {
      id: "dress-slip",
      label: "Slip dress",
      category: "Dresses",
      gender: "female",
      params: { ...base, bodyWidth: 43, neckWidthFront: 20, neckDropFront: 11, shoulderWidthFactor: 0.4 },
      features: {
        ...defaultTeeFeatures,
        sleeves: false,
        skirtLength: 52,
        skirtFlare: 0.2,
      },
    },
    {
      id: "dress-sundress",
      label: "Sundress",
      category: "Dresses",
      gender: "female",
      params: { ...base, bodyWidth: 44, bodyLength: 46, neckWidthFront: 24, neckDropFront: 4, neckDropBack: 4, shoulderWidthFactor: 0.34 },
      features: {
        ...defaultTeeFeatures,
        sleeves: false,
        skirtLength: 40,
        skirtFlare: 0.6,
      },
    },
    {
      id: "dress-wrap",
      label: "Wrap dress",
      category: "Dresses",
      gender: "female",
      params: { ...base, neckWidthFront: 16, neckDropFront: 16, sleeveLength: 22 },
      features: {
        ...defaultTeeFeatures,
        neckFinish: "band",
        placket: "half",
        skirtLength: 48,
        skirtFlare: 0.5,
      },
    },
    {
      id: "dress-shirt",
      label: "Shirt dress",
      category: "Dresses",
      gender: "female",
      params: { ...base, bodyWidth: 50, neckWidthFront: 15.5, neckDropFront: 6, sleeveLength: 22 },
      features: {
        ...defaultTeeFeatures,
        neckFinish: "shirt-collar",
        placket: "full",
        cuff: "barrel",
        skirtLength: 46,
        skirtFlare: 0.3,
      },
    },
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
    // 70 cm body, no hem band.
    params: { ...defaultTeeParams, ...longSleeve, sleeveLength: 56 },
    features: { ...defaultTeeFeatures, sleeveTaper: 0.72 },
  },
  {
    id: "top-sweatshirt",
    label: "Sweatshirt (crewneck)",
    category: "Tops",
    params: {
      ...defaultTeeParams,
      ...longSleeve,
      // 70 cm body + hem band.
      sleeveLength: 51,
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
      fabric: "fleece",
    },
  },
  {
    id: "top-hoodie",
    label: "Hoodie",
    category: "Tops",
    params: {
      ...defaultTeeParams,
      ...longSleeve,
      // 70 cm body + hem band.
      sleeveLength: 51,
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
      fabric: "fleece",
    },
  },
  {
    id: "top-polo",
    label: "Polo shirt",
    category: "Tops",
    params: { ...defaultTeeParams, neckWidthFront: 16, neckDropFront: 7 },
    features: {
      ...defaultTeeFeatures,
      neckFinish: "polo-collar",
      placket: "half",
    },
  },
  // Button-front woven shirts: identical geometry, different fabric/fit.
  ...wovenShirtPresets(),
  {
    id: "top-sweater",
    label: "Pullover / sweater",
    category: "Tops",
    params: {
      ...defaultTeeParams,
      ...longSleeve,
      // 70 cm body + hem band.
      sleeveLength: 52,
      bodyWidth: 56,
      bodyDepth: 25,
      neckWidthFront: 18,
      neckDropFront: 7,
    },
    features: {
      neckFinish: "band",
      hemBand: true,
      cuff: "ribbed",
      sleeveTaper: 0.62,
      fabric: "knit",
    },
  },
  {
    id: "top-turtleneck",
    label: "Turtleneck / roll neck",
    category: "Tops",
    gender: "male",
    params: {
      ...defaultTeeParams,
      ...longSleeve,
      // 70 cm body, no hem band.
      sleeveLength: 57,
      bodyWidth: 50,
      bodyDepth: 22,
      neckWidthFront: 15,
      neckDropFront: 3.5,
      neckDropBack: 2,
    },
    features: {
      neckFinish: "turtleneck",
      hemBand: false,
      // Straight sleeve end — the chunky ribbed band belongs to hoodies and
      // pullovers only.
      cuff: "raw",
      sleeveTaper: 0.58,
      fabric: "knit",
    },
  },
  {
    id: "top-turtleneck-f",
    label: "Turtleneck / roll neck",
    category: "Tops",
    gender: "female",
    // Slim women's roll neck: narrower body, fitted waist shaping, closer
    // shoulders and sleeves, shorter length — a streamlined silhouette, not
    // the relaxed men's tube.
    params: {
      ...defaultTeeParams,
      ...longSleeve,
      // 62 cm body, no hem band.
      sleeveLength: 50,
      bodyWidth: 43,
      bodyDepth: 18,
      bodyLength: 62,
      neckWidthFront: 14,
      neckDropFront: 3.5,
      neckDropBack: 2,
      shoulderWidthFactor: 0.72,
    },
    features: {
      neckFinish: "turtleneck",
      hemBand: false,
      cuff: "raw",
      sleeveTaper: 0.5,
      fabric: "knit",
      waistPinch: 0.14,
    },
  },
  {
    id: "top-cardigan",
    label: "Cardigan",
    category: "Tops",
    params: {
      ...defaultTeeParams,
      ...longSleeve,
      // 70 cm body + hem band.
      sleeveLength: 52,
      bodyWidth: 55,
      bodyDepth: 24,
      neckWidthFront: 17,
      neckDropFront: 10,
    },
    features: {
      neckFinish: "band",
      hemBand: true,
      cuff: "ribbed",
      sleeveTaper: 0.62,
      placket: "full",
      fabric: "knit",
    },
  },
  // Undergarment / sleeveless
  {
    id: "top-tank",
    label: "Tank top / sleeveless",
    category: "Tops",
    params: {
      ...defaultTeeParams,
      bodyWidth: 47,
      bodyDepth: 20,
      neckWidthFront: 17,
      neckDropFront: 10,
      armholeDepth: 23,
      shoulderWidthFactor: 0.68,
    },
    features: { ...defaultTeeFeatures, sleeves: false },
  },
  {
    id: "top-undershirt",
    label: "Undershirt",
    category: "Tops",
    params: {
      ...defaultTeeParams,
      bodyWidth: 45,
      bodyDepth: 18,
      neckDropFront: 9.5,
      neckWidthFront: 19,
      armholeDepth: 24,
      shoulderWidthFactor: 0.58,
    },
    features: { ...defaultTeeFeatures, sleeves: false },
  },
  {
    id: "top-compression",
    label: "Compression / base layer",
    category: "Tops",
    params: {
      ...defaultTeeParams,
      ...longSleeve,
      // 70 cm body, no hem band.
      sleeveLength: 58,
      bodyWidth: 44,
      bodyDepth: 17,
      neckWidthFront: 16,
      neckDropFront: 5,
    },
    features: { ...defaultTeeFeatures, sleeveTaper: 0.5 },
  },
  // Cropped / fitted tops
  {
    id: "top-crop-tee",
    label: "Crop top",
    category: "Tops",
    params: {
      ...defaultTeeParams,
      bodyWidth: 44,
      bodyLength: 42,
      bodyDepth: 18,
      neckWidthFront: 17,
      neckDropFront: 7,
      sleeveLength: 16,
    },
    features: { ...defaultTeeFeatures, sleeveTaper: 0.9 },
  },
  {
    id: "top-cropped-tank",
    label: "Cropped tank top",
    category: "Tops",
    params: {
      ...defaultTeeParams,
      bodyWidth: 42,
      bodyLength: 42,
      bodyDepth: 17,
      neckWidthFront: 18,
      neckDropFront: 9,
      armholeDepth: 24,
      shoulderWidthFactor: 0.5,
    },
    features: { ...defaultTeeFeatures, sleeves: false },
  },
  {
    id: "top-camp-shirt",
    label: "Camp shirt",
    category: "Tops",
    params: {
      ...defaultTeeParams,
      bodyWidth: 55,
      bodyDepth: 23,
      bodyLength: 68,
      neckWidthFront: 17,
      neckDropFront: 8,
      sleeveLength: 24,
    },
    features: {
      neckFinish: "shirt-collar",
      hemBand: false,
      cuff: "raw",
      sleeveTaper: 0.9,
      placket: "full",
      fabric: "woven",
    },
  },
  // Sports tops
  {
    id: "top-gym-tee",
    label: "Gym T-shirt",
    category: "Tops",
    params: { ...defaultTeeParams, bodyWidth: 51, bodyDepth: 21 },
    features: { ...defaultTeeFeatures },
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
      // 70 cm body, no hem band.
      sleeveLength: 55,
      armholeDepth: 26,
    },
    features: { ...defaultTeeFeatures, sleeveTaper: 0.75 },
  },
  // --- Bottoms archetype ---------------------------------------------------
  ...bottomsPresets(),
  ...skirtPresets(),
  // --- Dresses (bodice + lofted skirt) -------------------------------------
  ...dressPresets(),
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
  "top-sports-jersey": "top-gym-tee",
  "top-basketball-jersey": "top-tank",
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

/** True when a type is offered to the given gender (unisex types match both). */
function matchesGender(t: GarmentTemplateDef, gender: Gender): boolean {
  return !t.gender || t.gender === gender;
}

/** Garment types for a gender + category, narrowing the dropdown. */
export function templatesFor(
  gender: Gender,
  category: string,
): GarmentTemplateDef[] {
  const matches = garmentTemplates.filter(
    (t) => t.category === category && matchesGender(t, gender),
  );
  return matches.length > 0 ? matches : templatesForCategory(category);
}

/** Categories that actually have a type for the given gender, in display order. */
export function categoriesForGender(gender: Gender): string[] {
  const present = new Set(
    garmentTemplates.filter((t) => matchesGender(t, gender)).map((t) => t.category),
  );
  return CATEGORY_ORDER.filter((c) => present.has(c));
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
