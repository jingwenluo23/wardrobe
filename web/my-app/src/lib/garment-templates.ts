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
  {
    id: "bottom-straight",
    label: "Straight bottom",
    category: "Bottoms",
    params: { ...defaultTeeParams, bodyWidth: 46, bodyLength: 100 },
    features: { ...defaultTeeFeatures },
  },
  {
    id: "shoe-low",
    label: "Low shoe",
    category: "Shoes",
    params: { ...defaultTeeParams },
    features: { ...defaultTeeFeatures },
  },
];

const byId = new Map(garmentTemplates.map((t) => [t.id, t]));

export function getTemplate(id: string | null | undefined) {
  return (id && byId.get(id)) || null;
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
