import type {
  GarmentFeatures,
  GarmentParams,
  GarmentShapeEstimate,
} from "./garment-mesh";

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const mix = (a: number, b: number, weight: number) =>
  a + (b - a) * clamp(weight, 0, 1);

export function mergeShapeEstimates(
  front?: GarmentShapeEstimate,
  back?: GarmentShapeEstimate,
): GarmentShapeEstimate | undefined {
  const estimates = [front, back].filter(
    (shape): shape is GarmentShapeEstimate => Boolean(shape),
  );
  if (estimates.length === 0) return undefined;
  const total = estimates.reduce((sum, shape) => sum + shape.confidence, 0);
  const average = (key: keyof Omit<GarmentShapeEstimate, "confidence">) =>
    estimates.reduce(
      (sum, shape) => sum + shape[key] * shape.confidence,
      0,
    ) / total;
  return {
    bodyAspectRatio: average("bodyAspectRatio"),
    spanRatio: average("spanRatio"),
    shoulderRatio: average("shoulderRatio"),
    chestRatio: average("chestRatio"),
    waistRatio: average("waistRatio"),
    hemRatio: average("hemRatio"),
    neckWidthRatio: front?.neckWidthRatio ?? average("neckWidthRatio"),
    neckDepthRatio: front?.neckDepthRatio ?? average("neckDepthRatio"),
    confidence:
      estimates.reduce((sum, shape) => sum + shape.confidence, 0) /
      estimates.length,
  };
}

/**
 * Fit observable 2D proportions while retaining the template's scale and
 * manufacturing-friendly topology. Conservative blending prevents a poor
 * segmentation or perspective-heavy photo from distorting the garment.
 */
export function fitGarmentToShape(
  templateParams: GarmentParams,
  templateFeatures: GarmentFeatures,
  front?: GarmentShapeEstimate,
  back?: GarmentShapeEstimate,
): {
  params: GarmentParams;
  features: GarmentFeatures;
  confidence: number;
} {
  const shape = mergeShapeEstimates(front, back);
  if (!shape) {
    return {
      params: { ...templateParams },
      features: { ...templateFeatures },
      confidence: 0,
    };
  }

  // Cap photo influence at 72%: the template supplies depth, construction,
  // and plausible limits that two uncalibrated views cannot recover.
  const weight = clamp((shape.confidence - 0.25) / 0.73, 0, 1) * 0.72;
  if (
    templateFeatures.archetype === "bottoms" ||
    templateFeatures.archetype === "skirt"
  ) {
    const hip = templateParams.hipWidth ?? templateParams.bodyWidth;
    const rise = templateParams.rise ?? 0;
    const targetTotalLength = hip * clamp(shape.bodyAspectRatio, 0.45, 2.7);
    const targetWaist = hip * clamp(shape.shoulderRatio, 0.55, 1);
    const targetOpening = hip * clamp(shape.hemRatio, 0.18, 1.45);
    return {
      params: {
        ...templateParams,
        waistWidth: mix(
          templateParams.waistWidth ?? hip * 0.8,
          targetWaist,
          weight,
        ),
        inseam: mix(
          templateParams.inseam ?? targetTotalLength,
          Math.max(6, targetTotalLength - rise),
          weight,
        ),
        legOpening: mix(
          templateParams.legOpening ?? hip * 0.5,
          targetOpening,
          weight,
        ),
      },
      features: { ...templateFeatures },
      confidence: shape.confidence,
    };
  }

  const bodyWidth = templateParams.bodyWidth;
  // A photo can say a top is cropped or longline. It cannot say it is a dress.
  //
  // bodyAspectRatio comes from a silhouette, so it carries every distortion a
  // phone photo has: a garment shot from above foreshortens, one on a hanger
  // stretches, a hood or collar adds height the mesh's bodyLength does not
  // include. Multiplying bodyWidth by an unbounded ratio let a 70cm tee fit
  // out to 124cm, and because sleeveLength is almost always kept from the
  // template (see sleevesExtendSideways below), the torso ran away from the
  // sleeves instead of growing with them. Keep the fitted length in a band
  // around the template, which already encodes the right length for the
  // garment type the user picked.
  const targetLength = clamp(
    bodyWidth * clamp(shape.bodyAspectRatio, 0.75, 2.25),
    templateParams.bodyLength * 0.86,
    templateParams.bodyLength * 1.18,
  );
  const chest = Math.max(0.55, shape.chestRatio);
  const targetShoulder = clamp(
    (shape.shoulderRatio / chest) * 0.8,
    0.5,
    0.98,
  );
  const targetHem = clamp(shape.hemRatio / chest, 0.76, 1.38);
  const targetPinch = clamp(1 - shape.waistRatio / chest, 0, 0.2);
  const targetSleeve = clamp(
    (bodyWidth * Math.max(0, shape.spanRatio - 1)) / 2,
    4,
    bodyWidth * 0.85,
  );
  // spanRatio only measures how far the silhouette reaches SIDEWAYS. A sleeve
  // photographed on a body with the arms hanging down runs vertically and adds
  // almost nothing to the span, so this estimate collapses a 56cm hoodie sleeve
  // to a ~4cm stub. Only trust it when the sleeves clearly extend to the sides
  // (arms out, or a flat lay); otherwise keep the template's length, which
  // already encodes long vs short from the garment type the user chose.
  const sleevesExtendSideways = shape.spanRatio > 1.45;
  const targetNeckWidth = clamp(
    bodyWidth * shape.neckWidthRatio,
    bodyWidth * 0.16,
    bodyWidth * 0.52,
  );
  const fittedLength = mix(templateParams.bodyLength, targetLength, weight);
  const targetNeckDrop = clamp(
    fittedLength * shape.neckDepthRatio,
    1.5,
    fittedLength * 0.28,
  );

  return {
    params: {
      ...templateParams,
      bodyLength: mix(templateParams.bodyLength, targetLength, weight),
      shoulderWidthFactor: mix(
        templateParams.shoulderWidthFactor ?? 0.8,
        targetShoulder,
        weight,
      ),
      hemWidthFactor: mix(
        templateParams.hemWidthFactor ?? 1.03,
        targetHem,
        weight,
      ),
      sleeveLength:
        templateFeatures.sleeves === false || !sleevesExtendSideways
          ? templateParams.sleeveLength
          : mix(templateParams.sleeveLength, targetSleeve, weight),
      neckWidthFront: mix(
        templateParams.neckWidthFront,
        targetNeckWidth,
        weight * 0.8,
      ),
      neckDropFront: mix(
        templateParams.neckDropFront,
        targetNeckDrop,
        weight * 0.75,
      ),
    },
    features: {
      ...templateFeatures,
      waistPinch: mix(
        templateFeatures.waistPinch ?? 0,
        targetPinch,
        weight,
      ),
    },
    confidence: shape.confidence,
  };
}
