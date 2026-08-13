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
  // out to 124cm. Keep the fitted length in a band
  // around the template, which already encodes the right length for the
  // garment type the user picked.
  const targetLength = clamp(
    bodyWidth * clamp(shape.bodyAspectRatio, 0.75, 2.25),
    templateParams.bodyLength * 0.9,
    templateParams.bodyLength * 1.12,
  );
  const chest = Math.max(0.55, shape.chestRatio);
  const targetShoulder = clamp(
    (shape.shoulderRatio / chest) * 0.8,
    0.5,
    0.98,
  );
  const targetHem = clamp(shape.hemRatio / chest, 0.76, 1.38);
  const targetPinch = clamp(1 - shape.waistRatio / chest, 0, 0.2);
  const fittedLength = mix(templateParams.bodyLength, targetLength, weight);
  // The sleeve belongs to the same garment as the body.
  //
  // bodyLength and sleeveLength were fitted independently and could move in
  // OPPOSITE directions. On the striped dress shirt that is exactly what
  // happened: the body fitted out from 74cm to 82cm while the sideways sleeve
  // estimate pulled the other way, 61cm down to about 52cm — a 26% swing
  // between them — and the cuff finished roughly 18cm above the hem. A longer
  // body is a larger garment, and a larger garment has longer sleeves, so the
  // sleeve now follows the body by default.
  const lengthScale = fittedLength / templateParams.bodyLength;
  const scaledSleeve = templateParams.sleeveLength * lengthScale;
  // spanRatio only measures how far the silhouette reaches SIDEWAYS, so it can
  // only see a sleeve's LENGTH when the sleeves are held out. Straight out, the
  // span is body + two sleeves and a shirt reads about 3.3; at 45 degrees it is
  // still about 2.6. On a hanger with the sleeves hanging down it reads about
  // 1.3, and the sideways reach there is a small fraction of the real sleeve.
  // The old 1.45 threshold fired on those ordinary hanger and flat-lay shots
  // and shortened the sleeve to that fraction.
  const sleevesExtendSideways = shape.spanRatio > 2.1;
  // Even when the sleeves are out, perspective and a soft mask make this a
  // rough read, so keep it near the body's own proportion.
  const targetSleeve = clamp(
    (bodyWidth * Math.max(0, shape.spanRatio - 1)) / 2,
    scaledSleeve * 0.8,
    scaledSleeve * 1.2,
  );
  const targetNeckWidth = clamp(
    bodyWidth * shape.neckWidthRatio,
    bodyWidth * 0.16,
    bodyWidth * 0.52,
  );
  const targetNeckDrop = clamp(
    fittedLength * shape.neckDepthRatio,
    1.5,
    fittedLength * 0.28,
  );

  return {
    params: {
      ...templateParams,
      bodyLength: fittedLength,
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
        templateFeatures.sleeves === false
          ? templateParams.sleeveLength
          : sleevesExtendSideways
            ? mix(scaledSleeve, targetSleeve, weight)
            : scaledSleeve,
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
