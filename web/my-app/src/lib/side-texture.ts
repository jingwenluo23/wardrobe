import sharp from "sharp";

const SIDE_TILE_SIZE = 512;
const EDGE_BAND_RATIO = 0.08;

function dataUrlBuffer(url: string): Buffer {
  const comma = url.indexOf(",");
  if (comma < 0) {
    throw new Error("Invalid texture data URL.");
  }
  return Buffer.from(url.slice(comma + 1), "base64");
}

async function texturePixels(url: string) {
  const { data, info } = await sharp(dataUrlBuffer(url))
    .resize(SIDE_TILE_SIZE, SIDE_TILE_SIZE, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { data, width: info.width, height: info.height };
}

function smoothstep(value: number, low: number, high: number) {
  const t = Math.max(0, Math.min(1, (value - low) / (high - low)));
  return t * t * (3 - 2 * t);
}

function averageBand(
  pixels: Buffer,
  width: number,
  y: number,
  left: boolean,
): [number, number, number] {
  const band = Math.max(4, Math.round(width * EDGE_BAND_RATIO));
  const x0 = left ? 0 : width - band;
  const x1 = left ? band : width;
  let r = 0;
  let g = 0;
  let b = 0;
  let weight = 0;

  // Give pixels nearer the actual garment boundary more influence while
  // retaining a wide enough strip to preserve prints that approach the seam.
  for (let x = x0; x < x1; x += 1) {
    const edgeDistance = left ? x - x0 : x1 - 1 - x;
    const w = 1 + (band - edgeDistance) / band;
    const i = (y * width + x) * 3;
    r += pixels[i] * w;
    g += pixels[i + 1] * w;
    b += pixels[i + 2] * w;
    weight += w;
  }
  return [r / weight, g / weight, b / weight];
}

function colorDistance(
  a: readonly number[],
  b: readonly number[],
): number {
  return Math.sqrt(
    (a[0] - b[0]) ** 2 +
      (a[1] - b[1]) ** 2 +
      (a[2] - b[2]) ** 2,
  );
}

/**
 * Build a plausible side atlas from the visible boundary strips of the front
 * and back garment textures.
 *
 * The atlas runs front -> back along X and neck -> hem along Y. When neither
 * boundary contains a print, the output stays as the sampled fabric. When a
 * stripe or graphic reaches an edge, its colour is continued across the side
 * and blended toward the evidence at the opposite edge.
 */
export async function synthesizeSideTexture(input: {
  frontTextureUrl: string;
  backTextureUrl: string;
  fabricTextureUrl?: string;
}): Promise<string> {
  const [front, back, fabric] = await Promise.all([
    texturePixels(input.frontTextureUrl),
    texturePixels(input.backTextureUrl),
    input.fabricTextureUrl
      ? texturePixels(input.fabricTextureUrl)
      : texturePixels(input.frontTextureUrl),
  ]);

  const output = Buffer.alloc(SIDE_TILE_SIZE * SIDE_TILE_SIZE * 3);

  for (let y = 0; y < SIDE_TILE_SIZE; y += 1) {
    const frontLeft = averageBand(front.data, front.width, y, true);
    const frontRight = averageBand(front.data, front.width, y, false);
    const backLeft = averageBand(back.data, back.width, y, true);
    const backRight = averageBand(back.data, back.width, y, false);

    // One side atlas is shared by the left and right mesh edges. Prefer the
    // boundary with stronger non-fabric evidence so a stripe reaching only
    // one photographed side is not averaged away.
    const rowBaseIndex = (y * fabric.width + Math.floor(fabric.width / 2)) * 3;
    const rowBase = [
      fabric.data[rowBaseIndex],
      fabric.data[rowBaseIndex + 1],
      fabric.data[rowBaseIndex + 2],
    ] as const;
    const frontEdge =
      colorDistance(frontLeft, rowBase) >= colorDistance(frontRight, rowBase)
        ? frontLeft
        : frontRight;
    const backEdge =
      colorDistance(backLeft, rowBase) >= colorDistance(backRight, rowBase)
        ? backLeft
        : backRight;
    const designStrength = Math.max(
      colorDistance(frontEdge, rowBase),
      colorDistance(backEdge, rowBase),
    );
    const designMix = smoothstep(designStrength, 18, 58);

    for (let x = 0; x < SIDE_TILE_SIZE; x += 1) {
      const t0 = x / (SIDE_TILE_SIZE - 1);
      const t = t0 * t0 * (3 - 2 * t0);
      const fabricIndex = (y * fabric.width + x) * 3;
      for (let channel = 0; channel < 3; channel += 1) {
        const base = fabric.data[fabricIndex + channel];
        const bridge =
          frontEdge[channel] +
          (backEdge[channel] - frontEdge[channel]) * t;
        output[(y * SIDE_TILE_SIZE + x) * 3 + channel] = Math.round(
          base + (bridge - base) * designMix,
        );
      }
    }
  }

  const jpeg = await sharp(output, {
    raw: {
      width: SIDE_TILE_SIZE,
      height: SIDE_TILE_SIZE,
      channels: 3,
    },
  })
    .blur(0.35)
    .jpeg({ quality: 82 })
    .toBuffer();

  return "data:image/jpeg;base64," + jpeg.toString("base64");
}

/**
 * Bake the side atlas into the curved outer UV bands of a front or back
 * panel. The existing mesh already wraps those bands toward the side seam,
 * so this supplies a side surface without adding overlapping geometry or a
 * custom shader.
 */
export async function blendSideIntoPanel(input: {
  panelTextureUrl: string;
  sideTextureUrl: string;
  surface: "front" | "back";
}): Promise<string> {
  const [panel, side] = await Promise.all([
    texturePixels(input.panelTextureUrl),
    texturePixels(input.sideTextureUrl),
  ]);
  const output = Buffer.from(panel.data);
  const band = Math.round(SIDE_TILE_SIZE * 0.16);

  for (let y = 0; y < SIDE_TILE_SIZE; y += 1) {
    for (let x = 0; x < SIDE_TILE_SIZE; x += 1) {
      const edgeDistance = Math.min(x, SIDE_TILE_SIZE - 1 - x);
      if (edgeDistance >= band) {
        continue;
      }
      const seamAmount = 1 - edgeDistance / band;
      const blend = smoothstep(seamAmount, 0, 1);
      // X=0 is the front boundary of the side atlas, X=0.5 is the
      // lateral seam, and X=1 is the back boundary.
      const sideU =
        input.surface === "front"
          ? seamAmount * 0.5
          : 1 - seamAmount * 0.5;
      const sideX = Math.max(
        0,
        Math.min(SIDE_TILE_SIZE - 1, Math.round(sideU * (SIDE_TILE_SIZE - 1))),
      );
      const panelIndex = (y * SIDE_TILE_SIZE + x) * 3;
      const sideIndex = (y * SIDE_TILE_SIZE + sideX) * 3;
      for (let channel = 0; channel < 3; channel += 1) {
        output[panelIndex + channel] = Math.round(
          panel.data[panelIndex + channel] * (1 - blend) +
            side.data[sideIndex + channel] * blend,
        );
      }
    }
  }

  const jpeg = await sharp(output, {
    raw: {
      width: SIDE_TILE_SIZE,
      height: SIDE_TILE_SIZE,
      channels: 3,
    },
  })
    .jpeg({ quality: 84 })
    .toBuffer();

  return "data:image/jpeg;base64," + jpeg.toString("base64");
}
