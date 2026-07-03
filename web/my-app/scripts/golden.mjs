// Golden-image tests for every garment template.
//
// Renders each registry template on the /golden/<id> page and compares the
// screenshot against goldens/<id>.png, so a geometry fix for one archetype
// cannot silently deform another.
//
// Usage (with the app already running, e.g. `npm run build && npm start`):
//   node scripts/golden.mjs --update   # (re)record all goldens
//   node scripts/golden.mjs            # compare against recorded goldens
//
// Env: BASE_URL (default http://localhost:3000)
//      CHROMIUM_PATH (optional explicit browser binary)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright-core";
import sharp from "sharp";

const BASE_URL = process.env.BASE_URL ?? "http://localhost:3000";
const UPDATE = process.argv.includes("--update");
// Mean absolute per-channel difference (0-255) allowed before failing.
const THRESHOLD = 3;

const rootDir = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const goldenDir = path.join(rootDir, "goldens");
fs.mkdirSync(goldenDir, { recursive: true });

async function meanAbsDiff(bufferA, bufferB) {
  const size = 320;
  const [a, b] = await Promise.all(
    [bufferA, bufferB].map((buffer) =>
      sharp(buffer)
        .resize(size, size, { fit: "fill" })
        .removeAlpha()
        .raw()
        .toBuffer(),
    ),
  );
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    total += Math.abs(a[i] - b[i]);
  }
  return total / a.length;
}

const response = await fetch(BASE_URL + "/api/templates");
if (!response.ok) {
  console.error("Cannot reach " + BASE_URL + " - start the app first.");
  process.exit(2);
}
const { templates } = await response.json();

const executablePath =
  process.env.CHROMIUM_PATH ??
  (fs.existsSync("/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    ? "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
    : undefined);

const browser = await chromium.launch({
  executablePath,
  args: ["--no-sandbox", "--use-gl=swiftshader", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage({ viewport: { width: 720, height: 720 } });

let failures = 0;
for (const template of templates) {
  await page.goto(BASE_URL + "/golden/" + template.id, {
    waitUntil: "domcontentloaded",
  });
  await page.waitForSelector("canvas");
  await page.waitForTimeout(1800);
  const shot = await page.locator("canvas").screenshot();

  const goldenPath = path.join(goldenDir, template.id + ".png");
  if (UPDATE || !fs.existsSync(goldenPath)) {
    fs.writeFileSync(goldenPath, shot);
    console.log("recorded", template.id);
    continue;
  }
  const diff = await meanAbsDiff(shot, fs.readFileSync(goldenPath));
  if (diff > THRESHOLD) {
    failures += 1;
    fs.writeFileSync(path.join(goldenDir, template.id + ".actual.png"), shot);
    console.error(
      "FAIL " + template.id + " (diff " + diff.toFixed(2) + " > " + THRESHOLD + ")",
    );
  } else {
    console.log("ok   " + template.id + " (diff " + diff.toFixed(2) + ")");
  }
}

await browser.close();
if (failures > 0) {
  console.error(failures + " template(s) changed - inspect *.actual.png");
  process.exit(1);
}
console.log("all templates match");
