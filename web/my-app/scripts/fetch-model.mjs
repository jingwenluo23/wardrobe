// Vendor the clothes-segmentation model into the app.
//
// transformers.js fetches model weights from huggingface.co the first time the
// pipeline runs. That makes garment extraction depend on an outbound request
// during a user's upload: if the host cannot reach huggingface.co, or the
// request is throttled, extraction silently drops to the colour heuristic and
// the texture comes back visibly worse. Downloading the files ahead of time —
// at build or deploy — turns that into a local read.
//
//   npm run fetch-model
//
// Files land in models/<repo>/, which is where MODEL_DIR points the runtime.
// Safe to re-run: existing files are left alone. The script never fails the
// build; if the download does not work the runtime still falls back to
// fetching remotely, exactly as before.

import { createWriteStream } from "node:fs";
import { mkdir, stat, rename, unlink } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { pipeline as streamPipeline } from "node:stream/promises";
import { Readable } from "node:stream";

// Node's built-in fetch ignores HTTPS_PROXY unless NODE_USE_ENV_PROXY is set,
// so on a host that only reaches the internet through a proxy — corporate
// networks, CI runners, sandboxed build agents — every download here fails with
// a proxy error while curl on the same box succeeds.
//
// The flag is read when fetch's HTTP stack first initialises, which has already
// happened by the time this module runs, so assigning process.env here has no
// effect. Re-exec once with it set instead. Portable, needs no dependency, and
// costs one extra process only on proxied hosts.
if (
  (process.env.HTTPS_PROXY || process.env.https_proxy) &&
  !process.env.NODE_USE_ENV_PROXY
) {
  const { spawnSync } = await import("node:child_process");
  const { status } = spawnSync(
    process.execPath,
    [process.argv[1], ...process.argv.slice(2)],
    { stdio: "inherit", env: { ...process.env, NODE_USE_ENV_PROXY: "1" } },
  );
  process.exit(status ?? 0);
}

const REPO = "Xenova/segformer_b2_clothes";
const REVISION = "main";

// The q8 pipeline needs the config, the image preprocessor config, and the
// quantised ONNX weights. Nothing else is read at runtime.
const FILES = [
  "config.json",
  "preprocessor_config.json",
  "onnx/model_quantized.onnx",
];

const outRoot = resolve(
  process.env.WARDROBE_MODEL_DIR ?? join(process.cwd(), "models"),
);

async function exists(path) {
  try {
    const info = await stat(path);
    return info.isFile() && info.size > 0;
  } catch {
    return false;
  }
}

async function download(file) {
  const target = join(outRoot, REPO, file);
  if (await exists(target)) {
    console.log("  have    " + file);
    return true;
  }
  const url =
    "https://huggingface.co/" + REPO + "/resolve/" + REVISION + "/" + file;
  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok || !response.body) {
    throw new Error("HTTP " + response.status + " for " + file);
  }
  await mkdir(dirname(target), { recursive: true });
  // Write to a temporary name first so an interrupted download can never leave
  // a truncated file that later looks present and valid.
  const partial = target + ".partial";
  await streamPipeline(
    Readable.fromWeb(response.body),
    createWriteStream(partial),
  );
  await rename(partial, target);
  const { size } = await stat(target);
  console.log(
    "  fetched " + file + " (" + (size / 1024 / 1024).toFixed(1) + " MB)",
  );
  return true;
}

console.log("Vendoring " + REPO + " into " + outRoot);
let ok = true;
for (const file of FILES) {
  try {
    await download(file);
  } catch (error) {
    ok = false;
    console.warn(
      "  FAILED  " + file + ": " + (error?.message ?? String(error)),
    );
    await unlink(join(outRoot, REPO, file + ".partial")).catch(() => {});
  }
}

if (ok) {
  console.log("Model vendored: extraction will load it from disk.");
} else {
  console.warn(
    "Model not fully vendored. The app still runs and will try to fetch the\n" +
      "model at request time; re-run `npm run fetch-model` from a host that can\n" +
      "reach huggingface.co to remove that dependency.",
  );
}
// Never fail a build over this — the runtime has a fallback.
process.exit(0);
