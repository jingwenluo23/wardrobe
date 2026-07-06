import path from "node:path";
import { fileURLToPath } from "node:url";
import type { NextConfig } from "next";

const configDir = path.dirname(fileURLToPath(import.meta.url));

const nextConfig: NextConfig = {
  reactCompiler: true,
  // Native/ONNX packages must stay external to the server bundle.
  serverExternalPackages: [
    "@huggingface/transformers",
    "onnxruntime-node",
    "sharp",
    "better-sqlite3",
  ],
  turbopack: {
    root: configDir,
  },
  async redirects() {
    return [
      {
        source: "/settings/model-standards",
        destination: "/settings",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
