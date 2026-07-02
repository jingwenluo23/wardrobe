import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,
  // Native/ONNX packages must stay external to the server bundle.
  serverExternalPackages: [
    "@huggingface/transformers",
    "onnxruntime-node",
    "sharp",
  ],
  turbopack: {
    root: process.cwd(),
  },
};

export default nextConfig;
