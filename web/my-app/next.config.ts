import path from "node:path";
import { fileURLToPath } from "node:url";
import type { NextConfig } from "next";

const configDir = path.dirname(fileURLToPath(import.meta.url));

const nextConfig: NextConfig = {
  reactCompiler: true,
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
