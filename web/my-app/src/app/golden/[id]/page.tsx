"use client";

// Golden-test render page: shows a registry template as a flat-colour mesh
// with a fixed camera, so scripts/golden.mjs can screenshot every garment
// type deterministically. Not linked from the app UI.

import dynamic from "next/dynamic";
import { use } from "react";

import type { DraftMesh } from "@/lib/garment-mesh";
import { getTemplate } from "@/lib/garment-templates";

const GarmentMeshViewer = dynamic(
  () => import("../../outfits/garment-mesh-viewer"),
  { ssr: false },
);

export default function GoldenPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const template = getTemplate(id);

  if (!template) {
    return <main data-golden="missing">Unknown template: {id}</main>;
  }

  const mesh: DraftMesh = {
    assetUrl: "golden://" + template.id,
    generatedAt: 0,
    template: template.id,
    templateLabel: template.label,
    params: template.params,
    features: template.features,
    segmentation: { confidence: 1 },
    color: "#9a9da4",
    bounds: { width: 0, height: 0, depth: 0 },
  };

  return (
    <main data-golden={template.id} style={{ width: 640, height: 640 }}>
      <GarmentMeshViewer mesh={mesh} />
    </main>
  );
}
