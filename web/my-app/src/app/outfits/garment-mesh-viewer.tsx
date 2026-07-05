"use client";

import { Canvas } from "@react-three/fiber";
import { Bounds, OrbitControls } from "@react-three/drei";
import { useEffect, useMemo } from "react";
import * as THREE from "three";

import type { DraftMesh } from "@/lib/garment-mesh";
import { buildGarmentGeometry } from "./garment-blocks";

type GarmentMeshViewerProps = {
  mesh: DraftMesh;
  className?: string;
  /** Stage colour behind the garment (default: studio beige). */
  background?: string;
  /** Slow continuous turntable, for ambient/hero displays. */
  autoRotate?: boolean;
  /** Container height (default 560). */
  height?: number | string;
};

function loadTexture(url?: string) {
  if (!url) {
    return null;
  }
  const loaded = new THREE.TextureLoader().load(url);
  loaded.colorSpace = THREE.SRGBColorSpace;
  loaded.wrapS = THREE.ClampToEdgeWrapping;
  loaded.wrapT = THREE.ClampToEdgeWrapping;
  return loaded;
}

function fabricMaterial(
  color: string,
  map: THREE.Texture | null,
  bump?: { texture: THREE.Texture; scale: number } | null,
) {
  return new THREE.MeshPhysicalMaterial({
    // three.js multiplies the map by the material colour, so textured
    // surfaces must stay white or the photo gets darkened and prints are
    // crushed. Plain surfaces carry the fabric colour directly.
    color: map ? "#ffffff" : color,
    map: map ?? undefined,
    bumpMap: bump?.texture,
    bumpScale: bump?.scale ?? 0,
    roughness: 0.88,
    metalness: 0,
    sheen: 0.55,
    sheenColor: new THREE.Color("#ffffff"),
    sheenRoughness: 0.9,
    side: THREE.FrontSide,
  });
}

/**
 * Procedural fabric relief: chunky vertical ribs for sweater knit, fine
 * noise for fleece. Rendered to a small repeating canvas bump map so heavy
 * fabrics read as material, not just colour.
 */
function makeFabricBump(kind: "knit" | "fleece") {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const g = canvas.getContext("2d");
  if (!g) {
    return null;
  }
  g.fillStyle = "#808080";
  g.fillRect(0, 0, size, size);
  if (kind === "knit") {
    // Rib columns with a subtle stitch wave.
    for (let x = 0; x < size; x += 8) {
      const grad = g.createLinearGradient(x, 0, x + 8, 0);
      grad.addColorStop(0, "#9a9a9a");
      grad.addColorStop(0.45, "#6e6e6e");
      grad.addColorStop(1, "#9a9a9a");
      g.fillStyle = grad;
      g.fillRect(x, 0, 8, size);
    }
    g.globalAlpha = 0.25;
    for (let y = 0; y < size; y += 5) {
      g.fillStyle = y % 10 === 0 ? "#707070" : "#909090";
      g.fillRect(0, y, size, 2);
    }
    g.globalAlpha = 1;
  } else {
    // Seeded PRNG so the fleece speckle is deterministic (stable goldens).
    let seed = 0x9e3779b9;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 0xffffffff;
    };
    for (let i = 0; i < 2600; i += 1) {
      const v = 118 + Math.floor(rand() * 20);
      g.fillStyle = "rgb(" + v + "," + v + "," + v + ")";
      g.fillRect(rand() * size, rand() * size, 1.5, 1.5);
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(kind === "knit" ? 10 : 6, kind === "knit" ? 10 : 6);
  return texture;
}

function TeeModel({ mesh }: { mesh: DraftMesh }) {
  const geometry = useMemo(
    () => buildGarmentGeometry(mesh.params, mesh.features),
    [mesh.params, mesh.features],
  );

  // Geometry groups: 0 = front panel, 1 = back panel, 2 = sleeves/collar.
  // The extracted front photo maps to the front, the back photo to the back,
  // and the trims stay plain fabric colour.
  const materials = useMemo(() => {
    const frontTexture = loadTexture(mesh.extractedTextureUrl);
    const backTexture = loadTexture(mesh.extractedBackTextureUrl) ?? frontTexture;
    // Sleeves/collar use a plain fabric swatch sampled from the photo, so
    // their shading matches the torso's real fabric instead of a flat tint.
    const fabricTexture = loadTexture(mesh.fabricTextureUrl);
    if (fabricTexture) {
      fabricTexture.wrapS = THREE.RepeatWrapping;
      fabricTexture.wrapT = THREE.RepeatWrapping;
    }
    const fabricKind = mesh.features?.fabric;
    const bumpTexture =
      fabricKind === "knit" || fabricKind === "fleece"
        ? makeFabricBump(fabricKind)
        : null;
    const bump = bumpTexture
      ? { texture: bumpTexture, scale: fabricKind === "knit" ? 2.2 : 0.7 }
      : null;
    return [
      fabricMaterial(mesh.color, frontTexture, bump),
      fabricMaterial(mesh.color, backTexture, bump),
      fabricMaterial(mesh.color, fabricTexture, bump),
    ];
  }, [
    mesh.color,
    mesh.extractedTextureUrl,
    mesh.extractedBackTextureUrl,
    mesh.fabricTextureUrl,
    mesh.features?.fabric,
  ]);

  const innerMaterial = useMemo(
    () =>
      new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(mesh.color).multiplyScalar(0.82),
        roughness: 0.95,
        metalness: 0,
        side: THREE.BackSide,
      }),
    [mesh.color],
  );

  useEffect(() => {
    return () => {
      geometry.dispose();
      materials.forEach((material) => {
        material.map?.dispose();
        material.dispose();
      });
      innerMaterial.dispose();
    };
  }, [geometry, materials, innerMaterial]);

  return (
    <group>
      {/* Outside: front/back photo textures on their panels, plain trims. */}
      <mesh geometry={geometry} material={materials} castShadow receiveShadow />
      {/* Inside: plain fabric, slightly shaded — prints stay outside only. */}
      <mesh geometry={geometry} material={innerMaterial} />
    </group>
  );
}

export default function GarmentMeshViewer({
  mesh,
  className,
  background = "#f1ece0",
  autoRotate = false,
  height = 560,
}: GarmentMeshViewerProps) {
  return (
    <div
      className={className}
      style={{ position: "relative", width: "100%", height }}
    >
      <Canvas
        shadows
        resize={{ scroll: false }}
        camera={{ position: [0, 0.4, 4.2], fov: 38 }}
        gl={{ antialias: true, preserveDrawingBuffer: true }}
        style={{ position: "absolute", inset: 0, background }}
      >
        <color attach="background" args={[background]} />
        <hemisphereLight args={["#ffffff", "#cdbfa6", 1.1]} />
        <ambientLight intensity={0.35} />
        <directionalLight
          position={[3, 5, 4]}
          intensity={1.5}
          castShadow
          shadow-mapSize-width={1024}
          shadow-mapSize-height={1024}
        />
        <directionalLight position={[-4, 2, -3]} intensity={0.5} />
        <directionalLight position={[0, 1, -5]} intensity={0.35} />
        <Bounds fit clip observe margin={1.2}>
          <TeeModel mesh={mesh} />
        </Bounds>
        <OrbitControls
          makeDefault
          enablePan={false}
          autoRotate={autoRotate}
          autoRotateSpeed={1.4}
          minDistance={2.2}
          maxDistance={9}
          target={[0, 0, 0]}
        />
      </Canvas>
    </div>
  );
}
