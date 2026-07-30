"use client";

import { Canvas } from "@react-three/fiber";
import { Bounds, Environment, Lightformer, OrbitControls } from "@react-three/drei";
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
  loaded.anisotropy = 8;
  loaded.minFilter = THREE.LinearMipmapLinearFilter;
  loaded.magFilter = THREE.LinearFilter;
  loaded.generateMipmaps = true;
  return loaded;
}

type FabricKind = "jersey" | "fleece" | "knit";
type FabricSurface = "panel" | "trim" | "sleeve";

function fabricMaterial(
  color: string,
  map: THREE.Texture | null,
  kind: FabricKind,
  bump?: { texture: THREE.Texture; scale: number } | null,
) {
  const profile =
    kind === "fleece"
      ? { roughness: 0.96, sheen: 0.82, sheenRoughness: 0.96, specular: 0.22 }
      : kind === "knit"
        ? { roughness: 0.91, sheen: 0.72, sheenRoughness: 0.9, specular: 0.28 }
        : { roughness: 0.84, sheen: 0.58, sheenRoughness: 0.82, specular: 0.34 };
  return new THREE.MeshPhysicalMaterial({
    // three.js multiplies the map by the material colour, so textured
    // surfaces must stay white or the photo gets darkened and prints are
    // crushed. Plain surfaces carry the fabric colour directly.
    color: map ? "#ffffff" : color,
    map: map ?? undefined,
    bumpMap: bump?.texture,
    bumpScale: bump?.scale ?? 0,
    roughness: profile.roughness,
    metalness: 0,
    sheen: profile.sheen,
    sheenColor: new THREE.Color("#ffffff"),
    sheenRoughness: profile.sheenRoughness,
    specularIntensity: profile.specular,
    specularColor: new THREE.Color("#f4eee4"),
    ior: 1.46,
    side: THREE.FrontSide,
  });
}

/**
 * Procedural fabric relief in full UV space. It combines textile
 * microstructure with construction details, so even plain jersey has a fine
 * knit response and panels carry subtle hem/side topstitching.
 */
function makeFabricBump(kind: FabricKind, surface: FabricSurface) {
  const size = 512;
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
    // Broad yarn ribs with alternating stitch valleys.
    for (let x = 0; x < size; x += 12) {
      const grad = g.createLinearGradient(x, 0, x + 12, 0);
      grad.addColorStop(0, "#9a9a9a");
      grad.addColorStop(0.45, "#6e6e6e");
      grad.addColorStop(1, "#9a9a9a");
      g.fillStyle = grad;
      g.fillRect(x, 0, 12, size);
    }
    g.globalAlpha = 0.2;
    for (let y = 0; y < size; y += 9) {
      g.fillStyle = y % 18 === 0 ? "#707070" : "#969696";
      g.fillRect(0, y, size, 3);
    }
    g.globalAlpha = 1;
  } else if (kind === "fleece") {
    // Seeded PRNG so the fleece speckle is deterministic (stable goldens).
    let seed = 0x9e3779b9;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 0xffffffff;
    };
    for (let i = 0; i < 22000; i += 1) {
      const v = 116 + Math.floor(rand() * 25);
      g.fillStyle = "rgb(" + v + "," + v + "," + v + ")";
      const radius = 0.7 + rand() * 1.8;
      g.fillRect(rand() * size, rand() * size, radius, radius);
    }
  } else {
    // Fine single-jersey loops: crossed diagonal yarns with a faint vertical
    // wale. Kept low-contrast so logos and photographed prints remain clean.
    g.globalAlpha = 0.22;
    g.strokeStyle = "#a0a0a0";
    g.lineWidth = 1;
    for (let d = -size; d < size * 2; d += 6) {
      g.beginPath();
      g.moveTo(d, 0);
      g.lineTo(d - size, size);
      g.stroke();
      g.beginPath();
      g.moveTo(d, 0);
      g.lineTo(d + size, size);
      g.stroke();
    }
    g.globalAlpha = 0.12;
    g.fillStyle = "#666666";
    for (let x = 2; x < size; x += 8) {
      g.fillRect(x, 0, 1, size);
    }
    g.globalAlpha = 1;
  }

  if (surface === "panel") {
    const groove = (x1: number, y1: number, x2: number, y2: number) => {
      g.strokeStyle = "#5d5d5d";
      g.lineWidth = 3;
      g.beginPath();
      g.moveTo(x1, y1);
      g.lineTo(x2, y2);
      g.stroke();
    };
    const stitches = (x1: number, y1: number, x2: number, y2: number) => {
      g.strokeStyle = "#6b6b6b";
      g.lineWidth = 2;
      g.setLineDash([8, 7]);
      g.beginPath();
      g.moveTo(x1, y1);
      g.lineTo(x2, y2);
      g.stroke();
      g.setLineDash([]);
    };
    // Side seams and double-needle hem/shoulder stitching. Drawing both UV
    // extremes keeps the detail correct for mirrored back-panel UVs.
    groove(3, 0, 3, size);
    groove(size - 3, 0, size - 3, size);
    stitches(10, 0, 10, size);
    stitches(size - 10, 0, size - 10, size);
    groove(0, size - 4, size, size - 4);
    stitches(0, size - 12, size, size - 12);
    stitches(0, size - 19, size, size - 19);
    stitches(0, 10, size, 10);
  } else if (surface === "sleeve") {
    // Sleeve opening topstitching; both ends cover either UV orientation.
    g.strokeStyle = "#676767";
    g.lineWidth = 2;
    g.setLineDash([8, 7]);
    for (const y of [9, 17, size - 9, size - 17]) {
      g.beginPath();
      g.moveTo(0, y);
      g.lineTo(size, y);
      g.stroke();
    }
    g.setLineDash([]);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.anisotropy = 8;
  return texture;
}

/**
 * Stitched-seam relief for the hood: a centre-back seam groove with stitch
 * dashes, a neckline attachment seam, and edge stitching along the face
 * opening. Drawn in hood UV space (u: 0..1 across the sweep with 0.5 at the
 * centre-back, v: 0 at the neckline .. 1 at the crown/tail), clamped so the
 * lines land exactly on the seams. Fleece speckle keeps the fabric grain
 * consistent with the body.
 */
function makeHoodSeamBump(kind: FabricKind) {
  const size = 512;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const g = canvas.getContext("2d");
  if (!g) {
    return null;
  }
  const fabric = makeFabricBump(kind, "trim");
  const fabricCanvas = fabric?.image as HTMLCanvasElement | undefined;
  if (fabricCanvas) {
    g.drawImage(fabricCanvas, 0, 0, size, size);
    fabric?.dispose();
  } else {
    g.fillStyle = "#808080";
    g.fillRect(0, 0, size, size);
    if (fabric) {
      fabric.dispose();
    }
  }
  const groove = (x: number, w: number) => {
    // Recessed seam line with a soft highlight on each side.
    g.fillStyle = "#5c5c5c";
    g.fillRect(x - w / 2, 0, w, size);
    g.fillStyle = "#949494";
    g.fillRect(x - w / 2 - 2, 0, 2, size);
    g.fillRect(x + w / 2, 0, 2, size);
  };
  const stitches = (x: number) => {
    // Rows of short dashes flanking a seam, like topstitching.
    g.fillStyle = "#6a6a6a";
    for (let y = 4; y < size; y += 14) {
      g.fillRect(x - 1, y, 2, 7);
    }
  };
  // Centre-back seam (u = 0.5) with topstitching either side.
  groove(size / 2, 4);
  stitches(size / 2 - 9);
  stitches(size / 2 + 9);
  // Face-opening edge stitching (u ~ 0 and 1).
  stitches(10);
  stitches(size - 10);
  // Neckline attachment seam (v ~ 0.05): groove + dashes.
  g.fillStyle = "#5c5c5c";
  g.fillRect(0, size * 0.05 - 2, size, 4);
  g.fillStyle = "#6a6a6a";
  for (let x = 4; x < size; x += 14) {
    g.fillRect(x, size * 0.05 + 5, 7, 2);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  return texture;
}

function TeeModel({ mesh }: { mesh: DraftMesh }) {
  const geometry = useMemo(() => {
    const built = buildGarmentGeometry(mesh.params, mesh.features);
    // buildGarmentGeometry already centres each garment on its body (the hood
    // is excluded so it can't drag the orbit pivot off the torso), so no
    // extra recentre here — doing so would reintroduce the hood-skewed pivot.
    return built;
  }, [mesh.params, mesh.features]);

  // Geometry groups: 0 = front panel, 1 = back panel, 2 = seams/trims,
  // 3 = hood, 4 = sleeves. Front/back photos map to their panels; the hood
  // and sleeves get their own extracted regions when available, falling back
  // to the plain fabric swatch; remaining trims stay plain fabric.
  const materials = useMemo(() => {
    const frontTexture = loadTexture(mesh.extractedTextureUrl);
    const backTexture = loadTexture(mesh.extractedBackTextureUrl) ?? frontTexture;
    // Trims (neckbands, collars, plackets, cuffs) use a plain fabric swatch
    // sampled from the photo so their shading matches the real fabric. When
    // no swatch was extracted, fall back to the FRONT body texture rather than
    // the flat dominant colour — otherwise a big trim like a turtleneck collar
    // renders a different colour than the body it is knit from.
    const fabricTexture = loadTexture(mesh.fabricTextureUrl);
    if (fabricTexture) {
      fabricTexture.wrapS = THREE.RepeatWrapping;
      fabricTexture.wrapT = THREE.RepeatWrapping;
    }
    const trimTexture = fabricTexture ?? frontTexture;
    const sleeveTexture = loadTexture(mesh.sleeveTextureUrl) ?? trimTexture;
    const hoodTexture = loadTexture(mesh.hoodTextureUrl) ?? trimTexture;
    const fabricKind: FabricKind = mesh.features?.fabric ?? "jersey";
    const relief =
      fabricKind === "knit" ? 1.25 : fabricKind === "fleece" ? 0.55 : 0.16;
    const surfaceBump = (surface: FabricSurface) => {
      const texture = makeFabricBump(fabricKind, surface);
      return texture ? { texture, scale: relief } : null;
    };
    const frontBump = surfaceBump("panel");
    const backBump = surfaceBump("panel");
    const trimBump = surfaceBump("trim");
    const sleeveBump = surfaceBump("sleeve");
    // Hood (group 3): extracted hood region with stitched-seam relief —
    // centre-back seam, neckline seam, and edge stitching at the opening.
    const seamTexture =
      mesh.features?.neckFinish === "hood"
        ? makeHoodSeamBump(fabricKind)
        : null;
    const hoodBump = seamTexture
      ? { texture: seamTexture, scale: fabricKind === "fleece" ? 0.9 : 0.65 }
      : trimBump;
    return [
      fabricMaterial(mesh.color, frontTexture, fabricKind, frontBump),
      fabricMaterial(mesh.color, backTexture, fabricKind, backBump),
      fabricMaterial(mesh.color, trimTexture, fabricKind, trimBump),
      fabricMaterial(mesh.color, hoodTexture, fabricKind, hoodBump),
      fabricMaterial(mesh.color, sleeveTexture, fabricKind, sleeveBump),
    ];
  }, [
    mesh.color,
    mesh.extractedTextureUrl,
    mesh.extractedBackTextureUrl,
    mesh.fabricTextureUrl,
    mesh.sleeveTextureUrl,
    mesh.hoodTextureUrl,
    mesh.features?.fabric,
    mesh.features?.neckFinish,
  ]);

  const innerMaterial = useMemo(
    () => {
      const kind: FabricKind = mesh.features?.fabric ?? "jersey";
      const bumpTexture = makeFabricBump(kind, "trim");
      return new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(mesh.color).multiplyScalar(0.82),
        bumpMap: bumpTexture,
        bumpScale: kind === "knit" ? 0.8 : kind === "fleece" ? 0.35 : 0.1,
        roughness: kind === "fleece" ? 0.98 : 0.94,
        metalness: 0,
        sheen: 0.45,
        sheenRoughness: 0.95,
        side: THREE.BackSide,
      });
    },
    [mesh.color, mesh.features?.fabric],
  );

  useEffect(() => {
    return () => {
      geometry.dispose();
      const textures = new Set<THREE.Texture>();
      materials.forEach((material) => {
        if (material.map) textures.add(material.map);
        if (material.bumpMap) textures.add(material.bumpMap);
        material.dispose();
      });
      if (innerMaterial.bumpMap) textures.add(innerMaterial.bumpMap);
      textures.forEach((texture) => texture.dispose());
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
        dpr={[1, 2]}
        resize={{ scroll: false }}
        camera={{ position: [0, 0.4, 4.2], fov: 38 }}
        gl={{
          antialias: true,
          preserveDrawingBuffer: true,
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1.08,
        }}
        onCreated={({ gl }) => {
          gl.shadowMap.type = THREE.PCFSoftShadowMap;
        }}
        style={{ position: "absolute", inset: 0, background }}
      >
        <color attach="background" args={[background]} />
        <Environment resolution={128}>
          <Lightformer
            form="rect"
            intensity={2.8}
            color="#fff9ef"
            position={[0, 4, 5]}
            scale={[5, 5, 1]}
          />
          <Lightformer
            form="rect"
            intensity={1.8}
            color="#dce8ff"
            position={[-5, 1, 1]}
            rotation={[0, Math.PI / 2, 0]}
            scale={[3, 5, 1]}
          />
          <Lightformer
            form="rect"
            intensity={1.2}
            color="#ffe8d2"
            position={[4, 0, -3]}
            rotation={[0, -Math.PI / 3, 0]}
            scale={[3, 4, 1]}
          />
        </Environment>
        <hemisphereLight args={["#ffffff", "#cdbfa6", 0.8]} />
        <ambientLight intensity={0.18} />
        <directionalLight
          position={[3, 5, 4]}
          intensity={1.85}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
          shadow-bias={-0.0002}
        />
        <directionalLight position={[-4, 2, -3]} intensity={0.32} />
        <directionalLight position={[0, 1, -5]} intensity={0.25} />
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
