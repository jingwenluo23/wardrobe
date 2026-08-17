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

type FabricKind = "jersey" | "woven" | "fleece" | "knit";
type FabricSurface = "panel" | "trim" | "sleeve";

function fabricMaterial(
  color: string,
  map: THREE.Texture | null,
  kind: FabricKind,
  bump?: { texture: THREE.Texture; scale: number } | null,
) {
  // Sheen is ADDED on top of the base colour, so a strong white sheen lays a
  // pale film over the whole surface — brightest where the fabric faces the
  // viewer, which washed the print out behind a white haze across the chest.
  // Real cloth catches a soft, tinted highlight at grazing angles only, so
  // keep the amount low and tint it warm-grey rather than white. Printed
  // surfaces get even less, since any veil there is read as faded ink.
  const profile =
    kind === "fleece"
      ? { roughness: 0.96, sheen: 0.3, sheenRoughness: 0.96, specular: 0.16 }
      : kind === "knit"
        ? { roughness: 0.91, sheen: 0.26, sheenRoughness: 0.9, specular: 0.2 }
        : kind === "woven"
          ? { roughness: 0.92, sheen: 0.12, sheenRoughness: 0.94, specular: 0.17 }
        : { roughness: 0.84, sheen: 0.2, sheenRoughness: 0.82, specular: 0.24 };
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
    sheen: map ? profile.sheen * 0.6 : profile.sheen,
    sheenColor: new THREE.Color("#9b948a"),
    sheenRoughness: profile.sheenRoughness,
    specularIntensity: profile.specular,
    specularColor: new THREE.Color("#e8e2d8"),
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
  } else if (kind === "woven") {
    // Fine warp/weft yarns with occasional deterministic linen slubs. The
    // photographed pinstripes remain the visible pattern; this map only makes
    // the pale surface react like woven fabric rather than knitted loops.
    g.globalAlpha = 0.16;
    g.strokeStyle = "#9a9a9a";
    g.lineWidth = 1;
    for (let x = 0; x < size; x += 4) {
      g.beginPath();
      g.moveTo(x, 0);
      g.lineTo(x, size);
      g.stroke();
    }
    g.globalAlpha = 0.11;
    for (let y = 0; y < size; y += 5) {
      g.beginPath();
      g.moveTo(0, y);
      g.lineTo(size, y);
      g.stroke();
    }
    g.globalAlpha = 0.12;
    g.fillStyle = "#666666";
    for (let y = 19; y < size; y += 47) {
      const offset = (y * 17) % 61;
      g.fillRect(offset, y, size * 0.34, 1);
    }
    g.globalAlpha = 1;
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
    // Sleeves and the hood are large panels cut from the SAME printed cloth as
    // the body, so when no dedicated region was extracted they fall back to the
    // front panel's print — not to the plain swatch. The swatch is a blurred
    // single-colour patch, which is what rendered printed sleeves as flat
    // brown next to a fully patterned body. Only the small trims (neckband,
    // cuffs, plackets) keep the plain swatch, where a solid colour is right.
    // Woven checks/stripes must keep the fabric grain vertical across the
    // shoulder. Their sleeve UVs use a repeating world-space projection, so
    // pair them with the repeating fabric swatch instead of stretching the
    // one-off torso photograph around the sleeve tube.
    const sleeveTexture =
      mesh.features?.fabric === "woven" && fabricTexture
        ? fabricTexture
        : loadTexture(mesh.sleeveTextureUrl) ?? frontTexture ?? trimTexture;
    // Woven panels take the repeating swatch too, not the garment photograph.
    //
    // The photo tile is an unwrap of the whole garment, and on a flat-lay with
    // the sleeves lying against the body there is no way to find the side seams
    // in the mask — measured on a striped linen shirt, the mask run through the
    // centre spans 449-559px of a 561px garment at every height from 45% to 85%,
    // so body and sleeves are one region. Mapping that tile onto the front panel
    // paints a small picture of the entire shirt, sleeves and collar included,
    // across the chest.
    //
    // Cloth cut on the grain does not need registering. It needs the right
    // direction and the right pitch, which a patch of real interior fabric has
    // by construction: pickFabricSwatch already takes several stripe repeats
    // from the chest, clear of every edge, shadow and inpainted region. Tile it
    // at the garment's own scale. The cost is the chest pocket, which the mesh
    // does not model anyway.
    const panelFabric = (() => {
      if (mesh.features?.fabric !== "woven" || !fabricTexture) {
        return null;
      }
      // Clone so the repeat below cannot reach the sleeves, which project the
      // same swatch through world space and already carry their own scale.
      const tiled = fabricTexture.clone();
      // Mirror rather than wrap. The swatch is a photographed crop, so its left
      // and right edges do not match and every plain repeat boundary shows as a
      // hard vertical seam. Mirroring makes each join a reflection, which for a
      // vertical stripe is continuous.
      tiled.wrapS = THREE.MirroredRepeatWrapping;
      tiled.wrapT = THREE.ClampToEdgeWrapping;
      tiled.needsUpdate = true;
      // Across the body: one swatch per this much cloth, matching the
      // world-space period the sleeves use so the grain lines up at the seam.
      // Down the body: no repeat at all. A stripe has no vertical structure to
      // reproduce, and tiling vertically only adds horizontal seams — visible
      // as brick courses across the chest.
      const periodCm = 13.5;
      tiled.repeat.set(Math.max(1, mesh.params.bodyWidth / periodCm), 1);
      return tiled;
    })();
    const hoodTexture =
      loadTexture(mesh.hoodTextureUrl) ?? frontTexture ?? trimTexture;
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
      fabricMaterial(mesh.color, panelFabric ?? frontTexture, fabricKind, frontBump),
      fabricMaterial(mesh.color, panelFabric ?? backTexture, fabricKind, backBump),
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
    mesh.params.bodyWidth,
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
  background,
  autoRotate = false,
  height = 560,
}: GarmentMeshViewerProps) {
  const garmentColor = new THREE.Color(mesh.color);
  const brightFabric =
    garmentColor.r * 0.2126 +
      garmentColor.g * 0.7152 +
      garmentColor.b * 0.0722 >=
    0.7;
  // An ivory garment against the normal cream stage has almost no silhouette,
  // and full studio exposure clips its low-contrast pinstripes. Darken the
  // neutral stage and lights only for bright garments. An explicit background
  // supplied by a caller still wins.
  const stageBackground =
    background ?? (brightFabric ? "#c9c6bd" : "#f1ece0");
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
          // ACES deepens the shadow end, which exaggerates the print's own
          // contrast. Neutral tone mapping keeps the fabric reading as it does
          // in the photograph.
          toneMapping: THREE.NeutralToneMapping,
          toneMappingExposure: brightFabric ? 0.9 : 1.0,
        }}
        onCreated={({ gl }) => {
          gl.shadowMap.type = THREE.PCFSoftShadowMap;
        }}
        style={{ position: "absolute", inset: 0, background: stageBackground }}
      >
        <color attach="background" args={[stageBackground]} />
        <Environment resolution={128}>
          <Lightformer
            form="rect"
            intensity={brightFabric ? 2.25 : 2.8}
            color="#fff9ef"
            position={[0, 4, 5]}
            scale={[5, 5, 1]}
          />
          <Lightformer
            form="rect"
            intensity={brightFabric ? 1.45 : 1.8}
            color="#dce8ff"
            position={[-5, 1, 1]}
            rotation={[0, Math.PI / 2, 0]}
            scale={[3, 5, 1]}
          />
          <Lightformer
            form="rect"
            intensity={brightFabric ? 1.0 : 1.2}
            color="#ffe8d2"
            position={[4, 0, -3]}
            rotation={[0, -Math.PI / 3, 0]}
            scale={[3, 4, 1]}
          />
        </Environment>
        {/* Soft, even studio light. A hard key throws deep shadows across the
            folds, and on a photographic print that reads as harsh contrast
            rather than shape, so the key is eased back and the fill raised —
            closer to the diffuse lighting of a product shot. */}
        <hemisphereLight args={["#ffffff", "#d6cbb6", 0.95]} />
        <ambientLight intensity={0.38} />
        <directionalLight
          position={[3, 5, 4]}
          intensity={1.05}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
          shadow-bias={-0.0002}
        />
        <directionalLight position={[-4, 2, -3]} intensity={0.4} />
        <directionalLight position={[0, 1, -5]} intensity={0.3} />
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
