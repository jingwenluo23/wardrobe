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

function fabricMaterial(color: string, map: THREE.Texture | null) {
  return new THREE.MeshPhysicalMaterial({
    // three.js multiplies the map by the material colour, so textured
    // surfaces must stay white or the photo gets darkened and prints are
    // crushed. Plain surfaces carry the fabric colour directly.
    color: map ? "#ffffff" : color,
    map: map ?? undefined,
    roughness: 0.88,
    metalness: 0,
    sheen: 0.55,
    sheenColor: new THREE.Color("#ffffff"),
    sheenRoughness: 0.9,
    side: THREE.FrontSide,
  });
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
    return [
      fabricMaterial(mesh.color, frontTexture),
      fabricMaterial(mesh.color, backTexture),
      fabricMaterial(mesh.color, fabricTexture),
    ];
  }, [
    mesh.color,
    mesh.extractedTextureUrl,
    mesh.extractedBackTextureUrl,
    mesh.fabricTextureUrl,
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
}: GarmentMeshViewerProps) {
  return (
    <div
      className={className}
      style={{ position: "relative", width: "100%", height: 560 }}
    >
      <Canvas
        shadows
        resize={{ scroll: false }}
        camera={{ position: [0, 0.4, 4.2], fov: 38 }}
        gl={{ antialias: true, preserveDrawingBuffer: true }}
        style={{ position: "absolute", inset: 0, background: "#f1ece0" }}
      >
        <color attach="background" args={["#f1ece0"]} />
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
          minDistance={2.2}
          maxDistance={9}
          target={[0, 0, 0]}
        />
      </Canvas>
    </div>
  );
}
