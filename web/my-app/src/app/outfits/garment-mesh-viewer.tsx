"use client";

import { Canvas } from "@react-three/fiber";
import { Bounds, OrbitControls } from "@react-three/drei";
import { useEffect, useMemo } from "react";
import * as THREE from "three";

import type { DraftMesh } from "@/lib/garment-mesh";
import { buildTeeGeometry } from "./tee-geometry";

type GarmentMeshViewerProps = {
  mesh: DraftMesh;
  className?: string;
};

function TeeModel({ mesh }: { mesh: DraftMesh }) {
  const geometry = useMemo(() => buildTeeGeometry(mesh.params), [mesh.params]);

  const texture = useMemo(() => {
    if (!mesh.extractedTextureUrl) {
      return null;
    }
    const loaded = new THREE.TextureLoader().load(mesh.extractedTextureUrl);
    loaded.colorSpace = THREE.SRGBColorSpace;
    loaded.wrapS = THREE.ClampToEdgeWrapping;
    loaded.wrapT = THREE.ClampToEdgeWrapping;
    return loaded;
  }, [mesh.extractedTextureUrl]);

  useEffect(() => {
    return () => {
      geometry.dispose();
      texture?.dispose();
    };
  }, [geometry, texture]);

  return (
    <group>
      {/* Outside: printed/textured fabric. */}
      <mesh geometry={geometry} castShadow receiveShadow>
        <meshPhysicalMaterial
          color={mesh.color}
          map={texture ?? undefined}
          roughness={0.88}
          metalness={0}
          sheen={0.55}
          sheenColor="#ffffff"
          sheenRoughness={0.9}
          side={THREE.FrontSide}
        />
      </mesh>
      {/* Inside: plain fabric, slightly shaded — prints stay outside only. */}
      <mesh geometry={geometry}>
        <meshPhysicalMaterial
          color={new THREE.Color(mesh.color).multiplyScalar(0.82)}
          roughness={0.95}
          metalness={0}
          side={THREE.BackSide}
        />
      </mesh>
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
