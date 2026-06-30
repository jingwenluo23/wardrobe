"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

type DraftMeshAsset = {
  version: number;
  template: string;
  positions: number[];
  normals: number[];
  uvs: number[];
  indices: number[];
  segments?: Array<{
    id: "left-sleeve" | "front" | "back" | "right-sleeve";
    label: string;
    indexStart: number;
    indexCount: number;
  }>;
  bounds: {
    width: number;
    height: number;
    depth: number;
  };
  textureUrl?: string;
  extractedTextureUrl?: string;
  textureVersion?: string;
};

type DraftMesh = {
  assetUrl: string;
  generatedAt?: string;
  textureUrl?: string;
  extractedTextureUrl?: string;
};

type GarmentMeshViewerProps = {
  className?: string;
  mesh: DraftMesh;
};

type DisplayMode = "garment" | "avatar";
type ViewPreset = "front" | "back" | "left" | "right";

function NeutralMannequin({ visible }: { visible: boolean }) {
  if (!visible) {
    return null;
  }

  return (
    <group position={[0, -0.1, 0]}>
      <mesh castShadow position={[0, 1.55, 0]}>
        <sphereGeometry args={[0.13, 28, 20]} />
        <meshStandardMaterial color="#e4d8cf" roughness={1} />
      </mesh>
      <mesh castShadow position={[0, 1.2, 0]}>
        <cylinderGeometry args={[0.08, 0.1, 0.2, 20]} />
        <meshStandardMaterial color="#e7ddd5" roughness={1} />
      </mesh>
      <mesh castShadow position={[0, 0.75, 0]}>
        <capsuleGeometry args={[0.31, 0.82, 8, 16]} />
        <meshStandardMaterial
          color="#ebe6df"
          roughness={0.98}
          transparent={false}
        />
      </mesh>
      <mesh castShadow position={[-0.45, 0.72, 0]}>
        <capsuleGeometry args={[0.06, 0.76, 6, 12]} />
        <meshStandardMaterial
          color="#eee7e0"
          roughness={1}
          transparent={false}
        />
      </mesh>
      <mesh castShadow position={[0.45, 0.72, 0]}>
        <capsuleGeometry args={[0.06, 0.76, 6, 12]} />
        <meshStandardMaterial
          color="#eee7e0"
          roughness={1}
          transparent={false}
        />
      </mesh>
      <mesh castShadow position={[-0.15, -0.54, 0]}>
        <capsuleGeometry args={[0.08, 0.96, 6, 12]} />
        <meshStandardMaterial
          color="#ede7df"
          roughness={1}
          transparent={false}
        />
      </mesh>
      <mesh castShadow position={[0.15, -0.54, 0]}>
        <capsuleGeometry args={[0.08, 0.96, 6, 12]} />
        <meshStandardMaterial
          color="#ede7df"
          roughness={1}
          transparent={false}
        />
      </mesh>
    </group>
  );
}

function MeshAssetModel({
  asset,
  displayMode,
}: {
  asset: DraftMeshAsset;
  displayMode: DisplayMode;
}) {
  const [texture, setTexture] = useState<THREE.Texture | null>(null);

  const geometry = useMemo(() => {
    const nextGeometry = new THREE.BufferGeometry();
    nextGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(asset.positions, 3),
    );
    nextGeometry.setAttribute(
      "normal",
      new THREE.Float32BufferAttribute(asset.normals, 3),
    );
    nextGeometry.setAttribute("uv", new THREE.Float32BufferAttribute(asset.uvs, 2));
    nextGeometry.setIndex(asset.indices);
    nextGeometry.computeBoundingSphere();
    return nextGeometry;
  }, [asset]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  useEffect(() => {
    const source = asset.extractedTextureUrl || asset.textureUrl;
    if (!source) {
      return;
    }
    const textureSource = asset.textureVersion
      ? source + (source.includes("?") ? "&" : "?") + "v=" + encodeURIComponent(asset.textureVersion)
      : source;

    let disposed = false;
    const loader = new THREE.TextureLoader();
    loader.load(textureSource, (nextTexture) => {
      if (disposed) {
        nextTexture.dispose();
        return;
      }
      nextTexture.wrapS = THREE.ClampToEdgeWrapping;
      nextTexture.wrapT = THREE.ClampToEdgeWrapping;
      nextTexture.colorSpace = THREE.SRGBColorSpace;
      nextTexture.flipY = false;
      nextTexture.anisotropy = 8;
      nextTexture.needsUpdate = true;
      setTexture(nextTexture);
    });

    return () => {
      disposed = true;
    };
  }, [asset.extractedTextureUrl, asset.textureUrl, asset.textureVersion]);

  useEffect(
    () => () => {
      texture?.dispose();
    },
    [texture],
  );

  const material = useMemo(() => {
    if (texture) {
      return new THREE.MeshStandardMaterial({
        map: texture,
        color: "#ffffff",
        emissive: new THREE.Color("#2b2b2b"),
        emissiveIntensity: 0.24,
        transparent: false,
        roughness: 0.94,
        metalness: 0.01,
        side: THREE.FrontSide,
      });
    }

    return new THREE.MeshStandardMaterial({
      color: "#d8d0c5",
      emissive: new THREE.Color("#26231f"),
      emissiveIntensity: 0.12,
      roughness: 0.97,
      metalness: 0.01,
      side: THREE.FrontSide,
    });
  }, [texture]);

  const innerMaterial = useMemo(() => {
    if (texture) {
      return new THREE.MeshStandardMaterial({
        map: texture,
        color: "#f5eeea",
        emissive: new THREE.Color("#201d1a"),
        emissiveIntensity: 0.08,
        transparent: false,
        roughness: 0.98,
        metalness: 0.01,
        side: THREE.BackSide,
      });
    }

    return new THREE.MeshStandardMaterial({
      color: "#d8d0c5",
      emissive: new THREE.Color("#1f1b18"),
      emissiveIntensity: 0.05,
      roughness: 0.99,
      metalness: 0.01,
      side: THREE.BackSide,
    });
  }, [texture]);

  useEffect(
    () => () => {
      material.dispose();
    },
    [material],
  );

  useEffect(() => () => innerMaterial.dispose(), [innerMaterial]);

  return (
    <group position={[0, displayMode === "garment" ? 0.2 : 0.15, 0]}>
      <mesh castShadow geometry={geometry} material={material} />
      <mesh geometry={geometry} material={innerMaterial} renderOrder={-1} />
    </group>
  );
}

function ViewerScene({
  asset,
  displayMode,
  viewPreset,
}: {
  asset: DraftMeshAsset;
  displayMode: DisplayMode;
  viewPreset: ViewPreset;
}) {
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const { camera } = useThree();

  const framing = useMemo(() => {
    const maxDimension = Math.max(asset.bounds.width, asset.bounds.height, asset.bounds.depth);
    const baseDistance = Math.max(3.4, maxDimension * 3.15);
    const garmentCenterY = displayMode === "avatar" ? 0.52 : 0.18;

    return {
      distance: baseDistance,
      minDistance: Math.max(1.8, baseDistance * 0.58),
      maxDistance: Math.max(7.5, baseDistance * 2.4),
      targetY: garmentCenterY,
      cameraY: garmentCenterY + 0.12,
    };
  }, [asset.bounds.depth, asset.bounds.height, asset.bounds.width, displayMode]);

  useEffect(() => {
    const presetPositions: Record<ViewPreset, [number, number, number]> = {
      front: [0, framing.cameraY, framing.distance],
      back: [0, framing.cameraY, -framing.distance],
      left: [-framing.distance, framing.cameraY, 0],
      right: [framing.distance, framing.cameraY, 0],
    };

    const [x, y, z] = presetPositions[viewPreset];
    camera.position.set(x, y, z);

    if (controlsRef.current) {
      controlsRef.current.target.set(0, framing.targetY, 0);
      controlsRef.current.update();
    }
  }, [camera, framing, viewPreset]);

  return (
    <>
      <ambientLight intensity={1.28} />
      <directionalLight intensity={2.05} position={[2.8, 4.4, 3.2]} />
      <directionalLight intensity={1.05} position={[-2.2, 2.4, -1.6]} />
      <hemisphereLight intensity={0.68} color="#ffffff" groundColor="#ece7de" />
      <NeutralMannequin visible={displayMode === "avatar"} />
      <MeshAssetModel asset={asset} displayMode={displayMode} />
      <mesh
        receiveShadow
        position={[0, -1.3, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
      >
        <circleGeometry args={[1.9, 48]} />
        <shadowMaterial opacity={0.14} />
      </mesh>
      <OrbitControls
        autoRotate={false}
        enablePan={false}
        maxDistance={Math.max(framing.maxDistance, 11)}
        minPolarAngle={Math.PI / 2.85}
        minDistance={framing.minDistance}
        maxPolarAngle={Math.PI / 1.72}
        ref={controlsRef}
      />
    </>
  );
}

export default function GarmentMeshViewer({
  mesh,
  className,
}: GarmentMeshViewerProps) {
  const [asset, setAsset] = useState<DraftMeshAsset | null>(null);
  const [error, setError] = useState("");
  const [displayMode, setDisplayMode] = useState<DisplayMode>("garment");
  const [viewPreset, setViewPreset] = useState<ViewPreset>("front");

  useEffect(() => {
    let active = true;

    async function loadAsset() {
      try {
        setError("");
        const revision = mesh.generatedAt
          ? (mesh.assetUrl.includes("?") ? "&" : "?") +
            "revision=" +
            encodeURIComponent(mesh.generatedAt)
          : "";
        const response = await fetch(mesh.assetUrl + revision, { cache: "no-store" });
        if (!response.ok) {
          throw new Error("Unable to load the garment mesh.");
        }
        const payload = (await response.json()) as DraftMeshAsset;
        if (active) {
          setAsset(payload);
        }
      } catch (loadError) {
        if (active) {
          setError(
            loadError instanceof Error
              ? loadError.message
              : "Unable to load the garment mesh.",
          );
        }
      }
    }

    void loadAsset();

    return () => {
      active = false;
    };
  }, [mesh.assetUrl, mesh.generatedAt]);

  useEffect(() => {
    setDisplayMode("garment");
    setViewPreset("front");
  }, [mesh.assetUrl]);

  if (error) {
    return (
      <div
        className={
          className +
          " grid place-items-center rounded-[28px] bg-[#f3f0e9] text-sm font-semibold text-[#625c54]"
        }
      >
        {error}
      </div>
    );
  }

  if (!asset) {
    return (
      <div
        className={
          className +
          " grid place-items-center rounded-[28px] bg-[#f3f0e9] text-sm font-semibold text-[#625c54]"
        }
      >
        Loading mesh...
      </div>
    );
  }

  const usesExtractedTexture = Boolean(asset.extractedTextureUrl);

  return (
    <div className={className + " rounded-[28px] bg-[linear-gradient(180deg,#fbfaf7,#f1ede4)]"}>
      <div className="flex items-center justify-between gap-3 border-b border-[#e5dfd4] px-4 py-3">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[#857e73]">
            Render mode
          </p>
          <p className="mt-1 text-sm font-semibold text-[#232421]">
            {usesExtractedTexture ? "Garment extracted" : "Reference texture"}
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <div className="inline-flex rounded-full bg-white p-1 ring-1 ring-[#e5dfd4]">
            <button
              className={
                "rounded-full px-3 py-1.5 text-xs font-semibold transition " +
                (displayMode === "garment"
                  ? "bg-[#232421] text-white"
                  : "text-[#625c54]")
              }
              onClick={() => setDisplayMode("garment")}
              type="button"
            >
              Garment only
            </button>
            <button
              className={
                "rounded-full px-3 py-1.5 text-xs font-semibold transition " +
                (displayMode === "avatar"
                  ? "bg-[#232421] text-white"
                  : "text-[#625c54]")
              }
              onClick={() => setDisplayMode("avatar")}
              type="button"
            >
              On avatar
            </button>
          </div>
          <div className="inline-flex rounded-full bg-white p-1 ring-1 ring-[#e5dfd4]">
            {(["front", "back", "left", "right"] as ViewPreset[]).map((preset) => (
              <button
                key={preset}
                className={
                  "rounded-full px-3 py-1.5 text-xs font-semibold capitalize transition " +
                  (viewPreset === preset
                    ? "bg-[#9d6b4d] text-white"
                    : "text-[#625c54]")
                }
                onClick={() => setViewPreset(preset)}
                type="button"
              >
                {preset}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="h-[500px] w-full">
        <Canvas
          camera={{ fov: 32, position: [0.8, 0.3, 4.2] }}
          gl={{ antialias: true }}
          shadows
        >
          <color args={["#f6f3ec"]} attach="background" />
          <ViewerScene asset={asset} displayMode={displayMode} viewPreset={viewPreset} />
        </Canvas>
      </div>
    </div>
  );
}
