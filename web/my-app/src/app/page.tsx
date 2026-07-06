"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useMemo } from "react";

import type { DraftMesh } from "@/lib/garment-mesh";
import { garmentTemplates, getTemplate } from "@/lib/garment-templates";

const GarmentMeshViewer = dynamic(
  () => import("./outfits/garment-mesh-viewer"),
  { ssr: false },
);

const pipeline = [
  {
    step: "01",
    name: "Photograph",
    detail:
      "Front and back photos of any garment — on a hanger, flat, or worn. Phone camera is enough.",
  },
  {
    step: "02",
    name: "Isolate",
    detail:
      "Per-pixel clothes segmentation lifts the garment out of the scene: no wearer, no background, prints kept intact.",
  },
  {
    step: "03",
    name: "Reconstruct",
    detail:
      "A parametric CAD block rebuilds it in 3D — pattern-true seams, collars, cuffs — with your photos projected onto the panels.",
  },
];

const roadmap = [
  {
    phase: "NOW",
    title: "Clothes",
    items: [
      "43 garment templates across tops and bottoms",
      "ML extraction from photos, prints preserved",
      "Orbitable 3D preview of every piece you own",
    ],
  },
  {
    phase: "NEXT",
    title: "Your avatar & shoes",
    items: [
      "Faceless avatar tuned to your height and body shape",
      "Fit any saved garment on your proportions",
      "Shoe archetype joins the registry",
    ],
  },
  {
    phase: "LATER",
    title: "Everything else you wear",
    items: [
      "Scarves, gloves, hats, glasses, jewellery",
      "Snap a fitting-room photo, try it on digitally before you buy",
      "Community: outfit of the day, ideas, where-to-buy",
      "Brand collabs and fashion-show digital try-ons",
    ],
  },
];

export default function Home() {
  const heroMesh = useMemo<DraftMesh | null>(() => {
    const template = getTemplate("top-hoodie");
    if (!template) {
      return null;
    }
    return {
      assetUrl: "hero://" + template.id,
      generatedAt: 0,
      template: template.id,
      templateLabel: template.label,
      params: template.params,
      features: template.features,
      segmentation: { confidence: 1 },
      color: "#d8d3c6",
      bounds: { width: 0, height: 0, depth: 0 },
    };
  }, []);

  const topsCount = garmentTemplates.filter(
    (t) => t.category === "Tops",
  ).length;
  const bottomsCount = garmentTemplates.filter(
    (t) => t.category === "Bottoms",
  ).length;

  return (
    <main className="min-h-screen text-[#24262c]">
      {/* ---------------------------------------------------------- nav */}
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-6">
        <Link className="flex items-center gap-3" href="/">
          <div className="grid h-9 w-9 place-items-center rounded-md bg-[#9c7a33] font-mono text-sm font-bold text-[#241c08]">
            W
          </div>
          <span className="font-mono text-sm uppercase tracking-[0.22em] text-[#24262c]">
            Wardrobe
          </span>
        </Link>
        <nav className="flex items-center gap-2 text-sm">
          <Link
            className="rounded-md px-3 py-2 text-[#6b6f77] transition hover:text-[#24262c]"
            href="/outfits"
          >
            My wardrobe
          </Link>
          <span className="hidden rounded-md px-3 py-2 text-[#8b9099] sm:inline">
            Community
            <span className="ml-2 rounded-full border border-[#1e202617] px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest">
              soon
            </span>
          </span>
          <Link
            className="rounded-md bg-[#9c7a33] px-4 py-2 font-semibold text-[#241c08] transition hover:bg-[#b8934a]"
            href="/outfits"
          >
            Open wardrobe
          </Link>
        </nav>
      </header>

      {/* ---------------------------------------------------------- hero */}
      <section className="mx-auto grid w-full max-w-6xl items-center gap-10 px-6 pb-20 pt-10 lg:grid-cols-[1.05fr_1fr]">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.28em] text-[#9c7a33]">
            Digital wardrobe · 3D try-on engine
          </p>
          <h1 className="mt-4 text-5xl font-semibold leading-[1.04] tracking-tight sm:text-6xl">
            Every piece you own,
            <br />
            rebuilt in 3D.
          </h1>
          <p className="mt-6 max-w-xl text-lg leading-8 text-[#6b6f77]">
            Photograph your clothes once. Wardrobe isolates the garment,
            reconstructs it as a pattern-true 3D model, and files it into a
            closet you can search, orbit, and plan outfits with — so you
            always know what you own, and whether the next purchase earns its
            hanger.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-4">
            <Link
              className="rounded-md bg-[#9c7a33] px-6 py-3.5 text-sm font-semibold text-[#241c08] transition hover:bg-[#b8934a]"
              href="/outfits"
            >
              Digitize your first garment
            </Link>
            <span className="font-mono text-xs uppercase tracking-[0.18em] text-[#8b9099]">
              Web now · app later
            </span>
          </div>
          <dl className="mt-12 grid grid-cols-3 gap-4 border-t border-[#1e202617] pt-6 font-mono text-sm">
            <div>
              <dt className="text-[10px] uppercase tracking-[0.2em] text-[#8b9099]">
                Templates
              </dt>
              <dd className="mt-1 text-2xl text-[#24262c] tabular-nums">
                {garmentTemplates.length}
              </dd>
            </div>
            <div>
              <dt className="text-[10px] uppercase tracking-[0.2em] text-[#8b9099]">
                Tops / Bottoms
              </dt>
              <dd className="mt-1 text-2xl text-[#24262c] tabular-nums">
                {topsCount} / {bottomsCount}
              </dd>
            </div>
            <div>
              <dt className="text-[10px] uppercase tracking-[0.2em] text-[#8b9099]">
                Extraction
              </dt>
              <dd className="mt-1 text-2xl text-[#24262c]">ML</dd>
            </div>
          </dl>
        </div>

        <div className="relative">
          <div className="overflow-hidden rounded-xl border border-[#1e202617] bg-[#efece4]">
            {heroMesh ? (
              <GarmentMeshViewer
                autoRotate
                background="#efece4"
                height={520}
                mesh={heroMesh}
              />
            ) : null}
          </div>
          <div className="pointer-events-none absolute left-4 top-4 rounded-md border border-[#1e202617] bg-[#efece4]/70 px-3 py-2 font-mono text-[10px] uppercase tracking-[0.2em] text-[#6b6f77]">
            live mesh · hoodie block
          </div>
          <div className="pointer-events-none absolute bottom-4 right-4 rounded-md border border-[#1e202617] bg-[#efece4]/70 px-3 py-2 font-mono text-[10px] uppercase tracking-[0.2em] text-[#6b6f77]">
            drag to orbit
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------ pipeline */}
      <section className="border-t border-[#1e202617] bg-[#1e202608]">
        <div className="mx-auto w-full max-w-6xl px-6 py-16">
          <p className="font-mono text-xs uppercase tracking-[0.28em] text-[#9c7a33]">
            How a garment enters your wardrobe
          </p>
          <div className="mt-8 grid gap-6 md:grid-cols-3">
            {pipeline.map((stage) => (
              <div
                className="rounded-lg border border-[#1e202617] bg-[#ffffff]/70 backdrop-blur-xl p-6"
                key={stage.step}
              >
                <div className="flex items-baseline gap-3">
                  <span className="font-mono text-xs text-[#9c7a33]">
                    {stage.step}
                  </span>
                  <span className="h-px flex-1 bg-[#1e202617]" />
                  <h3 className="text-lg font-semibold">{stage.name}</h3>
                </div>
                <p className="mt-4 text-sm leading-6 text-[#6b6f77]">
                  {stage.detail}
                </p>
              </div>
            ))}
          </div>
          <p className="mt-6 max-w-3xl text-sm leading-6 text-[#8b9099]">
            The same extraction is built to survive store fitting rooms and
            busy backgrounds — point it at something on the rack and preview
            it in your wardrobe before you pay.
          </p>
        </div>
      </section>

      {/* ------------------------------------------------------- roadmap */}
      <section className="border-t border-[#1e202617]">
        <div className="mx-auto w-full max-w-6xl px-6 py-16">
          <p className="font-mono text-xs uppercase tracking-[0.28em] text-[#9c7a33]">
            Where this is going
          </p>
          <h2 className="mt-3 max-w-2xl text-3xl font-semibold tracking-tight">
            Clothes first. Then a body to put them on.
          </h2>
          <div className="mt-8 grid gap-6 md:grid-cols-3">
            {roadmap.map((column) => (
              <div
                className="rounded-lg border border-[#1e202617] bg-[#1e202608] p-6"
                key={column.phase}
              >
                <span
                  className={
                    "inline-block rounded-full px-3 py-1 font-mono text-[10px] uppercase tracking-[0.2em] " +
                    (column.phase === "NOW"
                      ? "bg-[#9c7a33] text-[#241c08]"
                      : "border border-[#1e202617] text-[#6b6f77]")
                  }
                >
                  {column.phase}
                </span>
                <h3 className="mt-4 text-xl font-semibold">{column.title}</h3>
                <ul className="mt-4 grid gap-3 text-sm leading-6 text-[#6b6f77]">
                  {column.items.map((item) => (
                    <li className="flex gap-3" key={item}>
                      <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-[#9c7a33]" />
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* -------------------------------------------------------- avatar */}
      <section className="border-t border-[#1e202617] bg-[#1e202608]">
        <div className="mx-auto grid w-full max-w-6xl gap-10 px-6 py-16 lg:grid-cols-2">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.28em] text-[#9c7a33]">
              The avatar
            </p>
            <h2 className="mt-3 text-3xl font-semibold tracking-tight">
              AAA-game customization.
              <br />
              Your proportions. No face.
            </h2>
            <p className="mt-5 max-w-xl text-sm leading-7 text-[#6b6f77]">
              The avatar is a mannequin, not a portrait: height, shoulders,
              chest, waist — tuned like a character creator, private by
              design. Every garment you digitize fits onto it, so outfit
              planning happens on your silhouette instead of a stock model,
              and a fitting-room photo answers &quot;does this work on
              me?&quot; before the till.
            </p>
          </div>
          <div className="grid content-center gap-3 font-mono text-sm text-[#6b6f77]">
            {[
              ["HEIGHT", "178 cm", "w-2/3"],
              ["SHOULDER", "46 cm", "w-1/2"],
              ["CHEST", "98 cm", "w-3/5"],
              ["WAIST", "82 cm", "w-2/5"],
            ].map(([label, value, bar]) => (
              <div
                className="flex items-center justify-between gap-4 rounded-md border border-[#1e202617] bg-[#ffffff]/70 backdrop-blur-xl px-4 py-3"
                key={label}
              >
                <span className="w-24 text-[10px] uppercase tracking-[0.24em] text-[#8b9099]">
                  {label}
                </span>
                <span className="tabular-nums text-[#24262c]">{value}</span>
                <div className="h-1 w-24 rounded-full bg-[#1e202617]">
                  <div className={"h-1 rounded-full bg-[#9c7a33] " + bar} />
                </div>
              </div>
            ))}
            <p className="mt-1 text-[10px] uppercase tracking-[0.2em] text-[#8b9099]">
              concept · sliders ship with the avatar phase
            </p>
          </div>
        </div>
      </section>

      {/* ----------------------------------------------------------- cta */}
      <section className="border-t border-[#1e202617]">
        <div className="mx-auto flex w-full max-w-6xl flex-col items-start gap-6 px-6 py-16 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">
              Start with the shirt on your chair.
            </h2>
            <p className="mt-2 text-sm text-[#6b6f77]">
              Two photos in, one 3D garment out. Your closet starts counting.
            </p>
          </div>
          <Link
            className="rounded-md bg-[#9c7a33] px-6 py-3.5 text-sm font-semibold text-[#241c08] transition hover:bg-[#b8934a]"
            href="/outfits"
          >
            Open my wardrobe
          </Link>
        </div>
      </section>

      <footer className="border-t border-[#1e202617]">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-2 px-6 py-8 font-mono text-[11px] uppercase tracking-[0.2em] text-[#8b9099] sm:flex-row sm:items-center sm:justify-between">
          <span>Wardrobe — reshaping the fashion industry</span>
          <span>web → app · clothes → shoes → everything</span>
        </div>
      </footer>
    </main>
  );
}
