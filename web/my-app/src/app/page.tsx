import Link from "next/link";

const closetItems = [
  {
    name: "Ivory linen shirt",
    type: "Top",
    palette: "bg-[#f3eee3]",
    accent: "bg-[#2f4f4a]",
    detail: "Breathable",
  },
  {
    name: "Charcoal pleated trouser",
    type: "Bottom",
    palette: "bg-[#34363a]",
    accent: "bg-[#b36f49]",
    detail: "Tailored",
  },
  {
    name: "Sage overshirt",
    type: "Layer",
    palette: "bg-[#8ea890]",
    accent: "bg-[#f0c56a]",
    detail: "Light layer",
  },
  {
    name: "Black leather loafer",
    type: "Shoes",
    palette: "bg-[#1f1f1f]",
    accent: "bg-[#d7dad1]",
    detail: "Polished",
  },
];

const plannedLooks = [
  "Client meeting",
  "Errands",
  "Dinner",
  "Rain day",
];

export default function Home() {
  return (
    <main className="min-h-screen bg-[#f8f7f2] text-[#232421]">
      <section className="mx-auto grid min-h-screen w-full max-w-7xl grid-cols-1 gap-8 px-5 py-6 sm:px-8 lg:grid-cols-[280px_1fr] lg:px-10">
        <aside className="flex flex-col justify-between border-b border-[#dad5c8] pb-6 lg:border-b-0 lg:border-r lg:pb-0 lg:pr-8">
          <div>
            <div className="flex items-center gap-3">
              <div className="grid h-11 w-11 place-items-center rounded-lg bg-[#243f3a] text-lg font-semibold text-[#f8f7f2]">
                W
              </div>
              <div>
                <p className="text-sm font-medium uppercase tracking-[0.18em] text-[#7d786d]">
                  Wardrobe
                </p>
                <h1 className="text-2xl font-semibold">Daily closet</h1>
              </div>
            </div>

            <nav className="mt-10 grid gap-2 text-sm font-medium text-[#5f5a52]">
              <Link
                className="rounded-lg bg-[#243f3a] px-3 py-2 text-white"
                href="/"
              >
                Overview
              </Link>
              <Link
                className="rounded-lg px-3 py-2 hover:bg-white hover:text-[#232421]"
                href="/outfits"
              >
                Outfits
              </Link>
            </nav>
          </div>

          <div className="mt-8 rounded-lg border border-[#d7d1c1] bg-white p-4">
            <p className="text-sm font-semibold text-[#232421]">Closet health</p>
            <div className="mt-4 space-y-3">
              <div>
                <div className="flex justify-between text-xs text-[#6d675f]">
                  <span>Worn this month</span>
                  <span>68%</span>
                </div>
                <div className="mt-2 h-2 rounded-full bg-[#e7e1d2]">
                  <div className="h-2 w-[68%] rounded-full bg-[#b36f49]" />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs text-[#6d675f]">
                  <span>Color balance</span>
                  <span>82%</span>
                </div>
                <div className="mt-2 h-2 rounded-full bg-[#e7e1d2]">
                  <div className="h-2 w-[82%] rounded-full bg-[#243f3a]" />
                </div>
              </div>
            </div>
          </div>
        </aside>

        <div className="space-y-8">
          <header className="grid gap-5 rounded-lg bg-[#243f3a] p-5 text-white md:grid-cols-[1fr_auto] md:p-7">
            <div>
              <p className="text-sm font-medium uppercase tracking-[0.18em] text-[#c9d4c9]">
                Today&apos;s edit
              </p>
              <h2 className="mt-3 max-w-2xl text-4xl font-semibold leading-tight md:text-5xl">
                Build a crisp warm-weather outfit from pieces you already own.
              </h2>
            </div>
            <div className="grid content-between gap-5 md:w-64">
              <div className="rounded-lg bg-white/10 p-4">
                <p className="text-sm text-[#dce4d9]">Weather</p>
                <p className="mt-1 text-2xl font-semibold">73F / light wind</p>
              </div>
              <Link
                className="rounded-lg bg-[#f0c56a] px-4 py-3 text-center text-sm font-semibold text-[#232421] transition hover:bg-[#ffd978]"
                href="/outfits"
              >
                Plan outfit
              </Link>
            </div>
          </header>

          <section className="grid gap-5 lg:grid-cols-[1.2fr_0.8fr]">
            <div className="rounded-lg bg-white p-5 shadow-sm ring-1 ring-[#e5dfd1]">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-sm font-medium uppercase tracking-[0.16em] text-[#857e73]">
                    Suggested look
                  </p>
                  <h3 className="mt-1 text-2xl font-semibold">
                    Linen shirt, trouser, sage layer
                  </h3>
                </div>
                <button className="rounded-lg border border-[#d8d1c3] px-3 py-2 text-sm font-semibold hover:bg-[#f8f7f2]">
                  Save
                </button>
              </div>

              <div className="mt-6 grid gap-4 sm:grid-cols-4">
                {closetItems.map((item) => (
                  <article
                    className="rounded-lg border border-[#e5dfd1] bg-[#fbfaf6] p-3"
                    key={item.name}
                  >
                    <div
                      className={`relative h-32 overflow-hidden rounded-md ${item.palette}`}
                    >
                      <div
                        className={`absolute left-1/2 top-5 h-20 w-14 -translate-x-1/2 rounded-t-full ${item.accent} opacity-90`}
                      />
                      <div className="absolute bottom-4 left-5 right-5 h-5 rounded-full bg-white/45" />
                    </div>
                    <p className="mt-3 text-xs font-medium uppercase tracking-[0.14em] text-[#81786e]">
                      {item.type}
                    </p>
                    <h4 className="mt-1 min-h-10 text-sm font-semibold leading-5">
                      {item.name}
                    </h4>
                    <p className="text-sm text-[#746d64]">{item.detail}</p>
                  </article>
                ))}
              </div>
            </div>

            <div className="rounded-lg bg-white p-5 shadow-sm ring-1 ring-[#e5dfd1]">
              <p className="text-sm font-medium uppercase tracking-[0.16em] text-[#857e73]">
                Week plan
              </p>
              <div className="mt-5 grid gap-3">
                {plannedLooks.map((look, index) => (
                  <div
                    className="flex items-center justify-between rounded-lg border border-[#e5dfd1] p-3"
                    key={look}
                  >
                    <div>
                      <p className="font-semibold">{look}</p>
                      <p className="text-sm text-[#746d64]">
                        {index + 2} pieces ready
                      </p>
                    </div>
                    <div className="flex -space-x-2">
                      <span className="h-8 w-8 rounded-full border-2 border-white bg-[#243f3a]" />
                      <span className="h-8 w-8 rounded-full border-2 border-white bg-[#b36f49]" />
                      <span className="h-8 w-8 rounded-full border-2 border-white bg-[#f0c56a]" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="grid gap-5 md:grid-cols-3">
            {[
              ["Items tracked", "42", "8 added this month"],
              ["Repeat winners", "12", "High-confidence staples"],
              ["Ready outfits", "7", "Built from uploaded pieces"],
            ].map(([label, value, note]) => (
              <div
                className="rounded-lg bg-white p-5 shadow-sm ring-1 ring-[#e5dfd1]"
                key={label}
              >
                <p className="text-sm font-medium uppercase tracking-[0.16em] text-[#857e73]">
                  {label}
                </p>
                <p className="mt-3 text-4xl font-semibold">{value}</p>
                <p className="mt-2 text-sm text-[#746d64]">{note}</p>
              </div>
            ))}
          </section>
        </div>
      </section>
    </main>
  );
}
