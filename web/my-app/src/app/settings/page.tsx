import Link from "next/link";

export default function SettingsPage() {
  return (
    <main className="min-h-screen bg-[#f8f7f2] text-[#232421]">
      <section className="mx-auto grid min-h-screen w-full max-w-7xl grid-cols-1 gap-8 px-5 py-6 sm:px-8 lg:grid-cols-[280px_1fr] lg:px-10">
        <aside className="border-b border-[#dad5c8] pb-6 lg:border-b-0 lg:border-r lg:pb-0 lg:pr-8">
          <Link className="flex items-center gap-3" href="/">
            <div className="grid h-11 w-11 place-items-center rounded-lg bg-[#243f3a] text-lg font-semibold text-[#f8f7f2]">
              W
            </div>
            <div>
              <p className="text-sm font-medium uppercase tracking-[0.18em] text-[#7d786d]">
                Wardrobe
              </p>
              <h1 className="text-2xl font-semibold">System settings</h1>
            </div>
          </Link>

          <nav className="mt-10 grid gap-2 text-sm font-medium text-[#5f5a52]">
            <Link
              className="rounded-lg px-3 py-2 hover:bg-white hover:text-[#232421]"
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
            <Link
              className="rounded-lg bg-[#243f3a] px-3 py-2 text-white"
              href="/settings"
            >
              Settings
            </Link>
          </nav>
        </aside>

        <div className="space-y-6">
          <header className="border-b border-[#dad5c8] pb-6">
            <p className="text-sm font-medium uppercase tracking-[0.18em] text-[#857e73]">
              System settings
            </p>
            <h2 className="mt-2 text-4xl font-semibold leading-tight md:text-5xl">
              Configuration
            </h2>
            <p className="mt-3 max-w-2xl text-[#625c54]">
              Manage the shared configuration behind garment reconstruction.
            </p>
          </header>

          <section className="rounded-[28px] border border-[#ded5c8] bg-white p-6 shadow-sm">
            <p className="text-sm font-semibold uppercase tracking-[0.16em] text-[#857e73]">
              Settings
            </p>
            <h3 className="mt-3 text-2xl font-semibold">No standalone standards page</h3>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-[#625c54]">
              The model standards reference page has been removed from this workspace.
            </p>
          </section>
        </div>
      </section>
    </main>
  );
}
