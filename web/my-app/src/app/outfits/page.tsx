"use client";

import Link from "next/link";
import { type ChangeEvent, type FormEvent, useMemo, useState } from "react";

type WardrobeItem = {
  name: string;
  category: string;
  color: string;
  status: string;
  worn: string;
  background: string;
  shape: string;
  imageUrls?: string[];
  modelStatus?: string;
  modelConfidence?: number;
};

type UploadPhoto = {
  name: string;
  url: string;
};

const categoryNames = [
  "All clothes",
  "Tops",
  "Bottoms",
  "Outerwear",
  "Socks",
  "Shoes",
  "Accessories",
];

const clothes: WardrobeItem[] = [
  {
    name: "Ivory linen shirt",
    category: "Top",
    color: "Ivory",
    status: "Uploaded",
    worn: "2 days ago",
    background: "bg-[#f3eee3]",
    shape: "bg-[#2f4f4a]",
  },
  {
    name: "White ribbed tee",
    category: "Top",
    color: "White",
    status: "Uploaded",
    worn: "Yesterday",
    background: "bg-[#f7f5ee]",
    shape: "bg-[#d8d2c5]",
  },
  {
    name: "Charcoal pleated trouser",
    category: "Bottom",
    color: "Charcoal",
    status: "Uploaded",
    worn: "Last week",
    background: "bg-[#34363a]",
    shape: "bg-[#b36f49]",
  },
  {
    name: "Washed straight denim",
    category: "Bottom",
    color: "Blue",
    status: "Uploaded",
    worn: "4 days ago",
    background: "bg-[#8aa2b5]",
    shape: "bg-[#263d52]",
  },
  {
    name: "Sage overshirt",
    category: "Outerwear",
    color: "Sage",
    status: "Uploaded",
    worn: "Today",
    background: "bg-[#8ea890]",
    shape: "bg-[#f0c56a]",
  },
  {
    name: "Navy chore jacket",
    category: "Outerwear",
    color: "Navy",
    status: "Uploaded",
    worn: "9 days ago",
    background: "bg-[#26364b]",
    shape: "bg-[#d8a460]",
  },
  {
    name: "Cream crew socks",
    category: "Socks",
    color: "Cream",
    status: "Uploaded",
    worn: "Unused",
    background: "bg-[#efe5d0]",
    shape: "bg-[#b7a27c]",
  },
  {
    name: "Black leather loafer",
    category: "Shoes",
    color: "Black",
    status: "Uploaded",
    worn: "3 days ago",
    background: "bg-[#1f1f1f]",
    shape: "bg-[#d7dad1]",
  },
  {
    name: "Canvas sneaker",
    category: "Shoes",
    color: "Oat",
    status: "Uploaded",
    worn: "Last week",
    background: "bg-[#ded4c0]",
    shape: "bg-[#4e625f]",
  },
];

const reconstructionStages = [
  "Align photos",
  "Find garment edges",
  "Build mesh draft",
  "Project fabric texture",
];

function categoryMatches(categoryName: string, itemCategory: string) {
  if (categoryName === "All clothes") {
    return true;
  }

  const normalizedCategory = categoryName.toLowerCase();
  const normalizedItem = itemCategory.toLowerCase();

  return (
    normalizedCategory === normalizedItem ||
    normalizedCategory === `${normalizedItem}s`
  );
}

function estimateDraftModel(photoCount: number) {
  return {
    confidence: Math.min(92, 48 + photoCount * 11),
    status: photoCount >= 4 ? "3D draft ready" : "3D draft",
  };
}

export default function OutfitsPage() {
  const [selectedCategory, setSelectedCategory] = useState("All clothes");
  const [wardrobeItems, setWardrobeItems] = useState(clothes);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [itemName, setItemName] = useState("");
  const [itemCategory, setItemCategory] = useState("Tops");
  const [uploadPhotos, setUploadPhotos] = useState<UploadPhoto[]>([]);

  const categories = useMemo(
    () =>
      categoryNames.map((name) => ({
        name,
        count:
          name === "All clothes"
            ? wardrobeItems.length
            : wardrobeItems.filter((item) => categoryMatches(name, item.category))
                .length,
      })),
    [wardrobeItems],
  );

  const visibleClothes = useMemo(
    () =>
      wardrobeItems.filter((item) =>
        categoryMatches(selectedCategory, item.category),
      ),
    [selectedCategory, wardrobeItems],
  );

  function handlePhotoUpload(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []).slice(0, 6);
    const photoDrafts = files.map((file) => ({
      name: file.name,
      url: URL.createObjectURL(file),
    }));

    setUploadPhotos((currentPhotos) =>
      [...currentPhotos, ...photoDrafts].slice(0, 6),
    );
    event.target.value = "";
  }

  function resetUploadForm() {
    setItemName("");
    setItemCategory("Tops");
    setUploadPhotos([]);
  }

  function closeUploadPrompt() {
    setIsUploadOpen(false);
    resetUploadForm();
  }

  function handleGenerateModel(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!itemName.trim() || uploadPhotos.length < 2) {
      return;
    }

    const draftModel = estimateDraftModel(uploadPhotos.length);
    const normalizedCategory =
      itemCategory === "Tops"
        ? "Top"
        : itemCategory === "Bottoms"
          ? "Bottom"
          : itemCategory;

    setWardrobeItems((currentItems) => [
      {
        name: itemName.trim(),
        category: normalizedCategory,
        color: "From photos",
        status: draftModel.status,
        worn: "Just uploaded",
        background: "bg-[#e8e2d3]",
        shape: "bg-[#243f3a]",
        imageUrls: uploadPhotos.map((photo) => photo.url),
        modelStatus: draftModel.status,
        modelConfidence: draftModel.confidence,
      },
      ...currentItems,
    ]);
    setSelectedCategory("All clothes");
    setIsUploadOpen(false);
    resetUploadForm();
  }

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
              <h1 className="text-2xl font-semibold">Outfits</h1>
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
              className="rounded-lg bg-[#243f3a] px-3 py-2 text-white"
              href="/outfits"
            >
              Outfits
            </Link>
          </nav>

          <div className="mt-10">
            <p className="text-sm font-medium uppercase tracking-[0.16em] text-[#857e73]">
              Categories
            </p>
            <div className="mt-4 grid gap-2">
              {categories.map((category) => (
                <button
                  type="button"
                  onClick={() => setSelectedCategory(category.name)}
                  className={`flex items-center justify-between rounded-lg px-3 py-2 text-left text-sm font-semibold ${
                    selectedCategory === category.name
                      ? "bg-white text-[#232421] shadow-sm ring-1 ring-[#e5dfd1]"
                      : "text-[#625c54] hover:bg-white"
                  }`}
                  key={category.name}
                >
                  <span>{category.name}</span>
                  <span className="text-xs text-[#857e73]">
                    {category.count}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </aside>

        <div className="space-y-6">
          <header className="flex flex-col justify-between gap-5 border-b border-[#dad5c8] pb-6 md:flex-row md:items-end">
            <div>
              <p className="text-sm font-medium uppercase tracking-[0.18em] text-[#857e73]">
                Outfit builder
              </p>
              <h2 className="mt-2 text-4xl font-semibold leading-tight md:text-5xl">
                Uploaded clothes
              </h2>
              <p className="mt-3 max-w-2xl text-[#625c54]">
                Browse uploaded wardrobe pieces by category before combining
                them into an outfit.
              </p>
            </div>
            <button
              className="rounded-lg bg-[#243f3a] px-4 py-3 text-sm font-semibold text-white transition hover:bg-[#1c332f]"
              onClick={() => setIsUploadOpen(true)}
              type="button"
            >
              Add clothes
            </button>
          </header>

          <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {visibleClothes.map((item) => (
              <article
                className="rounded-lg bg-white p-4 shadow-sm ring-1 ring-[#e5dfd1]"
                key={item.name}
              >
                <div
                  className={`relative h-52 overflow-hidden rounded-md ${item.background}`}
                  style={
                    item.imageUrls?.[0]
                      ? {
                          backgroundImage: `linear-gradient(to bottom, rgba(35, 36, 33, 0.08), rgba(35, 36, 33, 0.2)), url(${item.imageUrls[0]})`,
                          backgroundPosition: "center",
                          backgroundSize: "cover",
                        }
                      : undefined
                  }
                >
                  {item.imageUrls?.[0] ? (
                    <div className="absolute bottom-4 left-4 rounded-full bg-white/90 px-3 py-1 text-xs font-semibold text-[#243f3a]">
                      {item.imageUrls.length} photos
                    </div>
                  ) : (
                    <>
                      <div
                        className={`absolute left-1/2 top-8 h-28 w-20 -translate-x-1/2 rounded-t-full ${item.shape} opacity-90`}
                      />
                      <div className="absolute bottom-6 left-8 right-8 h-7 rounded-full bg-white/45" />
                    </>
                  )}
                  {item.modelStatus ? (
                    <div className="absolute right-4 top-4 h-24 w-20 rounded-lg bg-white/85 p-2 shadow-sm">
                      <div className="mx-auto h-14 w-10 rounded-t-full bg-[#243f3a] shadow-[8px_8px_0_#b36f49] [transform:perspective(120px)_rotateY(-24deg)]" />
                      <p className="mt-2 text-center text-[10px] font-semibold uppercase tracking-[0.12em] text-[#625c54]">
                        3D draft
                      </p>
                    </div>
                  ) : null}
                </div>
                <div className="mt-4 flex items-start justify-between gap-4">
                  <div>
                    <p className="text-xs font-medium uppercase tracking-[0.14em] text-[#81786e]">
                      {item.category}
                    </p>
                    <h3 className="mt-1 text-lg font-semibold">{item.name}</h3>
                  </div>
                  <span className="rounded-full bg-[#eef2ea] px-3 py-1 text-xs font-semibold text-[#2f4f4a]">
                    {item.status}
                  </span>
                </div>
                <div className="mt-4 grid grid-cols-2 gap-3 text-sm text-[#625c54]">
                  <div className="rounded-lg bg-[#f8f7f2] p-3">
                    <p className="text-xs uppercase tracking-[0.14em] text-[#857e73]">
                      Color
                    </p>
                    <p className="mt-1 font-semibold text-[#232421]">
                      {item.color}
                    </p>
                  </div>
                  <div className="rounded-lg bg-[#f8f7f2] p-3">
                    <p className="text-xs uppercase tracking-[0.14em] text-[#857e73]">
                      Last worn
                    </p>
                    <p className="mt-1 font-semibold text-[#232421]">
                      {item.worn}
                    </p>
                  </div>
                </div>
                {item.modelConfidence ? (
                  <div className="mt-3 rounded-lg border border-[#d8d1c3] bg-[#fbfaf6] p-3">
                    <div className="flex justify-between text-xs font-semibold text-[#625c54]">
                      <span>3D model confidence</span>
                      <span>{item.modelConfidence}%</span>
                    </div>
                    <div className="mt-2 h-2 rounded-full bg-[#e7e1d2]">
                      <div
                        className="h-2 rounded-full bg-[#243f3a]"
                        style={{ width: `${item.modelConfidence}%` }}
                      />
                    </div>
                  </div>
                ) : null}
              </article>
            ))}
          </section>
        </div>
      </section>

      {isUploadOpen ? (
        <div className="fixed inset-0 z-50 grid place-items-center bg-[#232421]/55 p-4">
          <form
            className="max-h-[92vh] w-full max-w-3xl overflow-y-auto rounded-lg bg-[#f8f7f2] p-5 shadow-2xl"
            onSubmit={handleGenerateModel}
          >
            <div className="flex items-start justify-between gap-4 border-b border-[#dad5c8] pb-4">
              <div>
                <p className="text-sm font-medium uppercase tracking-[0.18em] text-[#857e73]">
                  New garment
                </p>
                <h2 className="mt-1 text-3xl font-semibold">
                  Add clothing photos
                </h2>
              </div>
              <button
                className="rounded-lg border border-[#d8d1c3] px-3 py-2 text-sm font-semibold hover:bg-white"
                onClick={closeUploadPrompt}
                type="button"
              >
                Close
              </button>
            </div>

            <div className="mt-5 grid gap-5 lg:grid-cols-[1fr_280px]">
              <div className="space-y-4">
                <div className="grid gap-4 sm:grid-cols-2">
                  <label className="grid gap-2 text-sm font-semibold text-[#625c54]">
                    Item name
                    <input
                      className="rounded-lg border border-[#d8d1c3] bg-white px-3 py-3 text-base font-medium text-[#232421] outline-none focus:border-[#243f3a]"
                      onChange={(event) => setItemName(event.target.value)}
                      placeholder="Black knit polo"
                      type="text"
                      value={itemName}
                    />
                  </label>
                  <label className="grid gap-2 text-sm font-semibold text-[#625c54]">
                    Category
                    <select
                      className="rounded-lg border border-[#d8d1c3] bg-white px-3 py-3 text-base font-medium text-[#232421] outline-none focus:border-[#243f3a]"
                      onChange={(event) => setItemCategory(event.target.value)}
                      value={itemCategory}
                    >
                      {categoryNames
                        .filter((category) => category !== "All clothes")
                        .map((category) => (
                          <option key={category}>{category}</option>
                        ))}
                    </select>
                  </label>
                </div>

                <label className="grid min-h-44 cursor-pointer place-items-center rounded-lg border border-dashed border-[#bdb5a5] bg-white p-5 text-center transition hover:border-[#243f3a]">
                  <span className="text-base font-semibold text-[#232421]">
                    Upload front, back, and side photos
                  </span>
                  <span className="mt-1 text-sm text-[#746d64]">
                    Select at least two images to start a 3D draft.
                  </span>
                  <input
                    accept="image/*"
                    className="sr-only"
                    multiple
                    onChange={handlePhotoUpload}
                    type="file"
                  />
                </label>

                {uploadPhotos.length > 0 ? (
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                    {uploadPhotos.map((photo, index) => (
                      <div
                        className="relative h-32 overflow-hidden rounded-lg bg-[#e8e2d3]"
                        key={`${photo.name}-${index}`}
                        style={{
                          backgroundImage: `url(${photo.url})`,
                          backgroundPosition: "center",
                          backgroundSize: "cover",
                        }}
                      >
                        <span className="absolute bottom-2 left-2 rounded-full bg-white/90 px-2 py-1 text-[11px] font-semibold text-[#625c54]">
                          View {index + 1}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : null}
              </div>

              <aside className="rounded-lg border border-[#d8d1c3] bg-white p-4">
                <p className="text-sm font-medium uppercase tracking-[0.16em] text-[#857e73]">
                  3D pipeline
                </p>
                <div className="mt-4 grid gap-3">
                  {reconstructionStages.map((stage, index) => (
                    <div className="flex items-center gap-3" key={stage}>
                      <span
                        className={`grid h-8 w-8 place-items-center rounded-full text-xs font-bold ${
                          uploadPhotos.length >= 2
                            ? "bg-[#243f3a] text-white"
                            : "bg-[#ede7d8] text-[#746d64]"
                        }`}
                      >
                        {index + 1}
                      </span>
                      <span className="text-sm font-semibold text-[#232421]">
                        {stage}
                      </span>
                    </div>
                  ))}
                </div>

                <div className="mt-6 rounded-lg bg-[#f8f7f2] p-4">
                  <div className="mx-auto h-28 w-24 rounded-lg bg-white p-3 shadow-sm">
                    <div className="mx-auto h-20 w-14 rounded-t-full bg-[#243f3a] shadow-[10px_10px_0_#b36f49] [transform:perspective(140px)_rotateY(-24deg)]" />
                  </div>
                  <p className="mt-3 text-center text-sm font-semibold text-[#625c54]">
                    Preview model
                  </p>
                </div>
              </aside>
            </div>

            <div className="mt-5 flex flex-col-reverse justify-end gap-3 border-t border-[#dad5c8] pt-4 sm:flex-row">
              <button
                className="rounded-lg border border-[#d8d1c3] px-4 py-3 text-sm font-semibold hover:bg-white"
                onClick={closeUploadPrompt}
                type="button"
              >
                Cancel
              </button>
              <button
                className="rounded-lg bg-[#243f3a] px-4 py-3 text-sm font-semibold text-white transition hover:bg-[#1c332f] disabled:cursor-not-allowed disabled:bg-[#a6a096]"
                disabled={!itemName.trim() || uploadPhotos.length < 2}
                type="submit"
              >
                Generate 3D draft
              </button>
            </div>
          </form>
        </div>
      ) : null}
    </main>
  );
}
