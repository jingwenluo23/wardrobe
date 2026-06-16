"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import {
  type ChangeEvent,
  type FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { DraftMesh } from "@/lib/garment-mesh";

type DraftPipelineStatus = "processing" | "ready" | "failed";
type DraftStageStatus = "pending" | "active" | "done";

type DraftStage = {
  id: string;
  label: string;
  status: DraftStageStatus;
};

type ApiDraft = {
  id: string;
  name: string;
  category: string;
  color: string;
  status: DraftPipelineStatus;
  progress: number;
  currentStage: string;
  createdAt: string;
  photos: Array<{ url: string; role?: "front" | "back" | "side" }>;
  stages: DraftStage[];
  mesh?: DraftMesh;
  error?: string;
};

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
  draftId?: string;
  pipelineStatus?: DraftPipelineStatus;
  progress?: number;
  stages?: DraftStage[];
  mesh?: DraftMesh;
};

type UploadPhoto = {
  file: File;
  name: string;
  url: string;
  role: UploadRole;
};

type UploadRole = "front" | "back" | "side";

const categoryNames = [
  "All clothes",
  "Tops",
  "Bottoms",
  "Outerwear",
  "Socks",
  "Shoes",
  "Accessories",
];

const clothes: WardrobeItem[] = [];

const reconstructionStages = [
  "Extract garment region",
  "Build standard base mesh",
  "Fit neutral garment proportions",
  "Project garment texture",
];

const templateLabels: Record<string, string> = {
  "top-standard-tee": "Standard T-shirt",
  "top-fitted": "Fitted top",
  "outerwear-boxy": "Boxy outerwear",
  "bottom-straight": "Straight bottom",
  "shoe-low": "Low shoe",
};

function garmentPhotoLabel(role: UploadRole) {
  if (role === "front") {
    return "Front";
  }
  if (role === "back") {
    return "Back";
  }
  return "Side";
}

const uploadRoles: UploadRole[] = ["front", "back", "side"];

const GarmentMeshViewer = dynamic(() => import("./garment-mesh-viewer"), {
  ssr: false,
});

function categoryMatches(categoryName: string, itemCategory: string) {
  if (categoryName === "All clothes") {
    return true;
  }

  const normalizedCategory = categoryName.toLowerCase();
  const normalizedItem = itemCategory.toLowerCase();

  return (
    normalizedCategory === normalizedItem ||
    normalizedCategory === normalizedItem + "s"
  );
}

function draftToWardrobeItem(draft: ApiDraft): WardrobeItem {
  return {
    name: draft.name,
    category: draft.category,
    color: draft.color,
    status:
      draft.status === "ready"
        ? "Mesh ready"
        : draft.status === "failed"
          ? "Failed"
          : "Processing",
    worn: "Just uploaded",
    background: "bg-[#e8e2d3]",
    shape: "bg-[#243f3a]",
    imageUrls: draft.photos.map((photo) => photo.url),
    modelStatus: draft.currentStage,
    draftId: draft.id,
    pipelineStatus: draft.status,
    progress: draft.progress,
    stages: draft.stages,
    mesh: draft.mesh,
  };
}

export default function OutfitsPage() {
  const [selectedCategory, setSelectedCategory] = useState("All clothes");
  const [draftItems, setDraftItems] = useState<WardrobeItem[]>([]);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [itemName, setItemName] = useState("");
  const [itemCategory, setItemCategory] = useState("Tops");
  const [uploadPhotos, setUploadPhotos] = useState<
    Partial<Record<UploadRole, UploadPhoto>>
  >({});
  const [selectedDraftItem, setSelectedDraftItem] =
    useState<WardrobeItem | null>(null);
  const [selectedReferenceIndex, setSelectedReferenceIndex] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [deletingDraftId, setDeletingDraftId] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState("");

  const loadDrafts = useCallback(async () => {
    const response = await fetch("/api/drafts", { cache: "no-store" });
    if (!response.ok) {
      return;
    }
    const payload = (await response.json()) as { drafts: ApiDraft[] };
    const items = payload.drafts.map(draftToWardrobeItem);
    setDraftItems(items);
    setSelectedDraftItem((currentItem) => {
      if (!currentItem?.draftId) {
        return currentItem;
      }
      return items.find((item) => item.draftId === currentItem.draftId) ?? currentItem;
    });
  }, []);

  useEffect(() => {
    void loadDrafts();
  }, [loadDrafts]);

  useEffect(() => {
    setSelectedReferenceIndex(0);
  }, [selectedDraftItem?.draftId]);

  useEffect(() => {
    const hasProcessingDraft = draftItems.some(
      (item) => item.pipelineStatus === "processing",
    );
    if (!hasProcessingDraft) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void loadDrafts();
    }, 900);

    return () => window.clearInterval(intervalId);
  }, [draftItems, loadDrafts]);

  const wardrobeItems = useMemo(
    () => [...draftItems, ...clothes],
    [draftItems],
  );

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

  const orderedUploadPhotos = useMemo(
    () =>
      uploadRoles.flatMap((role) => {
        const photo = uploadPhotos[role];
        return photo ? [photo] : [];
      }),
    [uploadPhotos],
  );

  const hasRequiredUploadPhotos = Boolean(uploadPhotos.front && uploadPhotos.back);

  function handlePhotoUpload(
    role: UploadRole,
    event: ChangeEvent<HTMLInputElement>,
  ) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    const nextPhoto = {
      file,
      name: file.name,
      url: URL.createObjectURL(file),
      role,
    };

    setUploadPhotos((currentPhotos) => {
      const currentPhoto = currentPhotos[role];
      if (currentPhoto) {
        URL.revokeObjectURL(currentPhoto.url);
      }
      return {
        ...currentPhotos,
        [role]: nextPhoto,
      };
    });
    event.target.value = "";
  }

  function removeUploadPhoto(role: UploadRole) {
    setUploadPhotos((currentPhotos) => {
      const currentPhoto = currentPhotos[role];
      if (!currentPhoto) {
        return currentPhotos;
      }

      URL.revokeObjectURL(currentPhoto.url);
      return Object.fromEntries(
        Object.entries(currentPhotos).filter(([currentRole]) => currentRole !== role),
      ) as Partial<Record<UploadRole, UploadPhoto>>;
    });
  }

  function clearUploadPhotos() {
    setUploadPhotos((currentPhotos) => {
      Object.values(currentPhotos).forEach((photo) => {
        if (photo) {
          URL.revokeObjectURL(photo.url);
        }
      });
      return {};
    });
  }

  function resetUploadForm() {
    setItemName("");
    setItemCategory("Tops");
    setUploadError("");
    clearUploadPhotos();
  }

  function closeUploadPrompt() {
    if (isSubmitting) {
      return;
    }
    setIsUploadOpen(false);
    resetUploadForm();
  }

  async function handleGenerateModel(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!itemName.trim() || !hasRequiredUploadPhotos || isSubmitting) {
      return;
    }

    setIsSubmitting(true);
    setUploadError("");

    const formData = new FormData();
    formData.append("name", itemName.trim());
    formData.append("category", itemCategory);
    if (uploadPhotos.front) {
      formData.append("frontPhoto", uploadPhotos.front.file);
    }
    if (uploadPhotos.back) {
      formData.append("backPhoto", uploadPhotos.back.file);
    }
    if (uploadPhotos.side) {
      formData.append("sidePhoto", uploadPhotos.side.file);
    }

    try {
      const response = await fetch("/api/drafts", {
        method: "POST",
        body: formData,
      });
      const payload = (await response.json()) as { draft?: ApiDraft; error?: string };

      if (!response.ok || !payload.draft) {
        throw new Error(payload.error ?? "Unable to create the preview.");
      }

      const item = draftToWardrobeItem(payload.draft);
      setDraftItems((currentItems) => [
        item,
        ...currentItems.filter((currentItem) => currentItem.draftId !== item.draftId),
      ]);
      setSelectedCategory("All clothes");
      setIsUploadOpen(false);
      resetUploadForm();
    } catch (error) {
      setUploadError(
        error instanceof Error ? error.message : "Unable to create the preview.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleDeleteDraft(item: WardrobeItem) {
    if (!item.draftId || deletingDraftId) {
      return;
    }

    const confirmed = window.confirm(
      'Delete "' + item.name + '" from the uploaded clothes library?',
    );
    if (!confirmed) {
      return;
    }

    setDeletingDraftId(item.draftId);

    try {
      const response = await fetch("/api/drafts/" + item.draftId, {
        method: "DELETE",
      });
      const payload = (await response.json()) as { error?: string };

      if (!response.ok) {
        throw new Error(payload.error ?? "Unable to delete the uploaded item.");
      }

      setDraftItems((currentItems) =>
        currentItems.filter((currentItem) => currentItem.draftId !== item.draftId),
      );
      setSelectedDraftItem((currentItem) =>
        currentItem?.draftId === item.draftId ? null : currentItem,
      );
    } catch (error) {
      window.alert(
        error instanceof Error
          ? error.message
          : "Unable to delete the uploaded item.",
      );
    } finally {
      setDeletingDraftId(null);
    }
  }

  const selectedDraftViewCount = selectedDraftItem?.imageUrls?.length ?? 0;

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
            <Link
              className="rounded-lg px-3 py-2 hover:bg-white hover:text-[#232421]"
              href="/settings"
            >
              Settings
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
                  className={
                    "flex items-center justify-between rounded-lg px-3 py-2 text-left text-sm font-semibold " +
                    (selectedCategory === category.name
                      ? "bg-white text-[#232421] shadow-sm ring-1 ring-[#e5dfd1]"
                      : "text-[#625c54] hover:bg-white")
                  }
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
                Upload a few garment photos and the app fits them onto a known
                garment template to generate a quick 3D mesh preview.
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

          {visibleClothes.length === 0 ? (
            <section className="rounded-lg border border-dashed border-[#d8d1c3] bg-white/70 p-10 text-center shadow-sm">
              <p className="text-sm font-medium uppercase tracking-[0.16em] text-[#857e73]">
                Uploaded clothes
              </p>
              <h3 className="mt-2 text-2xl font-semibold text-[#232421]">
                Your library is empty
              </h3>
              <p className="mx-auto mt-3 max-w-xl text-sm leading-6 text-[#625c54]">
                The wardrobe starts at 0 items and only keeps the clothes you upload.
              </p>
            </section>
          ) : (
            <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {visibleClothes.map((item) => (
              <article
                className={
                  "rounded-lg bg-white p-4 shadow-sm ring-1 ring-[#e5dfd1]" +
                  (item.modelStatus ? " cursor-pointer transition hover:-translate-y-0.5" : "")
                }
                key={item.draftId ?? item.name}
                onClick={item.modelStatus ? () => setSelectedDraftItem(item) : undefined}
              >
                <div
                  className={"relative h-52 overflow-hidden rounded-md " + item.background}
                  style={
                    item.imageUrls?.[0]
                      ? {
                          backgroundImage:
                            "linear-gradient(to bottom, rgba(35, 36, 33, 0.08), rgba(35, 36, 33, 0.2)), url(" +
                            item.imageUrls[0] +
                            ")",
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
                        className={
                          "absolute left-1/2 top-8 h-28 w-20 -translate-x-1/2 rounded-t-full opacity-90 " +
                          item.shape
                        }
                      />
                      <div className="absolute bottom-6 left-8 right-8 h-7 rounded-full bg-white/45" />
                    </>
                  )}
                  {item.draftId ? (
                    <button
                      className="absolute right-3 top-3 rounded-full bg-white/92 px-3 py-1 text-[11px] font-semibold text-[#8f321f] shadow-sm ring-1 ring-[#e5dfd1] transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-60"
                      disabled={deletingDraftId === item.draftId}
                      onClick={(event) => {
                        event.stopPropagation();
                        void handleDeleteDraft(item);
                      }}
                      type="button"
                    >
                      {deletingDraftId === item.draftId ? "Deleting..." : "Delete"}
                    </button>
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
                {item.modelStatus ? (
                  <div className="mt-3 rounded-lg border border-[#d8d1c3] bg-[#fbfaf6] p-3">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#857e73]">
                          Mesh preview
                        </p>
                        <p className="mt-1 text-sm font-semibold text-[#232421]">
                          {item.modelStatus}
                        </p>
                      </div>
                      <button
                        className="rounded-lg bg-[#243f3a] px-3 py-2 text-xs font-semibold text-white transition hover:bg-[#1c332f]"
                        onClick={() => setSelectedDraftItem(item)}
                        type="button"
                      >
                        View
                      </button>
                    </div>
                    <div className="mt-3 h-2 rounded-full bg-[#e7e1d2]">
                      <div
                        className="h-2 rounded-full bg-[#243f3a] transition-all"
                        style={{ width: String(item.progress ?? 0) + "%" }}
                      />
                    </div>
                  </div>
                ) : null}
              </article>
            ))}
            </section>
          )}
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
                className="rounded-lg border border-[#d8d1c3] px-3 py-2 text-sm font-semibold hover:bg-white disabled:cursor-not-allowed disabled:opacity-60"
                disabled={isSubmitting}
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

                <div className="rounded-lg border border-[#d8d1c3] bg-white p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-base font-semibold text-[#232421]">
                        Upload garment photos
                      </p>
                      <p className="mt-1 text-sm text-[#746d64]">
                        Front and back are required. Side is optional and helps refine depth.
                      </p>
                    </div>
                    <div className="rounded-full bg-[#f8f7f2] px-3 py-1 text-xs font-semibold text-[#625c54]">
                      Required: Front + Back
                    </div>
                  </div>

                  <div className="mt-4 grid gap-3 sm:grid-cols-3">
                    {uploadRoles.map((role) => {
                      const photo = uploadPhotos[role];
                      const isRequired = role !== "side";

                      return (
                        <div
                          className="rounded-lg border border-dashed border-[#bdb5a5] bg-[#fbfaf6] p-3"
                          key={role}
                        >
                          <div className="flex items-center justify-between gap-3">
                            <p className="text-sm font-semibold text-[#232421]">
                              {garmentPhotoLabel(role)}
                            </p>
                            <span className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[#857e73]">
                              {isRequired ? "Required" : "Optional"}
                            </span>
                          </div>

                          <label className="mt-3 grid min-h-40 cursor-pointer place-items-center overflow-hidden rounded-lg border border-dashed border-[#cfc6b4] bg-white text-center transition hover:border-[#243f3a]">
                            {photo ? (
                              <div
                                className="relative h-full min-h-40 w-full bg-[#e8e2d3]"
                                style={{
                                  backgroundImage: "url(" + photo.url + ")",
                                  backgroundPosition: "center",
                                  backgroundSize: "cover",
                                }}
                              >
                                <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-[#232421]/70 to-transparent p-3 text-left text-xs font-semibold text-white">
                                  Replace {garmentPhotoLabel(role).toLowerCase()} photo
                                </div>
                              </div>
                            ) : (
                              <div className="px-4">
                                <p className="text-sm font-semibold text-[#232421]">
                                  Add {garmentPhotoLabel(role).toLowerCase()} photo
                                </p>
                                <p className="mt-1 text-xs leading-5 text-[#857e73]">
                                  {role === "front"
                                    ? "Flat front view of the garment."
                                    : role === "back"
                                      ? "Flat back view of the garment."
                                      : "Optional side profile for depth fitting."}
                                </p>
                              </div>
                            )}
                            <input
                              accept="image/*"
                              className="sr-only"
                              onChange={(event) => handlePhotoUpload(role, event)}
                              type="file"
                            />
                          </label>

                          <div className="mt-3 flex items-center justify-between gap-3">
                            <p className="truncate text-xs text-[#746d64]">
                              {photo ? photo.name : "No file selected"}
                            </p>
                            {photo ? (
                              <button
                                className="rounded-lg border border-[#d8d1c3] px-2 py-1 text-xs font-semibold text-[#625c54] transition hover:bg-white"
                                onClick={() => removeUploadPhoto(role)}
                                type="button"
                              >
                                Remove
                              </button>
                            ) : null}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {orderedUploadPhotos.length > 0 ? (
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                    {orderedUploadPhotos.map((photo) => (
                      <div
                        className="relative h-32 overflow-hidden rounded-lg bg-[#e8e2d3]"
                        key={photo.role}
                        style={{
                          backgroundImage: "url(" + photo.url + ")",
                          backgroundPosition: "center",
                          backgroundSize: "cover",
                        }}
                      >
                        <span className="absolute bottom-2 left-2 rounded-full bg-white/90 px-2 py-1 text-[11px] font-semibold text-[#625c54]">
                          {garmentPhotoLabel(photo.role)}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : null}

                {uploadError ? (
                  <p className="rounded-lg border border-[#c66b52] bg-[#fff3ef] px-3 py-2 text-sm font-semibold text-[#8f321f]">
                    {uploadError}
                  </p>
                ) : null}
              </div>

              <aside className="rounded-lg border border-[#d8d1c3] bg-white p-4">
                <p className="text-sm font-medium uppercase tracking-[0.16em] text-[#857e73]">
                  Mesh build
                </p>
                <div className="mt-4 grid gap-3">
                  {reconstructionStages.map((stage, index) => (
                    <div className="flex items-center gap-3" key={stage}>
                      <span
                        className={
                          "grid h-8 w-8 place-items-center rounded-full text-xs font-bold " +
                          (hasRequiredUploadPhotos
                            ? "bg-[#243f3a] text-white"
                            : "bg-[#ede7d8] text-[#746d64]")
                        }
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
                    Standard mesh first
                  </p>
                  <p className="mt-2 text-center text-xs leading-5 text-[#857e73]">
                    Shirt drafts now start from a neutral T-shirt blockout
                    before texture and photo details are applied.
                  </p>
                </div>
              </aside>
            </div>

            <div className="mt-5 flex flex-col-reverse justify-end gap-3 border-t border-[#dad5c8] pt-4 sm:flex-row">
              <button
                className="rounded-lg border border-[#d8d1c3] px-4 py-3 text-sm font-semibold hover:bg-white disabled:cursor-not-allowed disabled:opacity-60"
                disabled={isSubmitting}
                onClick={closeUploadPrompt}
                type="button"
              >
                Cancel
              </button>
              <button
                className="rounded-lg bg-[#243f3a] px-4 py-3 text-sm font-semibold text-white transition hover:bg-[#1c332f] disabled:cursor-not-allowed disabled:bg-[#a6a096]"
                disabled={!itemName.trim() || !hasRequiredUploadPhotos || isSubmitting}
                type="submit"
              >
                {isSubmitting ? "Creating..." : "Create 3D mesh"}
              </button>
            </div>
          </form>
        </div>
      ) : null}

      {selectedDraftItem ? (
        <div className="fixed inset-0 z-50 grid place-items-center bg-[#232421]/60 p-4">
          <div className="max-h-[92vh] w-full max-w-5xl overflow-y-auto rounded-lg bg-[#f8f7f2] p-5 shadow-2xl">
            <div className="flex items-start justify-between gap-4 border-b border-[#dad5c8] pb-4">
              <div>
                <p className="text-sm font-medium uppercase tracking-[0.18em] text-[#857e73]">
                  3D garment mesh
                </p>
                <h2 className="mt-1 text-3xl font-semibold">
                  {selectedDraftItem.name}
                </h2>
              </div>
              <button
                className="rounded-lg border border-[#d8d1c3] px-3 py-2 text-sm font-semibold hover:bg-white"
                onClick={() => setSelectedDraftItem(null)}
                type="button"
              >
                Close
              </button>
            </div>

            <div className="mt-5 grid gap-5 lg:grid-cols-[1fr_320px]">
              <div className="rounded-lg bg-[#ede7d8] p-6">
                {selectedDraftItem.mesh ? (
                  <GarmentMeshViewer
                    className="min-h-[560px] overflow-hidden rounded-lg border border-[#d8d1c3]"
                    key={
                      selectedDraftItem.mesh.assetUrl +
                      ":" +
                      selectedDraftItem.mesh.generatedAt
                    }
                    mesh={selectedDraftItem.mesh}
                  />
                ) : (
                  <div className="grid min-h-[560px] place-items-center rounded-lg border border-[#d8d1c3] bg-[#f8f7f2] p-4 text-sm font-semibold text-[#625c54]">
                    Building the mesh...
                  </div>
                )}

                <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-sm text-[#625c54]">
                  <div>
                    Drag to orbit the 3D mesh. Scroll to zoom. The view now auto-fits on open.
                  </div>
                  <div className="rounded-full bg-white px-3 py-1 font-semibold text-[#243f3a] shadow-sm ring-1 ring-[#e5dfd1]">
                    {selectedDraftViewCount === 0
                      ? "0 / 0"
                      : String(selectedReferenceIndex + 1) +
                        " / " +
                        String(selectedDraftViewCount)}
                  </div>
                </div>
                <p className="mt-3 text-sm leading-6 text-[#625c54]">
                  The preview starts from a neutral garment base. Uploaded
                  photos are used afterward for garment extraction, texture, and
                  detail fitting without reproducing a person&apos;s face.
                </p>
                {selectedDraftItem.mesh ? (
                  <div className="mt-4 grid gap-3 rounded-lg bg-white/70 p-4 text-sm text-[#4f4a43] ring-1 ring-[#e2dccf] sm:grid-cols-4">
                    <div>
                      <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[#857e73]">
                        Shape base
                      </p>
                      <p className="mt-1 font-semibold text-[#232421]">
                        {templateLabels[selectedDraftItem.mesh.template] ??
                          selectedDraftItem.mesh.template}
                      </p>
                    </div>
                    <div>
                      <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[#857e73]">
                        Extraction quality
                      </p>
                      <p className="mt-1 font-semibold text-[#232421]">
                        {Math.round(
                          selectedDraftItem.mesh.segmentation.confidence * 100,
                        )}
                        %
                      </p>
                    </div>
                    <div>
                      <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[#857e73]">
                        Surface texture
                      </p>
                      <p className="mt-1 font-semibold text-[#232421]">
                        {selectedDraftItem.mesh.extractedTextureUrl
                          ? "Extracted garment"
                          : "Full reference"}
                      </p>
                    </div>
                    <div>
                      <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[#857e73]">
                        Approx. size
                      </p>
                      <p className="mt-1 font-semibold text-[#232421]">
                        {selectedDraftItem.mesh.bounds.width.toFixed(2)} x{" "}
                        {selectedDraftItem.mesh.bounds.height.toFixed(2)} x{" "}
                        {selectedDraftItem.mesh.bounds.depth.toFixed(2)}
                      </p>
                    </div>
                  </div>
                ) : null}
              </div>

              <aside className="rounded-lg border border-[#d8d1c3] bg-white p-4">
                <p className="text-sm font-medium uppercase tracking-[0.16em] text-[#857e73]">
                  Reference captures
                </p>
                <p className="mt-2 text-xs leading-5 text-[#746d64]">
                  These photos now guide detail extraction after the base mesh
                  is built. The render uses the isolated garment texture, not
                  the full image.
                </p>
                <div className="mt-4 grid grid-cols-2 gap-3">
                  {selectedDraftItem.imageUrls?.map((url, index) => (
                    <button
                      className={
                        "relative h-28 overflow-hidden rounded-lg bg-[#e8e2d3] text-left ring-2 transition " +
                        (selectedReferenceIndex === index
                          ? "ring-[#243f3a]"
                          : "ring-transparent hover:ring-[#d8d1c3]")
                      }
                      key={selectedDraftItem.name + "-view-" + String(index)}
                      onClick={() => setSelectedReferenceIndex(index)}
                      style={{
                        backgroundImage: "url(" + url + ")",
                        backgroundPosition: "center",
                        backgroundSize: "cover",
                      }}
                      type="button"
                    >
                      <span className="absolute bottom-2 left-2 rounded-full bg-white/90 px-2 py-1 text-[11px] font-semibold text-[#625c54]">
                        View {index + 1}
                      </span>
                    </button>
                  ))}
                </div>
              </aside>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}
