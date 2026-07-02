import { NextResponse } from "next/server";

import { createDraft, listDrafts } from "@/lib/draft-store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json({ drafts: listDrafts() });
}

export async function POST(request: Request) {
  let formData: FormData;
  try {
    formData = await request.formData();
  } catch {
    return NextResponse.json(
      { error: "Expected a multipart form upload." },
      { status: 400 },
    );
  }

  const name = String(formData.get("name") ?? "").trim();
  const category = String(formData.get("category") ?? "Tops").trim();
  const templateId = String(formData.get("templateId") ?? "").trim() || null;
  const frontPhoto = formData.get("frontPhoto");
  const backPhoto = formData.get("backPhoto");
  const sidePhoto = formData.get("sidePhoto");

  if (!name) {
    return NextResponse.json(
      { error: "An item name is required." },
      { status: 400 },
    );
  }

  if (!(frontPhoto instanceof File) || !(backPhoto instanceof File)) {
    return NextResponse.json(
      { error: "Front and back photos are required." },
      { status: 400 },
    );
  }

  try {
    const draft = await createDraft({
      name,
      category: category || "Tops",
      templateId,
      frontPhoto,
      backPhoto,
      sidePhoto: sidePhoto instanceof File ? sidePhoto : null,
    });
    return NextResponse.json({ draft }, { status: 201 });
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Unable to process the uploaded photos.",
      },
      { status: 500 },
    );
  }
}
