import { createDraft, listDrafts } from "@/lib/draft-pipeline";

export const runtime = "nodejs";

export async function GET() {
  const drafts = await listDrafts();
  return Response.json(
    { drafts },
    {
      headers: {
        "Cache-Control": "no-store, max-age=0, must-revalidate",
      },
    },
  );
}

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const draft = await createDraft(formData);
    return Response.json({ draft }, { status: 201 });
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : "Unable to create draft." },
      { status: 400 },
    );
  }
}
