import { deleteDraft, getDraft } from "@/lib/draft-pipeline";

export const runtime = "nodejs";

type RouteContext = {
  params: Promise<{ id: string }>;
};

export async function GET(_request: Request, context: RouteContext) {
  const { id } = await context.params;
  const draft = await getDraft(id);

  if (!draft) {
    return Response.json({ error: "Draft not found." }, { status: 404 });
  }

  return Response.json({ draft });
}

export async function DELETE(_request: Request, context: RouteContext) {
  const { id } = await context.params;
  const deleted = await deleteDraft(id);

  if (!deleted) {
    return Response.json({ error: "Draft not found." }, { status: 404 });
  }

  return Response.json({ ok: true });
}
