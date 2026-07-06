import { getDraftMeshAsset } from "@/lib/draft-pipeline";

export const runtime = "nodejs";

type RouteContext = {
  params: Promise<{ id: string }>;
};

export async function GET(_request: Request, context: RouteContext) {
  const { id } = await context.params;
  const asset = await getDraftMeshAsset(id);

  if (!asset) {
    return Response.json({ error: "Mesh not found." }, { status: 404 });
  }

  return Response.json(asset, {
    headers: {
      "Cache-Control": "no-store, max-age=0, must-revalidate",
    },
  });
}
