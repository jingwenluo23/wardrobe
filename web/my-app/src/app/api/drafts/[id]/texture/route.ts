import { getDraftMeshTexture } from "@/lib/draft-pipeline";

export const runtime = "nodejs";

type RouteContext = {
  params: Promise<{ id: string }>;
};

export async function GET(_request: Request, context: RouteContext) {
  const { id } = await context.params;
  const texture = await getDraftMeshTexture(id);

  if (!texture) {
    return Response.json({ error: "Texture not found." }, { status: 404 });
  }

  return new Response(texture, {
    headers: {
      "Content-Type": "image/png",
      "Cache-Control": "no-store",
    },
  });
}
