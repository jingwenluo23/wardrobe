import { getDraftPhoto } from "@/lib/draft-pipeline";

export const runtime = "nodejs";

type RouteContext = {
  params: Promise<{ id: string; fileName: string }>;
};

export async function GET(_request: Request, context: RouteContext) {
  const { id, fileName } = await context.params;
  const photo = await getDraftPhoto(id, fileName);

  if (!photo) {
    return Response.json({ error: "Photo not found." }, { status: 404 });
  }

  return new Response(photo.content, {
    headers: {
      "Content-Type": photo.type,
      "Cache-Control": "private, max-age=31536000, immutable",
    },
  });
}
