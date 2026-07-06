import { NextResponse } from "next/server";

import { deleteDraft } from "@/lib/draft-store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const removed = deleteDraft(id);

  if (!removed) {
    return NextResponse.json(
      { error: "That draft no longer exists." },
      { status: 404 },
    );
  }

  return NextResponse.json({ ok: true });
}
