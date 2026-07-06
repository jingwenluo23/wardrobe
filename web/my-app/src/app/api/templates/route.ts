import { NextResponse } from "next/server";

import { garmentTemplates } from "@/lib/garment-templates";

export function GET() {
  return NextResponse.json({
    templates: garmentTemplates.map(({ id, label, category }) => ({
      id,
      label,
      category,
    })),
  });
}
