This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Garment CAD Asset

The T-shirt mesh used by the app is a checked-in static CAD asset:

- Canonical source asset: `src/lib/assets/tshirt_cad_source.json`
- JSON runtime asset: `src/lib/assets/tshirt_cad.json`
- OBJ inspection/export asset: `src/lib/assets/tshirt_cad.obj`

Runtime behavior:

- the web app does not procedurally generate the T-shirt mesh at request time
- draft generation copies the checked-in JSON asset and serves it to the viewer
- the viewer only loads and renders mesh data

Offline tooling:

- `npm run cad:refresh` is the single canonical asset refresh step
- it copies the checked-in source mesh from `src/lib/assets/tshirt_cad_source.json`, normalizes metadata/normals/bounds, and exports `src/lib/assets/tshirt_cad.obj`
- `scripts/legacy/generate_tshirt.py` is archived experimentation only and is not the runtime CAD source

If the silhouette changes, regenerate or reshape offline, commit the updated asset, and bump the mesh version in `src/lib/garment-mesh.ts` so cached drafts refresh.

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
