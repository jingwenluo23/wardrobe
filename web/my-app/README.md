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

## Garment segmentation model

Garment extraction uses `Xenova/segformer_b2_clothes` via transformers.js.
By default those weights are downloaded from huggingface.co the first time an
upload is processed, which makes extraction quality depend on an outbound
request at exactly the wrong moment: if the host cannot reach huggingface.co,
or the request is throttled, extraction silently falls back to a colour
heuristic and the texture comes back visibly worse (the mesh panel reports
82% extraction quality instead of ~95%).

Vendor the weights so the runtime reads them from disk instead:

```bash
npm run fetch-model
```

This downloads about 15 MB into `models/Xenova/segformer_b2_clothes/`, which
is gitignored. It also runs automatically before `npm run build`, so a deploy
that can reach huggingface.co bakes the model in. The script never fails a
build — if the download does not work the app still runs and falls back to
fetching at request time.

Overrides:

- `WARDROBE_MODEL_DIR` — where vendored weights live (default `./models`)
- `WARDROBE_MODEL_CACHE` — cache for runtime downloads (default a temp dir)

On startup the server logs which path it took, so it is easy to confirm:

```
[garment-segmentation] loading model from vendored weights at /app/models
[garment-segmentation] loading model from network (run `npm run fetch-model` to vendor it)
```
