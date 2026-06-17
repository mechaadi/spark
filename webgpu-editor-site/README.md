# Spark WebGPU Editor — Vercel deploy

A self-contained static build of the WebGPU splat editor (`examples/webgpu-editor`),
ready to host on Vercel (or any static host).

- `three`, `three/webgpu`, `three/tsl`, addons, `lil-gui`, and `stats.js` load from
  jsDelivr, pinned to the versions Spark is built against (three `0.180.0`,
  lil-gui `0.20.0`).
- Only the Spark bundle is hosted locally as `./spark.module.js`.
- Loads `butterfly.spz` from sparkjs.dev by default; append `?url=<splat-url>` to
  load any `.ply/.spz/.splat/.ksplat/.zip/.sog/.rad`.

WebGPU needs a **secure context** — Vercel serves over HTTPS, so it just works.
On a browser/GPU without WebGPU the page shows a fallback message.

## Deploy

`spark.module.js` is **gitignored** (it's a build artifact), so refresh it before
deploying. From the repo root:

```bash
npm run build                              # produces dist/spark.module.js (production)
cp dist/spark.module.js webgpu-editor-site/
cd webgpu-editor-site
npx vercel            # preview deploy  (or: npx vercel --prod)
```

`.vercelignore` is present so Vercel uploads `spark.module.js` even though git
ignores it. If you change Spark's source, re-run the two copy steps above and
redeploy.

## Notes

- Pin the CDN versions in `index.html`'s import map to match the `three` version
  Spark was built against — a mismatch can break TSL/WebGPU APIs.
- This mirrors `examples/webgpu-editor/index.html`; keep them in sync if you edit
  the editor.
