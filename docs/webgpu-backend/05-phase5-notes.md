# Phase 5 — Validation & polish

## Visual parity

- A fixed-pose A/B harness (`examples/webgpu-compare/{webgpu,webgl}.html`) renders
  the same butterfly at an identical transform + camera, one page per THREE build.
- The first comparison surfaced and fixed the **double-sRGB** color bug (see
  Phase 4 notes): `WebGPURenderer` applies a linear->sRGB output conversion, so
  the fragment now applies `srgbToLinear` (matching the WebGL `encodeLinear`
  path). After the fix, the two backends match closely: **identical splat
  shapes** (anisotropic covariance) and **matching color** (SH + sRGB).
- Residual: minor contrast / SH-sparkle difference, within the perceptual
  threshold (D8 allows non-bit-exact; rasterization & precision differ).
- D8 baseline: the butterfly asset exercises **SH3** (view-dependent / reflective)
  — the highest-divergence color case — and is matched. A **paged SOG** baseline
  is deferred along with paging itself (D4): paged scenes run the WebGL2 session.

### Automated diff (method)

Because the two backends require different THREE builds (one page each), an
automated pixel diff captures each canvas at the identical fixed pose
(`drawImage` onto a 2D canvas -> `getImageData`), stores the WebGPU pixels in
`localStorage`, then diffs against the WebGL capture on the next page. This was
exercised manually; wiring it into CI needs a headless WebGPU runner (the dev
preview browser de-prioritizes WebGPU contexts after many rapid reloads, which
makes long automated multi-reload sequences flaky — a runner concern, not a
backend one).

## Automatic fallback

`test/webgpuFallback.test.ts` (runs in Node, where there is no `navigator.gpu`)
verifies:
- `isWebGPUAvailable()` / `canRequestWebGPUAdapter()` report `false` (no throw),
- `resolveBackend()` falls back to `"webgl2"`,
- `forceBackend` (and the `?sparkBackend=` URL override) win when set.

`SparkRenderer.createRenderer()` additionally wraps `WebGPURenderer.init()` in a
try/catch that falls back to a classic `WebGLRenderer` on any hard failure.

## Benchmark (FPS vs splat count)

`examples/webgpu-benchmark/{webgpu,webgl}.html?count=N` generate N synthetic
splats (shared `gen.js`) and report rolling FPS.

Measured on the dev machine, **1,000,000 splats**, a deliberately
**overdraw-heavy** close-up scene (1M large overlapping splats filling the view):

| backend | FPS | time-to-first-correct-frame |
|---------|-----|-----------------------------|
| WebGPU  | ~9  | immediate (sorted in-frame) |
| WebGL2  | ~4  | ~40 s+ (readback+worker sort warmup) |

Two findings:
1. **Throughput**: WebGPU ~2x faster here. This scene is fragment-fill-bound
   (extreme overdraw), so it understates the sort difference — it is a stress
   test, not a sort-isolating microbenchmark. Spread-out / real scenes show
   higher absolute FPS for both.
2. **Latency**: WebGPU shows correct, fully-sorted splats *immediately*; the
   WebGL2 path renders black/partial for tens of seconds until its async
   readback+worker sort converges, then keeps lagging the camera (the popping
   the project set out to remove). This is the clearest demonstration of the
   in-frame-sort win.

5M/10M were not captured here: the dev preview browser's WebGPU context budget
makes repeated large-allocation reloads unreliable in this environment. The
harness is provided to run those counts (and real assets) on a normal browser.

### Sort scalability fix

The Phase-3 radix sort used a single-invocation O(histLen) prefix-sum scan, which
would dominate frame time at high splat counts. Phase 5 replaces it with a
**2-level block scan** (`SCAN_BLOCKS = 1024`): reduce each block, scan the 1024
block sums, then scan each block seeded with its base — O(histLen / 1024)
parallel work. The radix unit test mirrors this exactly and still passes all
cases (stability, ties, sentinels, tile boundaries).

## Examples & docs

- New examples: `webgpu-hello` (basic), `webgpu-compare` (A/B parity),
  `webgpu-benchmark` (FPS vs count). See `examples/README-webgpu.md`.
- Backend docs: `docs/webgpu-backend/00..05`.

## Status vs the original plan

| Phase | Status |
|-------|--------|
| 0 Exploration | done |
| 1 Design | done (D1-D8 folded in) |
| 2 Minimal E2E | done, verified on hardware |
| 3 GPU radix sort | done, unit-tested + hardware-verified, no-pop |
| 4 GPU cull+project+SH | done, visual parity + sRGB fix |
| 5 Validation | parity (visual + method), fallback (test), benchmark (1M), docs |

### Deferred (documented, by design)

- **Paging** (PagedSplats/SplatPager): WebGL2 session per D4; layout is
  paging-ready (sort/index operate on the dense active set).
- **dyno modifiers / SplatEdit / SplatSkinning / ExtSplats**: WebGL2 session per
  D5/D6 (WebGPU covers PackedSplats; ExtSplats upload is a small follow-up).
- **2DGS flat splats, depth-of-field** (focal/aperture): not ported.
- **Workgroup-shared radix** and **multi-mesh** WebGPU scenes: perf/scope
  follow-ups.
- **Headless WebGPU CI** for the automated parity diff.
