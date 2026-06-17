# Phase 3 — In-frame GPU radix sort

Goal: replace Phase 2's throwaway CPU sort with a multi-pass LSD GPU radix sort
over quantized 32-bit depth keys, produced and consumed in the same frame — no
GPU->CPU readback, no worker round-trip, no sort lag. Verify back-to-front
correctness and that orbiting no longer pops.

## What changed

- **Project pass now also emits sort inputs.** Per splat it writes a 32-bit
  depth key (`keyA`) and an identity payload (`idxA = i`). Inactive/culled
  splats get key `0xffffffff` so they sort to the end and are discarded by the
  raster (`projected.active == 0`). The key is `sparkDepthKey()`
  ([wgsl.ts](../../src/webgpu/wgsl.ts)): the standard float->sortable-uint flip,
  bit-inverted, so an **ascending** radix sort yields **far -> near** order.
- **GPU LSD radix sort** ([WebGPUSplatRenderer.ts](../../src/webgpu/WebGPUSplatRenderer.ts)
  `buildPipeline`): 4 passes of 8 bits over the 32-bit key, ping-pong key/index
  buffers, final sorted index in `idxA` which the raster indexes by instance.
- **Removed** the CPU sort, the per-frame order upload, and the CPU center
  precompute. `prepareFrame` now just updates uniforms and dispatches the
  compute pipeline; the raster consumes the GPU-sorted `idxA` the same frame.

## Radix sort structure (one tile per invocation)

Chosen for correctness-first with minimal API risk: **no atomics, no shared
memory, no barriers**. Each tile owns a disjoint span of `TILE = 256` elements
and writes to a disjoint output range, so there are no races. Per pass:

1. **count** (`numTiles` invocations): each tile builds a 256-bin histogram of
   its current 8-bit digit and writes it **digit-major** (`digit*numTiles + t`)
   to `tileHist`.
2. **scan** (1 invocation): a single exclusive prefix sum over the whole
   digit-major `tileHist`. Because of the digit-major layout, the prefix at
   `(digit, tile)` is exactly that bucket's global base offset — stable.
3. **scatter** (`numTiles` invocations): each tile re-reads its digits and emits
   `(key, payload)` to its running global offset, ping-ponging A<->B.

After 4 passes (even) the sorted payload is back in `idxA`.

This mirrors the PlayCanvas v2.19 GSplat radix architecture (multi-pass LSD,
per-tile histograms + global scan + stable scatter), reimplemented for Spark's
buffers. The one-tile-per-invocation layout trades peak throughput for
simplicity; a workgroup-shared version with parallel scan is a **Phase 5 perf**
item (the single-invocation scan over `numTiles*256` is the main bottleneck at
very high splat counts).

## Why orbiting no longer pops

By construction. The WebGL2 path computes the order from depths read back from a
*previous* frame (GPU readback + worker round-trip), so the visible order lags
the camera by 1-2+ frames — the cause of popping. The WebGPU path computes the
order in compute from the *current* frame's `modelView`/`projection` uniforms
and the raster consumes it in the same `renderer.compute(...)` +
`renderer.renderAsync(...)` submission. There is no frame where the render uses
a stale order, so there is nothing to pop.

## Verification

- ✅ **Algorithm correctness** — `test/webgpuRadixSort.test.ts` is a faithful CPU
  mirror of the exact tiled count/scan/scatter logic (same tiling, digit-major
  layout, 4 x 8-bit passes). It checks ascending output, valid permutation,
  **stability** vs a stable reference sort, heavy ties, all-equal,
  already-sorted, reversed, tile-boundary sizes (1, 255, 256, 257, 1000, 1024,
  5000), the `0xffffffff` sentinel sorting to the end, and the `sparkDepthKey`
  far->near monotonicity. All pass.
- ✅ **On real WebGPU hardware** — `examples/webgpu-hello/` renders the butterfly
  through the GPU radix sort with coherent depth ordering and correct alpha
  blending, animating across frames (the sort re-runs every frame), with **zero
  WebGPU validation warnings/errors**.
- Note on GPU buffer readback: TSL's `storage()` tracks the GPU buffer under the
  storage *node*, not the `StorageBufferAttribute`, so
  `renderer.getArrayBufferAsync(attribute)` cannot find it. Phase 5's parity and
  benchmark harness uses **canvas pixel-diff and FPS timing** (which need no
  buffer readback), so this does not block it; a node-level readback helper can
  be added later if raw buffer inspection is wanted.

## Still deferred (unchanged from the design)

- Isotropic (round) projection and no SH — **Phase 4** adds full anisotropic 2D
  covariance projection + SH<=3 evaluation in the compute pass, which is what
  closes the visual-quality gap vs WebGL.
- ExtSplats / Paged / dyno residency route to the WebGL2 session.
