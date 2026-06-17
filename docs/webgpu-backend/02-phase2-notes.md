# Phase 2 — Minimal end-to-end WebGPU path

Goal: get a single `SplatMesh` rendering through a WebGPU backend to prove the
plumbing (storage upload → compute dispatch → instanced raster), with a
throwaway sort, behind a runtime backend selector. WebGL2 stays the default and
is untouched.

## What landed

| File | Purpose |
|------|---------|
| `src/webgpu/capability.ts` | Dependency-free WebGPU detection + backend resolution (`isWebGPUAvailable`, `canRequestWebGPUAdapter`, `resolveBackend`, `backendFromUrl`). No `three` import, safe in the WebGL build. |
| `src/webgpu/wgsl.ts` | Small, single-function WGSL unpack helpers (center f16, RGBA u8, max scale). Bit-identical to `splatDefines.glsl` (D7). |
| `src/webgpu/WebGPUSplatRenderer.ts` | The Phase-2 backend: dynamic-imports `three/webgpu`+`three/tsl`, uploads the 16-byte packed buffer as `array<vec4<u32>>`, runs the project compute pass, rasters via a `MeshBasicNodeMaterial` into the shared depth buffer. |
| `src/SparkRenderer.ts` | Added static `SparkRenderer.createRenderer()` factory (capability detect + `WebGPURenderer.init()` + automatic WebGL2 fallback). Existing constructor/paths untouched. |
| `src/index.ts` | Exports `WebGPUSplatRenderer` + capability helpers. |
| `vite.config.ts` | Externalizes `three/webgpu` + `three/tsl`; excludes them from dep pre-bundling. |
| `examples/webgpu-hello/index.html` | Runnable WebGPU example + `?sparkBackend=` note. |

## Design decisions / how it maps to the directives

- **No static `three/webgpu` dependency (protects WebGL-only consumers).** The
  WebGPU module loads `three/webgpu` and `three/tsl` via `await import(...)`
  only when constructed/initialized. The ~85% WebGL path never fetches them and
  consumers don't need them in their import map. Types come from
  `typeof import("three/tsl")` so usage is still fully type-checked.
- **Shared depth (D1).** The instanced splat mesh renders in the same scene
  through `WebGPURenderer` with `depthTest=true`, `depthWrite=false`,
  premultiplied transparent — so it is occluded by opaque mesh geometry in-frame.
- **Two-renderer model (D2).** `createRenderer()` picks `WebGPURenderer` or the
  classic `WebGLRenderer` once; the WebGL2 backend code is unchanged and remains
  the literal fallback. (Note: the WebGPU THREE build does not ship the classic
  `WebGLRenderer`, so an app supporting both backends picks the THREE build at
  load — the example requires WebGPU and points at the classic examples for the
  WebGL2 path. A unified dual-build harness is a Phase 5 item.)
- **Bit-identical packed layout (D7).** Packed splats upload as
  `array<vec4<u32>>` with the exact 16-byte layout; WGSL `unpack2x16float`
  replaces GLSL `unpackHalf2x16`.
- **Residency scope (D5/D6).** Phase 2 supports `PackedSplats` only and throws a
  clear error otherwise; ExtSplats lands later in v1, Paged/dyno route to WebGL.

## Phase-2 simplifications (explicitly temporary)

- **Isotropic projection.** The compute pass projects each splat as a round quad
  sized by its max scale, not the full anisotropic 2D covariance. Full covariance
  projection is Phase 4 ("GPU cull + project") — the compute pass is already the
  right home for it.
- **Throwaway CPU sort.** Draw order is produced by a CPU sort of
  CPU-transformed centers each frame (no GPU readback, but O(N) JS work). This
  still exercises the GPU compute + raster path. The in-frame GPU radix sort
  (which removes the CPU work and the lag) is Phase 3.
- **Fixed `instanceCount`.** Draws `numSplats` instances; the vertex stage emits
  an off-screen degenerate quad for inactive/culled splats (`meta.active==0`),
  mirroring the WebGL `0xffffffff` sentinel. GPU-driven count via
  `IndirectStorageBufferAttribute` is a Phase-4 optimization.

## Public API additions (documented per task)

- `SparkRenderer.createRenderer(options?) => Promise<{ renderer, backend }>` —
  additive static; existing `new SparkRenderer({ renderer })` unchanged.
- `WebGPUSplatRenderer` (class) + `WebGPUSplatRendererOptions`.
- `isWebGPUAvailable`, `canRequestWebGPUAdapter`, `resolveBackend`,
  `backendFromUrl`, `SparkBackendKind`.
- `SparkRendererOptions.renderer` is still typed `THREE.WebGLRenderer`; the
  WebGPU path uses the separate `WebGPUSplatRenderer` rather than overloading the
  WebGL `SparkRenderer` constructor (which is WebGL-hardcoded). Unifying both
  under one `SplatBackend`-holding `SparkRenderer` is deferred to avoid a large
  refactor of the 2,100-line WebGL renderer; flagged for a later phase.

## Verification status

- ✅ **Type-checked.** `tsc --noEmit` is fully clean. Because `@types/three`
  ships full TSL/WebGPU typings, this validates the TSL/WGSL API usage (storage
  buffers, `wgslFn`, compute dispatch, node material, uniforms) against r180.
- ✅ **Lint/format clean** (`biome check`, all 88 files) and existing tests pass
  (via the pre-commit hook).
- ✅ **Runs on real WebGPU hardware.** Built the Rust/WASM package + dev bundle
  and loaded `examples/webgpu-hello/` in a WebGPU browser: the butterfly scene
  renders through `THREE.WebGPURenderer` (`backend: webgpu`), the per-frame loop
  (compute project pass → CPU sort → `renderAsync`) animates across frames, depth
  blending is correct, and there are **zero WebGPU validation warnings/errors**.
- ✅ **Bundle isolation confirmed.** The built `dist/spark.module.js` contains
  `WebGPUSplatRenderer` with **no static** `three/webgpu` / `three/tsl` imports —
  only the runtime `import("three/webgpu")` / `import("three/tsl")` — so
  WebGL-only consumers never load them.

### Toolchain note (building the Rust/WASM dep)

`npm run build:wasm` needs a rustc that has the `wasm32-unknown-unknown` target
*and* is new enough for the deps (≥1.88 for the `image` crate). A Homebrew-only
rust install fails (no wasm32 target); use rustup with its bin dir first on PATH:

```bash
rustup target add wasm32-unknown-unknown
rustup update stable                      # ensure rustc >= 1.88
PATH="$HOME/.cargo/bin:$PATH" npm run build:wasm
npm run build                             # rebuild dist with WebGPUSplatRenderer
npm run dev                               # serve examples
# open examples/webgpu-hello/ in a WebGPU-capable browser
```

`?sparkBackend=webgl2` is reserved for the dual-backend validation harness (Phase 5).

## Next: Phase 3

Replace the throwaway CPU sort with an in-frame multi-pass GPU LSD radix sort
over quantized 32-bit depth keys (8 bits × 4 passes, histogram→scan→scatter),
consumed the same frame. First sub-task: get Phase 2 rendering verified on real
hardware, then build the sort and confirm orbiting no longer pops.
