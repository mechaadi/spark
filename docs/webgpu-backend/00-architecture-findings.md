# Phase 0 — Architecture Findings

WebGPU compute-based rendering backend for Spark. This document maps the **real**
pipeline as it exists in the source (commit `155af7b`), corrects several inaccuracies
in the task brief, and identifies exactly where the WebGL2 assumptions live.

All citations are `file:line` from the repo root.

---

## TL;DR / executive summary

- Spark renders splats with a **single instanced draw** of a quad (`SplatGeometry`,
  an `InstancedBufferGeometry`), `instanceCount = activeSplats`. Each instance looks up
  its splat index from an **`ordering` integer texture** in the vertex shader.
- Sort order is produced by: render per-splat **depth to a float render target** →
  **async GPU→CPU readback** (`readRenderTargetPixelsAsync`) → **postMessage to a Web
  Worker** running a **Rust/WASM radix sort** → upload the resulting index array back
  into the `ordering` texture. This whole chain is `await`-driven and **decoupled from
  the draw**, which is the root cause of the popping/lag.
- The sort is **NOT a JS bucket sort** as the brief states — it is a two-pass 16-bit LSD
  **radix sort in Rust compiled to WASM** (`rust/spark-rs/src/sort.rs`,
  `sort32_splats`). A recent commit (`6d24120 "make sort32 fast"`) just optimized it.
- **`SparkViewpoint` does not exist anywhere in the codebase.** The brief lists it as a
  public API to preserve. There is no such class. Viewpoint/camera state lives on the
  `SplatAccumulator` (`viewOrigin`, `viewDirection`, `viewToWorld`) and on transient LOD
  state in `SparkRenderer`. **This needs a decision** (see §10).
- Splat data lives in GPU **`RGBA32UI` `DataArrayTexture`s** (integer textures), 16
  bytes/splat, fetched with `texelFetch` and bit-unpacked in-shader. SH coefficients are
  separate integer textures.
- `dyno` is a **GLSL-ES-3.00 code generator** with no backend abstraction. It emits GLSL
  text and builds `THREE.RawShaderMaterial`. Porting it to WGSL is a large, separate
  effort. **Recommendation: scope the first WebGPU pass to standard (non-dyno) splats.**
- THREE.js is **r180** (`^0.180.0`), which ships `three/webgpu` (`WebGPURenderer`) and
  `three/tsl`. The codebase currently imports **zero** WebGPU/TSL symbols.

---

## 1. Per-frame render flow (WebGL2 path)

`SparkRenderer` extends a THREE mesh and hooks `onBeforeRender`. The user drives a normal
`THREE.WebGLRenderer.render(scene, camera)` loop.

- Public entry: `SparkRenderer.render()` simply calls `this.renderer.render(scene, camera)`
  with a static `sparkOverride` set — `src/SparkRenderer.ts:1768`.
- The real work happens in the mesh's `onBeforeRender` — `src/SparkRenderer.ts:726-857`:
  1. Resolve the active render target & size (incl. WebXR quirks).
  2. Update shader uniforms from the camera (`renderToViewPos/Quat/Basis`, near/far,
     pixel-radius limits, 2DGS flag, blur, time).
  3. Set the draw size: `geometry.instanceCount = spark.activeSplats` —
     `src/SparkRenderer.ts:772`.
  4. Bind the current `ordering` texture uniform —
     `this.uniforms.ordering.value = spark.orderingTexture ?? SparkRenderer.emptyOrdering`
     (`src/SparkRenderer.ts:812`).
  5. If `autoUpdate` and the frame is new, call `updateInternal()` →
     `prepareGenerate()`/`generate()` and then `await this.driveSort()`
     (`src/SparkRenderer.ts:831-854`, `:988`).

### The instanced draw

`SplatGeometry` is an `InstancedBufferGeometry` holding one quad (4 verts, 2 tris) —
`src/SplatGeometry.ts:1-21`. THREE issues one instanced draw of `instanceCount` quads.
Per instance, the vertex shader resolves its splat index from the `ordering` texture:

```glsl
// src/shaders/splatVertex.glsl:48-57
ivec2 orderingCoord = ivec2((gl_InstanceID >> 2) & 4095, gl_InstanceID >> 14);
uint splatIndex = texelFetch(ordering, orderingCoord, 0)[gl_InstanceID & 3];
if (splatIndex == 0xffffffffu) { return; }      // culled / inactive sentinel
ivec3 texCoord = splatTexCoord(int(splatIndex));
uvec4 packedData = texelFetch(extSplats, texCoord, 0);
```

So **sorted order is consumed by reading `ordering[gl_InstanceID] → splatIndex`**, then
the splat attributes are fetched from the packed data texture. This is the seam the
WebGPU path must reproduce: produce a sorted index buffer the rasterizer indexes by
instance.

---

## 2. The sort pipeline (the thing we're replacing)

Driven by `driveSort()` — `src/SparkRenderer.ts:991-1110`. Fully `async`, guarded by a
`this.sorting` re-entrancy flag and a `this.sortDirty` flag.

**Stage A — depth compute (on GPU, in the generate pass).** Per-splat depth is written to
a float MRT output `target3` during splat generation, not in a dedicated pass:

```glsl
// src/dyno/output.ts:202-213  (OutputSplatDepth)
float metric = 1.0 / 0.0;                 // +inf = inactive
if (isGsplatActive(${gsplat}.flags)) {
  vec3 center = ${gsplat}.center - ${viewCenter};
  if (${sortRadial}) metric = length(center);            // radial
  else metric = dot(center, ${viewDir}) + 100.0;         // eye-depth + bias to reduce popping
}
target3 = floatToVec4(metric);
```

The generate pass renders into a `THREE.WebGLArrayRenderTarget`
(`RGBAIntegerFormat`/`UnsignedIntType` for packed data, plus a float layer for depth) —
`src/SplatAccumulator.ts:157-181`, dispatched via a fullscreen-quad "pseudo-compute"
(see §6).

**Stage B — async GPU→CPU readback.** `readbackDepth()` packs depth to a portable RGBA8
layout and reads it back asynchronously:

```ts
// src/SparkRenderer.ts:1712 / src/Readback.ts:248
renderer.readRenderTargetPixelsAsync(current.target, 0, 0, W, H, subReadback, ...)
```

`readRenderTargetPixelsAsync` is THREE's wrapper over WebGL2 `getBufferSubData` w/ a
PBO + fence. This is **one async boundary** — the result is not available the same frame.

**Stage C — worker round-trip to WASM radix sort.** The depth buffer is transferred to a
Web Worker:

```ts
// src/SparkRenderer.ts:1049
const result = await this.sortWorker.call("sortSplats32", { numSplats, readback, ordering });
```

`SplatWorker` posts `{id,name,args}` with transferables (`src/SplatWorker.ts:80`). The
worker handler `sortSplats32` (`src/worker.ts:86-97`) calls WASM `sort32_splats`.

The sort itself — **two-pass, 16-bit-radix LSD sort over inverted f32 depth bits**,
descending (far → near), excluding `+inf` (inactive) splats:

```
// rust/spark-rs/src/sort.rs  (sort32_splats / sort.rs:118-255)
// - DEPTH_INFINITY_F32 = 0x7f800000 marks inactive (excluded from output)
// - RADIX_BASE = 65536; pass 1 = low 16 bits, pass 2 = high 16 bits
// - returns active_splats (count of finite-depth splats)
```

(There is also an older `sort_splats` / `sort_splats` path at `worker.ts:82`; the live
one for the 32-bit ordering texture is `sort32_splats`.)

**Stage D — upload ordering + promote.** The returned index array is written into the
`ordering` `DataTexture` (`RGBA32UI`, 4096 × rows). First time allocates a new
`THREE.DataTexture`; subsequent updates use a **raw WebGL2 `texSubImage2D`**
(`uploadU32DataTextureRows`, `src/utils.ts:1687-1720`). Only after the sort resolves is
the freshly-sorted accumulator promoted to `display` (gated on `mappingVersion`) —
`src/SparkRenderer.ts:1103-1108`.

### Why it lags / pops

`driveSort()` chains **three async hops** (readback PBO fence → worker postMessage →
texSubImage upload), all behind `await`, and only swaps the visible `ordering` after they
complete. Meanwhile the rasterizer keeps drawing with the **previous** `ordering`
texture. The depth fed to the sort was captured from an earlier camera pose, so when the
camera orbits, the visible order is 1–2+ frames stale → splats reorder/pop. Above ~1M
splats the CPU/WASM sort + readback bandwidth also becomes a throughput wall.

The WebGPU goal is to collapse Stages A–D into in-frame compute passes (depth → GPU
radix sort → sorted index buffer) with **no readback and no worker**, consumed by the
raster pass the same frame.

---

## 3. Packed splat data layout (16 bytes/splat)

Each splat = **4 × uint32 = 16 bytes**, packed/unpacked in
`src/shaders/splatDefines.glsl:174-239`.

| Word | Bits | Field |
|------|------|-------|
| 0 | `[0:8]`,`[8:16]`,`[16:24]`,`[24:32]` | R, G, B, A (uint8 each; RGB rescaled by `rgbMin/Max`, A = opacity) |
| 1 | `[0:16]`,`[16:32]` | center.x, center.y (float16 via `packHalf2x16`) |
| 2 | `[0:16]` / `[16:24]` / `[24:32]` | center.z (float16) / quatX (uint8) / quatY (uint8) |
| 3 | `[0:8]`,`[8:16]`,`[16:24]` / `[24:32]` | scaleX, scaleY, scaleZ (uint8 log-space) / quatZ (uint8) |

- Scale: uint8 in log space, `scale = exp(lnScaleMin + (v-1)*step)`, `v=0` means flat/2DGS;
  `LN_SCALE_MIN=-12, LN_SCALE_MAX=9` (`src/defines.ts:7-8`).
- Rotation: 24-bit octahedral-XY (8/8) + 8-bit angle, decoded by `decodeQuatOctXy88R8`
  (`src/shaders/splatDefines.glsl:84-108`).
- Storage: `THREE.DataArrayTexture`, `format=RGBAIntegerFormat`, `type=UnsignedIntType`,
  `internalFormat="RGBA32UI"`, dims `SPLAT_TEX_WIDTH=2048 × 2048 × depth`
  (`src/PackedSplats.ts:719-728`, `src/defines.ts:26-28`). Index → texel via
  `splatTexCoord` (`splatDefines.glsl:502-507`).
- **`ExtSplats`**: optional extended encoding with float32 centers (second texture);
  **`PagedSplats`/`SplatPager`**: streams 256×256 (=65536-splat) pages into a fixed GPU
  texture pool with LRU eviction + an indirection "indices" texture
  (`src/SplatPager.ts:37-39,598-616,971-1000`).
- **Spherical harmonics**: separate integer textures (SH1 = `RG32UI`, SH2/SH3 =
  `RGBA32UI`), evaluated per-view-direction in shader (`src/PackedSplats.ts:1071-1270`).

### WebGL semantics that won't translate 1:1 to WebGPU

- Integer **textures as data arrays** (`RGBA32UI`/`RG32UI` + `texelFetch`) → WebGPU
  prefers **storage buffers** (`array<vec4<u32>>`) + `textureLoad`/direct index. We can
  keep the *bit layout* identical and just change the *binding* (texture → storage buffer).
- `addLayerUpdate()` lazy dirty-flag uploads (`SplatPager.ts:1018-1021`) → WebGPU needs
  explicit `queue.writeBuffer`/`writeTexture`.
- `packHalf2x16`/`unpackHalf2x16`, `packSnorm2x16` have **no WGSL builtins** — need helper
  functions (or native `f16` where available).
- `internalFormat` string negotiation → WGSL uses format enums (`rgba32uint`).

---

## 4. `dyno` programmable splats — scope decision input

`dyno` (`src/dyno/*`, `src/dyno.ts`) is a **runtime GLSL-ES-3.00 code generator**: users
compose typed nodes (math/trig/vecmat/logic/texture/control/convert/transform/uniforms/
splats); `DynoBlock.generateBlock()` topo-sorts them and each node emits **raw GLSL text**;
`DynoProgram` assembles it into a template and builds a `THREE.RawShaderMaterial`
(`glslVersion: THREE.GLSL3`) — `src/dyno/program.ts:45,107-121`.

- The user "modify splat" hook (`objectModifier`/`worldModifier` on `SplatMesh`) runs
  **inside the generate fragment shader** (render-to-texture), **not** a compute pass and
  **not** transform feedback. `GsplatGenerator: Dyno<{index:int} → {gsplat:Gsplat}>`
  (`src/SplatGenerator.ts:23`) is compiled into the fullscreen-quad generate material.
- It is **100% GLSL-coupled**: `texelFetch`/`texture`/`textureSize`, `packHalf2x16`,
  `floatBitsToUint`, sampler types, `#include`, GLSL builtins — no backend abstraction
  exists. Porting `dyno` to WGSL means a backend-emitter abstraction across ~60 node
  classes + WGSL helpers for the missing intrinsics. **Estimated 4–6 weeks**; standard
  (non-dyno) splats on WebGPU is **~1 week** of the shader/binding work.

**Decision input for Phase 1:** scope the initial WebGPU backend to **standard packed
splats (no dyno modifiers, no SH first cut optional)**, detect dyno usage and **fall back
to WebGL2** for any mesh that uses object/world modifiers. Do not break dyno.

---

## 5. Every place that assumes `THREE.WebGLRenderer`

- `SparkRendererOptions.renderer: THREE.WebGLRenderer` (required) + `readonly renderer`
  field — `src/SparkRenderer.ts:33,328`.
- `WEBGL_provoking_vertex` extension grab — `src/SparkRenderer.ts:565-572`.
- `renderer.getContext() as WebGL2RenderingContext` + raw `gl.*` calls (flush, texSubImage,
  pixelStorei, bindBuffer) — `src/SparkRenderer.ts:964,1629-1654,1725`; `src/utils.ts:1687`.
- `THREE.WebGLRenderTarget` / `WebGLArrayRenderTarget` / `WebGLCubeRenderTarget` —
  `src/SparkRenderer.ts:452-453,597-608,1893-1933`; `src/SplatAccumulator.ts:157-181`.
- `readRenderTargetPixelsAsync` readback — `src/SparkRenderer.ts:1712`; `src/Readback.ts:248`.
- `renderer.properties.get(texture).__webglTexture` internal access for direct uploads —
  `src/utils.ts:1687-1720`, `src/SparkRenderer.ts:1629-1654`.
- Save/restore of WebGL render state (`getRenderTarget`, `xr.enabled`, `autoClear`) —
  `src/SparkRenderer.ts:728-752`.

These are the integration points a backend abstraction must straddle.

---

## 6. Existing "GPU compute" mechanism (WebGL pseudo-compute)

Spark already does GPU data transforms via **fullscreen-quad render-to-texture**, not real
compute. Templates `computeUvec4.glsl`, `computeVec4.glsl`, `computeUvec4_Vec4.glsl`,
`computeUvec4x2_Vec4.glsl` (`src/shaders/`, registered in `src/shaders.ts`) have a
`{{ STATEMENTS }}` slot filled by dyno; a fragment's `gl_FragCoord` maps to a splat index
and the fragment outputs the packed splat (and depth via MRT). Invoked through a
`FullScreenQuad` against an array render target. This is the conceptual analog the WebGPU
path replaces with real `@compute` dispatches.

---

## 7. Build / THREE / public API surface

- THREE **r180** (`^0.180.0`, peer `>=0.180.0`) — ships `three/webgpu` (`WebGPURenderer`,
  `WebGPUBackend`, `StorageBufferAttribute`) and `three/tsl`. **No WebGPU/TSL imports exist
  in `src/` today.**
- Vite library build; GLSL imported as strings via **vite-plugin-glsl** (`vite.config.ts`).
  WGSL would need an analogous loader or `?raw` imports (also for the worker build).
- `three` is externalized in the bundle (`globals: { three: "THREE" }`).
- Public exports (`src/index.ts`) include `SparkRenderer`, `SplatMesh`, `SplatAccumulator`,
  `PackedSplats`, `ExtSplats`, `SplatPager`, `Readback`, the `dyno`/`generators`/
  `modifiers` namespaces, editing classes, etc. **`SparkViewpoint` is not exported and not
  defined** (see §10).
- Tests: `node --test` over `test/**/*.test.ts` — only a CPU util test exists. **No
  headless WebGL/WebGPU render harness, no visual-diff tooling.** Phase 5 must add one.

---

## 8. Minimal usage pattern (what must keep working)

```js
const renderer = new THREE.WebGLRenderer();
const spark = new SparkRenderer({ renderer });
scene.add(spark);
const mesh = new SplatMesh({ url: "foo.spz" });
scene.add(mesh);
renderer.setAnimationLoop(() => renderer.render(scene, camera));
```

`SparkRenderer` and `SplatMesh` are scene-graph `Object3D`s; the user owns the render loop
and the renderer. A WebGPU path must either accept a `THREE.WebGPURenderer` here or run a
raw-WebGPU side-channel keyed off the same scene — a Phase 1 decision.

---

## 9. Corrections to the task brief

1. **"CPU bucket sort on a worker thread (SplatWorker)"** — it is a **Rust/WASM two-pass
   16-bit radix sort** (`rust/spark-rs/src/sort.rs`, `sort32_splats`), invoked via the
   worker. Not a JS bucket sort. (The brief's intuition about the *cost/lag* is right;
   the mechanism is different.)
2. **"SparkViewpoint constructor options and methods"** — **no such class exists.** See §10.
3. **"reads them BACK to the CPU … per-splat camera distances"** — accurate; depth is an
   MRT float output during generate, read back via `readRenderTargetPixelsAsync`.
4. Depth metric is **radial distance** by default (`sortRadial=true`), not eye-Z; eye-Z is
   the non-radial fallback with a +100 bias.

## 10. Open question that needs your decision

The brief repeatedly names **`SparkViewpoint`** as a public API to preserve, but there is
no `SparkViewpoint` class, type, or export in the codebase. The nearest concepts are:

- per-frame camera state captured into `SplatAccumulator` (`viewOrigin`, `viewDirection`,
  `viewToWorld`) — `src/SplatAccumulator.ts:61-63`;
- transient LOD pose overrides on `SparkRenderer` (`lodPosOverride`, `lodQuatOverride`,
  `lastLod/currentLod`) — `src/SparkRenderer.ts:410-425`.

Either (a) the brief is working from an outdated/aspirational doc, or (b) "SparkViewpoint"
refers to a planned abstraction. **I'll treat the per-view sort/render state on
`SparkRenderer`+`SplatAccumulator` as the "viewpoint" surface to preserve**, unless you
tell me otherwise.

---

## 11. What this implies for the design (Phase 1 preview, not a commitment)

- Introduce a **backend abstraction** at the `SparkRenderer` seam: the WebGL2 path stays
  default; a WebGPU path is selected by capability detection + a force flag.
- Keep the **16-byte packed layout bit-for-bit**; change only the GPU binding (integer
  texture → storage buffer) and the unpack site (`texelFetch` → buffer index/`textureLoad`).
- Replace Stages A–D with in-frame compute: **cull/project → quantized depth keys → GPU
  LSD radix sort → sorted index storage buffer**, consumed the same frame by a lightweight
  raster pass that indexes splat data by `sortedIndex[instance]`.
- **Scope v1 to standard splats**; detect dyno modifiers / SH / paged streaming and fall
  back to WebGL2 per-mesh, documenting what's deferred.
- Build a **visual-parity + FPS-vs-count harness** (none exists today).

These are sketched for context only — the actual proposal, with the
THREE-WebGPU-vs-raw-WGSL decision and buffer layouts, comes in `01-design.md` after your
go-ahead.
