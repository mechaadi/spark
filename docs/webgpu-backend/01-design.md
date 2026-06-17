# Phase 1 — WebGPU Backend Design

Integration design for a WebGPU compute-based rendering backend in Spark. This folds in
the eight product-owner directives (**D1–D8**) as hard constraints and grounds every
decision in the verified r180 API surface and the Phase 0 architecture map
([00-architecture-findings.md](00-architecture-findings.md)).

**Status: proposal. No implementation code until approved.**

---

## 0. Directives as constraints (recap)

| # | Constraint | Where it lands in this design |
|---|-----------|-------------------------------|
| D1 | Build on `THREE.WebGPURenderer`, render into the **same depth buffer** as mesh geometry. No raw-WebGPU side-channel. | §2, §7 |
| D2 | **Two-renderer model**, chosen once at app init. Existing WebGL2 path untouched and is the literal fallback (not WebGPURenderer's internal WebGL fallback). | §2, §3 |
| D3 | **SH3 in scope.** Evaluate SH order 3 per view dir from SH storage buffers. | §4.3, §5.2 |
| D4 | **Paging is first-class.** Sort + sorted-index operate over the **resident/active set via indirection** from day one; whole-session WebGL fallback if paging can't land. | §4.4, §5, §6 |
| D5 | Formats decode upstream. Scope by **GPU residency** (PackedSplats, ExtSplats, PagedSplats). v1 covers PackedSplats **and** ExtSplats. | §4 |
| D6 | dyno modifiers / SplatEdit / SplatSkinning ⇒ **whole-session WebGL2**. Detect + warn. WGSL dyno emitter out of v1. | §8 |
| D7 | **16-byte packed layout bit-for-bit.** Storage buffers + WGSL unpack builtins. No native `f16` extension dependency. | §4.1, §5.2 |
| D8 | Phase 5 parity baseline includes an **SH3 reflective** scene and a **paged SOG** scene. | §10 |

---

## 1. Decision: THREE.WebGPURenderer + TSL/WGSL (not raw WebGPU, not pure TSL)

**Chosen:** Build the backend on `three/webgpu`'s `WebGPURenderer`, author the pipeline with
**TSL node graphs** for structure (storage bindings, compute dispatch wiring, material
nodes) and drop to **raw WGSL via `wgslFn()`** for the hot, hand-tuned kernels (bit-unpack,
octahedral-quat decode, SH evaluation, the radix-sort passes).

**Why TSL+WGSL hybrid rather than pure raw WebGPU:**
- D1 requires real WebGL↔mesh **depth interop in one frame**. Only a renderer that owns the
  scene's depth attachment can do this; `WebGPURenderer` shares one depth buffer across
  opaque + transparent draws automatically (verified: `depthTest=true`/`depthWrite=false`
  on a `NodeMaterial` tests against previously-drawn opaque geometry; `viewportDepthTexture()`
  is also available if we ever need to *sample* scene depth). A side-channel WebGPU canvas
  composited over WebGL cannot do this — D1 forbids it.
- TSL gives us the storage-buffer binding model, `instanceIndex` indexing, compute dispatch
  (`Fn(...).compute(count, [wg])`, `renderer.compute([pass0, pass1, ...])` runs **in
  submitted order**), and `NodeMaterial.vertexNode/fragmentNode` for the raster pass — all
  confirmed present in r180.
- `wgslFn()` lets us paste the performance-critical WGSL verbatim (no node-graph overhead
  for tight loops), which is exactly where the radix sort and SH eval live.

**Why not pure TSL:** the radix sort's histogram/scan/scatter with atomics and shared
workgroup memory is far cleaner and more predictable in raw WGSL than as a node graph.

**Why not raw WebGPU device:** loses D1 depth fusion and forces us to reimplement
swapchain/render-target/state management THREE already does.

**API-surface caveats discovered (design around them):**
- `await renderer.init()` is **mandatory** before any sync render/compute. App init becomes
  async (§3).
- r180 has **no easy GPU-driven instanceCount** (`drawIndirect` instance count). v1 therefore
  uses a **fixed `instanceCount = maxResident` + sentinel-discard** in the vertex stage —
  identical in spirit to the existing WebGL path (`splatVertex.glsl:54` already discards the
  `0xffffffff` sentinel). GPU-driven count via `IndirectStorageBufferAttribute` is a Phase-4
  optimization, not a v1 dependency.
- THREE ships **no bundled compute example** in the npm package; PlayCanvas Engine v2.19
  GSplat WebGPU remains our architecture/radix reference (reimplemented, not copied).

---

## 2. Coexistence & the two-renderer model (D1, D2)

There is exactly **one renderer per session**, selected once at construction. No per-frame,
no per-mesh backend mixing.

```
                       capability + content probe (app init)
                                   │
        ┌──────────────────────────┴──────────────────────────┐
   WebGPU OK AND scene is WebGPU-eligible            otherwise (no navigator.gpu,
        │                                            init() fails, OR scene uses
        ▼                                            dyno/SplatEdit/Skinning/—paging
  THREE.WebGPURenderer                               not-yet-landed)
  + Spark WebGPU backend (TSL/WGSL)                          │
                                                             ▼
                                              THREE.WebGLRenderer
                                              + Spark EXISTING WebGL2 backend (untouched)
```

- The WebGL2 path's code (RawShaderMaterial generate passes, Readback, SplatWorker, the
  whole `driveSort` chain) is **not modified**. It remains the literal fallback. We do **not**
  run the GLSL RawShaderMaterial path on top of WebGPURenderer's internal WebGL fallback
  (D2) — if WebGPU is unavailable we hand the app a classic `WebGLRenderer`.
- "WebGPU-eligible scene" = residency is PackedSplats/ExtSplats/(PagedSplats once landed)
  **and** no runtime dyno object/world modifier, SplatEdit, or SplatSkinning (D6, §8).

---

## 3. Where the backend slots in (API & selection)

### 3.1 Backend abstraction inside Spark

Introduce an internal `SplatBackend` interface that both paths implement, behind the existing
`SparkRenderer` public surface:

```ts
interface SplatBackend {
  // residency upload
  uploadPacked(mesh): void; uploadExt(mesh): void; uploadSh(mesh): void;
  // per-frame
  cullProject(view): void;          // compute: cull + project + depth key (+SH eval)
  sort(): void;                     // compute: GPU radix sort -> sorted index buffer
  raster(view, target): void;       // instanced quad draw, depth-tested
  // capabilities
  readonly kind: 'webgl2' | 'webgpu';
}
```

- `webgl2` backend = a thin wrapper that calls the **existing, unchanged** code paths
  (`SplatAccumulator.generate`, `driveSort`, etc.). This is purely an internal adapter so
  `SparkRenderer` can hold a `SplatBackend` without touching WebGL2 logic.
- `webgpu` backend = new code, this design.

`SparkRenderer` picks the backend from `renderer.isWebGPURenderer` at construction.

### 3.2 Public API changes (kept minimal; documented per task)

Current required option is `renderer: THREE.WebGLRenderer`
([SparkRenderer.ts:33](../../src/SparkRenderer.ts)). Changes:

1. **Widen the type** to `renderer: THREE.WebGLRenderer | THREE.WebGPURenderer` (both extend
   the THREE renderer surface Spark uses). Detect via `.isWebGPURenderer`.
2. **Add a static async helper** so apps don't hand-roll detection + `init()`:
   ```ts
   // additive, does not change existing constructor usage
   const { renderer, backend } = await SparkRenderer.createRenderer({
     canvas, preferWebGPU: true, forceBackend?: 'webgl2' | 'webgpu',  // test override
     scene?  // optional: lets Spark probe content (dyno/paging) to pick the fallback
   });
   ```
   It returns a ready (`init()`-awaited) `WebGPURenderer` when eligible, else a classic
   `WebGLRenderer`. Existing `new SparkRenderer({ renderer })` keeps working unchanged for
   apps that construct their own `WebGLRenderer`.
3. **Force flag for testing (Phase 2 deliverable):** `forceBackend` above + a
   `?sparkBackend=webgpu|webgl2` URL param honored by examples.

**No removals.** `SplatMesh`, `PackedSplats`, `ExtSplats`, `SparkRenderer` options/methods
stay. The `SparkViewpoint` named in the task brief does not exist (Phase 0 §10); directives
did not override that, so we keep treating the `SparkRenderer`+`SplatAccumulator` per-view
state as the "viewpoint surface" and leave it API-stable.

### 3.3 Build

- Import `WebGPURenderer` and TSL from `three/webgpu` / `three/tsl`; both already shipped in
  the r180 dependency, externalized like `three` in the bundle (`vite.config.ts`).
- WGSL kernels authored as `.wgsl` files loaded as strings (mirror the existing
  vite-plugin-glsl setup with a `?raw`/wgsl loader) so they're editable and syntax-checked,
  then passed to `wgslFn()`. No new runtime dep.

---

## 4. GPU residency: storage-buffer layouts (D4, D5, D7)

All splat source data moves from integer `DataArrayTexture`s to **storage buffers**, keeping
the **bit layout byte-identical** (D7). Unpack uses WGSL builtins, no `f16` extension.

### 4.1 PackedSplats — `array<vec4<u32>>` (16 bytes/splat, unchanged bits)

One `vec4<u32>` per splat, identical to the current RGBA32UI texel
([splatDefines.glsl:174-239](../../src/shaders/splatDefines.glsl)):

| word | bits | field | WGSL unpack |
|------|------|-------|-------------|
| .x | 8/8/8/8 | R,G,B,A u8 | `& 0xff`, `>> 8/16/24` |
| .y | 16/16 | center.x,y f16 | `unpack2x16float(word_y)` |
| .z | 16/8/8 | center.z f16, quatX,quatY u8 | `unpack2x16float(.z & 0xffff).x`, shifts |
| .w | 8/8/8/8 | scaleX,Y,Z u8(log), quatZ u8 | shifts + `exp()` |

- `pack2x16float/unpack2x16float ≡ packHalf2x16/unpackHalf2x16` (D7).
- Octahedral-XY-88 + angle-8 quat decode (`decodeQuatOctXy88R8`) ports as a `wgslFn` helper.
- Scale log decode, RGB rescale: straight arithmetic ports.

### 4.2 ExtSplats — second buffer, float32 centers (D5)

`array<vec4<u32>>` packed buffer **plus** an extended buffer carrying float32 centers
(reinterpret via `bitcast<f32>` / `uintBitsToFloat`). The cull/project kernel reads centers
from the ext buffer when present, otherwise from the f16 fields in the packed buffer. This is
the only residency branch in the project kernel.

### 4.3 Spherical harmonics — SH1/SH2/SH3 storage buffers (D3, D7)

Port the existing SH textures to storage buffers, **same bit packing**
([PackedSplats.ts:1071-1270](../../src/PackedSplats.ts)):

| level | source format | storage buffer | coeffs |
|-------|---------------|----------------|--------|
| SH1 | RG32UI | `array<vec2<u32>>` | 9 (3×3), int7-ish |
| SH2 | RGBA32UI | `array<vec4<u32>>` | 15 (5×3), int8 |
| SH3 | RGBA32UI | `array<vec4<u32>>` | 21 (7×3), int6 |

SH is evaluated **in the cull/project compute pass** (not the fragment shader): the view
direction is constant across a splat's quad, so resolving RGB once per splat and storing it
in the projected buffer keeps the raster fragment lightweight (aligns with the Phase 4 goal
of "lightweight raster from pre-projected data") and is the correct place for D3.

### 4.4 Paging indirection — resident set (D4)

`PagedSplats`/`SplatPager` keep an LRU page pool + an indirection "indices" texture mapping
draw-index → resident page offset ([SplatPager.ts:645-673](../../src/SplatPager.ts)). Ported:

- **`residentPool: array<vec4<u32>>`** — the page-pool packed data (+ ext/SH pools as needed).
- **`activeIndices: array<u32>`** — indirection: `activeIndices[i]` = resident offset of the
  i-th active splat. Length = `activeCount` (CPU-known per frame from the pager).

The cull/project pass iterates `i in [0, activeCount)` and fetches
`residentPool[activeIndices[i]]`. **Everything downstream (depth keys, radix sort, sorted
index buffer, raster) is keyed by the dense active-set index `i`, never by the backing store
index** — see §6 for why this makes paging an extension, not a rewrite.

### 4.5 Per-frame compute buffers

| buffer | type | size | role |
|--------|------|------|------|
| `projected` | `array<ProjSplat>` | maxResident | screen center, 2D conic (3f), resolved RGB, opacity, depth — written by project pass |
| `depthKeys` | `array<u32>` | maxResident | quantized sort key (§5.1) |
| `indexA/indexB` | `array<u32>` | maxResident | ping-pong payload (splat/active index); B is the sorted output |
| `histograms` | `array<atomic<u32>>` | passes×256×groups | radix histograms |
| `drawArgs` | `IndirectStorageBufferAttribute` | 1 | (Phase 4) GPU-driven instanceCount |

`ProjSplat` is `std430`-aligned; vec3s padded to 16 bytes.

---

## 5. Compute passes: cull → project → radix sort → raster

Dispatched per frame via `renderer.compute([cull_project, sort_pass0..3])` (sequential,
in-order), then a normal `renderer.render(scene, camera)` consumes the sorted buffer in the
same frame. **No readback, no worker** — this is the entire point.

### 5.1 Pass A — cull + project + key (+SH) (Phase 4 fully; Phase 2 minimal)

One compute invocation per active splat (`@workgroup_size(64)`):
1. Resolve source index (`activeIndices[i]` for paged, else `i`); fetch + unpack splat.
2. Transform center to view/clip; **frustum cull** (and min/max pixel-radius like
   [SparkRenderer.ts](../../src/SparkRenderer.ts) limits). Culled ⇒ write sentinel key +
   mark inactive.
3. Compute the **depth metric identical to the WebGL path**
   ([output.ts:202-213](../../src/dyno/output.ts)): radial `length(center−viewCenter)` by
   default (`sortRadial`), or `dot(center,viewDir)+bias` non-radial.
4. **Quantize to a sortable u32** depth key (monotonic float→uint flip so integer-ascending
   order = the painter's far→near order; +inf/culled ⇒ `0xffffffff`).
5. Project covariance to the 2D conic; **evaluate SH≤3** for view dir; write `projected[i]`
   and `depthKeys[i] = key`, `indexA[i] = i`.

For Phase 2 the project step can be minimal (center + constant size) to prove plumbing;
full cull/project/conic/SH land in Phase 4 / are gated by D3.

### 5.2 Passes B0–B3 — GPU LSD radix sort (Phase 3)

Multi-pass **least-significant-digit radix sort**, 8 bits/pass × **4 passes** over the 32-bit
`depthKeys`, payload = `indexA`. Per pass (raw WGSL via `wgslFn`, reference = PlayCanvas
v2.19 GSplat sort, reimplemented):
1. **Histogram**: each workgroup counts its tile's 256 buckets into shared memory, then adds
   to a global `histograms` via `atomicAdd`.
2. **Exclusive prefix-sum / scan** across buckets (and across workgroup partials) → global
   offsets.
3. **Scatter**: stable-write `(key,payload)` from `indexA→indexB` (ping-pong) at scanned
   offsets.

After 4 passes the payload buffer holds active-set indices ordered far→near. Inactive
(`0xffffffff` key) sort to the end and are never indexed by the raster (instanceCount ≤
activeCount, and the sentinel-discard belt). Key buffers double-buffered; no host sync.

**Phase 2 stand-in:** a trivial single-workgroup compute sort (or even a temporary CPU sort
of `depthKeys`) to prove the buffer plumbing before B0–B3 land. This is explicitly a throwaway.

### 5.3 Pass C — lightweight raster (Phase 2 onward)

Instanced quad geometry (reuse the existing 4-vert quad concept). `NodeMaterial` with:
- **vertexNode**: `let s = sortedIndex[instanceIndex];` (sentinel ⇒ emit degenerate
  off-screen vertex, mirroring `splatVertex.glsl:54`); read `projected[s]`; build the screen
  quad from the 2D conic. `instanceIndex`/`vertexIndex` are confirmed TSL builtins; storage
  read via `storage(buf,'...').element(idx)`.
- **fragmentNode**: Gaussian falloff from conic × `projected[s].rgb` × opacity. Minimal —
  SH/cull/projection already done in Pass A.
- **blend/depth**: premultiplied-alpha transparent (`material.transparent=true`,
  premultiplied blend to match `premultipliedAlpha` default), `depthTest=true`,
  `depthWrite=false` (§7).
- **instanceCount**: `maxResident`/`activeCount` (CPU-known), sentinel-discard for the rest.
  GPU-driven count via `drawArgs` indirect = Phase 4 optimization.

---

## 6. Paging design proof (D4 requirement (a))

D4 demands the sort/index layout *already* accommodate paging. It does, because **the sort
operates on the dense active-set index space, not the backing store**:

- The pager produces `activeCount` + `activeIndices[]` each frame (it already maintains this
  indirection for the WebGL path).
- Pass A is the *only* place that touches the backing store; it dereferences `activeIndices`
  and writes a **dense** `projected[i]`/`depthKeys[i]` for `i ∈ [0,activeCount)`.
- The radix sort, the sorted index buffer, and the raster all consume the **dense `i`
  space**. They are byte-for-byte identical whether the source is PackedSplats (where
  `i == sourceIndex`) or PagedSplats (where `i` maps through `activeIndices`).

So landing paging later changes **only Pass A's fetch** (one `activeIndices` indirection),
never the sort or raster. Until paging's Pass-A branch + pool upload land, **paged scenes
route to the whole-session WebGL fallback** (D4 requirement (b)) — never partially.

---

## 7. Depth fusion with mesh geometry (D1)

- Splats render through `WebGPURenderer` into the **scene's shared depth attachment**.
  Opaque meshes draw first (THREE's opaque-before-transparent ordering); splats draw as
  transparent with `depthTest=true`, `depthWrite=false`, so they are **occluded by closer
  opaque geometry** while still alpha-blending among themselves in the GPU-sorted order.
- This is the capability D1 says we must not foreclose. It comes "for free" from rendering in
  the same scene/renderer; no side-channel, no manual depth copy.
- `viewportDepthTexture()` / `viewportLinearDepth()` are available if we later want *soft*
  depth fade or to read scene depth in Pass A — noted as future, not v1.

---

## 8. dyno / SplatEdit / SplatSkinning routing (D6)

- At session init (and on mesh add), Spark **detects** any runtime `objectModifier` /
  `worldModifier` (dyno), `SplatEdit(s)`, or `SplatSkinning` on scene meshes.
- If present ⇒ **whole-session WebGLRenderer** path (no mixed backend), with a clear
  `console.warn` naming the unsupported feature and that the session fell back for parity.
- The **WGSL dyno emitter is explicitly out of v1 scope** and documented as such. dyno today
  is a GLSL-only code generator with no backend abstraction (Phase 0 §4); a WGSL backend is a
  separate 4–6 week effort. We do **not** silently break it — we fall back.

---

## 9. Phase mapping (what builds when, always runnable)

| Phase | Deliverable | Sort | Cull/Project | SH | Paging |
|-------|-------------|------|--------------|----|--------|
| 2 | One SplatMesh through WebGPU; force flag; buffers + raster plumbing | temp CPU/trivial | minimal | off | off (fallback) |
| 3 | GPU LSD radix sort, same-frame; orbit no-pop verified | **B0–B3** | minimal | off | off |
| 4 | Full GPU cull+project, pre-projected raster; (opt) indirect count | B0–B3 | **full** | **on (D3)** | design-ready |
| 5 | Parity + benchmark + fallback validation; examples/docs | — | — | — | land or formal fallback |

Each phase keeps the project building and the WebGL2 path the default; the WebGPU path is
opt-in via the force flag until Phase 5.

---

## 10. Verification & risks

### Phase 5 parity baseline (D8)
Perceptual/threshold diff (not bit-exact — rasterization & precision differ). Baseline asset
set **must** include: a standard PackedSplats scene, an ExtSplats scene, **an SH3 reflective
scene**, and **a paged SOG scene** (highest divergence risk). Harness: render identical
camera path on both backends, SSIM/▵E threshold gate. Plus an FPS-vs-count bench at 1M/5M/10M
on both paths, and an automated "WebGPU unavailable ⇒ WebGL2" fallback test.

### Risk register
| risk | mitigation |
|------|------------|
| r180 no easy GPU-driven instanceCount | fixed-max + sentinel-discard (matches WebGL path); indirect count deferred to Phase 4 |
| Radix sort correctness/stability under atomics | reimplement from PlayCanvas v2.19 reference; Phase 3 gate = orbit no-pop + back-to-front diff vs WASM sort |
| WGSL bit-unpack mismatch vs GLSL | unit-compare unpacked attributes WebGPU vs WebGL for a fixed asset before raster work |
| SH3 numerical divergence | D8 reflective asset in parity gate; evaluate SH in compute (same math as `evaluatePackedSH`) |
| `std430` alignment of `ProjSplat` (vec3 padding) | explicit 16-byte alignment; validated by attribute-compare test |
| async `renderer.init()` reshapes app startup | `SparkRenderer.createRenderer()` encapsulates it; examples updated |
| Paging Pass-A not landing in time | proven index/sort layout (§6) + whole-session WebGL fallback (D4b) |

### What I'll verify before declaring each phase done
- Phase 2: a butterfly renders identically-ish under `?sparkBackend=webgpu`, buffers upload,
  compute+raster dispatch without validation errors.
- Phase 3: orbiting a 1M+ scene shows no popping; sorted order matches the WASM radix sort
  ordering within quantization.
- Phase 4: vertex/fragment are lightweight (no per-fragment SH/cull); cull reduces drawn
  instances; visual unchanged.
- Phase 5: parity gate passes on the D8 baseline; benchmark numbers recorded; fallback test
  green.

---

## 11. Resolved scope decisions (product owner, Phase 1)
1. **`SparkRenderer.createRenderer()` helper — APPROVED.** Add the additive async factory as
   the blessed way to detect capability + pick backend + `await init()`. Examples adopt it.
   Existing `new SparkRenderer({ renderer })` stays working unchanged.
2. **SH eval in the project compute pass — APPROVED.** SH order ≤3 resolved once per splat in
   cull/project, RGB written to `projected[i]`; fragment stays lightweight (§4.3, §5.1).
3. **Paging deferred with whole-session WebGL fallback — APPROVED.** v1 WebGPU covers
   **PackedSplats + ExtSplats only**. **Paged (PagedSplats/SplatPager) scenes route to the
   WebGL2 session** (whole-session, never partial — D4b). The dense active-set sort/index
   layout (§6) keeps the path paging-*ready*; paged Pass-A fetch + GPU page-pool upload land
   in Phase 4/5, and the D8 paged-SOG parity gate is exercised once that lands.

### Updated v1 eligibility (supersedes §2 "WebGPU-eligible scene")
A session uses the WebGPU backend iff: `navigator.gpu` present and `init()` succeeds **AND**
residency is **PackedSplats or ExtSplats** (not PagedSplats) **AND** no runtime dyno
object/world modifier, SplatEdit, or SplatSkinning (D6). Otherwise → classic `WebGLRenderer`
+ existing untouched WebGL2 backend.

**Awaiting approval of this design before writing Phase 2 implementation code.**
