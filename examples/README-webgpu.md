# WebGPU backend examples

Experimental WebGPU rendering backend for Spark. It runs culling, projection,
spherical-harmonics color, and an LSD radix depth sort in GPU compute, producing
and consuming the sorted order in the **same frame** — no GPU→CPU readback, no
worker round-trip, no sort lag/popping. The classic WebGL2 path is unchanged and
remains the default + fallback.

See `docs/webgpu-backend/` for the full design and phase notes.

## Examples

| Folder | What it shows |
|--------|---------------|
| `webgpu-hello/` | Minimal single-SplatMesh render through the WebGPU backend. |
| `webgpu-compare/` | `webgpu.html` + `webgl.html`: the same splat at an identical fixed pose, one per backend, for A/B parity comparison. |
| `webgpu-benchmark/` | `webgpu.html` + `webgl.html`: generate `?count=N` synthetic splats and report rolling FPS per backend. |
| `webgpu-editor/` | Interactive viewer on the compute backend: orbit, drag-and-drop / `?url=` loading, SH toggle, render knobs, FPS. PackedSplats only — the full WebGL2 tool is `editor/`. |

These pages map `three` to the appropriate THREE build per backend (the WebGPU
build does not ship the classic `WebGLRenderer`, and vice-versa), so each backend
has its own page rather than a single dual-canvas page.

## Using the WebGPU backend in your app

```js
import * as THREE from "three";              // -> three/build/three.webgpu.js
import { WebGPURenderer } from "three/webgpu";
import {
  SplatMesh, WebGPUSplatRenderer, isWebGPUAvailable, requestWebGPUStorageLimits,
} from "@sparkjsdev/spark";

if (!isWebGPUAvailable()) {
  // fall back to the classic WebGL path (THREE.WebGLRenderer + SparkRenderer)
}

// Raise the storage-buffer limits to the adapter maxima — the per-splat
// projection buffer (64 B/splat) exceeds WebGPU's 128 MiB default binding size
// past ~2M splats. (createRenderer below does this for you.)
const requiredLimits = (await requestWebGPUStorageLimits()) ?? {};
const renderer = new WebGPURenderer({ antialias: false, requiredLimits });
await renderer.init();

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 1000);

const mesh = new SplatMesh({ url: "scene.spz" });
await mesh.initialized;

const spark = new WebGPUSplatRenderer({ renderer });
await spark.init();
scene.add(spark);
spark.setSplatMesh(mesh);          // PackedSplats meshes only (see scope below)

renderer.setAnimationLoop(async () => {
  await spark.renderFrame(scene, camera);   // compute (project + sort) + render
});
```

`SparkRenderer.createRenderer({ preferWebGPU, forceBackend })` is a convenience
factory that capability-detects, constructs the matching renderer, awaits
`init()`, and falls back to WebGL2 automatically. Append `?sparkBackend=webgl2`
or `?sparkBackend=webgpu` to force a backend for testing.

## Scope (v1)

- **Supported**: `PackedSplats` meshes, anisotropic covariance projection,
  spherical harmonics ≤ 3, in-frame GPU radix sort, shared-depth compositing with
  ordinary THREE mesh geometry.
- **Routed to the WebGL2 session** (per the design directives): `PagedSplats`
  streaming, `ExtSplats`, runtime `dyno` modifiers, `SplatEdit`, `SplatSkinning`.
  Detect these and select the classic `WebGLRenderer` + `SparkRenderer` path for
  the whole session.
- Single SplatMesh per `WebGPUSplatRenderer` in this version.

## Requirements

A WebGPU-capable browser. To build/run locally you also need the Rust/WASM
package built (`npm run build:wasm`) and the dev bundle (`npm run build`), then
`npm run dev` and open an example.
