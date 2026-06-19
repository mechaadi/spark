import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
import type { PackedSplats } from "../PackedSplats";
import type { SplatMesh } from "../SplatMesh";
import { RawRadixSort } from "./rawRadixSort";
import {
  WGSL_COV2D,
  WGSL_DEPTH_KEY16,
  WGSL_EVAL_SH1,
  WGSL_EVAL_SH2,
  WGSL_EVAL_SH3,
  WGSL_UNPACK_CENTER,
  WGSL_UNPACK_QUAT,
  WGSL_UNPACK_RGBA,
  WGSL_UNPACK_SCALES,
} from "./wgsl";

// The per-frame pipeline: a TSL project pass (anisotropic covariance + SH color,
// writing a normalized 16-bit depth key into keyA and an identity payload into
// idxA), then a stable LSD radix depth sort run as raw WebGPU compute in a single
// command buffer (see ./rawRadixSort — far fewer queue submits than dispatching
// each sort stage through THREE), then an instanced raster on THREE.WebGPURenderer.

// `three/tsl` and `three/webgpu` are loaded via dynamic import (see `load()`),
// never as static imports, so the default WebGL build does not force WebGL-only
// consumers to provide them in their import map. We capture the module
// namespaces by type only here.
type TSL = typeof import("three/tsl");
type WGPU = typeof import("three/webgpu");
type StorageBufferAttribute = InstanceType<WGPU["StorageBufferAttribute"]>;

export interface WebGPUSplatRendererOptions {
  /** A pre-initialized THREE.WebGPURenderer (await `renderer.init()` first). */
  renderer: WebGPURenderer;
  /**
   * Evaluate spherical-harmonics (view-dependent) color when the mesh has it.
   * Set false to render base color only (useful for A/B comparison). Can also
   * be toggled later via the `enableSh` property before `setSplatMesh`.
   * @default true
   */
  enableSh?: boolean;
}

/**
 * WebGPU splat backend (Phases 2-4).
 *
 * Renders a SplatMesh entirely on the GPU through THREE.WebGPURenderer:
 * packed-splat storage-buffer upload, a per-splat compute project pass (full
 * anisotropic 2D-covariance projection + SH<=3 view-dependent color), an
 * **in-frame stable GPU radix sort** by depth, and an instanced raster draw into
 * the scene's shared depth buffer. The sort is produced and consumed in the same
 * frame — no GPU->CPU readback, no worker round-trip, no sort lag.
 *
 * Scope/caveats (documented, replaced in later phases):
 * - Sort is a stable tiled LSD radix over a 16-bit depth key (2 x 8-bit passes,
 *   parallel prefix sum). Stability matters: equal-depth splats must keep a
 *   deterministic draw order or alpha blending flickers. 16-bit depth is enough
 *   for the tested scenes; a wider key or workgroup-shared layout is a future
 *   option.
 * - Only `PackedSplats` residency is supported here; ExtSplats/Paged are routed
 *   to the WebGL session per the design (directives D4/D5/D6).
 *
 * Usage:
 * ```
 * const spark = new WebGPUSplatRenderer({ renderer });
 * await spark.init();
 * scene.add(spark);
 * spark.setSplatMesh(mesh);          // after await mesh.initialized
 * // render loop:
 * await spark.renderFrame(scene, camera);
 * ```
 */
export class WebGPUSplatRenderer extends THREE.Group {
  readonly isWebGPUSplatRenderer = true;
  readonly renderer: WebGPURenderer;

  private tsl: TSL | null = null;
  private wgpu: WGPU | null = null;

  private numSplats = 0;
  private mesh: THREE.Mesh | null = null;

  // GPU storage buffers.
  private packedAttr: StorageBufferAttribute | null = null;
  private projAttr: StorageBufferAttribute | null = null;
  // Depth key + identity payload (project output / raw-sort I/O); the sorted
  // order lands back in idxA, which the raster reads. The sort's ping-pong +
  // histogram scratch lives in RawRadixSort, not here.
  private keyA: StorageBufferAttribute | null = null;
  private idxA: StorageBufferAttribute | null = null;
  // Spherical-harmonics coefficient buffers (present iff the mesh has SH).
  private sh1Attr: StorageBufferAttribute | null = null;
  private sh2Attr: StorageBufferAttribute | null = null;
  private sh3Attr: StorageBufferAttribute | null = null;
  private numSh = 0;
  private shMax = new THREE.Vector3(1, 1, 1);

  // Per-frame TSL compute (just the project pass now; the sort is raw WebGPU).
  private computeNodes: unknown[] = [];
  // Single-submit raw-WebGPU radix sort, sharing THREE's key/idx GPU buffers.
  private rawSort: RawRadixSort | null = null;
  private rawSortBound = false;

  // Uniform nodes (their `.value` is updated per frame).
  private uModelView: { value: THREE.Matrix4 } | null = null;
  private uModelView3: { value: THREE.Matrix3 } | null = null;
  private uProjection: { value: THREE.Matrix4 } | null = null;
  private uProjScale: { value: THREE.Vector2 } | null = null;
  private uRenderSize: { value: THREE.Vector2 } | null = null;
  private uViewOrigin: { value: THREE.Vector3 } | null = null;
  private uShMax: { value: THREE.Vector3 } | null = null;
  private uMaxStdDev: { value: number } | null = null;
  private uMinAlpha: { value: number } | null = null;
  private uMaxPixelRadius: { value: number } | null = null;
  private uMaxSplatScale: { value: number } | null = null;
  // Radial view-distance range of the model, recomputed per frame from the
  // object-space bounding sphere, so the 16-bit depth key resolves finely.
  private uDepthMin: { value: number } | null = null;
  private uDepthMax: { value: number } | null = null;
  // Object-space bounding sphere of the splat centers (set in setSplatMesh).
  private readonly boundCenter = new THREE.Vector3();
  private boundRadius = 0;

  // Per-frame reused temporaries (avoid allocation in the loop).
  private readonly tmpModelView = new THREE.Matrix4();
  private readonly tmpModelView3 = new THREE.Matrix3();
  private readonly tmpViewInverse = new THREE.Matrix4();
  private readonly tmpObjInverse = new THREE.Matrix4();
  private readonly tmpCamPos = new THREE.Vector3();
  private readonly tmpSphere = new THREE.Vector3();
  private readonly tmpSize = new THREE.Vector2();

  // Recompute gate: project + sort only re-run when something that affects the
  // projected/sorted result changed (camera, viewport, or a render param). A
  // static view (the common case) reuses last frame's projStore + sorted idxA
  // and just re-issues the raster — this is the single biggest per-frame saving,
  // since project+sort over all N splats is the dominant cost (PlayCanvas does
  // the same via a camera-movement epsilon). `recomputePending` forces the
  // first frame after a (re)build; the prev* snapshot detects view/param change.
  private recomputePending = true;
  private readonly prevModelView = new THREE.Matrix4();
  private readonly prevProjection = new THREE.Matrix4();
  private readonly prevSize = new THREE.Vector2(-1, -1);
  private prevStd = Number.NaN;
  private prevMinAlpha = Number.NaN;
  private prevMaxPixelRadius = Number.NaN;
  private prevMaxSplatScale = Number.NaN;

  maxStdDev = Math.sqrt(8.0);
  /**
   * Cull splats whose largest object-space scale exceeds this, to trim the big
   * diffuse "floater" blobs common in real captures. `Infinity` (default) keeps
   * everything; lower it toward the model size to remove background haze.
   */
  maxSplatScale = Number.POSITIVE_INFINITY;
  minAlpha = 0.5 * (1.0 / 255.0);
  maxPixelRadius = 512.0;
  /** Evaluate SH color (read on the next `setSplatMesh`). */
  enableSh = true;
  /**
   * Debug: skip the depth sort and render in the project pass's identity order.
   * The result is wrong (no painter's order), but A/B-ing fps with this on/off
   * isolates how much of the frame the sort actually costs vs project + raster.
   */
  skipSort = false;

  constructor(options: WebGPUSplatRendererOptions) {
    super();
    if (!options?.renderer) {
      throw new Error("WebGPUSplatRenderer requires a THREE.WebGPURenderer");
    }
    this.renderer = options.renderer;
    this.enableSh = options.enableSh ?? true;
    this.frustumCulled = false;
  }

  /** Dynamically load the WebGPU/TSL modules. Idempotent. */
  async init(): Promise<this> {
    if (!this.tsl || !this.wgpu) {
      const [tsl, wgpu] = await Promise.all([
        import("three/tsl"),
        import("three/webgpu"),
      ]);
      this.tsl = tsl;
      this.wgpu = wgpu;
    }
    return this;
  }

  /**
   * Upload a SplatMesh's packed data to the GPU and build the compute + raster
   * pipeline. Call after `await mesh.initialized`. Phase 2 supports a single
   * mesh; calling again replaces the previous one.
   */
  setSplatMesh(mesh: SplatMesh): void {
    if (!this.tsl || !this.wgpu) {
      throw new Error(
        "Call await WebGPUSplatRenderer.init() before setSplatMesh()",
      );
    }
    const packed: PackedSplats | undefined = mesh.packedSplats;
    if (!packed || !packed.packedArray) {
      throw new Error(
        "WebGPUSplatRenderer supports PackedSplats only; this mesh has no " +
          "packed data ready. ExtSplats/Paged scenes use the WebGL2 session.",
      );
    }

    this.disposeMesh();

    const numSplats = packed.numSplats;
    this.numSplats = numSplats;

    // Object-space bounding sphere of the splat centers: drives the per-frame
    // depth-key normalization (WGSL_DEPTH_KEY16) so all 16 key bits resolve
    // depth within the model rather than wasting range on the float exponent.
    const box = mesh.getBoundingBox(true);
    box.getCenter(this.boundCenter);
    this.boundRadius = box.getSize(this.tmpSphere).length() * 0.5;

    const packedArray = packed.packedArray.subarray(0, numSplats * 4);

    const { StorageBufferAttribute, MeshBasicNodeMaterial } = this.wgpu;

    // Source packed splats: one vec4<u32> per splat, bit-identical to WebGL.
    this.packedAttr = new StorageBufferAttribute(
      packedArray.slice() as unknown as Uint32Array,
      4,
    );
    // Projected per-splat data: 4 x vec4<f32> per splat (filled by the compute
    // pass): [ndc.xy, clipZ, clipW] / [axis1.xy, axis2.xy] / rgba / [stdDev,active,_,_].
    this.projAttr = new StorageBufferAttribute(
      new Float32Array(numSplats * 16),
      4,
    );
    // Depth key + identity payload written by the project pass; the raw-WebGPU
    // radix sort (which owns its own ping-pong + histogram scratch) permutes them
    // in place, landing the sorted order back in idxA for the raster.
    this.keyA = new StorageBufferAttribute(new Uint32Array(numSplats), 1);
    this.idxA = new StorageBufferAttribute(new Uint32Array(numSplats), 1);

    // Spherical harmonics: upload whichever bands the mesh carries (D3).
    // sh1 = 2 u32/splat (RG32UI), sh2/sh3 = 4 u32/splat (RGBA32UI).
    const extra = packed.extra as {
      sh1?: Uint32Array;
      sh2?: Uint32Array;
      sh3?: Uint32Array;
    };
    const enc = packed.splatEncoding ?? {};
    this.shMax.set(enc.sh1Max ?? 1, enc.sh2Max ?? 1, enc.sh3Max ?? 1);
    const shBuf = (src: Uint32Array, itemSize: number) =>
      new StorageBufferAttribute(
        src.subarray(0, numSplats * itemSize).slice() as unknown as Uint32Array,
        itemSize,
      );
    this.numSh = 0;
    if (this.enableSh && extra.sh1 && extra.sh1.length >= numSplats * 2) {
      this.sh1Attr = shBuf(extra.sh1, 2);
      this.numSh = 1;
      if (extra.sh2 && extra.sh2.length >= numSplats * 4) {
        this.sh2Attr = shBuf(extra.sh2, 4);
        this.numSh = 2;
        if (extra.sh3 && extra.sh3.length >= numSplats * 4) {
          this.sh3Attr = shBuf(extra.sh3, 4);
          this.numSh = 3;
        }
      }
    }

    this.buildPipeline();
    const material = this.buildMaterial(MeshBasicNodeMaterial);

    const geometry = makeQuadGeometry(numSplats);
    const drawMesh = new THREE.Mesh(geometry, material);
    drawMesh.frustumCulled = false;
    this.mesh = drawMesh;
    this.add(drawMesh);
  }

  private buildPipeline(): void {
    const tsl = this.tsl as TSL;
    const {
      Fn,
      If,
      Return,
      instanceIndex,
      vec2,
      vec3,
      vec4,
      float,
      uint,
      uniform,
      storage,
      wgslFn,
      normalize,
      select,
      min,
      max,
      sqrt,
      log,
    } = tsl;
    const N = this.numSplats;

    const unpackCenter = wgslFn(WGSL_UNPACK_CENTER);
    const unpackRgba = wgslFn(WGSL_UNPACK_RGBA);
    const unpackScales = wgslFn(WGSL_UNPACK_SCALES);
    const unpackQuat = wgslFn(WGSL_UNPACK_QUAT);
    const cov2DOf = wgslFn(WGSL_COV2D);
    const depthKey16 = wgslFn(WGSL_DEPTH_KEY16);

    const sba = (a: StorageBufferAttribute | null) =>
      a as StorageBufferAttribute;
    const packedStore = storage(sba(this.packedAttr), "uvec4", N).toReadOnly();
    const projStore = storage(sba(this.projAttr), "vec4", N * 4);
    // project writes the depth key + identity payload; the raw sort permutes them.
    const keyAStore = storage(sba(this.keyA), "uint", N);
    const idxAStore = storage(sba(this.idxA), "uint", N);

    const uModelView = uniform(new THREE.Matrix4());
    const uModelView3 = uniform(new THREE.Matrix3());
    const uProjection = uniform(new THREE.Matrix4());
    const uProjScale = uniform(new THREE.Vector2(1, 1));
    const uRenderSize = uniform(new THREE.Vector2(1, 1));
    const uViewOrigin = uniform(new THREE.Vector3());
    const uShMax = uniform(new THREE.Vector3(1, 1, 1));
    const uMaxStdDev = uniform(this.maxStdDev);
    const uMinAlpha = uniform(this.minAlpha);
    const uMaxPixelRadius = uniform(this.maxPixelRadius);
    const uMaxSplatScale = uniform(1e30);
    const uDepthMin = uniform(0.0);
    const uDepthMax = uniform(1.0);
    this.uDepthMin = uDepthMin as unknown as { value: number };
    this.uDepthMax = uDepthMax as unknown as { value: number };
    this.uModelView = uModelView as unknown as { value: THREE.Matrix4 };
    this.uModelView3 = uModelView3 as unknown as { value: THREE.Matrix3 };
    this.uProjection = uProjection as unknown as { value: THREE.Matrix4 };
    this.uProjScale = uProjScale as unknown as { value: THREE.Vector2 };
    this.uRenderSize = uRenderSize as unknown as { value: THREE.Vector2 };
    this.uViewOrigin = uViewOrigin as unknown as { value: THREE.Vector3 };
    this.uShMax = uShMax as unknown as { value: THREE.Vector3 };
    this.uMaxStdDev = uMaxStdDev as unknown as { value: number };
    this.uMinAlpha = uMinAlpha as unknown as { value: number };
    this.uMaxPixelRadius = uMaxPixelRadius as unknown as { value: number };
    this.uMaxSplatScale = uMaxSplatScale as unknown as { value: number };

    // SH evaluation kernels + storage stores (only the bands the mesh has).
    const sh1Eval = this.numSh >= 1 ? wgslFn(WGSL_EVAL_SH1) : null;
    const sh2Eval = this.numSh >= 2 ? wgslFn(WGSL_EVAL_SH2) : null;
    const sh3Eval = this.numSh >= 3 ? wgslFn(WGSL_EVAL_SH3) : null;
    const sh1S = this.sh1Attr
      ? storage(this.sh1Attr, "uvec2", N).toReadOnly()
      : null;
    const sh2S = this.sh2Attr
      ? storage(this.sh2Attr, "uvec4", N).toReadOnly()
      : null;
    const sh3S = this.sh3Attr
      ? storage(this.sh3Attr, "uvec4", N).toReadOnly()
      : null;

    // --- Pass A: project (full anisotropic covariance) + depth key + payload ---
    // Port of the standard-splat path in src/shaders/splatVertex.glsl: build the
    // view-space 3D covariance, project it through the perspective Jacobian to a
    // 2D covariance, eigen-decompose it into screen-space ellipse axes.
    const project = Fn(() => {
      const i = instanceIndex;
      // THREE dispatches ceil(count/workgroupSize) full workgroups with no range
      // guard, so when N isn't a multiple of the workgroup size the extra padding
      // invocations would write out of bounds (clamped/racing) -> garbage that
      // varies frame to frame on real hardware. Bail out of the over-run lanes.
      If(i.greaterThanEqual(uint(N)), () => {
        Return();
      });
      const pk = packedStore.element(i);
      const center = unpackCenter(pk);
      const rgba = unpackRgba(pk);
      const scales = unpackScales(pk);
      const quat = unpackQuat(pk);

      const base = i.mul(4);
      const zero = vec4(0.0, 0.0, 0.0, 0.0);
      projStore.element(base).assign(zero);
      projStore.element(base.add(1)).assign(zero);
      projStore.element(base.add(2)).assign(zero);
      projStore.element(base.add(3)).assign(zero);

      // Identity payload; inactive splats get the max 16-bit key so they sort
      // last (and are discarded by the raster via projected.active == 0).
      idxAStore.element(i).assign(i);
      keyAStore.element(i).assign(uint(0xffff));

      const viewC = uModelView.mul(vec4(center, 1.0)).xyz.toVar();
      const a = rgba.w.mul(2.0).toVar();
      const adj = uMaxStdDev.toVar();
      If(a.greaterThan(1.0), () => {
        a.assign(a.mul(4.0).sub(3.0).min(5.0));
        adj.assign(uMaxStdDev.add(a.sub(1.0).mul(0.7)));
      });

      const maxScale = max(scales.x, max(scales.y, scales.z));
      const active = viewC.z
        .lessThan(0.0)
        .and(maxScale.greaterThan(0.0))
        .and(maxScale.lessThanEqual(uMaxSplatScale))
        .and(a.greaterThanEqual(uMinAlpha));
      If(active, () => {
        const clip = uProjection.mul(vec4(viewC, 1.0)).toVar();
        If(clip.w.greaterThan(0.0), () => {
          // Projected 2D covariance (a, d, b) for [[a, b], [b, d]].
          const focal = uRenderSize.mul(0.5).mul(uProjScale); // pixels
          const cov = cov2DOf(uModelView3, scales, quat, focal, viewC);
          const covA0 = cov.x;
          const covD0 = cov.y;
          const covB = cov.z;

          // 0.5px Gaussian anti-alias blur + intensity compensation.
          const blur = float(0.3);
          const detOrig = covA0.mul(covD0).sub(covB.mul(covB));
          const covA = covA0.add(blur);
          const covD = covD0.add(blur);
          const det = covA.mul(covD).sub(covB.mul(covB));
          const blurAdjust = sqrt(max(0.0, detOrig.div(det)));
          const alpha = a.mul(blurAdjust);

          // Eigen-decomposition of the 2D covariance.
          const eigenAvg = covA.add(covD).mul(0.5);
          const eigenDelta = sqrt(max(0.0, eigenAvg.mul(eigenAvg).sub(det)));
          const eigen1 = eigenAvg.add(eigenDelta);
          const eigen2 = eigenAvg.sub(eigenDelta);
          const eigenVec1 = select(
            covB.abs().greaterThan(0.001),
            normalize(vec2(covB, eigen1.sub(covA))),
            select(covA.greaterThanEqual(covD), vec2(1.0, 0.0), vec2(0.0, 1.0)),
          );
          const eigenVec2 = vec2(eigenVec1.y, eigenVec1.x.negate());
          // Opacity-adaptive quad shrink (PlayCanvas clipCorner). The a<=1
          // Gaussian profile decays as alpha*exp(-0.5*r^2); it crosses the
          // alpha-discard floor (uMinAlpha) at r = sqrt(2*ln(alpha/uMinAlpha))
          // sigma — usually well inside the fixed `adj`-sigma quad. Shrink the
          // quad (and the fragment cutoff `cutoff`) to that radius so we stop
          // rasterizing the outer ring that the fragment would discard anyway.
          // +0.3 sigma margin keeps the visible edge intact; high-opacity (a>1)
          // splats use the bounded profile, so leave their quad at full `adj`.
          const cutoff = select(
            alpha.greaterThan(1.0),
            adj,
            min(adj, sqrt(max(0.0, log(alpha.div(uMinAlpha)).mul(2.0))).add(0.3)),
          ).toVar();
          const clipRatio = cutoff.div(adj);
          const scale1 = min(uMaxPixelRadius, adj.mul(sqrt(eigen1))).mul(clipRatio);
          const scale2 = min(uMaxPixelRadius, adj.mul(sqrt(eigen2))).mul(clipRatio);

          // NDC offsets per unit quad corner along each ellipse axis.
          const twoOverRS = vec2(2.0, 2.0).div(uRenderSize);
          const axis1 = eigenVec1.mul(scale1).mul(twoOverRS);
          const axis2 = eigenVec2.mul(scale2).mul(twoOverRS);

          // View-dependent color: base (SH DC, baked into rgba) + SH bands 1-3,
          // evaluated in object space (dir from camera to splat), per D3.
          const rgb = vec3(rgba.x, rgba.y, rgba.z).toVar();
          if (sh1Eval && sh1S) {
            const dir = normalize(center.sub(uViewOrigin)).toVar();
            rgb.addAssign(sh1Eval(sh1S.element(i), dir, uShMax.x));
            if (sh2Eval && sh2S) {
              rgb.addAssign(sh2Eval(sh2S.element(i), dir, uShMax.y));
            }
            if (sh3Eval && sh3S) {
              rgb.addAssign(sh3Eval(sh3S.element(i), dir, uShMax.z));
            }
            // The WebGL path bakes base+SH into an 8-bit color texture, which
            // clamps it to [0,1] (packSplatEncoding). Match that here, otherwise
            // extreme per-splat SH values blow past 1.0 and show as bright
            // speckle that WebGL never displays.
            rgb.assign(rgb.clamp(0.0, 1.0));
          }

          const ndc = clip.xyz.div(clip.w);
          projStore.element(base).assign(vec4(ndc.x, ndc.y, clip.z, clip.w));
          projStore
            .element(base.add(1))
            .assign(vec4(axis1.x, axis1.y, axis2.x, axis2.y));
          // sRGB->linear EOTF, hoisted out of the fragment shader. Overdraw
          // means each splat covers many fragments, so doing this once per
          // splat here is far cheaper than 3 pow ops per covered pixel. The
          // fragment then outputs the stored color straight; WebGPURenderer's
          // output linear->sRGB OETF cancels it, so on-screen color is
          // unchanged. (max(0) guards negative SH lobes, matching the WebGL path.)
          const lin = rgb.max(0.0);
          const rgbLin = select(
            lin.lessThanEqual(0.04045),
            lin.div(12.92),
            lin.add(0.055).div(1.055).pow(2.4),
          );
          projStore
            .element(base.add(2))
            .assign(vec4(rgbLin.x, rgbLin.y, rgbLin.z, alpha));
          projStore.element(base.add(3)).assign(vec4(cutoff, 1.0, 0.0, 0.0));
          // 16-bit far->near key, normalized to the model's view-depth range.
          keyAStore
            .element(i)
            .assign(depthKey16(viewC.length(), uDepthMin, uDepthMax));
        });
      });
    });

    // The depth sort runs as raw WebGPU compute in ONE command buffer (see
    // RawRadixSort) instead of ~10 separate THREE compute() submits. The TSL
    // project pass above writes the 16-bit depth key into keyA and the identity
    // payload into idxA; the raw sort permutes them in place (even pass count
    // lands the sorted order back in idxA), which the raster reads. The key/idx
    // GPU buffers are THREE-owned and shared with the raw sort by handle.
    this.computeNodes = [project().compute(N)];
    const device = (
      this.renderer as unknown as { backend?: { device?: unknown } }
    ).backend?.device;
    if (device) {
      this.rawSort = new RawRadixSort(device);
      this.rawSort.configure(N);
      this.rawSortBound = false;
    }
    // Force the first frame after a (re)build to run project + sort.
    this.recomputePending = true;
  }

  private buildMaterial(
    MeshBasicNodeMaterial: WGPU["MeshBasicNodeMaterial"],
  ): InstanceType<WGPU["MeshBasicNodeMaterial"]> {
    const tsl = this.tsl as TSL;
    const {
      Fn,
      If,
      Discard,
      instanceIndex,
      positionGeometry,
      vec4,
      float,
      varyingProperty,
      storage,
      mix,
      select,
    } = tsl;

    const projStore = storage(
      this.projAttr as StorageBufferAttribute,
      "vec4",
      this.numSplats * 4,
    ).toReadOnly();
    // Sorted draw order (far -> near) produced by the radix sort, in idxA.
    const orderStore = storage(
      this.idxA as StorageBufferAttribute,
      "uint",
      this.numSplats,
    ).toReadOnly();

    const vColor = varyingProperty("vec4", "vSplatColor");
    const vUv = varyingProperty("vec2", "vSplatUv");
    const vStd = varyingProperty("float", "vSplatStd");
    const uMinAlpha = this.uMinAlpha as { value: number };

    const material = new MeshBasicNodeMaterial();
    material.transparent = true;
    material.depthTest = true;
    material.depthWrite = false;
    material.premultipliedAlpha = true;
    material.side = THREE.DoubleSide;

    material.vertexNode = Fn(() => {
      const s = orderStore.element(instanceIndex);
      const b = s.mul(4);
      const p0 = projStore.element(b);
      const p1 = projStore.element(b.add(1));
      const p2 = projStore.element(b.add(2));
      const meta = projStore.element(b.add(3));

      const corner = positionGeometry.xy;
      vColor.assign(p2);
      vUv.assign(corner.mul(meta.x));
      vStd.assign(meta.x); // adjustedStdDev

      const off = corner.x.mul(p1.xy).add(corner.y.mul(p1.zw));
      const clipW = p0.w;
      const clipZ = p0.z;
      const pos = vec4(p0.xy.add(off).mul(clipW), clipZ, clipW);
      const offscreen = vec4(0.0, 0.0, 2.0, 1.0);
      return mix(offscreen, pos, meta.y);
    })();

    material.fragmentNode = Fn(() => {
      // Port of splatFragment.glsl. Circular cutoff, then the two falloff
      // profiles: a<=1 is a plain Gaussian; a>1 (high-opacity / LOD-inflated
      // splats) uses the bounded profile so alpha stays in [0,1] — without this
      // high-opacity splats render as hard, over-bright, non-blending dots.
      const z2 = vUv.dot(vUv);
      If(z2.greaterThan(vStd.mul(vStd)), () => {
        Discard();
      });
      const a = vColor.w;
      const fall = z2.mul(-0.5).exp();
      const aLow = a.mul(fall);
      const expo = a.mul(a).sub(1.0).div(Math.E).exp();
      const aHigh = float(1.0).sub(float(1.0).sub(fall).pow(expo));
      const alpha = select(a.greaterThan(1.0), aHigh, aLow).toVar();
      If(alpha.lessThan(uMinAlpha.value), () => {
        Discard();
      });
      // Color is already linear: the sRGB->linear EOTF now runs once per splat
      // in the project pass (projStore), not per fragment, so we just output it
      // straight here. WebGPURenderer's output linear->sRGB OETF cancels it.
      // Output STRAIGHT (non-premultiplied) color: NodeMaterial premultiplies it
      // itself when `premultipliedAlpha === true`. Outputting premultiplied here
      // would double-premultiply (rgb*alpha^2), darkening semi-transparent splats
      // to near nothing and leaving only opaque splats (a sparkle artifact).
      return vec4(vColor.xyz.max(0.0), alpha);
    })();

    return material;
  }

  /**
   * Update uniforms and dispatch the per-frame compute pipeline (project + GPU
   * radix sort). Call before rendering the scene. No GPU->CPU readback.
   */
  prepareFrame(camera: THREE.Camera): void {
    if (!this.mesh || !this.numSplats) {
      return;
    }
    this.updateMatrixWorld(true);
    camera.updateMatrixWorld(true);

    this.tmpViewInverse.copy(camera.matrixWorld).invert();
    this.tmpModelView.multiplyMatrices(
      this.tmpViewInverse,
      this.mesh.matrixWorld,
    );
    this.tmpModelView3.setFromMatrix4(this.tmpModelView);
    this.renderer.getDrawingBufferSize(this.tmpSize);
    // Camera position in the mesh's object space, for object-space SH viewDir.
    this.tmpObjInverse.copy(this.mesh.matrixWorld).invert();
    camera.getWorldPosition(this.tmpCamPos).applyMatrix4(this.tmpObjInverse);

    const proj = (camera as THREE.PerspectiveCamera).projectionMatrix;
    (this.uModelView as { value: THREE.Matrix4 }).value.copy(this.tmpModelView);
    (this.uModelView3 as { value: THREE.Matrix3 }).value.copy(
      this.tmpModelView3,
    );
    (this.uProjection as { value: THREE.Matrix4 }).value.copy(proj);
    (this.uProjScale as { value: THREE.Vector2 }).value.set(
      proj.elements[0],
      proj.elements[5],
    );
    (this.uRenderSize as { value: THREE.Vector2 }).value.copy(this.tmpSize);
    (this.uViewOrigin as { value: THREE.Vector3 }).value.copy(this.tmpCamPos);
    (this.uShMax as { value: THREE.Vector3 }).value.copy(this.shMax);
    (this.uMaxStdDev as { value: number }).value = this.maxStdDev;
    (this.uMinAlpha as { value: number }).value = this.minAlpha;
    (this.uMaxPixelRadius as { value: number }).value = this.maxPixelRadius;
    (this.uMaxSplatScale as { value: number }).value = Number.isFinite(
      this.maxSplatScale,
    )
      ? this.maxSplatScale
      : 1e30;

    // View-space radial-distance range of the model's bounding sphere, used to
    // normalize the 16-bit depth key. The view-space radius scales by the
    // model-view's largest axis scale (the view part is rigid).
    this.tmpSphere.copy(this.boundCenter).applyMatrix4(this.tmpModelView);
    const centerDist = this.tmpSphere.length();
    const e = this.tmpModelView3.elements;
    const scale = Math.max(
      Math.hypot(e[0], e[1], e[2]),
      Math.hypot(e[3], e[4], e[5]),
      Math.hypot(e[6], e[7], e[8]),
    );
    const r = this.boundRadius * scale + 1e-4;
    (this.uDepthMin as { value: number }).value = Math.max(0, centerDist - r);
    (this.uDepthMax as { value: number }).value = centerDist + r;

    // --- Recompute gate -----------------------------------------------------
    // Re-run project + sort only when the projected/sorted result would differ:
    // the camera/model transform, the viewport, or a render param changed.
    // Otherwise last frame's projStore + sorted idxA are still valid, so we skip
    // both passes (the dominant per-frame cost) and let renderAsync re-issue the
    // raster against the persisted GPU buffers. Damping is off, so a still
    // camera produces bit-identical matrices frame to frame.
    const viewChanged =
      this.recomputePending ||
      // Keep recomputing until the raw sort has bound its buffers (the bind
      // retries inside the sort block below, once the project pass has created
      // them), so a view that goes static mid-bind still finishes binding.
      (this.rawSort != null && !this.rawSortBound) ||
      !this.prevModelView.equals(this.tmpModelView) ||
      !this.prevProjection.equals(proj) ||
      !this.prevSize.equals(this.tmpSize) ||
      this.prevStd !== this.maxStdDev ||
      this.prevMinAlpha !== this.minAlpha ||
      this.prevMaxPixelRadius !== this.maxPixelRadius ||
      this.prevMaxSplatScale !== this.maxSplatScale;
    if (!viewChanged) {
      return;
    }
    this.prevModelView.copy(this.tmpModelView);
    this.prevProjection.copy(proj);
    this.prevSize.copy(this.tmpSize);
    this.prevStd = this.maxStdDev;
    this.prevMinAlpha = this.minAlpha;
    this.prevMaxPixelRadius = this.maxPixelRadius;
    this.prevMaxSplatScale = this.maxSplatScale;
    this.recomputePending = false;

    // 1) Project every splat (TSL compute): writes the depth key into keyA and
    //    the identity payload into idxA (one submit).
    for (const node of this.computeNodes) {
      this.renderer.compute(node as never);
    }
    // 2) Depth sort (raw WebGPU): one command buffer, in place on keyA/idxA. The
    //    project compute above creates keyA/idxA's GPU buffers, so bind their
    //    handles on the first frame. Queue ordering keeps project -> sort ->
    //    raster correct (separate submits are sequenced and memory-coherent).
    if (this.rawSort) {
      if (!this.rawSortBound) {
        const backend = (
          this.renderer as unknown as {
            backend?: { get(a: unknown): { buffer?: unknown } | undefined };
          }
        ).backend;
        const keyBuf = backend?.get(this.keyA)?.buffer;
        const idxBuf = backend?.get(this.idxA)?.buffer;
        if (keyBuf && idxBuf) {
          this.rawSort.bindIO(keyBuf as never, idxBuf as never);
          this.rawSortBound = true;
        }
      }
      if (this.rawSortBound && !this.skipSort) {
        this.rawSort.run();
      }
    }
  }

  /** Convenience: prepareFrame + renderer.renderAsync. */
  async renderFrame(scene: THREE.Scene, camera: THREE.Camera): Promise<void> {
    this.prepareFrame(camera);
    await this.renderer.renderAsync(scene, camera);
  }

  private disposeMesh(): void {
    if (this.mesh) {
      this.remove(this.mesh);
      this.mesh.geometry.dispose();
      (this.mesh.material as THREE.Material).dispose();
      this.mesh = null;
    }
    this.computeNodes = [];
    this.rawSort?.dispose();
    this.rawSort = null;
    this.rawSortBound = false;
  }

  dispose(): void {
    this.disposeMesh();
    this.rawSort?.dispose();
    this.rawSort = null;
    this.rawSortBound = false;
    this.packedAttr = null;
    this.projAttr = null;
    this.keyA = null;
    this.idxA = null;
    this.sh1Attr = null;
    this.sh2Attr = null;
    this.sh3Attr = null;
    this.numSh = 0;
  }
}

function makeQuadGeometry(
  instanceCount: number,
): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();
  // Quad corners in [-1, 1]; the splat ellipse half-axes scale these.
  const vertices = new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]);
  geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
  geometry.setIndex(
    new THREE.BufferAttribute(new Uint16Array([0, 1, 2, 0, 2, 3]), 1),
  );
  geometry.instanceCount = instanceCount;
  return geometry;
}
