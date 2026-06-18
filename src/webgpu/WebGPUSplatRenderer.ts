import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
import type { PackedSplats } from "../PackedSplats";
import type { SplatMesh } from "../SplatMesh";
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

// Stable LSD radix sort over a 16-bit depth key (radial distance normalized to
// the model's depth extent; see WGSL_DEPTH_KEY16), 2 passes of 8 bits. A
// *stable* sort is required: the draw order of splats within the same quantized
// depth must be deterministic frame to frame, otherwise alpha blending of
// overlapping splats flickers. The count is cooperative (one thread per splat,
// atomicAdd into a global per-tile histogram); the stable scatter stays per-tile
// (each tile owns a disjoint output range, no atomics). Inter-dispatch
// dependencies are run as separate (synchronized) passes.
const TILE = 256; // elements per tile
const RADIX_BITS = 8;
const RADIX = 1 << RADIX_BITS; // 256 buckets / digit
const RADIX_PASSES = 2; // 2 x 8 bits = 16-bit key
// Blocks for the 2-level parallel prefix sum over the per-tile histogram.
const SCAN_BLOCKS = 1024;

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
  // Radix-sort buffers: ping-pong key + index (payload), per-tile histogram +
  // scanned offsets, and the 2-level scan scratch. The sorted order lands in
  // idxA (even pass count), which the raster reads.
  private keyA: StorageBufferAttribute | null = null;
  private keyB: StorageBufferAttribute | null = null;
  private idxA: StorageBufferAttribute | null = null;
  private idxB: StorageBufferAttribute | null = null;
  private tileHist: StorageBufferAttribute | null = null;
  private tileBase: StorageBufferAttribute | null = null;
  private blockSum: StorageBufferAttribute | null = null;
  private blockBase: StorageBufferAttribute | null = null;
  // Spherical-harmonics coefficient buffers (present iff the mesh has SH).
  private sh1Attr: StorageBufferAttribute | null = null;
  private sh2Attr: StorageBufferAttribute | null = null;
  private sh3Attr: StorageBufferAttribute | null = null;
  private numSh = 0;
  private shMax = new THREE.Vector3(1, 1, 1);

  // Ordered list of compute kernels dispatched each frame (project + sort).
  private computeNodes: unknown[] = [];

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
    const numTiles = Math.ceil(numSplats / TILE);
    const histLen = numTiles * RADIX;

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
    // Radix-sort ping-pong buffers + per-tile histogram/offset + scan scratch.
    this.keyA = new StorageBufferAttribute(new Uint32Array(numSplats), 1);
    this.keyB = new StorageBufferAttribute(new Uint32Array(numSplats), 1);
    this.idxA = new StorageBufferAttribute(new Uint32Array(numSplats), 1);
    this.idxB = new StorageBufferAttribute(new Uint32Array(numSplats), 1);
    this.tileHist = new StorageBufferAttribute(new Uint32Array(histLen), 1);
    this.tileBase = new StorageBufferAttribute(new Uint32Array(histLen), 1);
    this.blockSum = new StorageBufferAttribute(new Uint32Array(SCAN_BLOCKS), 1);
    this.blockBase = new StorageBufferAttribute(
      new Uint32Array(SCAN_BLOCKS),
      1,
    );

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

    this.buildPipeline(numTiles);
    const material = this.buildMaterial(MeshBasicNodeMaterial);

    const geometry = makeQuadGeometry(numSplats);
    const drawMesh = new THREE.Mesh(geometry, material);
    drawMesh.frustumCulled = false;
    this.mesh = drawMesh;
    this.add(drawMesh);
  }

  private buildPipeline(numTiles: number): void {
    const tsl = this.tsl as TSL;
    const {
      Fn,
      If,
      Return,
      Loop,
      instanceIndex,
      workgroupId,
      atomicAdd,
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
    } = tsl;
    const N = this.numSplats;
    const histLen = numTiles * RADIX;
    const scanBlockSize = Math.ceil(histLen / SCAN_BLOCKS);

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
    const keyAStore = storage(sba(this.keyA), "uint", N);
    const keyBStore = storage(sba(this.keyB), "uint", N);
    const idxAStore = storage(sba(this.idxA), "uint", N);
    const idxBStore = storage(sba(this.idxB), "uint", N);
    const histStore = storage(sba(this.tileHist), "uint", histLen);
    // Atomic view of the same histogram buffer, for the cooperative count pass
    // (one thread per splat accumulating into global memory). The scan reads the
    // non-atomic `histStore` view in a later, synchronized pass.
    const histAtomic = storage(sba(this.tileHist), "uint", histLen).toAtomic();
    const baseStore = storage(sba(this.tileBase), "uint", histLen);
    const blockSumStore = storage(sba(this.blockSum), "uint", SCAN_BLOCKS);
    const blockBaseStore = storage(sba(this.blockBase), "uint", SCAN_BLOCKS);

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
          const scale1 = min(uMaxPixelRadius, adj.mul(sqrt(eigen1)));
          const scale2 = min(uMaxPixelRadius, adj.mul(sqrt(eigen2)));

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
          projStore
            .element(base.add(2))
            .assign(vec4(rgb.x, rgb.y, rgb.z, alpha));
          projStore.element(base.add(3)).assign(vec4(adj, 1.0, 0.0, 0.0));
          // 16-bit far->near key, normalized to the model's view-depth range.
          keyAStore
            .element(i)
            .assign(depthKey16(viewC.length(), uDepthMin, uDepthMax));
        });
      });
    });

    // --- Stable tiled radix sort ---
    // count (cooperative): one thread per splat accumulates the per-tile digit
    // histogram directly in global memory via atomicAdd. Workgroup size == TILE,
    // so workgroupId == tile index, and the layout (digit * numTiles + tile) is
    // bit-identical to the old serial count — scan + stable scatter are
    // unchanged. (TSL has no workgroup-shared atomics, so the accumulator is the
    // global buffer; the histogram must be zeroed first.) This replaces the old
    // ~numTiles serial threads (each looping TILE with a 256-entry private array)
    // with N parallel threads.
    const clearHist = Fn(() => {
      const i = instanceIndex;
      If(i.lessThan(uint(histLen)), () => {
        histStore.element(i).assign(uint(0));
      });
    });
    const countKernel = (keyIn: typeof keyAStore, shift: number) =>
      Fn(() => {
        const e = instanceIndex;
        If(e.lessThan(uint(N)), () => {
          const t = workgroupId.x; // workgroup size == TILE -> tile index
          const digit = keyIn
            .element(e)
            .shiftRight(shift)
            .bitAnd(RADIX - 1);
          atomicAdd(histAtomic.element(digit.mul(numTiles).add(t)), uint(1));
        });
      });

    // 2-level exclusive prefix sum over the digit-major histogram -> tileBase.
    const scanReduce = Fn(() => {
      If(instanceIndex.greaterThanEqual(uint(SCAN_BLOCKS)), () => {
        Return();
      });
      const start = instanceIndex.mul(scanBlockSize);
      const sum = uint(0).toVar();
      Loop(scanBlockSize, ({ i }) => {
        const e = start.add(i);
        If(e.lessThan(histLen), () => {
          sum.assign(sum.add(histStore.element(e)));
        });
      });
      blockSumStore.element(instanceIndex).assign(sum);
    });
    const scanBlocks = Fn(() => {
      // Dispatched with count 1, but THREE still launches a full workgroup, so
      // every lane but 0 must bail — otherwise 64 invocations race writing the
      // same serial prefix sum into blockBaseStore (the worst offender: this
      // dispatch is *always* padded regardless of N).
      If(instanceIndex.greaterThanEqual(uint(1)), () => {
        Return();
      });
      const running = uint(0).toVar();
      Loop(SCAN_BLOCKS, ({ i }) => {
        blockBaseStore.element(uint(i)).assign(running);
        running.assign(running.add(blockSumStore.element(uint(i))));
      });
    });
    const scanDown = Fn(() => {
      If(instanceIndex.greaterThanEqual(uint(SCAN_BLOCKS)), () => {
        Return();
      });
      const start = instanceIndex.mul(scanBlockSize);
      const running = blockBaseStore.element(instanceIndex).toVar();
      Loop(scanBlockSize, ({ i }) => {
        const e = start.add(i);
        If(e.lessThan(histLen), () => {
          baseStore.element(e).assign(running);
          running.assign(running.add(histStore.element(e)));
        });
      });
    });

    // scatter: each tile re-reads its digits and emits, in element order, to the
    // running global offset for that (digit, tile) -> stable. Disjoint output
    // ranges per (digit, tile), so no atomics needed.
    const scatterKernel = (
      keyIn: typeof keyAStore,
      idxIn: typeof idxAStore,
      keyOut: typeof keyAStore,
      idxOut: typeof idxAStore,
      shift: number,
    ) =>
      Fn(() => {
        const t = instanceIndex;
        If(t.greaterThanEqual(uint(numTiles)), () => {
          Return();
        });
        const offset = tsl.array("uint", RADIX).toVar();
        Loop(RADIX, ({ i }) => {
          const d = uint(i);
          offset.element(d).assign(baseStore.element(d.mul(numTiles).add(t)));
        });
        const start = t.mul(TILE);
        Loop(TILE, ({ i }) => {
          const e = start.add(i);
          If(e.lessThan(N), () => {
            const key = keyIn.element(e);
            const digit = key.shiftRight(shift).bitAnd(RADIX - 1);
            const pos = offset.element(digit).toVar();
            keyOut.element(pos).assign(key);
            idxOut.element(pos).assign(idxIn.element(e));
            offset.element(digit).assign(pos.add(1));
          });
        });
      });

    // Each stage is a read-after-write dependency, so each is dispatched as its
    // own (synchronized) pass in prepareFrame. After RADIX_PASSES (even) passes
    // the sorted payload is back in idxA, which the raster reads.
    const nodes: unknown[] = [project().compute(N)];
    let inKey = keyAStore;
    let inIdx = idxAStore;
    let outKey = keyBStore;
    let outIdx = idxBStore;
    for (let p = 0; p < RADIX_PASSES; p++) {
      const shift = p * RADIX_BITS;
      // Zero the histogram, then accumulate it cooperatively (workgroup size ==
      // TILE so workgroupId is the tile index).
      nodes.push(clearHist().compute(histLen));
      nodes.push(countKernel(inKey, shift)().compute(N, [TILE]));
      nodes.push(scanReduce().compute(SCAN_BLOCKS));
      nodes.push(scanBlocks().compute(1));
      nodes.push(scanDown().compute(SCAN_BLOCKS));
      nodes.push(
        scatterKernel(inKey, inIdx, outKey, outIdx, shift)().compute(numTiles),
      );
      [inKey, outKey] = [outKey, inKey];
      [inIdx, outIdx] = [outIdx, inIdx];
    }
    this.computeNodes = nodes;
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
      // The stored splat color is sRGB, but WebGPURenderer applies an output
      // linear->sRGB (sRGB OETF) conversion to the fragment color. Apply the
      // exact sRGB EOTF (sRGB->linear) here so the two cancel and the on-screen
      // color equals the stored sRGB value, matching the WebGL path.
      // max(0) guards against negative SH lobes.
      const c = vColor.xyz.max(0.0);
      const linear = select(
        c.lessThanEqual(0.04045),
        c.div(12.92),
        c.add(0.055).div(1.055).pow(2.4),
      );
      // Output STRAIGHT (non-premultiplied) color: NodeMaterial premultiplies it
      // itself when `premultipliedAlpha === true`. Outputting premultiplied here
      // would double-premultiply (rgb*alpha^2), darkening semi-transparent splats
      // to near nothing and leaving only opaque splats (a sparkle artifact).
      return vec4(linear, alpha);
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

    // GPU compute: project every splat, then radix-sort by depth in-frame.
    // Each stage is dispatched as its own compute pass: THREE runs an array of
    // compute nodes in a single WebGPU pass with no barrier between dispatches,
    // but these stages have read-after-write dependencies (project -> count ->
    // scan -> scatter). Separate passes are synchronized, avoiding the data
    // race that otherwise makes the sort order vary frame to frame (flicker).
    for (const node of this.computeNodes) {
      this.renderer.compute(node as never);
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
  }

  dispose(): void {
    this.disposeMesh();
    this.packedAttr = null;
    this.projAttr = null;
    this.keyA = null;
    this.keyB = null;
    this.idxA = null;
    this.idxB = null;
    this.tileHist = null;
    this.tileBase = null;
    this.blockSum = null;
    this.blockBase = null;
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
