import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
import type { PackedSplats } from "../PackedSplats";
import type { SplatMesh } from "../SplatMesh";
import {
  WGSL_COV2D,
  WGSL_DEPTH_KEY,
  WGSL_EVAL_SH1,
  WGSL_EVAL_SH2,
  WGSL_EVAL_SH3,
  WGSL_UNPACK_CENTER,
  WGSL_UNPACK_QUAT,
  WGSL_UNPACK_RGBA,
  WGSL_UNPACK_SCALES,
} from "./wgsl";

// Single-pass GPU counting sort over a 16-bit depth key (top 16 bits of the
// 32-bit far->near sort key). One pass means stability is not required, so the
// histogram + scatter can use atomics — only ~4 compute dispatches per frame
// (vs a multi-pass radix), and no inter-dispatch data race.
const DEPTH_BITS = 16;
const BUCKETS = 1 << DEPTH_BITS; // 65536
// The bucket prefix sum is a 2-level scan so it runs in parallel instead of one
// serial pass over all 65536 buckets (which is a fixed per-frame cost that
// otherwise dominates the frame regardless of splat count).
const SCAN_BLOCKS = 1024;
const SCAN_BLOCK_SIZE = BUCKETS / SCAN_BLOCKS; // 64

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
 * **in-frame GPU LSD radix sort** of depth keys, and an instanced raster draw
 * into the scene's shared depth buffer. The sort is produced and consumed in the
 * same frame — no GPU->CPU readback, no worker round-trip, no sort lag.
 *
 * Scope/caveats (documented, replaced in later phases):
 * - The radix sort uses a one-tile-per-invocation count/scatter with a 2-level
 *   parallel prefix-sum scan (no atomics). A workgroup-shared layout could
 *   reduce the size-1 dispatch overhead further.
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
  // Counting-sort buffers: per-splat 16-bit depth key, the sorted order output
  // (idxA, read by the raster), and the atomic bucket histogram + scanned offset.
  private keyA: StorageBufferAttribute | null = null;
  private idxA: StorageBufferAttribute | null = null;
  private hist: StorageBufferAttribute | null = null;
  private offset: StorageBufferAttribute | null = null;
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

  // Per-frame reused temporaries (avoid allocation in the loop).
  private readonly tmpModelView = new THREE.Matrix4();
  private readonly tmpModelView3 = new THREE.Matrix3();
  private readonly tmpViewInverse = new THREE.Matrix4();
  private readonly tmpObjInverse = new THREE.Matrix4();
  private readonly tmpCamPos = new THREE.Vector3();
  private readonly tmpSize = new THREE.Vector2();

  maxStdDev = Math.sqrt(8.0);
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
    // Counting-sort buffers: 16-bit key per splat, sorted order (idxA, read by
    // the raster), and the atomic histogram + scanned offset (BUCKETS entries).
    this.keyA = new StorageBufferAttribute(new Uint32Array(numSplats), 1);
    this.idxA = new StorageBufferAttribute(new Uint32Array(numSplats), 1);
    this.hist = new StorageBufferAttribute(new Uint32Array(BUCKETS), 1);
    this.offset = new StorageBufferAttribute(new Uint32Array(BUCKETS), 1);
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
      Loop,
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
      atomicAdd,
      atomicLoad,
      atomicStore,
    } = tsl;
    const N = this.numSplats;

    const unpackCenter = wgslFn(WGSL_UNPACK_CENTER);
    const unpackRgba = wgslFn(WGSL_UNPACK_RGBA);
    const unpackScales = wgslFn(WGSL_UNPACK_SCALES);
    const unpackQuat = wgslFn(WGSL_UNPACK_QUAT);
    const cov2DOf = wgslFn(WGSL_COV2D);
    const depthKey = wgslFn(WGSL_DEPTH_KEY);

    const sba = (a: StorageBufferAttribute | null) =>
      a as StorageBufferAttribute;
    const packedStore = storage(sba(this.packedAttr), "uvec4", N).toReadOnly();
    const projStore = storage(sba(this.projAttr), "vec4", N * 4);
    const keyAStore = storage(sba(this.keyA), "uint", N);
    const idxAStore = storage(sba(this.idxA), "uint", N);
    const histStore = storage(sba(this.hist), "uint", BUCKETS).toAtomic();
    const offsetStore = storage(sba(this.offset), "uint", BUCKETS).toAtomic();
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

      // Inactive splats get the max 16-bit key so they sort to the last bucket
      // (and are discarded by the raster via projected.active == 0).
      keyAStore.element(i).assign(uint(BUCKETS - 1));

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
          // 16-bit depth key = top 16 bits of the 32-bit far->near key.
          keyAStore.element(i).assign(depthKey(viewC.length()).shiftRight(16));
        });
      });
    });

    // --- Single-pass counting sort over the 16-bit depth key ---
    // clearHist: zero every histogram bucket (offset is fully (re)written by the
    // scan, so it needs no clear).
    const clearHist = Fn(() => {
      atomicStore(histStore.element(instanceIndex), uint(0));
    });
    // count: tally each splat into its depth bucket.
    const count = Fn(() => {
      atomicAdd(histStore.element(keyAStore.element(instanceIndex)), uint(1));
    });
    // scan: exclusive prefix sum hist -> offset (each bucket's first write
    // position) as a 2-level scan so it is parallel, not one serial pass over
    // all 65536 buckets. (1) each block sums its span -> blockSum, (2) one
    // invocation scans the block sums -> blockBase, (3) each block scans its
    // span seeded with its base -> offset.
    const scanReduce = Fn(() => {
      const start = instanceIndex.mul(SCAN_BLOCK_SIZE);
      const sum = uint(0).toVar();
      Loop(SCAN_BLOCK_SIZE, ({ i }) => {
        sum.assign(sum.add(atomicLoad(histStore.element(start.add(i)))));
      });
      blockSumStore.element(instanceIndex).assign(sum);
    });
    const scanBlocks = Fn(() => {
      const running = uint(0).toVar();
      Loop(SCAN_BLOCKS, ({ i }) => {
        blockBaseStore.element(uint(i)).assign(running);
        running.assign(running.add(blockSumStore.element(uint(i))));
      });
    });
    const scanDown = Fn(() => {
      const start = instanceIndex.mul(SCAN_BLOCK_SIZE);
      const running = blockBaseStore.element(instanceIndex).toVar();
      Loop(SCAN_BLOCK_SIZE, ({ i }) => {
        const e = start.add(i);
        atomicStore(offsetStore.element(e), running);
        running.assign(running.add(atomicLoad(histStore.element(e))));
      });
    });
    // scatter: each splat atomically claims the next slot in its bucket and
    // writes its index there. Unstable within a bucket, which is fine for a
    // single-pass sort (ties are at the same quantized depth).
    const scatter = Fn(() => {
      const i = instanceIndex;
      const pos = atomicAdd(offsetStore.element(keyAStore.element(i)), uint(1));
      idxAStore.element(pos).assign(i);
    });

    // project + clearHist are independent (disjoint buffers) -> one pass. count,
    // scan, scatter form a read-after-write chain -> each its own pass, since
    // THREE runs an array of compute nodes in ONE pass with no inter-dispatch
    // barrier (separate passes are synchronized). Dispatched in prepareFrame.
    this.computeNodes = [
      [project().compute(N), clearHist().compute(BUCKETS)],
      count().compute(N),
      scanReduce().compute(SCAN_BLOCKS),
      scanBlocks().compute(1),
      scanDown().compute(SCAN_BLOCKS),
      scatter().compute(N),
    ];
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
    this.idxA = null;
    this.hist = null;
    this.offset = null;
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
