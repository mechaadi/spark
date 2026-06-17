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

// Tiling for the GPU LSD radix sort: each tile owns a disjoint span of TILE
// elements and a disjoint output range, so no atomics or barriers are needed.
const TILE = 256; // elements per tile
const RADIX_BITS = 8;
const RADIX = 1 << RADIX_BITS; // 256 buckets
const RADIX_PASSES = 32 / RADIX_BITS; // 4 passes over a 32-bit key

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
 * - The radix sort uses a simple one-tile-per-invocation layout (no atomics);
 *   a workgroup-shared optimization is a Phase 5 perf item.
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
  // Radix-sort ping-pong key + payload (splat index) buffers, plus per-tile
  // histogram / scanned-offset scratch. The final sorted index lands in idxA.
  private keyA: StorageBufferAttribute | null = null;
  private keyB: StorageBufferAttribute | null = null;
  private idxA: StorageBufferAttribute | null = null;
  private idxB: StorageBufferAttribute | null = null;
  private tileHist: StorageBufferAttribute | null = null;
  private tileBase: StorageBufferAttribute | null = null;
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

  constructor(options: WebGPUSplatRendererOptions) {
    super();
    if (!options?.renderer) {
      throw new Error("WebGPUSplatRenderer requires a THREE.WebGPURenderer");
    }
    this.renderer = options.renderer;
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
    // Radix-sort ping-pong buffers + per-tile histogram/offset scratch.
    this.keyA = new StorageBufferAttribute(new Uint32Array(numSplats), 1);
    this.keyB = new StorageBufferAttribute(new Uint32Array(numSplats), 1);
    this.idxA = new StorageBufferAttribute(new Uint32Array(numSplats), 1);
    this.idxB = new StorageBufferAttribute(new Uint32Array(numSplats), 1);
    this.tileHist = new StorageBufferAttribute(new Uint32Array(histLen), 1);
    this.tileBase = new StorageBufferAttribute(new Uint32Array(histLen), 1);

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
    if (extra.sh1 && extra.sh1.length >= numSplats * 2) {
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
    } = tsl;
    const N = this.numSplats;
    const histLen = numTiles * RADIX;

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
    const keyBStore = storage(sba(this.keyB), "uint", N);
    const idxAStore = storage(sba(this.idxA), "uint", N);
    const idxBStore = storage(sba(this.idxB), "uint", N);
    const histStore = storage(sba(this.tileHist), "uint", histLen);
    const baseStore = storage(sba(this.tileBase), "uint", histLen);

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

      // Identity payload; inactive splats get the max key so they sort last
      // (and are discarded by the raster via projected.active == 0).
      idxAStore.element(i).assign(i);
      keyAStore.element(i).assign(uint(0xffffffff));

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
          keyAStore.element(i).assign(depthKey(viewC.length()));
        });
      });
    });

    // --- Radix sort kernels (one tile per invocation) ---
    // Per-tile histogram of the current 8-bit digit, written digit-major so a
    // single linear prefix sum over the whole array yields each (digit, tile)'s
    // global base offset.
    const countKernel = (keyIn: typeof keyAStore, shift: number) =>
      Fn(() => {
        const t = instanceIndex;
        const hist = tsl.array("uint", RADIX).toVar();
        Loop(RADIX, ({ i }) => {
          hist.element(uint(i)).assign(uint(0));
        });
        const start = t.mul(TILE);
        Loop(TILE, ({ i }) => {
          const e = start.add(i);
          If(e.lessThan(N), () => {
            const digit = keyIn
              .element(e)
              .shiftRight(shift)
              .bitAnd(RADIX - 1);
            hist.element(digit).assign(hist.element(digit).add(1));
          });
        });
        Loop(RADIX, ({ i }) => {
          const d = uint(i);
          histStore.element(d.mul(numTiles).add(t)).assign(hist.element(d));
        });
      });

    // Single-invocation exclusive prefix sum over the digit-major histogram.
    const scanKernel = () =>
      Fn(() => {
        const running = uint(0).toVar();
        Loop(histLen, ({ i }) => {
          baseStore.element(i).assign(running);
          running.assign(running.add(histStore.element(i)));
        });
      });

    // Stable scatter: each tile re-reads its digits and emits to the running
    // global offset for that (digit, tile).
    const scatterKernel = (
      keyIn: typeof keyAStore,
      idxIn: typeof idxAStore,
      keyOut: typeof keyAStore,
      idxOut: typeof idxAStore,
      shift: number,
    ) =>
      Fn(() => {
        const t = instanceIndex;
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

    const nodes: unknown[] = [project().compute(N)];
    let inKey = keyAStore;
    let inIdx = idxAStore;
    let outKey = keyBStore;
    let outIdx = idxBStore;
    for (let p = 0; p < RADIX_PASSES; p++) {
      const shift = p * RADIX_BITS;
      nodes.push(countKernel(inKey, shift)().compute(numTiles));
      nodes.push(scanKernel()().compute(1));
      nodes.push(
        scatterKernel(inKey, inIdx, outKey, outIdx, shift)().compute(numTiles),
      );
      [inKey, outKey] = [outKey, inKey];
      [inIdx, outIdx] = [outIdx, inIdx];
    }
    // After an even number of passes the sorted payload is back in idxA, which
    // the raster reads. (RADIX_PASSES is 4.)
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
      varyingProperty,
      storage,
      mix,
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

      const off = corner.x.mul(p1.xy).add(corner.y.mul(p1.zw));
      const clipW = p0.w;
      const clipZ = p0.z;
      const pos = vec4(p0.xy.add(off).mul(clipW), clipZ, clipW);
      const offscreen = vec4(0.0, 0.0, 2.0, 1.0);
      return mix(offscreen, pos, meta.y);
    })();

    material.fragmentNode = Fn(() => {
      const z2 = vUv.dot(vUv);
      const alpha = vColor.w.mul(z2.mul(-0.5).exp()).toVar();
      If(alpha.lessThan(uMinAlpha.value), () => {
        Discard();
      });
      // Match Spark's WebGL fragment (encodeLinear): the stored splat color is
      // sRGB; convert to linear so WebGPURenderer's linear->sRGB output yields
      // the same on-screen color. max(0) guards against negative SH lobes.
      const linear = vColor.xyz.max(0.0).pow(2.2);
      return vec4(linear.mul(alpha), alpha);
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
    this.renderer.compute(this.computeNodes as never);
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
