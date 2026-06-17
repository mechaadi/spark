import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
import type { PackedSplats } from "../PackedSplats";
import type { SplatMesh } from "../SplatMesh";
import { fromHalf } from "../utils";
import {
  WGSL_UNPACK_CENTER,
  WGSL_UNPACK_MAX_SCALE,
  WGSL_UNPACK_RGBA,
} from "./wgsl";

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
 * Phase 2 minimal WebGPU splat backend.
 *
 * Proves the end-to-end WebGPU plumbing for Spark: packed-splat storage-buffer
 * upload, a per-splat compute (project) dispatch, and an instanced raster draw
 * through THREE.WebGPURenderer into the scene's shared depth buffer.
 *
 * Scope/caveats for Phase 2 (documented, replaced in later phases):
 * - Projection is *isotropic* (round splats sized by max scale). Full
 *   anisotropic 2D-covariance projection lands in Phase 4 ("GPU cull+project").
 * - Sorting is a *throwaway CPU sort* of depths derived on the CPU; this still
 *   exercises the GPU compute + raster path. The GPU radix sort lands in Phase 3.
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
  private orderAttr: StorageBufferAttribute | null = null;
  private orderArray: Uint32Array | null = null;

  // Compute kernel node (built once per mesh).
  private computeNode: unknown = null;

  // CPU-side throwaway-sort scratch.
  private centers: Float32Array | null = null;
  private depths: Float32Array | null = null;
  private indexList: number[] = [];

  // Uniform nodes (their `.value` is updated per frame).
  private uModelView: { value: THREE.Matrix4 } | null = null;
  private uProjection: { value: THREE.Matrix4 } | null = null;
  private uProjScale: { value: THREE.Vector2 } | null = null;
  private uMaxStdDev: { value: number } | null = null;
  private uMinAlpha: { value: number } | null = null;

  // Per-frame reused temporaries (avoid allocation in the loop).
  private readonly tmpModelView = new THREE.Matrix4();
  private readonly tmpViewInverse = new THREE.Matrix4();

  maxStdDev = Math.sqrt(8.0);
  minAlpha = 0.5 * (1.0 / 255.0);

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
        "WebGPUSplatRenderer (Phase 2) supports PackedSplats only; this mesh " +
          "has no packed data ready. ExtSplats/Paged scenes use the WebGL2 session.",
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
    // Draw order: splat index per instance, refilled each frame by the CPU sort.
    this.orderArray = new Uint32Array(numSplats);
    for (let i = 0; i < numSplats; i++) this.orderArray[i] = i;
    this.orderAttr = new StorageBufferAttribute(this.orderArray, 1);

    this.buildComputePass();
    const material = this.buildMaterial(MeshBasicNodeMaterial);

    const geometry = makeQuadGeometry(numSplats);
    const drawMesh = new THREE.Mesh(geometry, material);
    drawMesh.frustumCulled = false;
    this.mesh = drawMesh;
    this.add(drawMesh);

    // Pre-unpack CPU-side centers (float16 -> float32) once for the throwaway sort.
    this.centers = new Float32Array(numSplats * 3);
    this.depths = new Float32Array(numSplats);
    this.indexList = new Array(numSplats);
    for (let i = 0; i < numSplats; i++) {
      const w1 = packedArray[i * 4 + 1];
      const w2 = packedArray[i * 4 + 2];
      this.centers[i * 3 + 0] = fromHalf(w1 & 0xffff);
      this.centers[i * 3 + 1] = fromHalf((w1 >>> 16) & 0xffff);
      this.centers[i * 3 + 2] = fromHalf(w2 & 0xffff);
      this.indexList[i] = i;
    }
  }

  private buildComputePass(): void {
    const tsl = this.tsl as TSL;
    const { Fn, If, instanceIndex, vec4, uniform, storage, wgslFn } = tsl;

    const unpackCenter = wgslFn(WGSL_UNPACK_CENTER);
    const unpackRgba = wgslFn(WGSL_UNPACK_RGBA);
    const unpackMaxScale = wgslFn(WGSL_UNPACK_MAX_SCALE);

    const packedStore = storage(
      this.packedAttr as StorageBufferAttribute,
      "uvec4",
      this.numSplats,
    ).toReadOnly();
    const projStore = storage(
      this.projAttr as StorageBufferAttribute,
      "vec4",
      this.numSplats * 4,
    );

    const uModelView = uniform(new THREE.Matrix4());
    const uProjection = uniform(new THREE.Matrix4());
    const uProjScale = uniform(new THREE.Vector2(1, 1));
    const uMaxStdDev = uniform(this.maxStdDev);
    const uMinAlpha = uniform(this.minAlpha);
    this.uModelView = uModelView as unknown as { value: THREE.Matrix4 };
    this.uProjection = uProjection as unknown as { value: THREE.Matrix4 };
    this.uProjScale = uProjScale as unknown as { value: THREE.Vector2 };
    this.uMaxStdDev = uMaxStdDev as unknown as { value: number };
    this.uMinAlpha = uMinAlpha as unknown as { value: number };

    const project = Fn(() => {
      const i = instanceIndex;
      const pk = packedStore.element(i);
      const center = unpackCenter(pk);
      const rgba = unpackRgba(pk);
      const maxScale = unpackMaxScale(pk);

      const base = i.mul(4);
      const zero = vec4(0.0, 0.0, 0.0, 0.0);
      projStore.element(base).assign(zero);
      projStore.element(base.add(1)).assign(zero);
      projStore.element(base.add(2)).assign(zero);
      projStore.element(base.add(3)).assign(zero);

      const viewC = uModelView.mul(vec4(center, 1.0)).xyz.toVar();
      const a = rgba.w.mul(2.0).toVar();
      const adj = uMaxStdDev.toVar();
      If(a.greaterThan(1.0), () => {
        a.assign(a.mul(4.0).sub(3.0).min(5.0));
        adj.assign(uMaxStdDev.add(a.sub(1.0).mul(0.7)));
      });

      const active = viewC.z
        .lessThan(0.0)
        .and(maxScale.greaterThan(0.0))
        .and(a.greaterThanEqual(uMinAlpha));
      If(active, () => {
        const clip = uProjection.mul(vec4(viewC, 1.0)).toVar();
        If(clip.w.greaterThan(0.0), () => {
          const ndc = clip.xyz.div(clip.w);
          const r = maxScale.mul(adj).div(viewC.z.negate());
          const ndcRx = uProjScale.x.mul(r);
          const ndcRy = uProjScale.y.mul(r);
          projStore.element(base).assign(vec4(ndc.x, ndc.y, clip.z, clip.w));
          projStore.element(base.add(1)).assign(vec4(ndcRx, 0.0, 0.0, ndcRy));
          projStore
            .element(base.add(2))
            .assign(vec4(rgba.x, rgba.y, rgba.z, a));
          projStore.element(base.add(3)).assign(vec4(adj, 1.0, 0.0, 0.0));
        });
      });
    });

    this.computeNode = project().compute(this.numSplats);
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
    const orderStore = storage(
      this.orderAttr as StorageBufferAttribute,
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
      return vec4(vColor.xyz.mul(alpha), alpha);
    })();

    return material;
  }

  /**
   * Update uniforms, dispatch the project compute pass, and refresh the draw
   * order via the throwaway CPU sort. Call before rendering the scene.
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

    const proj = (camera as THREE.PerspectiveCamera).projectionMatrix;
    (this.uModelView as { value: THREE.Matrix4 }).value.copy(this.tmpModelView);
    (this.uProjection as { value: THREE.Matrix4 }).value.copy(proj);
    (this.uProjScale as { value: THREE.Vector2 }).value.set(
      proj.elements[0],
      proj.elements[5],
    );
    (this.uMaxStdDev as { value: number }).value = this.maxStdDev;
    (this.uMinAlpha as { value: number }).value = this.minAlpha;

    // GPU compute: project every splat into the projected storage buffer.
    this.renderer.compute(this.computeNode as never);

    this.cpuSort();
  }

  /** Convenience: prepareFrame + renderer.renderAsync. */
  async renderFrame(scene: THREE.Scene, camera: THREE.Camera): Promise<void> {
    this.prepareFrame(camera);
    await this.renderer.renderAsync(scene, camera);
  }

  // Throwaway Phase 2 sort: transform centers by modelView on the CPU, order
  // back-to-front (far first). Replaced by the in-frame GPU radix sort in Phase 3.
  private cpuSort(): void {
    const n = this.numSplats;
    const centers = this.centers as Float32Array;
    const depths = this.depths as Float32Array;
    const m = this.tmpModelView.elements;
    for (let i = 0; i < n; i++) {
      const x = centers[i * 3];
      const y = centers[i * 3 + 1];
      const z = centers[i * 3 + 2];
      // view-space position via modelView (column-major elements)
      const vx = m[0] * x + m[4] * y + m[8] * z + m[12];
      const vy = m[1] * x + m[5] * y + m[9] * z + m[13];
      const vz = m[2] * x + m[6] * y + m[10] * z + m[14];
      depths[i] = vx * vx + vy * vy + vz * vz; // radial distance squared
    }
    const list = this.indexList;
    list.sort((a, b) => depths[b] - depths[a]); // far -> near
    const order = this.orderArray as Uint32Array;
    for (let i = 0; i < n; i++) order[i] = list[i];
    const attr = this.orderAttr as StorageBufferAttribute & {
      needsUpdate: boolean;
    };
    attr.needsUpdate = true;
  }

  private disposeMesh(): void {
    if (this.mesh) {
      this.remove(this.mesh);
      this.mesh.geometry.dispose();
      (this.mesh.material as THREE.Material).dispose();
      this.mesh = null;
    }
    this.computeNode = null;
  }

  dispose(): void {
    this.disposeMesh();
    this.packedAttr = null;
    this.projAttr = null;
    this.orderAttr = null;
    this.orderArray = null;
    this.centers = null;
    this.depths = null;
    this.indexList = [];
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
