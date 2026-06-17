import { WebGPURenderer } from 'three/webgpu';
import { SplatMesh } from '../SplatMesh';
import * as THREE from "three";
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
export declare class WebGPUSplatRenderer extends THREE.Group {
    readonly isWebGPUSplatRenderer = true;
    readonly renderer: WebGPURenderer;
    private tsl;
    private wgpu;
    private numSplats;
    private mesh;
    private packedAttr;
    private projAttr;
    private keyA;
    private keyB;
    private idxA;
    private idxB;
    private tileHist;
    private tileBase;
    private blockSum;
    private blockBase;
    private sh1Attr;
    private sh2Attr;
    private sh3Attr;
    private numSh;
    private shMax;
    private computeNodes;
    private uModelView;
    private uModelView3;
    private uProjection;
    private uProjScale;
    private uRenderSize;
    private uViewOrigin;
    private uShMax;
    private uMaxStdDev;
    private uMinAlpha;
    private uMaxPixelRadius;
    private uMaxSplatScale;
    private uDepthMin;
    private uDepthMax;
    private readonly boundCenter;
    private boundRadius;
    private readonly tmpModelView;
    private readonly tmpModelView3;
    private readonly tmpViewInverse;
    private readonly tmpObjInverse;
    private readonly tmpCamPos;
    private readonly tmpSphere;
    private readonly tmpSize;
    maxStdDev: number;
    /**
     * Cull splats whose largest object-space scale exceeds this, to trim the big
     * diffuse "floater" blobs common in real captures. `Infinity` (default) keeps
     * everything; lower it toward the model size to remove background haze.
     */
    maxSplatScale: number;
    minAlpha: number;
    maxPixelRadius: number;
    /** Evaluate SH color (read on the next `setSplatMesh`). */
    enableSh: boolean;
    constructor(options: WebGPUSplatRendererOptions);
    /** Dynamically load the WebGPU/TSL modules. Idempotent. */
    init(): Promise<this>;
    /**
     * Upload a SplatMesh's packed data to the GPU and build the compute + raster
     * pipeline. Call after `await mesh.initialized`. Phase 2 supports a single
     * mesh; calling again replaces the previous one.
     */
    setSplatMesh(mesh: SplatMesh): void;
    private buildPipeline;
    private buildMaterial;
    /**
     * Update uniforms and dispatch the per-frame compute pipeline (project + GPU
     * radix sort). Call before rendering the scene. No GPU->CPU readback.
     */
    prepareFrame(camera: THREE.Camera): void;
    /** Convenience: prepareFrame + renderer.renderAsync. */
    renderFrame(scene: THREE.Scene, camera: THREE.Camera): Promise<void>;
    private disposeMesh;
    dispose(): void;
}
