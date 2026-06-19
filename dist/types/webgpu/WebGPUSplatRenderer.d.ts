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
    private idxA;
    private sh1Attr;
    private sh2Attr;
    private sh3Attr;
    private numSh;
    private shMax;
    private computeNodes;
    private rawSort;
    private rawSortBound;
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
    private recomputePending;
    /** Debug counters: how often prepareFrame recomputes vs skips (gate). */
    computeRuns: number;
    skipRuns: number;
    private readonly prevModelView;
    private readonly prevProjection;
    private readonly prevSize;
    private prevStd;
    private prevMinAlpha;
    private prevMaxPixelRadius;
    private prevMaxSplatScale;
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
    /**
     * Debug: skip the depth sort and render in the project pass's identity order.
     * The result is wrong (no painter's order), but A/B-ing fps with this on/off
     * isolates how much of the frame the sort actually costs vs project + raster.
     */
    skipSort: boolean;
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
