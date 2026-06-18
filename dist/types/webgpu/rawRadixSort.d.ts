type GpuBuf = {
    destroy(): void;
};
export declare class RawRadixSort {
    private readonly device;
    private n;
    private numTiles;
    private histLen;
    private keyB;
    private idxB;
    private hist;
    private base;
    private blockSum;
    private blockBase;
    private paramsBuf;
    private scanParamsBuf;
    private countPipe;
    private scatterPipe;
    private scanReducePipe;
    private scanBlocksPipe;
    private scanDownPipe;
    private groups;
    constructor(device: any);
    /** (Re)allocate scratch + pipelines for a splat count. Call on setSplatMesh. */
    configure(numSplats: number): void;
    /**
     * Bind the external (THREE-owned) key + index buffers. `keyA`/`idxA` hold the
     * depth key + identity payload from the project pass and receive the sorted
     * result (even pass count). Rebuilds bind groups.
     */
    bindIO(keyA: GpuBuf, idxA: GpuBuf): void;
    /** Record + submit the whole sort as one command buffer. */
    run(): void;
    dispose(): void;
}
export {};
