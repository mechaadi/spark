// Raw-WebGPU stable LSD radix sort, recorded into a SINGLE command buffer.
//
// The TSL/THREE compute path dispatches every sort stage as its own
// `renderer.compute()` = its own `queue.submit()`. For an in-frame sort that is
// ~10-13 submits/frame, and the per-submit CPU overhead + GPU sync bubbles
// dominate the frame (confirmed sort-bound on hardware). Here we drop to raw
// WebGPU compute on THREE's own `GPUDevice`: all passes go into one
// `GPUCommandEncoder` (WebGPU inserts execution+memory barriers between compute
// passes automatically, so the stages stay correctly ordered) and we submit
// once. The per-workgroup histogram also lives in real workgroup-shared memory
// (`var<workgroup>` atomics), which TSL can't express.
//
// Render and projection stay on THREE.WebGPURenderer; only the sort drops to
// raw compute, reading/writing the same key+index storage buffers THREE
// allocated (shared via `backend.get(attr).buffer`).
//
// Algorithm: same stable tiled LSD radix as the TSL path — 2 passes x 8 bits
// over a 16-bit depth key. count (workgroup-shared atomic histogram, digit-major
// global layout) -> 2-level exclusive prefix sum -> stable scatter (one tile per
// workgroup, serial within the tile to preserve order). Bit-identical ordering
// to the TSL implementation, so the CPU unit test still describes it.

const TILE = 256; // elements per tile == workgroup size
const RADIX = 256; // 8-bit digit
const RADIX_PASSES = 2; // 16-bit key
const SCAN_BLOCKS = 1024;

const COUNT_WGSL = (shift: number) => /* wgsl */ `
@group(0) @binding(0) var<storage, read> keyIn : array<u32>;
@group(0) @binding(1) var<storage, read_write> hist : array<u32>;
@group(0) @binding(2) var<uniform> params : vec2<u32>; // x=N, y=numTiles

var<workgroup> lhist : array<atomic<u32>, ${RADIX}>;

@compute @workgroup_size(${TILE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_index) lid : u32,
        @builtin(workgroup_id) wid : vec3<u32>) {
  atomicStore(&lhist[lid], 0u);
  workgroupBarrier();
  let e = gid.x;
  if (e < params.x) {
    let d = (keyIn[e] >> ${shift}u) & ${RADIX - 1}u;
    atomicAdd(&lhist[d], 1u);
  }
  workgroupBarrier();
  // digit-major: hist[digit * numTiles + tile]; each (digit,tile) written once.
  hist[lid * params.y + wid.x] = atomicLoad(&lhist[lid]);
}
`;

const SCAN_REDUCE_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> hist : array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSum : array<u32>;
@group(0) @binding(2) var<uniform> sp : vec2<u32>; // x=histLen, y=blockSize

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let b = gid.x;
  if (b >= ${SCAN_BLOCKS}u) { return; }
  let start = b * sp.y;
  var sum = 0u;
  for (var i = 0u; i < sp.y; i = i + 1u) {
    let e = start + i;
    if (e < sp.x) { sum = sum + hist[e]; }
  }
  blockSum[b] = sum;
}
`;

const SCAN_BLOCKS_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> blockSum : array<u32>;
@group(0) @binding(1) var<storage, read_write> blockBase : array<u32>;

@compute @workgroup_size(1)
fn main() {
  var running = 0u;
  for (var i = 0u; i < ${SCAN_BLOCKS}u; i = i + 1u) {
    blockBase[i] = running;
    running = running + blockSum[i];
  }
}
`;

const SCAN_DOWN_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> hist : array<u32>;
@group(0) @binding(1) var<storage, read> blockBase : array<u32>;
@group(0) @binding(2) var<storage, read_write> base : array<u32>;
@group(0) @binding(3) var<uniform> sp : vec2<u32>; // x=histLen, y=blockSize

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let b = gid.x;
  if (b >= ${SCAN_BLOCKS}u) { return; }
  let start = b * sp.y;
  var running = blockBase[b];
  for (var i = 0u; i < sp.y; i = i + 1u) {
    let e = start + i;
    if (e < sp.x) {
      base[e] = running;
      running = running + hist[e];
    }
  }
}
`;

const SCATTER_WGSL = (shift: number) => /* wgsl */ `
@group(0) @binding(0) var<storage, read> keyIn : array<u32>;
@group(0) @binding(1) var<storage, read> idxIn : array<u32>;
@group(0) @binding(2) var<storage, read> base : array<u32>;
@group(0) @binding(3) var<storage, read_write> keyOut : array<u32>;
@group(0) @binding(4) var<storage, read_write> idxOut : array<u32>;
@group(0) @binding(5) var<uniform> params : vec2<u32>; // x=N, y=numTiles

var<workgroup> soffset : array<u32, ${RADIX}>;

@compute @workgroup_size(${TILE})
fn main(@builtin(local_invocation_index) lid : u32,
        @builtin(workgroup_id) wid : vec3<u32>) {
  let N = params.x;
  let numTiles = params.y;
  // Each digit's global start for this tile (parallel load across the workgroup).
  soffset[lid] = base[lid * numTiles + wid.x];
  workgroupBarrier();
  // Stable placement: emit this tile's elements in order. Serial within the
  // tile (one thread) keeps equal-depth draw order deterministic == no flicker.
  if (lid == 0u) {
    let start = wid.x * ${TILE}u;
    for (var j = 0u; j < ${TILE}u; j = j + 1u) {
      let e = start + j;
      if (e < N) {
        let key = keyIn[e];
        let d = (key >> ${shift}u) & ${RADIX - 1}u;
        let pos = soffset[d];
        soffset[d] = pos + 1u;
        keyOut[pos] = key;
        idxOut[pos] = idxIn[e];
      }
    }
  }
}
`;

/** Minimal GPU types (avoid a DOM lib dependency in this module's typing). */
type Dev = {
  createBuffer(d: object): GpuBuf;
  createShaderModule(d: { code: string }): object;
  createComputePipeline(d: object): GpuPipeline;
  createBindGroup(d: object): object;
  createCommandEncoder(): GpuEncoder;
  queue: {
    submit(b: object[]): void;
    writeBuffer(b: GpuBuf, o: number, d: BufferSource): void;
  };
};
type GpuBuf = { destroy(): void };
type GpuPipeline = { getBindGroupLayout(i: number): object };
type GpuEncoder = {
  beginComputePass(): GpuPass;
  finish(): object;
};
type GpuPass = {
  setPipeline(p: GpuPipeline): void;
  setBindGroup(i: number, g: object): void;
  dispatchWorkgroups(x: number): void;
  end(): void;
};

const STORAGE = 0x80; // GPUBufferUsage.STORAGE
const UNIFORM = 0x40; // GPUBufferUsage.UNIFORM
const COPY_DST = 0x08; // GPUBufferUsage.COPY_DST

export class RawRadixSort {
  private readonly device: Dev;
  private n = 0;
  private numTiles = 0;
  private histLen = 0;

  // Scratch buffers (owned here).
  private keyB: GpuBuf | null = null;
  private idxB: GpuBuf | null = null;
  private hist: GpuBuf | null = null;
  private base: GpuBuf | null = null;
  private blockSum: GpuBuf | null = null;
  private blockBase: GpuBuf | null = null;
  private paramsBuf: GpuBuf | null = null; // [N, numTiles]
  private scanParamsBuf: GpuBuf | null = null; // [histLen, blockSize]

  // Pipelines.
  private countPipe: GpuPipeline[] = [];
  private scatterPipe: GpuPipeline[] = [];
  private scanReducePipe: GpuPipeline | null = null;
  private scanBlocksPipe: GpuPipeline | null = null;
  private scanDownPipe: GpuPipeline | null = null;

  // Bind groups, rebuilt when the external key/idx buffers are (re)bound.
  private groups: {
    count: object[]; // [pass]
    scatter: object[]; // [pass]
    scanReduce: object;
    scanBlocks: object;
    scanDown: object;
  } | null = null;

  // biome-ignore lint/suspicious/noExplicitAny: THREE GPUDevice is untyped here.
  constructor(device: any) {
    this.device = device as Dev;
  }

  /** (Re)allocate scratch + pipelines for a splat count. Call on setSplatMesh. */
  configure(numSplats: number): void {
    this.dispose();
    this.n = numSplats;
    this.numTiles = Math.ceil(numSplats / TILE);
    this.histLen = this.numTiles * RADIX;
    const blockSize = Math.ceil(this.histLen / SCAN_BLOCKS);

    const d = this.device;
    const buf = (bytes: number, usage: number) =>
      d.createBuffer({ size: Math.max(4, bytes), usage });
    this.keyB = buf(numSplats * 4, STORAGE);
    this.idxB = buf(numSplats * 4, STORAGE);
    this.hist = buf(this.histLen * 4, STORAGE);
    this.base = buf(this.histLen * 4, STORAGE);
    this.blockSum = buf(SCAN_BLOCKS * 4, STORAGE);
    this.blockBase = buf(SCAN_BLOCKS * 4, STORAGE);
    this.paramsBuf = buf(8, UNIFORM | COPY_DST);
    this.scanParamsBuf = buf(8, UNIFORM | COPY_DST);
    d.queue.writeBuffer(
      this.paramsBuf,
      0,
      new Uint32Array([this.n, this.numTiles]),
    );
    d.queue.writeBuffer(
      this.scanParamsBuf,
      0,
      new Uint32Array([this.histLen, blockSize]),
    );

    const pipe = (code: string) =>
      d.createComputePipeline({
        layout: "auto",
        compute: { module: d.createShaderModule({ code }), entryPoint: "main" },
      });
    this.countPipe = [pipe(COUNT_WGSL(0)), pipe(COUNT_WGSL(8))];
    this.scatterPipe = [pipe(SCATTER_WGSL(0)), pipe(SCATTER_WGSL(8))];
    this.scanReducePipe = pipe(SCAN_REDUCE_WGSL);
    this.scanBlocksPipe = pipe(SCAN_BLOCKS_WGSL);
    this.scanDownPipe = pipe(SCAN_DOWN_WGSL);
    this.groups = null;
  }

  /**
   * Bind the external (THREE-owned) key + index buffers. `keyA`/`idxA` hold the
   * depth key + identity payload from the project pass and receive the sorted
   * result (even pass count). Rebuilds bind groups.
   */
  bindIO(keyA: GpuBuf, idxA: GpuBuf): void {
    const d = this.device;
    const bg = (
      pipe: GpuPipeline,
      entries: { binding: number; resource: { buffer: GpuBuf } }[],
    ) => d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    const b = (binding: number, buffer: GpuBuf) => ({
      binding,
      resource: { buffer },
    });

    const keyB = this.keyB as GpuBuf;
    const idxB = this.idxB as GpuBuf;
    const hist = this.hist as GpuBuf;
    const base = this.base as GpuBuf;
    const params = this.paramsBuf as GpuBuf;

    // Pass 0 reads keyA, writes keyB/idxB. Pass 1 reads keyB, writes keyA/idxA.
    this.groups = {
      count: [
        bg(this.countPipe[0], [b(0, keyA), b(1, hist), b(2, params)]),
        bg(this.countPipe[1], [b(0, keyB), b(1, hist), b(2, params)]),
      ],
      scatter: [
        bg(this.scatterPipe[0], [
          b(0, keyA),
          b(1, idxA),
          b(2, base),
          b(3, keyB),
          b(4, idxB),
          b(5, params),
        ]),
        bg(this.scatterPipe[1], [
          b(0, keyB),
          b(1, idxB),
          b(2, base),
          b(3, keyA),
          b(4, idxA),
          b(5, params),
        ]),
      ],
      scanReduce: bg(this.scanReducePipe as GpuPipeline, [
        b(0, hist),
        b(1, this.blockSum as GpuBuf),
        b(2, this.scanParamsBuf as GpuBuf),
      ]),
      scanBlocks: bg(this.scanBlocksPipe as GpuPipeline, [
        b(0, this.blockSum as GpuBuf),
        b(1, this.blockBase as GpuBuf),
      ]),
      scanDown: bg(this.scanDownPipe as GpuPipeline, [
        b(0, hist),
        b(1, this.blockBase as GpuBuf),
        b(2, base),
        b(3, this.scanParamsBuf as GpuBuf),
      ]),
    };
  }

  /** Record + submit the whole sort as one command buffer. */
  run(): void {
    if (!this.groups || this.n === 0) {
      return;
    }
    const d = this.device;
    const enc = d.createCommandEncoder();
    const tiles = this.numTiles;
    for (let p = 0; p < RADIX_PASSES; p++) {
      const pass = enc.beginComputePass();
      // count
      pass.setPipeline(this.countPipe[p]);
      pass.setBindGroup(0, this.groups.count[p]);
      pass.dispatchWorkgroups(tiles);
      pass.end();

      const sr = enc.beginComputePass();
      sr.setPipeline(this.scanReducePipe as GpuPipeline);
      sr.setBindGroup(0, this.groups.scanReduce);
      sr.dispatchWorkgroups(Math.ceil(SCAN_BLOCKS / 64));
      sr.end();

      const sb = enc.beginComputePass();
      sb.setPipeline(this.scanBlocksPipe as GpuPipeline);
      sb.setBindGroup(0, this.groups.scanBlocks);
      sb.dispatchWorkgroups(1);
      sb.end();

      const sd = enc.beginComputePass();
      sd.setPipeline(this.scanDownPipe as GpuPipeline);
      sd.setBindGroup(0, this.groups.scanDown);
      sd.dispatchWorkgroups(Math.ceil(SCAN_BLOCKS / 64));
      sd.end();

      const sc = enc.beginComputePass();
      sc.setPipeline(this.scatterPipe[p]);
      sc.setBindGroup(0, this.groups.scatter[p]);
      sc.dispatchWorkgroups(tiles);
      sc.end();
    }
    d.queue.submit([enc.finish()]);
  }

  dispose(): void {
    for (const b of [
      this.keyB,
      this.idxB,
      this.hist,
      this.base,
      this.blockSum,
      this.blockBase,
      this.paramsBuf,
      this.scanParamsBuf,
    ]) {
      b?.destroy();
    }
    this.keyB = this.idxB = this.hist = this.base = null;
    this.blockSum = this.blockBase = this.paramsBuf = this.scanParamsBuf = null;
    this.groups = null;
  }
}
