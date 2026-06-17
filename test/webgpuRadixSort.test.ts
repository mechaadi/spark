import assert from "node:assert";

// Faithful CPU mirror of the GPU LSD radix sort in
// src/webgpu/WebGPUSplatRenderer.ts (buildPipeline): one tile per invocation,
// digit-major per-tile histograms, a single exclusive prefix sum, and a stable
// scatter. This test validates the ALGORITHM the WGSL/TSL transcribes — the same
// tiling, the same digit-major layout, the same 4 x 8-bit passes.

const TILE = 256;
const RADIX = 256;
const PASSES = 4;

function radixSortTiled(input: Uint32Array): {
  keys: Uint32Array;
  order: Uint32Array;
} {
  const N = input.length;
  const numTiles = Math.ceil(N / TILE);
  const histLen = numTiles * RADIX;

  let inKey = Uint32Array.from(input);
  let outKey = new Uint32Array(N);
  let inIdx = new Uint32Array(N);
  for (let i = 0; i < N; i++) inIdx[i] = i;
  let outIdx = new Uint32Array(N);

  const hist = new Uint32Array(histLen);
  const base = new Uint32Array(histLen);

  for (let p = 0; p < PASSES; p++) {
    const shift = p * 8;

    // count: per-tile histogram, written digit-major (digit * numTiles + tile)
    hist.fill(0);
    for (let t = 0; t < numTiles; t++) {
      const local = new Uint32Array(RADIX);
      const start = t * TILE;
      for (let j = 0; j < TILE; j++) {
        const e = start + j;
        if (e < N) local[(inKey[e] >>> shift) & (RADIX - 1)]++;
      }
      for (let d = 0; d < RADIX; d++) hist[d * numTiles + t] = local[d];
    }

    // scan: single exclusive prefix sum over the whole digit-major histogram
    let running = 0;
    for (let m = 0; m < histLen; m++) {
      base[m] = running >>> 0;
      running = (running + hist[m]) >>> 0;
    }

    // scatter: stable, each tile emits to its running global offset per digit
    for (let t = 0; t < numTiles; t++) {
      const offset = new Uint32Array(RADIX);
      for (let d = 0; d < RADIX; d++) offset[d] = base[d * numTiles + t];
      const start = t * TILE;
      for (let j = 0; j < TILE; j++) {
        const e = start + j;
        if (e < N) {
          const key = inKey[e];
          const d = (key >>> shift) & (RADIX - 1);
          const pos = offset[d]++;
          outKey[pos] = key;
          outIdx[pos] = inIdx[e];
        }
      }
    }

    [inKey, outKey] = [outKey, inKey];
    [inIdx, outIdx] = [outIdx, inIdx];
  }
  return { keys: inKey, order: inIdx };
}

// Deterministic LCG so the test is reproducible.
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s;
  };
}

function checkSorted(input: Uint32Array, label: string): void {
  const { keys, order } = radixSortTiled(input);
  const N = input.length;

  // 1. keys ascending (non-decreasing)
  for (let i = 1; i < N; i++) {
    assert.ok(
      keys[i] >= keys[i - 1],
      `${label}: keys not ascending at ${i}: ${keys[i - 1]} > ${keys[i]}`,
    );
  }

  // 2. order is a permutation of [0, N) and keys[i] === input[order[i]]
  const seen = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    const v = order[i];
    assert.ok(v < N, `${label}: order out of bounds: ${v}`);
    assert.ok(!seen[v], `${label}: order has a duplicate index: ${v}`);
    seen[v] = 1;
    assert.strictEqual(
      keys[i],
      input[v],
      `${label}: key/order mismatch at ${i}`,
    );
  }

  // 3. stability: ties keep original index order (LSD radix must be stable)
  const reference = Array.from({ length: N }, (_, i) => i).sort(
    (a, b) => input[a] - input[b] || a - b,
  );
  for (let i = 0; i < N; i++) {
    assert.strictEqual(
      order[i],
      reference[i],
      `${label}: not a stable sort at ${i}`,
    );
  }
}

// Random arrays across sizes that exercise tile boundaries.
for (const N of [1, 255, 256, 257, 1000, 1024, 5000]) {
  const rng = makeRng(N + 1);
  const a = new Uint32Array(N);
  for (let i = 0; i < N; i++) a[i] = rng();
  checkSorted(a, `random N=${N}`);
}

// Heavy ties (only a few distinct keys) — stresses stability and bucketing.
{
  const N = 4096;
  const rng = makeRng(7);
  const a = new Uint32Array(N);
  for (let i = 0; i < N; i++) a[i] = rng() % 16;
  checkSorted(a, "many-ties");
}

// Degenerate inputs.
checkSorted(new Uint32Array(2048).fill(42), "all-equal");
checkSorted(
  Uint32Array.from({ length: 1000 }, (_, i) => i),
  "already-sorted",
);
checkSorted(
  Uint32Array.from({ length: 1000 }, (_, i) => 999 - i),
  "reversed",
);

// Inactive sentinels (0xffffffff) must sort to the very end.
{
  const N = 1000;
  const rng = makeRng(99);
  const a = new Uint32Array(N);
  let sentinels = 0;
  for (let i = 0; i < N; i++) {
    if (rng() % 3 === 0) {
      a[i] = 0xffffffff;
      sentinels++;
    } else {
      a[i] = rng() % 100000;
    }
  }
  const { keys } = radixSortTiled(a);
  for (let i = 0; i < sentinels; i++) {
    assert.strictEqual(
      keys[N - 1 - i],
      0xffffffff,
      "sentinels did not sort to the end",
    );
  }
  checkSorted(a, "with-sentinels");
}

// Depth-key mapping (mirror of WGSL sparkDepthKey): larger depth -> smaller key,
// so an ascending radix sort draws far -> near.
function depthKey(depth: number): number {
  const u = new Uint32Array(new Float32Array([depth]).buffer)[0];
  const mask = u & 0x80000000 ? 0xffffffff : 0x80000000;
  return ~(u ^ mask) >>> 0;
}
{
  const depths = [0, 0.001, 0.5, 1, 2, 10, 100, 1000, 1e6];
  for (let i = 1; i < depths.length; i++) {
    assert.ok(
      depthKey(depths[i]) < depthKey(depths[i - 1]),
      `depthKey not monotonically decreasing at depth ${depths[i]}`,
    );
  }
}

console.log("✅ All WebGPU radix sort test cases passed!");
