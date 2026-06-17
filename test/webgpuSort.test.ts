import assert from "node:assert";

// CPU mirror of the single-pass GPU counting sort in
// src/webgpu/WebGPUSplatRenderer.ts (buildPipeline): histogram -> exclusive
// prefix sum -> scatter by the bucket offset. Validates the algorithm the
// count/scan/scatter compute kernels implement.

function countingSort(
  keys: Uint32Array,
  buckets: number,
): { order: Uint32Array } {
  const n = keys.length;
  const hist = new Uint32Array(buckets);
  for (let i = 0; i < n; i++) hist[keys[i]]++;
  // exclusive prefix sum -> first write position per bucket
  const offset = new Uint32Array(buckets);
  let running = 0;
  for (let b = 0; b < buckets; b++) {
    offset[b] = running;
    running += hist[b];
  }
  // scatter
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const b = keys[i];
    order[offset[b]++] = i;
  }
  return { order };
}

function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s;
  };
}

function check(keys: Uint32Array, buckets: number, label: string): void {
  const { order } = countingSort(keys, buckets);
  const n = keys.length;
  // permutation of [0, n)
  const seen = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    assert.ok(order[i] < n, `${label}: order out of range`);
    assert.ok(!seen[order[i]], `${label}: duplicate in order`);
    seen[order[i]] = 1;
  }
  // keys are non-decreasing along the sorted order (correct ascending sort)
  for (let i = 1; i < n; i++) {
    assert.ok(
      keys[order[i]] >= keys[order[i - 1]],
      `${label}: not sorted ascending at ${i}`,
    );
  }
}

for (const n of [1, 255, 256, 257, 1000, 5000]) {
  const rng = makeRng(n + 1);
  const a = new Uint32Array(n);
  for (let i = 0; i < n; i++) a[i] = rng() % 256;
  check(a, 256, `random n=${n}`);
}
// heavy ties + degenerate
check(new Uint32Array(4096).fill(7), 256, "all-equal");
check(
  Uint32Array.from({ length: 1000 }, (_, i) => i % 256),
  256,
  "sorted-mod",
);
{
  // sentinel bucket (BUCKETS-1) must sort to the very end
  const N = 1000;
  const B = 256;
  const rng = makeRng(99);
  const a = new Uint32Array(N);
  let sentinels = 0;
  for (let i = 0; i < N; i++) {
    if (rng() % 4 === 0) {
      a[i] = B - 1;
      sentinels++;
    } else {
      a[i] = rng() % (B - 1);
    }
  }
  const { order } = countingSort(a, B);
  for (let i = 0; i < sentinels; i++) {
    assert.strictEqual(a[order[N - 1 - i]], B - 1, "sentinels not at the end");
  }
}

// 16-bit depth key (mirror of WGSL: depthKey(d) >> 16). Larger depth -> smaller
// key, so the ascending counting sort draws far -> near.
function depthKey16(depth: number): number {
  const u = new Uint32Array(new Float32Array([depth]).buffer)[0];
  const mask = u & 0x80000000 ? 0xffffffff : 0x80000000;
  return (~(u ^ mask) >>> 16) & 0xffff;
}
{
  const depths = [0, 0.001, 0.5, 1, 2, 10, 100, 1000, 1e5];
  for (let i = 1; i < depths.length; i++) {
    assert.ok(
      depthKey16(depths[i]) <= depthKey16(depths[i - 1]),
      `depthKey16 not non-increasing with depth at ${depths[i]}`,
    );
  }
  // active keys must stay below the inactive sentinel (0xffff)
  for (const d of depths) {
    assert.ok(depthKey16(d) < 0xffff, "active key collides with sentinel");
  }
}

console.log("✅ All WebGPU sort test cases passed!");
