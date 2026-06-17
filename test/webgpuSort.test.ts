import assert from "node:assert";

// CPU mirror of the stable tiled radix sort in
// src/webgpu/WebGPUSplatRenderer.ts (buildPipeline): per-tile digit-major
// histogram -> exclusive prefix sum -> stable scatter, ping-pong over
// RADIX_PASSES passes of RADIX_BITS each. Validates the algorithm the
// count/scan/scatter compute kernels implement, including STABILITY (the draw
// order of equal-depth splats must be deterministic, or alpha blending of
// overlapping splats flickers frame to frame).

const TILE = 256;
const RADIX = 256;
const RADIX_BITS = 8;
const RADIX_PASSES = 2; // 16-bit key

function radixSort(input: Uint32Array): {
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

  for (let p = 0; p < RADIX_PASSES; p++) {
    const shift = p * RADIX_BITS;
    // count: per-tile histogram, digit-major
    hist.fill(0);
    for (let t = 0; t < numTiles; t++) {
      const local = new Uint32Array(RADIX);
      for (let j = 0; j < TILE; j++) {
        const e = t * TILE + j;
        if (e < N) local[(inKey[e] >>> shift) & (RADIX - 1)]++;
      }
      for (let d = 0; d < RADIX; d++) hist[d * numTiles + t] = local[d];
    }
    // exclusive prefix sum
    let running = 0;
    for (let m = 0; m < histLen; m++) {
      base[m] = running >>> 0;
      running = (running + hist[m]) >>> 0;
    }
    // stable scatter (each tile emits in element order)
    for (let t = 0; t < numTiles; t++) {
      const offset = new Uint32Array(RADIX);
      for (let d = 0; d < RADIX; d++) offset[d] = base[d * numTiles + t];
      for (let j = 0; j < TILE; j++) {
        const e = t * TILE + j;
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

function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s;
  };
}

function check(input: Uint32Array, label: string): void {
  const { keys, order } = radixSort(input);
  const N = input.length;
  // permutation + ascending + key/order consistency
  const seen = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    assert.ok(order[i] < N, `${label}: order out of range`);
    assert.ok(!seen[order[i]], `${label}: duplicate in order`);
    seen[order[i]] = 1;
    assert.strictEqual(
      keys[i],
      input[order[i]],
      `${label}: key/order mismatch`,
    );
    if (i > 0) {
      assert.ok(keys[i] >= keys[i - 1], `${label}: not ascending at ${i}`);
    }
  }
  // STABILITY: equal keys keep original index order
  const ref = Array.from({ length: N }, (_, i) => i).sort(
    (a, b) => input[a] - input[b] || a - b,
  );
  for (let i = 0; i < N; i++) {
    assert.strictEqual(order[i], ref[i], `${label}: not a stable sort at ${i}`);
  }
}

for (const N of [1, 255, 256, 257, 1000, 5000]) {
  const rng = makeRng(N + 1);
  const a = new Uint32Array(N);
  for (let i = 0; i < N; i++) a[i] = rng() & 0xffff; // 16-bit keys
  check(a, `random n=${N}`);
}
// heavy ties stress stability
{
  const N = 4096;
  const rng = makeRng(7);
  const a = new Uint32Array(N);
  for (let i = 0; i < N; i++) a[i] = rng() % 16;
  check(a, "many-ties");
}
check(new Uint32Array(2048).fill(42), "all-equal");
// inactive sentinel (0xffff) sorts to the end
{
  const N = 1000;
  const rng = makeRng(99);
  const a = new Uint32Array(N);
  let sentinels = 0;
  for (let i = 0; i < N; i++) {
    if (rng() % 4 === 0) {
      a[i] = 0xffff;
      sentinels++;
    } else {
      a[i] = rng() % 0xffff;
    }
  }
  const { keys } = radixSort(a);
  for (let i = 0; i < sentinels; i++) {
    assert.strictEqual(keys[N - 1 - i], 0xffff, "sentinels not at the end");
  }
}

// 16-bit depth key (mirror of WGSL sparkDepthKey16): radial distance normalized
// to the model's [min,max] range. Larger distance -> smaller key, so the
// ascending sort draws far -> near. 0xffff is reserved for the inactive sentinel.
function depthKey16(dist: number, minDist: number, maxDist: number): number {
  const t = Math.min(
    1,
    Math.max(0, (dist - minDist) / Math.max(maxDist - minDist, 1e-6)),
  );
  return Math.round((1 - t) * 65534);
}
{
  // A tightly-clustered model (the case the raw float->top-16 key handled badly):
  // distances spanning a narrow band must still spread across the full key range.
  const minD = 2.5;
  const maxD = 3.5;
  const depths = [2.5, 2.6, 2.9, 3.0, 3.1, 3.4, 3.5];
  for (let i = 1; i < depths.length; i++) {
    assert.ok(
      depthKey16(depths[i], minD, maxD) < depthKey16(depths[i - 1], minD, maxD),
      `depthKey16 not strictly decreasing with depth at ${depths[i]}`,
    );
  }
  // Full-range spread: nearest -> ~0xfffe, farthest -> 0, all below the sentinel.
  assert.strictEqual(depthKey16(minD, minD, maxD), 65534, "near key not maxed");
  assert.strictEqual(depthKey16(maxD, minD, maxD), 0, "far key not zeroed");
  for (const d of depths) {
    assert.ok(depthKey16(d, minD, maxD) < 0xffff, "active key hits sentinel");
  }
  // Out-of-range distances clamp instead of wrapping.
  assert.strictEqual(
    depthKey16(1.0, minD, maxD),
    65534,
    "below-min not clamped",
  );
  assert.strictEqual(depthKey16(9.0, minD, maxD), 0, "above-max not clamped");
}

console.log("✅ All WebGPU sort test cases passed!");
