import assert from "node:assert";
import {
  backendFromUrl,
  canRequestWebGPUAdapter,
  isWebGPUAvailable,
  resolveBackend,
} from "../src/webgpu/capability.js";

// Runs under Node, where there is no `navigator`/`navigator.gpu` and no
// `location` — i.e. the "WebGPU unavailable" environment. This validates the
// automatic fallback decision logic without a browser.

// 1. With no navigator.gpu, detection must report unavailable.
assert.strictEqual(
  isWebGPUAvailable(),
  false,
  "isWebGPUAvailable should be false when navigator.gpu is absent",
);
assert.strictEqual(
  await canRequestWebGPUAdapter(),
  false,
  "canRequestWebGPUAdapter should be false (and not throw) without WebGPU",
);

// 2. Unforced resolution falls back to webgl2 when WebGPU is unavailable.
assert.strictEqual(
  await resolveBackend(),
  "webgl2",
  "resolveBackend() should fall back to webgl2 without WebGPU",
);

// 3. Forcing overrides detection (used by tests / the validation harness).
assert.strictEqual(
  await resolveBackend("webgl2"),
  "webgl2",
  "forceBackend webgl2 should win",
);
assert.strictEqual(
  await resolveBackend("webgpu"),
  "webgpu",
  "forceBackend webgpu should win even when detection is uncertain",
);

// 4. Without a `location`, the URL override is undefined (no throw).
assert.strictEqual(
  backendFromUrl(),
  undefined,
  "backendFromUrl should be undefined when location is absent",
);

console.log("✅ All WebGPU fallback test cases passed!");
