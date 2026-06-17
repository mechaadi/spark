// Runtime WebGPU capability detection for the Spark WebGPU backend.
//
// This module is intentionally dependency-free (no `three`, no `three/webgpu`)
// so it can be imported from the default WebGL build without pulling the WebGPU
// runtime into the static import graph. The actual WebGPU code in
// `WebGPUSplatRenderer` is loaded via dynamic import only when selected, so the
// ~15% of devices without WebGPU never fetch `three/webgpu` / `three/tsl`.

export type SparkBackendKind = "webgl2" | "webgpu";

/**
 * Returns true if the current environment exposes the WebGPU API. This only
 * checks for the presence of `navigator.gpu`; a full check that an adapter can
 * actually be requested happens during {@link WebGPUSplatRenderer} init, which
 * falls back automatically if adapter/device acquisition fails.
 */
export function isWebGPUAvailable(): boolean {
  return (
    typeof navigator !== "undefined" &&
    typeof (navigator as Navigator & { gpu?: unknown }).gpu === "object" &&
    (navigator as Navigator & { gpu?: unknown }).gpu != null
  );
}

/**
 * Asynchronously confirms a WebGPU adapter can be acquired. Use this for a
 * stronger guarantee than {@link isWebGPUAvailable} before committing to the
 * WebGPU backend. Returns false (rather than throwing) on any failure.
 */
export async function canRequestWebGPUAdapter(): Promise<boolean> {
  if (!isWebGPUAvailable()) {
    return false;
  }
  try {
    const gpu = (navigator as Navigator & { gpu: GPU }).gpu;
    const adapter = await gpu.requestAdapter();
    return adapter != null;
  } catch {
    return false;
  }
}

/**
 * Requests a WebGPU adapter and returns its storage-buffer limits, for use as
 * `requiredLimits` when constructing the THREE `WebGPURenderer`.
 *
 * The WebGPU default `maxStorageBufferBindingSize` is only 128 MiB, but Spark's
 * per-splat projection buffer is 64 bytes/splat, so a scene past ~2M splats
 * exceeds it and bind-group creation fails ("Binding size … is larger than the
 * maximum storage buffer binding size"). THREE forwards `requiredLimits`
 * verbatim to `requestDevice`, so the requested values must come from a real
 * adapter (asking for more than it supports would fail the device request).
 * Returns the adapter's true maxima, or undefined if WebGPU/the adapter is
 * unavailable.
 */
export async function requestWebGPUStorageLimits(): Promise<
  { maxStorageBufferBindingSize: number; maxBufferSize: number } | undefined
> {
  if (!isWebGPUAvailable()) {
    return undefined;
  }
  try {
    const gpu = (navigator as Navigator & { gpu: GPU }).gpu;
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      return undefined;
    }
    const { maxStorageBufferBindingSize, maxBufferSize } = adapter.limits;
    return { maxStorageBufferBindingSize, maxBufferSize };
  } catch {
    return undefined;
  }
}

/**
 * Resolves which backend to use given an optional forced override (e.g. from a
 * `?sparkBackend=` URL param in examples/tests) and runtime capability.
 *
 * - `force === "webgl2"` always returns `"webgl2"`.
 * - `force === "webgpu"` returns `"webgpu"` even if detection is uncertain, so
 *   tests can exercise the path; init will still fall back on hard failure.
 * - otherwise returns `"webgpu"` iff an adapter can be acquired.
 */
export async function resolveBackend(
  force?: SparkBackendKind,
): Promise<SparkBackendKind> {
  if (force === "webgl2") {
    return "webgl2";
  }
  if (force === "webgpu") {
    return "webgpu";
  }
  return (await canRequestWebGPUAdapter()) ? "webgpu" : "webgl2";
}

/**
 * Reads a forced-backend override from the current URL's query string
 * (`?sparkBackend=webgpu|webgl2`). Returns undefined if not present/invalid.
 * Used by examples to flip backends for visual comparison and by the Phase 5
 * validation harness.
 */
export function backendFromUrl(): SparkBackendKind | undefined {
  if (typeof location === "undefined") {
    return undefined;
  }
  const value = new URLSearchParams(location.search).get("sparkBackend");
  return value === "webgpu" || value === "webgl2" ? value : undefined;
}
