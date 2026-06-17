export type SparkBackendKind = "webgl2" | "webgpu";
/**
 * Returns true if the current environment exposes the WebGPU API. This only
 * checks for the presence of `navigator.gpu`; a full check that an adapter can
 * actually be requested happens during {@link WebGPUSplatRenderer} init, which
 * falls back automatically if adapter/device acquisition fails.
 */
export declare function isWebGPUAvailable(): boolean;
/**
 * Asynchronously confirms a WebGPU adapter can be acquired. Use this for a
 * stronger guarantee than {@link isWebGPUAvailable} before committing to the
 * WebGPU backend. Returns false (rather than throwing) on any failure.
 */
export declare function canRequestWebGPUAdapter(): Promise<boolean>;
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
export declare function requestWebGPUStorageLimits(): Promise<{
    maxStorageBufferBindingSize: number;
    maxBufferSize: number;
} | undefined>;
/**
 * Resolves which backend to use given an optional forced override (e.g. from a
 * `?sparkBackend=` URL param in examples/tests) and runtime capability.
 *
 * - `force === "webgl2"` always returns `"webgl2"`.
 * - `force === "webgpu"` returns `"webgpu"` even if detection is uncertain, so
 *   tests can exercise the path; init will still fall back on hard failure.
 * - otherwise returns `"webgpu"` iff an adapter can be acquired.
 */
export declare function resolveBackend(force?: SparkBackendKind): Promise<SparkBackendKind>;
/**
 * Reads a forced-backend override from the current URL's query string
 * (`?sparkBackend=webgpu|webgl2`). Returns undefined if not present/invalid.
 * Used by examples to flip backends for visual comparison and by the Phase 5
 * validation harness.
 */
export declare function backendFromUrl(): SparkBackendKind | undefined;
