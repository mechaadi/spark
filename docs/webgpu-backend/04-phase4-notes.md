# Phase 4 — GPU cull + projection + SH (visual parity)

Goal: move frustum culling, full anisotropic projection, and SH evaluation into
the compute pass so the vertex/fragment stay lightweight and the WebGPU output
matches the WebGL path. Landed in two parts.

## 4A — Anisotropic 2D-covariance projection

Replaced the Phase-2 isotropic (round) projection with the full anisotropic
covariance projection in the compute pass — a port of the standard-splat path in
`splatVertex.glsl`:

- Build the view-space 3D covariance `(M3·R·S)(M3·R·S)^T` from the unpacked
  scale + quaternion (`M3` = upper-left 3x3 of model-view).
- Project through the perspective Jacobian to a 2D covariance.
- Eigen-decompose into screen-space ellipse axes, with the 0.5px Gaussian
  anti-alias blur and the matching opacity (intensity) compensation.

The covariance + Jacobian math lives in one self-contained WGSL helper
(`sparkCov2D`, returns `(a, d, b)`), avoiding TSL matrix-construction typing
limits. The raster **vertex/fragment are unchanged** — they already consume the
projected `axis1`/`axis2` and `stdDev` from the projected buffer, so Phase 4A
only rewrote the compute projection.

Result: round "beady" splats became sharp anisotropic ellipses, matching WebGL
shape detail (clean wing edges, sharp eyespots, fine streaks).

## 4B — Spherical harmonics (SH ≤ 3), per directive D3

View-dependent color evaluated per splat in the project compute pass — a 1:1
port of `defineEvalPackedSH{1,2,3}` from `PackedSplats.ts`:

- Upload whichever SH bands the mesh carries
  (`packed.extra.sh1/sh2/sh3`) as storage buffers — SH1 = 2×u32 (`uvec2`,
  RG32UI-equivalent), SH2/SH3 = 4×u32 (`uvec4`, RGBA32UI), bit-identical.
- Sign-extend the packed `sint7` / `sint8` / `sint6` coefficients via
  `bitcast<i32>` + arithmetic shift, scaled by `shMax/{63,127,31}`.
- Evaluate the SH basis with the **object-space** camera→splat direction
  (`normalize(center − camPosObject)`, where `camPosObject = inverse(meshWorld) ·
  camWorldPos`) and add to the base (DC) color.
- Bands are gated at build time, so a mesh with only SH1 emits no SH2/SH3 code.

Result: the iridescent, view-dependent color (the magenta/cyan/teal shimmer) is
restored, reaching close visual parity with the WebGL path.

## Verification

- ✅ `tsc` + `biome` clean; existing tests pass.
- ✅ On real WebGPU hardware (`examples/webgpu-hello/`): the butterfly now renders
  with sharp anisotropic splats and SH iridescence, closely matching the WebGL
  `hello-world` example side by side. Zero WebGPU validation errors. (Direct A/B
  screenshots captured during development.)

## Known residual (for Phase 5 parity tuning)

- A slight saturation/brightness difference remains vs WebGL. This is
  **color-space / tone-mapping**, not the splat math: `WebGPURenderer` applies
  its own output color-space transform and tone mapping, which differ from the
  classic `WebGLRenderer` + Spark fragment's `encodeLinear`/sRGB handling. The
  Phase 5 perceptual-parity pass (threshold diff, not bit-exact per D8) will tune
  output color space / tone mapping to match, and add the `srgbToLinear` step the
  WebGL fragment applies when `encodeLinear` is set.
- Exact-pose A/B comparison is approximate here because both examples auto-rotate;
  the Phase 5 harness drives an identical fixed camera path on both backends.

## Compute pipeline now (per frame, in submission order)

`project (cull + anisotropic covariance + SH + depth key)` → `radix sort
(count → scan → scatter) × 4` → `renderer.render` (instanced raster reads sorted
`idxA` → projected ellipse). All in-frame, no readback.

## Still deferred

- ExtSplats / Paged / dyno residency route to the WebGL2 session (D4/D5/D6).
- 2DGS (flat-axis) splats and the focal/aperture depth-of-field path are not
  ported (standard splats only); they can be added if needed.
- Radix sort workgroup-shared optimization (Phase 5 perf).
