// Hand-written WGSL helper functions used by the WebGPU backend's compute pass.
//
// These are intentionally small, single-function, single-return WGSL snippets
// with no inter-function calls and no struct types. THREE's `wgslFn` parser
// (WGSLNodeFunction) treats the first `fn` in the source as the entry point and
// emits everything after its signature as the body, so multi-function / struct
// sources are fragile. Keeping each helper to one self-contained function keeps
// us firmly on the supported path.
//
// Bit layout matches Spark's 16-byte packed splat exactly (see
// `src/shaders/splatDefines.glsl` `unpackSplatEncoding`), so the WebGPU and
// WebGL2 paths decode identical bits (directive D7). `unpack2x16float` is the
// WGSL builtin equivalent of GLSL `unpackHalf2x16`.

/** Decode the 3 float16 center components (words 1 and low-16 of word 2). */
export const WGSL_UNPACK_CENTER = /* wgsl */ `
fn sparkUnpackCenter( packed : vec4<u32> ) -> vec3<f32> {
  let cxy = unpack2x16float( packed.y );
  let cz = unpack2x16float( packed.z & 0xffffu ).x;
  return vec3<f32>( cxy.x, cxy.y, cz );
}
`;

/** Decode the uint8 RGBA color/opacity (word 0) into [0,1] floats. */
export const WGSL_UNPACK_RGBA = /* wgsl */ `
fn sparkUnpackRgba( packed : vec4<u32> ) -> vec4<f32> {
  let w0 = packed.x;
  return vec4<f32>(
    f32( w0 & 0xffu ),
    f32( ( w0 >> 8u ) & 0xffu ),
    f32( ( w0 >> 16u ) & 0xffu ),
    f32( ( w0 >> 24u ) & 0xffu )
  ) / 255.0;
}
`;

/**
 * Decode the largest of the three log-space uint8 scales (word 3). A code of 0
 * means a flat (2DGS) axis -> 0.0, matching `LN_SCALE_MIN..LN_SCALE_MAX`
 * (-12..9) from `src/defines.ts`. Phase 2's isotropic projection only needs the
 * maximum extent; full anisotropic covariance lands in Phase 4.
 */
export const WGSL_UNPACK_MAX_SCALE = /* wgsl */ `
fn sparkUnpackMaxScale( packed : vec4<u32> ) -> f32 {
  let w3 = packed.w;
  let sx = w3 & 0xffu;
  let sy = ( w3 >> 8u ) & 0xffu;
  let sz = ( w3 >> 16u ) & 0xffu;
  let lnMin = -12.0;
  let lnStep = ( 9.0 - ( -12.0 ) ) / 254.0;
  let fx = select( exp( lnMin + f32( sx - 1u ) * lnStep ), 0.0, sx == 0u );
  let fy = select( exp( lnMin + f32( sy - 1u ) * lnStep ), 0.0, sy == 0u );
  let fz = select( exp( lnMin + f32( sz - 1u ) * lnStep ), 0.0, sz == 0u );
  return max( fx, max( fy, fz ) );
}
`;

/**
 * Map a non-negative depth (radial camera distance) to a 32-bit sort key such
 * that ascending integer sort orders splats **far -> near** (painter's order).
 *
 * The classic float->sortable-uint flip (`bits ^ 0x80000000` for f >= 0) is
 * monotonically increasing in depth; bit-inverting it makes the key decrease
 * with depth, so the radix sort's ascending output draws farthest first.
 */
export const WGSL_DEPTH_KEY = /* wgsl */ `
fn sparkDepthKey( depth : f32 ) -> u32 {
  let u = bitcast<u32>( depth );
  let mask = select( 0x80000000u, 0xffffffffu, ( u & 0x80000000u ) != 0u );
  return ~( u ^ mask );
}
`;
