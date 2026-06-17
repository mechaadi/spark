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

/** Decode the three log-space uint8 scales (word 3) into a vec3. */
export const WGSL_UNPACK_SCALES = /* wgsl */ `
fn sparkUnpackScales( packed : vec4<u32> ) -> vec3<f32> {
  let w3 = packed.w;
  let sx = w3 & 0xffu;
  let sy = ( w3 >> 8u ) & 0xffu;
  let sz = ( w3 >> 16u ) & 0xffu;
  let lnMin = -12.0;
  let lnStep = ( 9.0 - ( -12.0 ) ) / 254.0;
  return vec3<f32>(
    select( exp( lnMin + f32( sx - 1u ) * lnStep ), 0.0, sx == 0u ),
    select( exp( lnMin + f32( sy - 1u ) * lnStep ), 0.0, sy == 0u ),
    select( exp( lnMin + f32( sz - 1u ) * lnStep ), 0.0, sz == 0u )
  );
}
`;

/**
 * Decode the splat rotation quaternion (24-bit octahedral-XY-88 + 8-bit angle),
 * a port of `decodeQuatOctXy88R8` in splatDefines.glsl. The 24-bit code lives in
 * word 2 bits [16:32] and word 3 bits [24:32] of the packed splat.
 */
export const WGSL_UNPACK_QUAT = /* wgsl */ `
fn sparkUnpackQuat( packed : vec4<u32> ) -> vec4<f32> {
  let encoded = ( ( packed.z >> 16u ) & 0xffffu ) | ( ( packed.w >> 8u ) & 0xff0000u );
  let quantU = encoded & 0xffu;
  let quantV = ( encoded >> 8u ) & 0xffu;
  let angleInt = encoded >> 16u;
  let f = vec2<f32>( f32( quantU ) / 255.0 * 2.0 - 1.0, f32( quantV ) / 255.0 * 2.0 - 1.0 );
  var axis = vec3<f32>( f.x, f.y, 1.0 - abs( f.x ) - abs( f.y ) );
  let t = max( -axis.z, 0.0 );
  axis.x = axis.x + select( t, -t, axis.x >= 0.0 );
  axis.y = axis.y + select( t, -t, axis.y >= 0.0 );
  axis = normalize( axis );
  let theta = ( f32( angleInt ) / 255.0 ) * 3.14159265359;
  return vec4<f32>( axis * sin( theta * 0.5 ), cos( theta * 0.5 ) );
}
`;

/**
 * Compute the projected 2D covariance of a splat, returned as `(a, d, b)` for
 * the symmetric matrix `[[a, b], [b, d]]`. Port of the covariance + Jacobian
 * projection in splatVertex.glsl: build the object-space scale-rotate matrix
 * (inlined `scaleQuaternionToMatrix`), rotate into view space by `mv3` (the
 * upper-left 3x3 of the model-view matrix), form the 3D covariance, then project
 * through the perspective Jacobian. Self-contained (no helper calls / structs).
 */
export const WGSL_COV2D = /* wgsl */ `
fn sparkCov2D( mv3 : mat3x3<f32>, s : vec3<f32>, q : vec4<f32>, focal : vec2<f32>, viewC : vec3<f32> ) -> vec3<f32> {
  let rs = mat3x3<f32>(
    vec3<f32>( s.x * ( 1.0 - 2.0 * ( q.y * q.y + q.z * q.z ) ), s.x * ( 2.0 * ( q.x * q.y + q.w * q.z ) ), s.x * ( 2.0 * ( q.x * q.z - q.w * q.y ) ) ),
    vec3<f32>( s.y * ( 2.0 * ( q.x * q.y - q.w * q.z ) ), s.y * ( 1.0 - 2.0 * ( q.x * q.x + q.z * q.z ) ), s.y * ( 2.0 * ( q.y * q.z + q.w * q.x ) ) ),
    vec3<f32>( s.z * ( 2.0 * ( q.x * q.z + q.w * q.y ) ), s.z * ( 2.0 * ( q.y * q.z - q.w * q.x ) ), s.z * ( 1.0 - 2.0 * ( q.x * q.x + q.y * q.y ) ) )
  );
  let viewRS = mv3 * rs;
  let cov3D = viewRS * transpose( viewRS );
  let invZ = 1.0 / viewC.z;
  let j1 = focal * invZ;
  let j2 = -( j1 * viewC.xy ) * invZ;
  let jac = mat3x3<f32>(
    vec3<f32>( j1.x, 0.0, j2.x ),
    vec3<f32>( 0.0, j1.y, j2.y ),
    vec3<f32>( 0.0, 0.0, 0.0 )
  );
  let cov2D = transpose( jac ) * cov3D * jac;
  // (a, d, b) for [[a, b], [b, d]]; column-major so cov2D[0][1] is the off-diagonal.
  return vec3<f32>( cov2D[0][0], cov2D[1][1], cov2D[0][1] );
}
`;

// Spherical-harmonics evaluation, ported 1:1 from defineEvalPackedSH{1,2,3} in
// PackedSplats.ts. `dir` is the object-space direction from camera to splat.
// Signed coefficients are sign-extended via bitcast + arithmetic shift.

/** SH band 1 (3 coeffs, sint7 in 2x u32), scaled by sh1Max/63. */
export const WGSL_EVAL_SH1 = /* wgsl */ `
fn sparkEvalSH1( d : vec2<u32>, dir : vec3<f32>, shMax : f32 ) -> vec3<f32> {
  let x = d.x; let y = d.y;
  let c0 = vec3<f32>( f32( bitcast<i32>( x << 25u ) >> 25u ), f32( bitcast<i32>( x << 18u ) >> 25u ), f32( bitcast<i32>( x << 11u ) >> 25u ) );
  let c1 = vec3<f32>( f32( bitcast<i32>( x << 4u ) >> 25u ), f32( bitcast<i32>( ( x >> 3u ) | ( y << 29u ) ) >> 25u ), f32( bitcast<i32>( y << 22u ) >> 25u ) );
  let c2 = vec3<f32>( f32( bitcast<i32>( y << 15u ) >> 25u ), f32( bitcast<i32>( y << 8u ) >> 25u ), f32( bitcast<i32>( y << 1u ) >> 25u ) );
  let rgb = c0 * ( -0.4886025 * dir.y ) + c1 * ( 0.4886025 * dir.z ) + c2 * ( -0.4886025 * dir.x );
  return rgb * ( shMax / 63.0 );
}
`;

/** SH band 2 (5 coeffs, sint8 in 4x u32), scaled by sh2Max/127. */
export const WGSL_EVAL_SH2 = /* wgsl */ `
fn sparkEvalSH2( d : vec4<u32>, dir : vec3<f32>, shMax : f32 ) -> vec3<f32> {
  let x = d.x; let y = d.y; let z = d.z; let w = d.w;
  let c0 = vec3<f32>( f32( bitcast<i32>( x << 24u ) >> 24u ), f32( bitcast<i32>( x << 16u ) >> 24u ), f32( bitcast<i32>( x << 8u ) >> 24u ) );
  let c1 = vec3<f32>( f32( bitcast<i32>( x ) >> 24u ), f32( bitcast<i32>( y << 24u ) >> 24u ), f32( bitcast<i32>( y << 16u ) >> 24u ) );
  let c2 = vec3<f32>( f32( bitcast<i32>( y << 8u ) >> 24u ), f32( bitcast<i32>( y ) >> 24u ), f32( bitcast<i32>( z << 24u ) >> 24u ) );
  let c3 = vec3<f32>( f32( bitcast<i32>( z << 16u ) >> 24u ), f32( bitcast<i32>( z << 8u ) >> 24u ), f32( bitcast<i32>( z ) >> 24u ) );
  let c4 = vec3<f32>( f32( bitcast<i32>( w << 24u ) >> 24u ), f32( bitcast<i32>( w << 16u ) >> 24u ), f32( bitcast<i32>( w << 8u ) >> 24u ) );
  let xx = dir.x * dir.x; let yy = dir.y * dir.y; let zz = dir.z * dir.z;
  let rgb = c0 * ( 1.0925484 * dir.x * dir.y )
    + c1 * ( -1.0925484 * dir.y * dir.z )
    + c2 * ( 0.3153915 * ( 2.0 * zz - xx - yy ) )
    + c3 * ( -1.0925484 * dir.x * dir.z )
    + c4 * ( 0.5462742 * ( xx - yy ) );
  return rgb * ( shMax / 127.0 );
}
`;

/** SH band 3 (7 coeffs, sint6 in 4x u32), scaled by sh3Max/31. */
export const WGSL_EVAL_SH3 = /* wgsl */ `
fn sparkEvalSH3( d : vec4<u32>, dir : vec3<f32>, shMax : f32 ) -> vec3<f32> {
  let x = d.x; let y = d.y; let z = d.z; let w = d.w;
  let c0 = vec3<f32>( f32( bitcast<i32>( x << 26u ) >> 26u ), f32( bitcast<i32>( x << 20u ) >> 26u ), f32( bitcast<i32>( x << 14u ) >> 26u ) );
  let c1 = vec3<f32>( f32( bitcast<i32>( x << 8u ) >> 26u ), f32( bitcast<i32>( x << 2u ) >> 26u ), f32( bitcast<i32>( ( x >> 4u ) | ( y << 28u ) ) >> 26u ) );
  let c2 = vec3<f32>( f32( bitcast<i32>( y << 22u ) >> 26u ), f32( bitcast<i32>( y << 16u ) >> 26u ), f32( bitcast<i32>( y << 10u ) >> 26u ) );
  let c3 = vec3<f32>( f32( bitcast<i32>( y << 4u ) >> 26u ), f32( bitcast<i32>( ( y >> 2u ) | ( z << 30u ) ) >> 26u ), f32( bitcast<i32>( z << 24u ) >> 26u ) );
  let c4 = vec3<f32>( f32( bitcast<i32>( z << 18u ) >> 26u ), f32( bitcast<i32>( z << 12u ) >> 26u ), f32( bitcast<i32>( z << 6u ) >> 26u ) );
  let c5 = vec3<f32>( f32( bitcast<i32>( z ) >> 26u ), f32( bitcast<i32>( w << 26u ) >> 26u ), f32( bitcast<i32>( w << 20u ) >> 26u ) );
  let c6 = vec3<f32>( f32( bitcast<i32>( w << 14u ) >> 26u ), f32( bitcast<i32>( w << 8u ) >> 26u ), f32( bitcast<i32>( w << 2u ) >> 26u ) );
  let xx = dir.x * dir.x; let yy = dir.y * dir.y; let zz = dir.z * dir.z;
  let xy = dir.x * dir.y;
  let rgb = c0 * ( -0.5900436 * dir.y * ( 3.0 * xx - yy ) )
    + c1 * ( 2.8906114 * xy * dir.z )
    + c2 * ( -0.4570458 * dir.y * ( 4.0 * zz - xx - yy ) )
    + c3 * ( 0.3731763 * dir.z * ( 2.0 * zz - 3.0 * xx - 3.0 * yy ) )
    + c4 * ( -0.4570458 * dir.x * ( 4.0 * zz - xx - yy ) )
    + c5 * ( 1.4453057 * dir.z * ( xx - yy ) )
    + c6 * ( -0.5900436 * dir.x * ( xx - 3.0 * yy ) );
  return rgb * ( shMax / 31.0 );
}
`;

/**
 * Quantize a radial camera distance to a 16-bit sort key, normalized to the
 * scene's actual `[minDist, maxDist]` depth range so all 16 bits resolve depth
 * within the model.
 *
 * Taking the top 16 bits of the raw float instead would waste almost the entire
 * range on the (near-constant) sign + exponent of a tightly-clustered model:
 * e.g. a butterfly sitting at radial distance ~3 spans only ~64 distinct
 * top-16-bit codes, collapsing ~16k splats into each depth bucket. Equal-key
 * splats then blend in arbitrary index order, so view-dependent (e.g. green
 * iridescent) splats draw in front of splats that should occlude them -> visible
 * colour speckle. Normalizing to the model's depth extent gives the full 65535
 * codes meaningful spread.
 *
 * Larger distance -> smaller key, so the radix sort's ascending output draws
 * far -> near (painter's order). `0xffff` is reserved for the inactive sentinel,
 * so active keys span `0..0xfffe`.
 */
export const WGSL_DEPTH_KEY16 = /* wgsl */ `
fn sparkDepthKey16( dist : f32, minDist : f32, maxDist : f32 ) -> u32 {
  let t = clamp( ( dist - minDist ) / max( maxDist - minDist, 1e-6 ), 0.0, 1.0 );
  return u32( round( ( 1.0 - t ) * 65534.0 ) );
}
`;
