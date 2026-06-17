/** Decode the 3 float16 center components (words 1 and low-16 of word 2). */
export declare const WGSL_UNPACK_CENTER = "\nfn sparkUnpackCenter( packed : vec4<u32> ) -> vec3<f32> {\n  let cxy = unpack2x16float( packed.y );\n  let cz = unpack2x16float( packed.z & 0xffffu ).x;\n  return vec3<f32>( cxy.x, cxy.y, cz );\n}\n";
/** Decode the uint8 RGBA color/opacity (word 0) into [0,1] floats. */
export declare const WGSL_UNPACK_RGBA = "\nfn sparkUnpackRgba( packed : vec4<u32> ) -> vec4<f32> {\n  let w0 = packed.x;\n  return vec4<f32>(\n    f32( w0 & 0xffu ),\n    f32( ( w0 >> 8u ) & 0xffu ),\n    f32( ( w0 >> 16u ) & 0xffu ),\n    f32( ( w0 >> 24u ) & 0xffu )\n  ) / 255.0;\n}\n";
/**
 * Decode the largest of the three log-space uint8 scales (word 3). A code of 0
 * means a flat (2DGS) axis -> 0.0, matching `LN_SCALE_MIN..LN_SCALE_MAX`
 * (-12..9) from `src/defines.ts`. Phase 2's isotropic projection only needs the
 * maximum extent; full anisotropic covariance lands in Phase 4.
 */
export declare const WGSL_UNPACK_MAX_SCALE = "\nfn sparkUnpackMaxScale( packed : vec4<u32> ) -> f32 {\n  let w3 = packed.w;\n  let sx = w3 & 0xffu;\n  let sy = ( w3 >> 8u ) & 0xffu;\n  let sz = ( w3 >> 16u ) & 0xffu;\n  let lnMin = -12.0;\n  let lnStep = ( 9.0 - ( -12.0 ) ) / 254.0;\n  let fx = select( exp( lnMin + f32( sx - 1u ) * lnStep ), 0.0, sx == 0u );\n  let fy = select( exp( lnMin + f32( sy - 1u ) * lnStep ), 0.0, sy == 0u );\n  let fz = select( exp( lnMin + f32( sz - 1u ) * lnStep ), 0.0, sz == 0u );\n  return max( fx, max( fy, fz ) );\n}\n";
/** Decode the three log-space uint8 scales (word 3) into a vec3. */
export declare const WGSL_UNPACK_SCALES = "\nfn sparkUnpackScales( packed : vec4<u32> ) -> vec3<f32> {\n  let w3 = packed.w;\n  let sx = w3 & 0xffu;\n  let sy = ( w3 >> 8u ) & 0xffu;\n  let sz = ( w3 >> 16u ) & 0xffu;\n  let lnMin = -12.0;\n  let lnStep = ( 9.0 - ( -12.0 ) ) / 254.0;\n  return vec3<f32>(\n    select( exp( lnMin + f32( sx - 1u ) * lnStep ), 0.0, sx == 0u ),\n    select( exp( lnMin + f32( sy - 1u ) * lnStep ), 0.0, sy == 0u ),\n    select( exp( lnMin + f32( sz - 1u ) * lnStep ), 0.0, sz == 0u )\n  );\n}\n";
/**
 * Decode the splat rotation quaternion (24-bit octahedral-XY-88 + 8-bit angle),
 * a port of `decodeQuatOctXy88R8` in splatDefines.glsl. The 24-bit code lives in
 * word 2 bits [16:32] and word 3 bits [24:32] of the packed splat.
 */
export declare const WGSL_UNPACK_QUAT = "\nfn sparkUnpackQuat( packed : vec4<u32> ) -> vec4<f32> {\n  let encoded = ( ( packed.z >> 16u ) & 0xffffu ) | ( ( packed.w >> 8u ) & 0xff0000u );\n  let quantU = encoded & 0xffu;\n  let quantV = ( encoded >> 8u ) & 0xffu;\n  let angleInt = encoded >> 16u;\n  let f = vec2<f32>( f32( quantU ) / 255.0 * 2.0 - 1.0, f32( quantV ) / 255.0 * 2.0 - 1.0 );\n  var axis = vec3<f32>( f.x, f.y, 1.0 - abs( f.x ) - abs( f.y ) );\n  let t = max( -axis.z, 0.0 );\n  axis.x = axis.x + select( t, -t, axis.x >= 0.0 );\n  axis.y = axis.y + select( t, -t, axis.y >= 0.0 );\n  axis = normalize( axis );\n  let theta = ( f32( angleInt ) / 255.0 ) * 3.14159265359;\n  return vec4<f32>( axis * sin( theta * 0.5 ), cos( theta * 0.5 ) );\n}\n";
/**
 * Compute the projected 2D covariance of a splat, returned as `(a, d, b)` for
 * the symmetric matrix `[[a, b], [b, d]]`. Port of the covariance + Jacobian
 * projection in splatVertex.glsl: build the object-space scale-rotate matrix
 * (inlined `scaleQuaternionToMatrix`), rotate into view space by `mv3` (the
 * upper-left 3x3 of the model-view matrix), form the 3D covariance, then project
 * through the perspective Jacobian. Self-contained (no helper calls / structs).
 */
export declare const WGSL_COV2D = "\nfn sparkCov2D( mv3 : mat3x3<f32>, s : vec3<f32>, q : vec4<f32>, focal : vec2<f32>, viewC : vec3<f32> ) -> vec3<f32> {\n  let rs = mat3x3<f32>(\n    vec3<f32>( s.x * ( 1.0 - 2.0 * ( q.y * q.y + q.z * q.z ) ), s.x * ( 2.0 * ( q.x * q.y + q.w * q.z ) ), s.x * ( 2.0 * ( q.x * q.z - q.w * q.y ) ) ),\n    vec3<f32>( s.y * ( 2.0 * ( q.x * q.y - q.w * q.z ) ), s.y * ( 1.0 - 2.0 * ( q.x * q.x + q.z * q.z ) ), s.y * ( 2.0 * ( q.y * q.z + q.w * q.x ) ) ),\n    vec3<f32>( s.z * ( 2.0 * ( q.x * q.z + q.w * q.y ) ), s.z * ( 2.0 * ( q.y * q.z - q.w * q.x ) ), s.z * ( 1.0 - 2.0 * ( q.x * q.x + q.y * q.y ) ) )\n  );\n  let viewRS = mv3 * rs;\n  let cov3D = viewRS * transpose( viewRS );\n  let invZ = 1.0 / viewC.z;\n  let j1 = focal * invZ;\n  let j2 = -( j1 * viewC.xy ) * invZ;\n  let jac = mat3x3<f32>(\n    vec3<f32>( j1.x, 0.0, j2.x ),\n    vec3<f32>( 0.0, j1.y, j2.y ),\n    vec3<f32>( 0.0, 0.0, 0.0 )\n  );\n  let cov2D = transpose( jac ) * cov3D * jac;\n  // (a, d, b) for [[a, b], [b, d]]; column-major so cov2D[0][1] is the off-diagonal.\n  return vec3<f32>( cov2D[0][0], cov2D[1][1], cov2D[0][1] );\n}\n";
/** SH band 1 (3 coeffs, sint7 in 2x u32), scaled by sh1Max/63. */
export declare const WGSL_EVAL_SH1 = "\nfn sparkEvalSH1( d : vec2<u32>, dir : vec3<f32>, shMax : f32 ) -> vec3<f32> {\n  let x = d.x; let y = d.y;\n  let c0 = vec3<f32>( f32( bitcast<i32>( x << 25u ) >> 25u ), f32( bitcast<i32>( x << 18u ) >> 25u ), f32( bitcast<i32>( x << 11u ) >> 25u ) );\n  let c1 = vec3<f32>( f32( bitcast<i32>( x << 4u ) >> 25u ), f32( bitcast<i32>( ( x >> 3u ) | ( y << 29u ) ) >> 25u ), f32( bitcast<i32>( y << 22u ) >> 25u ) );\n  let c2 = vec3<f32>( f32( bitcast<i32>( y << 15u ) >> 25u ), f32( bitcast<i32>( y << 8u ) >> 25u ), f32( bitcast<i32>( y << 1u ) >> 25u ) );\n  let rgb = c0 * ( -0.4886025 * dir.y ) + c1 * ( 0.4886025 * dir.z ) + c2 * ( -0.4886025 * dir.x );\n  return rgb * ( shMax / 63.0 );\n}\n";
/** SH band 2 (5 coeffs, sint8 in 4x u32), scaled by sh2Max/127. */
export declare const WGSL_EVAL_SH2 = "\nfn sparkEvalSH2( d : vec4<u32>, dir : vec3<f32>, shMax : f32 ) -> vec3<f32> {\n  let x = d.x; let y = d.y; let z = d.z; let w = d.w;\n  let c0 = vec3<f32>( f32( bitcast<i32>( x << 24u ) >> 24u ), f32( bitcast<i32>( x << 16u ) >> 24u ), f32( bitcast<i32>( x << 8u ) >> 24u ) );\n  let c1 = vec3<f32>( f32( bitcast<i32>( x ) >> 24u ), f32( bitcast<i32>( y << 24u ) >> 24u ), f32( bitcast<i32>( y << 16u ) >> 24u ) );\n  let c2 = vec3<f32>( f32( bitcast<i32>( y << 8u ) >> 24u ), f32( bitcast<i32>( y ) >> 24u ), f32( bitcast<i32>( z << 24u ) >> 24u ) );\n  let c3 = vec3<f32>( f32( bitcast<i32>( z << 16u ) >> 24u ), f32( bitcast<i32>( z << 8u ) >> 24u ), f32( bitcast<i32>( z ) >> 24u ) );\n  let c4 = vec3<f32>( f32( bitcast<i32>( w << 24u ) >> 24u ), f32( bitcast<i32>( w << 16u ) >> 24u ), f32( bitcast<i32>( w << 8u ) >> 24u ) );\n  let xx = dir.x * dir.x; let yy = dir.y * dir.y; let zz = dir.z * dir.z;\n  let rgb = c0 * ( 1.0925484 * dir.x * dir.y )\n    + c1 * ( -1.0925484 * dir.y * dir.z )\n    + c2 * ( 0.3153915 * ( 2.0 * zz - xx - yy ) )\n    + c3 * ( -1.0925484 * dir.x * dir.z )\n    + c4 * ( 0.5462742 * ( xx - yy ) );\n  return rgb * ( shMax / 127.0 );\n}\n";
/** SH band 3 (7 coeffs, sint6 in 4x u32), scaled by sh3Max/31. */
export declare const WGSL_EVAL_SH3 = "\nfn sparkEvalSH3( d : vec4<u32>, dir : vec3<f32>, shMax : f32 ) -> vec3<f32> {\n  let x = d.x; let y = d.y; let z = d.z; let w = d.w;\n  let c0 = vec3<f32>( f32( bitcast<i32>( x << 26u ) >> 26u ), f32( bitcast<i32>( x << 20u ) >> 26u ), f32( bitcast<i32>( x << 14u ) >> 26u ) );\n  let c1 = vec3<f32>( f32( bitcast<i32>( x << 8u ) >> 26u ), f32( bitcast<i32>( x << 2u ) >> 26u ), f32( bitcast<i32>( ( x >> 4u ) | ( y << 28u ) ) >> 26u ) );\n  let c2 = vec3<f32>( f32( bitcast<i32>( y << 22u ) >> 26u ), f32( bitcast<i32>( y << 16u ) >> 26u ), f32( bitcast<i32>( y << 10u ) >> 26u ) );\n  let c3 = vec3<f32>( f32( bitcast<i32>( y << 4u ) >> 26u ), f32( bitcast<i32>( ( y >> 2u ) | ( z << 30u ) ) >> 26u ), f32( bitcast<i32>( z << 24u ) >> 26u ) );\n  let c4 = vec3<f32>( f32( bitcast<i32>( z << 18u ) >> 26u ), f32( bitcast<i32>( z << 12u ) >> 26u ), f32( bitcast<i32>( z << 6u ) >> 26u ) );\n  let c5 = vec3<f32>( f32( bitcast<i32>( z ) >> 26u ), f32( bitcast<i32>( w << 26u ) >> 26u ), f32( bitcast<i32>( w << 20u ) >> 26u ) );\n  let c6 = vec3<f32>( f32( bitcast<i32>( w << 14u ) >> 26u ), f32( bitcast<i32>( w << 8u ) >> 26u ), f32( bitcast<i32>( w << 2u ) >> 26u ) );\n  let xx = dir.x * dir.x; let yy = dir.y * dir.y; let zz = dir.z * dir.z;\n  let xy = dir.x * dir.y;\n  let rgb = c0 * ( -0.5900436 * dir.y * ( 3.0 * xx - yy ) )\n    + c1 * ( 2.8906114 * xy * dir.z )\n    + c2 * ( -0.4570458 * dir.y * ( 4.0 * zz - xx - yy ) )\n    + c3 * ( 0.3731763 * dir.z * ( 2.0 * zz - 3.0 * xx - 3.0 * yy ) )\n    + c4 * ( -0.4570458 * dir.x * ( 4.0 * zz - xx - yy ) )\n    + c5 * ( 1.4453057 * dir.z * ( xx - yy ) )\n    + c6 * ( -0.5900436 * dir.x * ( xx - 3.0 * yy ) );\n  return rgb * ( shMax / 31.0 );\n}\n";
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
export declare const WGSL_DEPTH_KEY16 = "\nfn sparkDepthKey16( dist : f32, minDist : f32, maxDist : f32 ) -> u32 {\n  let t = clamp( ( dist - minDist ) / max( maxDist - minDist, 1e-6 ), 0.0, 1.0 );\n  return u32( round( ( 1.0 - t ) * 65534.0 ) );\n}\n";
