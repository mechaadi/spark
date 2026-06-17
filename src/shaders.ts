import * as THREE from "three";

import computeUvec4Template from "./shaders/computeUvec4.glsl";
import computeUvec4Vec4Template from "./shaders/computeUvec4_Vec4.glsl";
import computeUvec4x2Vec4Template from "./shaders/computeUvec4x2_Vec4.glsl";
import computeVec4Template from "./shaders/computeVec4.glsl";
import splatDefines from "./shaders/splatDefines.glsl";
import splatFragment from "./shaders/splatFragment.glsl";
import splatVertex from "./shaders/splatVertex.glsl";

let shaders: Record<string, string> | null = null;

export function getShaders(): Record<string, string> {
  if (!shaders) {
    // `ShaderChunk` only exists on the WebGL build of three. The WebGPU build
    // (three.webgpu.js) omits it, so when Spark is imported alongside the WebGPU
    // backend with `three` mapped to that build, guard the registration —
    // otherwise this throws at init ("Cannot set properties of undefined"). The
    // WebGPU path uses WGSL and never consults this chunk.
    if (THREE.ShaderChunk) {
      // @ts-ignore
      THREE.ShaderChunk.splatDefines = splatDefines;
    }
    shaders = {
      splatVertex,
      splatFragment,
      computeVec4Template,
      computeUvec4Vec4Template,
      computeUvec4x2Vec4Template,
      computeUvec4Template,
    };
  }
  return shaders;
}
