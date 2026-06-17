import { setPackedSplat } from "@sparkjsdev/spark";

// Generate `count` synthetic splats packed into Spark's 16-byte layout: random
// positions in a cube, isotropic small scale, random opaque-ish colors. Shared
// by the webgpu.html / webgl.html benchmark pages so both render identical data.
export function generatePackedSplats(count) {
  const packedArray = new Uint32Array(count * 4);
  let s = 0x9e3779b9 >>> 0;
  const rnd = () => {
    s = (Math.imul(s ^ (s >>> 15), 0x2c1b3c6d) + 1) >>> 0;
    return s / 4294967296;
  };
  for (let i = 0; i < count; i++) {
    const x = (rnd() * 2 - 1) * 2.0;
    const y = (rnd() * 2 - 1) * 2.0;
    const z = (rnd() * 2 - 1) * 2.0;
    const sc = 0.008 + rnd() * 0.012;
    setPackedSplat(
      packedArray,
      i,
      x,
      y,
      z,
      sc,
      sc,
      sc,
      0,
      0,
      0,
      1,
      0.7,
      rnd(),
      rnd(),
      rnd(),
    );
  }
  return packedArray;
}

// Rolling FPS meter. Call tick() once per rendered frame; reads `value`.
export function makeFpsMeter() {
  const times = [];
  return {
    value: 0,
    tick(nowMs) {
      times.push(nowMs);
      while (times.length > 0 && nowMs - times[0] > 1000) times.shift();
      this.value = times.length;
      return this.value;
    },
  };
}
