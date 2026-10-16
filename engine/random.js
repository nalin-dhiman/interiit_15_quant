/** Deterministic primitive streams. No policy receives these generators. */
export function seedHash(value) {
  const text = String(value);
  let h = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function randomStream(seed, stream = "default") {
  let state = seedHash(`${seed}::${stream}`);
  let spare = null;
  return {
    uniform() {
      state += 0x6d2b79f5;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    },
    normal() {
      if (spare !== null) {
        const value = spare;
        spare = null;
        return value;
      }
      const u = Math.max(Number.MIN_VALUE, this.uniform());
      const v = this.uniform();
      const radius = Math.sqrt(-2 * Math.log(u));
      spare = radius * Math.sin(2 * Math.PI * v);
      return radius * Math.cos(2 * Math.PI * v);
    },
  };
}

/** Each event has a separate key: changing RFQ count cannot perturb prices. */
export function eventRandom(seed, stream, step, event = 0) {
  return randomStream(seed, `${stream}/${step}/${event}`);
}
