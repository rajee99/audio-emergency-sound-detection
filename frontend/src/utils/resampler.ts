/**
 * Continuous phase-preserving audio resampler.
 * Carries fractional phase and edge samples across block boundaries,
 * avoiding clicks, sample drift, and phase discontinuities.
 */
export class ContinuousResampler {
  private fromSampleRate: number;
  private toSampleRate: number;
  private phase: number = 0.0;
  private lastSample: number = 0.0;
  private hasLastSample: boolean = false;

  constructor(fromSampleRate: number, toSampleRate: number = 16000) {
    this.fromSampleRate = fromSampleRate;
    this.toSampleRate = toSampleRate;
  }

  public reset(): void {
    this.phase = 0.0;
    this.lastSample = 0.0;
    this.hasLastSample = false;
  }

  public process(input: Float32Array): Float32Array {
    if (this.fromSampleRate === this.toSampleRate || input.length === 0) {
      return input;
    }

    if (!this.hasLastSample) {
      this.lastSample = input[0];
      this.hasLastSample = true;
      this.phase = 0.0;
    }

    const ratio = this.fromSampleRate / this.toSampleRate;
    const L = input.length;
    const maxOutLen = Math.ceil(L / ratio) + 2;
    const temp = new Float32Array(maxOutLen);
    let count = 0;
    let t = this.phase;

    while (t < L - 1) {
      const k = Math.floor(t);
      const frac = t - k;

      const s0 = k === -1 ? this.lastSample : input[k];
      const s1 = input[k + 1];

      temp[count++] = s0 * (1.0 - frac) + s1 * frac;
      t += ratio;
    }

    this.phase = t - L;
    this.lastSample = input[L - 1];

    return new Float32Array(temp.subarray(0, count));
  }
}

