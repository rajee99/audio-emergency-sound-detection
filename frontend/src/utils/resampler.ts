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

  constructor(fromSampleRate: number, toSampleRate: number = 16000) {
    this.fromSampleRate = fromSampleRate;
    this.toSampleRate = toSampleRate;
  }

  public reset(): void {
    this.phase = 0.0;
    this.lastSample = 0.0;
  }

  public process(input: Float32Array): Float32Array {
    if (this.fromSampleRate === this.toSampleRate) {
      return input;
    }

    const ratio = this.fromSampleRate / this.toSampleRate;
    const available = input.length - this.phase;
    if (available <= 0) {
      this.phase -= input.length;
      return new Float32Array(0);
    }

    const outLen = Math.floor(available / ratio);
    const output = new Float32Array(outLen);

    for (let i = 0; i < outLen; i++) {
      const pos = this.phase + i * ratio;
      const idx = Math.floor(pos);
      const frac = pos - idx;

      const s0 = idx === 0 ? this.lastSample : input[idx - 1];
      const s1 = idx < input.length ? input[idx] : this.lastSample;
      output[i] = s0 * (1.0 - frac) + s1 * frac;
    }

    this.phase = (this.phase + outLen * ratio) - input.length;
    if (input.length > 0) {
      this.lastSample = input[input.length - 1];
    }

    return output;
  }
}
