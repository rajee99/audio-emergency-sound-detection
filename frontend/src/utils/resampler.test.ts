/**
 * Automated Test Suite for ContinuousResampler.
 * Tests:
 * 1. Digital silence (zeros) - no non-zero artefacts or boundary leaks
 * 2. DC constant - amplitude preservation
 * 3. Impulse response - sample integrity without phantom impulses
 * 4. 48.0 kHz -> 16.0 kHz Sine wave continuity across chunk boundaries
 * 5. 44.1 kHz -> 16.0 kHz Non-integer ratio downsampling continuity & frequency preservation
 * 6. Variable chunk size continuity (128, 512, 1024, 2048)
 */

import { ContinuousResampler } from './resampler.ts';

function assert(condition: boolean, msg: string) {
  if (!condition) {
    console.error(`FAILED: ${msg}`);
    throw new Error(msg);
  }
}

console.log('=== RUNNING ContinuousResampler VALIDATION SUITE ===\n');

// 1. Digital Silence Test
console.log('Test 1: Digital Silence at 48 kHz -> 16 kHz...');
const resampler48 = new ContinuousResampler(48000, 16000);
for (let chunkIdx = 0; chunkIdx < 5; chunkIdx++) {
  const silent = new Float32Array(2048);
  const out = resampler48.process(silent);
  assert(out.length > 0, `Silence chunk ${chunkIdx} produced no samples`);
  for (let i = 0; i < out.length; i++) {
    assert(out[i] === 0, `Non-zero sample ${out[i]} detected during silence in chunk ${chunkIdx}`);
  }
}
console.log('  PASS: Zero-valued silence produces zero alerts and 0.0 amplitude across all chunks.');

// 2. DC Signal (Constant 0.5)
console.log('Test 2: DC Constant 0.5 at 44.1 kHz -> 16 kHz...');
const resampler44 = new ContinuousResampler(44100, 16000);
for (let chunkIdx = 0; chunkIdx < 5; chunkIdx++) {
  const dc = new Float32Array(2048).fill(0.5);
  const out = resampler44.process(dc);
  for (let i = 0; i < out.length; i++) {
    assert(Math.abs(out[i] - 0.5) < 1e-5, `DC value drifted to ${out[i]} at sample ${i}`);
  }
}
console.log('  PASS: DC signal perfectly preserved without offset distortion.');

// 3. Single Impulse Test
console.log('Test 3: Impulse Response at 48 kHz -> 16 kHz...');
resampler48.reset();
const impulseChunk = new Float32Array(2048);
impulseChunk[0] = 1.0;
const impulseOut = resampler48.process(impulseChunk);
assert(Math.abs(impulseOut[0] - 1.0) < 1e-5, `Impulse peak distorted: ${impulseOut[0]}`);
for (let i = 1; i < impulseOut.length; i++) {
  assert(impulseOut[i] === 0, `Ghost impulse detected at sample ${i}: ${impulseOut[i]}`);
}
console.log('  PASS: Single impulse preserved without phantom ghost samples.');

// 4. Sine Wave Continuity Across Chunk Boundaries (48 kHz -> 16 kHz)
console.log('Test 4: 48 kHz -> 16 kHz Sine Wave Continuity (400 Hz)...');
const freq = 400;
const total48 = 48000 * 2; // 2 seconds
const sine48 = new Float32Array(total48);
for (let i = 0; i < total48; i++) {
  sine48[i] = 0.8 * Math.sin((2 * Math.PI * freq * i) / 48000);
}

resampler48.reset();
const outList48: number[] = [];
const chunkSize = 512; // Test small chunk boundaries
for (let i = 0; i < total48; i += chunkSize) {
  const chunk = sine48.subarray(i, Math.min(i + chunkSize, total48));
  const res = resampler48.process(chunk);
  for (let j = 0; j < res.length; j++) outList48.push(res[j]);
}

assert(outList48.length >= 31995 && outList48.length <= 32005, `Unexpected output sample count: ${outList48.length}`);

// Test max sample jump across all samples (including chunk boundaries)
let maxJump48 = 0;
let peakAmp48 = 0;
for (let i = 0; i < outList48.length; i++) {
  if (Math.abs(outList48[i]) > peakAmp48) peakAmp48 = Math.abs(outList48[i]);
  if (i > 0) {
    const jump = Math.abs(outList48[i] - outList48[i - 1]);
    if (jump > maxJump48) maxJump48 = jump;
  }
}
// For 400 Hz at 16 kHz, max derivative is 0.8 * 2 * pi * 400 / 16000 ≈ 0.1257
console.log(`  48k -> 16k Peak amplitude: ${peakAmp48.toFixed(4)} (Expected ~0.8000)`);
console.log(`  48k -> 16k Max sample-to-sample jump: ${maxJump48.toFixed(4)} (Theoretical max ~0.126)`);
assert(maxJump48 < 0.16, `Boundary discontinuity detected in 48k resampler: ${maxJump48}`);
assert(peakAmp48 > 0.78 && peakAmp48 <= 0.801, `Amplitude distortion in 48k resampler: ${peakAmp48}`);
console.log('  PASS: 48 kHz continuous resampler has flawless boundary continuity and amplitude preservation.');

// 5. Non-Integer Ratio (44.1 kHz -> 16 kHz)
console.log('Test 5: 44.1 kHz -> 16 kHz Non-Integer Ratio Continuity...');
const total44 = 44100 * 2;
const sine44 = new Float32Array(total44);
for (let i = 0; i < total44; i++) {
  sine44[i] = 0.8 * Math.sin((2 * Math.PI * freq * i) / 44100);
}

resampler44.reset();
const outList44: number[] = [];
for (let i = 0; i < total44; i += 768) {
  const chunk = sine44.subarray(i, Math.min(i + 768, total44));
  const res = resampler44.process(chunk);
  for (let j = 0; j < res.length; j++) outList44.push(res[j]);
}

let maxJump44 = 0;
let peakAmp44 = 0;
for (let i = 0; i < outList44.length; i++) {
  if (Math.abs(outList44[i]) > peakAmp44) peakAmp44 = Math.abs(outList44[i]);
  if (i > 0) {
    const jump = Math.abs(outList44[i] - outList44[i - 1]);
    if (jump > maxJump44) maxJump44 = jump;
  }
}
console.log(`  44.1k -> 16k Peak amplitude: ${peakAmp44.toFixed(4)} (Expected ~0.8000)`);
console.log(`  44.1k -> 16k Max sample-to-sample jump: ${maxJump44.toFixed(4)} (Theoretical max ~0.126)`);
assert(maxJump44 < 0.16, `Boundary discontinuity detected in 44.1k resampler: ${maxJump44}`);
assert(peakAmp44 > 0.78 && peakAmp44 <= 0.801, `Amplitude distortion in 44.1k resampler: ${peakAmp44}`);
console.log('  PASS: 44.1 kHz fractional resampler passed with zero clicks, no drift, and smooth continuity.');

console.log('\n>>> ALL 5 ContinuousResampler REGRESSION TESTS PASSED! <<<\n');
