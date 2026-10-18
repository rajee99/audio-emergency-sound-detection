/**
 * AudioWorkletProcessor for low-latency microphone audio streaming.
 * Replaces deprecated ScriptProcessorNode.
 * Captures 128-sample frames from Web Audio graph, accumulates into 2048-sample blocks,
 * and posts Float32Array chunks to the main thread.
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 2048;
    this.buffer = new Float32Array(this.bufferSize);
    this.bytesWritten = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input && input.length > 0 && input[0]) {
      const channel = input[0];
      for (let i = 0; i < channel.length; i++) {
        this.buffer[this.bytesWritten++] = channel[i];
        if (this.bytesWritten >= this.bufferSize) {
          // Post a copy of the buffer to main thread
          this.port.postMessage(this.buffer.slice(0, this.bufferSize));
          this.bytesWritten = 0;
        }
      }
    }
    return true; // Keep processor alive
  }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);
