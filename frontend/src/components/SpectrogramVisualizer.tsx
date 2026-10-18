import React, { useEffect, useRef } from 'react';
import { Layers } from 'lucide-react';

interface SpectrogramVisualizerProps {
  analyserNode: AnalyserNode | null;
  isMonitoring: boolean;
}

// Precompute 256-color Magma colormap RGB lookup table
const MAGMA_LUT: [number, number, number][] = [];
for (let i = 0; i < 256; i++) {
  const t = i / 255;
  // Magma-like polynomial interpolation
  const r = Math.min(255, Math.max(0, Math.round(255 * (1.1 * t * t + 0.1 * t))));
  const g = Math.min(255, Math.max(0, Math.round(255 * (Math.pow(t, 2.5) * 0.9))));
  const b = Math.min(255, Math.max(0, Math.round(255 * (Math.sin(t * Math.PI) * 0.7 + (t > 0.8 ? (t - 0.8) * 5 * 0.3 : 0)))));
  MAGMA_LUT.push([r, g, b]);
}

export const SpectrogramVisualizer: React.FC<SpectrogramVisualizerProps> = ({
  analyserNode,
  isMonitoring,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Initialize with dark background
    ctx.fillStyle = '#060913';
    ctx.fillRect(0, 0, width, height);

    let freqData: Uint8Array = new Uint8Array(0);
    if (analyserNode) {
      analyserNode.fftSize = 512;
      freqData = new Uint8Array(analyserNode.frequencyBinCount);
    }

    const render = () => {
      if (isMonitoring && analyserNode && freqData.length > 0) {
        analyserNode.getByteFrequencyData(freqData as any);

        // Shift existing image left by 3 pixels
        const shiftX = 3;
        ctx.drawImage(canvas, shiftX, 0, width - shiftX, height, 0, 0, width - shiftX, height);

        // Draw new slice on the right edge
        const sliceX = width - shiftX;
        const binCount = Math.min(freqData.length, 128); // focus on 0 to 4 kHz range where screeches & screams live
        const binHeight = height / binCount;

        for (let i = 0; i < binCount; i++) {
          // Invert y so low frequencies are at bottom
          const y = height - (i + 1) * binHeight;
          const val = freqData[i];
          const color = MAGMA_LUT[val] || [0, 0, 0];

          ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
          ctx.fillRect(sliceX, y, shiftX, binHeight + 1);
        }
      }

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [analyserNode, isMonitoring]);

  return (
    <div className="canvas-wrapper">
      <div className="canvas-overlay-tag flex items-center gap-1.5">
        <Layers size={12} className="text-orange-400" />
        <span>REAL-TIME SPECTROGRAM (0–4 kHz)</span>
      </div>
      <canvas
        ref={canvasRef}
        className="canvas-element"
        width={700}
        height={140}
      />
    </div>
  );
};
