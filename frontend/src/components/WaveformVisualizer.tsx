import React, { useEffect, useRef } from 'react';
import { Activity } from 'lucide-react';

interface WaveformVisualizerProps {
  analyserNode: AnalyserNode | null;
  isMonitoring: boolean;
  rms?: number;
}

export const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({
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

    let bufferLength = 1024;
    let dataArray = new Uint8Array(bufferLength);

    if (analyserNode) {
      bufferLength = analyserNode.fftSize;
      dataArray = new Uint8Array(bufferLength);
    }

    const render = () => {
      const width = canvas.width;
      const height = canvas.height;

      // Dark background with slight trail
      ctx.fillStyle = '#060913';
      ctx.fillRect(0, 0, width, height);

      // Grid Lines
      ctx.strokeStyle = 'rgba(51, 65, 85, 0.2)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      // Center zero line
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      // Top/bottom quarter lines
      ctx.moveTo(0, height / 4);
      ctx.lineTo(width, height / 4);
      ctx.moveTo(0, (height * 3) / 4);
      ctx.lineTo(width, (height * 3) / 4);
      ctx.stroke();

      if (isMonitoring && analyserNode) {
        analyserNode.getByteTimeDomainData(dataArray);

        ctx.lineWidth = 2;
        ctx.strokeStyle = '#06b6d4';
        ctx.shadowBlur = 8;
        ctx.shadowColor = 'rgba(6, 182, 212, 0.8)';
        ctx.beginPath();

        const sliceWidth = width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const v = dataArray[i] / 128.0; // 0.0 to 2.0
          const y = (v * height) / 2;

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }

          x += sliceWidth;
        }

        ctx.lineTo(width, height / 2);
        ctx.stroke();

        // Reset shadow
        ctx.shadowBlur = 0;
      } else {
        // Idle flatline
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = 'rgba(100, 116, 139, 0.4)';
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();
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
        <Activity size={12} className="text-cyan-400" />
        <span>LIVE PCM WAVEFORM (16 kHz)</span>
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
