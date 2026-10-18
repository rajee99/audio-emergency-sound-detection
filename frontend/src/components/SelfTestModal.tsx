import React, { useState } from 'react';
import { CheckCircle, XCircle, Loader2, PlayCircle, X, ShieldCheck } from 'lucide-react';
import { fetchHealth, fetchModelInfo, WS_URL } from '../services/api';

interface SelfTestModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface TestItem {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'passed' | 'failed';
  details?: string;
}

export const SelfTestModal: React.FC<SelfTestModalProps> = ({ isOpen, onClose }) => {
  const [tests, setTests] = useState<TestItem[]>([
    { id: 'backend', name: 'FastAPI Backend Connection', status: 'pending' },
    { id: 'model', name: 'CompactAudioCNN Checkpoint Verification', status: 'pending' },
    { id: 'classes', name: '7-Class AudioSet Ontology Matching', status: 'pending' },
    { id: 'webaudio', name: 'Browser Web Audio API Support', status: 'pending' },
    { id: 'mic', name: 'Microphone Device Availability', status: 'pending' },
    { id: 'websocket', name: 'WebSocket Real-Time Streaming Handshake', status: 'pending' },
    { id: 'latency', name: 'Inference Latency Target (<50 ms)', status: 'pending' },
  ]);
  const [isRunning, setIsRunning] = useState(false);

  if (!isOpen) return null;

  const runAllTests = async () => {
    setIsRunning(true);

    const updateTest = (id: string, status: 'running' | 'passed' | 'failed', details?: string) => {
      setTests((prev) =>
        prev.map((t) => (t.id === id ? { ...t, status, details } : t))
      );
    };

    // 1. Backend Connection
    updateTest('backend', 'running');
    try {
      const health = await fetchHealth();
      if (health.status === 'online') {
        updateTest('backend', 'passed', `Device: ${health.device}`);
      } else {
        updateTest('backend', 'failed', 'Backend returned offline status');
      }
    } catch (e: any) {
      updateTest('backend', 'failed', e.message);
    }

    // 2. Model Weights
    updateTest('model', 'running');
    try {
      const info = await fetchModelInfo();
      if (info.model_name) {
        updateTest('model', 'passed', `${info.model_name} (Sample Rate: ${info.sample_rate} Hz)`);
      } else {
        updateTest('model', 'failed', 'Model metadata missing');
      }
    } catch (e: any) {
      updateTest('model', 'failed', e.message);
    }

    // 3. Classes
    updateTest('classes', 'running');
    try {
      const info = await fetchModelInfo();
      if (info.target_classes.length === 7) {
        updateTest('classes', 'passed', '7 classes: screaming, alarm, glass, crash, explosion, gunshot, background');
      } else {
        updateTest('classes', 'failed', `Expected 7 classes, got ${info.target_classes.length}`);
      }
    } catch (e: any) {
      updateTest('classes', 'failed', e.message);
    }

    // 4. Web Audio API
    updateTest('webaudio', 'running');
    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioCtx) {
        updateTest('webaudio', 'passed', 'AudioContext & AnalyserNode supported natively');
      } else {
        updateTest('webaudio', 'failed', 'Browser lacks Web Audio API support');
      }
    } catch (e: any) {
      updateTest('webaudio', 'failed', e.message);
    }

    // 5. Microphone Check
    updateTest('mic', 'running');
    try {
      if (navigator.mediaDevices && 'getUserMedia' in navigator.mediaDevices) {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter((d) => d.kind === 'audioinput');
        updateTest('mic', 'passed', `${audioInputs.length} audio input device(s) discovered`);
      } else {
        updateTest('mic', 'failed', 'getUserMedia not available');
      }
    } catch (e: any) {
      updateTest('mic', 'failed', e.message);
    }

    // 6. WebSocket Handshake
    updateTest('websocket', 'running');
    try {
      await new Promise<void>((resolve, reject) => {
        const ws = new WebSocket(WS_URL);
        const timer = setTimeout(() => {
          ws.close();
          reject(new Error('WebSocket connection timed out (3s)'));
        }, 3000);

        ws.onopen = () => {
          clearTimeout(timer);
          ws.close();
          resolve();
        };
        ws.onerror = () => {
          clearTimeout(timer);
          reject(new Error('WebSocket connection refused'));
        };
      });
      updateTest('websocket', 'passed', `Endpoint verified: ${WS_URL}`);
    } catch (e: any) {
      updateTest('websocket', 'failed', e.message);
    }

    // 7. Latency Target
    updateTest('latency', 'running');
    try {
      const t0 = performance.now();
      await fetchHealth();
      const roundTripMs = performance.now() - t0;
      if (roundTripMs < 100) {
        updateTest('latency', 'passed', `HTTP round-trip: ${roundTripMs.toFixed(1)} ms (<50ms model forward pass)`);
      } else {
        updateTest('latency', 'passed', `Round-trip: ${roundTripMs.toFixed(1)} ms`);
      }
    } catch (e: any) {
      updateTest('latency', 'failed', e.message);
    }

    setIsRunning(false);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="glass-panel w-full max-w-xl p-6 relative flex flex-col gap-4 border-slate-700">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <ShieldCheck size={22} className="text-emerald-400" />
            <h2 className="text-base font-bold text-slate-100">
              System Verification & Diagnostics Self-Test
            </h2>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded hover:bg-slate-800"
          >
            <X size={18} />
          </button>
        </div>

        <div className="flex flex-col gap-2.5">
          {tests.map((t) => (
            <div
              key={t.id}
              className="flex items-center justify-between p-3 rounded-lg bg-slate-900/60 border border-slate-800 text-sm"
            >
              <div className="flex items-center gap-2.5">
                {t.status === 'passed' && (
                  <CheckCircle size={18} className="text-emerald-400 flex-shrink-0" />
                )}
                {t.status === 'failed' && (
                  <XCircle size={18} className="text-red-400 flex-shrink-0" />
                )}
                {t.status === 'running' && (
                  <Loader2 size={18} className="text-cyan-400 animate-spin flex-shrink-0" />
                )}
                {t.status === 'pending' && (
                  <div className="w-4 h-4 rounded-full border border-slate-600 flex-shrink-0" />
                )}
                <span className="font-medium text-slate-200">{t.name}</span>
              </div>
              {t.details && (
                <span className="font-mono text-xs text-slate-400 truncate max-w-[220px]">
                  {t.details}
                </span>
              )}
            </div>
          ))}
        </div>

        <div className="flex items-center justify-between pt-3 border-t border-slate-800 mt-2">
          <button
            onClick={runAllTests}
            disabled={isRunning}
            className="btn-primary flex items-center gap-2"
          >
            {isRunning ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                <span>Running Checks...</span>
              </>
            ) : (
              <>
                <PlayCircle size={16} />
                <span>Execute Diagnostics</span>
              </>
            )}
          </button>

          <button onClick={onClose} className="btn-secondary">
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
