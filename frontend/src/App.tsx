import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import type { ModelInfo, AlertData, TimelineEvent, TargetClass } from './types';
import { fetchModelInfo, fetchHealth, WS_URL } from './services/api';
import { Header } from './components/Header';
import { ControlBar } from './components/ControlBar';
import { AlertBanner } from './components/AlertBanner';
import { WaveformVisualizer } from './components/WaveformVisualizer';
import { SpectrogramVisualizer } from './components/SpectrogramVisualizer';
import { ClassificationPanel } from './components/ClassificationPanel';
import { EventTimeline } from './components/EventTimeline';
import { FileTestMode } from './components/FileTestMode';
import { SelfTestModal } from './components/SelfTestModal';
import { ModelInfoModal } from './components/ModelInfoModal';
import { ThresholdSettingsModal } from './components/ThresholdSettingsModal';
import { Clock, ShieldCheck, Volume2 } from 'lucide-react';

// Fast linear audio resampler for converting browser sample rate (e.g. 44.1k/48k) to 16k
function resampleBuffer(input: Float32Array, inputRate: number, targetRate = 16000): Float32Array {
  if (inputRate === targetRate) return input;
  const ratio = inputRate / targetRate;
  const newLength = Math.round(input.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const originalIndex = i * ratio;
    const indexFloor = Math.floor(originalIndex);
    const indexCeil = Math.min(input.length - 1, Math.ceil(originalIndex));
    const weight = originalIndex - indexFloor;
    result[i] = input[indexFloor] * (1 - weight) + input[indexCeil] * weight;
  }
  return result;
}

export const App: React.FC = () => {
  // App Modes & Status
  const [activeMode, setActiveMode] = useState<'live' | 'file'>('live');
  const [isMonitoring, setIsMonitoring] = useState<boolean>(false);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');

  // Model & Classification State
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [scores, setScores] = useState<Record<string, number>>({});
  const [thresholds, setThresholds] = useState<Record<string, number>>({
    screaming: 0.48,
    alarm: 0.54,
    glass_shatter: 0.49,
    smash_crash: 0.50,
    explosion: 0.48,
    gunshot: 0.50,
    background: 0.48,
  });
  const [topClass, setTopClass] = useState<string>('background');
  const [isScreamingDetected, setIsScreamingDetected] = useState<boolean>(false);
  const [latencyMs, setLatencyMs] = useState<number>(0);
  const [rms, setRms] = useState<number>(0);

  // Alerts & Timeline
  const [activeAlert, setActiveAlert] = useState<AlertData | null>(null);
  const [timelineEvents, setTimelineEvents] = useState<TimelineEvent[]>([]);
  const [monitoringDurationSec, setMonitoringDurationSec] = useState<number>(0);

  // Modals
  const [isSelfTestOpen, setIsSelfTestOpen] = useState<boolean>(false);
  const [isModelInfoOpen, setIsModelInfoOpen] = useState<boolean>(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);

  // Audio & WebSocket Refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const analyserNodeRef = useRef<AnalyserNode | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const monitoringTimerRef = useRef<number | null>(null);
  const lastAlertTimestampRef = useRef<number>(0);

  // Fetch initial model metadata and check backend connectivity
  useEffect(() => {
    let mounted = true;

    async function init() {
      try {
        const health = await fetchHealth();
        if (mounted && health.status === 'online') {
          setIsBackendConnected(true);
        }
      } catch {
        if (mounted) setIsBackendConnected(false);
      }

      try {
        const info = await fetchModelInfo();
        if (mounted) {
          setModelInfo(info);
          const initialTh: Record<string, number> = {};
          info.target_classes.forEach((c) => {
            initialTh[c.name] = c.threshold;
          });
          setThresholds(initialTh);
        }
      } catch (e) {
        console.warn('Could not fetch model info yet:', e);
      }

      // Enumerate audio input devices
      if (navigator.mediaDevices?.enumerateDevices) {
        try {
          const allDevs = await navigator.mediaDevices.enumerateDevices();
          const mics = allDevs.filter((d) => d.kind === 'audioinput');
          if (mounted) {
            setDevices(mics);
            if (mics.length > 0 && !selectedDeviceId) {
              setSelectedDeviceId(mics[0].deviceId);
            }
          }
        } catch (e) {
          console.warn('Device enumeration failed:', e);
        }
      }
    }

    init();
    const interval = setInterval(async () => {
      try {
        await fetchHealth();
        if (mounted) setIsBackendConnected(true);
      } catch {
        if (mounted) setIsBackendConnected(false);
      }
    }, 5000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  // Timer for monitoring duration
  useEffect(() => {
    if (isMonitoring) {
      const start = Date.now() - monitoringDurationSec * 1000;
      monitoringTimerRef.current = window.setInterval(() => {
        setMonitoringDurationSec(Math.floor((Date.now() - start) / 1000));
      }, 500);
    } else {
      if (monitoringTimerRef.current) clearInterval(monitoringTimerRef.current);
    }
    return () => {
      if (monitoringTimerRef.current) clearInterval(monitoringTimerRef.current);
    };
  }, [isMonitoring]);

  // Stop monitoring helper
  const stopMonitoring = useCallback(() => {
    setIsMonitoring(false);
    setActiveAlert(null);

    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();
      scriptProcessorRef.current = null;
    }
    if (analyserNodeRef.current) {
      analyserNodeRef.current.disconnect();
      analyserNodeRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
  }, []);

  // Start monitoring handler
  const startMonitoring = useCallback(async () => {
    try {
      // 1. Request microphone permission
      const constraints: MediaStreamConstraints = {
        audio: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId }, echoCancellation: false, autoGainControl: false, noiseSuppression: false }
          : { echoCancellation: false, autoGainControl: false, noiseSuppression: false },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      mediaStreamRef.current = stream;

      // Update devices list if labels were previously hidden
      const devList = await navigator.mediaDevices.enumerateDevices();
      setDevices(devList.filter((d) => d.kind === 'audioinput'));

      // 2. Initialize Web Audio API
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      const audioCtx = new AudioCtx();
      audioContextRef.current = audioCtx;

      const sourceNode = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.8;
      analyserNodeRef.current = analyser;

      // 4096 buffer size at 48k is ~85ms chunks
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current = processor;

      // 3. Connect WebSocket to FastAPI backend
      const ws = new WebSocket(WS_URL);
      socketRef.current = ws;
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        setIsBackendConnected(true);
        // Connect audio graph
        sourceNode.connect(analyser);
        analyser.connect(processor);
        processor.connect(audioCtx.destination);
        setIsMonitoring(true);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'prediction') {
            if (msg.scores) setScores(msg.scores);
            if (msg.top_class) setTopClass(msg.top_class);
            if (msg.inference_latency_ms) setLatencyMs(msg.inference_latency_ms);
            if (msg.rms !== undefined) setRms(msg.rms);
            if (msg.is_screaming_detected !== undefined) {
              setIsScreamingDetected(msg.is_screaming_detected);
            }

            // Handle detection alert
            if (msg.alert && msg.is_alert) {
              const alertObj: AlertData = msg.alert;
              setActiveAlert(alertObj);

              // Debounce timeline additions by at least 2.5 seconds
              const nowSec = msg.timestamp || 0;
              if (nowSec - lastAlertTimestampRef.current >= 2.5) {
                lastAlertTimestampRef.current = nowSec;
                const newEvt: TimelineEvent = {
                  id: `${nowSec}-${alertObj.event}-${Date.now()}`,
                  eventName: alertObj.event,
                  displayName: alertObj.event
                    .split('_')
                    .map((s: string) => s.charAt(0).toUpperCase() + s.slice(1))
                    .join(' '),
                  timestampSec: alertObj.timestamp,
                  score: alertObj.score,
                  threshold: alertObj.threshold,
                  timeString: new Date().toLocaleTimeString(),
                  isEmergency: true,
                };
                setTimelineEvents((prev) => [newEvt, ...prev].slice(0, 50));
              }
            } else if (!msg.is_emergency) {
              // Optionally decay active alert if signal has subsided
            }
          }
        } catch (e) {
          console.warn('WS message parse error:', e);
        }
      };

      ws.onerror = (e) => {
        console.error('WebSocket live streaming error:', e);
        setIsBackendConnected(false);
      };

      ws.onclose = () => {
        stopMonitoring();
      };

      // 4. Capture microphone Float32 samples and stream over WebSocket
      processor.onaudioprocess = (audioEvent) => {
        if (ws.readyState === WebSocket.OPEN) {
          const inputData = audioEvent.inputBuffer.getChannelData(0);
          // Resample from browser audio context rate to target 16,000 Hz
          const resampled16k = resampleBuffer(inputData, audioCtx.sampleRate, 16000);
          ws.send(resampled16k.buffer as ArrayBuffer);
        }
      };
    } catch (err: any) {
      alert(`Microphone access error: ${err.message || err}. Please ensure microphone permissions are granted in browser.`);
      stopMonitoring();
    }
  }, [selectedDeviceId, stopMonitoring]);

  const toggleMonitoring = () => {
    if (isMonitoring) {
      stopMonitoring();
    } else {
      startMonitoring();
    }
  };

  const handleResetTimeline = () => {
    setTimelineEvents([]);
    setActiveAlert(null);
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: 'reset' }));
    }
  };

  const handleThresholdChange = (className: string, val: number) => {
    setThresholds((prev) => {
      const updated = { ...prev, [className]: val };
      if (socketRef.current?.readyState === WebSocket.OPEN) {
        socketRef.current.send(JSON.stringify({ type: 'set_thresholds', thresholds: updated }));
      }
      return updated;
    });
  };

  const handleResetThresholdDefaults = () => {
    const defaults = {
      screaming: 0.48,
      alarm: 0.54,
      glass_shatter: 0.49,
      smash_crash: 0.50,
      explosion: 0.48,
      gunshot: 0.50,
      background: 0.48,
    };
    setThresholds(defaults);
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: 'set_thresholds', thresholds: defaults }));
    }
  };

  const configuredClasses: TargetClass[] = modelInfo?.target_classes || [
    { name: 'screaming', display_name: 'Screaming', is_emergency: true, threshold: 0.48 },
    { name: 'alarm', display_name: 'Alarm', is_emergency: true, threshold: 0.54 },
    { name: 'glass_shatter', display_name: 'Glass Shatter', is_emergency: true, threshold: 0.49 },
    { name: 'smash_crash', display_name: 'Crash / Smash', is_emergency: true, threshold: 0.50 },
    { name: 'explosion', display_name: 'Explosion', is_emergency: true, threshold: 0.48 },
    { name: 'gunshot', display_name: 'Gunshot', is_emergency: true, threshold: 0.50 },
    { name: 'background', display_name: 'Ambient / Background', is_emergency: false, threshold: 0.48 },
  ];

  const formatElapsed = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="app-container">
      {/* 1. Header */}
      <Header
        isMonitoring={isMonitoring}
        isBackendConnected={isBackendConnected}
        latencyMs={latencyMs}
        modelInfo={modelInfo}
        hasActiveAlert={activeAlert !== null}
        onOpenSelfTest={() => setIsSelfTestOpen(true)}
        onOpenModelInfo={() => setIsModelInfoOpen(true)}
      />

      {/* 2. Control Bar */}
      <ControlBar
        isMonitoring={isMonitoring}
        activeMode={activeMode}
        devices={devices}
        selectedDeviceId={selectedDeviceId}
        onToggleMonitoring={toggleMonitoring}
        onChangeMode={(m) => {
          if (isMonitoring) stopMonitoring();
          setActiveMode(m);
        }}
        onSelectDevice={setSelectedDeviceId}
        onResetTimeline={handleResetTimeline}
        onToggleSettings={() => setIsSettingsOpen(true)}
      />

      {/* 3. Emergency Alert Banner (when active) */}
      <AlertBanner
        alert={activeAlert}
        onDismiss={() => setActiveAlert(null)}
      />

      {/* 4. Main Body: Live Mode vs File Test Mode */}
      {activeMode === 'live' ? (
        <div className="dashboard-grid">
          {/* Left Column: Visualizers & Input Level */}
          <div className="glass-panel visualizer-card">
            <div className="card-header">
              <div className="card-title">
                <Volume2 size={18} className="text-cyan-400" />
                <span>Live Audio Signal Processing</span>
              </div>
              <div className="flex items-center gap-3 text-xs text-slate-400 font-mono">
                <span className="flex items-center gap-1">
                  <Clock size={12} />
                  Session: {formatElapsed(monitoringDurationSec)}
                </span>
                <span>Buffer: 16k Samples/s</span>
              </div>
            </div>

            {/* Oscilloscope Waveform */}
            <WaveformVisualizer
              analyserNode={analyserNodeRef.current}
              isMonitoring={isMonitoring}
              rms={rms}
            />

            {/* Rolling Spectrogram */}
            <SpectrogramVisualizer
              analyserNode={analyserNodeRef.current}
              isMonitoring={isMonitoring}
            />

            {/* Audio RMS Level Meter */}
            <div className="level-meter-wrapper">
              <span className="level-meter-label">INPUT LEVEL</span>
              <div className="level-bar">
                <div
                  className="level-bar-fill"
                  style={{ width: `${Math.min(100, isMonitoring ? rms * 250 : 0)}%` }}
                />
              </div>
              <span className="font-mono text-xs text-slate-400 w-12 text-right">
                {isMonitoring ? `${(rms * 100).toFixed(0)}%` : '0%'}
              </span>
            </div>

            {/* Event Timeline */}
            <EventTimeline
              events={timelineEvents}
              onClear={() => setTimelineEvents([])}
            />
          </div>

          {/* Right Column: Classification Results */}
          <div className="flex flex-col gap-5">
            <ClassificationPanel
              classes={configuredClasses}
              scores={scores}
              thresholds={thresholds}
              topClass={topClass}
              isScreamingDetected={isScreamingDetected}
            />
          </div>
        </div>
      ) : (
        /* File Test Mode */
        <FileTestMode />
      )}

      {/* Modals */}
      <SelfTestModal
        isOpen={isSelfTestOpen}
        onClose={() => setIsSelfTestOpen(false)}
      />

      <ModelInfoModal
        isOpen={isModelInfoOpen}
        onClose={() => setIsModelInfoOpen(false)}
        modelInfo={modelInfo}
      />

      <ThresholdSettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        classes={configuredClasses}
        thresholds={thresholds}
        onChangeThreshold={handleThresholdChange}
        onResetDefaults={handleResetThresholdDefaults}
      />

      {/* Footer Disclaimer */}
      <footer className="text-center text-xs text-slate-500 py-3 flex items-center justify-center gap-2">
        <ShieldCheck size={14} className="text-slate-600" />
        <span>
          Audio Emergency Sound Detector • FYDP 2026 • AudioSet 2017 Deep Acoustic Model • No Speech Recognition / Transcription
        </span>
      </footer>
    </div>
  );
};

export default App;
