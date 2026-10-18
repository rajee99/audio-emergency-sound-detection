import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import type { ModelInfo, AlertData, TimelineEvent, TargetClass, PersistenceClassState } from './types';
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
import { ContinuousResampler } from './utils/resampler';
import { Clock, ShieldCheck, Volume2 } from 'lucide-react';

export const App: React.FC = () => {
  // App Modes & Status
  const [activeMode, setActiveMode] = useState<'live' | 'file'>('live');
  const [isMonitoring, setIsMonitoring] = useState<boolean>(false);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');

  // Model & Classification State
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [rawScores, setRawScores] = useState<Record<string, number>>({});
  const [scores, setScores] = useState<Record<string, number>>({});
  const [thresholds, setThresholds] = useState<Record<string, number>>({
    screaming: 0.46,
    alarm: 0.54,
    glass_shatter: 0.42,
    smash_crash: 0.50,
    explosion: 0.42,
    gunshot: 0.42,
    background: 0.43,
  });
  const [persistenceState, setPersistenceState] = useState<Record<string, PersistenceClassState>>({});
  const [topClass, setTopClass] = useState<string>('background');
  const [isScreamingScoreHigh, setIsScreamingScoreHigh] = useState<boolean>(false);
  const [isScreamingAlert, setIsScreamingAlert] = useState<boolean>(false);
  const [latencyMs, setLatencyMs] = useState<number>(0);
  const [rms, setRms] = useState<number>(0);
  const [peak, setPeak] = useState<number>(0);
  const [isSilenceGated, setIsSilenceGated] = useState<boolean>(false);
  const [gateReason, setGateReason] = useState<string>('');
  const [silenceThresholdRms, setSilenceThresholdRms] = useState<number>(0.004);

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
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const resamplerRef = useRef<ContinuousResampler | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const monitoringTimerRef = useRef<number | null>(null);
  const lastAlertTimestampRef = useRef<number>(0);
  const lastFrontendLogRef = useRef<number>(0);

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
    setIsScreamingScoreHigh(false);
    setIsScreamingAlert(false);
    setIsSilenceGated(false);
    setGateReason('');
    setRms(0);
    setPeak(0);

    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
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
    if (resamplerRef.current) {
      resamplerRef.current.reset();
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

      // Connect source to analyser for 60fps canvas visualization
      sourceNode.connect(analyser);

      // Initialize continuous phase-preserving resampler (from hardware rate to 16,000 Hz)
      const resampler = new ContinuousResampler(audioCtx.sampleRate, 16000);
      resamplerRef.current = resampler;

      // 3. Connect WebSocket to FastAPI backend
      const ws = new WebSocket(WS_URL);
      socketRef.current = ws;
      ws.binaryType = 'arraybuffer';

      const sendResampledAudio = (rawPcm: Float32Array) => {
        if (ws.readyState === WebSocket.OPEN) {
          const resampled16k = resampler.process(rawPcm);
          if (resampled16k.length > 0) {
            ws.send(resampled16k.buffer.slice(resampled16k.byteOffset, resampled16k.byteOffset + resampled16k.byteLength) as ArrayBuffer);
          }

          // Input metrics calculation & diagnostic logging
          let sumSq = 0;
          let maxAmp = 0;
          for (let i = 0; i < rawPcm.length; i++) {
            const abs = Math.abs(rawPcm[i]);
            sumSq += abs * abs;
            if (abs > maxAmp) maxAmp = abs;
          }
          const chunkRms = Math.sqrt(sumSq / rawPcm.length);
          const now = performance.now();
          if (now - lastFrontendLogRef.current >= 2000) {
            lastFrontendLogRef.current = now;
            const dbfs = chunkRms > 0 ? (20 * Math.log10(chunkRms)).toFixed(1) : '-∞';
            console.log(
              `[MIC INGEST] HW: ${audioCtx.sampleRate}Hz -> 16000Hz | ` +
              `Chunk: ${rawPcm.length} -> ${resampled16k.length} samples | ` +
              `RMS: ${chunkRms.toFixed(5)} (${dbfs} dBFS) | Peak: ${maxAmp.toFixed(5)}`
            );
          }
        }
      };

      // 4. Setup AudioWorklet (Modern) with ScriptProcessorNode fallback
      let workletRegistered = false;
      if (audioCtx.audioWorklet && typeof audioCtx.audioWorklet.addModule === 'function') {
        try {
          await audioCtx.audioWorklet.addModule('/audio-capture-processor.js');
          const workletNode = new AudioWorkletNode(audioCtx, 'audio-capture-processor');
          workletNodeRef.current = workletNode;

          workletNode.port.onmessage = (event) => {
            const chunk = event.data as Float32Array;
            sendResampledAudio(chunk);
          };

          analyser.connect(workletNode);
          workletRegistered = true;
          console.log('[AUDIO] AudioWorklet registered successfully.');
        } catch (workletErr) {
          console.warn('[AUDIO] AudioWorklet registration failed, falling back to ScriptProcessor:', workletErr);
        }
      }

      if (!workletRegistered) {
        // Fallback for older browsers
        const processor = audioCtx.createScriptProcessor(2048, 1, 1);
        scriptProcessorRef.current = processor;
        processor.onaudioprocess = (audioEvent) => {
          const inputData = audioEvent.inputBuffer.getChannelData(0);
          sendResampledAudio(inputData);
        };
        analyser.connect(processor);
        processor.connect(audioCtx.destination);
        console.log('[AUDIO] ScriptProcessorNode fallback initialized.');
      }

      ws.onopen = () => {
        setIsBackendConnected(true);
        setIsMonitoring(true);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'prediction') {
            if (msg.raw_scores) setRawScores(msg.raw_scores);
            if (msg.scores) setScores(msg.scores);
            if (msg.persistence_state) setPersistenceState(msg.persistence_state);
            if (msg.top_class) setTopClass(msg.top_class);
            if (msg.inference_latency_ms) setLatencyMs(msg.inference_latency_ms);
            if (msg.rms !== undefined) setRms(msg.rms);
            if (msg.peak !== undefined) setPeak(msg.peak);
            if (msg.is_silence_gated !== undefined) setIsSilenceGated(msg.is_silence_gated);
            if (msg.gate_reason) setGateReason(msg.gate_reason);
            if (msg.silence_threshold_rms !== undefined) setSilenceThresholdRms(msg.silence_threshold_rms);

            // Explicit Screaming Distinction
            if (msg.is_screaming_score_high !== undefined) {
              setIsScreamingScoreHigh(msg.is_screaming_score_high);
            }
            if (msg.is_screaming_alert !== undefined) {
              setIsScreamingAlert(msg.is_screaming_alert);
            }

            // Handle alert emission
            if (msg.alert && msg.is_alert) {
              const alertObj: AlertData = msg.alert;
              setActiveAlert(alertObj);

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
                  triggerReason: alertObj.trigger_reason,
                };
                setTimelineEvents((prev) => [newEvt, ...prev].slice(0, 50));
              }
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
    setIsScreamingScoreHigh(false);
    setIsScreamingAlert(false);
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
      screaming: 0.46,
      alarm: 0.54,
      glass_shatter: 0.42,
      smash_crash: 0.50,
      explosion: 0.42,
      gunshot: 0.42,
      background: 0.43,
    };
    setThresholds(defaults);
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: 'set_thresholds', thresholds: defaults }));
    }
  };

  const configuredClasses: TargetClass[] = modelInfo?.target_classes || [
    { name: 'screaming', display_name: 'Screaming', is_emergency: true, threshold: 0.46, persistence_frames: 2 },
    { name: 'alarm', display_name: 'Alarm', is_emergency: true, threshold: 0.54, persistence_frames: 2 },
    { name: 'glass_shatter', display_name: 'Glass Shatter', is_emergency: true, threshold: 0.42, persistence_frames: 1 },
    { name: 'smash_crash', display_name: 'Crash / Smash', is_emergency: true, threshold: 0.50, persistence_frames: 1 },
    { name: 'explosion', display_name: 'Explosion', is_emergency: true, threshold: 0.42, persistence_frames: 1 },
    { name: 'gunshot', display_name: 'Gunshot', is_emergency: true, threshold: 0.42, persistence_frames: 1 },
    { name: 'background', display_name: 'Ambient / Background', is_emergency: false, threshold: 0.43, persistence_frames: 1 },
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

            {/* Audio RMS Level Meter & Gating Diagnostics */}
            <div className="level-meter-wrapper flex flex-col gap-1.5 p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="level-meter-label font-bold text-xs tracking-wider">INPUT LEVEL</span>
                  {isMonitoring && (
                    isSilenceGated ? (
                      <span className="text-[10px] bg-slate-800 text-cyan-300 border border-cyan-800/60 px-2 py-0.5 rounded font-mono font-semibold flex items-center gap-1 shadow-sm">
                        🔇 SILENCE GATE ACTIVE
                      </span>
                    ) : (
                      <span className="text-[10px] bg-emerald-950 text-emerald-400 border border-emerald-700/60 px-2 py-0.5 rounded font-mono font-semibold flex items-center gap-1">
                        🟢 AUDIO DETECTED
                      </span>
                    )
                  )}
                </div>
                <div className="font-mono text-[11px] text-slate-400 flex items-center gap-3">
                  <span title="Window Root Mean Square energy">
                    RMS: {isMonitoring ? `${(rms * 100).toFixed(2)}%` : '0.00%'} ({isMonitoring && rms > 0 ? `${(20 * Math.log10(rms)).toFixed(1)} dBFS` : '-∞ dBFS'})
                  </span>
                  <span title="Window Maximum Absolute Peak">
                    Peak: {isMonitoring ? peak.toFixed(4) : '0.0000'}
                  </span>
                  <span className="text-slate-500 text-[10px]" title="Configured silence gate threshold">
                    (Floor: {(silenceThresholdRms * 100).toFixed(2)}%)
                  </span>
                </div>
              </div>

              <div className="level-bar">
                <div
                  className={`level-bar-fill transition-all duration-100 ${
                    isSilenceGated ? 'bg-cyan-700/50' : 'bg-gradient-to-r from-cyan-500 to-emerald-400'
                  }`}
                  style={{ width: `${Math.min(100, isMonitoring ? Math.max(1, rms * 300) : 0)}%` }}
                />
              </div>

              {isMonitoring && isSilenceGated && gateReason && (
                <div className="text-[10.5px] text-cyan-400/80 font-mono italic">
                  {gateReason}
                </div>
              )}
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
              rawScores={rawScores}
              scores={scores}
              thresholds={thresholds}
              persistenceState={persistenceState}
              topClass={topClass}
              isScreamingScoreHigh={isScreamingScoreHigh}
              isScreamingAlert={isScreamingAlert}
              isSilenceGated={isSilenceGated}
              gateReason={gateReason}
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
