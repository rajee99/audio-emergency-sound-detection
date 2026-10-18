import React from 'react';
import { Activity, Radio, Server, ShieldAlert, Cpu } from 'lucide-react';
import type { ModelInfo } from '../types';

interface HeaderProps {
  isMonitoring: boolean;
  isBackendConnected: boolean;
  latencyMs: number;
  modelInfo: ModelInfo | null;
  hasActiveAlert: boolean;
  onOpenSelfTest: () => void;
  onOpenModelInfo: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  isMonitoring,
  isBackendConnected,
  latencyMs,
  modelInfo,
  hasActiveAlert,
  onOpenSelfTest,
  onOpenModelInfo,
}) => {
  return (
    <header className="glass-panel header-card">
      <div className="brand-section">
        <div className="brand-icon-wrapper">
          <ShieldAlert size={28} />
        </div>
        <div>
          <h1 className="brand-title">Audio Emergency Sound Detector</h1>
          <div className="brand-subtitle">
            <span>AudioSet 2017 Real-Time Acoustic Event Classification</span>
            <span>•</span>
            <span className="font-mono text-cyan-400">Strictly Non-Speech / Environmental</span>
          </div>
        </div>
      </div>

      <div className="status-cluster">
        {/* Monitoring State Pill */}
        {hasActiveAlert ? (
          <span className="badge-alert">
            <span className="dot-pulse" />
            ALERT TRIGGERED
          </span>
        ) : isMonitoring ? (
          <span className="badge-live">
            <span className="dot-pulse" />
            LIVE MONITORING
          </span>
        ) : (
          <span className="status-pill">
            <Radio size={14} className="text-slate-500" />
            <span>STANDBY</span>
          </span>
        )}

        {/* Backend Connection */}
        <div className={`status-pill ${isBackendConnected ? 'connected' : 'disconnected'}`}>
          <Server size={14} />
          <span>{isBackendConnected ? 'Backend Connected' : 'Backend Offline'}</span>
        </div>

        {/* Latency Pill */}
        {isMonitoring && latencyMs > 0 && (
          <div className="status-pill text-cyan-400 border-cyan-500/30">
            <Activity size={14} />
            <span className="font-mono">{latencyMs.toFixed(1)} ms</span>
          </div>
        )}

        {/* Model Info Pill */}
        <button
          onClick={onOpenModelInfo}
          className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1.5"
          title="View Model Configuration and Benchmarks"
        >
          <Cpu size={14} className="text-blue-400" />
          <span>{modelInfo?.model_name || 'CompactAudioCNN'}</span>
        </button>

        {/* Self-Test Button */}
        <button
          onClick={onOpenSelfTest}
          className="btn-secondary text-xs py-1.5 px-3 border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/10"
          title="Run System Verification & Diagnostics"
        >
          Diagnostics Self-Test
        </button>
      </div>
    </header>
  );
};
