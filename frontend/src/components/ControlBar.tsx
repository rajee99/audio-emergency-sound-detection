import React from 'react';
import { Mic, MicOff, RotateCcw, Sliders, Upload, Radio } from 'lucide-react';

interface ControlBarProps {
  isMonitoring: boolean;
  activeMode: 'live' | 'file';
  devices: MediaDeviceInfo[];
  selectedDeviceId: string;
  onToggleMonitoring: () => void;
  onChangeMode: (mode: 'live' | 'file') => void;
  onSelectDevice: (deviceId: string) => void;
  onResetTimeline: () => void;
  onToggleSettings: () => void;
}

export const ControlBar: React.FC<ControlBarProps> = ({
  isMonitoring,
  activeMode,
  devices,
  selectedDeviceId,
  onToggleMonitoring,
  onChangeMode,
  onSelectDevice,
  onResetTimeline,
  onToggleSettings,
}) => {
  return (
    <div className="glass-panel control-bar">
      <div className="control-group">
        {/* Mode Selector Tabs */}
        <div className="mode-tabs">
          <button
            className={`mode-tab flex items-center gap-2 ${activeMode === 'live' ? 'active' : ''}`}
            onClick={() => onChangeMode('live')}
          >
            <Radio size={15} />
            <span>Live Microphone</span>
          </button>
          <button
            className={`mode-tab flex items-center gap-2 ${activeMode === 'file' ? 'active' : ''}`}
            onClick={() => onChangeMode('file')}
          >
            <Upload size={15} />
            <span>File Test Mode</span>
          </button>
        </div>

        {/* Live Monitoring Button */}
        {activeMode === 'live' && (
          <button
            className={isMonitoring ? 'btn-danger' : 'btn-primary'}
            onClick={onToggleMonitoring}
          >
            {isMonitoring ? (
              <>
                <MicOff size={18} />
                <span>Stop Monitoring</span>
              </>
            ) : (
              <>
                <Mic size={18} />
                <span>Start Monitoring</span>
              </>
            )}
          </button>
        )}

        {/* Microphone Device Dropdown */}
        {activeMode === 'live' && devices.length > 0 && (
          <select
            className="device-select"
            value={selectedDeviceId}
            disabled={isMonitoring}
            onChange={(e) => onSelectDevice(e.target.value)}
            title={isMonitoring ? 'Stop monitoring to switch microphone' : 'Select audio input device'}
          >
            {devices.map((device, idx) => (
              <option key={device.deviceId || idx} value={device.deviceId}>
                {device.label || `Microphone ${idx + 1}`}
              </option>
            ))}
          </select>
        )}
      </div>

      <div className="control-group">
        {/* Reset Timeline Button */}
        <button
          className="btn-secondary"
          onClick={onResetTimeline}
          title="Clear detection timeline and reset temporal debounce"
        >
          <RotateCcw size={15} />
          <span>Reset Timeline</span>
        </button>

        {/* Settings / Thresholds Toggle */}
        <button
          className="btn-secondary"
          onClick={onToggleSettings}
          title="Configure Class Thresholds"
        >
          <Sliders size={15} />
          <span>Thresholds</span>
        </button>
      </div>
    </div>
  );
};
