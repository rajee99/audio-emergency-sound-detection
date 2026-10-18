import React from 'react';
import { X, Sliders, RotateCcw } from 'lucide-react';
import type { TargetClass } from '../types';

interface ThresholdSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  classes: TargetClass[];
  thresholds: Record<string, number>;
  onChangeThreshold: (className: string, value: number) => void;
  onResetDefaults: () => void;
}

export const ThresholdSettingsModal: React.FC<ThresholdSettingsModalProps> = ({
  isOpen,
  onClose,
  classes,
  thresholds,
  onChangeThreshold,
  onResetDefaults,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
      <div className="glass-panel w-full max-w-lg p-6 relative flex flex-col gap-4 border-slate-700">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <Sliders size={20} className="text-cyan-400" />
            <h2 className="text-base font-bold text-slate-100">
              Calibrated Decision Thresholds
            </h2>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded hover:bg-slate-800"
          >
            <X size={18} />
          </button>
        </div>

        <div className="text-xs text-slate-400">
          Thresholds calibrated on validation data to maximize life-safety recall while suppressing false positives on ambient sounds.
        </div>

        <div className="flex flex-col gap-3 max-h-80 overflow-y-auto pr-1">
          {classes.map((cls) => {
            const currentVal = thresholds[cls.name] ?? cls.threshold;
            return (
              <div
                key={cls.name}
                className="p-3 rounded bg-slate-900/60 border border-slate-800 flex flex-col gap-2"
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-200">{cls.display_name}</span>
                  <span className="font-mono font-bold text-cyan-400">
                    {(currentVal * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-[10px] text-slate-500 font-mono">0.10</span>
                  <input
                    type="range"
                    min="0.10"
                    max="0.90"
                    step="0.01"
                    value={currentVal}
                    onChange={(e) => onChangeThreshold(cls.name, parseFloat(e.target.value))}
                    className="flex-1 accent-blue-500 cursor-pointer"
                  />
                  <span className="text-[10px] text-slate-500 font-mono">0.90</span>
                </div>
              </div>
            );
          })}
        </div>

        <div className="flex items-center justify-between pt-3 border-t border-slate-800 mt-2">
          <button
            onClick={onResetDefaults}
            className="btn-secondary text-xs flex items-center gap-1.5"
          >
            <RotateCcw size={14} />
            <span>Reset Calibrated Defaults</span>
          </button>
          <button onClick={onClose} className="btn-primary text-xs">
            Save & Close
          </button>
        </div>
      </div>
    </div>
  );
};
