import React from 'react';
import type { TargetClass } from '../types';
import { Shield, Volume2, HelpCircle } from 'lucide-react';

interface ClassificationPanelProps {
  classes: TargetClass[];
  scores: Record<string, number>;
  thresholds: Record<string, number>;
  topClass: string;
  isScreamingDetected: boolean;
}

const CLASS_ICONS: Record<string, string> = {
  screaming: '🗣️',
  alarm: '🚨',
  glass_shatter: '🪟',
  smash_crash: '💥',
  explosion: '💣',
  gunshot: '🔫',
  background: '🌳',
};

export const ClassificationPanel: React.FC<ClassificationPanelProps> = ({
  classes,
  scores,
  thresholds,
  topClass,
  isScreamingDetected,
}) => {
  return (
    <div className="glass-panel classification-card">
      <div className="card-header">
        <div className="card-title">
          <Volume2 size={18} className="text-blue-400" />
          <span>Acoustic Event Classification</span>
        </div>
        <div className="text-xs text-slate-400 flex items-center gap-1">
          <HelpCircle size={13} />
          <span>Calibrated AudioSet 2017</span>
        </div>
      </div>

      <div className="flex flex-col gap-2.5">
        {classes.map((cls) => {
          const score = scores[cls.name] ?? 0.0;
          const threshold = thresholds[cls.name] ?? cls.threshold;
          const isTriggered = score >= threshold;
          const isTop = topClass === cls.name;
          const isScream = cls.name === 'screaming';

          return (
            <div
              key={cls.name}
              className={`class-row ${isTriggered && cls.is_emergency ? 'active-detection' : ''} ${
                isTop ? 'top-class' : ''
              }`}
            >
              <div className="class-row-header">
                <div className="class-info">
                  <span className="class-icon">{CLASS_ICONS[cls.name] || '🔊'}</span>
                  <span className="class-name">{cls.display_name}</span>
                  {cls.is_emergency ? (
                    <span className="class-badge emergency">Emergency Event</span>
                  ) : (
                    <span className="class-badge normal">Ambient</span>
                  )}
                  {isScream && isScreamingDetected && (
                    <span className="badge-alert text-[10px] py-0.5 px-2">
                      VOCAL DISTRESS
                    </span>
                  )}
                  {isTop && !isTriggered && (
                    <span className="text-[10px] text-cyan-400 font-mono font-semibold">
                      [LEADING]
                    </span>
                  )}
                </div>

                <div className="class-score-text">
                  <span className={isTriggered && cls.is_emergency ? 'text-red-400 font-bold' : ''}>
                    {(score * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Progress Track with Calibrated Threshold Pin */}
              <div className="progress-track" title={`Threshold: ${(threshold * 100).toFixed(0)}%`}>
                <div
                  className={`progress-fill ${cls.is_emergency && isTriggered ? 'alarm-fill' : ''}`}
                  style={{ width: `${Math.min(100, Math.max(0, score * 100))}%` }}
                />
                <div
                  className="threshold-marker"
                  style={{ left: `${threshold * 100}%` }}
                >
                  <div className="threshold-tooltip">
                    Thresh: {(threshold * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-2 text-[11px] text-slate-500 leading-relaxed border-t border-slate-800/60 pt-2 flex items-start gap-1.5">
        <Shield size={13} className="text-slate-400 flex-shrink-0 mt-0.5" />
        <span>
          Scores represent neural network posterior probabilities on 0.96s audio windows.
          Acoustic patterns trigger warnings when score exceeds the class threshold for 2 consecutive windows.
        </span>
      </div>
    </div>
  );
};
