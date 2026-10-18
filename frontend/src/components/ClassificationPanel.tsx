import React from 'react';
import type { TargetClass, PersistenceClassState } from '../types';
import { Shield, Volume2, Layers } from 'lucide-react';

interface ClassificationPanelProps {
  classes: TargetClass[];
  rawScores: Record<string, number>;
  scores: Record<string, number>; // Smoothed
  thresholds: Record<string, number>;
  persistenceState: Record<string, PersistenceClassState>;
  topClass: string;
  isScreamingScoreHigh: boolean;
  isScreamingAlert: boolean;
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
  rawScores,
  scores,
  thresholds,
  persistenceState,
  topClass,
  isScreamingScoreHigh,
  isScreamingAlert,
}) => {
  return (
    <div className="glass-panel classification-card">
      <div className="card-header">
        <div className="card-title">
          <Volume2 size={18} className="text-blue-400" />
          <span>Acoustic Event Classification</span>
        </div>
        <div className="text-xs text-slate-400 flex items-center gap-1.5 font-mono">
          <Layers size={13} className="text-cyan-400" />
          <span>Raw + EMA Smoothed (0.96s)</span>
        </div>
      </div>

      <div className="flex flex-col gap-2.5">
        {classes.map((cls) => {
          const smoothedScore = scores[cls.name] ?? 0.0;
          const rawScore = rawScores[cls.name] ?? smoothedScore;
          const threshold = thresholds[cls.name] ?? cls.threshold;
          const isTriggered = smoothedScore >= threshold;
          const isTop = topClass === cls.name;
          const isScream = cls.name === 'screaming';

          const pState = persistenceState[cls.name] || {
            count: 0,
            required: cls.persistence_frames || 2,
            persisted: false,
          };

          return (
            <div
              key={cls.name}
              className={`class-row ${pState.persisted && cls.is_emergency ? 'active-detection' : ''} ${
                isTop ? 'top-class' : ''
              }`}
            >
              <div className="class-row-header">
                <div className="class-info">
                  <span className="class-icon">{CLASS_ICONS[cls.name] || '🔊'}</span>
                  <span className="class-name">{cls.display_name}</span>

                  {cls.is_emergency ? (
                    <span className="class-badge emergency">Emergency</span>
                  ) : (
                    <span className="class-badge normal">Ambient</span>
                  )}

                  {/* Screaming Distinct Indicators */}
                  {isScream && isScreamingAlert && (
                    <span className="badge-alert text-[10px] py-0.5 px-2">
                      VOCAL DISTRESS ALERT (PERSISTED)
                    </span>
                  )}
                  {isScream && !isScreamingAlert && isScreamingScoreHigh && (
                    <span className="text-[10px] bg-amber-950/80 text-amber-300 border border-amber-600/40 rounded py-0.5 px-1.5 font-mono">
                      HIGH RAW SPIKE ({pState.count}/{pState.required})
                    </span>
                  )}

                  {/* Other Emergency Persisted Badges */}
                  {!isScream && cls.is_emergency && pState.persisted && (
                    <span className="badge-alert text-[10px] py-0.5 px-2">
                      ALERT PERSISTED
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2 font-mono text-xs">
                  {/* Explicit Raw Score */}
                  <span className="text-slate-400 text-[11px]" title="Instantaneous model forward pass score">
                    Raw: {(rawScore * 100).toFixed(0)}%
                  </span>

                  {/* Smoothed Score */}
                  <span
                    className={`font-bold ${
                      pState.persisted && cls.is_emergency
                        ? 'text-red-400'
                        : isTriggered
                        ? 'text-amber-400'
                        : 'text-slate-200'
                    }`}
                  >
                    {(smoothedScore * 100).toFixed(1)}%
                  </span>

                  {/* Persistence Counter */}
                  {cls.is_emergency && pState.count > 0 && (
                    <span
                      className={`text-[10px] px-1 py-0.5 rounded ${
                        pState.persisted
                          ? 'bg-red-950 text-red-300 border border-red-800'
                          : 'bg-slate-800 text-slate-300'
                      }`}
                      title="Consecutive windows above threshold"
                    >
                      {pState.count}/{pState.required}
                    </span>
                  )}
                </div>
              </div>

              {/* Progress Track with Calibrated Threshold Pin */}
              <div className="progress-track" title={`Threshold: ${(threshold * 100).toFixed(0)}%`}>
                <div
                  className={`progress-fill ${cls.is_emergency && isTriggered ? 'alarm-fill' : ''}`}
                  style={{ width: `${Math.min(100, Math.max(0, smoothedScore * 100))}%` }}
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
          Scores reflect acoustic neural network probabilities (raw instantaneous vs 3-frame rolling EMA).
          Alerts require smoothed score $\ge$ threshold for $K \ge 2$ consecutive windows to filter non-emergency spikes.
        </span>
      </div>
    </div>
  );
};
