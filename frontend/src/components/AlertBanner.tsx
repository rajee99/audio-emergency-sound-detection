import { AlertTriangle, Clock, Target, Info } from 'lucide-react';
import type { AlertData } from '../types';

interface AlertBannerProps {
  alert: AlertData | null;
  onDismiss: () => void;
}

export const AlertBanner: React.FC<AlertBannerProps> = ({ alert, onDismiss }) => {
  if (!alert) return null;

  const displayCategory = alert.event
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');

  return (
    <div className="glass-panel-alert alert-banner">
      <div className="alert-icon-pulse">
        <AlertTriangle size={24} />
      </div>

      <div className="flex-1">
        <div className="flex items-center justify-between">
          <div className="alert-title">
            Potential acoustic warning detected — human verification required
          </div>
          <button
            onClick={onDismiss}
            className="text-slate-400 hover:text-white text-xs px-2 py-1 rounded bg-black/30 hover:bg-black/50"
          >
            Dismiss
          </button>
        </div>

        <div className="alert-message">
          <strong>Acoustic Event:</strong> {displayCategory} detected ({alert.message || 'Exceeded calibrated persistence rule'})
        </div>

        <div className="alert-meta">
          <span className="flex items-center gap-1.5">
            <Clock size={12} className="text-red-400" />
            Timestamp: +{alert.timestamp.toFixed(2)}s
          </span>
          <span className="flex items-center gap-1.5">
            <Target size={12} className="text-red-400" />
            Confidence: {(alert.score * 100).toFixed(1)}%
          </span>
          <span>Calibrated Threshold: {(alert.threshold * 100).toFixed(0)}%</span>
          <span>Persistence: 2 consecutive windows</span>
        </div>

        <div className="alert-disclaimer flex items-center gap-1.5">
          <Info size={12} />
          <span>
            Ethical Constraint: Detections are acoustic indicators only, never proof of a crime, assault, or emergency.
          </span>
        </div>
      </div>
    </div>
  );
};
