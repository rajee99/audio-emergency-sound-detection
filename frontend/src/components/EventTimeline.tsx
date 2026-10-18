import React from 'react';
import type { TimelineEvent } from '../types';
import { Clock, Trash2 } from 'lucide-react';

interface EventTimelineProps {
  events: TimelineEvent[];
  onClear: () => void;
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

export const EventTimeline: React.FC<EventTimelineProps> = ({ events, onClear }) => {
  return (
    <div className="glass-panel timeline-card">
      <div className="card-header">
        <div className="card-title">
          <Clock size={18} className="text-cyan-400" />
          <span>Detection Event Timeline ({events.length})</span>
        </div>
        {events.length > 0 && (
          <button
            onClick={onClear}
            className="btn-secondary text-xs py-1 px-2.5 flex items-center gap-1 hover:text-red-400"
            title="Clear Event Log"
          >
            <Trash2 size={13} />
            <span>Clear</span>
          </button>
        )}
      </div>

      <div className="timeline-list">
        {events.length === 0 ? (
          <div className="empty-timeline">
            No emergency acoustic events detected in the current monitoring session.
          </div>
        ) : (
          events.map((evt, idx) => (
            <div
              key={evt.id}
              className={`timeline-item ${idx === 0 ? 'recent' : ''}`}
            >
              <div className="timeline-event-name">
                <span className="text-lg">{CLASS_ICONS[evt.eventName] || '🔊'}</span>
                <div>
                  <div className="font-semibold text-slate-200">
                    {evt.displayName}
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Threshold: {(evt.threshold * 100).toFixed(0)}%
                  </div>
                </div>
              </div>

              <div className="flex flex-col items-end gap-1">
                <span className="font-mono text-xs font-bold text-red-400 bg-red-950/60 px-2 py-0.5 rounded border border-red-800/40">
                  {(evt.score * 100).toFixed(1)}% Confidence
                </span>
                <span className="timeline-meta flex items-center gap-1">
                  <Clock size={11} />
                  +{evt.timestampSec.toFixed(2)}s ({evt.timeString})
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
