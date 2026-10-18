export interface TargetClass {
  name: string;
  display_name: string;
  is_emergency: boolean;
  threshold: number;
  persistence_frames?: number;
}

export interface ModelInfo {
  model_name: string;
  version: string;
  sample_rate: number;
  target_classes: TargetClass[];
  class_names: string[];
  window_duration_sec: number;
  hop_duration_sec: number;
  evaluation_metrics?: Record<string, any>;
  test_size_disclosure?: string;
}

export interface AlertData {
  event: string;
  score: number;
  threshold: number;
  persistence_count?: number;
  timestamp: number;
  message: string;
}

export interface PersistenceClassState {
  count: number;
  required: number;
  persisted: boolean;
}

export interface LivePredictionMessage {
  type: 'prediction' | 'error' | 'reset_ack' | 'config_ack';
  timestamp?: number;
  raw_scores?: Record<string, number>;
  scores?: Record<string, number>; // Smoothed
  thresholds?: Record<string, number>;
  persistence_state?: Record<string, PersistenceClassState>;
  top_class?: string;
  top_score?: number;
  top_raw_score?: number;
  is_emergency?: boolean;
  is_alert?: boolean;
  is_screaming_score_high?: boolean;
  is_screaming_alert?: boolean;
  alert?: AlertData | null;
  inference_latency_ms?: number;
  rms?: number;
  buffer_samples?: number;
  message?: string;
}

export interface TimelineEvent {
  id: string;
  eventName: string;
  displayName: string;
  timestampSec: number;
  score: number;
  threshold: number;
  timeString: string;
  isEmergency: boolean;
}

export interface FilePredictionWindow {
  start_sec: number;
  end_sec: number;
  raw_probabilities?: Record<string, number>;
  probabilities: Record<string, number>;
  persistence_counts?: Record<string, number>;
  detected_classes: string[];
  emergency_score: number;
}

export interface FilePredictResponse {
  filename: string;
  duration_sec: number;
  total_windows_processed: number;
  max_emergency_score: number;
  overall_detected_events: string[];
  alerts_emitted: Array<{
    timestamp_sec: number;
    event_category: string;
    confidence_score: number;
    detection_threshold: number;
    persistence_count?: number;
    message: string;
  }>;
  window_predictions: FilePredictionWindow[];
}
