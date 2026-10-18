export interface TargetClass {
  name: string;
  display_name: string;
  is_emergency: boolean;
  threshold: number;
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
}

export interface AlertData {
  event: string;
  score: number;
  threshold: number;
  timestamp: number;
  message: string;
}

export interface LivePredictionMessage {
  type: 'prediction' | 'error' | 'reset_ack' | 'config_ack';
  timestamp?: number;
  scores?: Record<string, number>;
  thresholds?: Record<string, number>;
  top_class?: string;
  top_score?: number;
  is_emergency?: boolean;
  is_alert?: boolean;
  is_screaming_detected?: boolean;
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
  probabilities: Record<string, number>;
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
    message: string;
  }>;
  window_predictions: FilePredictionWindow[];
}
