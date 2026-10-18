import React, { useState } from 'react';
import { Upload, FileAudio, CheckCircle2, AlertTriangle, Play, Pause, Loader2 } from 'lucide-react';
import { uploadPredictFile } from '../services/api';
import type { FilePredictResponse } from '../types';

export const FileTestMode: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<FilePredictResponse | null>(null);
  const audioRef = React.useRef<HTMLAudioElement | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setAudioUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleRunInference = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    try {
      const data = await uploadPredictFile(selectedFile);
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'File inference failed.');
    } finally {
      setLoading(false);
    }
  };

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  return (
    <div className="glass-panel p-6 flex flex-col gap-5">
      <div className="card-header">
        <div className="card-title">
          <FileAudio size={20} className="text-cyan-400" />
          <span>Controlled Audio File Evaluation</span>
        </div>
        <span className="text-xs px-2.5 py-1 rounded bg-blue-950/80 border border-blue-500/30 text-blue-300 font-mono">
          SOURCE: PRERECORDED WAV FILE
        </span>
      </div>

      {/* Upload Zone */}
      <label className="upload-dropzone block">
        <input
          type="file"
          accept=".wav,audio/wav"
          className="hidden"
          onChange={handleFileChange}
        />
        <Upload className="upload-icon" />
        <div className="text-sm font-semibold text-slate-200">
          {selectedFile ? selectedFile.name : 'Click or drag a .WAV audio file to test'}
        </div>
        <div className="text-xs text-slate-400 mt-1">
          Standard 16 kHz or multi-channel WAV will be resampled & converted to mono on the backend
        </div>
      </label>

      {/* Controls & Audio Player */}
      {selectedFile && audioUrl && (
        <div className="flex items-center justify-between p-4 rounded-lg bg-slate-900/70 border border-slate-800">
          <div className="flex items-center gap-3">
            <button
              onClick={togglePlay}
              className="w-10 h-10 rounded-full bg-blue-600 hover:bg-blue-500 flex items-center justify-center text-white"
            >
              {isPlaying ? <Pause size={18} /> : <Play size={18} className="ml-0.5" />}
            </button>
            <audio
              ref={audioRef}
              src={audioUrl}
              onEnded={() => setIsPlaying(false)}
              className="hidden"
            />
            <div>
              <div className="text-sm font-semibold text-slate-200">{selectedFile.name}</div>
              <div className="text-xs text-slate-400 font-mono">
                {(selectedFile.size / 1024).toFixed(1)} KB
              </div>
            </div>
          </div>

          <button
            onClick={handleRunInference}
            disabled={loading}
            className="btn-primary"
          >
            {loading ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                <span>Running Inference...</span>
              </>
            ) : (
              <span>Analyze with CompactAudioCNN</span>
            )}
          </button>
        </div>
      )}

      {error && (
        <div className="p-3 bg-red-950/60 border border-red-800/60 rounded text-red-300 text-xs flex items-center gap-2">
          <AlertTriangle size={15} />
          <span>{error}</span>
        </div>
      )}

      {/* Result Display */}
      {result && (
        <div className="file-result-card flex flex-col gap-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <CheckCircle2 size={18} className="text-emerald-400" />
              <span className="font-semibold text-sm text-slate-200">Analysis Complete</span>
            </div>
            <div className="text-xs font-mono text-slate-400">
              Duration: {result.duration_sec}s | Windows: {result.total_windows_processed}
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
            <div className="p-3 rounded bg-slate-900/60 border border-slate-800">
              <div className="text-xs text-slate-400">Max Emergency Score</div>
              <div className="text-xl font-bold font-mono text-cyan-400 mt-1">
                {(result.max_emergency_score * 100).toFixed(1)}%
              </div>
            </div>
            <div className="p-3 rounded bg-slate-900/60 border border-slate-800">
              <div className="text-xs text-slate-400">Detected Classes</div>
              <div className="text-sm font-bold text-slate-200 mt-1.5 capitalize">
                {result.overall_detected_events.length > 0
                  ? result.overall_detected_events.join(', ')
                  : 'None (Ambient)'}
              </div>
            </div>
            <div className="p-3 rounded bg-slate-900/60 border border-slate-800">
              <div className="text-xs text-slate-400">Alerts Emitted</div>
              <div className="text-xl font-bold font-mono text-red-400 mt-1">
                {result.alerts_emitted.length}
              </div>
            </div>
            <div className="p-3 rounded bg-slate-900/60 border border-slate-800">
              <div className="text-xs text-slate-400">Silence Gating</div>
              <div className="text-xs font-mono text-cyan-400 mt-2">
                {result.silence_gated_windows || 0} / {result.total_windows_processed} GATED
              </div>
            </div>
          </div>

          {result.alerts_emitted.length > 0 && (
            <div className="flex flex-col gap-2 mt-2">
              <div className="text-xs font-semibold text-slate-300">Emitted Alerts:</div>
              {result.alerts_emitted.map((a, i) => (
                <div
                  key={i}
                  className="p-3 rounded bg-red-950/40 border border-red-900/50 flex flex-col gap-1 text-xs"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-red-300">
                      🚨 {a.event_category.toUpperCase()}: {a.message}
                    </span>
                    <span className="font-mono text-slate-400">
                      Time: +{a.timestamp_sec.toFixed(2)}s | Confidence: {(a.confidence_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  {a.trigger_reason && (
                    <div className="text-[11px] font-mono text-cyan-400/90 italic">
                      Trigger reason: {a.trigger_reason}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

        </div>
      )}
    </div>
  );
};
