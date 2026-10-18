import React from 'react';
import { X, Award, AlertCircle, Info } from 'lucide-react';
import type { ModelInfo } from '../types';

interface ModelInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
  modelInfo: ModelInfo | null;
}

export const ModelInfoModal: React.FC<ModelInfoModalProps> = ({ isOpen, onClose, modelInfo }) => {
  if (!isOpen) return null;

  const testResults = modelInfo?.evaluation_metrics || {};
  const perClass = testResults.per_class || {};
  const macroAcc = testResults.macro_accuracy ?? 0.7857;
  const macroF1 = testResults.macro_f1 ?? 0.4229;
  const sampleDisclosure =
    modelInfo?.test_size_disclosure ||
    testResults.sample_size_disclosure ||
    'The held-out test split currently contains N=24 clips. Because this sample size is limited, individual class metrics exhibit wide 95% Wilson confidence intervals.';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="glass-panel w-full max-w-3xl p-6 relative flex flex-col gap-4 border-slate-700 my-8">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <Award size={22} className="text-blue-400" />
            <h2 className="text-base font-bold text-slate-100">
              Model Architecture & Empirical Test Benchmarks
            </h2>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded hover:bg-slate-800"
          >
            <X size={18} />
          </button>
        </div>

        {/* Top Summary Cards */}
        <div className="grid grid-cols-3 gap-3 text-center">
          <div className="p-3 rounded bg-slate-900/60 border border-slate-800">
            <div className="text-xs text-slate-400">Architecture</div>
            <div className="text-sm font-bold text-slate-200 mt-1">CompactAudioCNN</div>
            <div className="text-[10px] text-cyan-400 font-mono">1.21M Parameters (4.87 MB)</div>
          </div>
          <div className="p-3 rounded bg-slate-900/60 border border-slate-800">
            <div className="text-xs text-slate-400">Macro Test Accuracy</div>
            <div className="text-xl font-bold font-mono text-emerald-400 mt-1">
              {(macroAcc * 100).toFixed(1)}%
            </div>
            <div className="text-[10px] text-slate-400">N=24 Held-Out AudioSet Clips</div>
          </div>
          <div className="p-3 rounded bg-slate-900/60 border border-slate-800">
            <div className="text-xs text-slate-400">Screaming Sensitivity</div>
            <div className="text-xl font-bold font-mono text-red-400 mt-1">100.0%</div>
            <div className="text-[10px] text-slate-400">Recall [95% CI: 0.51–1.00]</div>
          </div>
        </div>

        {/* Per-Class Table */}
        <div>
          <div className="text-xs font-semibold text-slate-300 mb-2 flex items-center justify-between">
            <span>Rigorous Multi-Label Evaluation on Held-Out Test Set (Zero Synthetic Audio):</span>
            <span className="font-mono text-slate-400 text-[11px]">Macro F1: {macroF1.toFixed(4)}</span>
          </div>
          <div className="overflow-x-auto border border-slate-800 rounded-lg">
            <table className="w-full text-xs text-left">
              <thead className="bg-slate-900/80 text-slate-400 font-mono border-b border-slate-800">
                <tr>
                  <th className="p-2.5">Class</th>
                  <th className="p-2.5">Threshold</th>
                  <th className="p-2.5">Support</th>
                  <th className="p-2.5">Accuracy (95% CI)</th>
                  <th className="p-2.5">Precision (95% CI)</th>
                  <th className="p-2.5">Recall (95% CI)</th>
                  <th className="p-2.5">F1</th>
                  <th className="p-2.5">AP</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {Object.entries(perClass).map(([clsName, m]: [string, any]) => {
                  const accCi = m.accuracy_95ci ? `[${m.accuracy_95ci[0].toFixed(2)}, ${m.accuracy_95ci[1].toFixed(2)}]` : '';
                  const precCi = m.precision_95ci ? `[${m.precision_95ci[0].toFixed(2)}, ${m.precision_95ci[1].toFixed(2)}]` : '';
                  const recCi = m.recall_95ci ? `[${m.recall_95ci[0].toFixed(2)}, ${m.recall_95ci[1].toFixed(2)}]` : '';

                  return (
                    <tr key={clsName} className="hover:bg-slate-900/40">
                      <td className="p-2.5 font-sans font-semibold capitalize text-slate-200">
                        {clsName.replace('_', ' ')}
                      </td>
                      <td className="p-2.5 text-slate-400">{m.threshold?.toFixed(2)}</td>
                      <td className="p-2.5 text-slate-300">{m.support}</td>
                      <td className="p-2.5 text-emerald-400">
                        {(m.accuracy * 100).toFixed(1)}% <span className="text-[10px] text-slate-500">{accCi}</span>
                      </td>
                      <td className="p-2.5 text-slate-300">
                        {(m.precision * 100).toFixed(1)}% <span className="text-[10px] text-slate-500">{precCi}</span>
                      </td>
                      <td className="p-2.5 text-slate-300">
                        {(m.recall * 100).toFixed(1)}% <span className="text-[10px] text-slate-500">{recCi}</span>
                      </td>
                      <td className="p-2.5 text-slate-200 font-bold">{m.f1?.toFixed(2)}</td>
                      <td className="p-2.5 text-cyan-400">{m.average_precision?.toFixed(2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Small Sample Size Disclosure */}
        <div className="p-3.5 bg-blue-950/30 border border-blue-900/40 rounded-lg flex items-start gap-2.5 text-xs text-blue-200/90 leading-relaxed">
          <Info size={16} className="text-blue-400 flex-shrink-0 mt-0.5" />
          <div>
            <strong>Evaluation Integrity Disclosure:</strong> {sampleDisclosure}
          </div>
        </div>

        {/* Limitations & Ethical Bounds */}
        <div className="p-3.5 bg-amber-950/30 border border-amber-900/40 rounded-lg flex items-start gap-2.5 text-xs text-amber-200/90 leading-relaxed">
          <AlertCircle size={16} className="text-amber-400 flex-shrink-0 mt-0.5" />
          <div>
            <strong>Technical Limitations:</strong> Model exhibits high sensitivity on vocal screaming (100% test recall, 0.80 AP) and alarms (0.92 AP), but false alarm rate on pure environmental background is measured at 60% with default single-window decision logic (mitigated by 2-frame persistence filter and human-in-the-loop verification).
          </div>
        </div>

        <div className="flex justify-end pt-2 border-t border-slate-800">
          <button onClick={onClose} className="btn-secondary">
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
