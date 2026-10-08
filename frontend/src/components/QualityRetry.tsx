import React from 'react';
import { AlertTriangle, RefreshCw, CheckCircle2, HelpCircle } from 'lucide-react';
import { QualityDetails } from '../types/api';

interface QualityRetryProps {
  quality: QualityDetails;
  onReset: () => void;
}

export const QualityRetry: React.FC<QualityRetryProps> = ({ quality, onReset }) => {
  return (
    <div className="bg-amber-50 rounded-2xl border-2 border-amber-300 p-5 space-y-5 shadow-sm">
      {/* Header */}
      <div className="flex items-start gap-3">
        <div className="p-2.5 bg-amber-500 text-white rounded-xl flex-shrink-0 shadow-sm">
          <AlertTriangle className="w-6 h-6" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-amber-900 leading-tight">
            Video Quality Needs Improvement
          </h2>
          <p className="text-sm text-amber-800 mt-1 font-medium">
            The screening model requires clearer video evidence before evaluating gait patterns.
          </p>
        </div>
      </div>

      {/* Metric scores */}
      <div className="bg-white/90 rounded-xl p-4 border border-amber-200 grid grid-cols-2 gap-3 text-xs">
        <div>
          <span className="text-slate-500 block">Overall Quality Score</span>
          <span className="text-base font-bold text-slate-800">
            {Math.round(quality.quality_score * 100)}%
          </span>
        </div>
        <div>
          <span className="text-slate-500 block">Keypoint Coverage</span>
          <span className="text-base font-bold text-slate-800">
            {Math.round(quality.keypoint_coverage * 100)}%
          </span>
        </div>
        <div>
          <span className="text-slate-500 block">Motion Quality</span>
          <span className="text-base font-bold text-slate-800">
            {Math.round(quality.motion_quality * 100)}%
          </span>
        </div>
        <div>
          <span className="text-slate-500 block">Framing Quality</span>
          <span className="text-base font-bold text-slate-800">
            {Math.round(quality.framing_quality * 100)}%
          </span>
        </div>
      </div>

      {/* Detected issues */}
      {quality.issues && quality.issues.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs uppercase font-bold text-amber-900 tracking-wider">
            Detected Issues:
          </h4>
          <ul className="space-y-1.5">
            {quality.issues.map((issue, idx) => (
              <li key={idx} className="flex items-center gap-2 text-sm text-amber-900 bg-amber-100/70 px-3 py-1.5 rounded-lg">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-600 flex-shrink-0" />
                <span>{issue}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Actionable guidance */}
      {quality.user_guidance && quality.user_guidance.length > 0 && (
        <div className="bg-amber-100/80 rounded-xl p-4 border border-amber-300 space-y-2">
          <div className="flex items-center gap-2 text-amber-900 font-bold text-sm">
            <HelpCircle className="w-4 h-4 text-amber-700" />
            <span>How to get a successful screening:</span>
          </div>
          <ul className="space-y-1.5">
            {quality.user_guidance.map((guide, idx) => (
              <li key={idx} className="flex items-start gap-2 text-xs text-amber-900 font-medium">
                <CheckCircle2 className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
                <span>{guide}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Re-record Action */}
      <button
        onClick={onReset}
        className="w-full btn-primary py-3.5 flex items-center justify-center gap-2 font-bold text-base shadow-md"
      >
        <RefreshCw className="w-5 h-5" />
        Record or Select New Video
      </button>
    </div>
  );
};
