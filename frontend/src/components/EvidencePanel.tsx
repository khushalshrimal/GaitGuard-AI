import React, { useState } from 'react';
import { Activity, BarChart2, Info, ChevronDown, ChevronUp, AlertCircle } from 'lucide-react';
import { ExplanationDetails } from '../types/api';

interface EvidencePanelProps {
  explanation?: ExplanationDetails | null;
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({ explanation }) => {
  const [showDetails, setShowDetails] = useState(false);

  if (!explanation || !explanation.available) {
    return (
      <div className="bg-slate-50 rounded-2xl border border-slate-200 p-4 text-slate-500 text-xs flex items-center gap-2">
        <Info className="w-4 h-4 text-slate-400 flex-shrink-0" />
        <span>
          {explanation?.reason || 'Detailed SHAP evidence is unavailable for this sample.'}
        </span>
      </div>
    );
  }

  const topContributors = explanation.top_contributors || [];
  const derivedEvidence = explanation.derived_gait_evidence || [];

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 space-y-4 shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-slate-900 font-bold text-base">
          <BarChart2 className="w-5 h-5 text-emerald-700" />
          <h3>Model Gait Evidence & Attribution</h3>
        </div>
        <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-200">
          SHAP Explained
        </span>
      </div>

      {/* Level 1: Key Feature Attributions */}
      <div className="space-y-3">
        <h4 className="text-xs uppercase font-bold text-slate-500 tracking-wider">
          Level 1: Key Biomechanical Attributions
        </h4>
        <div className="space-y-2">
          {topContributors.slice(0, 5).map((item, idx) => {
            const isIncrease = item.direction === 'INCREASES_RISK';
            const absAttr = Math.abs(item.attribution);
            const percentage = Math.min(100, Math.round(absAttr * 500)); // normalized bar display

            return (
              <div key={idx} className="bg-slate-50 rounded-xl p-3 border border-slate-200 text-xs space-y-1.5">
                <div className="flex items-center justify-between font-semibold">
                  <span className="text-slate-800 capitalize">
                    {item.feature_name.replace(/_/g, ' ')}
                  </span>
                  <span
                    className={`font-bold ${
                      isIncrease ? 'text-rose-700' : 'text-emerald-700'
                    }`}
                  >
                    {isIncrease ? '+ Risk Contributor' : '- Risk Neutralizer'}
                  </span>
                </div>

                {/* Attribution visual bar */}
                <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden flex">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      isIncrease ? 'bg-rose-500' : 'bg-emerald-500'
                    }`}
                    style={{ width: `${Math.max(8, percentage)}%` }}
                  />
                </div>

                <div className="flex items-center justify-between text-slate-500 text-[11px]">
                  <span>Region: <strong className="text-slate-700 capitalize">{item.body_region}</strong></span>
                  <span>Modality: <strong className="text-slate-700 capitalize">{item.modality}</strong></span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Level 2: Derived Gait Domain Evidence (Expandable) */}
      {derivedEvidence.length > 0 && (
        <div className="pt-2 border-t border-slate-100">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="w-full flex items-center justify-between text-xs font-bold text-slate-700 py-2 hover:text-emerald-700 transition-colors"
          >
            <span className="flex items-center gap-1.5">
              <Activity className="w-4 h-4 text-emerald-600" />
              Level 2: Domain-Specific Gait Metrics ({derivedEvidence.length})
            </span>
            {showDetails ? (
              <ChevronUp className="w-4 h-4 text-slate-500" />
            ) : (
              <ChevronDown className="w-4 h-4 text-slate-500" />
            )}
          </button>

          {showDetails && (
            <div className="space-y-2 mt-2 pt-2 border-t border-slate-100">
              {derivedEvidence.map((ev, idx) => (
                <div key={idx} className="bg-emerald-50/60 rounded-xl p-3 border border-emerald-100 text-xs space-y-1">
                  <div className="flex items-center justify-between font-bold text-slate-800">
                    <span>{ev.name}</span>
                    <span className="font-mono text-emerald-900">{ev.value.toFixed(3)}</span>
                  </div>
                  <p className="text-slate-600 text-[11px] leading-relaxed">
                    {ev.interpretation}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Scientific Transparency Note */}
      <div className="bg-slate-50 rounded-xl p-3 border border-slate-200 text-[11px] text-slate-500 flex items-start gap-2">
        <AlertCircle className="w-4 h-4 text-slate-400 flex-shrink-0 mt-0.5" />
        <span>
          Attributions represent keypoint and gait velocity feature weightings from the Phase 7 BiLSTM model via exact Deep SHAP explanations.
        </span>
      </div>
    </div>
  );
};
