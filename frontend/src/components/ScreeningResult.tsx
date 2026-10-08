import React from 'react';
import { ShieldCheck, AlertTriangle, HelpCircle, RefreshCw, ChevronRight } from 'lucide-react';
import { InferenceDetails } from '../types/api';

interface ScreeningResultProps {
  inference: InferenceDetails;
  resultSummary: string;
  onReset: () => void;
}

export const ScreeningResult: React.FC<ScreeningResultProps> = ({
  inference,
  resultSummary,
  onReset
}) => {
  const { decision, calibrated_probability, confidence, uncertainty_margin } = inference;
  const riskPct = Math.round(calibrated_probability * 100);

  const getDecisionStyle = () => {
    switch (decision) {
      case 'NORMAL':
        return {
          bg: 'bg-emerald-50 border-emerald-300',
          badgeBg: 'bg-emerald-600 text-white',
          text: 'text-emerald-950',
          subtext: 'text-emerald-800',
          icon: ShieldCheck,
          title: 'Normal Gait Pattern',
          subtitle: 'No significant movement anomaly detected in this video segment.'
        };
      case 'LAMENESS_RISK':
        return {
          bg: 'bg-rose-50 border-rose-300',
          badgeBg: 'bg-rose-600 text-white',
          text: 'text-rose-950',
          subtext: 'text-rose-800',
          icon: AlertTriangle,
          title: 'Elevated Lameness Risk',
          subtitle: 'Biomechanical gait features show characteristics consistent with lameness risk.'
        };
      case 'INCONCLUSIVE':
      default:
        return {
          bg: 'bg-amber-50 border-amber-300',
          badgeBg: 'bg-amber-600 text-white',
          text: 'text-amber-950',
          subtext: 'text-amber-800',
          icon: HelpCircle,
          title: 'Inconclusive Screening Result',
          subtitle: 'Model predictions remain within the uncertainty margin. Further monitoring recommended.'
        };
    }
  };

  const style = getDecisionStyle();
  const Icon = style.icon;

  const getConfidenceBadge = () => {
    switch (confidence) {
      case 'HIGH':
        return 'bg-emerald-100 text-emerald-800 border-emerald-300';
      case 'MEDIUM':
        return 'bg-amber-100 text-amber-800 border-amber-300';
      case 'LOW':
      default:
        return 'bg-slate-100 text-slate-800 border-slate-300';
    }
  };

  return (
    <div className={`rounded-2xl border-2 p-5 space-y-5 shadow-sm ${style.bg}`}>
      {/* Top Banner */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <div className={`p-3 rounded-xl shadow-sm ${style.badgeBg}`}>
            <Icon className="w-7 h-7" />
          </div>
          <div>
            <h2 className={`text-2xl font-black tracking-tight ${style.text}`}>
              {style.title}
            </h2>
            <p className={`text-xs font-semibold mt-0.5 ${style.subtext}`}>
              {style.subtitle}
            </p>
          </div>
        </div>
      </div>

      {/* Primary Probability & Confidence Card */}
      <div className="bg-white/95 rounded-xl p-4 border border-slate-200 shadow-sm grid grid-cols-2 gap-4">
        <div>
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
            Lameness Risk Level
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-black text-slate-900">{riskPct}%</span>
            <span className="text-xs font-medium text-slate-500">
              calibrated prob.
            </span>
          </div>
          <div className="w-full bg-slate-100 h-2 rounded-full mt-2 overflow-hidden border border-slate-200">
            <div
              className={`h-full rounded-full transition-all duration-700 ${
                riskPct > 50 ? 'bg-rose-500' : riskPct > 30 ? 'bg-amber-500' : 'bg-emerald-500'
              }`}
              style={{ width: `${Math.max(5, riskPct)}%` }}
            />
          </div>
        </div>

        <div className="space-y-2 border-l border-slate-100 pl-4">
          <div>
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
              Confidence
            </span>
            <span
              className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-bold border mt-1 ${getConfidenceBadge()}`}
            >
              {confidence} CONFIDENCE
            </span>
          </div>
          <div>
            <span className="text-xs text-slate-500 block">Uncertainty Margin</span>
            <span className="text-xs font-mono font-semibold text-slate-700">
              ±{Math.round(uncertainty_margin * 100)}%
            </span>
          </div>
        </div>
      </div>

      {/* Summary Box */}
      {resultSummary && (
        <div className="bg-white/80 rounded-xl p-3.5 border border-slate-200 text-xs font-medium text-slate-700 leading-relaxed">
          <span className="font-bold text-slate-900 block mb-1">Screening Summary:</span>
          {resultSummary}
        </div>
      )}

      {/* Recommended Action Guidelines */}
      <div className="bg-white/90 rounded-xl p-4 border border-slate-200 space-y-2 text-xs">
        <h4 className="font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
          <ChevronRight className="w-4 h-4 text-emerald-600" />
          Recommended Next Steps:
        </h4>
        {decision === 'LAMENESS_RISK' && (
          <ul className="space-y-1.5 text-slate-700 font-medium pl-1">
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500 flex-shrink-0" />
              Schedule a visual and physical hoof inspection with herd personnel.
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500 flex-shrink-0" />
              Check for signs of swelling, sole bruising, or asymmetric weight bearing.
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500 flex-shrink-0" />
              Re-screen cow after 24–48 hours to monitor risk progression.
            </li>
          </ul>
        )}
        {decision === 'NORMAL' && (
          <ul className="space-y-1.5 text-slate-700 font-medium pl-1">
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 flex-shrink-0" />
              No immediate intervention required based on movement dynamics.
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 flex-shrink-0" />
              Maintain routine gait screening schedule during regular herd walks.
            </li>
          </ul>
        )}
        {decision === 'INCONCLUSIVE' && (
          <ul className="space-y-1.5 text-slate-700 font-medium pl-1">
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500 flex-shrink-0" />
              Record a fresh video with consistent lighting and parallel side view.
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500 flex-shrink-0" />
              Observe cow movement manually on a flat, non-slippery surface.
            </li>
          </ul>
        )}
      </div>

      {/* Action Button */}
      <button
        onClick={onReset}
        className="w-full btn-primary py-3.5 flex items-center justify-center gap-2 font-bold text-base shadow-md"
      >
        <RefreshCw className="w-5 h-5" />
        Screen Another Cow
      </button>
    </div>
  );
};
