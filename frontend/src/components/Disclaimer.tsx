import React from 'react';
import { ShieldAlert } from 'lucide-react';

interface DisclaimerProps {
  customText?: string;
}

export const Disclaimer: React.FC<DisclaimerProps> = ({ customText }) => {
  return (
    <div className="bg-slate-100 rounded-xl border border-slate-300 p-3.5 text-xs text-slate-600 flex items-start gap-2.5 shadow-inner">
      <ShieldAlert className="w-5 h-5 text-slate-500 flex-shrink-0 mt-0.5" />
      <div className="space-y-0.5">
        <span className="font-bold text-slate-700 block uppercase tracking-wider text-[10px]">
          Screening System Disclaimer
        </span>
        <p className="leading-relaxed text-[11px]">
          {customText ||
            'GaitGuard AI is an AI-assisted cattle gait screening tool. This output is not a veterinary diagnosis. Consult a qualified veterinarian for formal medical diagnosis and treatment decisions.'}
        </p>
      </div>
    </div>
  );
};
