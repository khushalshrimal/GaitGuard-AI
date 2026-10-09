import React, { useEffect, useState } from 'react';
import { Loader2, Video, CheckCircle2, Activity, BrainCircuit, ShieldCheck } from 'lucide-react';

const STAGES = [
  { icon: Video, label: 'Stage 1: Video Decoded & Format Verified' },
  { icon: CheckCircle2, label: 'Stage 2: Usable Frames Found & Uniformly Sampled (128 Window)' },
  { icon: Activity, label: 'Stage 3: Quadruped Cow / 17 Keypoints Detected' },
  { icon: BrainCircuit, label: 'Stage 4: Temporal Gait Features Prepared (128x76 Representation)' },
  { icon: ShieldCheck, label: 'Stage 5: BiLSTM Inference, Platt Calibration & SHAP Evidence Completed' }
];

export const ProcessingState: React.FC = () => {
  const [currentStage, setCurrentStage] = useState(0);

  useEffect(() => {
    const timer1 = setTimeout(() => setCurrentStage(1), 1000);
    const timer2 = setTimeout(() => setCurrentStage(2), 2200);
    const timer3 = setTimeout(() => setCurrentStage(3), 3600);
    const timer4 = setTimeout(() => setCurrentStage(4), 5000);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);
    };
  }, []);

  const Icon = STAGES[currentStage].icon;

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 text-center space-y-6">
      <div className="relative w-20 h-20 mx-auto flex items-center justify-center bg-emerald-50 rounded-full border border-emerald-200">
        <Icon className="w-10 h-10 text-emerald-700 animate-pulse" />
        <Loader2 className="absolute inset-0 w-full h-full text-emerald-600 animate-spin opacity-40" />
      </div>

      <div className="space-y-2">
        <h3 className="text-xl font-bold text-slate-800">Processing Video Pipeline</h3>
        <p className="text-sm font-medium text-emerald-800 bg-emerald-50 px-3 py-1.5 rounded-lg inline-block border border-emerald-200">
          {STAGES[currentStage].label}
        </p>
      </div>

      {/* Progress step indicators */}
      <div className="grid grid-cols-5 gap-1.5 pt-2">
        {STAGES.map((_, idx) => (
          <div
            key={idx}
            className={`h-2 rounded-full transition-all duration-500 ${
              idx <= currentStage ? 'bg-emerald-600' : 'bg-slate-200'
            }`}
          />
        ))}
      </div>

      <div className="bg-slate-50 rounded-xl p-3 border border-slate-200 text-left text-xs space-y-1.5">
        <div className="font-semibold text-slate-700">Measured Pipeline Progress:</div>
        <ul className="space-y-1 text-slate-600">
          {STAGES.slice(0, currentStage + 1).map((stage, idx) => (
            <li key={idx} className="flex items-center gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 flex-shrink-0" />
              <span>{stage.label}</span>
            </li>
          ))}
        </ul>
      </div>

      <p className="text-xs text-slate-500 italic">
        Please hold on — evidence-based gait screening typically takes 3-8 seconds.
      </p>
    </div>
  );
};
