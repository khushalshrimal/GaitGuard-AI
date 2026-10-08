import React, { useEffect, useState } from 'react';
import { Loader2, Video, Activity, BrainCircuit, ShieldCheck } from 'lucide-react';

const STAGES = [
  { icon: Video, label: 'Decoding & Analyzing Video Quality...' },
  { icon: Activity, label: 'Extracting 17 Biomechanical Keypoints...' },
  { icon: BrainCircuit, label: 'Evaluating BiLSTM Temporal Gait Model...' },
  { icon: ShieldCheck, label: 'Calibrating Probability & Generating SHAP Evidence...' }
];

export const ProcessingState: React.FC = () => {
  const [currentStage, setCurrentStage] = useState(0);

  useEffect(() => {
    const timer1 = setTimeout(() => setCurrentStage(1), 1500);
    const timer2 = setTimeout(() => setCurrentStage(2), 3500);
    const timer3 = setTimeout(() => setCurrentStage(3), 5500);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
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
        <h3 className="text-xl font-bold text-slate-800">Screening in Progress</h3>
        <p className="text-sm font-medium text-emerald-800 bg-emerald-50 px-3 py-1.5 rounded-lg inline-block border border-emerald-200">
          {STAGES[currentStage].label}
        </p>
      </div>

      {/* Progress step indicators */}
      <div className="grid grid-cols-4 gap-2 pt-2">
        {STAGES.map((s, idx) => (
          <div
            key={idx}
            className={`h-2 rounded-full transition-all duration-500 ${
              idx <= currentStage ? 'bg-emerald-600' : 'bg-slate-200'
            }`}
          />
        ))}
      </div>

      <p className="text-xs text-slate-500 italic">
        Please hold on — AI gait analysis typically takes 3-8 seconds.
      </p>
    </div>
  );
};
