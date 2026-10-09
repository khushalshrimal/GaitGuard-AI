import React from 'react';
import { Video, ShieldCheck, Cpu, BarChart2, CheckCircle2, ArrowRight } from 'lucide-react';

export const HowItWorks: React.FC = () => {
  const steps = [
    {
      num: '01',
      title: 'Video Capture & Quality Gate',
      desc: 'Upload a 3–15 second side-profile video of a walking cow. GaitGuard automatically verifies lighting, camera stability, framing, and keypoint coverage.',
      icon: Video,
      color: 'bg-emerald-50 text-emerald-700 border-emerald-200'
    },
    {
      num: '02',
      title: 'Anatomical Keypoint Extraction',
      desc: '17 anatomical keypoints (withers, spine, hocks, fetlocks, hoofs) are tracked across frames and cleaned using Savitzky-Golay filtering.',
      icon: Cpu,
      color: 'bg-emerald-50 text-emerald-700 border-emerald-200'
    },
    {
      num: '03',
      title: 'Biomechanical Feature Engineering',
      desc: 'Computes 76 normalized gait features including back-arch curvature, stance asymmetry, head nodding amplitude, and stride length variance.',
      icon: BarChart2,
      color: 'bg-emerald-50 text-emerald-700 border-emerald-200'
    },
    {
      num: '04',
      title: 'BiLSTM Temporal AI Model',
      desc: 'A Bidirectional LSTM neural network analyzes temporal movement patterns across time windows to estimate raw lameness risk.',
      icon: Cpu,
      color: 'bg-emerald-50 text-emerald-700 border-emerald-200'
    },
    {
      num: '05',
      title: 'Sigmoid Probability Calibration',
      desc: 'Raw predictions are calibrated using Platt Sigmoid scaling, ensuring probability estimates accurately reflect real-world risk confidence.',
      icon: ShieldCheck,
      color: 'bg-emerald-50 text-emerald-700 border-emerald-200'
    },
    {
      num: '06',
      title: 'Uncertainty Triage & SHAP Evidence',
      desc: 'Applies screening threshold (\u03c4 = 0.34) and inconclusive bounds ([0.24, 0.44]) alongside exact Deep SHAP feature attributions.',
      icon: CheckCircle2,
      color: 'bg-emerald-50 text-emerald-700 border-emerald-200'
    }
  ];

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Overview Banner */}
      <div className="bg-emerald-800 text-white p-6 rounded-2xl shadow-sm space-y-3">
        <h2 className="text-2xl font-black tracking-tight">How GaitGuard AI Works</h2>
        <p className="text-sm text-emerald-100 font-medium leading-relaxed">
          GaitGuard AI transforms standard smartphone videos of walking cattle into quantitative biomechanical gait insights using computer vision and temporal neural networks.
        </p>
      </div>

      {/* Process Pipeline Cards */}
      <div className="space-y-3">
        {steps.map((step, idx) => {
          const Icon = step.icon;
          return (
            <div key={idx} className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm flex items-start gap-4">
              <div className={`p-3 rounded-xl border flex-shrink-0 ${step.color}`}>
                <Icon className="w-6 h-6" />
              </div>
              <div className="space-y-1 flex-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-black text-emerald-700 tracking-wider">STEP {step.num}</span>
                </div>
                <h3 className="font-extrabold text-slate-900 text-base">{step.title}</h3>
                <p className="text-xs font-medium text-slate-600 leading-relaxed">{step.desc}</p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Target Users Banner */}
      <div className="bg-white rounded-2xl p-5 border border-slate-200 space-y-3 shadow-sm">
        <h3 className="font-extrabold text-slate-900 text-base flex items-center gap-2">
          <ArrowRight className="w-5 h-5 text-emerald-600" />
          Designed For Field Operations
        </h3>
        <ul className="space-y-2 text-xs text-slate-700 font-medium">
          <li className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0" />
            <strong>Dairy & Beef Farmers:</strong> Early screening during daily herd walks.
          </li>
          <li className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0" />
            <strong>Field Livestock Technicians:</strong> Rapid digital movement recording.
          </li>
          <li className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0" />
            <strong>Veterinary Professionals:</strong> Quantitative gait evidence support.
          </li>
        </ul>
      </div>
    </div>
  );
};
