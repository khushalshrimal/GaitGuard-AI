import React from 'react';
import { Camera, Sun, Footprints, MoveRight, Eye } from 'lucide-react';

export const CaptureGuide: React.FC = () => {
  return (
    <div className="card space-y-4 border-emerald-100 bg-emerald-50/40">
      <div className="flex items-center gap-2 text-emerald-900 font-bold text-base">
        <Eye className="w-5 h-5 text-emerald-700" />
        <h2>Recommended Recording Guidance</h2>
      </div>

      <ul className="space-y-2.5 text-sm text-slate-700">
        <li className="flex items-start gap-3">
          <div className="p-1.5 bg-white rounded-lg text-emerald-700 font-bold text-xs mt-0.5 shadow-sm border border-emerald-100">1</div>
          <span><strong className="text-slate-900">Perpendicular Side Profile:</strong> Record the cow from a 90° side view while it walks past.</span>
        </li>
        <li className="flex items-start gap-3">
          <div className="p-1.5 bg-white rounded-lg text-emerald-700 font-bold text-xs mt-0.5 shadow-sm border border-emerald-100">2</div>
          <span><strong className="text-slate-900">Distance & Framing:</strong> Stand 4 to 6 meters (12–20 ft) away so the full head, body, and hooves are in frame.</span>
        </li>
        <li className="flex items-start gap-3">
          <div className="p-1.5 bg-white rounded-lg text-emerald-700 font-bold text-xs mt-0.5 shadow-sm border border-emerald-100">3</div>
          <span><strong className="text-slate-900">Continuous Walk:</strong> Ensure cow walks continuously for 3 to 5 seconds (5–10 strides).</span>
        </li>
        <li className="flex items-start gap-3">
          <div className="p-1.5 bg-white rounded-lg text-emerald-700 font-bold text-xs mt-0.5 shadow-sm border border-emerald-100">4</div>
          <span><strong className="text-slate-900">Steady Camera & Good Light:</strong> Hold phone steady in well-lit daylight or alleyways.</span>
        </li>
      </ul>
    </div>
  );
};
