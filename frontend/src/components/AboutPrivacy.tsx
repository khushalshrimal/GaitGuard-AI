import React from 'react';
import { Shield, Lock, Trash2, AlertCircle, FileText } from 'lucide-react';

export const AboutPrivacy: React.FC = () => {
  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header Banner */}
      <div className="bg-slate-900 text-white p-6 rounded-2xl shadow-sm space-y-3">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-slate-800 rounded-xl">
            <Shield className="w-6 h-6 text-emerald-400" />
          </div>
          <div>
            <h2 className="text-2xl font-black tracking-tight">Privacy & Security Governance</h2>
            <p className="text-xs text-slate-300 font-medium">Zero-Persistence Data Policy & Legal Disclaimer</p>
          </div>
        </div>
      </div>

      {/* Privacy Guarantees */}
      <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm space-y-4">
        <h3 className="font-extrabold text-slate-900 text-lg flex items-center gap-2">
          <Lock className="w-5 h-5 text-emerald-700" />
          Zero Temporary Video Retention
        </h3>

        <div className="space-y-3 text-xs text-slate-700 font-medium leading-relaxed">
          <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 flex items-start gap-3">
            <Trash2 className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
            <div>
              <strong className="text-slate-900 block mb-0.5">Immediate Post-Inference Cleanup</strong>
              Uploaded video files are processed in isolated operating system temporary space and deleted immediately inside a deterministic execution block after keypoint extraction completes.
            </div>
          </div>

          <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 flex items-start gap-3">
            <Shield className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
            <div>
              <strong className="text-slate-900 block mb-0.5">Zero PII Collection</strong>
              No farmer names, phone numbers, GPS locations, farm address data, or owner credentials are required, collected, or stored on disk or server databases.
            </div>
          </div>

          <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 flex items-start gap-3">
            <Lock className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
            <div>
              <strong className="text-slate-900 block mb-0.5">Magic Byte Header Verification</strong>
              Server-side input validation inspects raw file header bytes (`ftyp`, `RIFF`, `WebM`), ensuring non-video files or malicious scripts are rejected prior to processing.
            </div>
          </div>
        </div>
      </div>

      {/* Non-Diagnostic Veterinary Disclaimer */}
      <div className="bg-amber-50 rounded-2xl p-5 border-2 border-amber-300 space-y-3 shadow-sm">
        <h3 className="font-extrabold text-amber-950 text-base flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-amber-600" />
          Medical & Veterinary Boundary Disclaimer
        </h3>
        <p className="text-xs text-amber-900 font-medium leading-relaxed">
          GaitGuard AI provides AI-assisted cattle gait and lameness-risk screening using movement analysis. It is <strong>NOT a veterinary diagnostic system</strong> and does not replace visual or physical assessment by a qualified veterinarian or hoof care specialist.
        </p>
        <p className="text-xs text-amber-800 leading-relaxed">
          Screening probabilities represent statistical movement anomalies relative to training baselines and should be used exclusively as a decision-support aid during routine herd triage.
        </p>
      </div>

      {/* Version Metadata Card */}
      <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm space-y-2">
        <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2">
          <FileText className="w-4 h-4 text-slate-500" />
          System Version Metadata
        </h3>
        <div className="grid grid-cols-2 gap-2 text-xs text-slate-600 pt-1 font-mono">
          <div className="bg-slate-50 p-2 rounded-lg border">API Version: <strong className="text-slate-900">v1</strong></div>
          <div className="bg-slate-50 p-2 rounded-lg border">Phase: <strong className="text-slate-900">Phase 17</strong></div>
          <div className="bg-slate-50 p-2 rounded-lg border">Model: <strong className="text-slate-900">BiLSTM-Mode-D</strong></div>
          <div className="bg-slate-50 p-2 rounded-lg border">Threshold: <strong className="text-slate-900">\u03c4 = 0.34</strong></div>
        </div>
      </div>
    </div>
  );
};
