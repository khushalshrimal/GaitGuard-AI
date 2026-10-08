import React, { useMemo } from 'react';
import { Play, RotateCcw, CheckCircle, FileVideo, HardDrive } from 'lucide-react';

interface VideoPreviewProps {
  file: File;
  onConfirm: () => void;
  onRetake: () => void;
  isSubmitting?: boolean;
}

export const VideoPreview: React.FC<VideoPreviewProps> = ({ file, onConfirm, onRetake, isSubmitting }) => {
  const videoUrl = useMemo(() => URL.createObjectURL(file), [file]);

  const sizeMb = (file.size / (1024 * 1024)).toFixed(2);

  return (
    <div className="card space-y-4">
      <div className="flex items-center justify-between border-b pb-3">
        <h3 className="font-bold text-slate-800 text-lg flex items-center gap-2">
          <FileVideo className="w-5 h-5 text-emerald-700" />
          Selected Video Preview
        </h3>
        <span className="text-xs font-semibold px-2.5 py-1 bg-emerald-100 text-emerald-800 rounded-full">
          Ready for Screening
        </span>
      </div>

      <div className="relative aspect-[4/3] bg-black rounded-xl overflow-hidden shadow-inner">
        <video
          src={videoUrl}
          controls
          playsInline
          className="w-full h-full object-contain"
        />
      </div>

      <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/80 text-xs space-y-1 text-slate-600">
        <div className="flex justify-between">
          <span className="font-semibold text-slate-700">Filename:</span>
          <span className="truncate max-w-[200px] text-slate-900 font-medium">{file.name}</span>
        </div>
        <div className="flex justify-between">
          <span className="font-semibold text-slate-700">File Size:</span>
          <span className="text-slate-900 font-medium">{sizeMb} MB</span>
        </div>
      </div>

      <div className="flex flex-col gap-2.5 pt-2">
        <button
          onClick={onConfirm}
          disabled={isSubmitting}
          className="btn-primary"
        >
          <CheckCircle className="w-5 h-5" />
          {isSubmitting ? 'Submitting Video...' : 'Screen This Video'}
        </button>
        <button
          onClick={onRetake}
          disabled={isSubmitting}
          className="btn-secondary"
        >
          <RotateCcw className="w-4 h-4" />
          Choose or Record Another Video
        </button>
      </div>
    </div>
  );
};
