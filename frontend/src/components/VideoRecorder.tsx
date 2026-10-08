import React, { useRef, useState, useEffect } from 'react';
import { Camera, Square, RefreshCw, AlertTriangle, ArrowLeft } from 'lucide-react';

interface VideoRecorderProps {
  onRecorded: (file: File) => void;
  onCancel: () => void;
}

export const VideoRecorder: React.FC<VideoRecorderProps> = ({ onRecorded, onCancel }) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [permissionError, setPermissionError] = useState<string | null>(null);

  useEffect(() => {
    let activeStream: MediaStream | null = null;
    const startCamera = async () => {
      try {
        setPermissionError(null);
        activeStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false
        });
        setStream(activeStream);
        if (videoRef.current) {
          videoRef.current.srcObject = activeStream;
        }
      } catch (err: any) {
        console.error('Camera access error:', err);
        setPermissionError('Camera access is unavailable. Please upload an existing video instead.');
      }
    };

    startCamera();

    return () => {
      if (activeStream) {
        activeStream.getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  useEffect(() => {
    let timer: any = null;
    if (isRecording) {
      timer = setInterval(() => {
        setRecordingSeconds(s => s + 1);
      }, 1000);
    } else {
      setRecordingSeconds(0);
    }
    return () => clearInterval(timer);
  }, [isRecording]);

  const startRecording = () => {
    if (!stream) return;
    chunksRef.current = [];
    try {
      const mimeType = MediaRecorder.isTypeSupported('video/mp4') ? 'video/mp4' : 'video/webm';
      const recorder = new MediaRecorder(stream, { mimeType });
      
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        const ext = mimeType.includes('mp4') ? '.mp4' : '.webm';
        const file = new File([blob], `camera_recorded_${Date.now()}${ext}`, { type: mimeType });
        onRecorded(file);
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    } catch (err) {
      console.error('Failed to start recorder:', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  if (permissionError) {
    return (
      <div className="card space-y-4 border-amber-200 bg-amber-50">
        <div className="flex items-center gap-3 text-amber-900 font-bold">
          <AlertTriangle className="w-6 h-6 text-amber-600" />
          <h3>Camera Access Required</h3>
        </div>
        <p className="text-sm text-amber-800">{permissionError}</p>
        <button onClick={onCancel} className="btn-secondary">
          <ArrowLeft className="w-5 h-5" />
          Back to Upload Options
        </button>
      </div>
    );
  }

  return (
    <div className="card space-y-4 bg-slate-900 text-white p-4">
      <div className="relative aspect-[4/3] bg-black rounded-xl overflow-hidden flex items-center justify-center">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
        />
        
        {isRecording && (
          <div className="absolute top-3 left-3 bg-red-600/90 text-white px-3 py-1 rounded-full text-xs font-bold flex items-center gap-2 animate-pulse">
            <span className="w-2 h-2 rounded-full bg-white" />
            REC {recordingSeconds}s
          </div>
        )}

        <div className="absolute bottom-2 left-0 right-0 text-center text-xs bg-black/60 py-1 text-slate-300">
          Keep full cow visible from side profile
        </div>
      </div>

      <div className="flex gap-3">
        {!isRecording ? (
          <button onClick={startRecording} className="btn-primary bg-red-600 hover:bg-red-700">
            <Camera className="w-6 h-6" />
            Start Recording
          </button>
        ) : (
          <button onClick={stopRecording} className="btn-primary bg-slate-800 hover:bg-slate-900 border-2 border-red-500 text-red-400">
            <Square className="w-6 h-6 fill-current" />
            Stop Recording ({recordingSeconds}s)
          </button>
        )}
        <button onClick={onCancel} className="btn-secondary text-slate-800">
          Cancel
        </button>
      </div>
    </div>
  );
};
