import React, { useState } from 'react';
import { Header } from './components/Header';
import { CaptureGuide } from './components/CaptureGuide';
import { VideoRecorder } from './components/VideoRecorder';
import { VideoPreview } from './components/VideoPreview';
import { ProcessingState } from './components/ProcessingState';
import { QualityRetry } from './components/QualityRetry';
import { ScreeningResult } from './components/ScreeningResult';
import { EvidencePanel } from './components/EvidencePanel';
import { Disclaimer } from './components/Disclaimer';
import { api } from './api/gaitguard';
import { ScreeningResponse } from './types/api';
import { Camera, Upload, AlertCircle, RefreshCw, Tag } from 'lucide-react';

type AppState = 'IDLE' | 'CAPTURING' | 'PREVIEW' | 'SUBMITTING' | 'RESULT' | 'RETRY' | 'ERROR';

export const App: React.FC = () => {
  const [appState, setAppState] = useState<AppState>('IDLE');
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [animalId, setAnimalId] = useState<string>('');
  const [screeningResponse, setScreeningResponse] = useState<ScreeningResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // File Upload Handler
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setVideoFile(e.target.files[0]);
      setAppState('PREVIEW');
    }
  };

  // Video Recorded Handler
  const handleVideoRecorded = (file: File) => {
    setVideoFile(file);
    setAppState('PREVIEW');
  };

  // Submit to Screening API
  const handleAnalyzeVideo = async () => {
    if (!videoFile) return;

    setAppState('SUBMITTING');
    setErrorMessage(null);

    try {
      const response = await api.screenVideo(videoFile, animalId || undefined);
      setScreeningResponse(response);

      if (response.status === 'retry' && response.video_quality) {
        setAppState('RETRY');
      } else if (response.status === 'success' && response.inference) {
        setAppState('RESULT');
      } else {
        setAppState('ERROR');
        setErrorMessage(response.result_summary || 'Screening returned an unexpected response.');
      }
    } catch (err: unknown) {
      setAppState('ERROR');
      if (err instanceof Error) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage('Unable to connect to the screening service. Please check your backend network connection.');
      }
    }
  };

  // Reset Application State
  const handleReset = () => {
    setVideoFile(null);
    setScreeningResponse(null);
    setErrorMessage(null);
    setAppState('IDLE');
  };

  return (
    <div className="min-h-screen bg-slate-100 flex flex-col font-sans pb-10 text-slate-800">
      <Header />

      <main className="max-w-md mx-auto w-full px-4 pt-4 space-y-4 flex-1">
        {/* State: IDLE */}
        {appState === 'IDLE' && (
          <div className="space-y-4">
            <CaptureGuide />

            {/* Optional Animal ID Field */}
            <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm space-y-2">
              <label className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
                <Tag className="w-4 h-4 text-emerald-700" />
                Cow ID / Ear Tag Number (Optional)
              </label>
              <input
                type="text"
                value={animalId}
                onChange={(e) => setAnimalId(e.target.value)}
                placeholder="e.g. COW-9812"
                className="w-full px-3 py-2.5 rounded-xl border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 font-medium"
              />
            </div>

            {/* Primary Action Buttons */}
            <div className="space-y-3 pt-2">
              <button
                onClick={() => setAppState('CAPTURING')}
                className="w-full btn-primary py-4 flex items-center justify-center gap-3 font-extrabold text-lg shadow-lg"
              >
                <Camera className="w-6 h-6" />
                Record Live Cattle Video
              </button>

              <label className="w-full btn-secondary py-3.5 flex items-center justify-center gap-2 font-bold text-base cursor-pointer shadow-sm">
                <Upload className="w-5 h-5 text-emerald-700" />
                <span>Upload Existing Video File</span>
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </div>

            <Disclaimer />
          </div>
        )}

        {/* State: CAPTURING */}
        {appState === 'CAPTURING' && (
          <VideoRecorder
            onRecorded={handleVideoRecorded}
            onCancel={() => setAppState('IDLE')}
          />
        )}

        {/* State: PREVIEW */}
        {appState === 'PREVIEW' && videoFile && (
          <VideoPreview
            file={videoFile}
            onConfirm={handleAnalyzeVideo}
            onRetake={handleReset}
            isSubmitting={false}
          />
        )}

        {/* State: SUBMITTING */}
        {appState === 'SUBMITTING' && <ProcessingState />}

        {/* State: RETRY */}
        {appState === 'RETRY' && screeningResponse?.video_quality && (
          <div className="space-y-4">
            <QualityRetry
              quality={screeningResponse.video_quality}
              onReset={handleReset}
            />
            <Disclaimer />
          </div>
        )}

        {/* State: RESULT */}
        {appState === 'RESULT' && screeningResponse?.inference && (
          <div className="space-y-4">
            <ScreeningResult
              inference={screeningResponse.inference}
              resultSummary={screeningResponse.result_summary}
              onReset={handleReset}
            />

            <EvidencePanel explanation={screeningResponse.explanation} />

            <Disclaimer customText={screeningResponse.disclaimer} />
          </div>
        )}

        {/* State: ERROR */}
        {appState === 'ERROR' && (
          <div className="bg-rose-50 border-2 border-rose-300 rounded-2xl p-5 space-y-4 text-center shadow-sm">
            <div className="w-12 h-12 bg-rose-500 text-white rounded-full flex items-center justify-center mx-auto shadow-sm">
              <AlertCircle className="w-6 h-6" />
            </div>
            <div className="space-y-1">
              <h3 className="text-lg font-bold text-rose-950">Screening Request Failed</h3>
              <p className="text-xs font-medium text-rose-800 leading-relaxed">
                {errorMessage || 'Unable to complete gait screening request.'}
              </p>
            </div>

            <button
              onClick={handleReset}
              className="w-full btn-primary py-3 flex items-center justify-center gap-2 font-bold text-sm"
            >
              <RefreshCw className="w-4 h-4" />
              Try Again
            </button>
          </div>
        )}
      </main>
    </div>
  );
};
