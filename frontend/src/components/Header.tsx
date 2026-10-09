import React, { useEffect, useState } from 'react';
import { Activity, CheckCircle2, XCircle, Info, ShieldCheck, Video } from 'lucide-react';
import { api } from '../api/gaitguard';

export type NavTab = 'screening' | 'how-it-works' | 'privacy';

interface HeaderProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, onTabChange }) => {
  const [isOnline, setIsOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let isMounted = true;
    const checkConnection = async () => {
      try {
        const health = await api.getHealth();
        if (isMounted) setIsOnline(health.status === 'ok');
      } catch {
        if (isMounted) setIsOnline(false);
      }
    };
    checkConnection();
    const interval = setInterval(checkConnection, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="bg-emerald-800 text-white shadow-md sticky top-0 z-40">
      <div className="max-w-md mx-auto px-4 pt-3 pb-2 space-y-3">
        {/* Logo & Connection Badge */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => onTabChange('screening')}>
            <div className="bg-emerald-700 p-2 rounded-xl">
              <Activity className="w-6 h-6 text-emerald-200" />
            </div>
            <div>
              <h1 className="font-extrabold text-xl leading-tight tracking-tight">GaitGuard AI</h1>
              <p className="text-[11px] text-emerald-200 font-medium">Cattle Gait Screening System</p>
            </div>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1 bg-emerald-900/70 rounded-full text-xs font-semibold border border-emerald-700/50">
            {isOnline === null ? (
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
            ) : isOnline ? (
              <>
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-emerald-200">API Connected</span>
              </>
            ) : (
              <>
                <XCircle className="w-3.5 h-3.5 text-rose-400" />
                <span className="text-rose-300">API Offline</span>
              </>
            )}
          </div>
        </div>

        {/* Responsive Navigation Tabs */}
        <nav className="flex bg-emerald-900/60 p-1 rounded-xl text-xs font-bold gap-1" aria-label="Main Navigation">
          <button
            onClick={() => onTabChange('screening')}
            className={`flex-1 py-1.5 px-2 rounded-lg flex items-center justify-center gap-1.5 transition-all ${
              activeTab === 'screening'
                ? 'bg-white text-emerald-900 shadow-sm'
                : 'text-emerald-100 hover:bg-emerald-800/80'
            }`}
          >
            <Video className="w-3.5 h-3.5" />
            <span>Screening</span>
          </button>

          <button
            onClick={() => onTabChange('how-it-works')}
            className={`flex-1 py-1.5 px-2 rounded-lg flex items-center justify-center gap-1.5 transition-all ${
              activeTab === 'how-it-works'
                ? 'bg-white text-emerald-900 shadow-sm'
                : 'text-emerald-100 hover:bg-emerald-800/80'
            }`}
          >
            <Info className="w-3.5 h-3.5" />
            <span>How It Works</span>
          </button>

          <button
            onClick={() => onTabChange('privacy')}
            className={`flex-1 py-1.5 px-2 rounded-lg flex items-center justify-center gap-1.5 transition-all ${
              activeTab === 'privacy'
                ? 'bg-white text-emerald-900 shadow-sm'
                : 'text-emerald-100 hover:bg-emerald-800/80'
            }`}
          >
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>About & Privacy</span>
          </button>
        </nav>
      </div>
    </header>
  );
};
