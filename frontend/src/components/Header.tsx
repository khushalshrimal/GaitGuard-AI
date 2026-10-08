import React, { useEffect, useState } from 'react';
import { Activity, CheckCircle2, XCircle } from 'lucide-react';
import { api } from '../api/gaitguard';

export const Header: React.FC = () => {
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
      <div className="max-w-md mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-emerald-700 p-2 rounded-xl">
            <Activity className="w-6 h-6 text-emerald-200" />
          </div>
          <div>
            <h1 className="font-extrabold text-xl leading-tight tracking-tight">GaitGuard AI</h1>
            <p className="text-xs text-emerald-200 font-medium">Cattle Gait Screening System</p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 px-3 py-1 bg-emerald-900/60 rounded-full text-xs font-semibold">
          {isOnline === null ? (
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
          ) : isOnline ? (
            <>
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-emerald-300">Online</span>
            </>
          ) : (
            <>
              <XCircle className="w-3.5 h-3.5 text-rose-400" />
              <span className="text-rose-300">Offline</span>
            </>
          )}
        </div>
      </div>
    </header>
  );
};
