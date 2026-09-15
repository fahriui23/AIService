import React from 'react';
import { Cpu, HardDrive, Zap } from 'lucide-react';
import { SystemStatus } from '../types';

interface NavbarProps {
  systemStatus: SystemStatus | null;
}

export const Navbar: React.FC<NavbarProps> = ({ systemStatus }) => {
  return (
    <header className="h-16 border-b border-slate-800 bg-dark-card/50 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-30">
      <div className="flex items-center gap-3">
        <div className="w-10 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-[10px] font-bold text-white shadow-lg shadow-blue-500/20">
          V14
        </div>
        <div>
          <h1 className="font-bold text-sm tracking-wide text-white">
            V14 ABSA
          </h1>
          <p className="text-[11px] text-slate-400">
            Aspect-Based Sentiment Analysis
          </p>
        </div>
      </div>

      <div className="flex items-center gap-6">
        {systemStatus && (
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60">
              <Zap className={`w-3.5 h-3.5 ${systemStatus.gpu_available ? 'text-emerald-400' : 'text-amber-400'}`} />
              <span className="text-slate-300 font-medium">
                {systemStatus.gpu_available ? systemStatus.gpu_name : 'CPU Fallback Mode'}
              </span>
              {systemStatus.gpu_available && (
                <span className="text-slate-400 font-mono">
                  ({systemStatus.vram_used_mb}MB / {systemStatus.vram_total_mb}MB)
                </span>
              )}
            </div>

            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60">
              <Cpu className="w-3.5 h-3.5 text-blue-400" />
              <span className="text-slate-300">
                CPU: <strong className="text-white font-mono">{systemStatus.cpu_usage_percent}%</strong>
              </span>
            </div>

            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60">
              <HardDrive className="w-3.5 h-3.5 text-purple-400" />
              <span className="text-slate-300">
                RAM: <strong className="text-white font-mono">{systemStatus.ram_used_gb} GB</strong>
              </span>
            </div>
          </div>
        )}
      </div>
    </header>
  );
};
