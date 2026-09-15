import React from 'react';
import { Settings, Cpu, ShieldCheck } from 'lucide-react';
import { SystemStatus } from '../types';

interface SettingsPageProps {
  systemStatus: SystemStatus | null;
}

export const SettingsPage: React.FC<SettingsPageProps> = ({ systemStatus }) => {
  return (
    <div className="space-y-6">
      <div className="bg-dark-card border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <h3 className="font-semibold text-white text-sm flex items-center gap-2">
          <Settings className="w-4 h-4 text-blue-400" />
          Pengaturan Aplikasi & Environment Status
        </h3>

        {systemStatus && (
          <div className="space-y-3 text-xs">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <h4 className="font-semibold text-white">Status Hardware & Akselerasi CUDA</h4>
              <div className="grid grid-cols-2 gap-4 text-slate-300">
                <div>GPU Available: <strong className={systemStatus.gpu_available ? 'text-emerald-400' : 'text-amber-400'}>{systemStatus.gpu_available ? 'Ya' : 'Tidak (CPU Fallback)'}</strong></div>
                <div>GPU Device: <strong className="text-white">{systemStatus.gpu_name}</strong></div>
                <div>Total VRAM: <strong className="text-white">{systemStatus.vram_total_mb} MB</strong></div>
                <div>CPU Cores: <strong className="text-white">{systemStatus.cpu_count} Cores</strong></div>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <h4 className="font-semibold text-white">Modus Evaluasi System</h4>
              <div className="text-slate-300">
                Default Mode: <strong className="text-amber-400 font-mono">proxy_without_human_gold</strong>
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                Seluruh evaluasi metrik pada aplikasi ini saat ini berjalan dalam modus proxy dan ditujukan untuk perbandingan eksperimen internal. Disclaimer permanen aktif pada seluruh dashboard evaluasi.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
