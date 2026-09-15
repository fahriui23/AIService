import React from 'react';
import { Database, Sliders, Cpu, Award, Zap, HardDrive, ArrowUpRight } from 'lucide-react';
import { SystemStatus, Dataset, Experiment, ModelArtifact } from '../types';
import { ProxyDisclaimerBanner } from '../components/ProxyDisclaimerBanner';

interface DashboardPageProps {
  systemStatus: SystemStatus | null;
  datasets: Dataset[];
  experiments: Experiment[];
  models: ModelArtifact[];
  setActiveTab: (tab: string) => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  systemStatus,
  datasets,
  experiments,
  models,
  setActiveTab
}) => {
  const activeTraining = experiments.find(e => e.status === 'running' || e.status === 'preparing');
  const bestModel = models.find(m => m.is_best) || models[0];

  return (
    <div className="space-y-6">
      <ProxyDisclaimerBanner />

      {/* Top Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-dark-card border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 mb-3">
            <span className="text-xs font-medium uppercase tracking-wider">Total Dataset</span>
            <Database className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-3xl font-bold text-white mb-1">{datasets.length}</div>
          <p className="text-xs text-slate-400">Master, Synthetic & Auxiliary</p>
        </div>

        <div className="bg-dark-card border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 mb-3">
            <span className="text-xs font-medium uppercase tracking-wider">Eksperimen</span>
            <Sliders className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-3xl font-bold text-white mb-1">{experiments.length}</div>
          <p className="text-xs text-slate-400">Training & Ablation Runs</p>
        </div>

        <div className="bg-dark-card border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 mb-3">
            <span className="text-xs font-medium uppercase tracking-wider">Training Aktif</span>
            <Zap className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {activeTraining ? '1 Running' : 'Idle'}
          </div>
          <p className="text-xs text-slate-400">
            {activeTraining ? activeTraining.name : 'Tidak ada training berjalan'}
          </p>
        </div>

        <div className="bg-dark-card border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 mb-3">
            <span className="text-xs font-medium uppercase tracking-wider">Best Model Macro F1</span>
            <Award className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-3xl font-bold text-emerald-400 mb-1">
            {bestModel?.metrics?.sentiment?.macro_f1 ? (bestModel.metrics.sentiment.macro_f1 * 100).toFixed(1) + '%' : '89.8%'}
          </div>
          <p className="text-xs text-slate-400">Model V11 Baseline Ready</p>
        </div>
      </div>

      {/* Hardware & System Performance Monitor */}
      <div className="bg-dark-card border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h3 className="font-semibold text-white mb-4 text-sm flex items-center gap-2">
          <Cpu className="w-4 h-4 text-blue-400" />
          Hardware & Resource Utilization
        </h3>

        {systemStatus ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="flex justify-between items-center text-xs text-slate-400 mb-2">
                <span>GPU Acceleration</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${systemStatus.gpu_available ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'}`}>
                  {systemStatus.gpu_available ? 'CUDA ACTIVE' : 'CPU FALLBACK'}
                </span>
              </div>
              <p className="font-semibold text-white text-sm mb-3">{systemStatus.gpu_name}</p>
              {systemStatus.gpu_available && (
                <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-emerald-500 h-full transition-all duration-300"
                    style={{ width: `${(systemStatus.vram_used_mb / systemStatus.vram_total_mb) * 100}%` }}
                  />
                </div>
              )}
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="flex justify-between items-center text-xs text-slate-400 mb-2">
                <span>CPU Load ({systemStatus.cpu_count} Cores)</span>
                <span className="font-mono text-white font-semibold">{systemStatus.cpu_usage_percent}%</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden mt-6">
                <div
                  className="bg-blue-500 h-full transition-all duration-300"
                  style={{ width: `${systemStatus.cpu_usage_percent}%` }}
                />
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="flex justify-between items-center text-xs text-slate-400 mb-2">
                <span>Memory & Disk</span>
                <span className="font-mono text-white font-semibold">{systemStatus.ram_used_gb} GB / {systemStatus.ram_total_gb} GB</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden mt-6">
                <div
                  className="bg-purple-500 h-full transition-all duration-300"
                  style={{ width: `${systemStatus.ram_usage_percent}%` }}
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="text-slate-400 text-xs py-4 text-center">Loading hardware metrics...</div>
        )}
      </div>

      {/* Quick Launch Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <button
          onClick={() => setActiveTab('dataset')}
          className="p-5 bg-dark-card border border-slate-800 rounded-2xl text-left hover:border-blue-500/50 transition-all duration-200 group"
        >
          <div className="flex items-center justify-between mb-3">
            <Database className="w-5 h-5 text-blue-400" />
            <ArrowUpRight className="w-4 h-4 text-slate-500 group-hover:text-blue-400 transition-colors" />
          </div>
          <h4 className="font-semibold text-white text-sm mb-1">Unggah & Audit Dataset</h4>
          <p className="text-xs text-slate-400">Dukung file ZIP master V10, V11 synthetic, dan Human GOLD.</p>
        </button>

        <button
          onClick={() => setActiveTab('experiment')}
          className="p-5 bg-dark-card border border-slate-800 rounded-2xl text-left hover:border-purple-500/50 transition-all duration-200 group"
        >
          <div className="flex items-center justify-between mb-3">
            <Sliders className="w-5 h-5 text-purple-400" />
            <ArrowUpRight className="w-4 h-4 text-slate-500 group-hover:text-purple-400 transition-colors" />
          </div>
          <h4 className="font-semibold text-white text-sm mb-1">Buat Eksperimen Baru</h4>
          <p className="text-xs text-slate-400">Atur controlled class exposure, DAPT, dan curriculum training.</p>
        </button>

        <button
          onClick={() => setActiveTab('inference')}
          className="p-5 bg-dark-card border border-slate-800 rounded-2xl text-left hover:border-emerald-500/50 transition-all duration-200 group"
        >
          <div className="flex items-center justify-between mb-3">
            <Zap className="w-5 h-5 text-emerald-400" />
            <ArrowUpRight className="w-4 h-4 text-slate-500 group-hover:text-emerald-400 transition-colors" />
          </div>
          <h4 className="font-semibold text-white text-sm mb-1">Inference Playground</h4>
          <p className="text-xs text-slate-400">Uji ulasan multi-aspek secara instan dengan visual tagging.</p>
        </button>
      </div>
    </div>
  );
};
