import React, { useEffect, useState } from 'react';
import { PlayCircle, StopCircle, RefreshCw, Terminal, CheckCircle2, Layers } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Experiment, TrainingMetric } from '../types';
import { api } from '../services/api';

interface TrainingMonitorPageProps {
  experimentId: string | null;
  experiments: Experiment[];
  onSelectExperiment: (id: string) => void;
}

export const TrainingMonitorPage: React.FC<TrainingMonitorPageProps> = ({
  experimentId,
  experiments,
  onSelectExperiment
}) => {
  const [currentExp, setCurrentExp] = useState<Experiment | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  const activeExpId = experimentId || (experiments.length > 0 ? experiments[experiments.length - 1].id : null);

  const fetchJobData = async () => {
    if (!activeExpId) return;
    try {
      const expData = await api.getExperimentDetail(activeExpId);
      setCurrentExp(expData);
      const metricsData = await api.getMetrics(activeExpId);
      setMetrics(metricsData);
      const logsData = await api.getLogs(activeExpId);
      setLogs(logsData);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchJobData();
    const interval = setInterval(fetchJobData, 1500);
    return () => clearInterval(interval);
  }, [activeExpId]);

  const handleStart = async () => {
    if (!activeExpId) return;
    setLoading(true);
    try {
      await api.startTraining(activeExpId);
      fetchJobData();
    } catch (err) {
      alert('Gagal memulai training.');
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = async () => {
    if (!activeExpId) return;
    try {
      await api.cancelTraining(activeExpId);
      fetchJobData();
    } catch (err) {
      alert('Gagal menghentikan training.');
    }
  };

  const stages = [
    'Stage 1: Dataset preparation',
    'Stage 2: Optional DAPT',
    'Stage 3: Normal supervised training',
    'Stage 4: Neutral hardening',
    'Stage 5: Relation hardening',
    'Stage 6: Calibration',
    'Stage 7: Canonical evaluation',
    'Stage 8: Model packaging'
  ];

  const currentStage = currentExp?.job?.stage || 'Stage 1: Dataset preparation';

  return (
    <div className="space-y-6">
      {/* Experiment Selector & Action Header */}
      <div className="bg-dark-card border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <label className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
            Pilih Eksperimen
          </label>
          <select
            value={activeExpId || ''}
            onChange={(e) => onSelectExperiment(e.target.value)}
            className="bg-slate-900 border border-slate-700/80 rounded-xl px-4 py-2 text-xs text-white focus:outline-none"
          >
            {experiments.map((e) => (
              <option key={e.id} value={e.id}>
                {e.name} ({e.status})
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleStart}
            disabled={loading || currentExp?.status === 'running'}
            className="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-medium shadow-lg shadow-emerald-500/20 flex items-center gap-2 disabled:opacity-50"
          >
            <PlayCircle className="w-4 h-4" />
            Start Training
          </button>
          <button
            onClick={handleCancel}
            disabled={currentExp?.status !== 'running'}
            className="px-4 py-2.5 bg-red-600/20 hover:bg-red-600/30 text-red-300 border border-red-500/30 rounded-xl text-xs font-medium flex items-center gap-2 disabled:opacity-50"
          >
            <StopCircle className="w-4 h-4" />
            Cancel Training
          </button>
        </div>
      </div>

      {/* 8-Stage Progress Tracker */}
      <div className="bg-dark-card border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <h3 className="font-semibold text-white text-sm flex items-center gap-2">
          <Layers className="w-4 h-4 text-blue-400" />
          Tahapan Training Lifecycle (8 Stages)
        </h3>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {stages.map((stg, idx) => {
            const isCompleted = stages.indexOf(currentStage) > idx || currentExp?.status === 'completed';
            const isCurrent = currentStage.includes(`Stage ${idx + 1}`) && currentExp?.status === 'running';
            return (
              <div
                key={stg}
                className={`p-3 rounded-xl border text-xs transition-all ${
                  isCompleted
                    ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                    : isCurrent
                    ? 'bg-blue-500/15 border-blue-500/40 text-blue-300 shadow-md shadow-blue-500/10 animate-pulse'
                    : 'bg-slate-900/60 border-slate-800 text-slate-500'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-semibold text-[10px]">STAGE {idx + 1}</span>
                  {isCompleted && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />}
                </div>
                <div className="font-medium text-[11px] truncate">{stg.split(': ')[1]}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Live Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-dark-card border border-slate-800 rounded-2xl p-6 shadow-xl">
          <h4 className="font-semibold text-white text-xs mb-4">Grafik Loss (Train vs Val Loss)</h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="epoch" stroke="#94a3b8" fontSize={10} />
                <YAxis stroke="#94a3b8" fontSize={10} />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', fontSize: '11px' }} />
                <Line type="monotone" dataKey="train_loss" stroke="#ef4444" strokeWidth={2} name="Train Loss" />
                <Line type="monotone" dataKey="val_loss" stroke="#3b82f6" strokeWidth={2} name="Val Loss" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-dark-card border border-slate-800 rounded-2xl p-6 shadow-xl">
          <h4 className="font-semibold text-white text-xs mb-4">Grafik Macro F1 & Sub-Metrics</h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="epoch" stroke="#94a3b8" fontSize={10} />
                <YAxis domain={[0.4, 1.0]} stroke="#94a3b8" fontSize={10} />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', fontSize: '11px' }} />
                <Line type="monotone" dataKey="macro_f1" stroke="#10b981" strokeWidth={2} name="Macro F1" />
                <Line type="monotone" dataKey="aspect_f1" stroke="#8b5cf6" strokeWidth={2} name="Aspect F1" />
                <Line type="monotone" dataKey="relation_f1" stroke="#f59e0b" strokeWidth={2} name="Relation F1" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Terminal Log Viewer */}
      <div className="bg-slate-950 border border-slate-800 rounded-2xl p-5 shadow-2xl font-mono text-xs text-slate-300">
        <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800 text-slate-400">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-emerald-400" />
            <span className="font-semibold text-white">Live Training Logs Terminal</span>
          </div>
          <span className="text-[10px] text-slate-500">Auto-refreshing every 1.5s</span>
        </div>
        <div className="h-48 overflow-y-auto space-y-1">
          {logs.map((log, idx) => (
            <div key={idx} className="leading-relaxed hover:bg-slate-900/60 px-1 rounded">
              {log}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
