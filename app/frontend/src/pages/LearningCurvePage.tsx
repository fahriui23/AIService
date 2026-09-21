import React, { useEffect, useState } from 'react';
import { TrendingUp, Download, Eye, EyeOff } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { TrainingMetric } from '../types';
import { api } from '../services/api';

interface LearningCurvePageProps {
  experimentId: string | null;
}

export const LearningCurvePage: React.FC<LearningCurvePageProps> = ({ experimentId }) => {
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const [showLoss, setShowLoss] = useState(true);
  const [showMacroF1, setShowMacroF1] = useState(true);
  const [showAspectF1, setShowAspectF1] = useState(true);
  const [showRelationF1, setShowRelationF1] = useState(true);

  useEffect(() => {
    async function loadMetrics() {
      try {
        const data = await api.getMetrics(experimentId || 'demo');
        if (data.length > 0) {
          setMetrics(data);
        } else {
          // Empty state is truthful: metrics only exist after a real training run.
          setMetrics([]);
        }
      } catch (err) {
        console.error(err);
      }
    }
    loadMetrics();
  }, [experimentId]);

  const downloadCSV = () => {
    const headers = 'epoch,train_loss,val_loss,macro_f1,aspect_f1,relation_f1,lr,vram_mb\n';
    const rows = metrics.map(m => `${m.epoch},${m.train_loss},${m.val_loss},${m.macro_f1},${m.aspect_f1},${m.relation_f1},${m.lr},${m.vram_mb}`).join('\n');
    const blob = new Blob([headers + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'training_history_v11.csv';
    a.click();
  };

  return (
    <div className="space-y-6">
      <div className="bg-dark-card border border-slate-800 rounded-2xl p-6 shadow-xl flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-white text-sm flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-emerald-400" />
            Learning Curve (Loss & Metrics Progression)
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Pantau pergerakan loss dan score F1 sepanjang epoch training V11.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={downloadCSV}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-medium shadow-lg shadow-blue-500/20 flex items-center gap-2"
          >
            <Download className="w-3.5 h-3.5" />
            Download CSV (training_history_v11.csv)
          </button>
        </div>
      </div>

      {/* Metric Toggle Buttons */}
      <div className="flex flex-wrap gap-2 text-xs">
        <button
          onClick={() => setShowLoss(!showLoss)}
          className={`px-3 py-1.5 rounded-lg border flex items-center gap-1.5 ${
            showLoss ? 'bg-red-500/15 border-red-500/40 text-red-300' : 'bg-slate-900 border-slate-800 text-slate-500'
          }`}
        >
          {showLoss ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
          Loss Curves
        </button>
        <button
          onClick={() => setShowMacroF1(!showMacroF1)}
          className={`px-3 py-1.5 rounded-lg border flex items-center gap-1.5 ${
            showMacroF1 ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300' : 'bg-slate-900 border-slate-800 text-slate-500'
          }`}
        >
          {showMacroF1 ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
          Macro F1
        </button>
        <button
          onClick={() => setShowAspectF1(!showAspectF1)}
          className={`px-3 py-1.5 rounded-lg border flex items-center gap-1.5 ${
            showAspectF1 ? 'bg-purple-500/15 border-purple-500/40 text-purple-300' : 'bg-slate-900 border-slate-800 text-slate-500'
          }`}
        >
          {showAspectF1 ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
          Aspect F1
        </button>
        <button
          onClick={() => setShowRelationF1(!showRelationF1)}
          className={`px-3 py-1.5 rounded-lg border flex items-center gap-1.5 ${
            showRelationF1 ? 'bg-amber-500/15 border-amber-500/40 text-amber-300' : 'bg-slate-900 border-slate-800 text-slate-500'
          }`}
        >
          {showRelationF1 ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
          Relation F1
        </button>
      </div>

      {/* Main Learning Curve Chart */}
      <div className="bg-dark-card border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="epoch" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', fontSize: '11px' }} />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              {showLoss && <Line type="monotone" dataKey="train_loss" stroke="#ef4444" strokeWidth={2} name="Train Loss" />}
              {showLoss && <Line type="monotone" dataKey="val_loss" stroke="#f87171" strokeWidth={2} strokeDasharray="4 4" name="Val Loss" />}
              {showMacroF1 && <Line type="monotone" dataKey="macro_f1" stroke="#10b981" strokeWidth={2.5} name="Macro F1" />}
              {showAspectF1 && <Line type="monotone" dataKey="aspect_f1" stroke="#8b5cf6" strokeWidth={2} name="Aspect F1" />}
              {showRelationF1 && <Line type="monotone" dataKey="relation_f1" stroke="#f59e0b" strokeWidth={2} name="Relation F1" />}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
