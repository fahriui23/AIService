import React, { useEffect, useState } from 'react';
import { GitCompare, Download, TrendingUp, Award } from 'lucide-react';
import { ProxyDisclaimerBanner } from '../components/ProxyDisclaimerBanner';
import { api } from '../services/api';

export const AblationPage: React.FC = () => {
  const [ablationRows, setAblationRows] = useState<any[]>([]);

  useEffect(() => {
    async function loadAblation() {
      try {
        const res = await api.getAblation();
        setAblationRows(res.ablation_results || []);
      } catch (err) {
        console.error(err);
      }
    }
    loadAblation();
  }, []);

  const exportCSV = () => {
    const headers = 'run_mode,experiment_name,macro_f1,positive_f1,negative_f1,neutral_f1,aspect_f1,relation_f1,delta_vs_baseline\n';
    const rows = ablationRows.map(r => `"${r.run_mode}","${r.experiment_name}",${r.macro_f1},${r.positive_f1},${r.negative_f1},${r.neutral_f1},${r.aspect_f1},${r.relation_f1},${r.delta_vs_baseline}`).join('\n');
    const blob = new Blob([headers + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ablation_results_v11.csv';
    a.click();
  };

  return (
    <div className="space-y-6">
      <ProxyDisclaimerBanner />

      <div className="bg-dark-card border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-white text-sm flex items-center gap-2">
              <GitCompare className="w-4 h-4 text-purple-400" />
              Matriks Perbandingan Ablation Study V11
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Bandingkan efek variasi DAPT, Silver, dan Synthetic V11 terhadap performa Macro F1.
            </p>
          </div>

          <button
            onClick={exportCSV}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-medium shadow-lg shadow-blue-500/20 flex items-center gap-2"
          >
            <Download className="w-3.5 h-3.5" />
            Export CSV (ablation_results_v11.csv)
          </button>
        </div>

        {/* Ablation Table */}
        <div className="overflow-x-auto pt-2">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-3 px-4">Run Mode</th>
                <th className="py-3 px-4">Macro F1</th>
                <th className="py-3 px-4">Pos F1</th>
                <th className="py-3 px-4">Neg F1</th>
                <th className="py-3 px-4">Neu F1</th>
                <th className="py-3 px-4">Aspect F1</th>
                <th className="py-3 px-4">Relation F1</th>
                <th className="py-3 px-4">Delta vs Baseline</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {ablationRows.map((r, idx) => (
                <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                  <td className="py-3 px-4 font-semibold text-white">
                    <span className="px-2.5 py-1 rounded-full text-[10px] bg-purple-500/15 text-purple-300 border border-purple-500/30">
                      {r.run_mode}
                    </span>
                  </td>
                  <td className="py-3 px-4 font-mono font-bold text-emerald-400">{(r.macro_f1 * 100).toFixed(1)}%</td>
                  <td className="py-3 px-4 font-mono text-slate-300">{(r.positive_f1 * 100).toFixed(1)}%</td>
                  <td className="py-3 px-4 font-mono text-slate-300">{(r.negative_f1 * 100).toFixed(1)}%</td>
                  <td className="py-3 px-4 font-mono text-slate-300">{(r.neutral_f1 * 100).toFixed(1)}%</td>
                  <td className="py-3 px-4 font-mono text-slate-300">{(r.aspect_f1 * 100).toFixed(1)}%</td>
                  <td className="py-3 px-4 font-mono text-slate-300">{(r.relation_f1 * 100).toFixed(1)}%</td>
                  <td className="py-3 px-4 font-mono font-semibold text-blue-400">
                    {r.delta_vs_baseline >= 0 ? `+${(r.delta_vs_baseline * 100).toFixed(1)}%` : `${(r.delta_vs_baseline * 100).toFixed(1)}%`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
