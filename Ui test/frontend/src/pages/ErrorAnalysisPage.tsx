import React, { useEffect, useState } from 'react';
import { AlertOctagon, Download, Search, Filter } from 'lucide-react';
import { ProxyDisclaimerBanner } from '../components/ProxyDisclaimerBanner';
import { api } from '../services/api';

interface ErrorAnalysisPageProps {
  experimentId: string | null;
}

export const ErrorAnalysisPage: React.FC<ErrorAnalysisPageProps> = ({ experimentId }) => {
  const [errorRows, setErrorRows] = useState<any[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedErrorType, setSelectedErrorType] = useState('ALL');

  useEffect(() => {
    async function loadErrors() {
      try {
        const report = await api.getReport(experimentId || 'demo');
        if (report.error_analysis) {
          setErrorRows(report.error_analysis);
        }
      } catch (err) {
        console.error(err);
      }
    }
    loadErrors();
  }, [experimentId]);

  const errorTypes = [
    'ALL',
    'prediction-label disagreement',
    'missed aspect',
    'spurious aspect',
    'wrong sentiment',
    'wrong taxonomy',
    'relation failure',
    'opinion extraction failure'
  ];

  const filteredRows = errorRows.filter((r) => {
    const matchesSearch = r.review_text.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesType = selectedErrorType === 'ALL' || r.error_type === selectedErrorType;
    return matchesSearch && matchesType;
  });

  const exportCSV = () => {
    const headers = 'review_text,gold_aspect,pred_aspect,gold_sentiment,pred_sentiment,gold_taxonomy,pred_taxonomy,confidence,error_type\n';
    const rows = filteredRows.map(r => `"${r.review_text.replace(/"/g, '""')}","${r.gold_aspect}","${r.pred_aspect}","${r.gold_sentiment}","${r.pred_sentiment}","${r.gold_taxonomy}","${r.pred_taxonomy}",${r.confidence},"${r.error_type}"`).join('\n');
    const blob = new Blob([headers + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'error_analysis_v11.csv';
    a.click();
  };

  return (
    <div className="space-y-6">
      <ProxyDisclaimerBanner />

      <div className="bg-dark-card border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <h3 className="font-semibold text-white text-sm flex items-center gap-2">
              <AlertOctagon className="w-4 h-4 text-red-400" />
              Tabel Error Analysis & Disagreement Categorization
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Catatan: Karena belum divalidasi Human GOLD, error dikategorikan sebagai "prediction-label disagreement".
            </p>
          </div>

          <button
            onClick={exportCSV}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-medium shadow-lg shadow-blue-500/20 flex items-center gap-2"
          >
            <Download className="w-3.5 h-3.5" />
            Export CSV (error_analysis_v11.csv)
          </button>
        </div>

        {/* Filter Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Cari teks ulasan..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700/80 rounded-xl pl-9 pr-3.5 py-2 text-xs text-white focus:outline-none"
            />
          </div>

          <div className="md:col-span-2 flex items-center gap-2 overflow-x-auto pb-1">
            <Filter className="w-4 h-4 text-slate-400 shrink-0" />
            {errorTypes.map((t) => (
              <button
                key={t}
                onClick={() => setSelectedErrorType(t)}
                className={`px-3 py-1.5 rounded-lg text-[11px] font-medium whitespace-nowrap transition-colors ${
                  selectedErrorType === t
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        {/* Error Table */}
        <div className="overflow-x-auto pt-2">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-3 px-4">Teks Ulasan</th>
                <th className="py-3 px-4">Gold Aspect</th>
                <th className="py-3 px-4">Pred Aspect</th>
                <th className="py-3 px-4">Gold Sent</th>
                <th className="py-3 px-4">Pred Sent</th>
                <th className="py-3 px-4">Taxonomy</th>
                <th className="py-3 px-4">Conf</th>
                <th className="py-3 px-4">Kategori Disagreement</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {filteredRows.map((r, idx) => (
                <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                  <td className="py-3 px-4 text-white font-medium max-w-xs truncate">{r.review_text}</td>
                  <td className="py-3 px-4 text-slate-300 font-mono">{r.gold_aspect || '-'}</td>
                  <td className="py-3 px-4 text-slate-300 font-mono">{r.pred_aspect || '-'}</td>
                  <td className="py-3 px-4 font-mono text-emerald-400">{r.gold_sentiment}</td>
                  <td className="py-3 px-4 font-mono text-red-400">{r.pred_sentiment}</td>
                  <td className="py-3 px-4 text-slate-400">{r.pred_taxonomy}</td>
                  <td className="py-3 px-4 text-slate-400 font-mono">{r.confidence}</td>
                  <td className="py-3 px-4">
                    <span className="px-2.5 py-1 rounded-full text-[10px] font-semibold bg-red-500/15 text-red-300 border border-red-500/30">
                      {r.error_type}
                    </span>
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
