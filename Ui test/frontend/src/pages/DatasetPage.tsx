import React, { useState } from 'react';
import { Upload, FileText, CheckCircle2, AlertCircle, Database, Layers, Eye } from 'lucide-react';
import { Dataset } from '../types';
import { api } from '../services/api';

interface DatasetPageProps {
  datasets: Dataset[];
  onDatasetUploaded: () => void;
}

export const DatasetPage: React.FC<DatasetPageProps> = ({ datasets, onDatasetUploaded }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setProgress(0);
    setError(null);

    try {
      const res = await api.uploadDataset(selectedFile, (pct) => setProgress(pct));
      setSelectedDataset(res);
      setSelectedFile(null);
      onDatasetUploaded();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Gagal mengunggah file. Pastikan format valid.');
    } finally {
      setUploading(false);
    }
  };

  const activeAudit = selectedDataset?.audit || (datasets.length > 0 ? datasets[0].audit : null);

  return (
    <div className="space-y-6">
      {/* Upload Box */}
      <div className="bg-dark-card border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h3 className="font-semibold text-white mb-4 text-sm flex items-center gap-2">
          <Upload className="w-4 h-4 text-blue-400" />
          Unggah File Dataset (ZIP, CSV, XLSX, JSON, Notebook)
        </h3>

        <div className="border-2 border-dashed border-slate-700/80 hover:border-blue-500/60 rounded-xl p-8 text-center transition-colors">
          <input
            type="file"
            id="file-upload"
            className="hidden"
            accept=".zip,.csv,.xlsx,.json,.ipynb"
            onChange={handleFileChange}
          />
          <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center justify-center">
            <FileText className="w-10 h-10 text-slate-500 mb-3" />
            <span className="text-sm font-medium text-slate-200">
              {selectedFile ? selectedFile.name : 'Drag and drop atau klik untuk memilih file'}
            </span>
            <span className="text-xs text-slate-400 mt-1">
              File dideteksi otomatis berdasarkan struktur isi (V10, V11 synthetic, GOLD pilot, SILVER, DAPT)
            </span>
          </label>
        </div>

        {selectedFile && (
          <div className="mt-4 flex items-center justify-between">
            <span className="text-xs font-mono text-slate-300">
              Size: {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
            </span>
            <button
              onClick={handleUpload}
              disabled={uploading}
              className="px-5 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-medium shadow-lg shadow-blue-500/20 disabled:opacity-50"
            >
              {uploading ? `Mengunggah (${progress}%)` : 'Mulai Upload & Audit'}
            </button>
          </div>
        )}

        {uploading && (
          <div className="w-full bg-slate-800 rounded-full h-1.5 mt-4 overflow-hidden">
            <div className="bg-blue-500 h-full transition-all duration-200" style={{ width: `${progress}%` }} />
          </div>
        )}

        {error && (
          <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-xl text-xs text-red-400 flex items-center gap-2">
            <AlertCircle className="w-4 h-4" />
            {error}
          </div>
        )}
      </div>

      {/* Dataset Inventory Table */}
      <div className="bg-dark-card border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h3 className="font-semibold text-white mb-4 text-sm flex items-center gap-2">
          <Database className="w-4 h-4 text-purple-400" />
          Daftar Dataset Tersedia
        </h3>

        {datasets.length === 0 ? (
          <p className="text-xs text-slate-400 text-center py-6">Belum ada dataset yang diunggah.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-3 px-4">Nama Dataset</th>
                  <th className="py-3 px-4">Tipe Dideteksi</th>
                  <th className="py-3 px-4">Jumlah Ulasan</th>
                  <th className="py-3 px-4">Jumlah Anotasi</th>
                  <th className="py-3 px-4">Human Approved</th>
                  <th className="py-3 px-4">Aksi</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {datasets.map((d) => (
                  <tr key={d.id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-3 px-4 font-medium text-white">{d.name}</td>
                    <td className="py-3 px-4">
                      <span className="px-2.5 py-1 rounded-full text-[10px] font-semibold bg-blue-500/15 text-blue-300 border border-blue-500/30">
                        {d.detected_type}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-300">{d.review_count}</td>
                    <td className="py-3 px-4 font-mono text-slate-300">{d.annotation_count}</td>
                    <td className="py-3 px-4 font-mono text-slate-300">{d.human_approved_count}</td>
                    <td className="py-3 px-4">
                      <button
                        onClick={() => setSelectedDataset(d)}
                        className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-[11px] flex items-center gap-1"
                      >
                        <Eye className="w-3.5 h-3.5" />
                        Audit Detail
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Quality Audit Dashboard */}
      {activeAudit && (
        <div className="bg-dark-card border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
          <h3 className="font-semibold text-white text-sm flex items-center gap-2">
            <Layers className="w-4 h-4 text-emerald-400" />
            Ringkasan Audit Kualitas Dataset ({selectedDataset?.name || 'Dataset Aktif'})
          </h3>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="text-slate-400 mb-1">Total Ulasan</div>
              <div className="text-2xl font-bold text-white font-mono">{activeAudit.review_count}</div>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="text-slate-400 mb-1">Objek Anotasi</div>
              <div className="text-2xl font-bold text-blue-400 font-mono">{activeAudit.annotation_count}</div>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="text-slate-400 mb-1">Duplikat Text</div>
              <div className="text-2xl font-bold text-amber-400 font-mono">{activeAudit.duplicate_count}</div>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="text-slate-400 mb-1">Invalid Offsets</div>
              <div className="text-2xl font-bold text-red-400 font-mono">{activeAudit.invalid_offset_count}</div>
            </div>
          </div>

          {/* Synthetic Template Breakdown */}
          {activeAudit.synthetic_templates && (
            <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800">
              <h4 className="font-semibold text-xs text-white mb-2">Analisis Template V11 Synthetic</h4>
              <div className="flex gap-6 text-xs text-slate-300 mb-3">
                <span>Raw Rows: <strong className="text-white font-mono">{activeAudit.synthetic_templates.raw_rows}</strong></span>
                <span>Effective Unique Templates: <strong className="text-emerald-400 font-mono">{activeAudit.synthetic_templates.unique_templates}</strong></span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
