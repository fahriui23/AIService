import React, { useEffect, useState } from 'react';
import { FolderArchive, Download, Award, Trash2, CheckCircle2 } from 'lucide-react';
import { ModelArtifact } from '../types';
import { api } from '../services/api';

export const ModelRegistryPage: React.FC = () => {
  const [models, setModels] = useState<ModelArtifact[]>([]);

  const loadModels = async () => {
    try {
      const data = await api.getModels();
      setModels(data);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  const handleActivate = async (id: string) => {
    try {
      await api.activateModel(id);
      loadModels();
    } catch (err) {
      alert('Gagal mengaktifkan model.');
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Apakah Anda yakin ingin menghapus model ini?')) return;
    try {
      await api.deleteModel(id);
      loadModels();
    } catch (err) {
      alert('Gagal menghapus model.');
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-dark-card border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <h3 className="font-semibold text-white text-sm flex items-center gap-2">
          <FolderArchive className="w-4 h-4 text-blue-400" />
          Model Artifact Registry & Packaging Store
        </h3>

        {models.length === 0 ? (
          <p className="text-xs text-slate-400 py-6 text-center">Belum ada model hasil training tersimpan.</p>
        ) : (
          <div className="space-y-3">
            {models.map((m) => (
              <div
                key={m.id}
                className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 text-xs"
              >
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-semibold text-white text-sm">{m.model_name}</h4>
                    {m.is_best && (
                      <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
                        <Award className="w-3 h-3" /> BEST ACTIVE MODEL
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-4 text-slate-400 text-[11px]">
                    <span>Run Mode: <strong className="text-slate-200">{m.run_mode}</strong></span>
                    <span>Base Encoder: <strong className="text-slate-200">{m.base_model}</strong></span>
                    <span>Evaluation Mode: <strong className="text-amber-400">{m.evaluation_mode}</strong></span>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  {!m.is_best && (
                    <button
                      onClick={() => handleActivate(m.id)}
                      className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs"
                    >
                      Set as Best
                    </button>
                  )}
                  <a
                    href={api.getModelDownloadUrl(m.id)}
                    download
                    className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-medium shadow-md flex items-center gap-1.5"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Download ZIP Bundle
                  </a>
                  <button
                    onClick={() => handleDelete(m.id)}
                    className="p-1.5 text-slate-500 hover:text-red-400 rounded-lg transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
