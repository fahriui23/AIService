import React, { useState } from 'react';
import { Sliders, Play, Cpu, Clock, HardDrive, CheckCircle2 } from 'lucide-react';
import { Dataset } from '../types';
import { api } from '../services/api';

interface ExperimentConfigPageProps {
  datasets: Dataset[];
  onExperimentCreated: (expId: string) => void;
}

export const ExperimentConfigPage: React.FC<ExperimentConfigPageProps> = ({ datasets, onExperimentCreated }) => {
  const [expName, setExpName] = useState('Eksperimen ABSA V11 — Balanced');
  const [runMode, setRunMode] = useState('dapt_silver_v11synthetic');
  const [baseModel, setBaseModel] = useState('indobenchmark/indobert-base-p1');
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(2e-5);
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>(datasets.map(d => d.id));

  // Controlled Class Exposure Sliders
  const [posTarget, setPosTarget] = useState(47.5);
  const [negTarget, setNegTarget] = useState(32.5);
  const [neuTarget, setNeuTarget] = useState(20.0);

  const [creating, setCreating] = useState(false);

  const applyPreset = (presetName: string) => {
    if (presetName === 'smoke') {
      setExpName('Quick Smoke Test');
      setEpochs(1);
      setBatchSize(8);
      setRunMode('base');
    } else if (presetName === 'balanced') {
      setExpName('Balanced Experiment — V11');
      setEpochs(3);
      setBatchSize(16);
      setRunMode('dapt_silver_v11synthetic');
    } else if (presetName === 'full') {
      setExpName('Full Training — Production Baseline');
      setEpochs(5);
      setBatchSize(32);
      setRunMode('dapt_silver_v11synthetic');
    }
  };

  const handleCreate = async () => {
    setCreating(true);
    try {
      const configPayload = {
        experiment_name: expName,
        run_mode: runMode,
        base_model: baseModel,
        number_of_epochs: epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        controlled_sampling_enabled: true,
        positive_exposure_target: posTarget,
        negative_exposure_target: negTarget,
        neutral_exposure_target: neuTarget,
        neutral_hardening_enabled: true,
        relation_hardening_enabled: true,
        calibration_enabled: true,
        dataset_ids: selectedDatasets
      };

      const res = await api.createExperiment(configPayload);
      onExperimentCreated(res.id);
    } catch (err) {
      alert('Gagal membuat eksperimen.');
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Presets */}
      <div className="bg-dark-card border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h3 className="font-semibold text-white mb-3 text-sm">Pilihan Preset Eksperimen</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <button
            type="button"
            onClick={() => applyPreset('smoke')}
            className="p-3 bg-slate-900/60 hover:bg-slate-800 border border-slate-800 rounded-xl text-left transition-colors"
          >
            <div className="font-semibold text-xs text-white mb-1">Quick Smoke Test</div>
            <p className="text-[11px] text-slate-400">1 Epoch, CPU/GPU ringan, verifikasi pipeline.</p>
          </button>
          <button
            type="button"
            onClick={() => applyPreset('balanced')}
            className="p-3 bg-blue-600/15 border border-blue-500/30 rounded-xl text-left transition-colors"
          >
            <div className="font-semibold text-xs text-blue-400 mb-1">Balanced Experiment</div>
            <p className="text-[11px] text-slate-400">3 Epochs, DAPT + Silver + Synthetic V11.</p>
          </button>
          <button
            type="button"
            onClick={() => applyPreset('full')}
            className="p-3 bg-slate-900/60 hover:bg-slate-800 border border-slate-800 rounded-xl text-left transition-colors"
          >
            <div className="font-semibold text-xs text-white mb-1">Full Training</div>
            <p className="text-[11px] text-slate-400">5 Epochs, full curriculum & hardening.</p>
          </button>
          <button
            type="button"
            onClick={() => setExpName('Custom Experiment')}
            className="p-3 bg-slate-900/60 hover:bg-slate-800 border border-slate-800 rounded-xl text-left transition-colors"
          >
            <div className="font-semibold text-xs text-white mb-1">Custom</div>
            <p className="text-[11px] text-slate-400">Kustomisasi seluruh parameter manual.</p>
          </button>
        </div>
      </div>

      {/* Main Config Form */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2 bg-dark-card border border-slate-800 rounded-2xl p-6 shadow-xl space-y-5">
          <h3 className="font-semibold text-white text-sm flex items-center gap-2">
            <Sliders className="w-4 h-4 text-purple-400" />
            Konfigurasi Parameter Training V11
          </h3>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">Nama Eksperimen</label>
            <input
              type="text"
              value={expName}
              onChange={(e) => setExpName(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3.5 py-2 text-xs text-white focus:outline-none focus:border-blue-500"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Run Mode</label>
              <select
                value={runMode}
                onChange={(e) => setRunMode(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3.5 py-2 text-xs text-white focus:outline-none focus:border-blue-500"
              >
                <option value="base">base</option>
                <option value="dapt_only">dapt_only</option>
                <option value="silver_only">silver_only</option>
                <option value="dapt_silver">dapt_silver</option>
                <option value="dapt_silver_v11synthetic">dapt_silver_v11synthetic (Default)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Base Model Encoder</label>
              <input
                type="text"
                value={baseModel}
                onChange={(e) => setBaseModel(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3.5 py-2 text-xs text-white focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Jumlah Epochs</label>
              <input
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                className="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3.5 py-2 text-xs text-white focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Batch Size</label>
              <input
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3.5 py-2 text-xs text-white focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Learning Rate</label>
              <input
                type="number"
                step="0.00001"
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                className="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3.5 py-2 text-xs text-white focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>

          {/* Controlled Class Exposure Sliders */}
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3">
            <h4 className="font-semibold text-xs text-white">Controlled Class Exposure Target</h4>

            <div>
              <div className="flex justify-between text-xs text-slate-300 mb-1">
                <span>Positive Exposure Target ({posTarget}%)</span>
              </div>
              <input
                type="range"
                min="30"
                max="60"
                value={posTarget}
                onChange={(e) => setPosTarget(Number(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs text-slate-300 mb-1">
                <span>Negative Exposure Target ({negTarget}%)</span>
              </div>
              <input
                type="range"
                min="20"
                max="50"
                value={negTarget}
                onChange={(e) => setNegTarget(Number(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs text-slate-300 mb-1">
                <span>Neutral Exposure Target ({neuTarget}%)</span>
              </div>
              <input
                type="range"
                min="10"
                max="35"
                value={neuTarget}
                onChange={(e) => setNeuTarget(Number(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </div>

        {/* Demand Estimator */}
        <div className="space-y-6">
          <div className="bg-dark-card border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
            <h3 className="font-semibold text-white text-sm">Perkiraan Kebutuhan Hardware</h3>

            <div className="flex items-center gap-3 p-3 bg-slate-900/60 rounded-xl border border-slate-800">
              <Cpu className="w-4 h-4 text-blue-400" />
              <div>
                <div className="text-[11px] text-slate-400">Estimasi VRAM GPU</div>
                <div className="text-xs font-semibold text-white font-mono">~3.8 GB (Fits RTX 3050+)</div>
              </div>
            </div>

            <div className="flex items-center gap-3 p-3 bg-slate-900/60 rounded-xl border border-slate-800">
              <Clock className="w-4 h-4 text-amber-400" />
              <div>
                <div className="text-[11px] text-slate-400">Estimasi Waktu Training</div>
                <div className="text-xs font-semibold text-white font-mono">
                  {epochs === 1 ? '~1.5 menit (Smoke Test)' : `~${epochs * 2.5} menit`}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3 p-3 bg-slate-900/60 rounded-xl border border-slate-800">
              <HardDrive className="w-4 h-4 text-purple-400" />
              <div>
                <div className="text-[11px] text-slate-400">Disk Checkpoint</div>
                <div className="text-xs font-semibold text-white font-mono">~450 MB</div>
              </div>
            </div>

            <button
              onClick={handleCreate}
              disabled={creating}
              className="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-medium shadow-lg shadow-blue-500/20 flex items-center justify-center gap-2"
            >
              <Play className="w-4 h-4" />
              {creating ? 'Membuat...' : 'Buat & Lanjut ke Training'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
