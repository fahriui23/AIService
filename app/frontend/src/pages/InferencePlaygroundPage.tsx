import React, { useEffect, useRef, useState } from 'react';
import {
  Activity, AlertCircle, CheckCircle2, Code, Download, FileSpreadsheet,
  Play, Sparkles, Upload, XCircle
} from 'lucide-react';
import {
  BatchRowResult, BatchStatusResponse, BatchUploadResponse, InferenceEngineInfo,
  InferenceEngineVersion, SingleInferenceResponse
} from '../types';
import { api } from '../services/api';

type Mode = 'single' | 'batch';
const terminalStates = new Set(['completed', 'failed', 'cancelled', 'timeout']);
const formatMs = (value?: number | null) => value == null ? '—' : `${value.toLocaleString('id-ID', { maximumFractionDigits: 2 })} ms`;
const formatDuration = (value?: number | null) => {
  if (value == null || !Number.isFinite(value)) return 'Menghitung…';
  const totalSeconds = Math.max(0, Math.round(value / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours) return `${hours} jam ${minutes} menit`;
  if (minutes) return `${minutes} menit ${seconds} detik`;
  return `${seconds} detik`;
};
const formatFinishTime = (remainingMs?: number | null) => remainingMs == null
  ? 'Menghitung…'
  : new Date(Date.now() + remainingMs).toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });

export const InferencePlaygroundPage: React.FC = () => {
  const [mode, setMode] = useState<Mode>('single');
  const [text, setText] = useState('Bandung no213 aplikasi ini sangat bagus');
  const [customerId, setCustomerId] = useState('');
  const [city, setCity] = useState('');
  const [province, setProvince] = useState('');
  const [threshold] = useState(0.1);
  const [profile, setProfile] = useState('maps_high_recall');
  const [engineVersion, setEngineVersion] = useState<InferenceEngineVersion>('v14');
  const [engines, setEngines] = useState<InferenceEngineInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<SingleInferenceResponse | null>(null);
  const [error, setError] = useState('');
  const [batchSize, setBatchSize] = useState(32);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [upload, setUpload] = useState<BatchUploadResponse | null>(null);
  const [batchStatus, setBatchStatus] = useState<BatchStatusResponse | null>(null);
  const [batchResults, setBatchResults] = useState<BatchRowResult[]>([]);
  const pollGeneration = useRef(0);

  useEffect(() => {
    api.getInferenceEngines().then(setEngines).catch(() => setError('Status bundle engine tidak dapat dimuat.'));
    const saved = localStorage.getItem('absa_active_batch');
    if (saved) {
      try {
        const active = JSON.parse(saved) as { jobId: string; token: string };
        setMode('batch');
        void resumeBatch(active.jobId, active.token);
      } catch { localStorage.removeItem('absa_active_batch'); }
    }
  }, []);

  const messageFromError = (err: any, fallback: string) =>
    err?.response?.data?.error?.message || err?.response?.data?.detail || fallback;

  const handleRunInference = async () => {
    if (!text.trim()) return;
    setLoading(true); setError('');
    try {
      setResponse(await api.runSingleInference(text, threshold, profile, engineVersion, {
        customer_id: customerId || undefined, city: city || undefined, province: province || undefined
      }));
    } catch (err: any) { setError(messageFromError(err, 'Gagal menjalankan inference.')); }
    finally { setLoading(false); }
  };

  const handleFile = async (file?: File) => {
    if (!file) return;
    setLoading(true); setError(''); setUpload(null); setBatchStatus(null); setBatchResults([]); setUploadProgress(0);
    try {
      setUpload(await api.uploadBatchCsv(file, engineVersion, batchSize, threshold, profile, setUploadProgress));
    } catch (err: any) { setError(messageFromError(err, 'CSV tidak dapat diunggah.')); }
    finally { setLoading(false); }
  };

  const pollBatch = async (jobId: string, token: string, generation: number) => {
    while (generation === pollGeneration.current) {
      const status = await api.getBatchStatus(jobId, token);
      setBatchStatus(status);
      const page = await api.getBatchResults(jobId, token, 0, 200);
      setBatchResults(page.results);
      if (terminalStates.has(status.status)) {
        localStorage.removeItem('absa_active_batch');
        break;
      }
      await new Promise(resolve => window.setTimeout(resolve, 800));
    }
  };

  const resumeBatch = async (jobId: string, token: string) => {
    const generation = ++pollGeneration.current;
    try { await pollBatch(jobId, token, generation); }
    catch { localStorage.removeItem('absa_active_batch'); }
  };

  const startBatch = async () => {
    if (!upload) return;
    setLoading(true); setError('');
    try {
      await api.startBatch(upload.job_id, upload.access_token);
      localStorage.setItem('absa_active_batch', JSON.stringify({ jobId: upload.job_id, token: upload.access_token }));
      const generation = ++pollGeneration.current;
      await pollBatch(upload.job_id, upload.access_token, generation);
    } catch (err: any) { setError(messageFromError(err, 'Batch tidak dapat dimulai.')); }
    finally { setLoading(false); }
  };

  const cancelBatch = async () => {
    if (!upload) return;
    try { await api.cancelBatch(upload.job_id, upload.access_token); }
    catch (err: any) { setError(messageFromError(err, 'Batch tidak dapat dibatalkan.')); }
  };

  const enginePicker = (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      {(engines.length ? engines.map(item => item.id) : ['v14'] as InferenceEngineVersion[]).map(version => {
        const engine = engines.find(item => item.id === version);
        const unavailable = engine?.available === false;
        return <button key={version} type="button" disabled={unavailable || loading || !!upload}
          onClick={() => { setEngineVersion(version); setResponse(null); }}
          className={`rounded-xl border p-3 text-left transition-all ${engineVersion === version
            ? 'border-emerald-500 bg-emerald-500/10 ring-1 ring-emerald-500/30'
            : 'border-slate-800 bg-slate-900 hover:border-slate-700'} ${unavailable ? 'opacity-50' : ''}`}>
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold">ABSA {version.toUpperCase()}</span>
            {unavailable ? <AlertCircle className="h-4 w-4 text-red-400"/> : <CheckCircle2 className="h-4 w-4 text-emerald-400"/>}
          </div>
          <p className="mt-1 text-[11px] text-slate-400">{engine?.model_version || 'Memuat metadata model…'}</p>
          {version === 'v14' && engine?.components && <p className="mt-1 text-[11px] text-amber-300">Aspect + opinion + sentiment model · relation/taxonomy fallback</p>}
          {version === 'v15' && engine?.components && <p className="mt-1 text-[11px] text-emerald-300">Joint relation + sentiment · hierarchical taxonomy</p>}
          {engine?.status === 'candidate_unpromoted' && <p className="mt-1 text-[10px] text-amber-300">Candidate · belum divalidasi Human GOLD</p>}
        </button>;
      })}
    </div>
  );

  return <div className="space-y-6">
    <div className="flex rounded-xl border border-slate-800 bg-slate-950 p-1">
      {([['single', 'Single Review'], ['batch', 'Batch CSV']] as const).map(([id, label]) =>
        <button key={id} onClick={() => setMode(id)} className={`flex-1 rounded-lg px-4 py-2 text-xs font-semibold ${mode === id ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-white'}`}>{label}</button>
      )}
    </div>

    {error && <div className="flex items-center gap-2 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-xs text-red-300"><AlertCircle className="h-4 w-4"/>{error}</div>}

    {mode === 'single' ? <>
      <section className="space-y-4 rounded-2xl border border-slate-800 bg-dark-card p-6 shadow-xl">
        <h3 className="flex items-center gap-2 text-sm font-semibold"><Sparkles className="h-4 w-4 text-emerald-400"/>Single Review + Metadata Enrichment</h3>
        {enginePicker}
        <div className="grid gap-3 md:grid-cols-3">
          <input value={customerId} onChange={e => setCustomerId(e.target.value)} placeholder="Customer ID (opsional)" className="input"/>
          <input value={city} onChange={e => setCity(e.target.value)} placeholder="Kota/Kabupaten (opsional)" className="input"/>
          <input value={province} onChange={e => setProvince(e.target.value)} placeholder="Provinsi (opsional)" className="input"/>
        </div>
        <textarea rows={7} value={text} onChange={e => setText(e.target.value)} placeholder="Masukkan review…" className="input w-full"/>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <select value={profile} onChange={e => setProfile(e.target.value)} className="input text-xs">
            <option value="maps_high_recall">maps_high_recall</option><option value="production_precision">production_precision</option><option value="scientific_balanced">scientific_balanced</option>
          </select>
          <button onClick={handleRunInference} disabled={loading} className="action"><Play className="h-4 w-4"/>{loading ? 'Menganalisis…' : 'Analisis Review'}</button>
        </div>
      </section>

      {response && <section className="space-y-4 rounded-2xl border border-slate-800 bg-dark-card p-6">
        <div className="flex flex-wrap items-center justify-between gap-3 text-xs"><b>Hasil {response.engine_version.toUpperCase()}</b><span className="font-mono text-slate-400">{formatMs(response.processing_time_ms)}</span></div>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Metric label="Customer ID" value={response.customer?.customer_id || '—'}/><Metric label="Customer Class" value={response.customer?.customer_class || 'UNKNOWN'}/>
          <Metric label="Kota/Kabupaten" value={response.location?.city_or_regency || response.location?.candidates?.join(' / ') || '—'}/><Metric label="Provinsi" value={response.location?.province || '—'}/>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-950 p-3 text-xs"><span className="text-slate-500">Teks bersih untuk ABSA</span><p className="mt-1 text-slate-200">{response.clean_review || response.text}</p></div>
        <div className="space-y-2">{response.results.map((item, index) => <div key={index} className="rounded-xl border border-slate-800 bg-slate-900/70 p-3 text-xs">
          <div className="grid gap-2 md:grid-cols-5"><b>{item.aspect}</b><span>{item.opinion}</span><span className={item.sentiment === 'positive' ? 'text-emerald-400' : item.sentiment === 'negative' ? 'text-red-400' : ''}>{item.sentiment}</span><span>{item.taxonomy}</span><span>{Math.round(item.confidence * 100)}%</span></div>
          {item.domain && <div className="mt-2 flex flex-wrap gap-2 border-t border-slate-800 pt-2 text-[10px] text-slate-400"><span>Domain: {item.domain}</span><span>Entity: {item.entity_id}</span><span>Issue: {item.issue_id}</span>{item.taxonomy_abstained && <span className="text-amber-300">Taxonomy abstain</span>}{item.taxonomy_low_confidence && !item.taxonomy_abstained && <span className="text-amber-300">Best effort · confidence rendah</span>}</div>}
        </div>)}</div>
        {!!response.warnings?.length && <div className="text-xs text-amber-300">Warning: {response.warnings.join(', ')}</div>}
        <details className="text-xs"><summary className="flex cursor-pointer items-center gap-2 text-slate-400"><Code className="h-4 w-4"/>Raw JSON</summary><pre className="mt-2 overflow-auto rounded-xl bg-slate-950 p-4 text-emerald-400">{JSON.stringify(response, null, 2)}</pre></details>
      </section>}
    </> : <>
      <section className="space-y-4 rounded-2xl border border-slate-800 bg-dark-card p-6">
        <h3 className="flex items-center gap-2 text-sm font-semibold"><FileSpreadsheet className="h-4 w-4 text-emerald-400"/>Batch CSV Analysis</h3>
        {!upload && enginePicker}
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex-1 cursor-pointer rounded-xl border border-dashed border-slate-700 bg-slate-900 p-5 text-center text-xs hover:border-emerald-500">
            <Upload className="mx-auto mb-2 h-5 w-5"/>Pilih CSV untuk validasi dan preview
            <input type="file" accept=".csv,text/csv" className="hidden" disabled={loading} onChange={e => void handleFile(e.target.files?.[0])}/>
          </label>
          <label className="text-xs text-slate-400">Batch size<input type="number" min={1} max={256} value={batchSize} disabled={!!upload} onChange={e => setBatchSize(Number(e.target.value))} className="input mt-1 block w-28"/></label>
        </div>
        {loading && !batchStatus && <div className="text-xs text-slate-400">Upload {uploadProgress}%</div>}
      </section>

      {upload && <section className="space-y-4 rounded-2xl border border-slate-800 bg-dark-card p-6">
        <div className="flex flex-wrap items-center justify-between gap-3"><h4 className="text-sm font-semibold">Preview CSV</h4><span className="rounded-full bg-emerald-500/15 px-3 py-1 text-xs text-emerald-300">Valid</span></div>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4"><Metric label="File" value={upload.file.name}/><Metric label="Jumlah baris" value={upload.file.row_count.toLocaleString('id-ID')}/><Metric label="Kolom review" value={upload.file.review_column}/><Metric label="Model / batch" value={`${upload.model_version.toUpperCase()} / ${upload.batch_size}`}/></div>
        {!!upload.validation.warnings.length && <div className="rounded-xl border border-blue-500/30 bg-blue-500/10 px-3 py-2 text-xs text-blue-200">Format ekspor VOC berbungkus terdeteksi dan dinormalisasi otomatis.</div>}
        <DataTable columns={upload.file.columns} rows={upload.preview}/>
        {!batchStatus && <div className="flex gap-2"><button onClick={() => { setUpload(null); setUploadProgress(0); }} className="secondary">Batalkan</button><button onClick={() => void startBatch()} className="action"><Play className="h-4 w-4"/>Mulai Analisis</button></div>}
      </section>}

      {batchStatus && <section className="space-y-4 rounded-2xl border border-slate-800 bg-dark-card p-6">
        <div className="flex items-center justify-between"><h4 className="flex items-center gap-2 text-sm font-semibold"><Activity className="h-4 w-4 text-blue-400"/>Progress Batch</h4><span className="text-xs uppercase text-slate-400">{batchStatus.status}</span></div>
        <div className="h-2 overflow-hidden rounded-full bg-slate-800"><div className="h-full bg-emerald-500 transition-all" style={{width: `${batchStatus.progress.percentage}%`}}/></div>
        <div className="grid gap-3 sm:grid-cols-2">
          <Metric label="Sudah berjalan" value={formatDuration(batchStatus.performance.elapsed_time_ms)}/>
          <Metric label="Selesai sekitar" value={terminalStates.has(batchStatus.status) ? 'Selesai' : formatFinishTime(batchStatus.performance.estimated_remaining_ms)}/>
        </div>
        <p className="text-xs text-slate-400">Memproses {batchStatus.progress.processed_rows.toLocaleString('id-ID')} / {batchStatus.progress.total_rows.toLocaleString('id-ID')} · Berhasil {batchStatus.progress.successful_rows} · Gagal {batchStatus.progress.failed_rows} · Dilewati {batchStatus.progress.skipped_rows}</p>
        {!terminalStates.has(batchStatus.status) && <button onClick={() => void cancelBatch()} className="secondary text-red-300"><XCircle className="h-4 w-4"/>Batalkan Proses</button>}
        {batchStatus.summary && <Summary summary={batchStatus.summary}/>}
        {!!batchResults.length && <ResultsTable rows={batchResults}/>}
        {terminalStates.has(batchStatus.status) && upload && <div className="flex gap-2"><button className="secondary" onClick={() => void api.downloadBatch(upload.job_id, upload.access_token, 'csv')}><Download className="h-4 w-4"/>CSV</button><button className="secondary" onClick={() => void api.downloadBatch(upload.job_id, upload.access_token, 'json')}><Download className="h-4 w-4"/>JSON</button></div>}
      </section>}
    </>}
  </div>;
};

const Metric = ({label, value}: {label: string; value: React.ReactNode}) => <div className="rounded-xl border border-slate-800 bg-slate-900 p-3"><div className="text-[10px] uppercase text-slate-500">{label}</div><div className="mt-1 truncate text-xs font-semibold text-slate-100">{value}</div></div>;
const DataTable = ({columns, rows}: {columns: string[]; rows: Record<string, string>[]}) => <div className="overflow-auto rounded-xl border border-slate-800"><table className="min-w-full text-left text-xs"><thead className="bg-slate-950 text-slate-400"><tr>{columns.map(c => <th key={c} className="px-3 py-2">{c}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i} className="border-t border-slate-800">{columns.map(c => <td key={c} className="max-w-sm truncate px-3 py-2">{row[c]}</td>)}</tr>)}</tbody></table></div>;
const Summary = ({summary}: {summary: Record<string, any>}) => <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6"><Metric label="Total waktu" value={formatMs(summary.total_processing_time_ms)}/><Metric label="Rata-rata" value={formatMs(summary.average_latency_ms)}/><Metric label="Median" value={formatMs(summary.median_latency_ms)}/><Metric label="P95" value={formatMs(summary.p95_latency_ms)}/><Metric label="P99" value={formatMs(summary.p99_latency_ms)}/><Metric label="Throughput" value={`${summary.throughput_rows_per_second || 0} row/dtk`}/></div>;
const sentimentStyle = (sentiment?: string) => {
  if (sentiment === 'positive') return 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300';
  if (sentiment === 'negative') return 'border-rose-500/20 bg-rose-500/10 text-rose-300';
  return 'border-slate-600/40 bg-slate-700/40 text-slate-300';
};

const StatusBadge = ({status}: {status: string}) => {
  const success = status === 'success';
  const skipped = status === 'empty_review' || status === 'skipped';
  const label = success ? 'Berhasil' : skipped ? 'Dilewati' : status.replace(/_/g, ' ');
  const style = success
    ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300'
    : skipped
      ? 'border-amber-500/20 bg-amber-500/10 text-amber-300'
      : 'border-rose-500/20 bg-rose-500/10 text-rose-300';

  return <span className={`inline-flex rounded-full border px-2.5 py-1 text-[10px] font-semibold capitalize ${style}`}>{label}</span>;
};

const ResultsTable = ({rows}: {rows: BatchRowResult[]}) => (
  <div className="space-y-3">
    <div className="flex flex-wrap items-end justify-between gap-2 px-1">
      <div>
        <h5 className="text-sm font-semibold text-slate-100">Hasil per baris</h5>
        <p className="mt-0.5 text-[11px] text-slate-500">Review, metadata, dan hasil analisis diringkas agar lebih mudah dibaca.</p>
      </div>
      <span className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-[11px] text-slate-400">
        {rows.length.toLocaleString('id-ID')} baris ditampilkan
      </span>
    </div>

    <div className="overflow-hidden rounded-2xl border border-slate-700/70 bg-slate-950/30">
      <div className="overflow-auto">
        <table className="min-w-[1180px] w-full text-left text-xs">
          <thead className="sticky top-0 z-10 bg-slate-950/95 text-[10px] uppercase tracking-wider text-slate-500 backdrop-blur">
            <tr>
              <th className="w-16 px-4 py-3 text-center">Baris</th>
              <th className="min-w-[270px] px-4 py-3">Review</th>
              <th className="w-32 px-4 py-3">Customer</th>
              <th className="w-40 px-4 py-3">Lokasi</th>
              <th className="min-w-[360px] px-4 py-3">Hasil analisis</th>
              <th className="w-32 px-4 py-3">Proses</th>
              <th className="w-44 px-4 py-3">Catatan</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/80">
            {rows.map(row => {
              const reviewChanged = row.clean_review && row.clean_review !== row.raw_review;
              const warning = row.warnings?.join(', ') || row.error_message;

              return (
                <tr key={row.row_number} className="group align-top transition-colors hover:bg-slate-800/35">
                  <td className="px-4 py-4 text-center">
                    <span className="inline-flex h-7 min-w-7 items-center justify-center rounded-lg bg-slate-800 px-2 font-semibold text-slate-300">
                      {row.row_number}
                    </span>
                  </td>
                  <td className="px-4 py-4">
                    <p className="leading-5 text-slate-100">{row.raw_review || <span className="italic text-slate-500">Review kosong</span>}</p>
                    {reviewChanged && (
                      <div className="mt-2 rounded-lg border border-slate-700/60 bg-slate-900/80 px-3 py-2">
                        <span className="text-[9px] font-semibold uppercase tracking-wider text-slate-500">Setelah dibersihkan</span>
                        <p className="mt-0.5 leading-4 text-slate-400">{row.clean_review}</p>
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-4">
                    {row.customer_id ? <p className="font-medium text-slate-200">ID {row.customer_id}</p> : <p className="text-slate-600">—</p>}
                    {row.customer_class && row.customer_class !== 'UNKNOWN' && (
                      <span className="mt-1.5 inline-flex rounded-full bg-blue-500/10 px-2 py-0.5 text-[10px] font-medium text-blue-300">{row.customer_class}</span>
                    )}
                  </td>
                  <td className="px-4 py-4 leading-5">
                    {row.city_or_regency || row.province ? (
                      <><p className="font-medium text-slate-200">{row.city_or_regency || 'Lokasi tidak spesifik'}</p><p className="text-[11px] text-slate-500">{row.province}</p></>
                    ) : <span className="text-slate-600">Tidak ada metadata</span>}
                  </td>
                  <td className="space-y-2 px-4 py-4">
                    {row.aspects?.length ? row.aspects.map((item, index) => (
                      <div key={index} className="rounded-xl border border-slate-700/60 bg-slate-900/70 p-2.5">
                        <div className="flex flex-wrap items-center gap-1.5">
                          <span className="font-semibold text-slate-100">{item.aspect}</span>
                          <span className={`rounded-full border px-2 py-0.5 text-[9px] font-semibold ${sentimentStyle(item.sentiment)}`}>{item.sentiment}</span>
                        </div>
                        <p className="mt-1 text-[11px] leading-4 text-slate-400">
                          {item.opinion || 'Tanpa opini'}
                          {(item.complaint_taxonomy || item.taxonomy) && <span className="text-slate-600"> · </span>}
                          <span className="text-slate-500">{item.complaint_taxonomy || item.taxonomy}</span>
                        </p>
                      </div>
                    )) : <span className="italic text-slate-500">Tidak ada aspek terdeteksi</span>}
                  </td>
                  <td className="px-4 py-4">
                    <StatusBadge status={row.processing_status}/>
                    <p className="mt-2 font-medium uppercase text-slate-400">{row.model_version}</p>
                    <p className="mt-0.5 text-[10px] tabular-nums text-slate-500">{formatMs(row.processing_time_ms)}</p>
                  </td>
                  <td className="px-4 py-4">
                    {warning ? <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 px-2.5 py-2 text-[11px] leading-4 text-amber-300">{warning}</div> : <span className="text-slate-600">Tidak ada catatan</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  </div>
);
