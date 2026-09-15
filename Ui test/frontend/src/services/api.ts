import axios from 'axios';
import {
  SystemStatus,
  Dataset,
  Experiment,
  TrainingMetric,
  SingleInferenceResponse,
  ModelArtifact,
  InferenceEngineInfo,
  InferenceEngineVersion,
  BatchUploadResponse,
  BatchStatusResponse,
  BatchRowResult
} from '../types';

const API_BASE = '/api';

export const api = {
  // System & Health
  async getSystemStatus(): Promise<SystemStatus> {
    const res = await axios.get(`${API_BASE}/system`);
    return res.data;
  },

  // Datasets
  async uploadDataset(file: File, onProgress?: (pct: number) => void): Promise<Dataset> {
    const formData = new FormData();
    formData.append('file', file);

    const res = await axios.post(`${API_BASE}/datasets/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (evt) => {
        if (evt.total && onProgress) {
          onProgress(Math.round((evt.loaded * 100) / evt.total));
        }
      }
    });
    return res.data;
  },

  async getDatasets(): Promise<Dataset[]> {
    const res = await axios.get(`${API_BASE}/datasets`);
    return res.data;
  },

  async getDatasetDetail(id: string): Promise<Dataset> {
    const res = await axios.get(`${API_BASE}/datasets/${id}`);
    return res.data;
  },

  async deleteDataset(id: string): Promise<void> {
    await axios.delete(`${API_BASE}/datasets/${id}`);
  },

  // Experiments
  async createExperiment(config: any): Promise<Experiment> {
    const res = await axios.post(`${API_BASE}/experiments`, config);
    return res.data;
  },

  async getExperiments(): Promise<Experiment[]> {
    const res = await axios.get(`${API_BASE}/experiments`);
    return res.data;
  },

  async getExperimentDetail(id: string): Promise<Experiment> {
    const res = await axios.get(`${API_BASE}/experiments/${id}`);
    return res.data;
  },

  async startTraining(id: string): Promise<void> {
    await axios.post(`${API_BASE}/experiments/${id}/start`);
  },

  async cancelTraining(id: string): Promise<void> {
    await axios.post(`${API_BASE}/experiments/${id}/cancel`);
  },

  async getLogs(id: string): Promise<string[]> {
    const res = await axios.get(`${API_BASE}/experiments/${id}/logs`);
    return res.data.logs || [];
  },

  async getMetrics(id: string): Promise<TrainingMetric[]> {
    const res = await axios.get(`${API_BASE}/experiments/${id}/metrics`);
    return res.data;
  },

  // Evaluation & Ablation
  async getReport(id: string): Promise<any> {
    const res = await axios.get(`${API_BASE}/reports/${id}`);
    return res.data;
  },

  async getAblation(): Promise<any> {
    const res = await axios.get(`${API_BASE}/ablation`);
    return res.data;
  },

  // Inference Playground
  async getInferenceEngines(): Promise<InferenceEngineInfo[]> {
    const res = await axios.get(`${API_BASE}/inference/engines`);
    return res.data.engines;
  },

  async runSingleInference(
    text: string,
    confidenceThreshold = 0.5,
    profile = 'production_precision',
    engineVersion: InferenceEngineVersion = 'v11',
    metadata: { customer_id?: string; city?: string; province?: string } = {}
  ): Promise<SingleInferenceResponse> {
    const res = await axios.post(`${API_BASE}/inference/single`, {
      text,
      confidence_threshold: confidenceThreshold,
      profile,
      engine_version: engineVersion,
      ...metadata
    });
    return res.data;
  },

  async runBatchInference(
    reviews: string[],
    confidenceThreshold = 0.5,
    profile = 'production_precision',
    engineVersion: InferenceEngineVersion = 'v11'
  ): Promise<any> {
    const res = await axios.post(`${API_BASE}/inference/batch`, {
      reviews,
      confidence_threshold: confidenceThreshold,
      profile,
      engine_version: engineVersion
    });
    return res.data;
  },

  async uploadBatchCsv(
    file: File, engineVersion: InferenceEngineVersion, batchSize: number,
    confidenceThreshold: number, profile: string, onProgress?: (pct: number) => void
  ): Promise<BatchUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('engine_version', engineVersion);
    formData.append('batch_size', String(batchSize));
    formData.append('confidence_threshold', String(confidenceThreshold));
    formData.append('profile', profile);
    const res = await axios.post(`${API_BASE}/batch/upload`, formData, {
      onUploadProgress: (evt) => evt.total && onProgress?.(Math.round(evt.loaded * 100 / evt.total))
    });
    return res.data;
  },

  async startBatch(jobId: string, token: string): Promise<void> {
    await axios.post(`${API_BASE}/batch/${jobId}/start`, {}, { headers: { 'X-Batch-Token': token } });
  },

  async getBatchStatus(jobId: string, token: string): Promise<BatchStatusResponse> {
    const res = await axios.get(`${API_BASE}/batch/${jobId}/status`, { headers: { 'X-Batch-Token': token } });
    return res.data;
  },

  async getBatchResults(jobId: string, token: string, offset = 0, limit = 200): Promise<{ results: BatchRowResult[]; summary?: Record<string, any> }> {
    const res = await axios.get(`${API_BASE}/batch/${jobId}/results`, {
      params: { offset, limit }, headers: { 'X-Batch-Token': token }
    });
    return res.data;
  },

  async cancelBatch(jobId: string, token: string): Promise<void> {
    await axios.post(`${API_BASE}/batch/${jobId}/cancel`, {}, { headers: { 'X-Batch-Token': token } });
  },

  async downloadBatch(jobId: string, token: string, format: 'csv' | 'json'): Promise<void> {
    const res = await axios.get(`${API_BASE}/batch/${jobId}/download`, {
      params: { format }, headers: { 'X-Batch-Token': token }, responseType: 'blob'
    });
    const href = URL.createObjectURL(res.data);
    const anchor = document.createElement('a');
    anchor.href = href;
    anchor.download = `${jobId}.${format}`;
    anchor.click();
    URL.revokeObjectURL(href);
  },

  // Model Registry
  async getModels(): Promise<ModelArtifact[]> {
    const res = await axios.get(`${API_BASE}/models`);
    return res.data;
  },

  async activateModel(id: string): Promise<void> {
    await axios.post(`${API_BASE}/models/${id}/activate`);
  },

  async deleteModel(id: string): Promise<void> {
    await axios.delete(`${API_BASE}/models/${id}`);
  },

  getModelDownloadUrl(id: string): string {
    return `${API_BASE}/models/${id}/download`;
  }
};
