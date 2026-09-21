export interface SystemStatus {
  gpu_available: boolean;
  gpu_name: string;
  vram_total_mb: number;
  vram_used_mb: number;
  vram_free_mb: number;
  cpu_count: number;
  cpu_usage_percent: number;
  ram_total_gb: number;
  ram_used_gb: number;
  ram_usage_percent: number;
  disk_total_gb: number;
  disk_used_gb: number;
  disk_free_gb: number;
  disk_usage_percent: number;
  evaluation_mode: string;
  proxy_disclaimer: string;
}

export interface Dataset {
  id: string;
  name: string;
  original_filename: string;
  checksum_sha256: string;
  detected_type: string;
  review_count: number;
  annotation_count: number;
  human_approved_count: number;
  audit?: Record<string, any>;
  created_at?: string;
}

export interface ExperimentConfig {
  experiment_name: string;
  run_mode: 'base' | 'dapt_only' | 'silver_only' | 'dapt_silver' | 'dapt_silver_v11synthetic';
  base_model: string;
  random_seed: number;
  batch_size: number;
  learning_rate: number;
  weight_decay: number;
  warmup_ratio: number;
  max_length: number;
  gradient_accumulation_steps: number;
  mixed_precision: boolean;
  number_of_epochs: number;
  early_stopping_patience: number;
  checkpoint_interval: number;
  evaluation_interval: number;
  neutral_hardening_enabled: boolean;
  neutral_hardening_epochs: number;
  controlled_sampling_enabled: boolean;
  positive_exposure_target: number;
  negative_exposure_target: number;
  neutral_exposure_target: number;
  dapt_enabled: boolean;
  silver_enabled: boolean;
  v11_synthetic_enabled: boolean;
  calibration_enabled: boolean;
  relation_hardening_enabled: boolean;
  dataset_ids: string[];
}

export interface Experiment {
  id: string;
  name: string;
  run_mode: string;
  base_model: string;
  status: 'created' | 'queued' | 'preparing' | 'running' | 'calibrating' | 'evaluating' | 'completed' | 'failed' | 'cancelled';
  stage?: string;
  config?: Record<string, any>;
  job?: {
    stage: string;
    current_epoch: number;
    total_epochs: number;
    current_batch: number;
    total_batches: number;
    error_message?: string;
  };
  created_at?: string;
}

export interface TrainingMetric {
  epoch: number;
  batch: number;
  stage: string;
  train_loss: number;
  val_loss: number;
  macro_f1: number;
  positive_f1: number;
  negative_f1: number;
  neutral_f1: number;
  aspect_f1: number;
  opinion_f1: number;
  relation_f1: number;
  lr: number;
  vram_mb: number;
  timestamp?: string;
}

export interface InferenceResult {
  aspect: string;
  opinion: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  taxonomy: string;
  relation: boolean;
  confidence: number;
  aspect_span?: [number, number];
  opinion_span?: [number, number] | null;
  domain?: string;
  domain_confidence?: number;
  entity_id?: string;
  issue_id?: string;
  taxonomy_confidence?: number;
  taxonomy_abstained?: boolean;
  taxonomy_low_confidence?: boolean;
  taxonomy_policy?: 'best_effort' | 'strict';
  needs_human_review?: boolean;
  review_reasons?: string[];
}

export type InferenceEngineVersion = 'v11' | 'v12' | 'v14' | 'v15';

export interface InferenceEngineInfo {
  id: InferenceEngineVersion;
  label: string;
  available: boolean;
  active: boolean;
  bundle: string;
  model_version: string;
  languages: string[];
  components?: Record<string, boolean> | null;
  status?: 'candidate_unpromoted' | null;
}

export interface SingleInferenceResponse {
  text: string;
  results: InferenceResult[];
  processing_time_ms: number;
  engine_version: InferenceEngineVersion;
  model_version?: string;
  model_source?: string;
  model_bundle_loaded?: boolean;
  inference_mode?: 'transformer_bundle' | 'rule_based_fallback';
  model_loading_mode?: 'resident' | 'streaming';
  model_load_error?: string;
  evaluation_mode?: string;
  disclaimer?: string;
  promotion_status?: 'candidate_unpromoted' | 'production_ready';
  human_gold_status?: 'not_available' | 'available';
  raw_text?: string;
  clean_review?: string;
  customer?: {
    customer_id: string | null;
    customer_class: string;
    classification_method: string | null;
    source?: string | null;
  };
  location?: {
    raw_location: string | null;
    city_or_regency: string | null;
    province: string | null;
    location_type: string | null;
    semantic_type?: string;
    status: 'resolved' | 'ambiguous' | 'not_found' | 'conflict';
    confidence: number;
    candidates: string[];
    evidence: string[];
  };
  absa?: { aspects: Array<InferenceResult & { complaint_taxonomy?: string | null }> };
  warnings?: string[];
  timing?: Record<string, number>;
}

export interface BatchUploadResponse {
  success: boolean;
  job_id: string;
  access_token: string;
  file: {
    name: string;
    size_bytes: number;
    row_count: number;
    columns: string[];
    review_column: string;
    encoding: string;
    delimiter: string;
    source_format?: string;
  };
  preview: Record<string, string>[];
  validation: { status: string; warnings: string[] };
  model_version: InferenceEngineVersion;
  batch_size: number;
}

export interface BatchStatusResponse {
  success: boolean;
  job_id: string;
  status: 'uploaded' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'timeout';
  model_version: string;
  progress: {
    total_rows: number;
    processed_rows: number;
    successful_rows: number;
    failed_rows: number;
    skipped_rows: number;
    percentage: number;
  };
  performance: {
    elapsed_time_ms: number;
    estimated_remaining_ms: number | null;
    throughput_rows_per_second: number;
  };
  summary?: Record<string, any> | null;
  error?: { code: string; message: string } | null;
}

export interface BatchRowResult {
  row_number: number;
  raw_review: string;
  clean_review: string;
  customer_id?: string | null;
  customer_class?: string;
  city_or_regency?: string | null;
  province?: string | null;
  location_status?: string;
  location_confidence?: number;
  aspects: Array<InferenceResult & { complaint_taxonomy?: string | null }>;
  model_version: string;
  processing_status: string;
  processing_time_ms: number;
  warnings: string[];
  error_message?: string;
}

export interface ModelArtifact {
  id: string;
  experiment_id: string;
  model_name: string;
  is_best: boolean;
  base_model: string;
  run_mode: string;
  metrics: Record<string, any>;
  evaluation_mode: string;
  created_at?: string;
}
