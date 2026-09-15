from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, JSON
from app.database.db import Base

class DatasetModel(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    checksum_sha256 = Column(String, nullable=False, index=True)
    detected_type = Column(String, nullable=False)  # e.g., v10_master, v11_synthetic, gold_pilot, silver, auxiliary
    review_count = Column(Integer, default=0)
    annotation_count = Column(Integer, default=0)
    human_approved_count = Column(Integer, default=0)
    audit_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ExperimentModel(Base):
    __tablename__ = "experiments"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    run_mode = Column(String, default="dapt_silver_v11synthetic")  # base, dapt_only, silver_only, dapt_silver, dapt_silver_v11synthetic
    base_model = Column(String, default="indobenchmark/indobert-base-p1")
    dataset_ids = Column(JSON, nullable=False)  # List of dataset IDs used
    config_json = Column(JSON, nullable=False)
    status = Column(String, default="created")  # created, queued, preparing, running, calibrating, evaluating, completed, failed, cancelled
    created_at = Column(DateTime, default=datetime.utcnow)

class TrainingJobModel(Base):
    __tablename__ = "training_jobs"

    id = Column(String, primary_key=True, index=True)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False, index=True)
    status = Column(String, default="queued")
    stage = Column(String, default="Stage 1: Dataset preparation")
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=0)
    current_batch = Column(Integer, default=0)
    total_batches = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    stack_trace = Column(Text, nullable=True)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)

class TrainingMetricModel(Base):
    __tablename__ = "training_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False, index=True)
    epoch = Column(Integer, nullable=False)
    batch = Column(Integer, default=0)
    stage = Column(String, nullable=True)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    macro_f1 = Column(Float, nullable=True)
    positive_f1 = Column(Float, nullable=True)
    negative_f1 = Column(Float, nullable=True)
    neutral_f1 = Column(Float, nullable=True)
    aspect_f1 = Column(Float, nullable=True)
    opinion_f1 = Column(Float, nullable=True)
    relation_f1 = Column(Float, nullable=True)
    lr = Column(Float, nullable=True)
    vram_mb = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ModelArtifactModel(Base):
    __tablename__ = "model_artifacts"

    id = Column(String, primary_key=True, index=True)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False, index=True)
    model_name = Column(String, nullable=False)
    is_best = Column(Boolean, default=False)
    base_model = Column(String, nullable=False)
    run_mode = Column(String, nullable=False)
    checkpoint_path = Column(String, nullable=False)
    model_path = Column(String, nullable=False)
    metrics_json = Column(JSON, nullable=False)
    checksum_manifest_json = Column(JSON, nullable=False)
    evaluation_mode = Column(String, default="proxy_without_human_gold")
    created_at = Column(DateTime, default=datetime.utcnow)

class EvaluationReportModel(Base):
    __tablename__ = "evaluation_reports"

    id = Column(String, primary_key=True, index=True)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False, index=True)
    metrics_json = Column(JSON, nullable=False)
    confusion_matrices_json = Column(JSON, nullable=True)
    error_analysis_json = Column(JSON, nullable=True)
    evaluation_mode = Column(String, default="proxy_without_human_gold")
    created_at = Column(DateTime, default=datetime.utcnow)

class InferenceHistoryModel(Base):
    __tablename__ = "inference_history"

    id = Column(String, primary_key=True, index=True)
    model_id = Column(String, nullable=False)
    input_text = Column(Text, nullable=False)
    output_json = Column(JSON, nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class ApplicationEventModel(Base):
    __tablename__ = "application_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    details_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
