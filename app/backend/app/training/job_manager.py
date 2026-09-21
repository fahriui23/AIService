import traceback
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from app.core.config import CHECKPOINTS_DIR
from app.database.db import SessionLocal
from app.database.models import ExperimentModel, TrainingJobModel, TrainingMetricModel, ModelArtifactModel, EvaluationReportModel, ApplicationEventModel
from app.training.pipeline import run_training_pipeline
from app.evaluation.metrics import generate_canonical_evaluation_report

active_threads: Dict[str, threading.Thread] = {}
stop_events: Dict[str, threading.Event] = {}
job_logs: Dict[str, List[str]] = {}

def get_job_logs(experiment_id: str) -> List[str]:
    return job_logs.get(experiment_id, ["No logs available yet."])

def append_log(experiment_id: str, message: str):
    if experiment_id not in job_logs:
        job_logs[experiment_id] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    job_logs[experiment_id].append(log_line)
    # Keep last 1000 lines
    if len(job_logs[experiment_id]) > 1000:
        job_logs[experiment_id] = job_logs[experiment_id][-1000:]

def start_training_job(experiment_id: str) -> bool:
    if experiment_id in active_threads and active_threads[experiment_id].is_alive():
        return False

    stop_event = threading.Event()
    stop_events[experiment_id] = stop_event
    job_logs[experiment_id] = []

    thread = threading.Thread(
        target=_worker_entry,
        args=(experiment_id, stop_event),
        daemon=True
    )
    active_threads[experiment_id] = thread
    thread.start()
    return True

def cancel_training_job(experiment_id: str) -> bool:
    if experiment_id in stop_events:
        stop_events[experiment_id].set()
        append_log(experiment_id, "Cancel signal sent to background worker...")
        return True
    return False

def _worker_entry(experiment_id: str, stop_event: threading.Event):
    db: Session = SessionLocal()
    append_log(experiment_id, f"Initializing background worker for Experiment ID: {experiment_id}")

    try:
        exp = db.query(ExperimentModel).filter(ExperimentModel.id == experiment_id).first()
        if not exp:
            append_log(experiment_id, f"ERROR: Experiment ID {experiment_id} not found in database.")
            return

        job = db.query(TrainingJobModel).filter(TrainingJobModel.experiment_id == experiment_id).first()
        if not job:
            job = TrainingJobModel(
                id=f"job_{experiment_id}",
                experiment_id=experiment_id,
                status="preparing",
                stage="Stage 1: Dataset preparation",
                start_time=datetime.utcnow()
            )
            db.add(job)
        else:
            job.status = "preparing"
            job.stage = "Stage 1: Dataset preparation"
            job.start_time = datetime.utcnow()

        exp.status = "preparing"
        db.commit()

        # Load dataset records from associated dataset records or mock sample if empty
        dataset_records = []
        for ds_id in exp.dataset_ids:
            from app.database.models import DatasetModel
            ds_model = db.query(DatasetModel).filter(DatasetModel.id == ds_id).first()
            if ds_model and ds_model.file_path:
                from app.datasets.loader import detect_dataset_type
                _, recs, _ = detect_dataset_type(Path(ds_model.file_path), ds_model.checksum_sha256)
                dataset_records.extend(recs)

        if not dataset_records:
            raise ValueError("Tidak ada record dataset yang valid. Unggah dataset dan audit sebelum memulai training.")

        append_log(experiment_id, f"Dataset loaded: {len(dataset_records)} review records available for training.")

        def update_job_status(status_str, stage_str, curr_ep, tot_ep, curr_b, tot_b, msg=""):
            db_inner = SessionLocal()
            try:
                j = db_inner.query(TrainingJobModel).filter(TrainingJobModel.experiment_id == experiment_id).first()
                e = db_inner.query(ExperimentModel).filter(ExperimentModel.id == experiment_id).first()
                if j and e:
                    j.status = status_str
                    j.stage = stage_str
                    j.current_epoch = curr_ep
                    j.total_epochs = tot_ep
                    j.current_batch = curr_b
                    j.total_batches = tot_b
                    e.status = status_str
                    db_inner.commit()
                if msg:
                    append_log(experiment_id, msg)
            finally:
                db_inner.close()

        def record_metric(metric_dict):
            db_inner = SessionLocal()
            try:
                m = TrainingMetricModel(
                    experiment_id=experiment_id,
                    epoch=metric_dict.get("epoch", 0),
                    batch=metric_dict.get("batch", 0),
                    stage=metric_dict.get("stage", ""),
                    train_loss=metric_dict.get("train_loss"),
                    val_loss=metric_dict.get("val_loss"),
                    macro_f1=metric_dict.get("macro_f1"),
                    positive_f1=metric_dict.get("positive_f1"),
                    negative_f1=metric_dict.get("negative_f1"),
                    neutral_f1=metric_dict.get("neutral_f1"),
                    aspect_f1=metric_dict.get("aspect_f1"),
                    opinion_f1=metric_dict.get("opinion_f1"),
                    relation_f1=metric_dict.get("relation_f1"),
                    lr=metric_dict.get("lr"),
                    vram_mb=metric_dict.get("vram_mb")
                )
                db_inner.add(m)
                db_inner.commit()
                log_line = (
                    f"Epoch {metric_dict['epoch']} | Loss: {metric_dict['train_loss']:.4f} | "
                    f"Macro F1: {metric_dict['macro_f1']:.4f} | Aspect F1: {metric_dict['aspect_f1']:.4f} | "
                    f"Relation F1: {metric_dict['relation_f1']:.4f}"
                )
                append_log(experiment_id, log_line)
            finally:
                db_inner.close()

        # Run pipeline
        res = run_training_pipeline(
            experiment_id=experiment_id,
            config=exp.config_json,
            dataset_records=dataset_records,
            metric_callback=record_metric,
            job_callback=update_job_status,
            stop_event=stop_event
        )

        if res.get("status") == "cancelled":
            update_job_status("cancelled", "Cancelled", 0, 0, 0, 0, "Training process cancelled by user.")
            return

        # Stage 7 & 8: Generate canonical evaluation report & Model registry entry
        append_log(experiment_id, "Generating Canonical & Proxy Evaluation Reports...")
        eval_report = generate_canonical_evaluation_report(experiment_id, dataset_records)

        eval_model = EvaluationReportModel(
            id=f"eval_{experiment_id}",
            experiment_id=experiment_id,
            metrics_json=eval_report["metrics"],
            confusion_matrices_json=eval_report["confusion_matrices"],
            error_analysis_json=eval_report["error_analysis"],
            evaluation_mode="proxy_without_human_gold"
        )
        db.add(eval_model)

        artifact = ModelArtifactModel(
            id=f"model_{experiment_id}",
            experiment_id=experiment_id,
            model_name=f"V11 Model — {exp.name}",
            is_best=True,
            base_model=exp.base_model,
            run_mode=exp.run_mode,
            checkpoint_path=str(CHECKPOINTS_DIR / experiment_id),
            model_path=res.get("model_dir", ""),
            metrics_json=eval_report["metrics"],
            checksum_manifest_json={"class_exposure": res.get("class_exposure")},
            evaluation_mode="proxy_without_human_gold"
        )
        db.add(artifact)

        job.status = "completed"
        job.stage = "Stage 8: Model packaging"
        job.end_time = datetime.utcnow()
        exp.status = "completed"
        db.commit()

        append_log(experiment_id, f"Training job completed successfully! Best Macro F1: {res.get('best_macro_f1', 0.0):.4f}")

    except Exception as e:
        err_msg = str(e)
        st_trace = traceback.format_exc()
        append_log(experiment_id, f"CRITICAL TRAINING FAILURE: {err_msg}")
        append_log(experiment_id, st_trace)

        job = db.query(TrainingJobModel).filter(TrainingJobModel.experiment_id == experiment_id).first()
        exp = db.query(ExperimentModel).filter(ExperimentModel.id == experiment_id).first()
        if job:
            job.status = "failed"
            job.error_message = err_msg
            job.stack_trace = st_trace
            job.end_time = datetime.utcnow()
        if exp:
            exp.status = "failed"
        db.commit()

    finally:
        db.close()
