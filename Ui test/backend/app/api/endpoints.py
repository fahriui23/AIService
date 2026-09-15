import uuid
import json
import os
from time import perf_counter
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Response, Query, Header
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from sqlalchemy.orm import Session

from app.core.config import (
    get_hardware_status, EVALUATION_MODE, PROXY_DISCLAIMER_MESSAGE, BATCH_JOBS_DIR,
    DEFAULT_BATCH_SIZE, MAX_CSV_FILE_SIZE_MB,
)
from app.database.db import get_db
from app.database.models import DatasetModel, ExperimentModel, TrainingJobModel, TrainingMetricModel, ModelArtifactModel, EvaluationReportModel, InferenceHistoryModel
from app.datasets.loader import save_uploaded_file, detect_dataset_type
from app.datasets.audit import audit_dataset_records
from app.training.job_manager import start_training_job, cancel_training_job, get_job_logs
from app.inference.engine import inference_engine_manager
from app.services.enrichment import analyze_enriched
from app.services.batch import BatchJobManager, BatchValidationError
from app.models.registry import list_model_artifacts, mark_best_model, delete_model_artifact, create_model_package_zip

router = APIRouter()
batch_job_manager = BatchJobManager(inference_engine_manager)
DEFAULT_ENGINE_VERSION = os.environ.get("ABSA_ENGINE_ONLY", "v11").strip().lower() or "v11"


def _batch_error(exc: BatchValidationError, status_code: int = 400):
    if exc.code == "JOB_NOT_FOUND":
        status_code = 404
    elif exc.code == "JOB_ACCESS_DENIED":
        status_code = 403
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": {"code": exc.code, "message": exc.message}},
    )

# --- System & Health ---
@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": os.environ.get("ABSA_SERVICE_NAME", "ABSA Multi-Engine Web API"),
        "version": os.environ.get("ABSA_SERVICE_VERSION", "v15.1"),
        "supported_engines": inference_engine_manager.supported_versions,
        "evaluation_mode": EVALUATION_MODE,
    }

@router.get("/system")
def get_system_status():
    return get_hardware_status()

# --- Datasets ---
@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_bytes = await file.read()
    try:
        saved_path, checksum = save_uploaded_file(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    detected_type, records, file_meta = detect_dataset_type(saved_path, checksum)
    audit_results = audit_dataset_records(records, detected_type)

    dataset_id = f"ds_{uuid.uuid4().hex[:10]}"
    ds_model = DatasetModel(
        id=dataset_id,
        name=Path(file.filename).stem,
        original_filename=file.filename,
        file_path=str(saved_path),
        checksum_sha256=checksum,
        detected_type=detected_type,
        review_count=audit_results["review_count"],
        annotation_count=audit_results["annotation_count"],
        human_approved_count=audit_results["human_approved_count"],
        audit_json=audit_results
    )
    db.add(ds_model)
    db.commit()
    db.refresh(ds_model)

    return {
        "id": ds_model.id,
        "name": ds_model.name,
        "original_filename": ds_model.original_filename,
        "checksum_sha256": ds_model.checksum_sha256,
        "detected_type": ds_model.detected_type,
        "review_count": ds_model.review_count,
        "annotation_count": ds_model.annotation_count,
        "audit": audit_results
    }

@router.get("/datasets")
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(DatasetModel).all()
    out = []
    for d in datasets:
        out.append({
            "id": d.id,
            "name": d.name,
            "original_filename": d.original_filename,
            "checksum_sha256": d.checksum_sha256,
            "detected_type": d.detected_type,
            "review_count": d.review_count,
            "annotation_count": d.annotation_count,
            "human_approved_count": d.human_approved_count,
            "created_at": d.created_at.isoformat() if d.created_at else None
        })
    return out

@router.get("/datasets/{dataset_id}")
def get_dataset_detail(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {
        "id": ds.id,
        "name": ds.name,
        "original_filename": ds.original_filename,
        "file_path": ds.file_path,
        "checksum_sha256": ds.checksum_sha256,
        "detected_type": ds.detected_type,
        "review_count": ds.review_count,
        "annotation_count": ds.annotation_count,
        "human_approved_count": ds.human_approved_count,
        "audit": ds.audit_json,
        "created_at": ds.created_at.isoformat() if ds.created_at else None
    }

@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.delete(ds)
    db.commit()
    return {"status": "success", "message": f"Dataset {dataset_id} deleted"}

# --- Experiments & Training ---
@router.post("/experiments")
def create_experiment(payload: Dict[str, Any], db: Session = Depends(get_db)):
    exp_id = f"exp_{uuid.uuid4().hex[:10]}"
    exp = ExperimentModel(
        id=exp_id,
        name=payload.get("experiment_name", "ABSA V11 Experiment"),
        run_mode=payload.get("run_mode", "dapt_silver_v11synthetic"),
        base_model=payload.get("base_model", "indobenchmark/indobert-base-p1"),
        dataset_ids=payload.get("dataset_ids", []),
        config_json=payload,
        status="created"
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return {"id": exp.id, "name": exp.name, "status": exp.status}

@router.get("/experiments")
def list_experiments(db: Session = Depends(get_db)):
    exps = db.query(ExperimentModel).all()
    out = []
    for e in exps:
        job = db.query(TrainingJobModel).filter(TrainingJobModel.experiment_id == e.id).first()
        out.append({
            "id": e.id,
            "name": e.name,
            "run_mode": e.run_mode,
            "base_model": e.base_model,
            "status": e.status,
            "stage": job.stage if job else "Created",
            "created_at": e.created_at.isoformat() if e.created_at else None
        })
    return out

@router.get("/experiments/{experiment_id}")
def get_experiment_detail(experiment_id: str, db: Session = Depends(get_db)):
    exp = db.query(ExperimentModel).filter(ExperimentModel.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    job = db.query(TrainingJobModel).filter(TrainingJobModel.experiment_id == experiment_id).first()
    return {
        "id": exp.id,
        "name": exp.name,
        "run_mode": exp.run_mode,
        "base_model": exp.base_model,
        "dataset_ids": exp.dataset_ids,
        "config": exp.config_json,
        "status": exp.status,
        "job": {
            "stage": job.stage if job else "N/A",
            "current_epoch": job.current_epoch if job else 0,
            "total_epochs": job.total_epochs if job else 0,
            "current_batch": job.current_batch if job else 0,
            "total_batches": job.total_batches if job else 0,
            "error_message": job.error_message if job else None
        } if job else None,
        "created_at": exp.created_at.isoformat() if exp.created_at else None
    }

@router.post("/experiments/{experiment_id}/start")
def start_experiment_training(experiment_id: str, db: Session = Depends(get_db)):
    exp = db.query(ExperimentModel).filter(ExperimentModel.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    success = start_training_job(experiment_id)
    if not success:
        raise HTTPException(status_code=400, detail="Job is already running")

    return {"status": "success", "message": f"Training started for experiment {experiment_id}"}

@router.post("/experiments/{experiment_id}/cancel")
def cancel_experiment_training(experiment_id: str):
    success = cancel_training_job(experiment_id)
    if not success:
        raise HTTPException(status_code=400, detail="No active training job to cancel")
    return {"status": "success", "message": f"Cancel signal sent for experiment {experiment_id}"}

@router.get("/experiments/{experiment_id}/logs")
def get_experiment_logs(experiment_id: str):
    logs = get_job_logs(experiment_id)
    return {"experiment_id": experiment_id, "logs": logs}

@router.get("/experiments/{experiment_id}/metrics")
def get_experiment_metrics(experiment_id: str, db: Session = Depends(get_db)):
    metrics = db.query(TrainingMetricModel).filter(TrainingMetricModel.experiment_id == experiment_id).order_by(TrainingMetricModel.id.asc()).all()
    out = []
    for m in metrics:
        out.append({
            "epoch": m.epoch,
            "batch": m.batch,
            "stage": m.stage,
            "train_loss": m.train_loss,
            "val_loss": m.val_loss,
            "macro_f1": m.macro_f1,
            "positive_f1": m.positive_f1,
            "negative_f1": m.negative_f1,
            "neutral_f1": m.neutral_f1,
            "aspect_f1": m.aspect_f1,
            "opinion_f1": m.opinion_f1,
            "relation_f1": m.relation_f1,
            "lr": m.lr,
            "vram_mb": m.vram_mb,
            "timestamp": m.timestamp.isoformat() if m.timestamp else None
        })
    return out

# --- Evaluation & Reports ---
@router.get("/reports/{experiment_id}")
def get_evaluation_report(experiment_id: str, db: Session = Depends(get_db)):
    report = db.query(EvaluationReportModel).filter(EvaluationReportModel.experiment_id == experiment_id).first()
    if not report:
        # Generate on the fly if missing
        from app.evaluation.metrics import generate_canonical_evaluation_report
        rep_dict = generate_canonical_evaluation_report(experiment_id, [])
        return rep_dict

    return {
        "experiment_id": experiment_id,
        "evaluation_mode": report.evaluation_mode,
        "disclaimer": PROXY_DISCLAIMER_MESSAGE,
        "metrics": report.metrics_json,
        "confusion_matrices": report.confusion_matrices_json,
        "error_analysis": report.error_analysis_json
    }

@router.get("/ablation")
def get_ablation_matrix(db: Session = Depends(get_db)):
    exps = db.query(ExperimentModel).filter(ExperimentModel.status == "completed").all()
    results = []

    # Standard run modes for V11 ablation
    standard_modes = ["base", "dapt_only", "silver_only", "dapt_silver", "dapt_silver_v11synthetic"]

    for exp in exps:
        rep = db.query(EvaluationReportModel).filter(EvaluationReportModel.experiment_id == exp.id).first()
        m = rep.metrics_json if rep else {}
        sent_m = m.get("sentiment", {})
        asp_m = m.get("aspect", {})
        rel_m = m.get("relation", {})

        results.append({
            "experiment_id": exp.id,
            "experiment_name": exp.name,
            "run_mode": exp.run_mode,
            "macro_f1": sent_m.get("macro_f1"),
            "positive_f1": sent_m.get("positive", {}).get("f1"),
            "negative_f1": sent_m.get("negative", {}).get("f1"),
            "neutral_f1": sent_m.get("neutral", {}).get("f1"),
            "aspect_f1": asp_m.get("f1"),
            "relation_f1": rel_m.get("f1"),
            "delta_vs_baseline": None
        })

    # No completed experiments means no ablation result is available.
    return {
        "ablation_results": results,
        "evaluation_mode": EVALUATION_MODE,
        "disclaimer": PROXY_DISCLAIMER_MESSAGE
    }

# --- Inference Playground ---
@router.get("/inference/engines")
def get_inference_engines():
    return {"engines": inference_engine_manager.list_engines()}


@router.post("/inference/single")
def run_single_inference(payload: Dict[str, Any], db: Session = Depends(get_db)):
    text = payload.get("review", payload.get("text", "Dokternya ramah tetapi antreannya lama."))
    threshold = float(payload.get("confidence_threshold", 0.5))
    profile = payload.get("profile", "production_precision")
    engine_version = str(payload.get("engine_version", DEFAULT_ENGINE_VERSION)).lower()

    try:
        res = analyze_enriched(
            inference_engine_manager, text, engine_version, threshold, profile,
            customer_id=payload.get("customer_id"), city=payload.get("city", payload.get("kota")),
            province=payload.get("province", payload.get("provinsi")),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Save to history
    inf_hist = InferenceHistoryModel(
        id=f"inf_{uuid.uuid4().hex[:10]}",
        model_id=payload.get("model_id", f"{engine_version}_default"),
        input_text=text,
        output_json=res,
        processing_time_ms=res["processing_time_ms"]
    )
    db.add(inf_hist)
    db.commit()

    return res

@router.post("/inference/batch")
def run_batch_inference(payload: Dict[str, Any]):
    reviews = payload.get("reviews", ["Kopinya enak tetapi nunggunya lama banget."])
    threshold = float(payload.get("confidence_threshold", 0.5))
    profile = payload.get("profile", "production_precision")
    engine_version = str(payload.get("engine_version", DEFAULT_ENGINE_VERSION)).lower()

    try:
        res = inference_engine_manager.analyze_batch(engine_version, reviews, threshold, profile)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"total": len(res), "engine_version": engine_version, "results": res}


# --- CSV Batch Analysis Jobs ---
@router.post("/batch/upload")
async def upload_batch_csv(
    file: UploadFile = File(...),
    engine_version: str = Form(DEFAULT_ENGINE_VERSION),
    batch_size: int = Form(DEFAULT_BATCH_SIZE),
    confidence_threshold: float = Form(0.5),
    profile: str = Form("production_precision"),
):
    upload_started = perf_counter()
    filename = Path(file.filename or "upload.csv").name
    if Path(filename).suffix.casefold() != ".csv":
        return _batch_error(BatchValidationError("INVALID_FILE_TYPE", "File harus menggunakan ekstensi .csv."))
    permitted_types = {"text/csv", "application/csv", "application/vnd.ms-excel", "text/plain", "application/octet-stream", ""}
    if (file.content_type or "").casefold() not in permitted_types:
        return _batch_error(BatchValidationError("INVALID_MIME_TYPE", "MIME type file bukan CSV."))
    upload_path = BATCH_JOBS_DIR / f"upload_{uuid.uuid4().hex}.csv"
    size_bytes = 0
    try:
        with upload_path.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size_bytes += len(chunk)
                if size_bytes > MAX_CSV_FILE_SIZE_MB * 1024 * 1024:
                    raise BatchValidationError("FILE_TOO_LARGE", f"Ukuran file melebihi {MAX_CSV_FILE_SIZE_MB} MB.")
                handle.write(chunk)
        job = batch_job_manager.create_job(
            upload_path, filename, size_bytes, str(engine_version).lower(), int(batch_size),
            float(confidence_threshold), profile,
        )
        response = batch_job_manager.upload_response(job)
        response["performance"] = {"upload_time_ms": round((perf_counter() - upload_started) * 1000, 3)}
        return response
    except BatchValidationError as exc:
        upload_path.unlink(missing_ok=True)
        return _batch_error(exc)
    except Exception:
        upload_path.unlink(missing_ok=True)
        return _batch_error(BatchValidationError("INVALID_CSV", "File CSV tidak dapat diproses."))
    finally:
        await file.close()


@router.post("/batch/{job_id}/start")
def start_batch_job(job_id: str, x_batch_token: Optional[str] = Header(None)):
    try:
        job = batch_job_manager.get(job_id, x_batch_token)
        batch_job_manager.start(job)
        return {"success": True, "job_id": job.id, "status": job.status}
    except BatchValidationError as exc:
        return _batch_error(exc)


@router.get("/batch/{job_id}/status")
def get_batch_status(job_id: str, x_batch_token: Optional[str] = Header(None)):
    try:
        return batch_job_manager.status_response(batch_job_manager.get(job_id, x_batch_token))
    except BatchValidationError as exc:
        return _batch_error(exc)


@router.get("/batch/{job_id}/results")
def get_batch_results(
    job_id: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000),
    x_batch_token: Optional[str] = Header(None),
):
    try:
        job = batch_job_manager.get(job_id, x_batch_token)
        results, available = batch_job_manager.read_results(job, offset, limit)
        return {
            "success": True, "job_id": job.id, "status": job.status, "model_version": job.engine_version,
            "summary": job.summary or None, "pagination": {
                "offset": offset, "limit": limit, "returned": len(results), "available": available,
                "has_more": offset + len(results) < available,
            }, "results": results,
        }
    except BatchValidationError as exc:
        return _batch_error(exc)


@router.get("/batch/{job_id}/download")
def download_batch_results(
    job_id: str, format: str = Query("csv", pattern="^(csv|json)$"),
    token: Optional[str] = Query(None), x_batch_token: Optional[str] = Header(None),
):
    try:
        job = batch_job_manager.get(job_id, x_batch_token or token)
        if format == "json":
            iterator = batch_job_manager.iter_json_download(job)
            media_type, suffix = "application/json; charset=utf-8", "json"
        else:
            iterator = batch_job_manager.iter_csv_download(job)
            media_type, suffix = "text/csv; charset=utf-8", "csv"
        return StreamingResponse(
            iterator, media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{job.id}.{suffix}"'},
        )
    except BatchValidationError as exc:
        return _batch_error(exc)


@router.post("/batch/{job_id}/cancel")
def cancel_batch_job(job_id: str, x_batch_token: Optional[str] = Header(None)):
    try:
        job = batch_job_manager.get(job_id, x_batch_token)
        batch_job_manager.cancel(job)
        return {"success": True, "job_id": job.id, "status": job.status}
    except BatchValidationError as exc:
        return _batch_error(exc)

# --- Model Registry ---
@router.get("/models")
def get_registered_models(db: Session = Depends(get_db)):
    return list_model_artifacts(db)

@router.post("/models/{model_id}/activate")
def activate_best_model(model_id: str, db: Session = Depends(get_db)):
    success = mark_best_model(db, model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model artifact not found")
    return {"status": "success", "message": f"Model {model_id} set as best active model"}

@router.delete("/models/{model_id}")
def remove_model_artifact(model_id: str, db: Session = Depends(get_db)):
    success = delete_model_artifact(db, model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model artifact not found")
    return {"status": "success", "message": f"Model {model_id} deleted"}

@router.get("/models/{model_id}/download")
def download_model_package(model_id: str, db: Session = Depends(get_db)):
    zip_bytes = create_model_package_zip(db, model_id)
    if not zip_bytes:
        raise HTTPException(status_code=404, detail="Model artifact not found")

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=absa_v11_model_{model_id}.zip"}
    )
