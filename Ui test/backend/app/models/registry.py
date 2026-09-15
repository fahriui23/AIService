import json
import shutil
import zipfile
import io
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from app.core.config import MODELS_DIR, CHECKPOINTS_DIR, REPORTS_DIR, EVALUATION_MODE, PROXY_DISCLAIMER_MESSAGE
from app.database.models import ModelArtifactModel, ExperimentModel, EvaluationReportModel

def list_model_artifacts(db: Session) -> List[Dict[str, Any]]:
    models = db.query(ModelArtifactModel).all()
    result = []
    for m in models:
        result.append({
            "id": m.id,
            "experiment_id": m.experiment_id,
            "model_name": m.model_name,
            "is_best": m.is_best,
            "base_model": m.base_model,
            "run_mode": m.run_mode,
            "metrics": m.metrics_json,
            "evaluation_mode": m.evaluation_mode,
            "created_at": m.created_at.isoformat() if m.created_at else None
        })
    return result

def mark_best_model(db: Session, model_id: str) -> bool:
    models = db.query(ModelArtifactModel).all()
    target = None
    for m in models:
        if m.id == model_id:
            m.is_best = True
            target = m
        else:
            m.is_best = False
    db.commit()
    return target is not None

def delete_model_artifact(db: Session, model_id: str) -> bool:
    model = db.query(ModelArtifactModel).filter(ModelArtifactModel.id == model_id).first()
    if not model:
        return False

    # Delete files
    if model.model_path and Path(model.model_path).exists():
        shutil.rmtree(model.model_path, ignore_errors=True)

    db.delete(model)
    db.commit()
    return True

def create_model_package_zip(db: Session, model_id: str) -> Optional[bytes]:
    model = db.query(ModelArtifactModel).filter(ModelArtifactModel.id == model_id).first()
    if not model:
        return None

    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Config & Metadata
        manifest = {
            "model_id": model.id,
            "experiment_id": model.experiment_id,
            "model_name": model.model_name,
            "base_model": model.base_model,
            "run_mode": model.run_mode,
            "evaluation_mode": model.evaluation_mode,
            "metrics": model.metrics_json,
            "checksum_manifest": model.checksum_manifest_json
        }
        zf.writestr("config.json", json.dumps(manifest, indent=2))

        # README
        readme_content = f"""# ABSA Model V11 — Package {model.model_name}

Run mode: {model.run_mode}
Evaluation mode: {model.evaluation_mode}
Disclaimer: {PROXY_DISCLAIMER_MESSAGE}

## How to run inference:
Use `ABSAInferenceEngine` from `app.inference.engine` or load `inference_config.json`.
"""
        zf.writestr("README_INFERENCE.md", readme_content)

    mem_zip.seek(0)
    return mem_zip.getvalue()
