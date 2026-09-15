import io
import zipfile
from pathlib import Path

import pytest

from app.datasets.loader import validate_zip_slip, sanitize_filename
from app.inference.engine import ABSAEngineManager, ABSAInferenceEngine
from app.inference.v14_engine import spans_overlap


def test_sanitize_filename_prevents_path_traversal():
    assert sanitize_filename("../../secret.csv") == "secret.csv"


def test_zip_slip_is_rejected(tmp_path: Path):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../escape.txt", "unsafe")
    archive.seek(0)
    with zipfile.ZipFile(archive) as zf:
        with pytest.raises(ValueError):
            validate_zip_slip(zf, tmp_path / "extract")


def test_single_inference_supports_multiple_aspects():
    result = ABSAInferenceEngine().analyze_single_review(
        "Dokternya ramah tetapi antreannya lama."
    )
    assert len(result["results"]) >= 2
    assert {item["sentiment"] for item in result["results"]} == {"positive", "negative"}


def test_inference_is_deterministic():
    engine = ABSAInferenceEngine()
    first = engine.analyze_single_review("Kopinya enak tetapi nunggunya lama banget.")
    second = engine.analyze_single_review("Kopinya enak tetapi nunggunya lama banget.")
    assert [(x["aspect"], x["confidence"]) for x in first["results"]] == [
        (x["aspect"], x["confidence"]) for x in second["results"]
    ]


def test_engine_registry_exposes_all_bundles():
    manager = ABSAEngineManager()
    engines = {engine["id"]: engine for engine in manager.list_engines()}

    assert set(engines) == {"v11", "v12", "v14", "v15"}
    assert engines["v11"]["available"] is True
    assert engines["v12"]["available"] is True
    assert engines["v12"]["languages"] == ["id", "en", "id-en"]
    assert engines["v14"]["available"] is True
    assert engines["v14"]["components"]["sentiment"] is True
    assert engines["v15"]["available"] is True
    assert engines["v15"]["model_version"] == "V15.1"
    assert engines["v15"]["components"]["joint_relation_sentiment"] is True
    assert engines["v15"]["status"] == "candidate_unpromoted"


def test_engine_registry_rejects_unknown_version():
    manager = ABSAEngineManager()
    with pytest.raises(ValueError, match="tidak tersedia"):
        manager._get_engine("v99")


def test_v14_rejects_overlapping_candidate_without_crashing():
    aspect = {"start": 0, "end": 8}
    opinion = {"start": 4, "end": 12}
    assert spans_overlap(aspect, opinion) is True
    assert spans_overlap(aspect, {"start": 8, "end": 12}) is False
