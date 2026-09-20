import csv
import io
import json
import time
from pathlib import Path

import pytest

from app.services.batch import BatchJobManager, BatchValidationError
from app.services.absa_adapter import normalize_absa_response
from app.services.enrichment import analyze_enriched, analyze_enriched_batch, location_resolver, resolve_customer


class FakeEngineManager:
    supported_versions = ["v11", "v12"]

    def analyze_single_review(self, version, text, confidence_threshold=0.5, profile="production_precision"):
        lowered = text.lower()
        if "model-error" in lowered:
            raise RuntimeError("private model failure")
        sentiment = "negative" if any(word in lowered for word in ("lambat", "terlambat", "buruk")) else "positive"
        aspect = "pengiriman" if "pengiriman" in lowered else "aplikasi"
        opinion = "lambat" if "lambat" in lowered else "bagus"
        return {
            "text": text,
            "results": [{
                "aspect": aspect, "opinion": opinion, "sentiment": sentiment,
                "taxonomy": "Delivery" if aspect == "pengiriman" else "Application",
                "relation": True, "confidence": 0.91,
            }],
            "processing_time_ms": 0.1, "engine_version": version, "model_version": version,
            "inference_mode": "rule_based_fallback",
        }

    def analyze_batch(self, version, reviews, confidence_threshold=0.5, profile="production_precision"):
        return [self.analyze_single_review(version, review, confidence_threshold, profile) for review in reviews]


def test_customer_id_patterns_classes_and_plain_numbers():
    for value in ("no213", "No 213", "NO:213", "id-213", "customer 213", "customer_id: 213"):
        customer, warnings, _ = resolve_customer(value)
        assert customer["customer_id"] == "213"
        assert customer["customer_class"] == "SILVER"
        assert not warnings
    assert resolve_customer("harga 213 ribu")[0]["customer_id"] is None
    assert resolve_customer("no00123")[0]["customer_id"] == "00123"
    assert resolve_customer("no9123")[0]["customer_class"] == "UNKNOWN"


def test_structured_customer_wins_and_conflict_is_reported():
    customer, warnings, _ = resolve_customer("no312 aplikasi bagus", "145")
    assert customer["customer_id"] == "145"
    assert customer["customer_class"] == "VIP"
    assert "customer_id_conflict" in warnings


@pytest.mark.parametrize(
    "text,status,canonical,province",
    [
        ("Surabaya no213 pengirimannya lambat", "resolved", "Kota Surabaya", "Jawa Timur"),
        ("no312 model ini sangat malang", "not_found", None, None),
        ("no145 saya tinggal di Kota Batu", "resolved", "Kota Batu", "Jawa Timur"),
        ("no245 produk ini keras seperti batu", "not_found", None, None),
        ("no345 cabang Solo pelayanannya lambat", "resolved", "Kota Surakarta", "Jawa Tengah"),
        ("no145 saya bekerja solo", "not_found", None, None),
        ("no345 aplikasi sering menyerang pengguna", "not_found", None, None),
    ],
)
def test_contextual_location_resolution(text, status, canonical, province):
    result, _ = location_resolver.resolve(text)
    assert result["status"] == status
    assert result["city_or_regency"] == canonical
    assert result["province"] == province


def test_city_regency_collision_is_ambiguous():
    bandung, _ = location_resolver.resolve("Bandung no123 aplikasi bagus")
    serang, _ = location_resolver.resolve("no245 pengiriman ke Serang terlambat")
    assert bandung["status"] == "ambiguous"
    assert set(bandung["candidates"]) == {"Kota Bandung", "Kabupaten Bandung"}
    assert serang["status"] == "ambiguous"
    assert set(serang["candidates"]) == {"Kota Serang", "Kabupaten Serang"}


def test_structured_location_and_single_response_are_canonical_and_backward_compatible():
    result = analyze_enriched(
        FakeEngineManager(), "no312 aplikasi sangat bagus", "v12",
        customer_id="213", city="Bandung", province="Jawa Barat",
    )
    assert result["text"] == "aplikasi sangat bagus"
    assert result["results"]
    assert result["customer"]["customer_class"] == "SILVER"
    assert result["location"]["city_or_regency"] == "Kota Bandung"
    assert result["absa"]["aspects"][0]["complaint_taxonomy"] == "Application"
    assert "customer_id_conflict" in result["warnings"]


def test_json_batch_endpoint_row_mapping_matches_single_inference_shape():
    # Mirrors the payload -> rows mapping in POST /api/inference/batch, so a
    # plain string and a metadata-carrying object both reach analyze_enriched_batch
    # and come back with the same canonical shape as /inference/single.
    reviews = ["aplikasi sangat bagus", {"review": "pengiriman lambat", "customer_id": "213", "city": "Bandung"}]
    rows = [
        {"raw_text": r} if isinstance(r, str) else {
            "raw_text": r.get("review", r.get("text", "")),
            "customer_id": r.get("customer_id"),
            "city": r.get("city", r.get("kota")),
            "province": r.get("province", r.get("provinsi")),
        }
        for r in reviews
    ]
    results = analyze_enriched_batch(FakeEngineManager(), rows, "v12")

    assert [item["success"] for item in results] == [True, True]
    assert results[0]["absa"]["aspects"][0]["complaint_taxonomy"] == "Application"
    assert results[1]["customer"]["customer_class"] == "SILVER"
    assert results[1]["location"]["city_or_regency"] == "Kota Bandung"
    assert "timing" in results[0] and "warnings" in results[0]


def _write_csv(path: Path, rows, fieldnames=("review",)):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _wait(job, timeout=5):
    deadline = time.monotonic() + timeout
    while job.status not in {"completed", "failed", "cancelled", "timeout"} and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job.status == "completed"


def test_csv_alias_preview_batch_metrics_order_and_download(tmp_path):
    source = tmp_path / "reviews.csv"
    _write_csv(source, [
        {"komentar": "Bandung no213 aplikasi ini sangat bagus", "customer": ""},
        {"komentar": "Surabaya no312 pengirimannya sangat lambat", "customer": ""},
        {"komentar": "", "customer": "145"},
    ], fieldnames=("customer", "komentar"))
    manager = BatchJobManager(FakeEngineManager(), tmp_path / "jobs")
    job = manager.create_job(source, "reviews.csv", source.stat().st_size, "v11", 8, 0.5, "production_precision")
    assert job.column_map["review"] == "komentar"
    assert job.row_count == 3
    assert len(job.preview) == 3
    with pytest.raises(BatchValidationError, match="Token"):
        manager.get(job.id, "wrong")

    manager.start(job)
    _wait(job)
    results, available = manager.read_results(job, 0, 10)
    assert available == 3
    assert [row["row_number"] for row in results] == [2, 3, 4]
    assert results[0]["customer_class"] == "SILVER"
    assert results[1]["city_or_regency"] == "Kota Surabaya"
    assert results[2]["processing_status"] == "empty_review"
    assert job.summary["successful_rows"] == 2
    assert job.summary["skipped_rows"] == 1
    assert "p95_latency_ms" in job.summary
    assert "throughput_rows_per_second" in job.summary
    assert not source.exists()
    assert json.loads("".join(manager.iter_json_download(job)))[0]["row_number"] == 2
    assert "original_komentar" in "".join(manager.iter_csv_download(job))


def test_invalid_csv_has_stable_error_code(tmp_path):
    source = tmp_path / "bad.csv"
    _write_csv(source, [{"title": "tanpa review"}], fieldnames=("title",))
    manager = BatchJobManager(FakeEngineManager(), tmp_path / "jobs")
    with pytest.raises(BatchValidationError) as caught:
        manager.create_job(source, "bad.csv", source.stat().st_size, "v11", 32, 0.5, "production_precision")
    assert caught.value.code == "REVIEW_COLUMN_NOT_FOUND"


def test_wrapped_voc_export_detects_isi_review(tmp_path):
    source = tmp_path / "voc.csv"
    source.write_text(
        '\"Review ID,\"\"Waktu Review\"\",\"\"Isi Review\"\"\";;;\n'
        '\"131023,\"\"2026-08-27 16:24:46 WIB\"\",\"\"Pelayanannya ramah, cepat dan baik.\"\"\";;;\n',
        encoding="utf-8",
    )
    manager = BatchJobManager(FakeEngineManager(), tmp_path / "jobs")
    job = manager.create_job(source, source.name, source.stat().st_size, "v11", 8, 0.5, "production_precision")
    assert job.source_format == "wrapped_voc_export"
    assert job.column_map["review"] == "Isi Review"
    assert job.preview[0]["Isi Review"] == "Pelayanannya ramah, cepat dan baik."


def test_row_model_error_does_not_stop_batch(tmp_path):
    source = tmp_path / "partial.csv"
    _write_csv(source, [{"review": "model-error"}, {"review": "no123 aplikasi bagus"}])
    manager = BatchJobManager(FakeEngineManager(), tmp_path / "jobs")
    job = manager.create_job(source, source.name, source.stat().st_size, "v12", 1, 0.5, "production_precision")
    manager.start(job)
    _wait(job)
    results, _ = manager.read_results(job, 0, 10)
    assert [row["processing_status"] for row in results] == ["model_error", "success"]
    assert "private model failure" not in json.dumps(results)


def test_export_guards_against_csv_formula_injection(tmp_path):
    source = tmp_path / "formula.csv"
    _write_csv(source, [{"review": "=HYPERLINK(\"bad\")"}])
    manager = BatchJobManager(FakeEngineManager(), tmp_path / "jobs")
    job = manager.create_job(source, source.name, source.stat().st_size, "v11", 1, 0.5, "production_precision")
    manager.start(job)
    _wait(job)
    exported = "".join(manager.iter_csv_download(job))
    assert "'=HYPERLINK" in exported


@pytest.mark.parametrize("version", ["v11", "v12"])
def test_canonical_adapter_supports_every_configured_model(version):
    result = analyze_enriched(FakeEngineManager(), "no123 aplikasi bagus", version)
    assert result["engine_version"] == version
    assert result["absa"]["aspects"][0]["sentiment"] == "positive"


def test_canonical_adapter_normalizes_indonesian_sentiment_labels():
    result = normalize_absa_response({"results": [{"aspect": "layanan", "opinion": "buruk", "sentiment": "negatif", "taxonomy": "Service Quality"}]})
    assert result["aspects"][0]["sentiment"] == "negative"
    assert result["aspects"][0]["complaint_taxonomy"] == "Service Quality"


def test_large_csv_stream_keeps_order(tmp_path):
    source = tmp_path / "large.csv"
    _write_csv(source, ({"review": f"no213 aplikasi bagus baris {index}"} for index in range(1000)))
    manager = BatchJobManager(FakeEngineManager(), tmp_path / "jobs")
    job = manager.create_job(source, source.name, source.stat().st_size, "v11", 64, 0.5, "production_precision")
    manager.start(job)
    _wait(job)
    first, available = manager.read_results(job, 0, 1)
    last, _ = manager.read_results(job, 999, 1)
    assert available == 1000
    assert first[0]["row_number"] == 2
    assert last[0]["row_number"] == 1001


def test_cancel_before_start_removes_temporary_upload(tmp_path):
    source = tmp_path / "cancel.csv"
    _write_csv(source, [{"review": "no123 aplikasi bagus"}])
    manager = BatchJobManager(FakeEngineManager(), tmp_path / "jobs")
    job = manager.create_job(source, source.name, source.stat().st_size, "v11", 1, 0.5, "production_precision")
    manager.cancel(job)
    assert job.status == "cancelled"
    assert not source.exists()


def test_timeout_preserves_safe_terminal_state(tmp_path, monkeypatch):
    import app.services.batch as batch_module
    source = tmp_path / "timeout.csv"
    _write_csv(source, [{"review": "no123 aplikasi bagus"}])
    manager = BatchJobManager(FakeEngineManager(), tmp_path / "jobs")
    job = manager.create_job(source, source.name, source.stat().st_size, "v11", 1, 0.5, "production_precision")
    monkeypatch.setattr(batch_module, "BATCH_TIMEOUT_SECONDS", -1)
    manager.start(job)
    deadline = time.monotonic() + 2
    while job.status not in {"timeout", "failed"} and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job.status == "timeout"
    assert job.summary["processed_rows"] == 0
