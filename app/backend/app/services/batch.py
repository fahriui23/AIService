import csv
import io
import json
import logging
import re
import secrets
import statistics
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from app.core.config import (
    BATCH_JOBS_DIR, BATCH_RESULT_TTL_SECONDS, BATCH_TIMEOUT_SECONDS, CSV_COLUMN_ALIASES, DEFAULT_BATCH_SIZE,
    MAX_BATCH_SIZE, MAX_CSV_FILE_SIZE_MB, MAX_CSV_ROWS, MAX_RETRY_PER_ROW, MAX_REVIEW_LENGTH,
)
from app.services.enrichment import analyze_enriched, analyze_enriched_batch


logger = logging.getLogger("absa.batch")


class BatchValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _safe_csv_value(value: Any) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if text.startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


@dataclass
class BatchJob:
    id: str
    owner_token: str
    filename: str
    source_path: Path
    results_path: Path
    size_bytes: int
    encoding: str
    delimiter: str
    source_format: str
    columns: List[str]
    column_map: Dict[str, Optional[str]]
    row_count: int
    preview: List[Dict[str, str]]
    engine_version: str
    batch_size: int
    confidence_threshold: float = 0.5
    profile: str = "production_precision"
    status: str = "uploaded"
    created_at: str = field(default_factory=_now)
    created_monotonic: float = field(default_factory=perf_counter, repr=False)
    queued_monotonic: Optional[float] = field(default=None, repr=False)
    queue_time_ms: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    processed_rows: int = 0
    successful_rows: int = 0
    failed_rows: int = 0
    skipped_rows: int = 0
    error: Optional[Dict[str, str]] = None
    elapsed_time_ms: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)
    cancellation: threading.Event = field(default_factory=threading.Event, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class BatchJobManager:
    def __init__(self, engine_manager: Any, jobs_dir: Path = BATCH_JOBS_DIR):
        self.engine_manager = engine_manager
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, BatchJob] = {}
        self._lock = threading.RLock()

    def _prune_expired(self) -> None:
        cutoff_age = BATCH_RESULT_TTL_SECONDS
        now = perf_counter()
        with self._lock:
            expired = [
                job_id for job_id, job in self._jobs.items()
                if job.status in {"completed", "failed", "cancelled", "timeout"}
                and now - job.created_monotonic > cutoff_age
            ]
            for job_id in expired:
                job = self._jobs.pop(job_id)
                job.results_path.unlink(missing_ok=True)

    @staticmethod
    def detect_encoding(path: Path) -> str:
        sample = path.read_bytes()[:65536]
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                sample.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue
        raise BatchValidationError("UNSUPPORTED_ENCODING", "Encoding CSV tidak didukung.")

    @staticmethod
    def _detect_dialect(path: Path, encoding: str) -> Tuple[csv.Dialect, str]:
        with path.open("r", encoding=encoding, newline="") as handle:
            sample = handle.read(65536)
        if not sample.strip():
            raise BatchValidationError("EMPTY_CSV", "File CSV kosong.")
        first_line = sample.splitlines()[0].strip() if sample.splitlines() else ""
        # Some VOC exports wrap the complete comma-separated record in another
        # CSV field and append three empty semicolon fields: "a,""b""";;;.
        # Decode that outer layer before parsing the actual CSV columns.
        if first_line.startswith('"') and first_line.endswith('";;;'):
            try:
                outer = next(csv.reader([first_line], delimiter=";"))
                inner = next(csv.reader([outer[0]], delimiter=","))
                if len(inner) > 1 and not any(value.strip() for value in outer[1:]):
                    return csv.excel, "wrapped_voc_export"
            except csv.Error:
                pass
        try:
            return csv.Sniffer().sniff(sample, delimiters=",;\t"), "standard"
        except csv.Error as exc:
            # A one-column CSV has no delimiter to sniff and is still valid.
            if not any(delimiter in first_line for delimiter in (",", ";", "\t")):
                return csv.excel, "standard"
            raise BatchValidationError("INVALID_CSV", "Delimiter atau struktur CSV tidak dapat dikenali.") from exc

    @staticmethod
    def _iter_csv_values(
        path: Path, encoding: str, dialect: csv.Dialect, source_format: str
    ) -> Iterator[List[str]]:
        with path.open("r", encoding=encoding, newline="") as handle:
            if source_format == "wrapped_voc_export":
                # These exports are one logical record per physical line. Some
                # source reviews contain unescaped quote/semicolon combinations,
                # so parsing the outer wrapper as CSV would split valid reviews.
                for line in handle:
                    inner = re.sub(r";+\s*$", "", line.rstrip("\r\n"))
                    if not inner.strip():
                        continue
                    if len(inner) >= 2 and inner.startswith('"') and inner.endswith('"'):
                        inner = inner[1:-1].replace('""', '"')
                    try:
                        yield next(csv.reader([inner], delimiter=","))
                    except csv.Error as exc:
                        raise BatchValidationError("MALFORMED_CSV", "Baris CSV VOC tidak dapat dibaca.") from exc
                return
            yield from csv.reader(handle, dialect=dialect)

    @staticmethod
    def _repair_wrapped_values(
        values: List[str], columns: List[str], column_map: Dict[str, Optional[str]], source_format: str
    ) -> List[str]:
        if len(values) == len(columns) or source_format != "wrapped_voc_export":
            return values
        review_column = column_map.get("review")
        if not review_column or len(values) < len(columns):
            return values
        review_index = columns.index(review_column)
        trailing_count = len(columns) - review_index - 1
        trailing_start = len(values) - trailing_count
        if trailing_start <= review_index:
            return values
        return values[:review_index] + [",".join(values[review_index:trailing_start])] + values[trailing_start:]

    @staticmethod
    def _column_map(columns: List[str]) -> Dict[str, Optional[str]]:
        normalized = {column.strip().casefold(): column for column in columns}
        return {
            role: next((normalized[alias.casefold()] for alias in aliases if alias.casefold() in normalized), None)
            for role, aliases in CSV_COLUMN_ALIASES.items()
        }

    def create_job(
        self, path: Path, filename: str, size_bytes: int, engine_version: str,
        batch_size: int, confidence_threshold: float, profile: str,
    ) -> BatchJob:
        self._prune_expired()
        if path.suffix.casefold() != ".csv":
            raise BatchValidationError("INVALID_FILE_TYPE", "File harus menggunakan ekstensi .csv.")
        if size_bytes > MAX_CSV_FILE_SIZE_MB * 1024 * 1024:
            raise BatchValidationError("FILE_TOO_LARGE", f"Ukuran file melebihi {MAX_CSV_FILE_SIZE_MB} MB.")
        if not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise BatchValidationError("INVALID_BATCH_SIZE", f"Batch size harus antara 1 dan {MAX_BATCH_SIZE}.")
        if engine_version not in self.engine_manager.supported_versions:
            raise BatchValidationError("MODEL_NOT_AVAILABLE", f"Model {engine_version} tidak tersedia.")

        encoding = self.detect_encoding(path)
        dialect, source_format = self._detect_dialect(path, encoding)
        preview: List[Dict[str, str]] = []
        row_count = 0
        try:
            reader = self._iter_csv_values(path, encoding, dialect, source_format)
            try:
                fieldnames = next(reader)
            except StopIteration:
                fieldnames = []
            if not fieldnames or any(name is None or not str(name).strip() for name in fieldnames):
                    raise BatchValidationError("HEADER_NOT_FOUND", "Header CSV tidak ditemukan atau tidak valid.")
            columns = [str(name).strip() for name in fieldnames]
            if len({name.casefold() for name in columns}) != len(columns):
                raise BatchValidationError("DUPLICATE_COLUMNS", "CSV memiliki nama kolom duplikat.")
            column_map = self._column_map(columns)
            if not column_map["review"]:
                raise BatchValidationError("REVIEW_COLUMN_NOT_FOUND", "Kolom review tidak ditemukan pada file CSV.")
            for values in reader:
                row_count += 1
                values = self._repair_wrapped_values(values, columns, column_map, source_format)
                if len(values) != len(columns):
                    raise BatchValidationError("MALFORMED_CSV", f"Struktur CSV rusak pada baris {row_count + 1}.")
                row = dict(zip(columns, values))
                if row_count > MAX_CSV_ROWS:
                    raise BatchValidationError("TOO_MANY_ROWS", f"Jumlah baris melebihi batas {MAX_CSV_ROWS}.")
                if row_count <= 10:
                    preview.append({str(key): str(value or "") for key, value in row.items()})
        except csv.Error as exc:
            raise BatchValidationError("MALFORMED_CSV", "Struktur CSV rusak atau tidak konsisten.") from exc
        if row_count == 0:
            raise BatchValidationError("EMPTY_CSV", "CSV tidak memiliki baris data.")

        job_id = f"batch_{uuid.uuid4().hex[:12]}"
        job = BatchJob(
            id=job_id, owner_token=secrets.token_urlsafe(24), filename=Path(filename).name,
            source_path=path, results_path=self.jobs_dir / f"{job_id}.jsonl", size_bytes=size_bytes,
            encoding=encoding, delimiter=dialect.delimiter, source_format=source_format,
            columns=columns, column_map=column_map,
            row_count=row_count, preview=preview, engine_version=engine_version, batch_size=batch_size,
            confidence_threshold=confidence_threshold, profile=profile,
        )
        with self._lock:
            self._jobs[job.id] = job
        logger.info("batch_created job_id=%s rows=%d model=%s batch_size=%d", job.id, row_count, engine_version, batch_size)
        return job

    def get(self, job_id: str, owner_token: Optional[str]) -> BatchJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            raise BatchValidationError("JOB_NOT_FOUND", "Batch job tidak ditemukan atau sudah kedaluwarsa.")
        if not owner_token or not secrets.compare_digest(owner_token, job.owner_token):
            raise BatchValidationError("JOB_ACCESS_DENIED", "Token akses batch tidak valid.")
        return job

    def upload_response(self, job: BatchJob) -> Dict[str, Any]:
        return {
            "success": True, "job_id": job.id, "access_token": job.owner_token,
            "file": {
                "name": job.filename, "size_bytes": job.size_bytes, "row_count": job.row_count,
                "columns": job.columns, "review_column": job.column_map["review"],
                "encoding": job.encoding, "delimiter": "tab" if job.delimiter == "\t" else job.delimiter,
                "source_format": job.source_format,
            },
            "preview": job.preview,
            "validation": {
                "status": "valid",
                "warnings": ["wrapped_voc_export_normalized"] if job.source_format == "wrapped_voc_export" else [],
            },
            "model_version": job.engine_version, "batch_size": job.batch_size,
        }

    def start(self, job: BatchJob) -> None:
        with job.lock:
            if job.status != "uploaded":
                raise BatchValidationError("INVALID_JOB_STATE", f"Job berstatus {job.status} dan tidak dapat dimulai.")
            job.status = "queued"
            job.queued_monotonic = perf_counter()
        thread = threading.Thread(target=self._run, args=(job,), name=job.id, daemon=True)
        thread.start()

    def cancel(self, job: BatchJob) -> None:
        with job.lock:
            if job.status in {"completed", "failed", "cancelled", "timeout"}:
                raise BatchValidationError("INVALID_JOB_STATE", f"Job berstatus {job.status} dan tidak dapat dibatalkan.")
            job.cancellation.set()
            if job.status in {"uploaded", "queued"}:
                job.status = "cancelled"
                job.completed_at = _now()
                self._remove_source(job)
        logger.info("batch_cancel_requested job_id=%s", job.id)

    def _iter_rows(self, job: BatchJob) -> Iterator[Tuple[int, Dict[str, str]]]:
        dialect = type("JobDialect", (csv.excel,), {"delimiter": job.delimiter})
        reader = self._iter_csv_values(job.source_path, job.encoding, dialect, job.source_format)
        fieldnames = next(reader)
        column_map = self._column_map(fieldnames)
        for row_number, values in enumerate(reader, start=2):
            values = self._repair_wrapped_values(values, fieldnames, column_map, job.source_format)
            if len(values) != len(fieldnames):
                raise BatchValidationError("MALFORMED_CSV", f"Struktur CSV rusak pada baris {row_number}.")
            yield row_number, {
                str(key): str(value or "") for key, value in zip(fieldnames, values)
            }

    def _run(self, job: BatchJob) -> None:
        started = perf_counter()
        job.queue_time_ms = max(0.0, (started - (job.queued_monotonic or started)) * 1000)
        latencies: List[float] = []
        preprocessing_total = inference_total = postprocessing_total = 0.0
        job.started_at = _now()
        job.status = "running"
        logger.info("batch_started job_id=%s", job.id)
        try:
            with job.results_path.open("w", encoding="utf-8") as output:
                row_iterator = self._iter_rows(job)
                while True:
                    chunk = list(islice(row_iterator, job.batch_size))
                    if not chunk:
                        break
                    elapsed = perf_counter() - started
                    if job.cancellation.is_set():
                        job.status = "cancelled"
                        break
                    if elapsed > BATCH_TIMEOUT_SECONDS:
                        job.status = "timeout"
                        break
                    for result in self._process_chunk(job, chunk):
                        output.write(json.dumps(result, ensure_ascii=False) + "\n")
                        output.flush()
                        job.processed_rows += 1
                        status = result["processing_status"]
                        if status == "success":
                            job.successful_rows += 1
                        elif status in {"skipped", "empty_review", "invalid_input"}:
                            job.skipped_rows += 1
                        else:
                            job.failed_rows += 1
                        latency = float(result.get("processing_time_ms", 0.0))
                        latencies.append(latency)
                        timing = result.get("timing", {})
                        preprocessing_total += float(timing.get("preprocessing_time_ms", 0.0))
                        inference_total += float(timing.get("inference_time_ms", 0.0))
                        postprocessing_total += float(timing.get("postprocessing_time_ms", 0.0))
                        job.elapsed_time_ms = (perf_counter() - started) * 1000
            if job.cancellation.is_set() and job.status == "running":
                job.status = "cancelled"
            elif job.status == "running":
                job.status = "completed"
        except Exception as exc:  # Never expose internals through the API.
            logger.exception("batch_system_failure job_id=%s", job.id)
            job.status = "failed"
            job.error = {"code": "BATCH_SYSTEM_ERROR", "message": "Proses batch mengalami kegagalan sistem."}
        finally:
            job.elapsed_time_ms = (perf_counter() - started) * 1000
            job.completed_at = _now()
            job.summary = self._summary(job, latencies, preprocessing_total, inference_total, postprocessing_total)
            self._remove_source(job)
            logger.info(
                "batch_finished job_id=%s status=%s success=%d failed=%d elapsed_ms=%.2f",
                job.id, job.status, job.successful_rows, job.failed_rows, job.elapsed_time_ms,
            )

    def _process_row(self, job: BatchJob, row_number: int, row: Dict[str, str]) -> Dict[str, Any]:
        review = row.get(job.column_map["review"] or "", "").strip()
        base = {"row_number": row_number, "original_fields": row, "raw_review": review, "model_version": job.engine_version}
        if not review:
            return {**base, "clean_review": "", "processing_status": "empty_review", "processing_time_ms": 0.0,
                    "aspects": [], "warnings": ["empty_review"]}
        if len(review) > MAX_REVIEW_LENGTH:
            return {**base, "clean_review": review, "processing_status": "invalid_input", "processing_time_ms": 0.0,
                    "aspects": [], "error_code": "REVIEW_TOO_LONG", "error_message": "Review melebihi batas panjang.",
                    "warnings": ["review_too_long"]}
        customer_id = row.get(job.column_map["customer_id"] or "") or None
        city = row.get(job.column_map["city"] or "") or None
        province = row.get(job.column_map["province"] or "") or None
        row_started = perf_counter()
        last_error: Optional[Exception] = None
        for _attempt in range(MAX_RETRY_PER_ROW + 1):
            try:
                enriched = analyze_enriched(
                    self.engine_manager, review, job.engine_version, job.confidence_threshold, job.profile,
                    customer_id=customer_id, city=city, province=province,
                )
                latency = (perf_counter() - row_started) * 1000
                return {
                    **base, "clean_review": enriched["clean_review"],
                    "customer_id": enriched["customer"]["customer_id"],
                    "customer_class": enriched["customer"]["customer_class"],
                    "city_or_regency": enriched["location"].get("city_or_regency"),
                    "province": enriched["location"].get("province"),
                    "location_status": enriched["location"]["status"],
                    "location_confidence": enriched["location"]["confidence"],
                    "location": enriched["location"], "aspects": enriched["absa"]["aspects"],
                    "processing_status": "success", "processing_time_ms": round(latency, 3),
                    "per_row_latency_ms": round(latency, 3), "timing": enriched["timing"],
                    "warnings": enriched["warnings"],
                }
            except Exception as exc:
                last_error = exc
        latency = (perf_counter() - row_started) * 1000
        logger.warning("batch_row_failed job_id=%s row=%d error=%s", job.id, row_number, type(last_error).__name__)
        return {
            **base, "clean_review": review, "processing_status": "model_error",
            "processing_time_ms": round(latency, 3), "aspects": [],
            "error_code": "MODEL_INFERENCE_ERROR", "error_message": "Gagal menjalankan prediksi pada baris ini.",
            "warnings": [],
        }

    def _process_chunk(
        self, job: BatchJob, chunk: List[Tuple[int, Dict[str, str]]]
    ) -> List[Dict[str, Any]]:
        completed: Dict[int, Dict[str, Any]] = {}
        valid_rows: List[Dict[str, Any]] = []
        valid_entries: List[Tuple[int, int, Dict[str, str], str, Dict[str, Any]]] = []
        for position, (row_number, row) in enumerate(chunk):
            review = row.get(job.column_map["review"] or "", "").strip()
            base = {"row_number": row_number, "original_fields": row, "raw_review": review, "model_version": job.engine_version}
            if not review:
                completed[position] = {
                    **base, "clean_review": "", "processing_status": "empty_review",
                    "processing_time_ms": 0.0, "aspects": [], "warnings": ["empty_review"],
                }
                continue
            if len(review) > MAX_REVIEW_LENGTH:
                completed[position] = {
                    **base, "clean_review": review, "processing_status": "invalid_input",
                    "processing_time_ms": 0.0, "aspects": [], "error_code": "REVIEW_TOO_LONG",
                    "error_message": "Review melebihi batas panjang.", "warnings": ["review_too_long"],
                }
                continue
            metadata = {
                "raw_text": review,
                "customer_id": row.get(job.column_map["customer_id"] or "") or None,
                "city": row.get(job.column_map["city"] or "") or None,
                "province": row.get(job.column_map["province"] or "") or None,
            }
            valid_entries.append((position, row_number, row, review, base))
            valid_rows.append(metadata)

        if valid_rows:
            chunk_started = perf_counter()
            try:
                enriched_rows = analyze_enriched_batch(
                    self.engine_manager, valid_rows, job.engine_version,
                    job.confidence_threshold, job.profile,
                )
                shared_latency = (perf_counter() - chunk_started) * 1000 / len(valid_rows)
                for entry, enriched in zip(valid_entries, enriched_rows):
                    position, _row_number, _row, _review, base = entry
                    completed[position] = {
                        **base, "clean_review": enriched["clean_review"],
                        "customer_id": enriched["customer"]["customer_id"],
                        "customer_class": enriched["customer"]["customer_class"],
                        "city_or_regency": enriched["location"].get("city_or_regency"),
                        "province": enriched["location"].get("province"),
                        "location_status": enriched["location"]["status"],
                        "location_confidence": enriched["location"]["confidence"],
                        "location": enriched["location"], "aspects": enriched["absa"]["aspects"],
                        "processing_status": "success", "processing_time_ms": round(shared_latency, 3),
                        "per_row_latency_ms": round(shared_latency, 3), "timing": enriched["timing"],
                        "warnings": enriched["warnings"],
                    }
            except Exception:
                # Preserve row-level isolation when one record or a model batch fails.
                for position, row_number, row, _review, _base in valid_entries:
                    completed[position] = self._process_row(job, row_number, row)
        return [completed[index] for index in range(len(chunk))]

    @staticmethod
    def _summary(job: BatchJob, latencies: List[float], preprocessing: float, inference: float, postprocessing: float) -> Dict[str, Any]:
        elapsed_seconds = job.elapsed_time_ms / 1000
        return {
            "total_rows": job.row_count, "processed_rows": job.processed_rows,
            "successful_rows": job.successful_rows, "failed_rows": job.failed_rows,
            "skipped_rows": job.skipped_rows, "total_processing_time_ms": round(job.elapsed_time_ms, 3),
            "average_latency_ms": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "median_latency_ms": round(statistics.median(latencies), 3) if latencies else 0.0,
            "p95_latency_ms": round(_percentile(latencies, 0.95), 3),
            "p99_latency_ms": round(_percentile(latencies, 0.99), 3),
            "min_latency_ms": round(min(latencies), 3) if latencies else 0.0,
            "max_latency_ms": round(max(latencies), 3) if latencies else 0.0,
            "throughput_rows_per_second": round(job.successful_rows / elapsed_seconds, 3) if elapsed_seconds else 0.0,
            "batch_size": job.batch_size, "model_version": job.engine_version,
            "preprocessing_time_ms": round(preprocessing, 3), "inference_time_ms": round(inference, 3),
            "batch_inference_latency_ms": round(inference, 3),
            "end_to_end_latency_ms": round(job.elapsed_time_ms, 3),
            "postprocessing_time_ms": round(postprocessing, 3), "cold_start_latency_ms": None,
            "warmup_executed": False, "queue_time_ms": round(job.queue_time_ms, 3),
        }

    @staticmethod
    def _remove_source(job: BatchJob) -> None:
        try:
            job.source_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("temporary_file_cleanup_failed job_id=%s", job.id)

    def status_response(self, job: BatchJob) -> Dict[str, Any]:
        with job.lock:
            elapsed = job.elapsed_time_ms
            throughput = job.processed_rows / (elapsed / 1000) if elapsed else 0.0
            remaining = job.row_count - job.processed_rows
            return {
                "success": True, "job_id": job.id, "status": job.status, "model_version": job.engine_version,
                "progress": {
                    "total_rows": job.row_count, "processed_rows": job.processed_rows,
                    "successful_rows": job.successful_rows, "failed_rows": job.failed_rows,
                    "skipped_rows": job.skipped_rows,
                    "percentage": round(job.processed_rows / job.row_count * 100, 2) if job.row_count else 0.0,
                },
                "performance": {
                    "elapsed_time_ms": round(elapsed, 3),
                    "queue_time_ms": round(job.queue_time_ms, 3),
                    "estimated_remaining_ms": round(remaining / throughput * 1000, 3) if throughput else None,
                    "throughput_rows_per_second": round(throughput, 3),
                },
                "summary": job.summary or None, "error": job.error,
            }

    def read_results(self, job: BatchJob, offset: int, limit: int) -> Tuple[List[Dict[str, Any]], int]:
        results: List[Dict[str, Any]] = []
        available = 0
        if job.results_path.exists():
            with job.results_path.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    available += 1
                    if index >= offset and len(results) < limit:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return results, available

    def iter_json_download(self, job: BatchJob) -> Iterable[str]:
        yield "["
        first = True
        if job.results_path.exists():
            with job.results_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    if not first:
                        yield ","
                    yield line.strip()
                    first = False
        yield "]"

    def iter_csv_download(self, job: BatchJob) -> Iterable[str]:
        columns = [
            "row_number", *[f"original_{column}" for column in job.columns], "raw_review", "clean_review",
            "customer_id", "customer_class", "city_or_regency", "province", "location_status",
            "location_confidence", "aspects", "model_version", "processing_status", "processing_time_ms",
            "warnings", "error_code", "error_message",
        ]
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        yield "\ufeff" + buffer.getvalue()
        if not job.results_path.exists():
            return
        with job.results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                flat = {key: item.get(key) for key in columns}
                for column in job.columns:
                    flat[f"original_{column}"] = item.get("original_fields", {}).get(column)
                flat = {key: _safe_csv_value(value) for key, value in flat.items()}
                buffer.seek(0)
                buffer.truncate(0)
                writer.writerow(flat)
                yield buffer.getvalue()
