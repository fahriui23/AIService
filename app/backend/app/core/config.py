import os
import sys
import shutil
import psutil
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = Path(os.environ.get("ABSA_ROOT_DIR", BASE_DIR))

STORAGE_DIR = ROOT_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
EXTRACTED_DIR = STORAGE_DIR / "extracted"
RUNS_DIR = STORAGE_DIR / "runs"
CHECKPOINTS_DIR = STORAGE_DIR / "checkpoints"
MODELS_DIR = STORAGE_DIR / "models"
REPORTS_DIR = STORAGE_DIR / "reports"
BATCH_JOBS_DIR = STORAGE_DIR / "batch_jobs"
MODEL_BUNDLE_DIRS = {
    "v11": ROOT_DIR / "absa_v11_web_bundle",
    "v12": ROOT_DIR / "absa_v12_multilingual_bundle",
    "v14": ROOT_DIR / "absa_v14_context_hardened_seed_42",
    "v15": ROOT_DIR / "absa_v15_1_production_fp16",
}
# Kept for callers that instantiate ABSAInferenceEngine directly.
PRETRAINED_BUNDLE_DIR = MODEL_BUNDLE_DIRS["v11"]

DB_PATH = STORAGE_DIR / "absa_v11_metadata.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Security & Constraints
MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 500))
ALLOWED_EXTENSIONS = {".zip", ".csv", ".xlsx", ".json", ".ipynb"}

# Metadata enrichment and CSV batch processing. All limits are configurable
# without changing application code.
CUSTOMER_CLASS_MAP = {"1": "VIP", "2": "SILVER", "3": "REGULER"}
LOCATION_CONFIDENCE_THRESHOLD = float(os.environ.get("LOCATION_CONFIDENCE_THRESHOLD", "0.80"))
MAX_REVIEW_LENGTH = int(os.environ.get("MAX_REVIEW_LENGTH", "10000"))
MAX_CSV_FILE_SIZE_MB = int(os.environ.get("MAX_CSV_FILE_SIZE_MB", "100"))
MAX_CSV_ROWS = int(os.environ.get("MAX_CSV_ROWS", "100000"))
DEFAULT_BATCH_SIZE = int(os.environ.get("DEFAULT_BATCH_SIZE", "32"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "256"))
BATCH_TIMEOUT_SECONDS = int(os.environ.get("BATCH_TIMEOUT_SECONDS", "1800"))
MAX_RETRY_PER_ROW = int(os.environ.get("MAX_RETRY_PER_ROW", "1"))
BATCH_RESULT_TTL_SECONDS = int(os.environ.get("BATCH_RESULT_TTL_SECONDS", "86400"))
CSV_COLUMN_ALIASES = {
    "review": ["review", "review_text", "text", "comment", "komentar", "ulasan", "isi review"],
    "customer_id": ["customer_id", "customer", "id"],
    "city": ["city", "kota", "regency", "kabupaten"],
    "province": ["province", "provinsi"],
}

EVALUATION_MODE = "proxy_without_human_gold"
PROXY_DISCLAIMER_MESSAGE = (
    "Evaluation menggunakan proxy labels dan belum divalidasi menggunakan Human GOLD. "
    "Metrik digunakan untuk perbandingan eksperimen internal dan bukan klaim performa produksi."
)

DEFAULT_CONTROLLED_CLASS_EXPOSURE = {
    "positive": 47.5,
    "negative": 32.5,
    "neutral": 20.0
}

def ensure_directories():
    for d in [STORAGE_DIR, UPLOADS_DIR, EXTRACTED_DIR, RUNS_DIR, CHECKPOINTS_DIR, MODELS_DIR, REPORTS_DIR, BATCH_JOBS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

ensure_directories()

def get_hardware_status() -> Dict[str, Any]:
    # Check PyTorch CUDA availability
    gpu_available = False
    gpu_name = "N/A (CPU Mode)"
    vram_total_mb = 0.0
    vram_used_mb = 0.0
    vram_free_mb = 0.0

    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            mem_total = torch.cuda.get_device_properties(0).total_memory
            mem_reserved = torch.cuda.memory_reserved(0)
            mem_allocated = torch.cuda.memory_allocated(0)
            vram_total_mb = round(mem_total / (1024 * 1024), 2)
            vram_used_mb = round(mem_reserved / (1024 * 1024), 2)
            vram_free_mb = round((mem_total - mem_reserved) / (1024 * 1024), 2)
    except Exception:
        pass

    # CPU & RAM
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count(logical=True)
    ram = psutil.virtual_memory()
    ram_total_gb = round(ram.total / (1024 ** 3), 2)
    ram_used_gb = round(ram.used / (1024 ** 3), 2)
    ram_percent = ram.percent

    # Disk
    disk = psutil.disk_usage(str(STORAGE_DIR))
    disk_total_gb = round(disk.total / (1024 ** 3), 2)
    disk_used_gb = round(disk.used / (1024 ** 3), 2)
    disk_free_gb = round(disk.free / (1024 ** 3), 2)
    disk_percent = disk.percent

    return {
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "vram_total_mb": vram_total_mb,
        "vram_used_mb": vram_used_mb,
        "vram_free_mb": vram_free_mb,
        "cpu_count": cpu_count,
        "cpu_usage_percent": cpu_percent,
        "ram_total_gb": ram_total_gb,
        "ram_used_gb": ram_used_gb,
        "ram_usage_percent": ram_percent,
        "disk_total_gb": disk_total_gb,
        "disk_used_gb": disk_used_gb,
        "disk_free_gb": disk_free_gb,
        "disk_usage_percent": disk_percent,
        "evaluation_mode": EVALUATION_MODE,
        "proxy_disclaimer": PROXY_DISCLAIMER_MESSAGE,
    }
