import os
import re
import io
import json
import zipfile
import hashlib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from app.core.config import UPLOADS_DIR, EXTRACTED_DIR, MAX_UPLOAD_SIZE_MB, ALLOWED_EXTENSIONS

def sanitize_filename(filename: str) -> str:
    # Strip both POSIX and Windows path components before sanitizing.
    basename = re.split(r'[\\/]', str(filename))[-1]
    cleaned = re.sub(r'[^a-zA-Z0-9_\.\-]', '_', basename)
    return cleaned.lstrip('.') or "uploaded_dataset"

def compute_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def validate_zip_slip(zip_ref: zipfile.ZipFile, extract_to: Path) -> None:
    extract_to_abs = extract_to.resolve()
    for member in zip_ref.infolist():
        member_path = (extract_to / member.filename).resolve()
        try:
            member_path.relative_to(extract_to_abs)
        except ValueError:
            raise ValueError(f"ZIP Slip security hazard detected in file: {member.filename}")

def save_uploaded_file(file_bytes: bytes, original_filename: str) -> Tuple[Path, str]:
    if len(file_bytes) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise ValueError(f"File size exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB}MB")

    sanitized_name = sanitize_filename(original_filename)
    ext = Path(sanitized_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext}. Allowed: {ALLOWED_EXTENSIONS}")

    checksum = compute_sha256(file_bytes)
    target_path = UPLOADS_DIR / f"{checksum}_{sanitized_name}"

    with open(target_path, "wb") as f:
        f.write(file_bytes)

    # Extract if ZIP
    if ext == ".zip":
        extract_dir = EXTRACTED_DIR / checksum
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as zf:
            validate_zip_slip(zf, extract_dir)
            zf.extractall(extract_dir)

    return target_path, checksum

def detect_dataset_type(file_path: Path, checksum: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Auto-detect dataset type by content structure, not filename.
    Returns: (detected_type, records_list, audit_metadata)
    Types: v10_master, v11_synthetic, gold_pilot, silver, dapt_corpus, auxiliary
    """
    ext = file_path.suffix.lower()
    extracted_dir = EXTRACTED_DIR / checksum if ext == ".zip" else None

    files_to_check = []
    if extracted_dir and extracted_dir.exists():
        for root, _, files in os.walk(extracted_dir):
            for file in files:
                files_to_check.append(Path(root) / file)
    else:
        files_to_check.append(file_path)

    # Content inspection
    detected_type = "auxiliary"
    all_records = []
    file_info = []

    has_human_approved = False
    has_v11_synthetic = False
    has_v10_master = False
    has_aspect_cols = False

    for fp in files_to_check:
        fname = fp.name.lower()
        if fp.suffix.lower() in [".csv", ".tsv", ".xlsx", ".json"]:
            try:
                df = None
                if fp.suffix.lower() == ".csv":
                    df = pd.read_csv(fp)
                elif fp.suffix.lower() == ".tsv":
                    df = pd.read_csv(fp, sep="\t")
                elif fp.suffix.lower() == ".xlsx":
                    df = pd.read_excel(fp)
                elif fp.suffix.lower() == ".json":
                    with open(fp, "r", encoding="utf-8") as jf:
                        jdata = json.load(jf)
                        if isinstance(jdata, list):
                            df = pd.DataFrame(jdata)
                        elif isinstance(jdata, dict) and "reviews" in jdata:
                            df = pd.DataFrame(jdata["reviews"])

                if df is not None:
                    cols = [c.lower() for c in df.columns]
                    if "human_status" in cols or "human_approved" in cols:
                        has_human_approved = True
                    if any("synthetic" in c for c in cols) or "template_id" in cols or "v11" in fname:
                        has_v11_synthetic = True
                    if "aspect_term" in cols or "aspect" in cols or "taxonomy" in cols:
                        has_aspect_cols = True
                    if "gmaps_reviews" in fname or "v10" in fname or "all_datasets" in fname:
                        has_v10_master = True

                    file_info.append({
                        "filename": fp.name,
                        "rows": len(df),
                        "columns": list(df.columns)
                    })

                    # Parse records into standard dict structure if text column present
                    text_col = next((c for c in df.columns if c.lower() in ["text", "review_text", "review", "content"]), None)
                    if text_col:
                        for _, row in df.iterrows():
                            rec = row.to_dict()
                            rec["text"] = str(rec[text_col])
                            all_records.append(rec)
            except Exception:
                continue

    if has_human_approved:
        detected_type = "gold_pilot"
    elif has_v11_synthetic:
        detected_type = "v11_synthetic"
    elif has_v10_master or (has_aspect_cols and len(all_records) > 500):
        detected_type = "v10_master"
    elif has_aspect_cols:
        detected_type = "silver"
    elif len(all_records) > 0:
        detected_type = "dapt_corpus"

    return detected_type, all_records, {"file_info": file_info}
