import re
import json
import time
import gc
import os
import threading
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import torch
import psutil
from app.core.config import MODEL_BUNDLE_DIRS, PRETRAINED_BUNDLE_DIR, EVALUATION_MODE, PROXY_DISCLAIMER_MESSAGE
from app.training.pipeline import canonical_taxonomy_mapper
from app.inference.v14_engine import V14InferenceEngine
from app.inference.v15_engine import V15InferenceEngine

try:
    from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer
except ImportError:  # pragma: no cover - handled by the runtime health check
    AutoModelForSequenceClassification = AutoModelForTokenClassification = AutoTokenizer = None

class ABSAInferenceEngine:
    def __init__(self, model_bundle_dir: Optional[Path] = None, engine_version: str = "v11"):
        self.bundle_dir = model_bundle_dir or PRETRAINED_BUNDLE_DIR
        self.engine_version = engine_version
        self.config = {}
        self._load_bundle_config()
        self._tokenizer = None
        self._models: Dict[str, Any] = {}
        self._model_load_error: Optional[str] = None
        self._runtime_failed = False
        self._stream_models = False
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._relation_threshold = float(self.config.get("inference", {}).get("relation_threshold", 0.5))
        self._load_transformer_components()

    def _load_bundle_config(self):
        cfg_file = self.bundle_dir / "inference_config.json"
        if cfg_file.exists():
            with open(cfg_file, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "inference": {
                    "aspect_threshold": 0.75,
                    "opinion_threshold": 0.90,
                    "relation_threshold": 0.2
                }
            }

    def analyze_single_review(
        self,
        text: str,
        confidence_threshold: float = 0.5,
        profile: str = "production_precision"
    ) -> Dict[str, Any]:
        """
        Extract aspect, opinion, sentiment, relation, and taxonomy for a single review.
        Supports multi-aspect reviews.
        """
        start_time = time.time()
        text_str = str(text).strip()
        if not text_str:
            return self._response_metadata({
                "text": text,
                "results": [],
                "processing_time_ms": 0.0,
                "disclaimer": PROXY_DISCLAIMER_MESSAGE
            }, used_bundle=False)

        # Use the selected local bundle first. Operational candidate recovery is
        # enabled only by the maps_high_recall profile.
        results = []
        if self.model_bundle_loaded:
            try:
                results = self._analyze_with_bundle(text_str, confidence_threshold, profile)
            except Exception as exc:
                # Keep the API usable when local hardware cannot load a component.
                self._model_load_error = f"{type(exc).__name__}: {exc}"
                self._runtime_failed = True
                self._models.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if results:
            return self._response_metadata({
                "text": text_str,
                "results": results,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "evaluation_mode": EVALUATION_MODE,
                "disclaimer": PROXY_DISCLAIMER_MESSAGE,
            }, used_bundle=True)

        # Sentence segmentation & clause splitting
        clauses = re.split(r'[,.\n;]|\btetapi\b|\btapi\b|\bnamun\b|\bcuma\b|\bhanya\b', text_str, flags=re.IGNORECASE)

        aspect_keywords = [
            ("dokternya", "ramah", "positive", "Staff Attitude"),
            ("dokter", "baik", "positive", "Staff Attitude"),
            ("antreannya", "lama", "negative", "Waiting Time"),
            ("antrean", "panjang", "negative", "Waiting Time"),
            ("kopinya", "enak", "positive", "Beverage Quality"),
            ("kopi", "pahit", "negative", "Beverage Quality"),
            ("makanan", "lezat", "positive", "Food Quality"),
            ("makanan", "dingin", "negative", "Food Quality"),
            ("pelayanan", "cepat", "positive", "Service Speed"),
            ("pelayanan", "buruk", "negative", "Service Quality"),
            ("tempatnya", "bersih", "positive", "Cleanliness"),
            ("tempat", "nyaman", "positive", "Ambience"),
            ("harga", "murah", "positive", "Price & Value"),
            ("harga", "mahal", "negative", "Price & Value"),
            ("parkiran", "luas", "positive", "Parking"),
            ("fasilitas", "lengkap", "positive", "Facilities")
        ]

        matched_spans = set()

        # Check for explicit keywords or regex pattern matches in text
        for kw_asp, kw_opn, kw_sent, kw_tax in aspect_keywords:
            if kw_asp in text_str.lower():
                # Locate aspect span
                s_asp = text_str.lower().find(kw_asp)
                e_asp = s_asp + len(kw_asp)
                asp_actual = text_str[s_asp:e_asp]

                opn_actual = kw_opn
                if kw_opn in text_str.lower():
                    s_opn = text_str.lower().find(kw_opn)
                    opn_actual = text_str[s_opn:s_opn + len(kw_opn)]

                # Check sentiment context (negation flips positive)
                sent = kw_sent
                if any(neg in text_str.lower() for neg in ["tidak ", "gak ", "kurang ", "bukan "]):
                    if sent == "positive" and kw_asp in text_str.lower():
                        sent = "negative"

                # Deterministic confidence keeps inference reproducible and testable.
                conf = 0.93 if kw_opn in text_str.lower() else 0.84
                if conf >= confidence_threshold and asp_actual.lower() not in matched_spans:
                    matched_spans.add(asp_actual.lower())
                    results.append({
                        "aspect": asp_actual,
                        "opinion": opn_actual,
                        "sentiment": sent,
                        "taxonomy": kw_tax or canonical_taxonomy_mapper(asp_actual, text_str),
                        "relation": True,
                        "confidence": conf,
                        "aspect_span": [s_asp, e_asp]
                    })

        # Fallback clause parser if no keyword matched
        if not results:
            words = text_str.split()
            first_word = words[0] if words else "Ulasan"
            last_word = words[-1] if len(words) > 1 else "bagus"
            tax = canonical_taxonomy_mapper(first_word, text_str)

            results.append({
                "aspect": first_word,
                "opinion": last_word,
                "sentiment": "positive" if any(w in text_str.lower() for w in ["enak", "bagus", "ramah", "bersih", "suka"]) else "negative",
                "taxonomy": tax,
                "relation": True,
                "confidence": 0.85,
                "aspect_span": [0, len(first_word)]
            })

        processing_time = round((time.time() - start_time) * 1000, 2)

        return self._response_metadata({
            "text": text_str,
            "results": results,
            "processing_time_ms": processing_time,
            "evaluation_mode": EVALUATION_MODE,
            "disclaimer": PROXY_DISCLAIMER_MESSAGE
        }, used_bundle=False)

    def analyze_batch(
        self,
        reviews: List[str],
        confidence_threshold: float = 0.5,
        profile: str = "production_precision",
    ) -> List[Dict[str, Any]]:
        batch_out = []
        for r in reviews:
            batch_out.append(self.analyze_single_review(r, confidence_threshold, profile))
        return batch_out

    def _response_metadata(self, response: Dict[str, Any], used_bundle: bool) -> Dict[str, Any]:
        response.update({
            "engine_version": self.engine_version,
            "model_version": self.config.get("project_version", self.config.get("model_version", self.engine_version)),
            "model_source": self.bundle_dir.name,
            "model_bundle_loaded": self.model_bundle_loaded,
            "inference_mode": "transformer_bundle" if used_bundle else "rule_based_fallback",
            "model_loading_mode": "streaming" if self._stream_models else "resident",
        })
        if self._model_load_error:
            response["model_load_error"] = self._model_load_error
        return response

    def _load_transformer_components(self):
        """Load the selected V11 or V12 artifacts from the local bundle."""
        if not all((AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification)):
            self._model_load_error = "Dependency transformers tidak tersedia."
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.bundle_dir / "tokenizer"), local_files_only=True
            )
            model_bytes = sum(
                (self.bundle_dir / folder / "model.safetensors").stat().st_size
                for folder, _ in self._component_types().values()
            )
            if self._device.type == "cuda":
                available_bytes = torch.cuda.mem_get_info(self._device)[0]
            else:
                available_bytes = psutil.virtual_memory().available
            loading_preference = os.environ.get("ABSA_MODEL_LOADING", "auto").strip().lower()
            self._stream_models = loading_preference == "stream" or (
                loading_preference == "auto" and model_bytes > available_bytes * 0.65
            )
            if not self._stream_models:
                for name in self._component_types():
                    self._models[name] = self._load_single_model(name)
        except Exception as exc:
            # A malformed/incomplete local bundle must not prevent fallback inference.
            self._model_load_error = f"{type(exc).__name__}: {exc}"
            self._runtime_failed = True
            self._tokenizer = None
            self._models.clear()

    @property
    def model_bundle_loaded(self) -> bool:
        return not self._runtime_failed and self._tokenizer is not None and (self._stream_models or len(self._models) == 5)

    @staticmethod
    def _component_types() -> Dict[str, Tuple[str, Any]]:
        return {
            "aspect_model": ("aspect_tagger", AutoModelForTokenClassification),
            "opinion_model": ("opinion_tagger", AutoModelForTokenClassification),
            "sentiment_model": ("sentiment_classifier", AutoModelForSequenceClassification),
            "relation_model": ("relation_classifier", AutoModelForSequenceClassification),
            "taxonomy_model": ("taxonomy_classifier", AutoModelForSequenceClassification),
        }

    def _load_single_model(self, model_name: str):
        folder, model_class = self._component_types()[model_name]
        model = model_class.from_pretrained(
            str(self.bundle_dir / folder),
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        return model.to(self._device).eval()

    def _acquire_model(self, model_name: str):
        return self._load_single_model(model_name) if self._stream_models else self._models[model_name]

    @staticmethod
    def _release_streamed_model() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _predict_sequences(self, model_name: str, texts: List[str], max_length: int = 256) -> List[torch.Tensor]:
        if not texts:
            return []
        model = self._acquire_model(model_name)
        try:
            outputs: List[torch.Tensor] = []
            # Small batches keep V12 usable on CPU-only and low-memory machines.
            for start in range(0, len(texts), 8):
                encoded = self._tokenizer(
                    texts[start:start + 8],
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                with torch.inference_mode():
                    logits = model(**encoded).logits.detach().cpu()
                outputs.extend(logits[index] for index in range(len(logits)))
            return outputs
        finally:
            if self._stream_models:
                del model
                self._release_streamed_model()

    def _analyze_with_bundle(self, text: str, confidence_threshold: float, profile: str) -> List[Dict[str, Any]]:
        max_length = int(self.config.get("inference", {}).get("span_max_length", 192))
        stride = int(self.config.get("inference", {}).get("span_stride", 64))
        encoded = self._tokenizer(
            text, return_offsets_mapping=True, return_tensors="pt",
            return_overflowing_tokens=True, stride=stride,
            truncation=True, max_length=max_length, padding=False
        )
        offsets_by_window = encoded.pop("offset_mapping")
        model_inputs = {
            key: value.to(self._device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }

        def spans_from(model_name: str) -> List[Tuple[str, int, int, float]]:
            all_spans = []
            model = self._acquire_model(model_name)
            try:
                for window in range(len(offsets_by_window)):
                    window_inputs = {key: value[window:window + 1] for key, value in model_inputs.items()}
                    offsets = offsets_by_window[window].tolist()
                    with torch.inference_mode():
                        logits = model(**window_inputs).logits[0]
                    probs = torch.softmax(logits, dim=-1)
                    labels = probs.argmax(dim=-1).tolist()
                    current = None
                    for index, label_id in enumerate(labels):
                        start, end = offsets[index]
                        if end <= start:
                            continue
                        label = self.config.get("labels", {}).get("bio_id2label", {}).get(str(label_id), "O")
                        score = float(probs[index, label_id].item())
                        if label == "B" or (label == "I" and current is None):
                            if current:
                                all_spans.append(current)
                            current = [text[start:end], start, end, score]
                        elif label == "I" and current:
                            current[2] = end
                            current[0] = text[current[1]:end]
                            current[3] = min(current[3], score)
                        elif current:
                            all_spans.append(current)
                            current = None
                    if current:
                        all_spans.append(current)
            finally:
                if self._stream_models:
                    del model
                    self._release_streamed_model()
            repaired = []
            for term, start, end, score in all_spans:
                while start > 0 and (text[start - 1].isalnum() or text[start - 1] in "_-"):
                    start -= 1
                while end < len(text) and (text[end].isalnum() or text[end] in "_-"):
                    end += 1
                repaired.append([text[start:end], start, end, score])
            return repaired

        aspects = spans_from("aspect_model")
        opinions = spans_from("opinion_model")
        if profile == "maps_high_recall":
            # Recover explicit operational heads missed by BIO tagging.
            for match in re.finditer(r"\b(?:proses(?:nya)?|pelayanan(?:nya)?|layanan(?:nya)?|antrean(?:nya)?|antrian(?:nya)?|waktu\s+tunggu)\b", text, re.I):
                if not any(start <= match.start() < end or match.start() <= start < match.end() for _, start, end, _ in aspects):
                    nearby = text[match.end():min(len(text), match.end() + 60)].lower()
                    if re.search(r"\b(lama|lambat|cepat|nunggu|menunggu|terlambat)\b", nearby):
                        aspects.append([text[match.start():match.end()], match.start(), match.end(), 0.72])
        aspects.sort(key=lambda item: (item[1], item[2]))
        if not aspects:
            return []

        labels = self.config.get("labels", {})
        sentiment_labels = labels.get("sentiment_id2label", {})
        taxonomy_labels = labels.get("taxonomy_id2label", {})
        candidates = []
        for aspect, start, end, aspect_conf in aspects:
            def sentence_key(position: int) -> int:
                return len(re.findall(r"[.!?;]", text[:position]))
            same_sentence = [item for item in opinions if sentence_key(item[1]) == sentence_key(start) and abs(item[1] - end) <= 90]
            opinion = min(same_sentence, key=lambda item: abs(item[1] - end)) if same_sentence else None
            marked = text[:start] + " [ASP] " + text[start:end] + " [/ASP] "
            if opinion:
                os_, oe = opinion[1], opinion[2]
                if os_ >= end:
                    marked += text[end:os_] + " [OPN] " + text[os_:oe] + " [/OPN] " + text[oe:]
                else:
                    marked = text[:os_] + " [OPN] " + text[os_:oe] + " [/OPN] " + text[oe:start] + " [ASP] " + text[start:end] + " [/ASP] " + text[end:]
            candidates.append((aspect, start, end, aspect_conf, opinion, marked))

        sentiment_logits = self._predict_sequences("sentiment_model", [item[5] for item in candidates])
        taxonomy_logits = self._predict_sequences("taxonomy_model", [item[5] for item in candidates])
        relation_texts = []
        relation_candidate_indexes = []
        for index, (aspect, start, end, _, opinion, _) in enumerate(candidates):
            if opinion:
                relation_texts.append(
                    f"{text[:start]} [ASP] {text[start:end]} [/ASP] {text[end:opinion[1]]} "
                    f"[OPN] {text[opinion[1]:opinion[2]]} [/OPN] {text[opinion[2]:]}"
                )
                relation_candidate_indexes.append(index)
        relation_logits = self._predict_sequences("relation_model", relation_texts)
        relation_probabilities = {
            candidate_index: float(torch.softmax(logits, dim=-1)[1].item())
            for candidate_index, logits in zip(relation_candidate_indexes, relation_logits)
        }

        results = []
        for index, (aspect, start, end, aspect_conf, opinion, marked) in enumerate(candidates):
            # Condition sentiment on the aspect and its nearest related opinion;
            # whole-review classification causes errors such as "pelayanan agak lama"
            # being assigned the sentiment of another positive clause.
            sentiment_probs = torch.softmax(sentiment_logits[index] / float(self.config.get("inference", {}).get("sentiment_temperature", 1.0)), dim=-1)
            taxonomy_probs = torch.softmax(taxonomy_logits[index], dim=-1)
            relation_probability = relation_probabilities.get(index)
            sent_id = int(sentiment_probs.argmax().item())
            tax_id = int(taxonomy_probs.argmax().item())
            confidence = min(aspect_conf, float(sentiment_probs[sent_id]), float(taxonomy_probs[tax_id]))
            context = text[max(0, start - 80):min(len(text), end + 120)].lower()
            opinion_context = opinion[0].lower() if opinion else context
            if any(word in opinion_context for word in ("lama", "nunggu", "menunggu", "antrean", "antrian")):
                sent_id = 1
            confidence = min(0.99, confidence + 0.02)
            if confidence < confidence_threshold:
                continue
            results.append({
                "aspect": aspect.strip(),
                "opinion": opinion[0].strip() if opinion else "",
                "sentiment": sentiment_labels.get(str(sent_id), "neutral"),
                "taxonomy": self._taxonomy_from_context(aspect, opinion[0] if opinion else "", taxonomy_labels.get(str(tax_id), "Other")),
                "relation": bool(opinion and (relation_probability is None or relation_probability >= self._relation_threshold)),
                "relation_probability": relation_probability,
                "confidence": round(confidence, 4),
                "aspect_span": [start, end],
            })
        return results

    @staticmethod
    def _taxonomy_from_context(aspect: str, opinion: str, predicted: str) -> str:
        value = f"{aspect} {opinion}".lower()
        if re.search(r"\b(musholla|mushola|musala|masjid)\b", value):
            return "Prayer Room / Facilities"
        if re.search(r"\b(harga|biaya|tarif|price)\b", aspect.lower()):
            return "Price & Value"
        if re.search(r"\b(antrean|antrian|nunggu|menunggu|waktu tunggu)\b", value):
            return "Waiting Time"
        if re.search(r"\b(lama|lambat|cepat|terlambat|delay)\b", opinion.lower()) and re.search(r"\b(pelayanan|layanan|proses|service|servis|pesanan|order)\b", aspect.lower()):
            return "Service Speed"
        return predicted

class ABSAEngineManager:
    """Lazily load one engine at a time so model bundles do not exhaust RAM/VRAM."""

    def __init__(self, bundle_dirs: Optional[Dict[str, Path]] = None):
        self.bundle_dirs = bundle_dirs or MODEL_BUNDLE_DIRS
        self._active_version: Optional[str] = None
        self._active_engine: Optional[Any] = None
        self._lock = threading.RLock()

    @property
    def supported_versions(self) -> List[str]:
        return list(self.bundle_dirs.keys())

    def list_engines(self) -> List[Dict[str, Any]]:
        engines = []
        for version, bundle_dir in self.bundle_dirs.items():
            config_file = bundle_dir / "inference_config.json"
            config: Dict[str, Any] = {}
            if config_file.exists():
                try:
                    with open(config_file, "r", encoding="utf-8") as handle:
                        config = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    config = {}
            nested_config = config.get("config", {})
            if version == "v15":
                manifest_file = bundle_dir / "deployment_manifest.json"
                manifest: Dict[str, Any] = {}
                if manifest_file.exists():
                    try:
                        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        manifest = {}
                required = [bundle_dir / path / "model.safetensors" for path in manifest.get("required_models", [])]
                required.append(bundle_dir / "v14_runtime" / "context_hardened" / "seed_42" / "models" / "v14_context_hardened" / "tokenizer")
                components = {
                    "aspect": True, "opinion": True, "joint_relation_sentiment": True,
                    "taxonomy_domain": True, "taxonomy_entity": True, "taxonomy_issue": True,
                }
                available = manifest_file.exists() and len(manifest.get("required_models", [])) == 6 and all(path.exists() for path in required)
                model_version = manifest.get("bundle_version", "V15.1")
                languages = ["id", "en", "id-en"]
                status = "candidate_unpromoted"
            elif version == "v14":
                components = config.get("components", {})
                required = [bundle_dir / "tokenizer", bundle_dir / "absa_v14" / "inference.py"] + [
                    bundle_dir / name / "model.safetensors"
                    for name, present in components.items() if present
                ]
                available = config_file.exists() and all(path.exists() for path in required)
                model_version = nested_config.get("project_version", config.get("project_version", config.get("model_version", version)))
                languages = nested_config.get("mvp_languages", config.get("languages", ["id"]))
                status = None
            else:
                components = config.get("components")
                required = [
                    bundle_dir / "tokenizer",
                    bundle_dir / "aspect_tagger" / "model.safetensors",
                    bundle_dir / "opinion_tagger" / "model.safetensors",
                    bundle_dir / "sentiment_classifier" / "model.safetensors",
                    bundle_dir / "relation_classifier" / "model.safetensors",
                    bundle_dir / "taxonomy_classifier" / "model.safetensors",
                ]
                available = config_file.exists() and all(path.exists() for path in required)
                model_version = nested_config.get("project_version", config.get("project_version", config.get("model_version", version)))
                languages = nested_config.get("mvp_languages", config.get("languages", ["id"]))
                status = None
            engines.append({
                "id": version,
                "label": "V14 ABSA" if version == "v14" and len(self.bundle_dirs) == 1 else f"Engine {version.upper()}",
                "available": available,
                "active": version == self._active_version,
                "bundle": bundle_dir.name,
                "model_version": model_version,
                "languages": languages,
                "components": components,
                "status": status,
            })
        return engines

    def _get_engine(self, engine_version: str) -> ABSAInferenceEngine:
        version = str(engine_version).lower()
        if version not in self.bundle_dirs:
            supported = ", ".join(self.supported_versions)
            raise ValueError(f"Engine '{engine_version}' tidak tersedia. Pilihan: {supported}.")
        if self._active_engine is None or self._active_version != version:
            if self._active_engine is not None and hasattr(self._active_engine, "close"):
                self._active_engine.close()
            self._active_engine = None
            self._active_version = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            engine_class = V15InferenceEngine if version == "v15" else (V14InferenceEngine if version == "v14" else ABSAInferenceEngine)
            self._active_engine = engine_class(self.bundle_dirs[version], engine_version=version)
            self._active_version = version
        return self._active_engine

    def analyze_single_review(
        self,
        engine_version: str,
        text: str,
        confidence_threshold: float = 0.5,
        profile: str = "production_precision",
    ) -> Dict[str, Any]:
        with self._lock:
            return self._get_engine(engine_version).analyze_single_review(text, confidence_threshold, profile)

    def analyze_batch(
        self,
        engine_version: str,
        reviews: List[str],
        confidence_threshold: float = 0.5,
        profile: str = "production_precision",
    ) -> List[Dict[str, Any]]:
        with self._lock:
            return self._get_engine(engine_version).analyze_batch(reviews, confidence_threshold, profile)


_engine_only = os.environ.get("ABSA_ENGINE_ONLY", "").strip().lower()
if _engine_only:
    if _engine_only not in MODEL_BUNDLE_DIRS:
        supported = ", ".join(MODEL_BUNDLE_DIRS)
        raise RuntimeError(f"ABSA_ENGINE_ONLY='{_engine_only}' tidak valid. Pilihan: {supported}.")
    inference_engine_manager = ABSAEngineManager({_engine_only: MODEL_BUNDLE_DIRS[_engine_only]})
else:
    inference_engine_manager = ABSAEngineManager()
