"""Inference adapter for the exported ABSA V15.1 FP16 production candidate.

V15 reuses the V14 span taggers and replaces the old relation/sentiment and
flat taxonomy heads with a joint four-class head plus hierarchical taxonomy.
The exported bundle intentionally contains weights only, so this adapter keeps
the web API contract stable while reconstructing the small custom joint module.
"""
from __future__ import annotations

import gc
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch
from torch import nn

from app.core.config import EVALUATION_MODE, PROXY_DISCLAIMER_MESSAGE
from app.inference.v14_engine import V14InferenceEngine

try:
    from safetensors.torch import load_file
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
    )
except ImportError:  # pragma: no cover - surfaced through model_load_error
    load_file = AutoConfig = AutoModel = None
    AutoModelForSequenceClassification = AutoModelForTokenClassification = AutoTokenizer = None


JOINT_LABELS = ["no_relation", "related_positive", "related_negative", "related_neutral"]
SENTIMENT_LABELS = ["positive", "negative", "neutral"]
NEGATION_RE = re.compile(r"(?i)(?<!\w)(?:tidak|tak|bukan|belum|kurang|ga|gak|nggak|ngga|not|never|hardly)(?!\w)")
CONTRAST_RE = re.compile(r"(?i)(?<!\w)(?:tapi|tetapi|namun|melainkan|sedangkan|but|however|although|whereas|yet)(?!\w)")


class JointRelationSentimentModel(nn.Module):
    """Exact inference architecture used to save the V15 joint checkpoint."""

    def __init__(self, encoder: nn.Module, marker_ids: List[int], feature_size: int = 8, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config
        self.marker_ids = [int(value) for value in marker_ids]
        self.feature_size = int(feature_size)
        self.dropout_rate = float(dropout)
        hidden = encoder.config.hidden_size
        width = hidden * 5 + 2 + self.feature_size
        self.project = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout))
        self.joint_classifier = nn.Linear(hidden, len(JOINT_LABELS))
        self.sentiment_classifier = nn.Linear(hidden, len(SENTIMENT_LABELS))

    def _pool(self, hidden: torch.Tensor, ids: torch.Tensor, open_id: int, close_id: int):
        open_mask, close_mask = ids.eq(open_id), ids.eq(close_id)
        if not torch.all(open_mask.sum(1) == 1) or not torch.all(close_mask.sum(1) == 1):
            raise ValueError("Marker pasangan V15 hilang atau duplikat.")
        lower, upper = open_mask.long().argmax(1), close_mask.long().argmax(1)
        if not torch.all(upper > lower + 1):
            raise ValueError("Span bertanda V15 kosong atau terbalik.")
        positions = torch.arange(ids.shape[1], device=ids.device)[None, :]
        mask = (positions > lower[:, None]) & (positions < upper[:, None])
        pooled = (hidden * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return pooled, lower

    def forward(self, input_ids, attention_mask=None, pair_features=None, **kwargs):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state
        aspect, aspect_left = self._pool(hidden, input_ids, *self.marker_ids[:2])
        opinion, opinion_left = self._pool(hidden, input_ids, *self.marker_ids[2:])
        denominator = attention_mask.sum(1).clamp_min(1) if attention_mask is not None else input_ids.new_full(
            (len(input_ids),), input_ids.shape[1]
        )
        distance = torch.stack(
            ((opinion_left - aspect_left) / denominator, torch.abs(opinion_left - aspect_left) / denominator), -1
        ).to(hidden.dtype)
        if pair_features is None:
            pair_features = hidden.new_zeros((hidden.shape[0], self.feature_size))
        representation = torch.cat(
            (hidden[:, 0], aspect, opinion, aspect * opinion, torch.abs(aspect - opinion), distance,
             pair_features.to(hidden.dtype)), -1
        )
        shared = self.project(representation)
        joint_logits = self.joint_classifier(shared)
        return SimpleNamespace(
            logits=joint_logits,
            joint_logits=joint_logits,
            sentiment_logits=self.sentiment_classifier(shared),
        )

    @classmethod
    def from_pretrained(cls, path: Path, device: torch.device):
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if config.get("joint_labels") != JOINT_LABELS:
            raise ValueError("Peta label joint V15 tidak cocok dengan runtime.")
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(path / "encoder_config", local_files_only=True))
        model = cls(encoder, config["marker_ids"], config.get("feature_size", 8), config.get("dropout", 0.1))
        model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
        return model.to(device).eval()


def _sentence_index(text: str, position: int) -> int:
    return len(re.findall(r"[.!?]+(?:\s+|$)", text[:position]))


def _clause_index(text: str, position: int) -> int:
    return len(re.findall(r"[,;:]|(?i:(?<!\w)(?:tapi|tetapi|namun|but|however|yet)(?!\w))", text[:position]))


def pair_features(text: str, aspect: Dict[str, Any], opinion: Dict[str, Any]) -> List[float]:
    left = min(int(aspect["start"]), int(opinion["start"]))
    right = max(int(aspect["end"]), int(opinion["end"]))
    hull = text[left:right]
    lower = hull.casefold()
    has_id = bool(re.search(r"\b(?:tidak|banget|bagus|buruk|pelayanan|harga|yang|dan|tapi)\b", lower))
    has_en = bool(re.search(r"\b(?:not|very|good|bad|service|price|the|and|but)\b", lower))
    distance = abs(int(aspect["start"]) - int(opinion["start"])) / max(1, len(text))
    return [
        min(1.0, distance),
        float(_sentence_index(text, int(aspect["start"])) == _sentence_index(text, int(opinion["start"]))),
        float(_clause_index(text, int(aspect["start"])) == _clause_index(text, int(opinion["start"]))),
        float(bool(NEGATION_RE.search(hull))),
        float(bool(CONTRAST_RE.search(hull))),
        0.0,
        float(has_id),
        float(has_en),
    ]


class V15InferenceEngine(V14InferenceEngine):
    """Memory-aware end-to-end V15.1 adapter with V11/V12/V14 API compatibility."""

    component_types = {
        "aspect": AutoModelForTokenClassification,
        "opinion": AutoModelForTokenClassification,
        "joint_relation_sentiment": JointRelationSentimentModel,
        "taxonomy_domain": AutoModelForSequenceClassification,
        "taxonomy_entity": AutoModelForSequenceClassification,
        "taxonomy_issue": AutoModelForSequenceClassification,
    }

    def __init__(self, bundle_dir: Path, engine_version: str = "v15"):
        self.bundle_dir = Path(bundle_dir)
        self.engine_version = engine_version
        self.v14_root = self.bundle_dir / "v14_runtime" / "context_hardened" / "seed_42"
        self.span_root = self.v14_root / "models" / "v14_context_hardened"
        self.head_root = (
            self.bundle_dir / "v15_runtime" / "context_hardened" / "models" /
            "v15_required_heads" / "seed_42"
        )
        self.config: Dict[str, Any] = {}
        self.runtime_config: Dict[str, Any] = {}
        self.labels: Dict[str, Dict[str, str]] = {}
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self._models: Dict[str, Any] = {}
        self._tokenizer = None
        self._model_load_error: Optional[str] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_config()
        self._load_helpers()
        self._load_runtime()

    def _load_config(self) -> None:
        manifest = json.loads((self.bundle_dir / "deployment_manifest.json").read_text(encoding="utf-8"))
        report_path = self.bundle_dir / "v15_runtime" / "context_hardened" / "reports" / "v15_one_seed_pilot_reports.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        seed_report = report.get("42", {})
        v14_config_path = self.bundle_dir.parent / "absa_v14_context_hardened_seed_42" / "inference_config.json"
        v14_config = json.loads(v14_config_path.read_text(encoding="utf-8")) if v14_config_path.exists() else {}
        self.config = manifest
        self.runtime_config = {
            **v14_config.get("config", {}),
            "project_version": manifest.get("bundle_version", "V15.1"),
            "model_version": manifest.get("bundle_version", "V15.1"),
            "taxonomy_version": "v15.1",
            "joint_max_length": 448,
            "taxonomy_max_length": 384,
            "joint_relation_sentiment_usable": True,
            "taxonomy_compatibility_mask": seed_report.get("taxonomy_compatibility_mask", {}),
            "taxonomy_output_policy": os.environ.get("ABSA_V15_TAXONOMY_POLICY", "best_effort").strip().lower(),
            "diagnostic_only_taxonomy": (
                seed_report.get("taxonomy_support", {}).get("diagnostic_only_entities", []) +
                seed_report.get("taxonomy_support", {}).get("diagnostic_only_issues", [])
            ),
        }
        self.labels = {
            "bio": v14_config.get("labels", {}).get("bio", {"0": "O", "1": "B", "2": "I"}),
            "joint": {str(index): label for index, label in enumerate(JOINT_LABELS)},
        }
        for name in ("taxonomy_domain", "taxonomy_entity", "taxonomy_issue"):
            config = json.loads((self.head_root / name / "config.json").read_text(encoding="utf-8"))
            self.labels[name] = {str(key): value for key, value in config.get("id2label", {}).items()}
        self.profiles = v14_config.get("profiles") or {
            "scientific_balanced": {"aspect_threshold": 0.7, "opinion_threshold": 0.55, "recovery": False, "dual_view": False},
            "production_precision": {"aspect_threshold": 0.9, "opinion_threshold": 0.9, "recovery": False, "dual_view": False},
            "maps_high_recall": {"aspect_threshold": 0.1, "opinion_threshold": 0.55, "recovery": True, "dual_view": True},
        }

    def _load_helpers(self) -> None:
        helper_bundle = self.bundle_dir.parent / "absa_v14_context_hardened_seed_42"
        original = self.bundle_dir
        try:
            self.bundle_dir = helper_bundle
            super()._load_helpers()
        finally:
            self.bundle_dir = original

    def _component_path(self, name: str) -> Path:
        return self.span_root / f"{name}_tagger" if name in {"aspect", "opinion"} else self.head_root / name

    def _required_components(self) -> List[str]:
        return list(self.component_types)

    def _load_runtime(self) -> None:
        if not all((AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoModel, load_file)):
            self._model_load_error = "Dependency transformers/safetensors tidak tersedia."
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.span_root / "tokenizer"), local_files_only=True, use_fast=True)
            preference = os.environ.get("ABSA_MODEL_LOADING", "auto").strip().lower()
            # Six XLM-R checkpoints are large; stream unless resident mode is explicitly requested.
            self._stream_models = preference != "resident"
            if not self._stream_models:
                for name in self._required_components():
                    self._models[name] = self._load_model(name)
        except Exception as exc:
            self._model_load_error = f"{type(exc).__name__}: {exc}"
            self.close()

    def _load_model(self, name: str):
        path = self._component_path(name)
        if name == "joint_relation_sentiment":
            return JointRelationSentimentModel.from_pretrained(path, self._device)
        return self.component_types[name].from_pretrained(
            str(path), local_files_only=True, low_cpu_mem_usage=True
        ).to(self._device).eval()

    def _label(self, task: str, index: int) -> str:
        return self.labels[task].get(str(index), str(index))

    def _encode_taxonomy(self, text: str, aspect: Dict[str, Any], opinion: Dict[str, Any], domain: str, entity: Optional[str] = None):
        marked = self.mark_pair(text, aspect["start"], aspect["end"], opinion["start"], opinion["end"])
        condition = f"domain={domain}" + (f" entity={entity}" if entity else "")
        return self.encode_marked(
            self._tokenizer, marked, int(self.runtime_config.get("taxonomy_max_length", 384)), target=condition
        )

    def analyze_batch(self, reviews: List[str], confidence_threshold: float = 0.5,
                      profile: str = "production_precision") -> List[Dict[str, Any]]:
        if not self.model_bundle_loaded:
            raise RuntimeError(f"Bundle V15.1 tidak dapat dimuat: {self._model_load_error or 'runtime tidak tersedia'}")
        if profile not in self.profiles:
            raise ValueError(f"Profile '{profile}' tidak tersedia untuk V15.1.")
        texts = [str(review or "").strip() for review in reviews]
        for text in texts:
            if len(text) > int(self.runtime_config.get("max_input_chars", 30000)):
                raise ValueError("Review melebihi panjang maksimum V15.1.")
            if any(marker in text for marker in ("[ASP]", "[/ASP]", "[OPN]", "[/OPN]")):
                raise ValueError("Review berisi marker internal ABSA yang tidak diizinkan.")

        started = time.perf_counter()
        profile_config = self.profiles[profile]
        aspects_by_review = self._extract_batch(texts, "aspect", profile_config)
        opinions_by_review = self._extract_batch(texts, "opinion", profile_config)
        if profile_config.get("recovery"):
            for index, text in enumerate(texts):
                aspects_by_review[index], opinions_by_review[index] = self._recover(
                    text, aspects_by_review[index], opinions_by_review[index]
                )

        candidates: List[Dict[str, Any]] = []
        diagnostics: List[Dict[str, Any]] = []
        for review_index, (text, aspects, opinions) in enumerate(zip(texts, aspects_by_review, opinions_by_review)):
            proposals = sorted(
                ((aspect, opinion) for aspect in aspects for opinion in opinions),
                key=lambda pair: abs(pair[0]["start"] - pair[1]["start"]),
            )[: int(self.runtime_config.get("max_candidate_pairs", 512))]
            diagnostics.append({"candidate_pairs": len(proposals), "scored_pairs": 0, "rejected_pair_count": 0})
            for aspect, opinion in proposals:
                candidates.append({"review_index": review_index, "aspect": aspect, "opinion": opinion})

        joint_model = self._acquire_model("joint_relation_sentiment")
        try:
            for candidate in candidates:
                text = texts[candidate["review_index"]]
                aspect, opinion = candidate["aspect"], candidate["opinion"]
                try:
                    marked = self.mark_pair(text, aspect["start"], aspect["end"], opinion["start"], opinion["end"])
                    encoded = self.encode_marked(
                        self._tokenizer, marked, int(self.runtime_config.get("joint_max_length", 448))
                    )
                    inputs = {key: torch.tensor([value], device=self._device) for key, value in encoded.items()}
                    inputs["pair_features"] = torch.tensor(
                        [pair_features(text, aspect, opinion)], dtype=torch.float32, device=self._device
                    )
                    with torch.inference_mode():
                        probabilities = joint_model(**inputs).joint_logits[0].softmax(-1)
                    label_index = int(probabilities.argmax())
                    label = JOINT_LABELS[label_index]
                    candidate.update({
                        "joint_label": label,
                        "sentiment": label.removeprefix("related_") if label != "no_relation" else None,
                        "joint_confidence": float(probabilities[label_index]),
                        "relation_confidence": float(1.0 - probabilities[0]),
                        "joint_probabilities": {name: float(probabilities[i]) for i, name in enumerate(JOINT_LABELS)},
                    })
                    diagnostics[candidate["review_index"]]["scored_pairs"] += 1
                except (ValueError, self.PairTooWide):
                    candidate["joint_label"] = "no_relation"
                    diagnostics[candidate["review_index"]]["rejected_pair_count"] += 1
        finally:
            self._release_model(joint_model)

        related = [candidate for candidate in candidates if candidate.get("joint_label") != "no_relation"]
        domain_by_review: Dict[int, Tuple[str, float]] = {}
        domain_model = self._acquire_model("taxonomy_domain")
        try:
            for index, text in enumerate(texts):
                encoded = self._tokenizer(
                    text, truncation=True, max_length=int(self.runtime_config.get("taxonomy_max_length", 384)),
                    return_tensors="pt"
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                with torch.inference_mode():
                    probabilities = domain_model(**encoded).logits[0].softmax(-1)
                label_index = int(probabilities.argmax())
                domain_by_review[index] = (self._label("taxonomy_domain", label_index), float(probabilities[label_index]))
        finally:
            self._release_model(domain_model)

        entity_model = self._acquire_model("taxonomy_entity")
        try:
            for candidate in related:
                domain, _ = domain_by_review[candidate["review_index"]]
                encoded = self._encode_taxonomy(
                    texts[candidate["review_index"]], candidate["aspect"], candidate["opinion"], domain
                )
                inputs = {key: torch.tensor([value], device=self._device) for key, value in encoded.items()}
                with torch.inference_mode():
                    probabilities = entity_model(**inputs).logits[0].softmax(-1)
                label_index = int(probabilities.argmax())
                candidate["entity_id"] = self._label("taxonomy_entity", label_index)
                candidate["entity_confidence"] = float(probabilities[label_index])
        finally:
            self._release_model(entity_model)

        issue_model = self._acquire_model("taxonomy_issue")
        try:
            for candidate in related:
                review_index = candidate["review_index"]
                domain, _ = domain_by_review[review_index]
                entity = candidate["entity_id"]
                encoded = self._encode_taxonomy(texts[review_index], candidate["aspect"], candidate["opinion"], domain, entity)
                inputs = {key: torch.tensor([value], device=self._device) for key, value in encoded.items()}
                with torch.inference_mode():
                    logits = issue_model(**inputs).logits[0]
                allowed = set(self.runtime_config.get("taxonomy_compatibility_mask", {}).get(domain, {}).get(entity, []))
                if allowed:
                    blocked = torch.tensor(
                        [self._label("taxonomy_issue", i) not in allowed for i in range(len(logits))], device=logits.device
                    )
                    logits = logits.masked_fill(blocked, float("-inf"))
                probabilities = logits.softmax(-1)
                label_index = int(probabilities.argmax())
                issue = self._label("taxonomy_issue", label_index)
                taxonomy_confidence = min(candidate["entity_confidence"], float(probabilities[label_index]))
                unsupported = set(self.runtime_config.get("diagnostic_only_taxonomy", []))
                low_confidence = taxonomy_confidence < 0.6
                unsupported_label = entity in unsupported or issue in unsupported
                strict_policy = self.runtime_config.get("taxonomy_output_policy") == "strict"
                abstained = strict_policy and (unsupported_label or low_confidence)
                candidate.update({
                    "issue_id": issue if not abstained else "ABSTAIN",
                    "taxonomy": f"{entity}#{issue}" if not abstained else "OTHER#ABSTAIN",
                    "taxonomy_confidence": taxonomy_confidence,
                    "taxonomy_abstained": abstained,
                    "taxonomy_low_confidence": low_confidence,
                    "taxonomy_unsupported_label": unsupported_label,
                    "taxonomy_policy": "strict" if strict_policy else "best_effort",
                })
        finally:
            self._release_model(issue_model)

        results_by_review: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for candidate in related:
            index = candidate["review_index"]
            text = texts[index]
            aspect, opinion = candidate["aspect"], candidate["opinion"]
            confidence = min(
                float(aspect.get("score", 0.0)), float(opinion.get("score", 0.0)),
                float(candidate["joint_confidence"]), float(candidate["relation_confidence"]),
            )
            if confidence < confidence_threshold:
                continue
            domain, domain_confidence = domain_by_review[index]
            reasons = ["candidate_unpromoted", "human_gold_not_available"]
            if candidate.get("taxonomy_abstained"):
                reasons.append("taxonomy_abstained")
            elif candidate.get("taxonomy_low_confidence"):
                reasons.append("taxonomy_low_confidence_best_effort")
            if candidate.get("taxonomy_unsupported_label"):
                reasons.append("taxonomy_label_diagnostic_only")
            results_by_review[index].append({
                "aspect": text[aspect["start"]:aspect["end"]],
                "opinion": text[opinion["start"]:opinion["end"]],
                "sentiment": candidate["sentiment"],
                "taxonomy": candidate["taxonomy"],
                "relation": True,
                "confidence": round(confidence, 4),
                "aspect_span": [aspect["start"], aspect["end"]],
                "opinion_span": [opinion["start"], opinion["end"]],
                "relation_probability": candidate["relation_confidence"],
                "sentiment_probabilities": {
                    label.removeprefix("related_"): probability
                    for label, probability in candidate["joint_probabilities"].items() if label != "no_relation"
                },
                "domain": domain,
                "domain_confidence": domain_confidence,
                "entity_id": candidate["entity_id"],
                "issue_id": candidate["issue_id"],
                "taxonomy_confidence": candidate["taxonomy_confidence"],
                "taxonomy_abstained": candidate["taxonomy_abstained"],
                "taxonomy_low_confidence": candidate["taxonomy_low_confidence"],
                "taxonomy_policy": candidate["taxonomy_policy"],
                "output_type": "triplet",
                "needs_human_review": True,
                "review_reasons": reasons,
                **self.negation_scope(text, opinion),
            })

        elapsed_ms = (time.perf_counter() - started) * 1000
        return [{
            "text": text,
            "results": sorted(results_by_review[index], key=lambda item: item["aspect_span"]),
            "processing_time_ms": round(elapsed_ms / max(1, len(texts)), 2),
            "engine_version": self.engine_version,
            "model_version": "V15.1",
            "model_source": self.bundle_dir.name,
            "model_bundle_loaded": True,
            "inference_mode": "transformer_bundle",
            "model_loading_mode": "streaming" if self._stream_models else "resident",
            "evaluation_mode": EVALUATION_MODE,
            "disclaimer": PROXY_DISCLAIMER_MESSAGE,
            "promotion_status": "candidate_unpromoted",
            "human_gold_status": "not_available",
            "diagnostics": diagnostics[index],
        } for index, text in enumerate(texts)]
