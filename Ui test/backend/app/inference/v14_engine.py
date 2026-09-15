"""Memory-aware adapter for the exported ABSA V14 bundle.

The exported V14 loader keeps every XLM-R model resident.  That needs more than
3 GB just for weights, so this web adapter loads one component at a time by
default and still uses the bundle's tokenizer, profiles, recovery resources,
pair decoder, and sentiment checkpoint.
"""
from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch

from app.core.config import EVALUATION_MODE, PROXY_DISCLAIMER_MESSAGE

try:
    from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForSequenceClassification = AutoModelForTokenClassification = AutoTokenizer = None


def spans_overlap(first: Dict[str, Any], second: Dict[str, Any]) -> bool:
    """Return True when two half-open character spans intersect."""
    return max(int(first["start"]), int(second["start"])) < min(int(first["end"]), int(second["end"]))


class V14InferenceEngine:
    component_types = {
        "aspect": AutoModelForTokenClassification,
        "opinion": AutoModelForTokenClassification,
        "sentiment": AutoModelForSequenceClassification,
    }

    def __init__(self, bundle_dir: Path, engine_version: str = "v14"):
        self.bundle_dir = Path(bundle_dir)
        self.engine_version = engine_version
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
        with (self.bundle_dir / "inference_config.json").open("r", encoding="utf-8") as handle:
            self.config = json.load(handle)
        self.runtime_config = self.config.get("config", {})
        self.labels = self.config.get("labels", {})
        self.profiles = self.config.get("profiles", {})

    def _load_helpers(self) -> None:
        bundle_path = str(self.bundle_dir.resolve())
        if bundle_path not in sys.path:
            sys.path.insert(0, bundle_path)
        from absa_v14 import legacy_resources
        from absa_v14.inference import global_decode, merge_spans, negation_scope
        from absa_v14.pair import PairTooWide, encode_marked, mark_pair

        self.resources = legacy_resources
        self.global_decode = global_decode
        self.merge_spans = merge_spans
        self.negation_scope = negation_scope
        self.PairTooWide = PairTooWide
        self.encode_marked = encode_marked
        self.mark_pair = mark_pair

    def _load_runtime(self) -> None:
        if not all((AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification)):
            self._model_load_error = "Dependency transformers tidak tersedia."
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.bundle_dir / "tokenizer"), local_files_only=True, use_fast=True
            )
            model_bytes = sum(
                (self.bundle_dir / name / "model.safetensors").stat().st_size
                for name, present in self.config.get("components", {}).items()
                if present and name in self.component_types
            )
            available = torch.cuda.mem_get_info(self._device)[0] if self._device.type == "cuda" else psutil.virtual_memory().available
            preference = os.environ.get("ABSA_MODEL_LOADING", "auto").strip().lower()
            self._stream_models = preference == "stream" or (preference == "auto" and model_bytes > available * 0.65)
            if preference == "resident":
                self._stream_models = False
            if not self._stream_models:
                for name in self._required_components():
                    self._models[name] = self._load_model(name)
        except Exception as exc:
            self._model_load_error = f"{type(exc).__name__}: {exc}"
            self.close()

    def _required_components(self) -> List[str]:
        configured = self.config.get("components", {})
        return [name for name in self.component_types if configured.get(name, False)]

    @property
    def model_bundle_loaded(self) -> bool:
        return self._tokenizer is not None and self._model_load_error is None

    def _load_model(self, name: str):
        loader = self.component_types[name]
        return loader.from_pretrained(
            str(self.bundle_dir / name), local_files_only=True, low_cpu_mem_usage=True
        ).to(self._device).eval()

    def _acquire_model(self, name: str):
        if name not in self._required_components():
            return None
        return self._load_model(name) if self._stream_models else self._models[name]

    def _release_model(self, model: Any) -> None:
        if self._stream_models and model is not None:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def close(self) -> None:
        self._models.clear()
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _spans_with_model(self, text: str, kind: str, threshold: float, dual: bool, model: Any) -> List[Dict[str, Any]]:
        views: List[Tuple[str, Any]] = [(text, None)]
        if dual:
            aligned = self.resources.normalize_multilingual_text_with_alignment(text)
            if aligned["normalized_text"] != text:
                views.append((aligned["normalized_text"], aligned["norm_to_raw"]))
        spans: List[Dict[str, Any]] = []
        for view, mapping in views:
            encoded = self._tokenizer(
                view,
                truncation=True,
                max_length=int(self.runtime_config.get("span_max_length", 256)),
                stride=int(self.runtime_config.get("span_stride", 64)),
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
            )
            for window, offsets in enumerate(encoded["offset_mapping"]):
                inputs = {
                    key: torch.tensor([encoded[key][window]], device=self._device)
                    for key in ("input_ids", "attention_mask")
                }
                with torch.inference_mode():
                    probabilities = model(**inputs).logits[0].softmax(-1).cpu()
                label_ids = probabilities.argmax(-1).tolist()
                current: Optional[Dict[str, Any]] = None
                found: List[Dict[str, Any]] = []
                for index, ((start, end), label_id) in enumerate(zip(offsets, label_ids)):
                    if end <= start:
                        continue
                    tag = self.labels["bio"].get(str(label_id), "O")
                    if tag == "B":
                        if current:
                            found.append(current)
                        current = {"start": start, "end": end}
                    elif tag == "I":
                        if current is None:
                            current = {"start": start, "end": end, "window_fragment": True}
                        current["end"] = end
                    elif current:
                        found.append(current)
                        current = None
                if current:
                    found.append(current)
                for span in found:
                    hits = [
                        index for index, (start, end) in enumerate(offsets)
                        if end > start and start < span["end"] and end > span["start"]
                    ]
                    span["score"] = (
                        sum(float(probabilities[index, label_ids[index]]) for index in hits) / len(hits)
                        if hits else 0.0
                    )
                    if span["score"] < threshold:
                        continue
                    if mapping:
                        raw_positions = mapping[span["start"]:span["end"]]
                        if not raw_positions:
                            continue
                        span["start"] = min(position[0] for position in raw_positions)
                        span["end"] = max(position[1] for position in raw_positions)
                    span["source_view"] = "normalized" if mapping else "raw"
                    span["is_recovery"] = False
                    spans.append(span)
        return self.merge_spans(spans)

    def _extract_batch(self, texts: List[str], kind: str, profile_config: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        model = self._acquire_model(kind)
        if model is None:
            return [[] for _ in texts]
        try:
            return [
                self._spans_with_model(
                    text, kind, float(profile_config[f"{kind}_threshold"]), bool(profile_config.get("dual_view")), model
                )
                for text in texts
            ]
        finally:
            self._release_model(model)

    def _sentiments(self, requests: List[Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]]) -> List[Tuple[Optional[str], float, Dict[str, float]]]:
        model = self._acquire_model("sentiment")
        if model is None:
            return [(None, 0.0, {}) for _ in requests]
        results = []
        try:
            for text, aspect, opinion in requests:
                marked = self.mark_pair(
                    text, aspect["start"], aspect["end"],
                    opinion["start"] if opinion else None, opinion["end"] if opinion else None,
                )
                encoded = self.encode_marked(
                    self._tokenizer,
                    marked,
                    int(self.runtime_config.get("sentiment_max_length", 384)),
                    target=text[aspect["start"]:aspect["end"]],
                    require_opinion=opinion is not None,
                )
                inputs = {key: torch.tensor([value], device=self._device) for key, value in encoded.items()}
                with torch.inference_mode():
                    probabilities = (
                        model(**inputs).logits[0] / float(self.runtime_config.get("sentiment_temperature", 1.0))
                    ).softmax(-1)
                predicted = int(probabilities.argmax())
                label_map = self.labels["sentiment"]
                results.append((
                    label_map[str(predicted)],
                    float(probabilities[predicted]),
                    {label_map[str(index)]: float(value) for index, value in enumerate(probabilities)},
                ))
            return results
        finally:
            self._release_model(model)

    def _recover(self, text: str, aspects: List[Dict[str, Any]], opinions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        for current, recover in ((aspects, self.resources.recover_aspects), (opinions, self.resources.recover_opinions)):
            for item in recover(text):
                current.append({**item, "is_recovery": True, "score": min(0.60, float(item.get("score", 0.5)))})
        repaired = []
        for aspect in aspects:
            for fragment in self.resources.split_coordinated_aspect(text, aspect):
                repaired.append({**fragment, "is_recovery": True} if fragment.get("coordination_repair") else fragment)
        return self.merge_spans(repaired), self.merge_spans(opinions)

    def _taxonomy(self, text: str, aspect: Dict[str, Any]) -> Tuple[str, float]:
        mapped = self.resources.taxonomy_mapper(text[aspect["start"]:aspect["end"]], text)
        return mapped.get("normalized_category", "Other"), float(mapped.get("taxonomy_confidence", 0.0))

    def analyze_batch(
        self, reviews: List[str], confidence_threshold: float = 0.5, profile: str = "production_precision"
    ) -> List[Dict[str, Any]]:
        if not self.model_bundle_loaded:
            raise RuntimeError(f"Bundle V14 tidak dapat dimuat: {self._model_load_error or 'runtime tidak tersedia'}")
        if profile not in self.profiles:
            raise ValueError(f"Profile '{profile}' tidak tersedia untuk V14.")
        texts = [str(review or "").strip() for review in reviews]
        for text in texts:
            if len(text) > int(self.runtime_config.get("max_input_chars", 30000)):
                raise ValueError("Review melebihi panjang maksimum V14.")
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

        planned: List[Tuple[int, Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = []
        diagnostics: List[Dict[str, Any]] = []
        for review_index, (text, aspects, opinions) in enumerate(zip(texts, aspects_by_review, opinions_by_review)):
            candidates = []
            overlap_rejected = 0
            proposals = sorted(
                ((aspect, opinion) for aspect in aspects for opinion in opinions),
                key=lambda pair: abs(pair[0]["start"] - pair[1]["start"]),
            )
            for aspect, opinion in proposals[: int(self.runtime_config.get("max_candidate_pairs", 512))]:
                if spans_overlap(aspect, opinion):
                    overlap_rejected += 1
                    continue
                between = text[min(aspect["end"], opinion["end"]):max(aspect["start"], opinion["start"])]
                if profile_config.get("recovery") and not re.search(
                    r"[.!?;]|\b(?:tapi|but|melainkan|namun)\b", between, re.I
                ):
                    candidates.append({
                        "aspect": aspect,
                        "opinion": opinion,
                        "score": 0.5 / (1 + abs(aspect["start"] - opinion["start"]) / 100),
                        "method": "heuristic_review_only_v14",
                    })
            pairs, rejected = self.global_decode(
                text, candidates, 0.1, int(self.runtime_config.get("max_opinions_per_aspect", 8))
            )
            paired = set()
            for pair in pairs:
                planned.append((review_index, pair["aspect"], pair["opinion"], pair))
                paired.add((pair["aspect"]["start"], pair["aspect"]["end"]))
            for aspect in aspects:
                if (aspect["start"], aspect["end"]) not in paired:
                    planned.append((review_index, aspect, None, None))
            diagnostics.append({
                "candidate_pairs": len(proposals),
                "rejected_pair_count": len(rejected) + overlap_rejected,
                "overlap_rejected_count": overlap_rejected,
            })

        sentiment_requests = [(texts[index], aspect, opinion) for index, aspect, opinion, _ in planned]
        sentiments = self._sentiments(sentiment_requests)
        results_by_review: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for plan, sentiment_result in zip(planned, sentiments):
            review_index, aspect, opinion, pair = plan
            sentiment, sentiment_confidence, probabilities = sentiment_result
            text = texts[review_index]
            taxonomy, taxonomy_confidence = self._taxonomy(text, aspect)
            pair_score = float(pair["score"]) if pair else 1.0
            confidence = min(float(aspect.get("score", 0.0)), sentiment_confidence, pair_score)
            if confidence < confidence_threshold:
                continue
            results_by_review[review_index].append({
                "aspect": text[aspect["start"]:aspect["end"]],
                "opinion": text[opinion["start"]:opinion["end"]] if opinion else "",
                "sentiment": sentiment or "neutral",
                "taxonomy": taxonomy,
                "relation": pair is not None,
                "confidence": round(confidence, 4),
                "aspect_span": [aspect["start"], aspect["end"]],
                "opinion_span": [opinion["start"], opinion["end"]] if opinion else None,
                "relation_probability": float(pair["score"]) if pair else None,
                "sentiment_probabilities": probabilities,
                "taxonomy_confidence": taxonomy_confidence,
                "output_type": "triplet" if pair else "aspect_only",
                "needs_human_review": True,
                "review_reasons": ["relation_not_human_validated", "taxonomy_heuristic_v14"],
                **self.negation_scope(text, opinion),
            })

        elapsed_ms = (time.perf_counter() - started) * 1000
        responses = []
        for index, text in enumerate(texts):
            responses.append({
                "text": text,
                "results": sorted(results_by_review[index], key=lambda item: item["aspect_span"]),
                "processing_time_ms": round(elapsed_ms / max(1, len(texts)), 2),
                "engine_version": self.engine_version,
                "model_version": self.runtime_config.get("project_version", self.config.get("version", "v14")),
                "model_source": self.bundle_dir.name,
                "model_bundle_loaded": True,
                "inference_mode": "transformer_bundle",
                "model_loading_mode": "streaming" if self._stream_models else "resident",
                "evaluation_mode": EVALUATION_MODE,
                "disclaimer": PROXY_DISCLAIMER_MESSAGE,
                "diagnostics": diagnostics[index],
            })
        return responses

    def analyze_single_review(
        self, text: str, confidence_threshold: float = 0.5, profile: str = "production_precision"
    ) -> Dict[str, Any]:
        return self.analyze_batch([text], confidence_threshold, profile)[0]
