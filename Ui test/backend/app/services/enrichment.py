import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import CUSTOMER_CLASS_MAP, LOCATION_CONFIDENCE_THRESHOLD, MAX_REVIEW_LENGTH
from app.services.absa_adapter import normalize_absa_response


logger = logging.getLogger("absa.enrichment")


CUSTOMER_PATTERN = re.compile(
    r"(?<![\w])(?P<label>customer(?:_id)?|cust(?:omer)?\s*id|no|id)\s*[:#-]?\s*(?P<id>\d+)(?!\w)",
    re.IGNORECASE,
)


def _compact_spaces(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return re.sub(r"\s+([,.;!?])", r"\1", value)


def resolve_customer(raw_text: str, structured_id: Any = None) -> Tuple[Dict[str, Any], List[str], List[Tuple[int, int]]]:
    warnings: List[str] = []
    matches = list(CUSTOMER_PATTERN.finditer(raw_text or ""))
    text_ids = [match.group("id") for match in matches]
    if len(set(text_ids)) > 1:
        warnings.append("multiple_customer_ids")

    field_id: Optional[str] = None
    if structured_id is not None and str(structured_id).strip():
        candidate = str(structured_id).strip()
        if re.fullmatch(r"\d+", candidate):
            field_id = candidate
        else:
            warnings.append("invalid_structured_customer_id")

    text_id = text_ids[0] if text_ids else None
    if field_id and text_id and field_id != text_id:
        warnings.append("customer_id_conflict")
    customer_id = field_id or text_id
    customer_class = CUSTOMER_CLASS_MAP.get(customer_id[0], "UNKNOWN") if customer_id else "UNKNOWN"
    if customer_id and customer_class == "UNKNOWN":
        warnings.append("unsupported_customer_prefix")

    masked = None if not customer_id else (customer_id[:2] + "*" * max(1, len(customer_id) - 2))
    logger.debug(
        "customer_resolved id=%s class=%s source=%s warnings=%s",
        masked, customer_class, "structured_field" if field_id else ("review_text" if text_id else None), warnings,
    )

    return ({
        "customer_id": customer_id,
        "customer_class": customer_class,
        "classification_method": "first_digit_rule" if customer_id else None,
        "source": "structured_field" if field_id else ("review_text" if text_id else None),
    }, warnings, [(m.start(), m.end()) for m in matches])


@dataclass(frozen=True)
class Region:
    code: str
    name: str
    location_type: str
    province: str
    aliases: Tuple[str, ...]


class IndonesiaLocationResolver:
    LOCATION_PREFIX = re.compile(
        r"(?:\b(?:di|dari|ke|menuju|asal|domisili|alamat|daerah|wilayah|cabang|outlet)\s+|"
        r"\b(?:berasal|tinggal|dikirim|pengiriman)\s+(?:dari|di|ke)\s+)$",
        re.IGNORECASE,
    )
    NEGATIVE_PATTERNS = (
        r"\b(?:sangat|amat|begitu|nasibnya)\s+malang\b",
        r"\b(?:melempar|seperti|keras seperti)\s+batu\b",
        r"\b(?:bekerja|bermain|tampil)\s+solo\b",
    )

    def __init__(self, data_path: Optional[Path] = None):
        path = data_path or Path(__file__).resolve().parent.parent / "data" / "indonesia_regions.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.source = payload["source"]
        self.version = payload["version"]
        self.provinces: Dict[str, str] = {}
        self.province_aliases: Dict[str, str] = {}
        for code, name, aliases in payload["provinces"]:
            self.provinces[name.casefold()] = name
            for alias in [name, *aliases]:
                self.province_aliases[self._normal(alias)] = name
        self.regions: List[Region] = []
        self.alias_index: Dict[str, List[Region]] = {}
        for code, name, location_type, province, aliases in payload["regions"]:
            expanded = {self._normal(name), *(self._normal(alias) for alias in aliases)}
            region = Region(code, name, location_type, province, tuple(sorted(expanded, key=len, reverse=True)))
            self.regions.append(region)
            for alias in region.aliases:
                self.alias_index.setdefault(alias, []).append(region)
        self.sorted_aliases = sorted(self.alias_index, key=len, reverse=True)

    @staticmethod
    def _normal(value: Any) -> str:
        text = str(value or "").casefold().replace("kab.", "kabupaten ")
        text = re.sub(r"[^\w\s-]", " ", text, flags=re.UNICODE)
        return _compact_spaces(text.replace("_", " "))

    @staticmethod
    def _empty(source: Optional[str] = None) -> Dict[str, Any]:
        return {
            "raw_location": None, "city_or_regency": None, "province": None,
            "location_type": None, "semantic_type": "unknown_location_type",
            "source": source, "status": "not_found", "confidence": 0.0,
            "candidates": [], "evidence": [], "span": None,
        }

    def _province_from_value(self, value: Any) -> Optional[str]:
        return self.province_aliases.get(self._normal(value))

    def _make_result(
        self, raw: str, candidates: Sequence[Region], source: str, evidence: List[str],
        confidence: float, span: Optional[Tuple[int, int]] = None, province: Optional[str] = None,
    ) -> Dict[str, Any]:
        unique = list({candidate.code: candidate for candidate in candidates}.values())
        if province:
            compatible = [candidate for candidate in unique if candidate.province == province]
            if compatible:
                unique = compatible
                evidence.append("province_relation_valid")
            elif unique:
                return {
                    **self._empty(source), "raw_location": raw, "province": province,
                    "status": "conflict", "confidence": round(min(confidence, 0.49), 2),
                    "candidates": [candidate.name for candidate in unique],
                    "evidence": [*evidence, "city_province_conflict"], "span": list(span) if span else None,
                }
        if len(unique) != 1:
            return {
                **self._empty(source), "raw_location": raw,
                "province": province or (unique[0].province if unique and len({c.province for c in unique}) == 1 else None),
                "status": "ambiguous", "confidence": round(min(confidence, 0.69), 2),
                "candidates": [candidate.name for candidate in unique],
                "evidence": [*evidence, "multiple_master_data_matches"], "span": list(span) if span else None,
            }
        selected = unique[0]
        status = "resolved" if confidence >= LOCATION_CONFIDENCE_THRESHOLD else "ambiguous"
        return {
            "raw_location": raw, "city_or_regency": selected.name if status == "resolved" else None,
            "province": selected.province, "location_type": selected.location_type if status == "resolved" else None,
            "semantic_type": self._semantic_type(source, evidence), "source": source, "status": status,
            "confidence": round(confidence, 2),
            "candidates": [] if status == "resolved" else [selected.name],
            "evidence": evidence, "span": list(span) if span else None,
            "region_code": selected.code if status == "resolved" else None,
        }

    @staticmethod
    def _semantic_type(source: str, evidence: Sequence[str]) -> str:
        if source == "structured_field":
            return "customer_location"
        joined = " ".join(evidence)
        if "pengiriman" in joined or "dikirim" in joined:
            return "delivery_destination"
        if "cabang" in joined or "outlet" in joined:
            return "store_location"
        return "mentioned_location"

    def resolve(self, raw_text: str, city: Any = None, province: Any = None) -> Tuple[Dict[str, Any], List[str]]:
        warnings: List[str] = []
        structured_province = self._province_from_value(province) if province else None
        if province and not structured_province:
            warnings.append("unknown_structured_province")

        if city is not None and str(city).strip():
            raw_city = str(city).strip()
            normalized = self._normal(raw_city)
            candidates = self.alias_index.get(normalized, [])
            # A `city` structured field deliberately prefers a city when the bare
            # name also identifies a regency. Explicit kabupaten text remains a regency.
            if candidates and not re.match(r"^(?:kab|kabupaten)\b", normalized):
                city_candidates = [item for item in candidates if item.location_type == "city"]
                candidates = city_candidates or candidates
            if candidates:
                result = self._make_result(
                    raw_city, candidates, "structured_field", ["structured_city_field", "exact_master_data_match"],
                    0.99, province=structured_province,
                )
                text_result, _ = self._resolve_text(raw_text)
                if text_result["status"] == "resolved" and result.get("city_or_regency") != text_result.get("city_or_regency"):
                    warnings.append("location_source_conflict")
                    result["evidence"].append("review_text_mentions_different_location")
                return result, warnings
            warnings.append("unknown_structured_city")

        text_result, text_warnings = self._resolve_text(raw_text)
        warnings.extend(text_warnings)
        if structured_province:
            if text_result["status"] != "not_found" and text_result.get("province") != structured_province:
                text_result["status"] = "conflict"
                text_result["confidence"] = min(text_result["confidence"], 0.49)
                text_result["evidence"].append("structured_province_conflict")
                warnings.append("location_source_conflict")
            elif text_result["status"] == "not_found":
                text_result.update({
                    "raw_location": str(province), "province": structured_province, "location_type": "province",
                    "source": "structured_field", "status": "resolved", "confidence": 0.99,
                    "evidence": ["structured_province_field"],
                })
        logger.debug(
            "location_resolved status=%s city=%s province=%s confidence=%s warnings=%s",
            text_result.get("status"), text_result.get("city_or_regency"), text_result.get("province"),
            text_result.get("confidence"), warnings,
        )
        return text_result, warnings

    def _resolve_text(self, raw_text: str) -> Tuple[Dict[str, Any], List[str]]:
        text = raw_text or ""
        search_text = text.casefold()
        negative_spans = [
            match.span()
            for pattern in self.NEGATIVE_PATTERNS
            for match in re.finditer(pattern, search_text, flags=re.IGNORECASE)
        ]

        found: List[Tuple[int, int, str, List[Region], float, List[str]]] = []
        for alias in self.sorted_aliases:
            alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
            for match in re.finditer(rf"(?<!\w){alias_pattern}(?!\w)", search_text):
                if any(match.start() < end and match.end() > start for start, end in negative_spans):
                    continue
                before = search_text[max(0, match.start() - 35):match.start()]
                score = 0.42
                evidence = ["exact_master_data_match"]
                prefix = self.LOCATION_PREFIX.search(before)
                if prefix:
                    signal = prefix.group(0).strip()
                    score += 0.43
                    evidence.append(f"location_context: {signal}")
                if re.search(r"(?:^|\s)(?:kota|kabupaten|kab)\s*$", before):
                    score = max(score, 0.97)
                    evidence.append("explicit_administrative_type")
                original_fragment = text[match.start():match.end()]
                if original_fragment[:1].isupper():
                    score += 0.05
                    evidence.append("capitalized_candidate")
                if match.start() == 0 and re.match(r"\s+(?:no|id|customer)", search_text[match.end():], re.I):
                    score += 0.43
                    evidence.append("leading_metadata_position")
                found.append((match.start(), match.end(), original_fragment or alias, self.alias_index[alias], min(score, 0.99), evidence))

        # Prefer the longest overlapping alias, then the strongest contextual match.
        found.sort(key=lambda item: (item[0], -(item[1] - item[0]), -item[4]))
        deduped = []
        for item in found:
            if any(item[0] >= kept[0] and item[1] <= kept[1] for kept in deduped):
                continue
            deduped.append(item)
        if deduped:
            if len(deduped) > 1:
                warnings = ["multiple_locations_detected"]
            else:
                warnings = []
            best = max(deduped, key=lambda item: item[4])
            result = self._make_result(best[2], best[3], "review_text", best[5], best[4], (best[0], best[1]))
            if len(deduped) > 1:
                result["evidence"].append("multiple_locations_in_review")
                result["mentioned_locations"] = [item[2] for item in deduped]
            return result, warnings

        # Conservative fuzzy matching: only capitalized words near a location cue.
        for match in re.finditer(r"\b[A-Z][A-Za-z]{4,}\b", text):
            before = text[max(0, match.start() - 30):match.start()].casefold()
            if not self.LOCATION_PREFIX.search(before):
                continue
            token = self._normal(match.group(0))
            ranked = sorted(
                ((SequenceMatcher(None, token, alias).ratio(), alias) for alias in self.alias_index if " " not in alias),
                reverse=True,
            )
            if ranked and ranked[0][0] >= 0.84:
                close = [alias for ratio, alias in ranked if ranked[0][0] - ratio <= 0.03]
                candidates = [region for alias in close for region in self.alias_index[alias]]
                return self._make_result(
                    match.group(0), candidates, "fuzzy_review_text", ["fuzzy_master_data_match", f"score:{ranked[0][0]:.2f}"],
                    min(0.91, ranked[0][0]), (match.start(), match.end()),
                ), ["fuzzy_location_match"]
        return self._empty(), []


location_resolver = IndonesiaLocationResolver()


def clean_review_text(raw_text: str, customer_spans: Sequence[Tuple[int, int]], location: Dict[str, Any]) -> str:
    spans = list(customer_spans)
    if location.get("status") == "resolved" and location.get("source") in {"review_text", "fuzzy_review_text"}:
        span = location.get("span")
        if span:
            spans.append((int(span[0]), int(span[1])))
    clean = raw_text
    for start, end in sorted(spans, reverse=True):
        clean = clean[:start] + " " + clean[end:]
    clean = re.sub(r"^[\s,;:.-]+|[\s,;:.-]+$", "", clean)
    return _compact_spaces(clean)


def analyze_enriched(
    engine_manager: Any, raw_text: Any, engine_version: str = "v11", confidence_threshold: float = 0.5,
    profile: str = "production_precision", customer_id: Any = None, city: Any = None, province: Any = None,
) -> Dict[str, Any]:
    started = perf_counter()
    text = str(raw_text or "")
    if len(text) > MAX_REVIEW_LENGTH:
        raise ValueError(f"Review melebihi batas {MAX_REVIEW_LENGTH} karakter.")
    customer, customer_warnings, customer_spans = resolve_customer(text, customer_id)
    location, location_warnings = location_resolver.resolve(text, city, province)
    clean_review = clean_review_text(text, customer_spans, location)
    preprocessing_ms = (perf_counter() - started) * 1000

    inference_started = perf_counter()
    raw_absa = engine_manager.analyze_single_review(engine_version, clean_review, confidence_threshold, profile)
    inference_ms = (perf_counter() - inference_started) * 1000
    post_started = perf_counter()
    model_warnings = []
    if raw_absa.get("promotion_status") == "candidate_unpromoted":
        model_warnings.append("V15.1 masih candidate_unpromoted dan belum memiliki validasi Human GOLD.")
    warnings = list(dict.fromkeys([*customer_warnings, *location_warnings, *model_warnings]))
    canonical = normalize_absa_response(raw_absa)
    postprocessing_ms = (perf_counter() - post_started) * 1000
    total_ms = (perf_counter() - started) * 1000

    # Existing top-level fields are intentionally retained for backward compatibility.
    return {
        **raw_absa,
        "success": True,
        "raw_text": text,
        "clean_review": clean_review,
        "customer": customer,
        "location": location,
        "absa": canonical,
        "model_version": raw_absa.get("model_version", engine_version),
        "warnings": warnings,
        "timing": {
            "preprocessing_time_ms": round(preprocessing_ms, 3),
            "inference_time_ms": round(inference_ms, 3),
            "postprocessing_time_ms": round(postprocessing_ms, 3),
            "end_to_end_latency_ms": round(total_ms, 3),
        },
    }


def analyze_enriched_batch(
    engine_manager: Any, rows: Sequence[Dict[str, Any]], engine_version: str = "v11",
    confidence_threshold: float = 0.5, profile: str = "production_precision",
) -> List[Dict[str, Any]]:
    """Enrich a chunk while allowing large models to execute a real batch pass."""
    prepared = []
    for row in rows:
        started = perf_counter()
        text = str(row.get("raw_text") or "")
        if len(text) > MAX_REVIEW_LENGTH:
            raise ValueError(f"Review melebihi batas {MAX_REVIEW_LENGTH} karakter.")
        customer, customer_warnings, customer_spans = resolve_customer(text, row.get("customer_id"))
        location, location_warnings = location_resolver.resolve(text, row.get("city"), row.get("province"))
        clean_review = clean_review_text(text, customer_spans, location)
        prepared.append({
            "raw_text": text,
            "clean_review": clean_review,
            "customer": customer,
            "location": location,
            "warnings": list(dict.fromkeys([*customer_warnings, *location_warnings])),
            "preprocessing_time_ms": (perf_counter() - started) * 1000,
        })

    inference_started = perf_counter()
    raw_results = engine_manager.analyze_batch(
        engine_version, [item["clean_review"] for item in prepared], confidence_threshold, profile
    )
    inference_ms = (perf_counter() - inference_started) * 1000
    if len(raw_results) != len(prepared):
        raise RuntimeError("Jumlah hasil batch model tidak sama dengan jumlah input.")
    per_row_inference_ms = inference_ms / max(1, len(prepared))

    outputs = []
    for item, raw_absa in zip(prepared, raw_results):
        post_started = perf_counter()
        canonical = normalize_absa_response(raw_absa)
        postprocessing_ms = (perf_counter() - post_started) * 1000
        total_ms = item["preprocessing_time_ms"] + per_row_inference_ms + postprocessing_ms
        model_warnings = []
        if raw_absa.get("promotion_status") == "candidate_unpromoted":
            model_warnings.append("V15.1 masih candidate_unpromoted dan belum memiliki validasi Human GOLD.")
        outputs.append({
            **raw_absa,
            "success": True,
            "raw_text": item["raw_text"],
            "clean_review": item["clean_review"],
            "customer": item["customer"],
            "location": item["location"],
            "absa": canonical,
            "model_version": raw_absa.get("model_version", engine_version),
            "warnings": list(dict.fromkeys([*item["warnings"], *model_warnings])),
            "timing": {
                "preprocessing_time_ms": round(item["preprocessing_time_ms"], 3),
                "inference_time_ms": round(per_row_inference_ms, 3),
                "postprocessing_time_ms": round(postprocessing_ms, 3),
                "end_to_end_latency_ms": round(total_ms, 3),
            },
        })
    return outputs
