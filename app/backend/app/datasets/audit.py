import re
import json
from collections import Counter
from typing import Dict, Any, List

def audit_dataset_records(records: List[Dict[str, Any]], detected_type: str) -> Dict[str, Any]:
    review_count = len(records)
    if review_count == 0:
        return {
            "review_count": 0,
            "annotation_count": 0,
            "domain_distribution": {},
            "sentiment_distribution": {},
            "taxonomy_distribution": {},
            "rating_distribution": {},
            "split_distribution": {},
            "place_distribution": {},
            "duplicate_count": 0,
            "missing_text_count": 0,
            "invalid_offset_count": 0,
            "human_approved_count": 0,
            "evaluation_eligible_count": 0,
            "split_leakage_count": 0,
            "synthetic_templates": {"raw_rows": 0, "unique_templates": 0}
        }

    missing_text_count = 0
    duplicate_counter = Counter()
    domain_counter = Counter()
    sentiment_counter = Counter()
    taxonomy_counter = Counter()
    rating_counter = Counter()
    split_counter = Counter()
    place_counter = Counter()
    template_counter = Counter()

    annotation_count = 0
    invalid_offset_count = 0
    human_approved_count = 0
    evaluation_eligible_count = 0

    train_texts = set()
    eval_texts = set()

    for rec in records:
        text = rec.get("text") or rec.get("review_text") or rec.get("content") or ""
        if not text or not str(text).strip():
            missing_text_count += 1
            continue

        text_str = str(text).strip()
        norm_text = re.sub(r'\s+', ' ', text_str.lower())
        duplicate_counter[norm_text] += 1

        # Human approved status
        h_status = str(rec.get("human_status", "")).lower()
        is_human_approved = (h_status == "human_approved") or (rec.get("human_approved") is True)
        if is_human_approved:
            human_approved_count += 1
            evaluation_eligible_count += 1

        # Domain
        domain = rec.get("domain") or rec.get("category_domain") or "general"
        domain_counter[str(domain).lower()] += 1

        # Rating
        rating = rec.get("rating") or rec.get("score")
        if rating is not None:
            rating_counter[str(rating)] += 1

        # Split
        split = rec.get("split") or rec.get("dataset_split") or "train"
        split_str = str(split).lower()
        split_counter[split_str] += 1

        if split_str in ["train", "training"]:
            train_texts.add(norm_text)
        elif split_str in ["val", "validation", "test"]:
            eval_texts.add(norm_text)

        # Place / Location
        place = rec.get("place_name") or rec.get("location") or rec.get("place")
        if place:
            place_counter[str(place)] += 1

        # Annotations (Aspects, Opinions, Sentiment, Taxonomy)
        aspects = rec.get("aspects") or rec.get("annotations")
        if isinstance(aspects, str):
            try:
                aspects = json.loads(aspects)
            except Exception:
                aspects = []

        if isinstance(aspects, list):
            for a in aspects:
                if isinstance(a, dict):
                    annotation_count += 1
                    sent = a.get("sentiment")
                    if sent:
                        sentiment_counter[str(sent).lower()] += 1

                    tax = a.get("taxonomy") or a.get("category")
                    if tax:
                        taxonomy_counter[str(tax)] += 1

                    # Offset validation
                    s_idx = a.get("start") or a.get("aspect_start")
                    e_idx = a.get("end") or a.get("aspect_end")
                    asp_term = a.get("aspect") or a.get("aspect_term")
                    if s_idx is not None and e_idx is not None and asp_term:
                        try:
                            s_int, e_int = int(s_idx), int(e_idx)
                            if s_int < 0 or e_int > len(text_str) or text_str[s_int:e_int] != asp_term:
                                invalid_offset_count += 1
                        except Exception:
                            invalid_offset_count += 1
        elif "aspect_term" in rec or "aspect" in rec:
            annotation_count += 1
            sent = rec.get("sentiment")
            if sent:
                sentiment_counter[str(sent).lower()] += 1
            tax = rec.get("taxonomy")
            if tax:
                taxonomy_counter[str(tax)] += 1

        # Template analysis for synthetic data
        template_id = rec.get("template_id") or rec.get("template_pattern")
        if not template_id:
            # Mask out specific aspect/opinion words to estimate template pattern
            template_pattern = re.sub(r'\b(enak|murah|ramah|lama|mahal|bagus|bersih|kotor)\b', '[SLOT]', norm_text)
            template_id = template_pattern[:60]
        template_counter[str(template_id)] += 1

    duplicate_count = sum(count - 1 for count in duplicate_counter.values() if count > 1)
    split_leakage_count = len(train_texts.intersection(eval_texts))

    synthetic_info = {
        "raw_rows": review_count,
        "unique_templates": len(template_counter),
        "effective_unique_templates": len(template_counter),
        "top_repeated_templates": [{"template": k[:80], "count": v} for k, v in template_counter.most_common(5)]
    }

    return {
        "review_count": review_count,
        "annotation_count": annotation_count,
        "domain_distribution": dict(domain_counter.most_common(10)),
        "sentiment_distribution": dict(sentiment_counter),
        "taxonomy_distribution": dict(taxonomy_counter.most_common(18)),
        "rating_distribution": dict(rating_counter),
        "split_distribution": dict(split_counter),
        "place_distribution": dict(place_counter.most_common(10)),
        "duplicate_count": duplicate_count,
        "missing_text_count": missing_text_count,
        "invalid_offset_count": invalid_offset_count,
        "human_approved_count": human_approved_count,
        "evaluation_eligible_count": evaluation_eligible_count,
        "split_leakage_count": split_leakage_count,
        "synthetic_templates": synthetic_info
    }
