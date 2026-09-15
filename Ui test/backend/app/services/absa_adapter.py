"""Normalize every ABSA engine response without coupling metadata to a model."""
from typing import Any, Dict


SENTIMENT_ALIASES = {
    "positive": "positive", "positif": "positive", "pos": "positive",
    "negative": "negative", "negatif": "negative", "neg": "negative",
    "neutral": "neutral", "netral": "neutral", "neu": "neutral",
}


def normalize_absa_response(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    aspects = []
    for item in raw_result.get("results", []):
        raw_sentiment = str(item.get("sentiment", "neutral")).strip().casefold()
        aspects.append({
            "aspect": item.get("aspect"),
            "opinion": item.get("opinion"),
            "sentiment": SENTIMENT_ALIASES.get(raw_sentiment, raw_sentiment),
            "complaint_taxonomy": item.get("complaint_taxonomy", item.get("taxonomy")),
            "confidence": item.get("confidence"),
            "relation": item.get("relation"),
            "domain": item.get("domain"),
            "domain_confidence": item.get("domain_confidence"),
            "entity_id": item.get("entity_id"),
            "issue_id": item.get("issue_id"),
            "taxonomy_confidence": item.get("taxonomy_confidence"),
            "taxonomy_abstained": item.get("taxonomy_abstained"),
            "taxonomy_low_confidence": item.get("taxonomy_low_confidence"),
            "taxonomy_policy": item.get("taxonomy_policy"),
            "needs_human_review": item.get("needs_human_review"),
            "review_reasons": item.get("review_reasons", []),
        })
    return {"aspects": aspects}
