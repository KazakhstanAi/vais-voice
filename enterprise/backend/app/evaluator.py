from __future__ import annotations

from statistics import mean
from typing import Any


def normalise(value: str) -> str:
    return " ".join(value.lower().replace("ё", "е").split())


def evaluate_case(case: dict[str, Any], response: Any) -> dict[str, Any]:
    expected_citations = {(item["document_id"], item["section"]) for item in case["expected_citations"]}
    actual_citations = {(item.get("document_id", ""), item.get("section", "")) for item in response.citations}
    answer_correct = float(normalise(case["expected_answer"]) in normalise(response.answer))
    correct_citations = expected_citations & actual_citations
    citation_precision = len(correct_citations) / len(actual_citations) if actual_citations else 0.0
    citation_completeness = len(correct_citations) / len(expected_citations) if expected_citations else None
    unsupported = any(claim.lower() in response.answer.lower() for claim in case.get("forbidden_claims", []))
    invalid_citation = bool(actual_citations - expected_citations)
    hallucination = float(unsupported or invalid_citation)
    cross_language = answer_correct if case["question_language"] not in case["document_languages"] else None
    version_awareness = citation_completeness if case["category"] == "version_awareness" else None
    return {
        "case_id": case["case_id"], "question": case["question"], "expected_answer": case["expected_answer"],
        "raw_answer": response.answer, "citations": response.citations, "latency_ms": response.latency_ms,
        "status": "completed", "metrics": {
            "answer_correctness": answer_correct, "citation_precision": citation_precision,
            "citation_completeness": citation_completeness, "hallucination_rate": hallucination,
            "retrieval_recall_at_k": None, "version_awareness": version_awareness,
            "cross_language_score": cross_language,
        },
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    values: dict[str, list[float]] = {}
    for result in results:
        for name, value in result["metrics"].items():
            if value is not None:
                values.setdefault(name, []).append(value)
    latency = sorted(item["latency_ms"] for item in results)
    metrics = {name: {"value": round(mean(items), 4), "metric_source": "deterministic"} for name, items in values.items()}
    metrics["retrieval_recall_at_k"] = {"value": None, "metric_source": "not_evaluated"}
    metrics["latency_mean_ms"] = {"value": round(mean(latency), 2), "metric_source": "deterministic"}
    metrics["latency_p50_ms"] = {"value": latency[(len(latency) - 1) // 2], "metric_source": "deterministic"}
    metrics["latency_p95_ms"] = {"value": latency[min(len(latency) - 1, int(len(latency) * 0.95))], "metric_source": "deterministic"}
    return metrics
