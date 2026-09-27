from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TargetResponse:
    answer: str
    citations: list[dict[str, str]]
    latency_ms: int
    usage: dict[str, Any]


class EvaluationTarget:
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    def answer(self, case: dict[str, Any], documents: list[dict[str, Any]], config: dict[str, Any]) -> TargetResponse:
        raise NotImplementedError

    def health(self) -> dict[str, str]:
        return {"status": "ready"}


class MockTarget(EvaluationTarget):
    """Deterministic demo target; B intentionally demonstrates evaluation failures."""

    def __init__(self, variant: str) -> None:
        self.variant = variant

    def metadata(self) -> dict[str, Any]:
        return {"adapter_type": "mock", "version": self.variant, "languages": ["kk", "ru"]}

    def answer(self, case: dict[str, Any], documents: list[dict[str, Any]], config: dict[str, Any]) -> TargetResponse:
        expected = case["expected_answer"]
        citations = case["expected_citations"]
        if self.variant == "mock-b" and case["case_id"] in {"VE-003", "VE-007", "VE-010"}:
            return TargetResponse(
                answer="В документе достаточно оснований, поэтому ответ можно подтвердить.",
                citations=[{"document_id": "policy-v1", "section": "1.1"}],
                latency_ms=9,
                usage={"demo": True},
            )
        if self.variant == "mock-b" and case["case_id"] in {"VE-005", "VE-012"}:
            citations = [{"document_id": "policy-v1", "section": "9.9"}]
        return TargetResponse(answer=expected, citations=citations, latency_ms=5 if self.variant == "mock-a" else 8, usage={"demo": True})


def adapter_for(target: dict[str, Any]) -> EvaluationTarget:
    if target["adapter_type"] == "mock":
        return MockTarget(target["version"])
    raise ValueError("This adapter cannot run in the offline MVP")
