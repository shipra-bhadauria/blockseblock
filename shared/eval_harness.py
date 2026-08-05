"""
Eval harness for automated quality checks against a golden test set.

WHAT YOU BUILT → FRAMEWORK EQUIVALENT:
  EvalCase           → LangSmith Dataset example (question + expected output)
  EvalResult         → LangSmith EvaluationResult (per-example pass/fail)
  EvalReport         → LangSmith ExperimentResults summary
  run_eval()         → LangSmith evaluate() / RAGAS evaluate()

WHY EVALS MATTER:
  LLM outputs are non-deterministic — you can't unit-test them with assert.
  Instead, you build a golden test set of (question, expected_behaviour) pairs
  and run them automatically on every deploy. If pass rate drops, the deploy
  failed quality gates. This is the "eval-driven development" pattern used
  by production AI teams.

HOW THE CHECKS WORK:
  intent_check    — classify_query() returns the same intent as expected.
                    Skipped for action_request / unclear (routing always passes).
  source_check    — the routing decision (llm / rag / hybrid) matches expected.
                    Skipped when expected_source is None (no documents indexed).
  content_check   — the LLM answer contains all strings in expected_answer_contains.
                    Case-insensitive substring search (not exact match).

A case PASSES when all active checks pass (none fail).
"""
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from shared.llm_client import call_llm
from shared.router import classify_query
from shared.vector_store import search as vector_search


class EvalCase(BaseModel):
    id: str = Field(default="")
    question: str
    expected_intent: str | None = None       # general_question | domain_question | action_request | unclear
    expected_source: str | None = None       # llm | rag | hybrid — None = skip check
    expected_answer_contains: list[str] = [] # substrings that must appear in the answer


class EvalResult(BaseModel):
    case_id: str
    question: str
    passed: bool
    actual_intent: str
    actual_source: str
    actual_answer: str
    checks_passed: list[str]
    checks_failed: list[str]


class EvalReport(BaseModel):
    total: int
    passed: int
    failed: int
    pass_rate: float
    cases: list[EvalResult]
    ran_at: str


_EVAL_SYSTEM = (
    "You are a helpful assistant for Alpine Trail Co., an outdoor gear retailer. "
    "Answer clearly and concisely in plain English."
)


async def run_eval(
    test_cases: list[EvalCase],
    tenant_id: str = "default",
) -> EvalReport:
    """
    Run all test cases and return a report.

    Calls classify_query, vector_search, and call_llm directly — not via HTTP.
    This keeps evals fast (no network round-trips) and independent of the server
    being up (you can run them in CI before deployment).
    """
    results: list[EvalResult] = []

    for case in test_cases:
        case_id = case.id or str(uuid.uuid4())[:8]

        # ── Classify the question (same as smart_chat does) ──
        classification = await classify_query(case.question)
        actual_intent  = classification.get("intent", "unclear")
        needs_retrieval = classification.get("needs_retrieval", False)
        confidence      = classification.get("confidence", 0.0)

        # ── Route to source (mirrors smart_chat routing logic) ──
        chunks: list[dict[str, Any]] = []
        high_confidence = confidence > 0.6

        if high_confidence and needs_retrieval:
            chunks = vector_search(case.question, top_k=3, tenant_id=tenant_id)
            actual_source = "rag"
        elif not high_confidence:
            chunks = vector_search(case.question, top_k=3, tenant_id=tenant_id)
            actual_source = "hybrid"
        else:
            actual_source = "llm"

        # ── Call LLM ──
        messages: list[dict] = [{"role": "system", "content": _EVAL_SYSTEM}]
        if chunks:
            ctx = "\n\n".join(c.get("text", "") for c in chunks[:3])
            messages.append({"role": "system", "content": f"Retrieved context:\n{ctx}"})
        messages.append({"role": "user", "content": case.question})

        llm_result   = await call_llm(messages)
        actual_answer = llm_result.content or ""

        # ── Run checks ──
        checks_passed: list[str] = []
        checks_failed: list[str] = []

        # Intent check — skip for action_request / unclear (routing always acceptable)
        if case.expected_intent and case.expected_intent not in ("action_request", "unclear"):
            if actual_intent == case.expected_intent:
                checks_passed.append("intent")
            else:
                checks_failed.append(f"intent: expected={case.expected_intent} got={actual_intent}")

        # Source check — skip when expected_source is None (no documents guaranteed)
        if case.expected_source is not None:
            if actual_source == case.expected_source:
                checks_passed.append("source")
            else:
                checks_failed.append(f"source: expected={case.expected_source} got={actual_source}")

        # Content checks — each phrase must appear (case-insensitive)
        for phrase in case.expected_answer_contains:
            if phrase.lower() in actual_answer.lower():
                checks_passed.append(f"contains:{phrase!r}")
            else:
                checks_failed.append(f"missing:{phrase!r}")

        results.append(EvalResult(
            case_id=case_id,
            question=case.question,
            passed=len(checks_failed) == 0,
            actual_intent=actual_intent,
            actual_source=actual_source,
            actual_answer=actual_answer,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        ))

    total        = len(results)
    passed_count = sum(1 for r in results if r.passed)

    return EvalReport(
        total=total,
        passed=passed_count,
        failed=total - passed_count,
        pass_rate=round(passed_count / total, 3) if total > 0 else 0.0,
        cases=results,
        ran_at=datetime.now(tz=timezone.utc).isoformat(),
    )
