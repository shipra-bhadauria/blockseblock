"""
Long-term retrieval memory for Feature 6, Part C.

WHAT THIS SOLVES:
  Feature 3's sliding window handles SHORT-term memory — the last 20 messages
  of one conversation. But in a real consulting engagement, a user might
  interact across WEEKS and many separate sessions. Two problems emerge:

  1. The system has no memory of PREVIOUS sessions once they age out — it
     may re-explain things already covered or contradict earlier answers.
  2. Repeated retrieval of popular chunks doesn't build a "relationship"
     with the tenant's evolving knowledge needs.

  This module introduces RETRIEVAL MEMORY — distinct from Feature 3's
  CONVERSATION memory. Feature 3 remembers what was SAID. This remembers
  what was RETRIEVED and why — a completely different signal.

KEY FUNCTION: build_knowledge_digest()
  Pulls recent retrieval log entries, extracts unique topics, and makes
  ONE summarization LLM call to produce a 2-3 sentence KnowledgeDigest.
  The digest is injected into the Smart Router's system prompt so the model
  has cross-session context without the prompt growing unboundedly.

  This is the "summary-based memory" pattern from Resource 3, applied
  specifically to retrieval history instead of raw chat.

Public API:
  log_retrieval(session_id, tenant_id, query, chunk_ids, retrieval_method)
  get_recent_retrievals(tenant_id, limit) → list[RetrievalLogEntry]
  build_knowledge_digest(tenant_id)       → KnowledgeDigest | None
  get_current_digest(tenant_id)           → KnowledgeDigest | None
"""
import json
from datetime import datetime, timezone

from shared.llm_client import call_llm
from shared.models import KnowledgeDigest, RetrievalLogEntry

# In-memory stores — parallels session_store.py pattern.
# In production: replace with a database table (see Feature 11).
_retrieval_log: list[RetrievalLogEntry] = []
_digests: dict[str, KnowledgeDigest] = {}  # keyed by tenant_id


def log_retrieval(
    session_id: str,
    tenant_id: str,
    query: str,
    chunk_ids: list[str],
    retrieval_method: str = "vector",
) -> None:
    """
    Record that the Smart Router retrieved chunks in response to a query.

    Called automatically by smart_chat whenever source != "llm".
    Over many sessions these entries accumulate into a picture of
    what the tenant cares about — the raw material for build_knowledge_digest().
    """
    _retrieval_log.append(RetrievalLogEntry(
        session_id=session_id,
        tenant_id=tenant_id,
        query=query,
        chunks_retrieved=chunk_ids,
        timestamp=datetime.now(tz=timezone.utc),
        retrieval_method=retrieval_method,
    ))


def get_recent_retrievals(tenant_id: str, limit: int = 50) -> list[RetrievalLogEntry]:
    """
    Return the most recent retrieval log entries for a tenant.

    Returns entries most-recent first, capped at `limit`.
    Used by GET /api/retrieval-memory/recent and by build_knowledge_digest().
    """
    tenant_entries = [e for e in _retrieval_log if e.tenant_id == tenant_id]
    return sorted(tenant_entries, key=lambda e: e.timestamp, reverse=True)[:limit]


async def build_knowledge_digest(tenant_id: str) -> KnowledgeDigest | None:
    """
    Build (or rebuild) a KnowledgeDigest from recent retrieval history.

    THIS IS THE CORE TODO FOR PART C:
    1. Get recent retrievals for this tenant.
    2. Extract the unique queries (what the user has been asking about).
    3. Make ONE summarization LLM call to produce a 2-3 sentence digest.
    4. Store the digest in _digests[tenant_id] and return it.

    The digest is injected into the Smart Router's system prompt as:
    "Context about this user's history: {digest.summary}"

    This is the 'summary-based memory' pattern from Resource 3 — applied
    to RETRIEVAL history, not raw chat history.

    Returns None if there is no retrieval history for this tenant yet.
    """
    entries = get_recent_retrievals(tenant_id, limit=50)
    if not entries:
        return None

    unique_queries = list(dict.fromkeys(e.query for e in entries))[:20]
    sessions_covered = len({e.session_id for e in entries})

    summarize_prompt = (
        "Here are the topics and questions this user has asked about across recent sessions:\n\n"
        + "\n".join(f"- {q}" for q in unique_queries)
        + "\n\nWrite a 2-3 sentence summary of what this user seems to care about "
        "and what knowledge areas have already been covered. Be concise and specific. "
        "Respond with ONLY a JSON object: "
        '{"summary": "<2-3 sentence summary>", "topics": ["<topic1>", "<topic2>", ...]}'
    )

    result = await call_llm(
        messages=[
            {"role": "system", "content": "You summarize user inquiry history concisely."},
            {"role": "user", "content": summarize_prompt},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(result.content or "{}")
        digest = KnowledgeDigest(
            tenant_id=tenant_id,
            summary=data.get("summary", ""),
            topics_covered=data.get("topics", []),
            last_updated=datetime.now(tz=timezone.utc),
            source_session_count=sessions_covered,
        )
        _digests[tenant_id] = digest
        return digest
    except Exception:
        return None


def get_current_digest(tenant_id: str) -> KnowledgeDigest | None:
    """Return the most recently built digest for a tenant, or None."""
    return _digests.get(tenant_id)
