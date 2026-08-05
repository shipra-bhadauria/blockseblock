"""
Smart Router for Feature 6: the intelligence layer that decides whether and
how to retrieve before generating a response.

This module contains two things:
  1. classify_query() — a cheap, focused LLM call that classifies the query
     before the main response call. This is the "Anti-RAG" pattern: instead
     of blindly retrieving for every query, the router first DECIDES whether
     retrieval is needed, then lets the LLM reason over the retrieved context.

  2. PAGEINDEX_ROUTING — an inline comment block documenting the optional
     PageIndex routing path for professional document domains.

ANTI-RAG PHILOSOPHY:
  A naive RAG system retrieves for every query and then hopes the context
  helps. The problem: retrieval for "what is 2 + 2?" costs tokens, adds
  noise, and can confuse the model with irrelevant chunks.

  The Smart Router changes this: classify_query() makes a cheap LLM call
  to decide BEFORE retrieving. This is what makes the difference between
  a search engine that searches everything and an assistant that knows
  when to look things up.

  "Anti-RAG" is the name for this pattern — RAG with a pre-retrieval
  intelligence step that decides if retrieval is worth doing at all.
"""
import json

from shared.llm_client import call_llm

_CLASSIFIER_SYSTEM_PROMPT = """You are a query classifier for a document Q&A assistant.

Analyze the user's query and respond ONLY with a JSON object (no markdown, no extra text):
{
  "needs_retrieval": <true if this question likely needs uploaded domain documents, false if answerable from general knowledge>,
  "confidence": <0.0-1.0 — how certain you are about this classification>,
  "reasoning": "<one sentence explaining your decision>",
  "query_type": "<one of: general | domain | professional_document | ambiguous>"
}

Query type definitions:
- "general": common knowledge — greetings, math, basic facts, general how-to questions.
  These don't need domain documents (e.g. "What is the capital of France?", "Hello").
- "domain": questions about the organization's products, services, policies, or procedures
  that would benefit from uploaded documents (e.g. "What is your return policy?").
- "professional_document": questions requiring precise navigation of structured professional
  documents — financial filings, legal contracts, regulatory documents, technical specs.
  The answer requires finding a specific section, not just a semantically similar passage.
  (e.g. "What was net revenue in Q3?", "What are the termination clauses?").
- "ambiguous": genuinely unclear whether domain documents would help.

Confidence guide:
- 0.9–1.0: very clear (obvious greeting vs obvious domain question)
- 0.7–0.8: reasonably clear but some uncertainty
- 0.4–0.6: genuinely ambiguous — could go either way
- Below 0.4: you really can't tell

Be strict with "general": only use it when you're confident retrieval won't help."""


async def classify_query(query: str) -> dict:
    """
    Classify whether a query needs document retrieval and how to retrieve.

    Makes a focused, cheap LLM call with a low temperature to get a
    deterministic classification. This is the pre-retrieval step that
    prevents wasteful retrieval for general-knowledge questions.

    Returns:
      needs_retrieval: bool  — True → retrieve; False → answer directly
      confidence:      float — how sure the classifier is (0.0–1.0)
      reasoning:       str   — one-sentence explanation (for debugging)
      query_type:      str   — "general" | "domain" | "professional_document" | "ambiguous"

    The "professional_document" query_type is the signal for the optional
    PageIndex routing path. See the PAGEINDEX_ROUTING comment block below.

    Fallback: if classification fails for any reason, default to
    needs_retrieval=True, confidence=0.5 (hybrid path) — better to
    retrieve unnecessarily than to skip retrieval when it's needed.
    """
    result = await call_llm(
        messages=[
            {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(result.content or "{}")
        return {
            "needs_retrieval": bool(data.get("needs_retrieval", True)),
            "confidence": min(1.0, max(0.0, float(data.get("confidence", 0.5)))),
            "reasoning": str(data.get("reasoning", "")),
            "query_type": data.get("query_type", "ambiguous"),
        }
    except Exception:
        return {
            "needs_retrieval": True,
            "confidence": 0.5,
            "reasoning": "Classification failed — defaulting to retrieval (safe fallback).",
            "query_type": "ambiguous",
        }


def detect_modality(
    has_audio: bool = False,
    has_image: bool = False,
    content_type: str | None = None,
) -> str:
    """
    Detect the input modality of an incoming request.

    This is the Feature 10 extension of the Feature 6 Smart Router concept:
    instead of routing between "use docs" vs "use general knowledge," we now
    route between "text / voice / vision." The pattern is identical — inspect
    the request, decide which pipeline to invoke, pass through to it.

    Priority: explicit file uploads take precedence over Content-Type sniffing.

    Args:
        has_audio:    True when an audio file is present in the request.
        has_image:    True when an image file is present in the request.
        content_type: The HTTP Content-Type header (used when no explicit files).

    Returns:
        "voice"  — caller should invoke the STT → LLM → TTS pipeline
        "vision" — caller should invoke the image analysis (VLM) pipeline
        "text"   — caller should invoke the normal text chat pipeline
    """
    if has_audio:
        return "voice"
    if has_image:
        return "vision"
    if content_type:
        ct = content_type.lower()
        if "audio" in ct:
            return "voice"
        if "image" in ct:
            return "vision"
    return "text"


# =============================================================================
# PAGEINDEX ROUTING (optional — for professional document domains)
#
# When classify_query returns query_type="professional_document" AND
# PageIndex is available (ENABLE_PAGEINDEX=true in .env AND a tree JSON
# has been pre-built for the relevant document), route retrieval to
# PageIndex tree search instead of Chroma vector search:
#
#   from pageindex import PageIndex  # pip install -r VectifyAI/PageIndex
#   pi = PageIndex.load(tree_json_path)  # pre-built tree JSON
#   result = pi.retrieve(query)
#   chunks_used = [{
#       "text": result.text,
#       "filename": result.source,
#       "chunk_index": 0,
#       "score": 1.0,
#       "document_id": "",
#       "retrieval_method": "pageindex",
#   }]
#
# This replaces vector similarity with reasoning-based tree navigation
# for document types where similarity ≠ relevance is most harmful:
# financial filings, legal contracts, regulatory documents.
#
# The rest of the routing logic (LLM generation with context, source badge
# in SmartChatResponse, session storage) is identical — only retrieval differs.
#
# INTEGRATION IN FEATURE 6'S SMART ROUTER (solution/main.py):
#   if classification["query_type"] == "professional_document" and settings.enable_pageindex:
#       chunks_used = pageindex_retrieve(query)   # ← PageIndex path
#       source = "pageindex"
#       retrieval_method = "pageindex"
#   else:
#       chunks_used = vector_search(query, ...)   # ← Vector RAG path
#       source = "rag"
#       retrieval_method = "vector"
#
# See github.com/VectifyAI/PageIndex for building the tree index.
# Cloud service + MCP server at pageindex.ai.
# =============================================================================
