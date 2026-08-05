"""
Shared Pydantic models for the AI Engineering Bootcamp.

This file holds data models that are used by more than one feature.
Each feature may also define its own local models inside its own directory.

New models are added here as they are introduced in the course:
  - Feature 2 adds: StructuredResponse
  - Feature 3 adds: Message, Session
  - Feature 4 adds: Document, Chunk
  - Feature 6 adds: SmartChatResponse, RetrievalLogEntry, KnowledgeDigest
  - Feature 8 adds: AgentTask
"""
from datetime import datetime
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Feature 2: Structured output
# =============================================================================

class StructuredResponse(BaseModel):
    """
    A structured reply from the AI assistant that includes classification metadata
    alongside the answer text.

    Instead of returning plain text, the assistant analyzes the query, decides
    what kind of question it is, and returns that classification together with its
    answer and a confidence estimate. This lets the UI (and any downstream code)
    make decisions based on the *type* of question — for example, showing a
    warning when confidence is low, or routing action requests to a separate flow.
    """

    intent: Literal["general_question", "domain_question", "action_request", "unclear"] = Field(
        description=(
            "What kind of request the user made. "
            "'general_question' = factual/knowledge query unrelated to the domain. "
            "'domain_question' = a question specifically about the assistant's domain. "
            "'action_request' = the user wants something DONE (book, schedule, find, send, etc.). "
            "'unclear' = the query is ambiguous or doesn't fit the other categories."
        )
    )

    answer: str = Field(
        description=(
            "The assistant's response to the user's query, written in plain English. "
            "For action_request intents, this should explain what action would be taken "
            "and any information still needed to complete it."
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How confident the assistant is in its answer, from 0.0 (not confident at all) "
            "to 1.0 (completely certain). A confidence below 0.5 usually means the answer "
            "should be verified — this is surfaced visually in the UI."
        ),
    )

    sources_needed: bool = Field(
        description=(
            "True if this answer would be significantly better with access to domain documents "
            "(uploaded in Feature 4+). False if the answer is reliably answerable from "
            "general knowledge or the system prompt alone. "
            "This flag is used in Week 2 to decide whether to trigger RAG retrieval."
        )
    )


# =============================================================================
# Feature 3: Conversation memory
# =============================================================================

class Message(BaseModel):
    """
    A single message in a conversation — either from the user or the assistant.

    Messages are stored in order inside a Session and sent to the LLM as
    conversation history so the assistant can refer back to earlier exchanges.
    """

    role: Literal["user", "assistant"] = Field(
        description=(
            "Who sent this message. 'user' is the person typing; "
            "'assistant' is the AI's reply."
        )
    )

    content: str = Field(
        description="The text of the message."
    )

    timestamp: datetime = Field(
        description="When this message was recorded, in UTC."
    )


class Session(BaseModel):
    """
    A single conversation thread between a user and the assistant.

    A session holds the full ordered history of messages for one conversation.
    When the user starts a 'New Chat', a new session is created with a fresh
    empty history — previous sessions remain accessible in the sidebar.
    """

    id: str = Field(
        description="Unique identifier for this session (a UUID)."
    )

    created_at: datetime = Field(
        description="When this session was started, in UTC."
    )

    messages: List[Message] = Field(
        default_factory=list,
        description=(
            "All messages in this conversation, oldest first. "
            "The assistant uses this list as context when generating each new reply."
        ),
    )

    tenant_id: str = Field(
        default="default",
        description=(
            "Which tenant owns this session. 'default' = single-tenant mode. "
            "Set by the X-Tenant-ID request header when ENABLE_MULTI_TENANT=true (Feature 6, Part B)."
        ),
    )


# =============================================================================
# Feature 4: Document ingestion
# =============================================================================

class Document(BaseModel):
    """
    Metadata record for an uploaded document.

    When a file is uploaded it immediately gets a Document record with
    status='processing'. Once text extraction and chunking complete, status
    flips to 'ready' and chunk_count is updated. On failure: 'error'.

    The extracted text itself is NOT stored here — chunks are stored separately
    in Chunk objects. This keeps the metadata record lightweight.
    """

    id: str = Field(
        description="Unique identifier for this document (a UUID)."
    )

    filename: str = Field(
        description="Original filename as uploaded by the user (e.g. 'report.pdf')."
    )

    uploaded_at: datetime = Field(
        description="When this document was uploaded, in UTC."
    )

    status: Literal["processing", "ready", "error"] = Field(
        description=(
            "'processing' while text is being extracted and chunked. "
            "'ready' once chunks are stored and available for retrieval. "
            "'error' if extraction or chunking failed."
        )
    )

    chunk_count: int = Field(
        default=0,
        description="Number of text chunks created from this document. 0 while processing.",
    )

    chunking_strategy: str = Field(
        default="sentence",
        description=(
            "Which chunking strategy was used during ingestion. "
            "One of: 'sentence' (sentence-aware fixed-size, default), "
            "'paragraph' (paragraph-based), 'page' (one chunk per PDF page). "
            "Stored so the UI can display it and future re-ingestion can reproduce the same split."
        ),
    )

    tenant_id: str = Field(
        default="default",
        description=(
            "Which tenant owns this document. 'default' = single-tenant mode. "
            "Set from the X-Tenant-ID header when ENABLE_MULTI_TENANT=true (Feature 6, Part B)."
        ),
    )


class Chunk(BaseModel):
    """
    A single text chunk extracted from a Document.

    Documents are split into chunks so that only the most relevant passages
    are sent to the LLM on each query (RAG). Each Chunk stores the text and
    its position within the source document.

    In Feature 5 these chunks are converted to vector embeddings for semantic search.
    """

    id: str = Field(
        description="Unique identifier for this chunk (a UUID)."
    )

    document_id: str = Field(
        description="ID of the Document this chunk belongs to."
    )

    text: str = Field(
        description="The raw text content of this chunk."
    )

    chunk_index: int = Field(
        description="Zero-based position of this chunk within its document. Chunk 0 is the first."
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary key-value metadata. At minimum contains 'filename' and 'chunk_index'. "
            "Feature 5 may add 'embedding_model', 'token_count', etc."
        ),
    )


# =============================================================================
# Feature 6: Smart Router
# =============================================================================

class SmartChatResponse(BaseModel):
    """
    Response from POST /api/chat/smart — the Smart Router endpoint.

    Unlike StructuredResponse (which classifies the question), SmartChatResponse
    exposes HOW the answer was generated: whether the router retrieved context,
    which chunks it used, and how confident it was in the routing decision.

    This transparency lets the UI show source badges (llm / rag / hybrid /
    pageindex) that teach students how the routing decision affected the answer.
    """

    answer: str = Field(
        description="The assistant's answer in plain text."
    )

    source: Literal["llm", "rag", "hybrid", "pageindex"] = Field(
        description=(
            "'llm' = answered directly, no retrieval. "
            "'rag' = answer grounded in retrieved document chunks. "
            "'hybrid' = retrieval attempted but router was uncertain; context used as supplement. "
            "'pageindex' = answer grounded in PageIndex tree-navigation retrieval (optional path)."
        )
    )

    chunks_used: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "The document chunks that were retrieved and included in the context. "
            "Each dict has: text, filename, chunk_index, score, document_id. "
            "Empty list when source='llm'."
        ),
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "The router's confidence in its routing decision (from classify_query). "
            "High confidence (>0.6) → deterministic routing. "
            "Low confidence (0.4–0.6) → hybrid path."
        ),
    )

    retrieval_method: str = Field(
        description=(
            "'vector' = ChromaDB similarity search was used. "
            "'pageindex' = PageIndex tree navigation was used. "
            "'none' = no retrieval was performed."
        )
    )


# =============================================================================
# Feature 6: Part C — Long-term retrieval memory
# =============================================================================

class RetrievalLogEntry(BaseModel):
    """
    A single retrieval event — logged whenever the Smart Router retrieves chunks.

    Retrieval memory is distinct from conversation memory (Feature 3):
    Feature 3 remembers WHAT was said; this logs WHAT was retrieved and WHY.
    Over many sessions, these logs reveal which topics a tenant asks about most
    and which documents are used most — the raw material for KnowledgeDigest.
    """

    session_id: str = Field(description="Which session triggered this retrieval.")
    tenant_id: str = Field(default="default", description="Which tenant this belongs to.")
    query: str = Field(description="The user's question that triggered retrieval.")
    chunks_retrieved: List[str] = Field(
        description="IDs of the chunks that were retrieved (format: '{document_id}_{chunk_index}')."
    )
    timestamp: datetime = Field(description="When this retrieval happened, in UTC.")
    retrieval_method: str = Field(
        default="vector",
        description="'vector' or 'pageindex' — which retrieval path was used.",
    )


class KnowledgeDigest(BaseModel):
    """
    A summarized view of a tenant's retrieval history — built by
    build_knowledge_digest() in shared/retrieval_memory.py.

    Injected into the Smart Router's system prompt as:
    "Context about this user's history: {summary}"

    This is the 'summary-based memory' pattern from Resource 3 applied to
    RETRIEVAL history, not raw chat — it compresses cross-session patterns
    into a brief paragraph that fits in every prompt without growing unboundedly.
    """

    tenant_id: str
    summary: str = Field(description="2-3 sentence LLM-generated summary of retrieval history.")
    topics_covered: List[str] = Field(
        description="Key topics/themes extracted from recent queries."
    )
    last_updated: datetime
    source_session_count: int = Field(
        description="How many sessions' worth of retrieval data this digest covers."
    )


# =============================================================================
# Feature 8: Multi-step agent task
# =============================================================================

class AgentTask(BaseModel):
    """
    A long-running multi-step agent task tracked across the planning and
    execution phases.

    Lifecycle:
      "planning"  → make_plan() is generating the step list
      "executing" → execute_plan() is running steps one by one
      "done"      → all steps complete; result holds the final answer
      "error"     → an exception occurred; error holds the message

    The task is created by POST /api/agent/plan and polled via
    GET /api/agent/status/{task_id}. The UI renders live progress by
    watching steps_completed grow between polls.
    """

    id: str = Field(description="Unique task ID (UUID).")
    status: Literal["planning", "executing", "done", "error"] = Field(
        description="Current lifecycle state of the task."
    )
    message: str = Field(description="The original user request that started this task.")
    plan: List[str] | None = Field(
        default=None,
        description="Ordered list of step instructions produced by make_plan(). None until planning is complete.",
    )
    steps_completed: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Steps that have finished executing. Each entry: "
            "{step_index, step (instruction text), result (agent answer), tools_used}. "
            "Grows one entry at a time as execute_plan() progresses."
        ),
    )
    result: str | None = Field(
        default=None,
        description="Final synthesized answer, set when status='done'.",
    )
    error: str | None = Field(
        default=None,
        description="Error message, set when status='error'.",
    )
    session_id: str = Field(
        default="default",
        description="Session used to maintain conversation context across steps.",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant that owns this task (for multi-tenant isolation).",
    )
    created_at: datetime = Field(description="When this task was created, in UTC.")
    updated_at: datetime = Field(description="When this task was last updated, in UTC.")
