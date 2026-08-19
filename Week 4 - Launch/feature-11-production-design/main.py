"""
Feature 11: Production Design — starter

Your task: make this AI assistant production-ready.

Four TODOs in this file. Everything else (F1-F10 endpoints) is wired up
and working — you're adding the production layer on top.

Part A — Observability (TODOs 1–3):
  TODO 1: Initialize structured JSON logging at startup
  TODO 2: Attach RequestIDMiddleware and TimingMiddleware to the app
  TODO 3: Enable rate limiting on /api/chat and /api/sessions/{id}/agent/run

Part B — Eval Harness (TODO 4):
  TODO 4: Implement the eval run logic in POST /api/eval/run

Hints:
  - shared/logging_config.py  → setup_logging()
  - shared/middleware.py      → RequestIDMiddleware, TimingMiddleware
  - shared/metrics.py         → get_metrics(), set_eval_result()
  - shared/eval_harness.py    → EvalCase, EvalReport, run_eval()
  - slowapi docs: https://slowapi.readthedocs.io/en/latest/

Run with:
    uvicorn main:app --reload --port 8000
"""
import base64
import json
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared import metrics
from shared.agent import run_agent_with_mcp
from shared.document_store import (
    delete_document, get_chunks, get_document, list_documents,
    save_chunk, save_document, update_document,
)
from shared.eval_harness import EvalCase, EvalReport, run_eval
from shared.ingestion import CHUNKING_STRATEGIES, extract_pages, extract_text
from shared.llm_client import analyze_image, call_llm, synthesize_speech, transcribe_audio
from shared.mcp_client import (
    SERVER_REGISTRY, call_mcp_tool, get_mcp_tool_schemas, list_mcp_tools,
)
from shared.models import (
    AgentTask, Chunk, Document, Message, SmartChatResponse, StructuredResponse,
)
from shared.planner import execute_plan, make_plan
from shared.provider_check import check_provider_config
from shared.retrieval_memory import (
    build_knowledge_digest, get_current_digest, get_recent_retrievals, log_retrieval,
)
from shared.router import classify_query, detect_modality
from shared.session_store import add_message, create_session, get_session, list_sessions
from shared.task_store import create_task, get_task, update_task
from shared.tenant_context import get_tenant_id
from shared.vector_store import (
    add_chunks, delete_document_chunks, get_stats as vector_get_stats, search as vector_search,
)

CONTEXT_WINDOW_SIZE = 20
_EVAL_CASES_PATH = Path(__file__).parent.parent / "tests" / "eval_cases_example.json"

OPENAPI_TAGS = [
    {"name": "F1 · Hello AI",        "description": "**Week 1 · Feature 1** — Basic text chat."},
    {"name": "F2 · Prompt Mastery",   "description": "**Week 1 · Feature 2** — Structured JSON-mode responses."},
    {"name": "F3 · AI Memory",        "description": "**Week 1 · Feature 3** — Session management."},
    {"name": "F4 · Feed the Brain",   "description": "**Week 2 · Feature 4** — Document ingestion."},
    {"name": "F5 · Find the Answer",  "description": "**Week 2 · Feature 5** — Semantic search."},
    {"name": "F6 · Smart Router",     "description": "**Week 2 · Feature 6** — Intelligent routing + tenant isolation + retrieval memory."},
    {"name": "F7 · First Agent",      "description": "**Week 3 · Feature 7** — Tool-calling agent (ReAct pattern)."},
    {"name": "F8 · Multi-Step Agent", "description": "**Week 3 · Feature 8** — Plan-and-Execute agent with background execution."},
    {"name": "F9 · MCP Integration",  "description": "**Week 3 · Feature 9** — Model Context Protocol server integration."},
    {"name": "F10 · Multimodal AI",   "description": "**Week 4 · Feature 10** — Voice (STT/TTS) and Vision (VLM)."},
    {"name": "F11 · Production Design", "description": "**Week 4 · Feature 11** — Observability, rate limiting, and eval harness."},
    {"name": "Infrastructure",        "description": "Health check and provider info."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # =========================================================================
    # TODO 1: Initialize structured JSON logging.
    #
    # Import setup_logging from shared.logging_config and call it here.
    # This replaces the default plain-text log format with machine-readable
    # JSON lines — each log line becomes a searchable JSON object in your
    # log aggregator (Datadog, Loki, CloudWatch).
    #
    # from shared.logging_config import setup_logging
    # setup_logging()
    # =========================================================================
    from shared.logging_config import setup_logging
    setup_logging()
    await check_provider_config()
    yield


app = FastAPI(
    title="My AI BlockSeBlock Assistant",
    version="11.0.0",
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
)

# =============================================================================
# TODO 2: Add request tracing and latency middleware.
#
# Import RequestIDMiddleware and TimingMiddleware from shared.middleware.
# Add them both to `app` using app.add_middleware().
#
# Order matters: RequestIDMiddleware should be added first (outermost) so the
# request_id is set before TimingMiddleware records the request.
#
# from shared.middleware import RequestIDMiddleware, TimingMiddleware
# app.add_middleware(RequestIDMiddleware)
# app.add_middleware(TimingMiddleware)
# =============================================================================
from shared.middleware import RequestIDMiddleware, TimingMiddleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(TimingMiddleware)

# =============================================================================
# TODO 3: Enable rate limiting with slowapi.
#
# Install: pip install slowapi
# Then uncomment the block below.
#
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.errors import RateLimitExceeded
# from slowapi.util import get_remote_address
#
# limiter = Limiter(key_func=get_remote_address)
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
#
# Then add `http_request: Request` as the first parameter of the endpoints
# you want to rate-limit and decorate them:
#
#   @app.post("/api/chat")
#   @limiter.limit("60/minute")
#   async def chat(http_request: Request, request: ChatRequest) -> ChatResponse:
# =============================================================================
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

limiter = Limiter(key_func = get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class SessionSummary(BaseModel):
    id: str
    created_at: str
    message_count: int
    title: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    document_id: str | None = None

class SmartChatRequest(BaseModel):
    message: str

class AgentRequest(BaseModel):
    message: str

class PlanRequest(BaseModel):
    message: str

class McpExecuteRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = {}

class VoiceChatResponse(BaseModel):
    transcript: str
    answer: str
    audio_base64: str
    audio_mime: str = "audio/mpeg"

class VisionAnalyzeResponse(BaseModel):
    answer: str
    model_used: str

class MultimodalChatResponse(BaseModel):
    modality: str
    answer: str
    source: str | None = None
    chunks_used: list[dict] = []
    confidence: float | None = None
    retrieval_method: str | None = None
    transcript: str | None = None
    audio_base64: str | None = None
    audio_mime: str | None = None
    model_used: str | None = None


# ---------------------------------------------------------------------------
# Features 1–3 (carry-forward)
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse, tags=["F1 · Hello AI"])
@limiter.limit("60/minute")
async def chat(request: ChatRequest) -> ChatResponse:
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant for [YOUR_DOMAIN]. Answer clearly and concisely."},
        {"role": "user", "content": request.message},
    ]
    result = await call_llm(messages)
    return ChatResponse(response=result.content or "")


_STRUCTURED_SYSTEM_PROMPT = """You are a helpful AI assistant for [YOUR_DOMAIN].
Respond ONLY with JSON: {"intent": "...", "answer": "...", "confidence": 0.0, "sources_needed": false}"""


def _parse_structured(raw_text: str) -> StructuredResponse:
    try:
        return StructuredResponse(**json.loads(raw_text))
    except Exception:
        return StructuredResponse(intent="unclear", answer=raw_text or "", confidence=0.0, sources_needed=False)


@app.post("/api/chat/structured", response_model=StructuredResponse, tags=["F2 · Prompt Mastery"])
async def chat_structured(request: ChatRequest) -> StructuredResponse:
    result = await call_llm(
        [{"role": "system", "content": _STRUCTURED_SYSTEM_PROMPT}, {"role": "user", "content": request.message}],
        temperature=0.3, response_format={"type": "json_object"},
    )
    return _parse_structured(result.content or "")


@app.post("/api/sessions", tags=["F3 · AI Memory"])
async def new_session(tenant_id: str = Depends(get_tenant_id)) -> dict:
    return {"session_id": create_session(tenant_id=tenant_id)}


@app.get("/api/sessions", response_model=list[SessionSummary], tags=["F3 · AI Memory"])
async def sessions_list() -> list[SessionSummary]:
    summaries = []
    for s in list_sessions():
        first_user_msg = next((m.content for m in s.messages if m.role == "user"), "")
        title = (first_user_msg[:60] + "…") if len(first_user_msg) > 60 else (first_user_msg or "New conversation")
        summaries.append(SessionSummary(id=s.id, created_at=s.created_at.isoformat(), message_count=len(s.messages), title=title))
    return summaries


@app.post("/api/sessions/{session_id}/chat", response_model=StructuredResponse, tags=["F3 · AI Memory"])
async def session_chat(session_id: str, request: ChatRequest, tenant_id: str = Depends(get_tenant_id)) -> StructuredResponse:
    session = get_session(session_id, tenant_id=tenant_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    messages: list[dict] = [{"role": "system", "content": _STRUCTURED_SYSTEM_PROMPT}]
    for msg in session.messages[-CONTEXT_WINDOW_SIZE:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.message})
    add_message(session_id, "user", request.message)
    result = await call_llm(messages, temperature=0.3, response_format={"type": "json_object"})
    structured = _parse_structured(result.content or "")
    add_message(session_id, "assistant", structured.answer)
    return structured


@app.get("/api/sessions/{session_id}/history", response_model=list[Message], tags=["F3 · AI Memory"])
async def session_history(session_id: str, tenant_id: str = Depends(get_tenant_id)) -> list[Message]:
    session = get_session(session_id, tenant_id=tenant_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session.messages


# ---------------------------------------------------------------------------
# Feature 4: Document ingestion (carry-forward)
# ---------------------------------------------------------------------------

@app.post("/api/documents/upload", response_model=Document, tags=["F4 · Feed the Brain"])
async def upload_document(file: UploadFile = File(...), strategy: str = Form("sentence"), tenant_id: str = Depends(get_tenant_id)) -> Document:
    if strategy not in CHUNKING_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unknown strategy '{strategy}'.")
    filename = file.filename or "unknown"
    doc = save_document(filename, tenant_id=tenant_id)
    try:
        file_bytes = await file.read()
        text = extract_text(file_bytes, filename)
        pages = extract_pages(file_bytes, filename)
        chunk_dicts = CHUNKING_STRATEGIES[strategy](text, pages)
        for cd in chunk_dicts:
            save_chunk(Chunk(id=str(uuid.uuid4()), document_id=doc.id, text=cd["text"], chunk_index=cd["chunk_index"],
                             metadata={"filename": filename, "chunk_index": cd["chunk_index"]}))
        add_chunks(doc.id, [cd["text"] for cd in chunk_dicts],
                   [{"filename": filename, "chunk_index": cd["chunk_index"]} for cd in chunk_dicts], tenant_id=tenant_id)
        update_document(doc.id, status="ready", chunk_count=len(chunk_dicts), chunking_strategy=strategy)
    except Exception as exc:
        update_document(doc.id, status="error", chunk_count=0, chunking_strategy=strategy)
        raise HTTPException(status_code=422, detail=str(exc))
    return get_document(doc.id, tenant_id=tenant_id)  # type: ignore[return-value]


@app.get("/api/documents", response_model=list[Document], tags=["F4 · Feed the Brain"])
async def documents_list(tenant_id: str = Depends(get_tenant_id)) -> list[Document]:
    return list_documents(tenant_id=tenant_id)


@app.delete("/api/documents/{doc_id}", tags=["F4 · Feed the Brain"])
async def remove_document(doc_id: str, tenant_id: str = Depends(get_tenant_id)) -> dict:
    if get_document(doc_id, tenant_id=tenant_id) is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    delete_document_chunks(doc_id)
    delete_document(doc_id, tenant_id=tenant_id)
    return {"deleted": doc_id}


@app.get("/api/documents/{doc_id}/chunks", response_model=list[Chunk], tags=["F4 · Feed the Brain"])
async def document_chunks(doc_id: str, tenant_id: str = Depends(get_tenant_id)) -> list[Chunk]:
    if get_document(doc_id, tenant_id=tenant_id) is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return get_chunks(doc_id)


# ---------------------------------------------------------------------------
# Feature 5: Semantic search (carry-forward)
# ---------------------------------------------------------------------------

@app.post("/api/search", tags=["F5 · Find the Answer"])
async def search_documents(req: SearchRequest, tenant_id: str = Depends(get_tenant_id)) -> list[dict]:
    return vector_search(req.query, top_k=req.top_k, filters={"document_id": req.document_id} if req.document_id else None, tenant_id=tenant_id)


@app.get("/api/search/stats", tags=["F5 · Find the Answer"])
async def search_stats() -> dict:
    return vector_get_stats()


# ---------------------------------------------------------------------------
# Feature 6: Smart Router (carry-forward)
# ---------------------------------------------------------------------------

_SMART_SYSTEM_PROMPT = "You are a helpful AI assistant for [YOUR_DOMAIN]. Answer clearly and concisely in plain English."
_SMART_RAG_SYSTEM_PROMPT = "You are a helpful AI assistant for [YOUR_DOMAIN]. Use the provided document excerpts to answer accurately."
_SMART_HYBRID_SYSTEM_PROMPT = "You are a helpful AI assistant for [YOUR_DOMAIN]. Some excerpts have been retrieved — use them if helpful."


def _build_context_block(chunks: list[dict]) -> str:
    lines = ["--- RETRIEVED CONTEXT ---"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"\n[{i}] From: {chunk.get('filename', 'unknown')} (chunk {chunk.get('chunk_index', 0)})")
        lines.append(chunk.get("text", ""))
    lines.append("--- END CONTEXT ---")
    return "\n".join(lines)


@app.post("/api/sessions/{session_id}/chat/smart", response_model=SmartChatResponse, tags=["F6 · Smart Router"])
async def smart_chat(session_id: str, request: SmartChatRequest, tenant_id: str = Depends(get_tenant_id)) -> SmartChatResponse:
    from shared.config import settings
    session = get_session(session_id, tenant_id=tenant_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    classification = await classify_query(request.message)
    chunks_used: list[dict] = []
    source = "llm"
    retrieval_method = "none"
    system_prompt = _SMART_SYSTEM_PROMPT
    high_confidence = classification["confidence"] > 0.6
    if high_confidence and classification["needs_retrieval"]:
        chunks_used = vector_search(request.message, top_k=5, tenant_id=tenant_id)
        source = "rag"
        retrieval_method = "vector"
        system_prompt = _SMART_RAG_SYSTEM_PROMPT
    elif not high_confidence:
        chunks_used = vector_search(request.message, top_k=3, tenant_id=tenant_id)
        source = "hybrid"
        retrieval_method = "vector"
        system_prompt = _SMART_HYBRID_SYSTEM_PROMPT
    if settings.enable_long_term_context:
        digest = get_current_digest(tenant_id)
        if digest and digest.summary:
            system_prompt += f"\n\nContext about this user's history: {digest.summary}"
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if chunks_used:
        messages.append({"role": "system", "content": _build_context_block(chunks_used)})
    for msg in session.messages[-CONTEXT_WINDOW_SIZE:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.message})
    result = await call_llm(messages)
    answer = result.content or ""
    add_message(session_id, "user", request.message)
    add_message(session_id, "assistant", answer)
    if chunks_used and settings.enable_long_term_context:
        chunk_ids = [f"{c.get('document_id', '')}_{c.get('chunk_index', 0)}" for c in chunks_used]
        log_retrieval(session_id=session_id, tenant_id=tenant_id, query=request.message, chunk_ids=chunk_ids, retrieval_method=retrieval_method)
    return SmartChatResponse(answer=answer, source=source, chunks_used=chunks_used, confidence=classification["confidence"], retrieval_method=retrieval_method)  # type: ignore[arg-type]


@app.get("/api/tenant/info", tags=["F6 · Smart Router"])
async def tenant_info(tenant_id: str = Depends(get_tenant_id)) -> dict:
    from shared.config import settings
    return {"tenant_id": tenant_id, "multi_tenant_enabled": settings.enable_multi_tenant,
            "document_count": len(list_documents(tenant_id=tenant_id))}


@app.post("/api/retrieval-memory/rebuild", tags=["F6 · Smart Router"])
async def retrieval_memory_rebuild(tenant_id: str = Depends(get_tenant_id)) -> dict:
    digest = await build_knowledge_digest(tenant_id=tenant_id)
    if digest is None:
        return {"message": "No retrieval history found."}
    return {"summary": digest.summary, "topics_covered": digest.topics_covered,
            "source_session_count": digest.source_session_count, "last_updated": digest.last_updated.isoformat()}


@app.get("/api/retrieval-memory/digest", tags=["F6 · Smart Router"])
async def retrieval_memory_digest(tenant_id: str = Depends(get_tenant_id)) -> dict:
    digest = get_current_digest(tenant_id=tenant_id)
    if digest is None:
        return {"message": "No digest built yet."}
    return {"summary": digest.summary, "topics_covered": digest.topics_covered,
            "source_session_count": digest.source_session_count, "last_updated": digest.last_updated.isoformat()}


@app.get("/api/retrieval-memory/recent", tags=["F6 · Smart Router"])
async def retrieval_memory_recent(limit: int = 20, tenant_id: str = Depends(get_tenant_id)) -> list[dict]:
    return [{"session_id": e.session_id, "query": e.query, "timestamp": e.timestamp.isoformat(), "retrieval_method": e.retrieval_method}
            for e in get_recent_retrievals(tenant_id=tenant_id, limit=limit)]


# ---------------------------------------------------------------------------
# Feature 7: Agent (carry-forward)
# ---------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/agent/run", tags=["F7 · First Agent"])
@limiter.limit("20/minute")
async def agent_run(session_id: str, request: AgentRequest, tenant_id: str = Depends(get_tenant_id)) -> dict:
    session = get_session(session_id, tenant_id=tenant_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return await run_agent_with_mcp(message=request.message, session_id=session_id, tenant_id=tenant_id)


# ---------------------------------------------------------------------------
# Feature 8: Multi-step agent (carry-forward)
# ---------------------------------------------------------------------------

@app.post("/api/agent/plan", tags=["F8 · Multi-Step Agent"])
async def agent_plan(request: PlanRequest, background_tasks: BackgroundTasks, tenant_id: str = Depends(get_tenant_id)) -> dict:
    session_id = create_session(tenant_id=tenant_id)
    task = create_task(message=request.message, session_id=session_id, tenant_id=tenant_id)
    plan = await make_plan(request.message)
    update_task(task.id, plan=plan)
    background_tasks.add_task(execute_plan, task.id)
    return {"task_id": task.id, "session_id": session_id, "plan": plan}


@app.get("/api/agent/status/{task_id}", tags=["F8 · Multi-Step Agent"])
async def agent_status(task_id: str) -> dict:
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return task.model_dump()


# ---------------------------------------------------------------------------
# Feature 9: MCP (carry-forward)
# ---------------------------------------------------------------------------

@app.get("/api/mcp/servers", tags=["F9 · MCP Integration"])
async def mcp_servers() -> list[dict]:
    return [{"name": s["name"], "transport": s.get("transport", "stdio"),
             "enabled": s.get("enabled", False), "description": s.get("description", "")} for s in SERVER_REGISTRY]


@app.get("/api/mcp/tools", tags=["F9 · MCP Integration"])
async def mcp_tools_list() -> list[dict]:
    try:
        return await list_mcp_tools(use_cache=False)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not list MCP tools: {exc}")


@app.post("/api/mcp/execute", tags=["F9 · MCP Integration"])
async def mcp_execute(request: McpExecuteRequest) -> dict:
    try:
        result = await call_mcp_tool(request.tool_name, request.arguments)
        return {"tool_name": request.tool_name, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Feature 10: Voice + Vision (carry-forward)
# ---------------------------------------------------------------------------

@app.post("/api/voice/transcribe", tags=["F10 · Multimodal AI"])
async def voice_transcribe(audio: UploadFile = File(...)) -> dict:
    audio_bytes = await audio.read()
    text = await transcribe_audio(audio_bytes, audio.filename or "audio.webm")
    return {"text": text, "filename": audio.filename}


@app.post("/api/voice/chat", response_model=VoiceChatResponse, tags=["F10 · Multimodal AI"])
async def voice_chat(audio: UploadFile = File(...), session_id: str = Form(""), tenant_id: str = Depends(get_tenant_id)) -> VoiceChatResponse:
    audio_bytes = await audio.read()
    transcript = await transcribe_audio(audio_bytes, audio.filename or "audio.webm")
    if session_id:
        session = get_session(session_id, tenant_id=tenant_id)
        if session:
            smart_resp = await smart_chat(session_id, SmartChatRequest(message=transcript), tenant_id)
            answer = smart_resp.answer
        else:
            result = await call_llm([{"role": "system", "content": _SMART_SYSTEM_PROMPT}, {"role": "user", "content": transcript}])
            answer = result.content or ""
    else:
        result = await call_llm([{"role": "system", "content": _SMART_SYSTEM_PROMPT}, {"role": "user", "content": transcript}])
        answer = result.content or ""
    audio_base64 = ""
    try:
        audio_out = await synthesize_speech(answer)
        audio_base64 = base64.b64encode(audio_out).decode()
    except NotImplementedError:
        pass
    return VoiceChatResponse(transcript=transcript, answer=answer, audio_base64=audio_base64)


@app.post("/api/vision/analyze", response_model=VisionAnalyzeResponse, tags=["F10 · Multimodal AI"])
async def vision_analyze(image: UploadFile = File(...), prompt: str = Form("Describe this image."), detail: str = Form("auto")) -> VisionAnalyzeResponse:
    image_bytes = await image.read()
    result = await analyze_image(image_bytes, prompt, detail)
    return VisionAnalyzeResponse(answer=result.content or "", model_used=result.model)


@app.post("/api/vision/chat", response_model=VisionAnalyzeResponse, tags=["F10 · Multimodal AI"])
async def vision_chat(image: UploadFile = File(...), prompt: str = Form("Describe this image."), detail: str = Form("auto"),
                      session_id: str = Form(""), tenant_id: str = Depends(get_tenant_id)) -> VisionAnalyzeResponse:
    image_bytes = await image.read()
    result = await analyze_image(image_bytes, prompt, detail)
    answer = result.content or ""
    if session_id:
        session = get_session(session_id, tenant_id=tenant_id)
        if session:
            add_message(session_id, "user", f"[IMAGE] {prompt}")
            add_message(session_id, "assistant", answer)
    return VisionAnalyzeResponse(answer=answer, model_used=result.model)


@app.post("/api/chat/multimodal", response_model=MultimodalChatResponse, tags=["F10 · Multimodal AI"])
async def multimodal_chat(session_id: str = Form(...), message: str = Form(""), audio: UploadFile | None = File(None),
                          image: UploadFile | None = File(None), prompt: str = Form(""), tenant_id: str = Depends(get_tenant_id)) -> MultimodalChatResponse:
    has_audio = audio is not None and bool(audio.filename)
    has_image = image is not None and bool(image.filename)
    modality  = detect_modality(has_audio=has_audio, has_image=has_image)
    if modality == "voice":
        audio_bytes = await audio.read()  # type: ignore[union-attr]
        transcript  = await transcribe_audio(audio_bytes, audio.filename or "audio.webm")  # type: ignore[union-attr]
        smart_resp  = await smart_chat(session_id, SmartChatRequest(message=transcript), tenant_id)
        audio_base64 = ""
        try:
            audio_out = await synthesize_speech(smart_resp.answer)
            audio_base64 = base64.b64encode(audio_out).decode()
        except NotImplementedError:
            pass
        return MultimodalChatResponse(modality="voice", answer=smart_resp.answer, source=smart_resp.source,
                                      chunks_used=list(smart_resp.chunks_used), confidence=smart_resp.confidence,
                                      retrieval_method=smart_resp.retrieval_method, transcript=transcript,
                                      audio_base64=audio_base64, audio_mime="audio/mpeg")
    if modality == "vision":
        image_bytes  = await image.read()  # type: ignore[union-attr]
        text_prompt  = prompt or message or "Describe this image in detail."
        result       = await analyze_image(image_bytes, text_prompt)
        answer       = result.content or ""
        session      = get_session(session_id, tenant_id=tenant_id)
        if session:
            add_message(session_id, "user", f"[IMAGE] {text_prompt}")
            add_message(session_id, "assistant", answer)
        return MultimodalChatResponse(modality="vision", answer=answer, model_used=result.model)
    smart_resp = await smart_chat(session_id, SmartChatRequest(message=message), tenant_id)
    return MultimodalChatResponse(modality="text", answer=smart_resp.answer, source=smart_resp.source,
                                  chunks_used=list(smart_resp.chunks_used), confidence=smart_resp.confidence,
                                  retrieval_method=smart_resp.retrieval_method)


# ---------------------------------------------------------------------------
# Feature 11: Observability endpoints (Part A)
# ---------------------------------------------------------------------------

@app.get("/api/metrics", tags=["F11 · Production Design"])
async def get_metrics_endpoint() -> dict:
    return metrics.get_metrics()


# ---------------------------------------------------------------------------
# Feature 11: Eval harness (Part B)
# ---------------------------------------------------------------------------

@app.post("/api/eval/run", tags=["F11 · Production Design"])
async def eval_run() -> EvalReport:
    # =========================================================================
    # TODO 4: Implement the eval run logic.
    #
    # Steps:
    #   1. Check that _EVAL_CASES_PATH exists (raise 404 if not).
    #   2. Load and parse the JSON file into a list of EvalCase objects.
    #   3. Call run_eval(cases) — it's async, so await it.
    #   4. Store the report with metrics.set_eval_result(report.model_dump()).
    #   5. Return the report.
    #
    # Hint: json.loads(_EVAL_CASES_PATH.read_text()) gives you a list of dicts.
    # =========================================================================
    #raise HTTPException(status_code=501, detail="TODO 4: implement eval_run()")
    if not _EVAL_CASES_PATH.exists():
        raise HTTPException(status_code = 404, detail = f"Eval cases file not found: {_EVAL_CASES_PATH}")

    cases_data = json.loads(_EVAL_CASES_PATH.read_text())
    cases = [EvalCase(**c) for c in cases_data]

    report = await run_eval(cases)

    metrics.set_eval_result(report.model_dump())
    return report


@app.get("/api/eval/last", tags=["F11 · Production Design"])
async def eval_last() -> dict:
    data = metrics.get_metrics().get("eval_last_run")
    if data is None:
        raise HTTPException(status_code=404, detail="No eval has been run yet. POST /api/eval/run first.")
    return data


# ---------------------------------------------------------------------------
# Health + provider info
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["Infrastructure"])
async def health() -> dict:
    from shared.config import settings
    return {
        "status":   "ok",
        "version":  "11.0.0",
        "provider": settings.llm_provider,
        "metrics":  metrics.get_metrics(),
    }


@app.get("/api/provider-info", tags=["Infrastructure"])
async def provider_info() -> dict:
    from shared.config import settings
    llm_name = settings.llm_provider.lower().strip()
    model_map = {
        "openai": settings.openai_model, "anthropic": settings.anthropic_model,
        "cohere": settings.cohere_model, "ollama": settings.ollama_model,
        "groq": settings.groq_model, "azure": settings.azure_openai_deployment_name,
        "bedrock": settings.bedrock_model_id, "vertex": settings.vertex_model,
    }
    voice_name = settings.effective_voice_provider().lower().strip()
    vlm_name   = settings.effective_vlm_provider().lower().strip()
    return {
        "llm_provider":  llm_name,
        "llm_model":     model_map.get(llm_name, "unknown"),
        "voice_provider": voice_name if voice_name != llm_name else None,
        "voice_model":    model_map.get(voice_name) if voice_name != llm_name else None,
        "vlm_provider":   vlm_name if vlm_name != llm_name else None,
        "vlm_model":      settings.vlm_model if vlm_name != llm_name else None,
    }


_ui_path = Path(__file__).resolve().parents[2] / "ui"
if _ui_path.exists():
    app.mount("/", StaticFiles(directory=str(_ui_path), html=True), name="ui")
