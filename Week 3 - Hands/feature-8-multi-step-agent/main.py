"""
Feature 8: Multi-Step Agent — starter

Run with:
    cd week-3-hands/feature-8-multi-step-agent/starter
    uvicorn main:app --reload --port 8000

The two new endpoints in this feature:
  POST /api/agent/plan              — creates a task, plans, starts background execution
  GET  /api/agent/status/{task_id} — polls task progress

Your task: implement make_plan() and execute_plan() in planner.py.
The endpoints below are mostly complete — they wire up your functions.
"""
import json
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Resolve shared/ from the repo root (3 levels up from starter/).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import planner from the LOCAL starter directory, not shared/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from planner import execute_plan, make_plan

from shared.agent import run_agent
from shared.document_store import (
    delete_document,
    get_chunks,
    get_document,
    list_documents,
    save_chunk,
    save_document,
    update_document,
)
from shared.ingestion import CHUNKING_STRATEGIES, extract_pages, extract_text
from shared.llm_client import call_llm
from shared.models import (
    AgentTask,
    Chunk,
    Document,
    Message,
    SmartChatResponse,
    StructuredResponse,
)
from shared.provider_check import check_provider_config
from shared.retrieval_memory import (
    build_knowledge_digest,
    get_current_digest,
    get_recent_retrievals,
    log_retrieval,
)
from shared.router import classify_query
from shared.session_store import add_message, create_session, get_session, list_sessions
from shared.task_store import create_task, get_task, update_task
from shared.tenant_context import get_tenant_id
from shared.vector_store import (
    add_chunks,
    delete_document_chunks,
    get_stats as vector_get_stats,
    search as vector_search,
)

CONTEXT_WINDOW_SIZE = 20


@asynccontextmanager
async def lifespan(app: FastAPI):
    await check_provider_config()
    yield


app = FastAPI(
    title="My AI BlockSeBlock Assistant",
    description="Domain-Specific AI Assistant — AI Engineering Bootcamp, BlockseBlock",
    version="8.0.0-starter",
    lifespan=lifespan,
)


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


# ---------------------------------------------------------------------------
# Features 1–7: Carry-forward (unchanged from Feature 7 solution)
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant for [YOUR_DOMAIN]. Answer clearly and concisely."},
        {"role": "user", "content": request.message},
    ]
    result = await call_llm(messages)
    return ChatResponse(response=result.content or "")


_STRUCTURED_SYSTEM_PROMPT = """You are a helpful AI assistant for [YOUR_DOMAIN].
Respond ONLY with a JSON object with fields: intent, answer, confidence, sources_needed."""


def _parse_structured(raw_text: str) -> StructuredResponse:
    try:
        return StructuredResponse(**json.loads(raw_text))
    except Exception:
        return StructuredResponse(intent="unclear", answer=raw_text or "", confidence=0.0, sources_needed=False)


@app.post("/api/chat/structured", response_model=StructuredResponse)
async def chat_structured(request: ChatRequest) -> StructuredResponse:
    result = await call_llm(
        [{"role": "system", "content": _STRUCTURED_SYSTEM_PROMPT}, {"role": "user", "content": request.message}],
        temperature=0.3, response_format={"type": "json_object"},
    )
    return _parse_structured(result.content or "")


@app.post("/api/sessions")
async def new_session(tenant_id: str = Depends(get_tenant_id)) -> dict:
    return {"session_id": create_session(tenant_id=tenant_id)}


@app.get("/api/sessions", response_model=list[SessionSummary])
async def sessions_list() -> list[SessionSummary]:
    summaries = []
    for s in list_sessions():
        first_user_msg = next((m.content for m in s.messages if m.role == "user"), "")
        title = (first_user_msg[:60] + "…") if len(first_user_msg) > 60 else (first_user_msg or "New conversation")
        summaries.append(SessionSummary(id=s.id, created_at=s.created_at.isoformat(),
                                         message_count=len(s.messages), title=title))
    return summaries


@app.post("/api/sessions/{session_id}/chat", response_model=StructuredResponse)
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


@app.get("/api/sessions/{session_id}/history", response_model=list[Message])
async def session_history(session_id: str, tenant_id: str = Depends(get_tenant_id)) -> list[Message]:
    session = get_session(session_id, tenant_id=tenant_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session.messages


@app.post("/api/documents/upload", response_model=Document)
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
            save_chunk(Chunk(id=str(uuid.uuid4()), document_id=doc.id, text=cd["text"],
                             chunk_index=cd["chunk_index"], metadata={"filename": filename, "chunk_index": cd["chunk_index"]}))
        add_chunks(doc.id, [cd["text"] for cd in chunk_dicts],
                   [{"filename": filename, "chunk_index": cd["chunk_index"]} for cd in chunk_dicts], tenant_id=tenant_id)
        update_document(doc.id, status="ready", chunk_count=len(chunk_dicts), chunking_strategy=strategy)
    except Exception as exc:
        update_document(doc.id, status="error", chunk_count=0, chunking_strategy=strategy)
        raise HTTPException(status_code=422, detail=str(exc))
    return get_document(doc.id, tenant_id=tenant_id)  # type: ignore[return-value]


@app.get("/api/documents", response_model=list[Document])
async def documents_list(tenant_id: str = Depends(get_tenant_id)) -> list[Document]:
    return list_documents(tenant_id=tenant_id)


@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: str, tenant_id: str = Depends(get_tenant_id)) -> dict:
    if get_document(doc_id, tenant_id=tenant_id) is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    delete_document_chunks(doc_id)
    delete_document(doc_id, tenant_id=tenant_id)
    return {"deleted": doc_id}


@app.get("/api/documents/{doc_id}/chunks", response_model=list[Chunk])
async def document_chunks(doc_id: str, tenant_id: str = Depends(get_tenant_id)) -> list[Chunk]:
    if get_document(doc_id, tenant_id=tenant_id) is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return get_chunks(doc_id)


@app.post("/api/search")
async def search_documents(req: SearchRequest, tenant_id: str = Depends(get_tenant_id)) -> list[dict]:
    return vector_search(req.query, top_k=req.top_k, filters={"document_id": req.document_id} if req.document_id else None, tenant_id=tenant_id)


@app.get("/api/search/stats")
async def search_stats() -> dict:
    return vector_get_stats()


@app.post("/api/sessions/{session_id}/chat/smart", response_model=SmartChatResponse)
async def smart_chat(session_id: str, request: SmartChatRequest, tenant_id: str = Depends(get_tenant_id)) -> SmartChatResponse:
    from shared.config import settings
    session = get_session(session_id, tenant_id=tenant_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    classification = await classify_query(request.message)
    chunks_used: list[dict] = []
    source = "llm"
    retrieval_method = "none"
    system_prompt = "You are a helpful AI assistant for [YOUR_DOMAIN]."
    if classification["confidence"] > 0.6 and classification["needs_retrieval"]:
        chunks_used = vector_search(request.message, top_k=5, tenant_id=tenant_id)
        source = "rag"
        retrieval_method = "vector"
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for msg in session.messages[-CONTEXT_WINDOW_SIZE:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.message})
    result = await call_llm(messages)
    answer = result.content or ""
    add_message(session_id, "user", request.message)
    add_message(session_id, "assistant", answer)
    return SmartChatResponse(answer=answer, source=source, chunks_used=chunks_used,  # type: ignore[arg-type]
                              confidence=classification["confidence"], retrieval_method=retrieval_method)


@app.get("/api/tenant/info")
async def tenant_info(tenant_id: str = Depends(get_tenant_id)) -> dict:
    from shared.config import settings
    return {"tenant_id": tenant_id, "multi_tenant_enabled": settings.enable_multi_tenant,
            "document_count": len(list_documents(tenant_id=tenant_id))}


@app.post("/api/retrieval-memory/rebuild")
async def retrieval_memory_rebuild(tenant_id: str = Depends(get_tenant_id)) -> dict:
    digest = await build_knowledge_digest(tenant_id=tenant_id)
    if digest is None:
        return {"message": "No retrieval history found."}
    return {"summary": digest.summary, "topics_covered": digest.topics_covered}


@app.get("/api/retrieval-memory/digest")
async def retrieval_memory_digest(tenant_id: str = Depends(get_tenant_id)) -> dict:
    digest = get_current_digest(tenant_id=tenant_id)
    if digest is None:
        return {"message": "No digest built yet."}
    return {"summary": digest.summary, "topics_covered": digest.topics_covered}


@app.get("/api/retrieval-memory/recent")
async def retrieval_memory_recent(limit: int = 20, tenant_id: str = Depends(get_tenant_id)) -> list[dict]:
    return [{"session_id": e.session_id, "query": e.query, "timestamp": e.timestamp.isoformat()}
            for e in get_recent_retrievals(tenant_id=tenant_id, limit=limit)]


@app.post("/api/sessions/{session_id}/agent/run")
async def agent_run(session_id: str, request: AgentRequest, tenant_id: str = Depends(get_tenant_id)) -> dict:
    session = get_session(session_id, tenant_id=tenant_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return await run_agent(message=request.message, session_id=session_id, tenant_id=tenant_id)


# ---------------------------------------------------------------------------
# Feature 8: Multi-step agent — YOUR TODO IS IN planner.py
# ---------------------------------------------------------------------------

@app.post("/api/agent/plan")
async def agent_plan(
    request: PlanRequest,
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    session_id = create_session(tenant_id=tenant_id)
    task = create_task(message=request.message, session_id=session_id, tenant_id=tenant_id)

    try:
        plan = await make_plan(request.message)
        update_task(task.id, plan=plan)
        background_tasks.add_task(execute_plan, task.id)
        return {"task_id": task.id, "session_id": session_id, "plan": plan}
    except NotImplementedError:
        raise HTTPException(
            status_code=501,
            detail="make_plan() is not implemented yet. Open starter/planner.py and complete TODO STEP 1 and STEP 2.",
        )


@app.get("/api/agent/status/{task_id}")
async def agent_status(task_id: str) -> dict:
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return task.model_dump()


# ---------------------------------------------------------------------------
# Health + provider info
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/provider-info")
async def provider_info():
    from shared.config import settings
    llm_name = settings.llm_provider.lower().strip()
    return {"llm_provider": llm_name, "llm_model": getattr(settings, f"{llm_name}_model", "unknown")}


_ui_path = Path(__file__).resolve().parents[2] / "ui"
if _ui_path.exists():
    app.mount("/", StaticFiles(directory=str(_ui_path), html=True), name="ui")
