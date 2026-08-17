"""
Feature 9: MCP Integration — starter

Your tasks for this feature:
  1. Open starter/mcp_client.py and implement list_mcp_tools() (STEP 1)
  2. Open starter/mcp_client.py and implement call_mcp_tool() (STEP 2)
  3. Restart this server and verify GET /api/mcp/tools returns the demo tools
  4. Add at least one real tool to shared/domain_mcp_server.py
  5. Set ENABLE_DOMAIN_MCP_SERVER=true in .env and restart — your tool appears

New endpoints in this feature:
  GET  /api/mcp/servers   — list connected MCP servers
  GET  /api/mcp/tools     — list all tools from all connected servers
  POST /api/mcp/execute   — directly invoke an MCP tool by name

The agent endpoint (/api/sessions/{id}/agent/run) already uses run_agent_with_mcp,
which merges local tools + MCP tools. Once you implement mcp_client.py, MCP tools
will appear in the agent's tool list automatically.

Run with:
    uvicorn main:app --reload --port 8000

IMPORTANT: This starter carries forward ALL working code from Feature 8.
Only the MCP-specific functions in starter/mcp_client.py are stubbed.
"""
import json
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Feature 9: import run_agent_with_mcp (falls back to run_agent if MCP not configured)
from shared.agent import run_agent_with_mcp
from shared.document_store import (
    delete_document, get_chunks, get_document, list_documents,
    save_chunk, save_document, update_document,
)
from shared.ingestion import CHUNKING_STRATEGIES, extract_pages, extract_text
from shared.llm_client import call_llm
from shared.models import (
    AgentTask, Chunk, Document, Message, SmartChatResponse, StructuredResponse,
)
from shared.planner import execute_plan, make_plan
from shared.provider_check import check_provider_config
from shared.retrieval_memory import (
    build_knowledge_digest, get_current_digest, get_recent_retrievals, log_retrieval,
)
from shared.router import classify_query
from shared.session_store import add_message, create_session, get_session, list_sessions
from shared.task_store import create_task, get_task, update_task
from shared.tenant_context import get_tenant_id
from shared.vector_store import (
    add_chunks, delete_document_chunks, get_stats as vector_get_stats, search as vector_search,
)

# Feature 9: import mcp_client from this starter's local copy.
# Once you implement list_mcp_tools() and call_mcp_tool() in mcp_client.py,
# these imports will work. Until then, NotImplementedError is caught below.
try:
    from mcp_client import SERVER_REGISTRY, call_mcp_tool, list_mcp_tools
    _MCP_CLIENT_AVAILABLE = True
except ImportError:
    _MCP_CLIENT_AVAILABLE = False
    SERVER_REGISTRY = []
    list_mcp_tools = None
    call_mcp_tool = None

CONTEXT_WINDOW_SIZE = 20

OPENAPI_TAGS = [
    {"name": "F1 · Hello AI",         "description": "**Week 1 · Feature 1** — Basic text chat."},
    {"name": "F2 · Prompt Mastery",    "description": "**Week 1 · Feature 2** — Structured JSON-mode responses."},
    {"name": "F3 · AI Memory",         "description": "**Week 1 · Feature 3** — Session management and conversation history."},
    {"name": "F4 · Feed the Brain",    "description": "**Week 2 · Feature 4** — Document upload, chunking, and indexing."},
    {"name": "F5 · Find the Answer",   "description": "**Week 2 · Feature 5** — Semantic search over document chunks."},
    {"name": "F6 · Smart Router",      "description": "**Week 2 · Feature 6** — Intelligent routing between LLM and RAG. Includes tenant isolation (Part B) and retrieval memory (Part C)."},
    {"name": "F7 · First Agent",       "description": "**Week 3 · Feature 7** — Tool-calling agent (ReAct pattern)."},
    {"name": "F8 · Multi-Step Agent",  "description": "**Week 3 · Feature 8** — Plan-and-Execute agent with background execution and polling."},
    {"name": "F9 · MCP Integration",   "description": "**Week 3 · Feature 9** — MCP tool servers. Implement mcp_client.py to unlock these endpoints."},
    {"name": "Infrastructure",         "description": "Health check and active provider info."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    await check_provider_config()
    yield


app = FastAPI(
    title="My AI BlockSeBlock Assistant",
    description=(
        "Domain-Specific AI Assistant — AI Engineering Bootcamp, BlockseBlock\n\n"
        "Endpoints are grouped by **course feature**. "
        "F9 · MCP Integration endpoints return HTTP 501 until you implement `mcp_client.py`."
    ),
    version="9.0.0",
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
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

class McpExecuteRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Features 1–8: Carry-forward (complete, no TODOs)
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse, tags=["F1 · Hello AI"])
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


@app.post("/api/search", tags=["F5 · Find the Answer"])
async def search_documents(req: SearchRequest, tenant_id: str = Depends(get_tenant_id)) -> list[dict]:
    return vector_search(req.query, top_k=req.top_k, filters={"document_id": req.document_id} if req.document_id else None, tenant_id=tenant_id)


@app.get("/api/search/stats", tags=["F5 · Find the Answer"])
async def search_stats() -> dict:
    return vector_get_stats()


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


@app.post("/api/sessions/{session_id}/agent/run", tags=["F7 · First Agent"])
async def agent_run(session_id: str, request: AgentRequest, tenant_id: str = Depends(get_tenant_id)) -> dict:
    session = get_session(session_id, tenant_id=tenant_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return await run_agent_with_mcp(message=request.message, session_id=session_id, tenant_id=tenant_id)


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
# Feature 9: MCP endpoints
# ---------------------------------------------------------------------------

@app.get("/api/mcp/servers", tags=["F9 · MCP Integration"])
async def mcp_servers() -> list[dict]:
    """List connected MCP servers with name, transport, and enabled status."""
    return [
        {
            "name":        s["name"],
            "transport":   s.get("transport", "stdio"),
            "enabled":     s.get("enabled", False),
            "description": s.get("description", ""),
        }
        for s in SERVER_REGISTRY
    ]


@app.get("/api/mcp/tools", tags=["F9 · MCP Integration"])
async def mcp_tools_list() -> list[dict]:
    """
    List all tools available from connected MCP servers.

    Returns HTTP 501 until you implement list_mcp_tools() in mcp_client.py
    (TODO STEP 1). Once implemented, this returns the demo server's tools.
    """
    if list_mcp_tools is None:
        raise HTTPException(status_code=503, detail="MCP client not available — check mcp_client.py import.")
    try:
        tools = await list_mcp_tools(use_cache=False)
        return tools
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=f"Not implemented: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not list MCP tools: {exc}")


@app.post("/api/mcp/execute", tags=["F9 · MCP Integration"])
async def mcp_execute(request: McpExecuteRequest) -> dict:
    """
    Directly invoke an MCP tool by name.

    Returns HTTP 501 until you implement call_mcp_tool() in mcp_client.py
    (TODO STEP 2). Once implemented, you can test tools from the UI or curl.
    """
    if call_mcp_tool is None:
        raise HTTPException(status_code=503, detail="MCP client not available — check mcp_client.py import.")
    try:
        result = await call_mcp_tool(request.tool_name, request.arguments)
        return {"tool_name": request.tool_name, "result": result}
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=f"Not implemented: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Health + provider info
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["Infrastructure"])
async def health():
    return {"status": "ok"}


@app.get("/api/provider-info", tags=["Infrastructure"])
async def provider_info():
    from shared.config import settings
    llm_name = settings.llm_provider.lower().strip()
    model_map = {
        "openai": settings.openai_model, "anthropic": settings.anthropic_model,
        "cohere": settings.cohere_model, "ollama": settings.ollama_model,
        "groq": settings.groq_model, "azure": settings.azure_openai_deployment_name,
        "bedrock": settings.bedrock_model_id, "vertex": settings.vertex_model,
    }
    voice_name = settings.effective_voice_provider().lower().strip()
    return {
        "llm_provider": llm_name,
        "llm_model": model_map.get(llm_name, "unknown"),
        "voice_provider": voice_name if voice_name != llm_name else None,
        "voice_model": model_map.get(voice_name) if voice_name != llm_name else None,
    }


_ui_path = Path(__file__).resolve().parents[2] / "ui"
if _ui_path.exists():
    app.mount("/", StaticFiles(directory=str(_ui_path), html=True), name="ui")
