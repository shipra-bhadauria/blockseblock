"""
MCP Client — Feature 9 starter stub.

Your task: implement list_mcp_tools() and call_mcp_tool() so the agent can
discover and call tools from connected MCP servers.

WHAT MCP IS:
  MCP (Model Context Protocol) is a standard for connecting AI applications
  to tools and data sources. Any MCP-compatible client can call any
  MCP-compatible server — the protocol handles the handshake.

  The Python MCP SDK (package: "mcp") provides:
    - StdioServerParameters: config for a server started as a subprocess
    - stdio_client(params): context manager that opens stdin/stdout streams
    - ClientSession(read, write): manages the client-server session
    - session.initialize(): performs the MCP handshake
    - session.list_tools(): returns the server's tool list
    - session.call_tool(name, arguments): invokes a tool on the server

TRANSPORTS:
  stdio — the server runs as a subprocess; we communicate via stdin/stdout.
          This is what we use in this course (simplest, no network required).
  SSE   — the server is a persistent HTTP service. For remote/shared servers.
  HTTP  — newer bidirectional transport for production deployments.

HOW TOOLS FLOW:
  MCP server (shared/mcp_demo_server.py)
      ↓ stdio
  mcp_client.py (this file) → list_mcp_tools(), call_mcp_tool()
      ↓
  shared/agent.py run_agent_with_mcp()
      ↓ merged tool list
  LLM (sees local + MCP tools as one list)
      ↓ tool_calls
  back here: call_mcp_tool()
      ↓
  MCP server executes the tool and returns result

GOOGLE ADK EQUIVALENT:
  from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
  mcp_tools = MCPToolset(connection_params=StdioServerParameters(
      command="python", args=["-m", "shared.mcp_demo_server"]
  ))
  # ADK's MCPToolset does what this file does — 4 lines vs ~100.
  # Now you know what those 4 lines are doing.
"""
import asyncio
import json
import os
import sys
from typing import Any, Optional

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None

# Server registry — same as shared/mcp_client.py
def _build_server_registry() -> list[dict]:
    demo_server_cmd = [sys.executable, "-m", "shared.mcp_demo_server"]
    registry = [
        {"name": "demo", "command": demo_server_cmd, "enabled": True, "transport": "stdio",
         "description": "Course demo server — weather, news, exchange rate (all mock data)"},
    ]
    if os.getenv("ENABLE_DOMAIN_MCP_SERVER", "false").lower() == "true":
        registry.append({
            "name": "domain", "command": [sys.executable, "-m", "shared.domain_mcp_server"],
            "enabled": True, "transport": "stdio",
            "description": "Your domain MCP server",
        })
    return registry

SERVER_REGISTRY: list[dict] = _build_server_registry()
_tool_cache: dict[str, list[dict]] = {}


async def _list_tools_from_server(server_entry: dict) -> list[dict]:
    """Connect to a server subprocess, list its tools, and disconnect. (Given — no TODO.)"""
    if not _MCP_AVAILABLE:
        raise ImportError("Run: pip install mcp")
    params = StdioServerParameters(command=server_entry["command"][0], args=server_entry["command"][1:])
    tools: list[dict] = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            for tool in result.tools:
                tools.append({"name": tool.name, "description": tool.description or "",
                               "inputSchema": tool.inputSchema or {}, "server": server_entry["name"]})
    return tools


async def list_mcp_tools(use_cache: bool = True) -> list[dict]:
    """
    Return the combined list of tools from all enabled MCP servers.

    =========================================================================
    TODO STEP 1: Implement this function.

    For each entry in SERVER_REGISTRY where entry["enabled"] is True:

      a. Check the cache:
         if use_cache and entry["name"] in _tool_cache:
             all_tools.extend(_tool_cache[entry["name"]])
             continue

      b. Call: tools = await _list_tools_from_server(entry)
         Cache: _tool_cache[entry["name"]] = tools
         Extend: all_tools.extend(tools)

      c. Wrap each server in try/except Exception as exc:
         On failure: print a warning to stderr and continue.
         (One unreachable server shouldn't crash the whole list.)

    Return all_tools.
    =========================================================================
    """
    if not _MCP_AVAILABLE:
        return []
    #raise NotImplementedError(
     #   "TODO STEP 1: Implement list_mcp_tools(). See the comments above."
    #)

    all_tools: list[dict] = []

    for entry in SERVER_REGISTRY:
        if not entry["enabled"]:
            continue
        try:
            if use_cache and entry["name"] in _tool_cache:
                all_tools.extend(_tool_cache[entry["name"]])
                continue

            tools = await _list_tools_from_server(entry)
            _tool_cache[entry["name"]] = tools
            all_tools.extend(tools)

        except Exception as exc:
            print(f"Warning: Could not list tools from '{entry['name']}: {exc}", file=sys.stderr)
            continue

    return all_tools


async def _call_tool_on_server(server_entry: dict, tool_name: str, arguments: dict) -> dict:
    """Connect to a server, call a tool, and return the JSON result. (Given — no TODO.)"""
    if not _MCP_AVAILABLE:
        raise ImportError("Run: pip install mcp")
    params = StdioServerParameters(command=server_entry["command"][0], args=server_entry["command"][1:])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            if result.content:
                raw = result.content[0].text
                try:
                    return json.loads(raw)
                except (json.JSONDecodeError, AttributeError):
                    return {"result": str(raw)}
            return {"result": None}


async def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """
    Invoke a named MCP tool and return its result.

    =========================================================================
    TODO STEP 2: Implement this function.

    1. If _tool_cache is empty, populate it:
       if not _tool_cache:
           await list_mcp_tools()

    2. Find which server owns this tool:
       owner_name = None
       for server_name, tools in _tool_cache.items():
           if any(t["name"] == tool_name for t in tools):
               owner_name = server_name
               break

    3. If owner_name is None:
       return {"error": f"Tool '{tool_name}' not found in any connected MCP server."}

    4. Find the server_entry in SERVER_REGISTRY:
       server_entry = next((s for s in SERVER_REGISTRY if s["name"] == owner_name), None)
       if server_entry is None:
           return {"error": f"Server '{owner_name}' not found in registry."}

    5. Call and return:
       try:
           return await _call_tool_on_server(server_entry, tool_name, arguments)
       except Exception as exc:
           return {"error": str(exc)}
    =========================================================================
    """
    #raise NotImplementedError(
     #   "TODO STEP 2: Implement call_mcp_tool(). See the comments above."
    #)

    async def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
        if not _tool_cache:
            await list_mcp_tools()

        owner_name = None
        for server_name, tools in _tool_cache.items():
            if any(t["name"] == tool_name for t in tools):
                owner_name = server_name
                break

        if owner_name is None:
            return {"error": f"Tool '{tool_name}' not found in any connected MCP Server."}

        server_entry = next((s for s in SERVER_REGISTRY if s["name"] == owner_name), None)
        if server_entry is None:
            return {"error": f"Server '{owner_name}' not found in registry." }

        try:
            return await _call_tool_on_server(server_entry, tool_name, arguments)
        except Exception as exc:
            return {"error": str(exc)}


async def get_mcp_tool_schemas() -> list[dict]:
    """Return OpenAI-format schemas for all MCP tools. (Given — uses list_mcp_tools.)"""
    tools = await list_mcp_tools()
    schemas = []
    for tool in tools:
        schemas.append({
            "type": "function",
            "function": {
                "name":        tool["name"],
                "description": f"[MCP:{tool['server']}] {tool['description']}",
                "parameters":  tool.get("inputSchema", {"type": "object", "properties": {}}),
            },
        })
    return schemas
