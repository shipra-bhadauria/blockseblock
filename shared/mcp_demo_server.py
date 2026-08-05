"""
MCP Demo Server for Feature 9: MCP Integration.

This is a LOCAL MCP server exposing 3 mock tools over stdio.
We built a local server rather than depending on an external one because:
  - Zero external infrastructure required (no API keys, no signup)
  - Students can inspect and modify the server side by side with the client
  - Demonstrates the SAME pattern used to build production MCP servers

Run standalone:
    python -m shared.mcp_demo_server

The client (shared/mcp_client.py) connects to it via stdio transport —
launching this as a subprocess and communicating via stdin/stdout.

TOOLS EXPOSED:
  get_weather(location: str) → mock weather data
  search_news(query: str, max_results: int) → mock news headlines
  get_exchange_rate(from_currency: str, to_currency: str) → mock FX rate

These all return mock data. In a real integration you would replace the
mock return values with actual API calls (OpenWeatherMap, NewsAPI, etc.).

MCP SDK NOTE:
  This server uses the official Python MCP SDK (package: "mcp").
  The SDK handles the JSON-RPC protocol layer — you just define tools
  with the @server.tool() decorator and return results as dicts or strings.

  The same server code works with ANY MCP-compatible client:
    - Our hand-rolled client (shared/mcp_client.py)
    - Claude Desktop
    - Cursor
    - Google ADK's MCPToolset
    - LangChain MCP adapters
  This is the entire point of the MCP standard.
"""
import asyncio
import json
import random

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types as mcp_types
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

# =============================================================================
# Tool implementations — replace mock data with real API calls in production
# =============================================================================

def _get_weather(location: str) -> dict:
    conditions = ["sunny", "partly cloudy", "overcast", "light rain", "clear"]
    return {
        "location": location,
        "temperature_c": round(random.uniform(10, 32), 1),
        "condition": random.choice(conditions),
        "humidity_pct": random.randint(40, 85),
        "source": "mock — replace with OpenWeatherMap API in production",
    }


def _search_news(query: str, max_results: int = 3) -> dict:
    headlines = [
        f"Breaking: Major developments in {query} sector",
        f"Analysis: What the latest {query} news means for markets",
        f"Report: {query} trends show surprising reversal",
        f"Opinion: The hidden story behind {query}",
        f"Update: Experts weigh in on {query} situation",
    ]
    return {
        "query": query,
        "results": [
            {"headline": h, "source": "Mock News Agency", "published_at": "2026-07-01T12:00:00Z"}
            for h in headlines[:max(1, min(max_results, len(headlines)))]
        ],
        "source": "mock — replace with NewsAPI or similar in production",
    }


def _get_exchange_rate(from_currency: str, to_currency: str) -> dict:
    base_rates = {
        ("USD", "EUR"): 0.92, ("EUR", "USD"): 1.09,
        ("USD", "GBP"): 0.79, ("GBP", "USD"): 1.27,
        ("USD", "JPY"): 157.2, ("JPY", "USD"): 0.0064,
        ("USD", "INR"): 83.5, ("INR", "USD"): 0.012,
    }
    rate = base_rates.get((from_currency.upper(), to_currency.upper()))
    if rate is None:
        rate = round(random.uniform(0.5, 2.0), 4)
    return {
        "from": from_currency.upper(),
        "to": to_currency.upper(),
        "rate": rate,
        "timestamp": "2026-07-01T12:00:00Z",
        "source": "mock — replace with a real FX API (e.g., exchangerate-api.com) in production",
    }


# =============================================================================
# MCP Server definition
# =============================================================================

if _MCP_AVAILABLE:
    server = Server("bsb-demo-mcp-server")

    @server.list_tools()
    async def list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name="get_weather",
                description="Get the current weather for a location. Returns temperature, condition, and humidity.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name or location (e.g. 'London', 'New York')"},
                    },
                    "required": ["location"],
                },
            ),
            mcp_types.Tool(
                name="search_news",
                description="Search for recent news headlines about a topic.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (e.g. 'AI regulation', 'climate change')"},
                        "max_results": {"type": "integer", "description": "Maximum number of results (1-5, default 3)", "default": 3},
                    },
                    "required": ["query"],
                },
            ),
            mcp_types.Tool(
                name="get_exchange_rate",
                description="Get the current exchange rate between two currencies.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "from_currency": {"type": "string", "description": "Source currency code (e.g. 'USD', 'EUR', 'GBP')"},
                        "to_currency":   {"type": "string", "description": "Target currency code (e.g. 'USD', 'EUR', 'GBP')"},
                    },
                    "required": ["from_currency", "to_currency"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
        if name == "get_weather":
            result = _get_weather(arguments["location"])
        elif name == "search_news":
            result = _search_news(arguments["query"], arguments.get("max_results", 3))
        elif name == "get_exchange_rate":
            result = _get_exchange_rate(arguments["from_currency"], arguments["to_currency"])
        else:
            result = {"error": f"Unknown tool '{name}'"}
        return [mcp_types.TextContent(type="text", text=json.dumps(result, indent=2))]

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

else:
    async def main():
        raise RuntimeError(
            "The 'mcp' package is not installed. Run: pip install mcp\n"
            "See requirements.txt for the full dependency list."
        )


if __name__ == "__main__":
    asyncio.run(main())
