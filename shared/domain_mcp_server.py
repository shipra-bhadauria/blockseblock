"""
Domain MCP Server template for Feature 9: MCP Integration.

This is YOUR MCP server for YOUR domain. Replace the two placeholder tools
(search_domain_records, get_domain_status) with real ones — database lookups,
API calls, file system access — and your agent gains domain-specific capabilities
it can share with ANY MCP-compatible client, including Claude Desktop, Cursor,
Google ADK, and other AI tools.

Run standalone:
    python -m shared.domain_mcp_server

Enable via .env:
    ENABLE_DOMAIN_MCP_SERVER=true

WHAT TO REPLACE:
  1. The tool names and descriptions — make them meaningful for your domain.
     E.g., "search_domain_records" → "search_customer_orders" for e-commerce,
     or "search_patient_records" for healthcare.
  2. The inputSchema for each tool — match the fields your domain needs.
  3. The tool implementations (marked with # TODO) — call your actual data source
     instead of returning mock data.

EXAMPLE REPLACEMENTS BY DOMAIN:

  Healthcare:
    search_domain_records → search_patient_appointments(patient_id, date_range)
    get_domain_status     → get_doctor_availability(doctor_id, date)

  E-commerce:
    search_domain_records → search_orders(customer_email, status)
    get_domain_status     → get_inventory_level(product_sku)

  HR / Internal tools:
    search_domain_records → search_employee_records(department, role)
    get_domain_status     → get_policy_update(policy_name)

Once you have real tools here, your agent (Feature 9) will use them alongside
the Feature 7 local tools and the Feature 9 demo tools — all in the same
TOOLS_REGISTRY, all available to the LLM without it knowing which are "local"
vs "MCP".
"""
import asyncio
import json

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types as mcp_types
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

# =============================================================================
# Domain tool implementations
# Replace these stubs with real logic for your domain.
# =============================================================================

def _search_domain_records(query: str, limit: int = 5) -> dict:
    # TODO: Replace with a real database query, API call, or file search.
    # Example for e-commerce: look up orders matching the query string.
    # Example for HR: search the employee directory.
    return {
        "query": query,
        "results": [
            {"id": f"record-{i+1}", "title": f"[YOUR_DOMAIN] record matching '{query}' #{i+1}", "summary": "Replace this with real data from your domain."}
            for i in range(min(limit, 3))
        ],
        "total": min(limit, 3),
        "note": "This is mock data. Replace _search_domain_records() with your real implementation.",
    }


def _get_domain_status(entity_id: str) -> dict:
    # TODO: Replace with a real status lookup for your domain.
    # Example for e-commerce: look up order status by order_id.
    # Example for healthcare: look up appointment status by appointment_id.
    return {
        "entity_id": entity_id,
        "status": "active",
        "last_updated": "2026-07-01T12:00:00Z",
        "details": f"Status of {entity_id} in [YOUR_DOMAIN]. Replace with real lookup.",
        "note": "This is mock data. Replace _get_domain_status() with your real implementation.",
    }


# =============================================================================
# MCP Server definition
# =============================================================================

if _MCP_AVAILABLE:
    server = Server("bsb-domain-mcp-server")

    @server.list_tools()
    async def list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                # TODO: Replace name, description, and inputSchema for your domain.
                name="search_domain_records",
                description=(
                    "Search [YOUR_DOMAIN] records matching a query. "
                    "Replace this description with what this tool actually searches — "
                    "e.g., 'Search customer orders by keyword or customer email.'"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            # TODO: Update description for your domain.
                            "description": "Search query — e.g. customer name, order ID, keyword",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            mcp_types.Tool(
                # TODO: Replace name, description, and inputSchema for your domain.
                name="get_domain_status",
                description=(
                    "Get the current status of a [YOUR_DOMAIN] entity by ID. "
                    "Replace this description with what entity this looks up — "
                    "e.g., 'Get the current status of an order by order ID.'"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            # TODO: Update description for your domain.
                            "description": "The ID of the entity to look up (order ID, appointment ID, employee ID, etc.)",
                        },
                    },
                    "required": ["entity_id"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
        if name == "search_domain_records":
            result = _search_domain_records(arguments["query"], arguments.get("limit", 5))
        elif name == "get_domain_status":
            result = _get_domain_status(arguments["entity_id"])
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
