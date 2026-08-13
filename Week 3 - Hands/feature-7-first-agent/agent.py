"""
Agent execution engine — Feature 7 starter.

Your task: implement run_agent() so it runs the two-call tool-calling loop.

The TOOLS_REGISTRY is already wired for you (all three example tools are registered).
You only need to fill in the TODO sections in run_agent().

THE AGENT LOOP (what you're building):
  user message
      ↓
  LLM (with tool schemas) ──→ tool_calls: [{name, arguments}]
      ↓
  execute each tool via TOOLS_REGISTRY → results
      ↓
  LLM again (with results) ──→ final natural-language answer
      ↓
  return {result, steps, tools_used}

If the LLM doesn't call any tools, return its direct text response immediately.
No second call needed — the first response IS the answer.

Test messages to trigger each tool:
  "Is there an opening on Friday at 3 PM?"   → check_availability
  "I can't log in to my account"             → create_ticket
  "What are your business hours?"             → lookup_info
  "What is the capital of France?"           → no tool (direct LLM answer)
"""
import json
from typing import Any

from shared.llm_client import call_llm
from shared.session_store import add_message, get_session
from shared.tools import (
    CHECK_AVAILABILITY_SCHEMA,
    CREATE_TICKET_SCHEMA,
    LOOKUP_INFO_SCHEMA,
    check_availability,
    create_ticket,
    lookup_info,
)

# =============================================================================
# TOOLS_REGISTRY — already wired, no changes needed here.
#
# Maps tool name → (callable, OpenAI-format schema).
# When you add your own tools in shared/tools.py, register them here.
# =============================================================================

TOOLS_REGISTRY: dict[str, tuple[Any, dict]] = {
    "check_availability": (check_availability, CHECK_AVAILABILITY_SCHEMA),
    "create_ticket":      (create_ticket, CREATE_TICKET_SCHEMA),
    "lookup_info":        (lookup_info, LOOKUP_INFO_SCHEMA),
}

_TOOL_SCHEMAS = [schema for _, schema in TOOLS_REGISTRY.values()]

_AGENT_SYSTEM_PROMPT = """You are a helpful AI assistant for [YOUR_DOMAIN].
You have access to tools that can check availability, create support tickets,
and look up factual information. Use these tools whenever the user's request
would benefit from real data — don't guess at facts you could look up.

When you use a tool, wait for its result before responding. Synthesize the
tool results into a clear, helpful answer in plain English.
If a tool fails, acknowledge it and offer alternatives."""


async def run_agent(
    message: str,
    session_id: str,
    tenant_id: str = "default",
) -> dict:
    """
    Run the agent loop for one user message.

    Returns:
        {
          "result":     str  — the final natural-language answer
          "steps":      list — [{tool, args, result}] for each tool executed
          "tools_used": list — names of tools called (empty list if no tools used)
        }
    """
    # =========================================================================
    # SETUP: Build the initial message list from session history.
    # (This part is given — no TODO here.)
    # =========================================================================
    session = get_session(session_id, tenant_id=tenant_id)
    history: list[dict] = []
    if session:
        recent = session.messages[-20:]
        for msg in recent:
            history.append({"role": msg.role, "content": msg.content})

    messages: list[dict] = [
        {"role": "system", "content": _AGENT_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": message},
    ]

    # =========================================================================
    # TODO STEP 1: Make the first LLM call with tools.
    #
    # Call call_llm() with:
    #   - messages=messages
    #   - tools=_TOOL_SCHEMAS        ← this is what tells the LLM what tools exist
    #   - temperature=0.3            ← lower = more deterministic tool selection
    #   - max_tokens=1000
    #
    # Store the result in: first_response
    # =========================================================================
    #raise NotImplementedError(
     #   "TODO STEP 1: Call call_llm() with messages and tools=_TOOL_SCHEMAS. "
      #  "See the docstring above."
    #)

    first_response = await call_llm(
        messages=messages,
        tools=_TOOL_SCHEMAS,
        temperature=0.3,
        max_tokens = 1000,
    )

    steps: list[dict] = []
    tools_used: list[str] = []

    # =========================================================================
    # TODO STEP 2: Check if the LLM called any tools.
    #
    # If first_response.tool_calls is empty (the LLM answered directly):
    #   - Get the answer from first_response.content
    #   - Call add_message() twice to persist both turns to session history
    #   - Return {"result": answer, "steps": [], "tools_used": []}
    #
    # Hint: first_response.tool_calls is a list of dicts:
    #   [{"id": "call_abc", "name": "check_availability", "arguments": {...}}]
    # =========================================================================

    # =========================================================================
    # TODO STEP 3: Execute each tool call.
    #
    # For each tc in first_response.tool_calls:
    #   a. Look up the function: fn, _ = TOOLS_REGISTRY.get(tc["name"], (None, None))
    #   b. If fn is None: tool_result = {"error": f"Unknown tool '{tc['name']}'"}
    #   c. Otherwise: tool_result = fn(**tc["arguments"])
    #      (wrap in try/except — return {"error": str(exc)} on failure)
    #   d. Append {"tool": tc["name"], "args": tc["arguments"], "result": tool_result}
    #      to steps, and tc["name"] to tools_used.
    #   e. Append the tool result as a "tool" role message:
    #      messages.append({
    #          "role": "tool",
    #          "tool_call_id": tc["id"],
    #          "content": json.dumps(tool_result),
    #      })
    #
    # IMPORTANT: Before appending tool results, you MUST first append the
    # assistant's tool-call message so the LLM knows what it previously decided:
    #   messages.append({
    #       "role": "assistant",
    #       "content": first_response.content,   # may be None
    #       "tool_calls": [
    #           {
    #               "id": tc["id"],
    #               "type": "function",
    #               "function": {
    #                   "name": tc["name"],
    #                   "arguments": json.dumps(tc["arguments"]),
    #               },
    #           }
    #           for tc in first_response.tool_calls
    #       ],
    #   })
    # =========================================================================

    # =========================================================================
    # TODO STEP 4: Make the second LLM call to synthesize the final answer.
    #
    # Call call_llm() again with:
    #   - messages=messages   ← now includes tool results from Step 3
    #   - temperature=0.7
    #   - max_tokens=1000
    #   - NO tools parameter (we want a natural-language response, not more calls)
    #
    # Get the answer from second_response.content.
    # Call add_message() twice to persist both turns to session history.
    # Return {
    #     "result": answer,
    #     "steps": steps,
    #     "tools_used": list(dict.fromkeys(tools_used)),  # deduplicated
    # }
    # =========================================================================


    if not first_response.tool_calls:
        answer = first_response.content or ""
        add_message(session_id, "user", message)
        add_message(session_id,"assistant",answer)
        return {"result":answer, "steps": [], "tools_used": []}

    messages.append({
        "role":"assistant",
        "content": first_response.content,
        "tool_calls": [{
            "id": tc["id"],
            "type": "function",
            "function":{
                "name": tc["name"],
                "arguments": json.dumps(tc["arguments"]),
            },
        }
        for tc in first_response.tool_calls
        ],
    })

    for tc in first_response.tool_calls:
        fn, _ = TOOLS_REGISTRY.get(tc["name"], (None, None))
        if fn is None:
            tool_result = {"error": f"Unknown tool '{tc['name']}'"}
        else:
            try:
                tool_result = fn(**tc["arguments"])
            except Exception as exc:
                tool_result = {"error": str(exc)}

        steps.append ({"tool" : tc["name"], "args": tc["arguments"], "result" : tool_result})
        tools_used.append(tc["name"])

        messages.append({
            "role":"tool",
            "tool_call_id": tc["id"],
            "content": json.dumps(tool_result),
        })

        second_response = await call_llm(
            messages = messages,
            temperature = 0.7,
            max_tokens = 1000,
        )

        answer = second_response.content or ""
        add_message(session_id, "user", message)
        add_message(session_id,"assistant", answer)

        return {
            "result" : answer,
            "steps" : steps,
            "tools_used": list(dict.fromkeys(tools_used)),
        }
