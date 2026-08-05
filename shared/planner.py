"""
Task planner for Feature 8: Multi-Step Agent.

Implements the Plan-and-Execute agent pattern:
  1. make_plan()    — LLM call that decomposes a request into 2-5 concrete steps.
  2. execute_plan() — async loop that runs each step through run_agent(),
                      tracks progress, and synthesizes a final answer.

WHAT THIS IS (agent design pattern):
  Plan-and-Execute separates PLANNING from DOING. The LLM commits to a plan
  before execution starts — this prevents the common failure where a single-turn
  agent gets distracted mid-task and loses track of the overall goal.

FRAMEWORK EQUIVALENT:
  In LangGraph this is modelled as a directed graph with two node types:
    - "plan" node → calls make_plan()
    - "execute" node → runs one step of the plan
  The edge from "execute" back to itself (or to END) is the loop we implement
  in execute_plan() below. LangGraph adds conditional branching (retry on failure,
  route to human review) that our simpler linear loop doesn't support.

PUBLIC API:
  make_plan(message)               → list[str]
  execute_plan(task_id)            → None  (updates task in-place via task_store)
"""
import json

from shared.agent import run_agent
from shared.llm_client import call_llm
from shared.task_store import get_task, update_task

_PLANNER_SYSTEM_PROMPT = """You are a task planner. Your job is to break a user's request
into a sequence of 2 to 5 concrete, actionable steps that an AI agent can execute one by one.

Each step should:
- Be a standalone instruction that can be executed independently
- Be specific enough that an AI agent knows exactly what to do
- Build on the results of previous steps where needed

Respond ONLY with a JSON array of strings — no preamble, no markdown fences.
Example:
["Check availability for Friday at 3 PM", "Create a support ticket for the user", "Look up business hours"]
"""

_SYNTHESIZER_SYSTEM_PROMPT = """You are a helpful AI assistant. You have just completed
a multi-step task. Below are the results of each step.

Write a clear, concise final summary for the user that:
- Directly answers their original request
- Incorporates the key information from each step's result
- Is written in plain English (no JSON, no bullet points unless natural)
- Does not mention "steps" or the planning process — just give the answer
"""


async def make_plan(message: str) -> list[str]:
    """
    Ask the LLM to decompose a user request into 2–5 concrete steps.

    Returns a list of step strings. Falls back to a single-step plan
    if parsing fails so execute_plan() always has something to run.
    """
    messages = [
        {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    response = await call_llm(
        messages=messages,
        temperature=0.3,
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    raw = response.content or "[]"
    try:
        parsed = json.loads(raw)
        # The LLM may return {"steps": [...]} instead of [...] directly
        if isinstance(parsed, dict):
            for key in ("steps", "plan", "tasks"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
        if isinstance(parsed, list) and parsed:
            return [str(s) for s in parsed[:5]]  # cap at 5 steps
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: treat the whole request as one step
    return [message]


async def execute_plan(task_id: str) -> None:
    """
    Execute every step in an AgentTask's plan using the Feature 7 agent loop.

    This function runs as a FastAPI BackgroundTask — it updates the AgentTask
    in task_store after each step so the polling endpoint shows live progress.

    After all steps complete, calls the LLM once more to synthesize a final
    answer from all step results and marks the task done.

    On any exception the task is marked status="error" with the error message.
    """
    task = get_task(task_id)
    if task is None:
        return

    update_task(task_id, status="executing", steps_completed=[])

    steps_completed: list[dict] = []
    try:
        plan = task.plan or []
        for i, step in enumerate(plan):
            # Run this step through the single-turn agent (Feature 7).
            step_result = await run_agent(
                message=step,
                session_id=task.session_id,
                tenant_id=task.tenant_id,
            )
            step_record = {
                "step_index": i,
                "step": step,
                "result": step_result.get("result", ""),
                "tools_used": step_result.get("tools_used", []),
            }
            steps_completed.append(step_record)
            # Update the task after every step so the UI can show live progress.
            update_task(task_id, steps_completed=list(steps_completed))

        # Synthesize a final answer from all step results.
        step_summary = "\n".join(
            f"Step {r['step_index'] + 1} ({r['step']}): {r['result']}"
            for r in steps_completed
        )
        synth_messages = [
            {"role": "system", "content": _SYNTHESIZER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Original request: {task.message}\n\nStep results:\n{step_summary}"},
        ]
        synth_response = await call_llm(synth_messages, temperature=0.7, max_tokens=800)
        final_result = synth_response.content or "Task completed."

        update_task(task_id, status="done", result=final_result)

    except Exception as exc:
        update_task(task_id, status="error", error=str(exc))
