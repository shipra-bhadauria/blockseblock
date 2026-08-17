"""
Task planner for Feature 8: Multi-Step Agent — starter stub.

Your task: implement make_plan() and execute_plan() so the agent can
decompose a request into steps and execute each step in sequence.

THE PLAN-AND-EXECUTE PATTERN (what you're building):
  user request
      ↓
  make_plan() → ["Step 1: check availability", "Step 2: create ticket", ...]
      ↓
  execute_plan() — for each step:
      run_agent(step)  → {result, steps, tools_used}
      update task state (the UI polls to see progress)
      ↓
  LLM synthesis call  → final answer combining all step results
      ↓
  task.status = "done", task.result = final_answer

Why plan first?
  Single-turn agents (Feature 7) can lose track of the overall goal when
  executing a complex multi-step task. Planning first commits the agent to
  a structure before execution starts — it can't get distracted halfway.

Test messages:
  "Check availability for Friday at 3 PM, then create a support ticket
   with the result and look up our business hours"   → 3-step plan
  "What is the weather today?"                       → probably 1-step plan
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

    Returns a list of step strings.

    =========================================================================
    TODO STEP 1: Build messages and call the LLM.

    messages = [
        {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]

    Call call_llm() with:
      - messages=messages
      - temperature=0.3
      - max_tokens=500
      - response_format={"type": "json_object"}

    Store the result in: response
    =========================================================================

    =========================================================================
    TODO STEP 2: Parse the JSON response.

    raw = response.content or "[]"
    parsed = json.loads(raw)

    The LLM may return {"steps": [...]} instead of a bare list.
    Check if parsed is a dict; if so, look for a "steps", "plan", or "tasks"
    key that holds a list, and use that list.

    Cap at 5 items: parsed[:5]
    Return [str(s) for s in parsed]

    Fall back to [message] (single-step plan) if parsing fails.
    =========================================================================
    """
    #raise NotImplementedError(
     #   "TODO: Implement make_plan(). See the step-by-step comments above."
    #)

    async def make_plan(message: str) -> list[str]:
        #step 1 : LLM call
        messages = [
            {"role" : "system", "content" : _PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]
        response = await call_llm(
            messages = messages,
            temperature = 0.3,
            max_tokens = 500,
            response_format = {"type":"json_object"},
        )

        #step 2 : Parse response
        try:
            raw = response.content or "[]"
            parsed = json.loads(raw)

            #If LLM has some steps 
            if isinstance(parsed, dict):
                for key in ("steps", "plan", "tasks"):
                    if isinstance(parsed.get(key), list):
                        parsed = parsed[key]
                        break

            parsed = parsed[:5]
            return [str(s) for s in parsed]

        except Exception:
            return[message] 


async def execute_plan(task_id: str) -> None:
    """
    Execute every step in an AgentTask's plan using the Feature 7 agent loop.

    This function runs as a FastAPI BackgroundTask — it updates the AgentTask
    after each step so the polling endpoint shows live progress.

    =========================================================================
    TODO STEP 3: Run each step through the agent.

    task = get_task(task_id)
    update_task(task_id, status="executing", steps_completed=[])

    steps_completed = []
    plan = task.plan or []

    For each i, step in enumerate(plan):
      a. Call: step_result = await run_agent(
             message=step,
             session_id=task.session_id,
             tenant_id=task.tenant_id,
         )
      b. Build a step record:
         step_record = {
             "step_index": i,
             "step": step,
             "result": step_result.get("result", ""),
             "tools_used": step_result.get("tools_used", []),
         }
      c. Append to steps_completed, then:
         update_task(task_id, steps_completed=list(steps_completed))
         (This write-after-every-step is what makes the UI's live progress work.)
    =========================================================================

    =========================================================================
    TODO STEP 4: Synthesize the final answer.

    Build a step_summary string:
      "Step 1 (check availability): There is an opening on Friday at 3 PM."
      "Step 2 (create ticket): Ticket #1234 created."
      ...

    Call call_llm() with:
      messages = [
          {"role": "system", "content": _SYNTHESIZER_SYSTEM_PROMPT},
          {"role": "user", "content":
              f"Original request: {task.message}\n\nStep results:\n{step_summary}"},
      ]
      temperature=0.7
      max_tokens=800

    Get the final answer from synth_response.content.
    Call: update_task(task_id, status="done", result=final_result)

    Wrap the ENTIRE execute logic in try/except Exception as exc:
      update_task(task_id, status="error", error=str(exc))
    =========================================================================
    """
    #raise NotImplementedError(
     #   "TODO: Implement execute_plan(). See the step-by-step comments above."
    #)

    try: 
        task = get_task(task_id)
        update_task(task_id, status="executing", steps_completed=[])

        steps_completed = []
        plan = task.plan or []

        #step 3 execute
        for i , step in enumerate(plan):
            step_result = await run_agent(
                message = step,
                session_id= task.session_id,
                tenant_id = task.tenant_id,
            )

            step_record = {
                "step_index":i,
                "step":step,
                "result":step_result.get("result", ""),
                "tools_used": step_result.get("tools_used", []),
            }
            steps_completed.append(step_record)
            update_task(task_id, steps_completed=list(steps_completed))

        #step 4: synthesis
        step_summary = "\n".join(
            f"Step {r['step_index'] + 1} ({r['step']}): {r['result']}"
            for r in steps_completed
        )

        synth_response = await call_llm(
            messages = [
                {"role":"system", "content": _SYNTHESIZER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Original request: {task.message}\n\nStep results:\n{step_summary}"},
            ],
            temperature = 0.7,
            max_tokens = 800,
        )

        final_result = synth_response.content or ""
        update_task(task_id, status="done", result= final_result)

    except Exception as exc:
        update_task(task_id, status="error", error=str(exc))
