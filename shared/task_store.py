"""
In-memory task store for Feature 8: Multi-Step Agent.

Stores AgentTask objects by ID so the background execute_plan() function and
the GET /api/agent/status/{task_id} polling endpoint can share state without
a database.

Same pattern as session_store.py — a plain dict at module level, lost on
server restart. For production, swap with a Redis or Postgres-backed store.
"""
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from shared.models import AgentTask


_tasks: dict[str, AgentTask] = {}


def create_task(
    message: str,
    session_id: str = "default",
    tenant_id: str = "default",
) -> AgentTask:
    """Create a new AgentTask in 'planning' status and return it."""
    task_id = str(uuid.uuid4())
    now = datetime.now(tz=timezone.utc)
    task = AgentTask(
        id=task_id,
        status="planning",
        message=message,
        session_id=session_id,
        tenant_id=tenant_id,
        created_at=now,
        updated_at=now,
    )
    _tasks[task_id] = task
    return task


def get_task(task_id: str) -> Optional[AgentTask]:
    """Return the task with the given ID, or None if not found."""
    return _tasks.get(task_id)


def update_task(task_id: str, **fields: Any) -> None:
    """
    Update one or more fields on an existing task.

    Automatically sets updated_at to the current UTC time.
    Silently does nothing if the task_id is not found.

    Example:
        update_task(task_id, status="executing", plan=["Step 1", "Step 2"])
    """
    task = _tasks.get(task_id)
    if task is None:
        return
    for field, value in fields.items():
        if hasattr(task, field):
            setattr(task, field, value)
    task.updated_at = datetime.now(tz=timezone.utc)
