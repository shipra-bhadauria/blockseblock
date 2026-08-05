"""
Example tools for Feature 7: First Agent.

These three tools represent generic domain actions — a support ticket system,
an availability checker, and a knowledge base lookup. They use mock/hardcoded
data so the agent loop works immediately, without any external API calls.

YOUR TASK:
  Replace these three functions (and their schemas) with tools relevant to
  your own domain. Keep the same pattern:
    1. A plain Python function that takes typed arguments and returns a dict.
    2. A TOOL_SCHEMA dict alongside it in OpenAI function-calling format.
    3. A registration entry in shared/agent.py's TOOLS_REGISTRY.

WHAT MAKES A GOOD TOOL:
  - One action per tool (small, focused scope)
  - Clear description — the LLM reads this to decide when to call it
  - ≤4 parameters — more parameters confuse the LLM's argument generation
  - Returns a flat dict — easier for the LLM to summarize in natural language
  - Never raises exceptions — return {"error": "..."} instead so the agent
    can incorporate the failure into its response

DEEPAGENT NOTE (WWW 2026):
  This pre-registered tool registry is the foundation of most production agents.
  The research frontier (e.g., DeepAgent, WWW 2026, github.com/RUC-NLPIR/DeepAgent)
  replaces this with dynamic tool discovery from large tool libraries — the agent
  searches for tools it needs from 16,000+ RapidAPIs within its reasoning process,
  rather than using only pre-registered ones. If you're curious where tool-calling
  agents are heading, start there.
"""
import random
import uuid


# =============================================================================
# Tool 1: Availability check
# =============================================================================

def check_availability(date: str, time: str) -> dict:
    """
    Check whether an appointment slot is available on the given date and time.

    Returns a dict indicating availability, and if available, a booking
    reference the caller can use to confirm.

    Args:
        date: Date string in any common format (e.g. "2024-01-15", "next Monday").
        time: Time string (e.g. "10:00 AM", "14:30", "afternoon").

    Returns:
        {"available": bool, "date": str, "time": str, "booking_ref": str | None}
    """
    # Mock implementation — replace with a real calendar API call.
    is_available = random.choice([True, True, True, False])  # 75% available
    return {
        "available": is_available,
        "date": date,
        "time": time,
        "booking_ref": f"REF-{uuid.uuid4().hex[:6].upper()}" if is_available else None,
        "next_available": "tomorrow at 2:00 PM" if not is_available else None,
    }


CHECK_AVAILABILITY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_availability",
        "description": (
            "Check whether an appointment or booking slot is available on a given date and time. "
            "Use this when the user asks about scheduling, bookings, appointments, or availability. "
            "Returns whether the slot is open and a booking reference if it is."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "The date to check, e.g. '2024-01-15', 'next Monday', 'tomorrow'.",
                },
                "time": {
                    "type": "string",
                    "description": "The time to check, e.g. '10:00 AM', '2:30 PM', 'morning', 'afternoon'.",
                },
            },
            "required": ["date", "time"],
        },
    },
}


# =============================================================================
# Tool 2: Support ticket creation
# =============================================================================

def create_ticket(subject: str, description: str, priority: str = "normal") -> dict:
    """
    Create a support ticket and return its ID and estimated response time.

    Args:
        subject: One-line summary of the issue.
        description: Detailed description of the problem or request.
        priority: Urgency level — "low", "normal", or "high". Defaults to "normal".

    Returns:
        {"ticket_id": str, "subject": str, "priority": str,
         "status": str, "estimated_response": str}
    """
    # Mock implementation — replace with a real ticketing API call (Zendesk, Jira, etc.).
    valid_priorities = {"low", "normal", "high"}
    if priority not in valid_priorities:
        priority = "normal"

    response_times = {
        "low": "3–5 business days",
        "normal": "1–2 business days",
        "high": "within 4 hours",
    }

    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    return {
        "ticket_id": ticket_id,
        "subject": subject,
        "priority": priority,
        "status": "open",
        "estimated_response": response_times[priority],
        "confirmation": f"Your ticket {ticket_id} has been created successfully.",
    }


CREATE_TICKET_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_ticket",
        "description": (
            "Create a support ticket for an issue, complaint, or request. "
            "Use this when the user reports a problem, asks for help with a specific issue, "
            "or requests something that requires follow-up action from the team. "
            "Returns a ticket ID and estimated response time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "A short one-line summary of the issue or request.",
                },
                "description": {
                    "type": "string",
                    "description": "A detailed description of the issue, including relevant context.",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high"],
                    "description": (
                        "Urgency level. Use 'high' for urgent issues (service outage, data loss, "
                        "blocking a deadline). Use 'low' for general questions or nice-to-haves. "
                        "Default 'normal' for everything else."
                    ),
                },
            },
            "required": ["subject", "description"],
        },
    },
}


# =============================================================================
# Tool 3: Information lookup
# =============================================================================

_INFO_KNOWLEDGE_BASE = {
    "hours": {
        "topic": "hours",
        "answer": "We are open Monday–Friday 9:00 AM to 6:00 PM, and Saturday 10:00 AM to 4:00 PM. We are closed on Sundays and public holidays.",
        "last_updated": "2024-01-01",
    },
    "pricing": {
        "topic": "pricing",
        "answer": "Pricing depends on the service. Basic plan: $29/month. Pro plan: $79/month. Enterprise: custom pricing. All plans include a 14-day free trial.",
        "last_updated": "2024-01-01",
    },
    "policy": {
        "topic": "policy",
        "answer": "Refunds are available within 30 days of purchase. Cancellations can be made at any time; you will retain access until the end of your billing period.",
        "last_updated": "2024-01-01",
    },
    "shipping": {
        "topic": "shipping",
        "answer": "Standard shipping takes 5–7 business days. Express shipping (2–3 days) is available at an additional cost. Free shipping on orders over $50.",
        "last_updated": "2024-01-01",
    },
    "contact": {
        "topic": "contact",
        "answer": "You can reach us at support@example.com, or call 1-800-555-0100 during business hours. Live chat is available on the website.",
        "last_updated": "2024-01-01",
    },
}


def lookup_info(topic: str) -> dict:
    """
    Look up factual information about a topic from the organization's knowledge base.

    Covers common topics like hours, pricing, policies, shipping, and contact details.
    Returns "not found" if the topic isn't in the knowledge base.

    Args:
        topic: The topic to look up. Supported: "hours", "pricing", "policy",
               "shipping", "contact".

    Returns:
        {"topic": str, "answer": str, "found": bool}
    """
    # Normalize the topic — strip whitespace, lowercase, handle plurals.
    normalized = topic.lower().strip().rstrip("s")  # "policies" -> "polic" won't match, handled below
    # Try direct match first, then check if any key starts with the normalized topic.
    result = _INFO_KNOWLEDGE_BASE.get(normalized)
    if result is None:
        for key, val in _INFO_KNOWLEDGE_BASE.items():
            if key.startswith(normalized) or normalized.startswith(key):
                result = val
                break

    if result:
        return {"found": True, **result}
    return {
        "found": False,
        "topic": topic,
        "answer": f"I don't have specific information about '{topic}' in my knowledge base. Try asking about: hours, pricing, policy, shipping, or contact.",
    }


LOOKUP_INFO_SCHEMA = {
    "type": "function",
    "function": {
        "name": "lookup_info",
        "description": (
            "Look up factual information about the organization — such as business hours, "
            "pricing plans, refund and cancellation policies, shipping times, or contact details. "
            "Use this when the user asks a specific factual question that has a definitive answer "
            "stored in the knowledge base. Supported topics: hours, pricing, policy, shipping, contact."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "The topic to look up. One of: 'hours', 'pricing', 'policy', "
                        "'shipping', 'contact'. Use the closest matching topic."
                    ),
                },
            },
            "required": ["topic"],
        },
    },
}
