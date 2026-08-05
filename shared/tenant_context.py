"""
Tenant context utilities for Feature 6, Part B: multi-tenant isolation.

WHAT IS MULTI-TENANCY?
  When a single deployed instance of this assistant serves MULTIPLE
  organizations ("tenants"), each tenant's data — documents, sessions,
  conversation history, vector embeddings — must be completely invisible
  to every other tenant. A search by Tenant A must NEVER retrieve content
  belonging to Tenant B, even if their questions are semantically similar.
  This is a security boundary, not just an organizational nicety: getting
  it wrong is a data breach.

HOW TENANT ISOLATION IS ENFORCED:
  1. Identification: tenant_id is read from the X-Tenant-ID request header.
     In production this would come from a JWT claim or API key lookup (see
     note below). The header approach here is simplified for learning.

  2. Storage: all Document, Session, and vector store records include
     tenant_id. All list/search operations filter by it.

  3. The critical enforcement point is the VECTOR DATABASE filter:
       collection.query(where={"tenant_id": tenant_id}, ...)
     Application-level filtering (fetch then discard non-matching results)
     is fragile — a bug means a leak. Database-level filtering means
     cross-tenant chunks are never even retrieved.

PRODUCTION NOTE:
  In production, tenant_id typically comes from an authenticated context,
  not a raw header:
  - API Gateway pattern: resolve API key → tenant_id before the request
    reaches your service. The service trusts the resolved header.
  - JWT pattern: validate the JWT, extract the "tenant" claim. Never trust
    client-supplied tenant values without authentication.
  - Subdomain pattern: route acme.yoursaas.com vs globex.yoursaas.com to
    the same service with different tenant_id in the request context.
  See Resource 6, Section 2 for the full production pattern comparison.

This module is ONLY active when ENABLE_MULTI_TENANT=true in .env.
When disabled, get_tenant_id() always returns "default" and all existing
Feature 1-5 behavior is unchanged.
"""
from fastapi import Request

DEFAULT_TENANT = "default"


def get_tenant_id(request: Request) -> str:
    """
    FastAPI dependency: extract tenant_id from the X-Tenant-ID request header.

    Returns "default" when:
      - ENABLE_MULTI_TENANT=false (single-tenant mode — Part B disabled)
      - The X-Tenant-ID header is absent or empty

    Usage in an endpoint:
      @app.post("/api/chat/smart")
      async def smart_chat(req: SmartChatRequest, tenant_id: str = Depends(get_tenant_id)):
          ...

    In production, validate that the header value matches an authenticated
    identity rather than accepting it at face value. See module docstring.
    """
    from shared.config import settings

    if not settings.enable_multi_tenant:
        return DEFAULT_TENANT

    tenant = request.headers.get("X-Tenant-ID", "").strip()
    return tenant if tenant else DEFAULT_TENANT
