"""
🔐 BONUS MODULE — PROMPT 0.5.1 (Part A)
Not required to complete any feature. Adds production-grade secrets
management for students building toward real deployment or working
with sensitive data.

Secrets loader — abstracted to support both .env (development) and a
secrets vault (production/sensitive). Set SECRETS_PROVIDER in your
environment to switch:
  SECRETS_PROVIDER=env       → load from .env (default, same as before)
  SECRETS_PROVIDER=infisical → load from Infisical vault
  SECRETS_PROVIDER=doppler   → load from Doppler vault

WHY THIS MATTERS:
  Storing API keys in a .env file is the minimum viable approach — fine
  for learning, risky for anything real. In 2025, 10 million+ credentials
  were leaked from GitHub; every one started with a key stored in a file
  that got committed. A secrets vault is the standard alternative: keys
  are stored in an encrypted, audited central store and injected into the
  application at runtime — they never appear in code or config files.

QUICK START — Infisical:
  1. Sign up at app.infisical.com (free individual plan)
  2. Create a project, add your secrets (OPENAI_API_KEY etc.)
  3. Create a Machine Identity (Client Credentials) under Project Settings
  4. Add to .env:
       SECRETS_PROVIDER=infisical
       INFISICAL_CLIENT_ID=<your-machine-id>
       INFISICAL_CLIENT_SECRET=<your-machine-secret>
       INFISICAL_PROJECT_ID=<your-project-id>
  5. Install: pip install infisical-sdk

  Note: INFISICAL_CLIENT_ID/SECRET are machine identity credentials —
  NOT your actual API keys. They can be rotated independently and have
  scoped permissions. Safe to store in .env.

QUICK START — Doppler:
  1. Sign up at doppler.com (free for 5 users)
  2. Install CLI: brew install dopplerhq/cli/doppler (macOS)
  3. Run: doppler setup (links this project)
  4. Start the server: doppler run -- uvicorn main:app --reload
     Doppler injects secrets as environment variables before your app
     starts — os.environ["OPENAI_API_KEY"] just works.
  5. In code: set SECRETS_PROVIDER=doppler; all secrets come from env

Verify your vault integration is working:
  python -c "from shared.secrets import get_secret; print(get_secret('OPENAI_API_KEY')[:4] + '...')"
"""
import os
from functools import lru_cache


def _load_from_infisical(secret_name: str) -> str:
    """Fetch a single secret from the Infisical vault."""
    try:
        from infisical_sdk import InfisicalSDKClient
    except ImportError:
        raise ImportError(
            "infisical-sdk is not installed. "
            "Run: pip install infisical-sdk"
        )

    client_id = os.environ.get("INFISICAL_CLIENT_ID", "")
    client_secret = os.environ.get("INFISICAL_CLIENT_SECRET", "")
    project_id = os.environ.get("INFISICAL_PROJECT_ID", "")

    if not all([client_id, client_secret, project_id]):
        raise ValueError(
            "Infisical provider requires INFISICAL_CLIENT_ID, "
            "INFISICAL_CLIENT_SECRET, and INFISICAL_PROJECT_ID in .env. "
            "See shared/secrets.py for setup instructions."
        )

    client = InfisicalSDKClient(host="https://app.infisical.com")
    client.auth.universal_auth.login(
        client_id=client_id,
        client_secret=client_secret,
    )
    secret = client.secrets.get_secret_by_name(
        secret_name=secret_name,
        project_id=project_id,
        environment_slug=os.getenv("INFISICAL_ENV", "dev"),
    )
    return secret.secret_value


def get_secret(name: str) -> str:
    """
    Retrieve a secret by name from the configured provider.

    Falls back to os.environ (i.e., .env via pydantic_settings) if
    provider is 'env' or unset. Raises ValueError if the secret is
    missing and the provider is 'env'.

    For Doppler: run your app with `doppler run -- uvicorn main:app`.
    Secrets are injected as env vars before the process starts, so
    os.environ already contains them when this function runs.
    """
    provider = os.getenv("SECRETS_PROVIDER", "env").lower()

    if provider == "infisical":
        return _load_from_infisical(name)

    elif provider == "doppler":
        # Doppler injects secrets as environment variables at process start.
        # Nothing special required here — the injection already happened.
        value = os.environ.get(name)
        if not value:
            raise ValueError(
                f"Secret '{name}' not found in environment. "
                f"If using Doppler, make sure you started the server with: "
                f"doppler run -- uvicorn main:app --reload"
            )
        return value

    else:
        # Default: read from os.environ (populated by .env via pydantic_settings)
        value = os.environ.get(name)
        if not value:
            raise ValueError(
                f"Secret '{name}' not found in environment. "
                f"Check your .env file — make sure {name} is set."
            )
        return value


@lru_cache(maxsize=None)
def get_secret_cached(name: str) -> str:
    """
    Cached version — use for secrets that don't rotate during a single
    server process lifetime (all API keys in this course qualify).

    Cache is per-process; a restart re-fetches from the vault.
    To clear during testing: get_secret_cached.cache_clear()
    """
    return get_secret(name)
