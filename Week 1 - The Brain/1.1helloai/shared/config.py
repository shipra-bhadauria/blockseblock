"""
Centralized configuration for the AI Engineering Bootcamp.

All environment variables are read here. No other file in this project should
call os.getenv() directly — import `settings` from this module instead.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and the .env file."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        # Unknown env vars in .env are silently ignored — students can have
        # provider-specific vars set without breaking providers they aren't using.
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Provider routing
    # -------------------------------------------------------------------------

    # Which provider to use for chat/agents/RAG.
    # Valid values: "groq" | "openai" | "anthropic" | "cohere" | "ollama"
    #               "azure" | "bedrock" | "vertex" | "custom"
    # groq is recommended for beginners — free tier, no credit card, very fast.
    llm_provider: str = "groq"

    # Optional: use a different provider for voice (STT/TTS) in Feature 10.
    # Defaults to llm_provider if not set.
    voice_provider: str = ""

    # Optional: use a separate provider for image understanding (Feature 10 Vision).
    vlm_provider: str = ""

    # -------------------------------------------------------------------------
    # Groq (recommended for beginners — free tier at console.groq.com)
    # -------------------------------------------------------------------------
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # -------------------------------------------------------------------------
    # OpenAI
    # -------------------------------------------------------------------------
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # Optional: override the OpenAI base URL (e.g. for a proxy or compatible server).
    # For Groq/Azure/Bedrock/Vertex, use their dedicated sections below instead.
    openai_base_url: str = ""

    # -------------------------------------------------------------------------
    # Anthropic
    # -------------------------------------------------------------------------
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # -------------------------------------------------------------------------
    # Cohere
    # -------------------------------------------------------------------------
    cohere_api_key: str = ""
    cohere_model: str = "command-r-plus"

    # -------------------------------------------------------------------------
    # Ollama (local SLM — no API key needed)
    # -------------------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # -------------------------------------------------------------------------
    # Azure OpenAI
    # -------------------------------------------------------------------------
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment_name: str = ""
    azure_openai_api_version: str = "2024-02-01"

    # -------------------------------------------------------------------------
    # AWS Bedrock
    # -------------------------------------------------------------------------
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    bedrock_model_id: str = ""

    # -------------------------------------------------------------------------
    # GCP Vertex AI
    # -------------------------------------------------------------------------
    gcp_project_id: str = ""
    gcp_region: str = "us-central1"
    vertex_model: str = ""

    # -------------------------------------------------------------------------
    # Custom OpenAI-compatible endpoint (Together AI, Fireworks, vLLM, etc.)
    # -------------------------------------------------------------------------
    custom_base_url: str = ""
    custom_api_key: str = ""
    custom_model: str = ""

    # -------------------------------------------------------------------------
    # Vector database (Feature 5+)
    # -------------------------------------------------------------------------
    vector_db_path: str = "./data/vectordb"
    embedding_model: str = "text-embedding-3-small"

    # -------------------------------------------------------------------------
    # Feature 6: Smart Router feature flags
    # -------------------------------------------------------------------------

    # Enable PageIndex routing for professional document queries.
    # Requires a tree index pre-built with run_pageindex.py.
    # See github.com/VectifyAI/PageIndex for setup.
    enable_pageindex: bool = False

    # Enable multi-tenant isolation (Part B — enterprise extension).
    # When True, every document/session/vector is scoped to a tenant_id
    # derived from the X-Tenant-ID request header.
    enable_multi_tenant: bool = False

    # Enable long-term retrieval memory (Part C — enterprise extension).
    # When True, retrieval events are logged and can be summarised into a
    # KnowledgeDigest that is injected into the smart chat system prompt.
    enable_long_term_context: bool = False

    # -------------------------------------------------------------------------
    # Bonus Module 0.5.1 (Part A): Secrets management
    # -------------------------------------------------------------------------
    # "env" = read from .env (default). "infisical" or "doppler" = vault.
    # See shared/secrets.py for setup instructions.
    secrets_provider: str = "env"

    # Infisical machine identity (used when secrets_provider="infisical")
    infisical_client_id: str = ""
    infisical_client_secret: str = ""
    infisical_project_id: str = ""
    infisical_env: str = "dev"

    # -------------------------------------------------------------------------
    # Bonus Module 0.5.1 (Part B): Ollama local privacy hardening
    # -------------------------------------------------------------------------
    # When True, OllamaProvider logs startup warnings for -cloud models and
    # plain-text history logging. See docs/ollama-privacy-guide.md.
    enable_ollama_privacy_checks: bool = False

    # -------------------------------------------------------------------------
    # Feature 9: MCP Integration
    # -------------------------------------------------------------------------
    # Enable the domain-specific MCP server (shared/domain_mcp_server.py).
    # When True, the domain server is registered alongside the demo server and
    # its tools appear in /api/mcp/tools and the agent's tool list.
    enable_domain_mcp_server: bool = False

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    app_port: int = 8000
    app_host: str = "0.0.0.0"

    # -------------------------------------------------------------------------
    # Feature 10: Voice + Vision models
    # -------------------------------------------------------------------------
    stt_model: str = "whisper-large-v3"   # speech-to-text (Groq default)
    tts_model: str = ""                    # text-to-speech (OpenAI default when set)
    vlm_model: str = "llava"              # vision-language model (Ollama default)

    # -------------------------------------------------------------------------
    # Session management (Feature 3+)
    # -------------------------------------------------------------------------
    session_ttl_seconds: int = 3600

    # -------------------------------------------------------------------------
    # Rate limiting (Feature 12)
    # -------------------------------------------------------------------------
    rate_limit_per_minute: int = 60

    # -------------------------------------------------------------------------
    # Domain / assistant identity
    # -------------------------------------------------------------------------
    assistant_name: str = "My AI Assistant"
    assistant_description: str = (
        "You are a helpful assistant. Replace this with your domain description."
    )

    def effective_voice_provider(self) -> str:
        """Return the provider to use for voice features.

        Falls back to the main LLM provider if VOICE_PROVIDER is not set,
        so students using a speech-capable provider everywhere don't need
        to set this separately.
        """
        return self.voice_provider or self.llm_provider

    def effective_vlm_provider(self) -> str:
        """Return the provider to use for image analysis (VLM).

        Falls back to the main LLM provider if VLM_PROVIDER is not set.
        If your main provider doesn't support vision (groq, cohere), set
        VLM_PROVIDER=openai or VLM_PROVIDER=ollama with a vision model.
        """
        return self.vlm_provider or self.llm_provider


# Single shared instance — import this object, don't instantiate Settings yourself.
settings = Settings()
