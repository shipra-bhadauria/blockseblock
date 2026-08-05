"""
Public interface for all LLM calls in the AI Engineering Bootcamp.

Every feature imports from here. The underlying provider is configured entirely
through .env — switching providers never requires changing feature code.

Quick reference:
    from shared.llm_client import call_llm, transcribe_audio, synthesize_speech, analyze_image
    from shared.providers.base import LLMResponse
"""
import logging
from typing import Optional, List

from shared.providers.base import LLMResponse
from shared.providers.factory import get_provider

logger = logging.getLogger(__name__)


async def call_llm(
    messages: List[dict],
    temperature: float = 0.7,
    max_tokens: int = 1000,
    tools: Optional[List[dict]] = None,
    response_format: Optional[dict] = None,
    **kwargs,
) -> LLMResponse:
    """
    Send a conversation to the configured LLM and return the response.

    Args:
        messages: Conversation history as a list of message dicts in the
                  OpenAI style: [{"role": "system"|"user"|"assistant", "content": "..."}].
                  Every provider's implementation translates this into its own
                  native format internally.
        temperature: Controls randomness. 0 = predictable, 1 = creative.
        max_tokens: Maximum response length in tokens (~¾ of a word each).
        tools: OpenAI-style tool schemas if the model should be able to call
               functions. None means plain chat with no tools.
        response_format: OpenAI-style hint for structured output, e.g.
                         {"type": "json_object"} to request JSON.
        **kwargs: Reserved for future use; currently ignored.

    Returns:
        LLMResponse with:
          .content     — the model's text reply (str or None if only tools called)
          .tool_calls  — list of {"id", "name", "arguments"} dicts (empty if none)
          .provider    — which provider answered ("openai", "anthropic", etc.)
          .model       — which model was used
          .raw         — the original provider response dict, for debugging
    """
    provider = get_provider("llm")
    return await provider.chat(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        response_format=response_format,
    )


async def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    """
    Convert spoken audio to text (Speech-to-Text).

    Uses the VOICE_PROVIDER if set in .env, otherwise the main LLM_PROVIDER.
    This lets students run chat on a provider that doesn't support speech while
    still using a speech-capable provider for Feature 10.

    Args:
        audio_bytes: Raw audio file bytes (WAV, MP3, WebM, etc.).
        filename: The original filename including extension (used to detect format).

    Returns:
        The transcribed text as a plain string.
    """
    provider = get_provider("voice")
    return await provider.transcribe(audio_bytes, filename)


async def synthesize_speech(text: str, voice: str = "default") -> bytes:
    """
    Convert text to spoken audio (Text-to-Speech).

    Uses the VOICE_PROVIDER if set in .env, otherwise the main LLM_PROVIDER.

    Args:
        text: The text to speak aloud.
        voice: Provider-specific voice name. Use "default" for the provider's
               default voice.

    Returns:
        Raw audio bytes (typically MP3), ready to send to the browser.
    """
    provider = get_provider("voice")
    return await provider.synthesize_speech(text, voice)


async def analyze_image(
    image_bytes: bytes,
    prompt: str,
    detail: str = "auto",
) -> LLMResponse:
    """
    Analyze an image with a text prompt (Vision Language Model / VLM).

    Uses the VLM_PROVIDER if set in .env, otherwise the main LLM_PROVIDER.
    Set VLM_MODEL in .env to select the vision model:
      - OpenAI: gpt-4o or gpt-4o-mini (both support vision)
      - Ollama: llava, phi3:vision, or llava-phi3 (free, local, private)

    Args:
        image_bytes: Raw image bytes (JPEG, PNG, WebP, GIF).
        prompt:      Question or instruction for the model about the image.
        detail:      "auto" (default), "high" (fine text), or "low" (fast overview).

    Returns:
        LLMResponse with .content containing the answer/analysis.
        Falls back gracefully with a helpful message if the provider does not
        support vision — instead of crashing, it returns an LLMResponse explaining
        what to set in .env.
    """
    try:
        provider = get_provider("vlm")
        return await provider.analyze_image(image_bytes, prompt, detail)
    except NotImplementedError as exc:
        logger.warning(
            "VLM_MODEL does not support image input — %s. "
            "Set VLM_PROVIDER=openai (gpt-4o) or VLM_PROVIDER=ollama (llava) in .env.",
            exc,
        )
        return LLMResponse(
            content=(
                "Image analysis is not available with the current configuration. "
                "Set VLM_PROVIDER=openai and VLM_MODEL=gpt-4o, or "
                "VLM_PROVIDER=ollama and VLM_MODEL=llava in your .env file."
            ),
            provider="none",
            model="none",
        )
