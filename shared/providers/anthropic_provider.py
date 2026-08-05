"""
Provider implementation that wraps the Anthropic Messages API.

Key differences from the OpenAI API that this class handles transparently:
  1. The system message is a separate `system` parameter, not a message in the list.
  2. Tool schemas use Anthropic's format ("input_schema" not "parameters").
  3. Responses can contain multiple content blocks (text + tool_use mixed).
  4. JSON mode is not a native API feature — we approximate it via prompting.
  5. Anthropic does not provide speech (STT/TTS) APIs.
"""
import anthropic
from typing import Optional, List

from shared.config import settings
from shared.providers.base import LLMProvider, LLMResponse


def _extract_system_message(messages: List[dict]) -> tuple[str, List[dict]]:
    """
    Split an OpenAI-style messages list into (system_content, remaining_messages).

    Anthropic requires the system instruction to be passed as a separate parameter,
    not mixed into the messages array. This helper does that extraction.
    """
    system_parts: List[str] = []
    other_messages: List[dict] = []

    for msg in messages:
        if msg.get("role") == "system":
            system_parts.append(msg["content"])
        else:
            other_messages.append(msg)

    return "\n\n".join(system_parts), other_messages


def _translate_tools(openai_tools: List[dict]) -> List[dict]:
    """
    Convert OpenAI-style tool schemas to Anthropic's format.

    OpenAI:   {"type": "function", "function": {"name", "description", "parameters"}}
    Anthropic: {"name", "description", "input_schema"}
    """
    anthropic_tools = []
    for tool in openai_tools:
        fn = tool.get("function", {})
        anthropic_tools.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                # "parameters" in OpenAI is a JSON Schema dict — Anthropic calls
                # the same thing "input_schema".
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return anthropic_tools


class AnthropicProvider(LLMProvider):
    """Talks to the Anthropic Messages API with full translation to/from the common interface."""

    def __init__(self) -> None:
        self._api_key = settings.anthropic_api_key
        self._model = settings.anthropic_model
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

    async def chat(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        tools: Optional[List[dict]] = None,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        """Send messages to the Anthropic API and return a normalized response."""
        system_content, remaining_messages = _extract_system_message(messages)

        # Anthropic doesn't have a native JSON mode. If JSON output is requested,
        # we append an instruction to the system prompt. Callers should still use
        # tolerant JSON parsing (try/except) because this is a best-effort workaround,
        # not a hard guarantee.
        if response_format and response_format.get("type") == "json_object":
            system_content += "\n\nRespond with valid JSON only. Do not include any explanation, markdown, or text outside the JSON object."

        kwargs: dict = {}
        if tools:
            kwargs["tools"] = _translate_tools(tools)

        response = await self._client.messages.create(
            model=self._model,
            system=system_content or anthropic.NOT_GIVEN,
            messages=remaining_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Anthropic returns content as a list of blocks; we normalize these.
        text_parts: list[str] = []
        normalized_tool_calls: list[dict] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                normalized_tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        # block.input is already a dict for Anthropic — no JSON parsing needed.
                        "arguments": block.input,
                    }
                )

        return LLMResponse(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=normalized_tool_calls,
            raw=response.model_dump(),
            provider="anthropic",
            model=self._model,
        )

    async def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        raise NotImplementedError(
            "Anthropic does not provide speech APIs. "
            "Set VOICE_PROVIDER=openai (or another speech-capable provider) in .env "
            "to use voice features while keeping LLM_PROVIDER=anthropic for chat. "
            "See docs/provider-setup-guide.md for details."
        )

    async def synthesize_speech(self, text: str, voice: str = "default") -> bytes:
        raise NotImplementedError(
            "Anthropic does not provide speech APIs. "
            "Set VOICE_PROVIDER=openai (or another speech-capable provider) in .env "
            "to use voice features while keeping LLM_PROVIDER=anthropic for chat. "
            "See docs/provider-setup-guide.md for details."
        )

    async def analyze_image(self, image_bytes: bytes, prompt: str, detail: str = "auto") -> LLMResponse:
        """Analyze an image using Claude's native vision capabilities (Claude 3+ all support vision)."""
        import base64

        b64 = base64.b64encode(image_bytes).decode()

        # Anthropic's vision format: image goes BEFORE the text prompt in the content array.
        # The "detail" parameter doesn't apply here; Anthropic controls resolution internally.
        # Note: media_type is declared as image/jpeg — works for JPEG, PNG, GIF, WebP.
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )

        text_parts = [block.text for block in response.content if block.type == "text"]
        return LLMResponse(
            content="\n".join(text_parts) if text_parts else None,
            provider="anthropic",
            model=self._model,
            raw=response.model_dump(),
        )
