"""60dB large language model for LiveKit Agents.

Streams an OpenAI-style chat completion (SSE) and republishes each token as
a LiveKit ``ChatChunk`` so the agent speaks incrementally. Tool/function
calling is intentionally out of scope for this voice pipeline.
"""

from __future__ import annotations

import json

from livekit.agents import llm, utils

from .config import SixtyDBConfig, http_session


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str = "60db-chat",
        temperature: float = 0.7,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        self._config = SixtyDBConfig.resolve(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        conn_options=None,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = 1,
        parallel_tool_calls: bool | None = None,
    ) -> "LLMStream":
        return LLMStream(
            self,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            config=self._config,
            model=self._model,
            temperature=temperature
            if temperature is not None
            else self._temperature,
        )


def _to_wire_messages(chat_ctx: llm.ChatContext) -> list[dict]:
    """Flatten a LiveKit ChatContext into OpenAI-style message dicts."""
    messages: list[dict] = []
    for msg in chat_ctx.messages:
        content = msg.content
        if isinstance(content, list):
            # Voice pipeline content is text; join any string parts and drop
            # non-text (e.g. images), which this provider doesn't accept.
            content = " ".join(c for c in content if isinstance(c, str))
        messages.append({"role": str(msg.role), "content": content or ""})
    return messages


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm_: "LLM",
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
        config: SixtyDBConfig,
        model: str,
        temperature: float,
    ) -> None:
        super().__init__(llm_, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)
        self._config = config
        self._model = model
        self._temperature = temperature

    async def _run(self) -> None:
        request_id = utils.shortuuid()

        # 60dB API: OpenAI-compatible streaming chat completion.
        body = {
            "model": self._model,
            "temperature": self._temperature,
            "stream": True,
            "messages": _to_wire_messages(self._chat_ctx),
        }

        async with http_session().post(
            self._config.url("/chat/completions"),
            json=body,
            headers=self._config.auth_headers,
        ) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break

                event = json.loads(data)
                delta = (
                    event.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content")
                )
                if not delta:
                    continue

                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        request_id=request_id,
                        choices=[
                            llm.Choice(
                                delta=llm.ChoiceDelta(
                                    role="assistant", content=delta
                                ),
                                index=0,
                            )
                        ],
                    )
                )
