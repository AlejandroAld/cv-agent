"""Capa de proveedor de modelo.

El servidor habla Open Responses hacia afuera y Chat Completions hacia adentro.
Esa traducción vive aquí, así que cambiar de proveedor es una variable de
entorno y no tocar el agent loop.

Proveedores: openai | azure | compatible (cualquier base URL OpenAI-compatible)
| mock (determinista, para tests y CI sin gastar tokens).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .core import get_settings, log_event


class LLMError(RuntimeError):
    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


def _endpoint_y_headers() -> tuple[str, dict[str, str], str]:
    s = get_settings()
    if s.provider == "azure":
        if not (s.azure_endpoint and s.azure_deployment):
            raise LLMError("Azure OpenAI mal configurado: falta endpoint o deployment.", 500)
        url = (
            f"{s.azure_endpoint}/openai/deployments/{s.azure_deployment}"
            f"/chat/completions?api-version={s.azure_api_version}"
        )
        return url, {"api-key": s.azure_api_key}, s.azure_deployment
    if s.provider == "compatible":
        if not s.compat_base_url:
            raise LLMError("LLM_BASE_URL no configurado.", 500)
        return f"{s.compat_base_url}/chat/completions", {"Authorization": f"Bearer {s.compat_api_key}"}, s.model
    # openai por defecto
    if not s.openai_api_key:
        raise LLMError("OPENAI_API_KEY no configurado.", 500)
    return "https://api.openai.com/v1/chat/completions", {"Authorization": f"Bearer {s.openai_api_key}"}, s.model


def _payload(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float | None,
    max_tokens: int | None,
    model: str,
    stream: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens:
        body["max_tokens"] = max_tokens
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


async def stream_chat(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Emite eventos normalizados del proveedor.

    Tipos emitidos:
      {"t": "text", "delta": str}
      {"t": "tool", "index": int, "id": str|None, "name": str|None, "args_delta": str}
      {"t": "done", "finish_reason": str|None, "usage": dict|None}
    """
    s = get_settings()

    if s.provider == "mock":
        async for ev in _mock_stream(messages, tools):
            yield ev
        return

    url, headers, model = _endpoint_y_headers()
    headers["Content-Type"] = "application/json"
    body = _payload(messages, tools, temperature, max_tokens, model, stream=True)

    usage: dict[str, Any] | None = None
    finish_reason: str | None = None

    async with httpx.AsyncClient(timeout=s.request_timeout_s) as client:
        async with client.stream("POST", url, headers=headers, json=body) as resp:
            if resp.status_code >= 400:
                detalle = (await resp.aread()).decode("utf-8", "replace")[:400]
                log_event("llm_error", status=resp.status_code, detalle=detalle)
                raise LLMError(f"El proveedor de modelo respondió {resp.status_code}.", 502)

            async for linea in resp.aiter_lines():
                if not linea or not linea.startswith("data:"):
                    continue
                data = linea[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}

                if delta.get("content"):
                    yield {"t": "text", "delta": delta["content"]}

                for tc in delta.get("tool_calls") or []:
                    yield {
                        "t": "tool",
                        "index": tc.get("index", 0),
                        "id": tc.get("id"),
                        "name": (tc.get("function") or {}).get("name"),
                        "args_delta": (tc.get("function") or {}).get("arguments") or "",
                    }

    yield {"t": "done", "finish_reason": finish_reason, "usage": usage}


# ---------------------------------------------------------------------------
# Mock determinista: permite correr tests de contrato y CI sin credenciales.
# ---------------------------------------------------------------------------
async def _mock_stream(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> AsyncIterator[dict[str, Any]]:
    ultimo = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            ultimo = str(m.get("content") or "")
            break
    ya_hubo_tool = any(m.get("role") == "tool" for m in messages)

    quiere_tool = (
        tools
        and not ya_hubo_tool
        and any(k in ultimo.lower() for k in ("proyecto", "vacante", "encaj", "contacto"))
    )
    if quiere_tool:
        yield {"t": "tool", "index": 0, "id": "call_mock_1", "name": "buscar_en_perfil", "args_delta": ""}
        yield {"t": "tool", "index": 0, "id": None, "name": None, "args_delta": json.dumps({"consulta": ultimo[:60]})}
        yield {"t": "done", "finish_reason": "tool_calls", "usage": None}
        return

    texto = f"[mock] Recibí: {ultimo[:120]}" if ultimo else "[mock] Sin entrada."
    for pedazo in texto.split(" "):
        yield {"t": "text", "delta": pedazo + " "}
    yield {"t": "done", "finish_reason": "stop", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
