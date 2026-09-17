"""Capa de proveedor de modelo: Responses API.

El servidor habla Open Responses hacia afuera y hacia adentro. La migración
desde Chat Completions no fue estética: los modelos de razonamiento (serie
GPT-5) rechazan `temperature`, `top_p` y las penalties, usan
`max_completion_tokens` en vez de `max_tokens`, y para tool calling requieren
esta API. Mandarles el cuerpo viejo devuelve 400.

Proveedores: azure | openai | compatible (cualquier base URL que exponga
/responses) | mock (determinista, para tests y CI sin gastar tokens).
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


# ---------------------------------------------------------------------------
# Resolución de proveedor
# ---------------------------------------------------------------------------
def _base_azure(endpoint: str) -> str:
    """El portal de Foundry entrega el endpoint ya con /openai/v1.

    Se acepta con o sin el sufijo para que configurar la variable no dependa de
    dónde se haya copiado la URL.
    """
    base = endpoint.rstrip("/")
    return base if base.endswith("/openai/v1") else f"{base}/openai/v1"


def _endpoint_y_headers() -> tuple[str, dict[str, str], str]:
    """Devuelve (url de /responses, headers de auth, modelo a pedir)."""
    s = get_settings()

    if s.provider == "azure":
        if not (s.azure_endpoint and s.azure_deployment):
            raise LLMError("Azure OpenAI mal configurado: falta endpoint o deployment.", 500)
        url = f"{_base_azure(s.azure_endpoint)}/responses"
        # La v1 GA no pide api-version; sólo se agrega si alguien la define.
        if s.azure_api_version:
            url += f"?api-version={s.azure_api_version}"
        # Azure autentica con el header api-key, no con Bearer.
        return url, {"api-key": s.azure_api_key}, s.azure_deployment

    if s.provider == "compatible":
        if not s.compat_base_url:
            raise LLMError("LLM_BASE_URL no configurado.", 500)
        return (
            f"{s.compat_base_url}/responses",
            {"Authorization": f"Bearer {s.compat_api_key}"},
            s.model,
        )

    if not s.openai_api_key:
        raise LLMError("OPENAI_API_KEY no configurado.", 500)
    return (
        "https://api.openai.com/v1/responses",
        {"Authorization": f"Bearer {s.openai_api_key}"},
        s.model,
    )


def construir_cuerpo(
    entrada: list[dict[str, Any]],
    instructions: str | None,
    tools: list[dict[str, Any]] | None,
    max_output_tokens: int | None,
    reasoning_effort: str | None,
    model: str,
) -> dict[str, Any]:
    """Cuerpo de POST /responses.

    NUNCA incluye temperature, top_p, presence_penalty, frequency_penalty ni
    max_tokens: un modelo de razonamiento responde 400 ante cualquiera de
    ellos. Si el cliente los manda, se hacen eco en el objeto Response pero no
    viajan al proveedor.
    """
    cuerpo: dict[str, Any] = {
        "model": model,
        "input": entrada,
        "stream": True,
        # El estado de conversación lo lleva este servidor, no el proveedor.
        "store": False,
    }
    if instructions:
        cuerpo["instructions"] = instructions
    if tools:
        cuerpo["tools"] = tools
        cuerpo["tool_choice"] = "auto"
    if max_output_tokens:
        cuerpo["max_output_tokens"] = max_output_tokens
    if reasoning_effort:
        cuerpo["reasoning"] = {"effort": reasoning_effort}
    return cuerpo


def _normalizar_usage(u: dict[str, Any] | None) -> dict[str, Any] | None:
    """Deja el usage listo para `construir_response`, que lee ambos dialectos."""
    if not u:
        return None
    salida = dict(u)
    if u.get("input_tokens_details") and "prompt_tokens_details" not in salida:
        salida["prompt_tokens_details"] = u["input_tokens_details"]
    if u.get("output_tokens_details") and "completion_tokens_details" not in salida:
        salida["completion_tokens_details"] = u["output_tokens_details"]
    return salida


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
async def stream_agente(
    entrada: list[dict[str, Any]],
    *,
    instructions: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Emite eventos normalizados del proveedor.

    Tipos emitidos:
      {"t": "text", "delta": str}
      {"t": "tool", "call_id": str, "name": str, "arguments": str}
      {"t": "done", "usage": dict|None, "output": list}

    Los argumentos de una llamada llegan completos en `response.output_item.done`,
    así que no hay ensamblado por índice: cuando se emite un evento "tool", el
    JSON de argumentos ya está entero.

    `output` son los items crudos del modelo —incluidos los de razonamiento—,
    que el bucle vuelve a mandar tal cual en la siguiente vuelta.
    """
    s = get_settings()

    if s.provider == "mock":
        async for ev in _mock_stream(entrada, tools):
            yield ev
        return

    url, headers, model = _endpoint_y_headers()
    headers["Content-Type"] = "application/json"
    cuerpo = construir_cuerpo(
        entrada, instructions, tools, max_output_tokens, reasoning_effort, model
    )

    usage: dict[str, Any] | None = None
    items: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=s.request_timeout_s) as client:
        async with client.stream("POST", url, headers=headers, json=cuerpo) as resp:
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
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue

                tipo = ev.get("type")

                if tipo == "response.output_text.delta":
                    if ev.get("delta"):
                        yield {"t": "text", "delta": ev["delta"]}

                elif tipo == "response.output_item.done":
                    item = ev.get("item") or {}
                    items.append(item)
                    if item.get("type") == "function_call":
                        yield {
                            "t": "tool",
                            "call_id": item.get("call_id") or item.get("id") or "",
                            "name": item.get("name") or "",
                            "arguments": item.get("arguments") or "{}",
                        }

                elif tipo in ("response.completed", "response.incomplete"):
                    respuesta = ev.get("response") or {}
                    usage = respuesta.get("usage") or usage
                    # El objeto final manda sobre lo acumulado item por item.
                    if respuesta.get("output"):
                        items = respuesta["output"]

                elif tipo in ("response.failed", "error"):
                    respuesta = ev.get("response") or {}
                    detalle = (respuesta.get("error") or {}).get("message") or ev.get("message") or ""
                    log_event("llm_error", tipo=tipo, detalle=str(detalle)[:400])
                    raise LLMError("El proveedor de modelo no completó la respuesta.", 502)

    yield {"t": "done", "usage": _normalizar_usage(usage), "output": items}


# ---------------------------------------------------------------------------
# Mock determinista: permite correr tests de contrato y CI sin credenciales.
# ---------------------------------------------------------------------------
def _ultimo_texto_usuario(entrada: list[dict[str, Any]]) -> str:
    for item in reversed(entrada):
        if item.get("type") == "message" and item.get("role") == "user":
            return "".join(
                p.get("text", "") for p in item.get("content") or [] if isinstance(p, dict)
            )
    return ""


async def _mock_stream(
    entrada: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> AsyncIterator[dict[str, Any]]:
    """Imita la forma real: siempre hay un item de razonamiento en el output."""
    ultimo = _ultimo_texto_usuario(entrada)
    ya_hubo_tool = any(i.get("type") == "function_call_output" for i in entrada)

    razonamiento = {"type": "reasoning", "id": "rs_mock_1", "summary": []}

    quiere_tool = (
        tools
        and not ya_hubo_tool
        and any(k in ultimo.lower() for k in ("proyecto", "vacante", "encaj", "contacto"))
    )
    if quiere_tool:
        llamada = {
            "type": "function_call",
            "id": "fc_mock_1",
            "call_id": "call_mock_1",
            "name": "buscar_en_perfil",
            "arguments": json.dumps({"consulta": ultimo[:60]}),
            "status": "completed",
        }
        yield {
            "t": "tool",
            "call_id": llamada["call_id"],
            "name": llamada["name"],
            "arguments": llamada["arguments"],
        }
        yield {"t": "done", "usage": None, "output": [razonamiento, llamada]}
        return

    texto = f"[mock] Recibí: {ultimo[:120]}" if ultimo else "[mock] Sin entrada."
    for pedazo in texto.split(" "):
        yield {"t": "text", "delta": pedazo + " "}
    yield {
        "t": "done",
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "output": [
            razonamiento,
            {
                "type": "message",
                "id": "msg_mock_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": texto + " ", "annotations": []}],
            },
        ],
    }
