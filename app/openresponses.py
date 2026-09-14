"""Traducción entre el formato de cable Open Responses y Chat Completions.

Referencia: https://www.openresponses.org/specification
"""

from __future__ import annotations

import json
import secrets
import time
from typing import Any


def nuevo_id(prefijo: str) -> str:
    return f"{prefijo}_{secrets.token_hex(16)}"


# ---------------------------------------------------------------------------
# Entrada: items Open Responses -> mensajes Chat Completions
# ---------------------------------------------------------------------------
def _partes_a_contenido(partes: Any) -> Any:
    """Convierte content (string o lista de partes) al formato del proveedor."""
    if isinstance(partes, str):
        return partes
    if not isinstance(partes, list):
        return ""

    trozos: list[dict[str, Any]] = []
    solo_texto = True
    for parte in partes:
        if not isinstance(parte, dict):
            if isinstance(parte, str):
                trozos.append({"type": "text", "text": parte})
            continue
        tipo = parte.get("type")
        if tipo in ("input_text", "output_text", "text"):
            trozos.append({"type": "text", "text": parte.get("text", "")})
        elif tipo == "refusal":
            trozos.append({"type": "text", "text": parte.get("refusal", "")})
        elif tipo == "input_image":
            url = parte.get("image_url")
            if url:
                solo_texto = False
                trozos.append({"type": "image_url", "image_url": {"url": url, "detail": parte.get("detail", "auto")}})
        elif tipo == "input_file":
            nombre = parte.get("filename") or parte.get("file_url") or "archivo"
            trozos.append({"type": "text", "text": f"[archivo adjunto: {nombre}]"})

    if solo_texto:
        return "\n".join(t.get("text", "") for t in trozos).strip()
    return trozos


def input_a_mensajes(entrada: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Devuelve (mensajes de chat, instrucciones de rol system/developer)."""
    mensajes: list[dict[str, Any]] = []
    sistemas: list[str] = []

    if entrada is None:
        return mensajes, sistemas
    if isinstance(entrada, str):
        return [{"role": "user", "content": entrada}], sistemas
    if isinstance(entrada, dict):
        entrada = [entrada]

    pendientes_tool: dict[str, str] = {}

    for item in entrada:
        if not isinstance(item, dict):
            if isinstance(item, str):
                mensajes.append({"role": "user", "content": item})
            continue

        tipo = item.get("type") or ("message" if item.get("role") else None)

        if tipo == "message":
            rol = item.get("role", "user")
            contenido = _partes_a_contenido(item.get("content"))
            if rol in ("system", "developer"):
                if contenido:
                    sistemas.append(contenido if isinstance(contenido, str) else json.dumps(contenido))
                continue
            if rol == "assistant":
                mensajes.append({"role": "assistant", "content": contenido or ""})
            else:
                mensajes.append({"role": "user", "content": contenido})

        elif tipo == "function_call":
            call_id = item.get("call_id") or item.get("id") or nuevo_id("call")
            pendientes_tool[call_id] = item.get("name", "")
            mensajes.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": item.get("name", ""), "arguments": item.get("arguments", "{}")},
                        }
                    ],
                }
            )

        elif tipo == "function_call_output":
            salida = item.get("output")
            if not isinstance(salida, str):
                salida = json.dumps(salida, ensure_ascii=False, default=str)
            mensajes.append({"role": "tool", "tool_call_id": item.get("call_id", ""), "content": salida})

        # reasoning / compaction / item_reference: se ignoran, este servidor no
        # los persiste y el spec permite tratarlos como opacos.

    return mensajes, sistemas


# ---------------------------------------------------------------------------
# Salida: objeto Response
# ---------------------------------------------------------------------------
def item_mensaje(texto: str, item_id: str | None = None, status: str = "completed") -> dict[str, Any]:
    return {
        "type": "message",
        "id": item_id or nuevo_id("msg"),
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": texto, "annotations": []}],
    }


def item_function_call(call_id: str, nombre: str, argumentos: str, item_id: str | None = None) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": item_id or nuevo_id("fc"),
        "call_id": call_id,
        "name": nombre,
        "arguments": argumentos or "{}",
        "status": "completed",
    }


def construir_response(
    *,
    response_id: str,
    modelo: str,
    salida: list[dict[str, Any]],
    peticion: dict[str, Any],
    status: str = "completed",
    usage: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    creado: int | None = None,
) -> dict[str, Any]:
    """Objeto Response con todos los campos que el spec marca como required."""
    ahora = int(time.time())
    u = usage or {}
    entrada_tok = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
    salida_tok = int(u.get("completion_tokens") or u.get("output_tokens") or 0)

    return {
        "id": response_id,
        "object": "response",
        "created_at": creado or ahora,
        "completed_at": ahora if status in ("completed", "incomplete", "failed") else None,
        "status": status,
        "error": error,
        "incomplete_details": None,
        "model": modelo,
        "previous_response_id": peticion.get("previous_response_id"),
        "instructions": peticion.get("instructions"),
        "output": salida,
        # conveniencia: varios SDKs la leen directo
        "output_text": "".join(
            c.get("text", "")
            for it in salida
            if it.get("type") == "message"
            for c in it.get("content", [])
            if c.get("type") == "output_text"
        ),
        "tools": peticion.get("tools") or [],
        "tool_choice": peticion.get("tool_choice") or "auto",
        "truncation": peticion.get("truncation") or "disabled",
        "parallel_tool_calls": bool(peticion.get("parallel_tool_calls", True)),
        "text": peticion.get("text") or {"format": {"type": "text"}},
        "top_p": peticion.get("top_p", 1.0),
        "presence_penalty": peticion.get("presence_penalty", 0.0),
        "frequency_penalty": peticion.get("frequency_penalty", 0.0),
        "top_logprobs": peticion.get("top_logprobs", 0),
        "temperature": peticion.get("temperature", 1.0),
        "reasoning": {"effort": (peticion.get("reasoning") or {}).get("effort"), "summary": None},
        "max_output_tokens": peticion.get("max_output_tokens"),
        "max_tool_calls": peticion.get("max_tool_calls"),
        "store": bool(peticion.get("store", False)),
        "background": bool(peticion.get("background", False)),
        "service_tier": peticion.get("service_tier") or "default",
        "metadata": peticion.get("metadata") or {},
        "safety_identifier": peticion.get("safety_identifier"),
        "prompt_cache_key": peticion.get("prompt_cache_key"),
        "usage": {
            "input_tokens": entrada_tok,
            "output_tokens": salida_tok,
            "total_tokens": int(u.get("total_tokens") or (entrada_tok + salida_tok)),
            "input_tokens_details": {"cached_tokens": int((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0))},
            "output_tokens_details": {"reasoning_tokens": int((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0))},
        },
    }


def cuerpo_error(mensaje: str, tipo: str = "invalid_request", code: str | None = None, param: str | None = None) -> dict[str, Any]:
    return {"error": {"message": mensaje, "type": tipo, "param": param, "code": code}}


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------
class Emisor:
    """Numera los eventos y los formatea como SSE.

    El spec exige que el campo `event:` coincida con el `type` del cuerpo y que
    el evento terminal sea literalmente `[DONE]`.
    """

    def __init__(self) -> None:
        self._n = 0

    def evento(self, tipo: str, **campos: Any) -> str:
        self._n += 1
        cuerpo = {"type": tipo, "sequence_number": self._n, **campos}
        return f"event: {tipo}\ndata: {json.dumps(cuerpo, ensure_ascii=False)}\n\n"

    @staticmethod
    def fin() -> str:
        return "data: [DONE]\n\n"
