"""Formato de cable Open Responses: normalización de entrada y armado de salida.

El proveedor consume los mismos items que expone el spec, así que aquí ya no
hay traducción de formato: hay normalización de lo que manda el cliente y
construcción del objeto Response que se le devuelve.

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
# Entrada: normalización a items de Open Responses
# ---------------------------------------------------------------------------
# La Responses API consume los mismos items que expone Open Responses, así que
# aquí ya no se traduce a mensajes de chat: sólo se normaliza lo que manda el
# cliente a la forma exacta que el proveedor acepta.

_PARTES_TEXTO = ("input_text", "output_text", "text", "summary_text")


def _tipo_texto(rol: str) -> str:
    return "output_text" if rol == "assistant" else "input_text"


def _item_usuario(texto: str) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": texto}]}


def _normalizar_contenido(partes: Any, rol: str) -> list[dict[str, Any]]:
    """Content de un item message, en partes válidas para la Responses API.

    El tipo de la parte depende del rol: la API rechaza un `input_text` dentro
    de un mensaje del assistant y un `output_text` dentro de uno del usuario,
    así que se fuerza el que corresponde en vez de confiar en lo que llegó.
    """
    tipo_txt = _tipo_texto(rol)

    if isinstance(partes, str):
        return [{"type": tipo_txt, "text": partes}] if partes else []
    if not isinstance(partes, list):
        return []

    salida: list[dict[str, Any]] = []
    for parte in partes:
        if isinstance(parte, str):
            if parte:
                salida.append({"type": tipo_txt, "text": parte})
            continue
        if not isinstance(parte, dict):
            continue

        tipo = parte.get("type")
        if tipo in _PARTES_TEXTO:
            texto = parte.get("text", "")
            if texto:
                salida.append({"type": tipo_txt, "text": texto})
        elif tipo == "refusal":
            texto = parte.get("refusal", "")
            if texto:
                salida.append({"type": tipo_txt, "text": texto})
        elif tipo == "input_image" and rol != "assistant":
            url = parte.get("image_url")
            if url:
                salida.append(
                    {"type": "input_image", "image_url": url, "detail": parte.get("detail", "auto")}
                )
        elif tipo == "input_file":
            nombre = parte.get("filename") or parte.get("file_url") or "archivo"
            salida.append({"type": tipo_txt, "text": f"[archivo adjunto: {nombre}]"})

    return salida


def _texto_plano(contenido: Any) -> str:
    """Aplana el content de un mensaje system/developer a texto."""
    if isinstance(contenido, str):
        return contenido.strip()
    partes = _normalizar_contenido(contenido, "user")
    return "\n".join(p.get("text", "") for p in partes if p.get("type") == "input_text").strip()


def input_a_items(entrada: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Normaliza el campo `input` a items de Open Responses.

    Devuelve (items, textos_system). Los roles system/developer no viajan como
    items: se extraen aparte para concatenarlos a `instructions`, que es donde
    la Responses API espera las reglas del agente.
    """
    items: list[dict[str, Any]] = []
    sistemas: list[str] = []

    if entrada is None:
        return items, sistemas
    if isinstance(entrada, str):
        return ([_item_usuario(entrada)] if entrada else []), sistemas
    if isinstance(entrada, dict):
        entrada = [entrada]
    if not isinstance(entrada, list):
        return items, sistemas

    for item in entrada:
        if isinstance(item, str):
            if item:
                items.append(_item_usuario(item))
            continue
        if not isinstance(item, dict):
            continue

        tipo = item.get("type") or ("message" if item.get("role") else None)

        if tipo == "message":
            rol_crudo = item.get("role", "user")
            if rol_crudo in ("system", "developer"):
                texto = _texto_plano(item.get("content"))
                if texto:
                    sistemas.append(texto)
                continue
            rol = "assistant" if rol_crudo == "assistant" else "user"
            contenido = _normalizar_contenido(item.get("content"), rol)
            if not contenido:
                if rol == "assistant":
                    continue  # un turno vacío del assistant no aporta contexto
                contenido = [{"type": "input_text", "text": ""}]
            items.append({"type": "message", "role": rol, "content": contenido})

        elif tipo == "function_call":
            items.append(
                {
                    "type": "function_call",
                    "call_id": item.get("call_id") or item.get("id") or nuevo_id("call"),
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments") or "{}",
                }
            )

        elif tipo == "function_call_output":
            salida = item.get("output")
            if not isinstance(salida, str):
                salida = json.dumps(salida, ensure_ascii=False, default=str)
            items.append(
                {"type": "function_call_output", "call_id": item.get("call_id", ""), "output": salida}
            )

        # reasoning / compaction / item_reference del cliente: se ignoran. Un
        # item de razonamiento sólo es válido junto al id que emitió el propio
        # proveedor, así que reenviar el de otra respuesta sería un 400.

    return items, sistemas


# ---------------------------------------------------------------------------
# Higiene del historial
# ---------------------------------------------------------------------------
def sin_reasoning(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Quita los items de razonamiento.

    El razonamiento es estado interno del modelo: vuelve al proveedor en la
    siguiente vuelta del bucle, pero nunca sale hacia el cliente.
    """
    return [i for i in items if i.get("type") != "reasoning"]


def _tamano(items: list[dict[str, Any]]) -> int:
    return sum(len(json.dumps(i, default=str)) for i in items)


def _call_id(item: dict[str, Any]) -> str:
    return item.get("call_id") or item.get("id") or ""


def recortar(items: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    """Corta el historial más viejo hasta caber en el presupuesto de entrada.

    Nunca deja un `function_call_output` huérfano: si el recorte se lleva un
    `function_call`, se lleva también su resultado. Un output sin su llamada no
    es una respuesta degradada, es un 400 del proveedor.
    """
    restantes = list(items)

    while _tamano(restantes) > max_chars and len(restantes) > 1:
        fuera = restantes.pop(0)
        if fuera.get("type") == "function_call":
            cid = _call_id(fuera)
            restantes = [
                i
                for i in restantes
                if not (i.get("type") == "function_call_output" and _call_id(i) == cid)
            ]

    # Y al revés: un output cuya llamada ya no está también se descarta.
    llamadas = {_call_id(i) for i in restantes if i.get("type") == "function_call"}
    return [
        i for i in restantes if i.get("type") != "function_call_output" or _call_id(i) in llamadas
    ]


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
    extra_metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Objeto Response con todos los campos que el spec marca como required.

    `extra_metadata` son pares que el servidor añade a la `metadata` del
    cliente (que se conserva entera): lo que el bucle hizo con herramientas
    internas, para que una corrida grabada desde fuera pueda saberlo.
    """
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
        # Default true, como la plataforma. El servidor lo respeta: ver _guardar_estado.
        "store": bool(peticion.get("store", True)),
        "background": bool(peticion.get("background", False)),
        "service_tier": peticion.get("service_tier") or "default",
        "metadata": {**(peticion.get("metadata") or {}), **(extra_metadata or {})},
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
