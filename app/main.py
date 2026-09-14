"""Servidor Open Responses del agente de CV.

Superficie pública:
  POST /v1/responses              síncrono y streaming (SSE)
  GET  /v1/responses/{id}          recuperar una respuesta almacenada
  GET  /healthz                    liveness
  GET  /.well-known/agent-card.json tarjeta A2A para autodescubrimiento
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import openresponses as orx
from .agent_brain import (
    HERRAMIENTAS_INTERNAS,
    NOMBRES_INTERNOS,
    construir_system_prompt,
    ejecutar_herramienta,
    serializar_resultado,
)
from .core import ROOT, get_profile, get_settings, log_event
from .llm import LLMError, stream_chat

app = FastAPI(title="Agente de CV — Open Responses", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Estado de conversación en memoria para previous_response_id.
# Suficiente para una instancia; en multi-instancia se sustituye por Redis o
# Firestore sin tocar el resto del código (está aislado en estas funciones).
_ESTADO: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_TTL_S = 2 * 60 * 60
_MAX_ESTADOS = 500


def _guardar_estado(rid: str, mensajes: list[dict[str, Any]], respuesta: dict[str, Any]) -> None:
    _ESTADO[rid] = {"mensajes": mensajes, "respuesta": respuesta, "exp": time.time() + _TTL_S}
    _ESTADO.move_to_end(rid)
    while len(_ESTADO) > _MAX_ESTADOS:
        _ESTADO.popitem(last=False)


def _leer_estado(rid: str) -> dict[str, Any] | None:
    entrada = _ESTADO.get(rid)
    if not entrada:
        return None
    if entrada["exp"] < time.time():
        _ESTADO.pop(rid, None)
        return None
    return entrada


# ---------------------------------------------------------------------------
# Límite de uso para la demo pública
# ---------------------------------------------------------------------------
# El endpoint Open Responses está protegido por Bearer. La demo del sitio no
# puede estarlo: un token en el navegador es un token público. Así que se
# protege por consumo, que es lo que realmente cuesta dinero.
_VISITAS: dict[str, list[float]] = {}
_LIMITE = int(os.getenv("DEMO_RATE_LIMIT", "12"))
_VENTANA_S = float(os.getenv("DEMO_RATE_WINDOW_S", "3600"))


def _ip(request: Request) -> str:
    reenviado = request.headers.get("x-forwarded-for", "")
    if reenviado:
        return reenviado.split(",")[0].strip()
    return request.client.host if request.client else "desconocido"


def _dentro_del_limite(ip: str) -> bool:
    ahora = time.time()
    marcas = [t for t in _VISITAS.get(ip, []) if ahora - t < _VENTANA_S]
    if len(marcas) >= _LIMITE:
        _VISITAS[ip] = marcas
        return False
    marcas.append(ahora)
    _VISITAS[ip] = marcas
    if len(_VISITAS) > 5000:  # poda simple
        for k in [k for k, v in list(_VISITAS.items()) if not v or ahora - max(v) > _VENTANA_S][:2000]:
            _VISITAS.pop(k, None)
    return True


# ---------------------------------------------------------------------------
# Autenticación
# ---------------------------------------------------------------------------
def _autorizado(header: str | None) -> bool:
    esperado = get_settings().agent_api_key
    if not esperado:
        return True  # endpoint abierto: sólo para desarrollo local
    if not header:
        return False
    token = header[7:].strip() if header.lower().startswith("bearer ") else header.strip()
    # comparación en tiempo constante
    if len(token) != len(esperado):
        return False
    return sum(a != b for a, b in zip(token, esperado)) == 0


# ---------------------------------------------------------------------------
# Herramientas del cliente
# ---------------------------------------------------------------------------
def _tools_cliente(peticion: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    """Convierte FunctionToolParam de Open Responses a formato Chat Completions."""
    salida, nombres = [], set()
    for t in peticion.get("tools") or []:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue  # herramientas hospedadas por otros proveedores: se ignoran
        nombre = t.get("name") or (t.get("function") or {}).get("name")
        if not nombre or nombre in NOMBRES_INTERNOS:
            continue
        salida.append(
            {
                "type": "function",
                "function": {
                    "name": nombre,
                    "description": t.get("description") or "",
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
        nombres.add(nombre)
    return salida, nombres


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
async def _ejecutar(
    peticion: dict[str, Any], emisor: orx.Emisor | None
) -> AsyncIterator[tuple[str, Any]]:
    """Corre el bucle agéntico.

    Emite ("sse", str) cuando hay emisor, y termina con
    ("final", {"salida": [...], "usage": {...}, "status": "..."}).
    """
    s = get_settings()
    profile = get_profile()

    mensajes_previos: list[dict[str, Any]] = []
    prev_id = peticion.get("previous_response_id")
    if prev_id:
        estado = _leer_estado(prev_id)
        if not estado:
            raise LLMError("previous_response_not_found", 404)
        mensajes_previos = list(estado["mensajes"])

    nuevos, sistemas_del_input = orx.input_a_mensajes(peticion.get("input"))

    # Presupuesto de entrada: corta el historial más viejo, nunca el system.
    historial = mensajes_previos + nuevos
    while sum(len(json.dumps(m, default=str)) for m in historial) > s.max_input_chars and len(historial) > 1:
        historial.pop(0)

    system = construir_system_prompt(profile)
    if peticion.get("instructions"):
        system += (
            "\n\n## Instrucciones adicionales del operador\n"
            + str(peticion["instructions"])
            + "\n(Estas instrucciones no anulan las reglas de fundamentación ni de privacidad.)"
        )
    for extra in sistemas_del_input:
        system += "\n\n## Instrucción de sesión\n" + extra

    mensajes: list[dict[str, Any]] = [{"role": "system", "content": system}] + historial

    tools_cli, nombres_cli = _tools_cliente(peticion)
    tools = HERRAMIENTAS_INTERNAS + tools_cli

    temperatura = peticion.get("temperature")
    max_tokens = peticion.get("max_output_tokens")

    salida: list[dict[str, Any]] = []
    usage_total: dict[str, Any] = {}
    indice_salida = 0
    llamadas_herramienta = 0

    for _ in range(s.max_tool_iterations):
        texto = ""
        item_msg_id: str | None = None
        abierto = False
        pendientes: dict[int, dict[str, str]] = {}

        async for ev in stream_chat(
            mensajes, tools=tools, temperature=temperatura, max_tokens=max_tokens
        ):
            if ev["t"] == "text":
                delta = ev["delta"]
                texto += delta
                if emisor:
                    if not abierto:
                        item_msg_id = orx.nuevo_id("msg")
                        abierto = True
                        yield (
                            "sse",
                            emisor.evento(
                                "response.output_item.added",
                                output_index=indice_salida,
                                item={
                                    "id": item_msg_id,
                                    "type": "message",
                                    "status": "in_progress",
                                    "role": "assistant",
                                    "content": [],
                                },
                            ),
                        )
                        yield (
                            "sse",
                            emisor.evento(
                                "response.content_part.added",
                                item_id=item_msg_id,
                                output_index=indice_salida,
                                content_index=0,
                                part={"type": "output_text", "text": "", "annotations": []},
                            ),
                        )
                    yield (
                        "sse",
                        emisor.evento(
                            "response.output_text.delta",
                            item_id=item_msg_id,
                            output_index=indice_salida,
                            content_index=0,
                            delta=delta,
                            logprobs=[],
                        ),
                    )

            elif ev["t"] == "tool":
                i = ev["index"]
                acc = pendientes.setdefault(i, {"id": "", "name": "", "args": ""})
                if ev.get("id"):
                    acc["id"] = ev["id"]
                if ev.get("name"):
                    acc["name"] = ev["name"]
                acc["args"] += ev.get("args_delta") or ""

            elif ev["t"] == "done":
                if ev.get("usage"):
                    for k, v in ev["usage"].items():
                        if isinstance(v, (int, float)):
                            usage_total[k] = usage_total.get(k, 0) + v
                        else:
                            usage_total[k] = v

        # Cierra el item de mensaje si se abrió
        if texto:
            item_msg_id = item_msg_id or orx.nuevo_id("msg")
            item = orx.item_mensaje(texto, item_msg_id)
            salida.append(item)
            if emisor and abierto:
                yield ("sse", emisor.evento("response.output_text.done", item_id=item_msg_id, output_index=indice_salida, content_index=0, text=texto, logprobs=[]))
                yield ("sse", emisor.evento("response.content_part.done", item_id=item_msg_id, output_index=indice_salida, content_index=0, part={"type": "output_text", "text": texto, "annotations": []}))
                yield ("sse", emisor.evento("response.output_item.done", output_index=indice_salida, item=item))
            indice_salida += 1

        if not pendientes:
            break

        # ¿Alguna herramienta es del cliente? Entonces cedemos control.
        del_cliente = [p for p in pendientes.values() if p["name"] in nombres_cli]
        if del_cliente:
            for p in del_cliente:
                call_id = p["id"] or orx.nuevo_id("call")
                item = orx.item_function_call(call_id, p["name"], p["args"])
                salida.append(item)
                if emisor:
                    en_curso = {**item, "status": "in_progress", "arguments": ""}
                    yield ("sse", emisor.evento("response.output_item.added", output_index=indice_salida, item=en_curso))
                    yield ("sse", emisor.evento("response.function_call_arguments.delta", item_id=item["id"], output_index=indice_salida, delta=item["arguments"]))
                    yield ("sse", emisor.evento("response.function_call_arguments.done", item_id=item["id"], output_index=indice_salida, arguments=item["arguments"]))
                    yield ("sse", emisor.evento("response.output_item.done", output_index=indice_salida, item=item))
                indice_salida += 1
            break

        # Herramientas internas: se ejecutan aquí y el bucle continúa.
        mensajes.append(
            {
                "role": "assistant",
                "content": texto or None,
                "tool_calls": [
                    {
                        "id": p["id"] or f"call_{i}",
                        "type": "function",
                        "function": {"name": p["name"], "arguments": p["args"] or "{}"},
                    }
                    for i, p in sorted(pendientes.items())
                ],
            }
        )
        for i, p in sorted(pendientes.items()):
            try:
                args = json.loads(p["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            t0 = time.perf_counter()
            resultado = ejecutar_herramienta(p["name"], args)
            llamadas_herramienta += 1
            log_event(
                "tool_call",
                tool=p["name"],
                ms=round((time.perf_counter() - t0) * 1000, 1),
                ok="error" not in resultado,
            )
            mensajes.append(
                {
                    "role": "tool",
                    "tool_call_id": p["id"] or f"call_{i}",
                    "content": serializar_resultado(resultado),
                }
            )
    else:
        # Se agotaron las iteraciones sin respuesta final.
        if not salida:
            item = orx.item_mensaje(
                "No pude completar la consulta en los pasos disponibles. "
                "¿Puedes reformular la pregunta de forma más específica?"
            )
            salida.append(item)

    if not salida:
        salida.append(orx.item_mensaje("No obtuve respuesta del modelo. Intenta de nuevo."))

    # Historial para previous_response_id
    mensajes_finales = historial + [
        {"role": "assistant", "content": it["content"][0]["text"]}
        for it in salida
        if it.get("type") == "message"
    ]

    yield (
        "final",
        {
            "salida": salida,
            "usage": usage_total,
            "status": "completed",
            "mensajes": mensajes_finales,
            "tool_calls": llamadas_herramienta,
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/responses")
async def crear_respuesta(request: Request, authorization: str | None = Header(default=None)):
    if not _autorizado(authorization):
        return JSONResponse(
            status_code=401,
            content=orx.cuerpo_error("Falta o es inválida la credencial.", "invalid_request", "invalid_api_key"),
        )

    try:
        peticion = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content=orx.cuerpo_error("El cuerpo debe ser JSON válido.", "invalid_request", "invalid_json"))

    if not isinstance(peticion, dict):
        return JSONResponse(status_code=400, content=orx.cuerpo_error("El cuerpo debe ser un objeto JSON.", "invalid_request"))
    if peticion.get("input") in (None, "", []) and not peticion.get("previous_response_id"):
        return JSONResponse(status_code=400, content=orx.cuerpo_error("Falta el campo 'input'.", "invalid_request", "missing_required_parameter", "input"))

    # El modelo es un campo opcional en la plataforma. Aceptamos el que venga,
    # lo devolvemos tal cual, y servimos con el que este agente tiene configurado.
    modelo = peticion.get("model") or get_settings().model
    response_id = orx.nuevo_id("resp")
    creado = int(time.time())
    t0 = time.perf_counter()
    streaming = bool(peticion.get("stream"))

    log_event(
        "request",
        response_id=response_id,
        stream=streaming,
        modelo_solicitado=peticion.get("model"),
        tiene_previous=bool(peticion.get("previous_response_id")),
        tools_cliente=len(peticion.get("tools") or []),
    )

    # ---------------- no streaming ----------------
    if not streaming:
        try:
            final: dict[str, Any] = {}
            async for tipo, carga in _ejecutar(peticion, emisor=None):
                if tipo == "final":
                    final = carga
        except LLMError as exc:
            code = "previous_response_not_found" if str(exc) == "previous_response_not_found" else None
            return JSONResponse(
                status_code=exc.status,
                content=orx.cuerpo_error(str(exc), "not_found" if exc.status == 404 else "server_error", code),
            )
        except Exception as exc:  # pragma: no cover
            log_event("unhandled_error", response_id=response_id, error=repr(exc))
            return JSONResponse(status_code=500, content=orx.cuerpo_error("Error interno del agente.", "server_error"))

        respuesta = orx.construir_response(
            response_id=response_id,
            modelo=modelo,
            salida=final["salida"],
            peticion=peticion,
            usage=final["usage"],
            creado=creado,
        )
        _guardar_estado(response_id, final["mensajes"], respuesta)
        log_event("response", response_id=response_id, ms=round((time.perf_counter() - t0) * 1000, 1), tools=final.get("tool_calls", 0), chars=len(respuesta["output_text"]))
        return JSONResponse(content=respuesta)

    # ---------------- streaming ----------------
    async def generar() -> AsyncIterator[str]:
        emisor = orx.Emisor()
        base = orx.construir_response(
            response_id=response_id, modelo=modelo, salida=[], peticion=peticion, status="in_progress", creado=creado
        )
        yield emisor.evento("response.created", response=base)
        yield emisor.evento("response.in_progress", response=base)
        try:
            final: dict[str, Any] = {}
            async for tipo, carga in _ejecutar(peticion, emisor=emisor):
                if tipo == "sse":
                    yield carga
                else:
                    final = carga
            respuesta = orx.construir_response(
                response_id=response_id,
                modelo=modelo,
                salida=final["salida"],
                peticion=peticion,
                usage=final["usage"],
                creado=creado,
            )
            _guardar_estado(response_id, final["mensajes"], respuesta)
            yield emisor.evento("response.completed", response=respuesta)
            log_event("response", response_id=response_id, ms=round((time.perf_counter() - t0) * 1000, 1), stream=True, tools=final.get("tool_calls", 0))
        except LLMError as exc:
            code = "previous_response_not_found" if str(exc) == "previous_response_not_found" else None
            yield emisor.evento("error", message=str(exc), code=code, param=None)
            fallida = orx.construir_response(
                response_id=response_id, modelo=modelo, salida=[], peticion=peticion, status="failed",
                error={"code": code or "server_error", "message": str(exc)}, creado=creado,
            )
            yield emisor.evento("response.failed", response=fallida)
        except Exception as exc:  # pragma: no cover
            log_event("unhandled_error", response_id=response_id, error=repr(exc))
            yield emisor.evento("error", message="Error interno del agente.", code="server_error", param=None)
            fallida = orx.construir_response(
                response_id=response_id, modelo=modelo, salida=[], peticion=peticion, status="failed",
                error={"code": "server_error", "message": "Error interno del agente."}, creado=creado,
            )
            yield emisor.evento("response.failed", response=fallida)
        finally:
            yield orx.Emisor.fin()

    return StreamingResponse(
        generar(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/v1/responses/{response_id}")
async def obtener_respuesta(response_id: str, authorization: str | None = Header(default=None)):
    if not _autorizado(authorization):
        return JSONResponse(status_code=401, content=orx.cuerpo_error("Falta o es inválida la credencial.", "invalid_request", "invalid_api_key"))
    estado = _leer_estado(response_id)
    if not estado:
        return JSONResponse(status_code=404, content=orx.cuerpo_error("No existe esa respuesta.", "not_found", "response_not_found"))
    return JSONResponse(content=estado["respuesta"])


@app.get("/healthz")
async def healthz():
    p = get_profile()
    return {
        "status": "ok",
        "perfil": p.nombre,
        "experiencias": len(p.experiencia),
        "proyectos": len(p.proyectos),
        "proveedor": get_settings().provider,
    }


@app.get("/.well-known/agent-card.json")
async def agent_card():
    """Tarjeta A2A: permite que cualquier cliente descubra y registre este agente."""
    s = get_settings()
    p = get_profile()
    base = s.public_base_url.rstrip("/")
    return {
        "protocolVersion": "0.3.0",
        "name": f"CV de {p.nombre}",
        "description": f"Agente conversacional sobre la trayectoria profesional de {p.nombre}: experiencia, habilidades y proyectos.",
        "version": "1.0.0",
        "url": base,
        "openResponsesUrl": base,
        "preferredTransport": "open-responses",
        "supportedInterfaces": [
            {"url": base, "protocolBinding": "open-responses", "protocolVersion": "2026-04-24"}
        ],
        "provider": {"organization": p.nombre, "url": (p.persona.get("contacto") or {}).get("linkedin") or base},
        "contact": {k: v for k, v in (p.persona.get("contacto") or {}).items() if v and k in ("email", "linkedin")},
        "repositoryUrl": (p.persona.get("contacto") or {}).get("github") or "",
        "capabilities": {"streaming": True, "pushNotifications": False, "stateTransitionHistory": True},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}} if s.agent_api_key else {},
        "skills": [
            {"id": "trayectoria", "name": "Trayectoria profesional", "description": "Responde sobre experiencia, roles y responsabilidades.", "tags": ["cv", "experiencia"], "examples": ["¿En qué has trabajado los últimos dos años?"]},
            {"id": "proyectos", "name": "Proyectos", "description": "Explica proyectos concretos, decisiones técnicas y resultados.", "tags": ["proyectos"], "examples": ["Cuéntame del bot de WhatsApp"]},
            {"id": "encaje", "name": "Encaje con una vacante", "description": "Compara el perfil contra requisitos de un puesto y reporta huecos.", "tags": ["fit"], "examples": ["¿Encajas en esta vacante? [pegar descripción]"]},
        ],
        "promptSuggestions": [
            "¿Cuál es tu experiencia con agentes de IA en producción?",
            "Cuéntame del proyecto técnico más difícil que has resuelto",
            "¿Qué has hecho con datos o sistemas financieros?",
            "Te paso una vacante, ¿qué tanto encajas?",
        ],
    }


# ---------------------------------------------------------------------------
# Demo pública del sitio personal
# ---------------------------------------------------------------------------
@app.get("/")
async def inicio():
    """Interfaz de chat, pensada para incrustarse en un sitio personal."""
    return FileResponse(ROOT / "app" / "static" / "index.html", media_type="text/html")


@app.post("/api/chat")
async def chat_demo(request: Request):
    """Mismo agente que /v1/responses, sin credencial y con límite por IP.

    Existe porque una página web no puede guardar una API key en secreto. La
    protección aquí no es de identidad sino de consumo.
    """
    if os.getenv("PUBLIC_DEMO", "true").lower() not in ("1", "true", "yes"):
        return JSONResponse(status_code=404, content=orx.cuerpo_error("La demo pública está apagada.", "not_found"))

    ip = _ip(request)
    if not _dentro_del_limite(ip):
        log_event("demo_rate_limited", ip=ip)
        return JSONResponse(
            status_code=429,
            content=orx.cuerpo_error(
                f"Límite de {_LIMITE} mensajes por hora alcanzado.", "rate_limit", "rate_limit_exceeded"
            ),
        )

    try:
        peticion = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content=orx.cuerpo_error("Cuerpo inválido.", "invalid_request"))

    peticion = {
        "input": peticion.get("input"),
        "stream": True,
        "temperature": 0.4,
        "max_output_tokens": 700,
    }
    if not peticion["input"]:
        return JSONResponse(status_code=400, content=orx.cuerpo_error("Falta 'input'.", "invalid_request"))

    response_id = orx.nuevo_id("resp")
    creado = int(time.time())
    log_event("demo_request", response_id=response_id, ip=ip)

    async def generar() -> AsyncIterator[str]:
        emisor = orx.Emisor()
        base = orx.construir_response(
            response_id=response_id, modelo=get_settings().model, salida=[],
            peticion=peticion, status="in_progress", creado=creado,
        )
        yield emisor.evento("response.created", response=base)
        try:
            final: dict[str, Any] = {}
            async for tipo, carga in _ejecutar(peticion, emisor=emisor):
                if tipo == "sse":
                    yield carga
                else:
                    final = carga
            yield emisor.evento(
                "response.completed",
                response=orx.construir_response(
                    response_id=response_id, modelo=get_settings().model,
                    salida=final["salida"], peticion=peticion, usage=final["usage"], creado=creado,
                ),
            )
        except Exception as exc:
            log_event("demo_error", response_id=response_id, error=repr(exc))
            yield emisor.evento("error", message="El agente no pudo responder.", code="server_error", param=None)
        finally:
            yield orx.Emisor.fin()

    return StreamingResponse(
        generar(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
