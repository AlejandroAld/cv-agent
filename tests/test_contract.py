"""Tests de contrato Open Responses.

Corren con LLM_PROVIDER=mock, así que validan el protocolo sin gastar tokens
ni necesitar credenciales. Es lo que corre en cada push.
"""

from __future__ import annotations

import json
import os
import re

import pytest

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("AGENT_API_KEY", "test-key")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

H = {"Authorization": "Bearer test-key"}

CAMPOS_REQUERIDOS = [
    "id", "object", "created_at", "completed_at", "status", "error",
    "incomplete_details", "model", "previous_response_id", "instructions",
    "output", "tools", "tool_choice", "truncation", "parallel_tool_calls",
    "text", "top_p", "presence_penalty", "frequency_penalty", "top_logprobs",
    "temperature", "reasoning", "usage", "max_output_tokens", "max_tool_calls",
    "store", "background", "service_tier", "metadata", "safety_identifier",
    "prompt_cache_key",
]


@pytest.fixture(scope="module")
def cli():
    return TestClient(app)


def _sse(cli, body):
    with cli.stream("POST", "/v1/responses", headers=H, json=body) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = b"".join(r.iter_bytes()).decode()
    eventos = []
    for bloque in [b for b in raw.split("\n\n") if b.strip()]:
        data = re.search(r"^data: (.+)$", bloque, re.M)
        if not data:
            continue
        if data.group(1).strip() == "[DONE]":
            eventos.append(("[DONE]", None))
            continue
        ev = re.search(r"^event: (.+)$", bloque, re.M)
        eventos.append((ev.group(1) if ev else None, json.loads(data.group(1))))
    return eventos


# --- auth ------------------------------------------------------------------
def test_sin_credencial_da_401(cli):
    r = cli.post("/v1/responses", json={"input": "hola"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


def test_credencial_incorrecta_da_401(cli):
    r = cli.post("/v1/responses", headers={"Authorization": "Bearer malo"}, json={"input": "hola"})
    assert r.status_code == 401


# --- objeto Response -------------------------------------------------------
def test_response_trae_todos_los_campos_requeridos(cli):
    d = cli.post("/v1/responses", headers=H, json={"model": "x", "input": "hola"}).json()
    faltantes = [c for c in CAMPOS_REQUERIDOS if c not in d]
    assert not faltantes, f"faltan campos: {faltantes}"
    assert d["object"] == "response"
    assert d["status"] == "completed"


def test_item_message_bien_formado(cli):
    d = cli.post("/v1/responses", headers=H, json={"input": "hola"}).json()
    item = d["output"][0]
    assert item["type"] == "message"
    assert item["role"] == "assistant"
    assert item["status"] == "completed"
    assert item["id"]
    parte = item["content"][0]
    assert parte["type"] == "output_text"
    assert isinstance(parte["annotations"], list)


def test_usage_tiene_desglose(cli):
    u = cli.post("/v1/responses", headers=H, json={"input": "hola"}).json()["usage"]
    assert "input_tokens_details" in u and "cached_tokens" in u["input_tokens_details"]
    assert "output_tokens_details" in u and "reasoning_tokens" in u["output_tokens_details"]


# --- entrada ---------------------------------------------------------------
def test_input_como_string(cli):
    assert cli.post("/v1/responses", headers=H, json={"input": "texto plano"}).status_code == 200


def test_input_como_items_con_historial(cli):
    entrada = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hola"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Qué tal"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Sigue"}]},
    ]
    assert cli.post("/v1/responses", headers=H, json={"input": entrada}).status_code == 200


def test_mensaje_system_en_el_input_se_acepta(cli):
    entrada = [
        {"type": "message", "role": "system", "content": "Sé breve"},
        {"type": "message", "role": "user", "content": "Hola"},
    ]
    assert cli.post("/v1/responses", headers=H, json={"input": entrada}).status_code == 200


def test_input_faltante_da_400(cli):
    r = cli.post("/v1/responses", headers=H, json={})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "input"


def test_json_invalido_da_400(cli):
    r = cli.post(
        "/v1/responses",
        headers={**H, "Content-Type": "application/json"},
        content=b"{no json",
    )
    assert r.status_code == 400


# --- streaming -------------------------------------------------------------
def test_secuencia_de_eventos_sse(cli):
    eventos = _sse(cli, {"input": "hola", "stream": True})
    tipos = [t for t, _ in eventos]

    assert tipos[0] == "response.created"
    assert tipos[1] == "response.in_progress"
    assert tipos[-1] == "[DONE]"
    assert "response.completed" in tipos

    # el ciclo de vida del item respeta el orden del spec
    for esperado in (
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
    ):
        assert esperado in tipos, f"falta evento {esperado}"
    assert tipos.index("response.output_item.added") < tipos.index("response.content_part.added")
    assert tipos.index("response.content_part.done") < tipos.index("response.output_item.done")


def test_event_coincide_con_type_y_seq_es_monotonico(cli):
    eventos = _sse(cli, {"input": "hola", "stream": True})
    seqs = []
    for tipo, payload in eventos:
        if payload is None:
            continue
        assert tipo == payload["type"], "el campo event: debe coincidir con type"
        seqs.append(payload["sequence_number"])
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_deltas_reconstruyen_el_texto_final(cli):
    eventos = _sse(cli, {"input": "hola", "stream": True})
    deltas = "".join(p["delta"] for t, p in eventos if t == "response.output_text.delta")
    final = next(p for t, p in eventos if t == "response.completed")["response"]
    assert deltas == final["output_text"]


# --- estado ----------------------------------------------------------------
def test_previous_response_id_encadena(cli):
    r1 = cli.post("/v1/responses", headers=H, json={"input": "uno", "store": True}).json()
    r2 = cli.post("/v1/responses", headers=H, json={"input": "dos", "previous_response_id": r1["id"]}).json()
    assert r2["previous_response_id"] == r1["id"]
    assert r2["status"] == "completed"


def test_previous_response_id_inexistente(cli):
    r = cli.post("/v1/responses", headers=H, json={"input": "x", "previous_response_id": "resp_nope"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "previous_response_not_found"


def test_recuperar_respuesta(cli):
    rid = cli.post("/v1/responses", headers=H, json={"input": "hola"}).json()["id"]
    assert cli.get(f"/v1/responses/{rid}", headers=H).json()["id"] == rid
    assert cli.get("/v1/responses/resp_nope", headers=H).status_code == 404


# --- parámetros de la plataforma ------------------------------------------
def test_eco_de_parametros(cli):
    d = cli.post(
        "/v1/responses",
        headers=H,
        json={
            "input": "hola",
            "instructions": "Sé breve",
            "temperature": 0.7,
            "reasoning": {"effort": "medium"},
            "metadata": {"origen": "prueba"},
            "truncation": "auto",
        },
    ).json()
    assert d["instructions"] == "Sé breve"
    assert d["temperature"] == 0.7
    assert d["reasoning"]["effort"] == "medium"
    # La metadata del cliente se conserva; el servidor añade sus claves agent_*.
    assert d["metadata"]["origen"] == "prueba"
    assert d["truncation"] == "auto"


def test_modelo_opcional(cli):
    assert cli.post("/v1/responses", headers=H, json={"input": "hola"}).json()["model"]


# --- herramientas del cliente ---------------------------------------------
def test_function_tool_del_cliente_cede_control(cli, monkeypatch):
    async def fake(entrada, *, instructions=None, tools=None, max_output_tokens=None, reasoning_effort=None):
        yield {"t": "tool", "call_id": "call_1", "name": "get_weather", "arguments": '{"city":"GDL"}'}
        yield {
            "t": "done",
            "usage": None,
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                 "name": "get_weather", "arguments": '{"city":"GDL"}'},
            ],
        }

    import app.main as main

    monkeypatch.setattr(main, "stream_agente", fake)
    body = {
        "input": "clima",
        "tools": [{"type": "function", "name": "get_weather", "description": "clima", "parameters": {"type": "object", "properties": {}}}],
    }
    d = cli.post("/v1/responses", headers=H, json=body).json()
    item = d["output"][0]
    assert item["type"] == "function_call"
    assert item["call_id"] == "call_1"
    assert item["name"] == "get_weather"
    assert json.loads(item["arguments"]) == {"city": "GDL"}
    assert all(it["type"] != "reasoning" for it in d["output"])


# --- descubrimiento --------------------------------------------------------
def test_healthz(cli):
    assert cli.get("/healthz").json()["status"] == "ok"


def test_agent_card(cli):
    card = cli.get("/.well-known/agent-card.json").json()
    for campo in ("name", "description", "version", "url", "capabilities", "skills"):
        assert campo in card
    assert card["capabilities"]["streaming"] is True
    assert len(card["skills"]) >= 1


def test_agent_card_declara_la_interfaz_en_una_sola_version_de_a2a(cli):
    """La tarjeta decía protocolVersion 0.3.0 y usaba supportedInterfaces, que
    es de v1.0. Ningún parser veía una interfaz completa."""
    card = cli.get("/.well-known/agent-card.json").json()

    assert card["protocolVersion"] == "1.0"
    iface = card["supportedInterfaces"][0]
    assert set(iface) >= {"url", "protocolBinding", "protocolVersion"}
    # El protocolVersion de la interfaz es la versión de A2A, no la fecha del
    # spec de Open Responses.
    assert iface["protocolVersion"] == "1.0"
    assert iface["url"] == card["url"], "la interfaz preferida y url deben coincidir"


def test_agent_card_usa_una_uri_como_binding_propio(cli):
    """A2A registra JSONRPC, GRPC y HTTP+JSON. Un binding propio va como URI,
    para no chocar con valores futuros del núcleo."""
    card = cli.get("/.well-known/agent-card.json").json()

    binding = card["supportedInterfaces"][0]["protocolBinding"]
    assert binding.startswith("https://"), f"binding no es URI: {binding!r}"
    assert card["preferredTransport"] == binding, "v0.3 y v1.0 deben coincidir"
    assert card["additionalInterfaces"][0]["transport"] == binding


# --- herramientas internas -------------------------------------------------
def test_herramientas_internas_no_alucinan():
    """Kubernetes ya NO sirve como ejemplo de hueco: el perfil lo cubre.

    n8n corre autohospedado sobre Kubernetes y está en habilidades. Se usan
    Terraform y Rust, que sí están ausentes del perfil; si algún día dejan de
    estarlo, este test debe cambiar, no el código.
    """
    from app.agent_brain import ejecutar_herramienta

    vacio = ejecutar_herramienta("buscar_en_perfil", {"consulta": "terraform rust"})
    assert vacio["encontrados"] == 0 and vacio["nota"]

    hit = ejecutar_herramienta("buscar_en_perfil", {"consulta": "n8n WhatsApp"})
    assert hit["encontrados"] > 0

    encaje = ejecutar_herramienta("evaluar_encaje", {"requisitos": ["Terraform", "Firebase"]})
    por_req = {e["requisito"]: e["cobertura"] for e in encaje["evaluacion"]}
    assert por_req["Terraform"] == "sin_evidencia", "debe reportar honestamente lo que no cubre"
    assert por_req["Firebase"] == "directa"

    assert ejecutar_herramienta("obtener_detalle", {"id": "nope"})["error"] == "not_found"


def test_contacto_no_expone_datos_privados():
    from app.agent_brain import ejecutar_herramienta

    contacto = ejecutar_herramienta("obtener_contacto", {})["contacto"]
    assert "telefono" not in contacto and "direccion" not in contacto


def test_system_prompt_incluye_el_perfil():
    from app.agent_brain import construir_system_prompt

    p = construir_system_prompt()
    assert "PERFIL" in p and "fundamentación" in p.lower()


# --- Responses API: lo que viaja al proveedor ------------------------------
class _RespuestaFalsa:
    """Respuesta de httpx con un stream SSE grabado."""

    def __init__(self, lineas: list[str], status: int = 200) -> None:
        self._lineas = lineas
        self.status_code = status

    async def aread(self) -> bytes:
        return b""

    async def aiter_lines(self):
        for linea in self._lineas:
            yield linea


class _CtxStream:
    def __init__(self, resp): self._resp = resp
    async def __aenter__(self): return self._resp
    async def __aexit__(self, *a): return False


def _sse_proveedor(texto: str) -> list[str]:
    """La secuencia exacta que emite la Responses API, comprobada en vivo."""
    item = {"id": "msg_1", "type": "message", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": texto, "annotations": []}]}
    eventos: list[dict] = [
        {"type": "response.created", "response": {"id": "resp_x", "status": "in_progress"}},
        {"type": "response.in_progress", "response": {"id": "resp_x", "status": "in_progress"}},
        {"type": "response.output_item.added", "item": {"id": "rs_1", "type": "reasoning", "summary": []}},
        {"type": "response.output_item.done", "item": {"id": "rs_1", "type": "reasoning", "summary": []}},
        {"type": "response.output_item.added", "item": {**item, "status": "in_progress", "content": []}},
        {"type": "response.content_part.added", "item_id": "msg_1"},
    ]
    pedazos = texto.split(" ")
    for i, pedazo in enumerate(pedazos):
        delta = pedazo if i == len(pedazos) - 1 else pedazo + " "
        eventos.append({"type": "response.output_text.delta", "item_id": "msg_1", "delta": delta})
    eventos += [
        {"type": "response.output_text.done", "item_id": "msg_1", "text": texto},
        {"type": "response.content_part.done", "item_id": "msg_1"},
        {"type": "response.output_item.done", "item": item},
        {"type": "response.completed", "response": {
            "id": "resp_x", "status": "completed",
            "output": [{"id": "rs_1", "type": "reasoning", "summary": []}, item],
            "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18,
                      "input_tokens_details": {"cached_tokens": 3},
                      "output_tokens_details": {"reasoning_tokens": 0}},
        }},
    ]
    return [f"data: {json.dumps(e)}" for e in eventos] + ["data: [DONE]"]


@pytest.fixture
def azure_falso(monkeypatch):
    """Apunta el proveedor a un Azure falso y captura lo que se le manda."""
    import app.llm as llm
    from app.core import Settings

    capturas: list[dict] = []

    ajustes = Settings(
        provider="azure",
        azure_endpoint="https://cv-agent-foundry.openai.azure.com/openai/v1",
        azure_api_key="clave-secreta",
        azure_deployment="gpt-5-mini",
        azure_api_version="",
        reasoning_effort="minimal",
    )

    class ClienteFalso:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

        def stream(self, metodo, url, headers=None, json=None):
            capturas.append({"metodo": metodo, "url": url, "headers": headers, "cuerpo": json})
            return _CtxStream(_RespuestaFalsa(_sse_proveedor("hola desde Azure")))

    monkeypatch.setattr(llm, "get_settings", lambda: ajustes)
    monkeypatch.setattr(llm.httpx, "AsyncClient", ClienteFalso)
    return capturas


def test_el_cuerpo_al_proveedor_no_lleva_temperature_ni_max_tokens(cli, azure_falso):
    """Un modelo de razonamiento responde 400 ante cualquiera de esos campos.

    El cliente puede mandarlos —se hacen eco en el Response— pero no viajan.
    """
    d = cli.post(
        "/v1/responses",
        headers=H,
        json={
            "input": "hola",
            "temperature": 0.7,
            "top_p": 0.4,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5,
            "max_output_tokens": 700,
        },
    ).json()

    assert len(azure_falso) == 1, "debió haber exactamente una llamada al proveedor"
    cuerpo = azure_falso[0]["cuerpo"]

    for prohibido in ("temperature", "max_tokens", "top_p", "presence_penalty", "frequency_penalty"):
        assert prohibido not in cuerpo, f"{prohibido} no debe viajar al proveedor"

    # lo que sí debe ir
    assert cuerpo["max_output_tokens"] == 700
    assert cuerpo["model"] == "gpt-5-mini"
    assert cuerpo["stream"] is True
    assert cuerpo["store"] is False
    assert cuerpo["reasoning"] == {"effort": "minimal"}
    assert cuerpo["instructions"].count("PERFIL") >= 1
    assert all(m.get("role") != "system" for m in cuerpo["input"]), "el prompt va en instructions"

    # y el eco al cliente no cambia
    assert d["temperature"] == 0.7
    assert d["output_text"] == "hola desde Azure"


def test_ruta_y_auth_de_azure(cli, azure_falso):
    cli.post("/v1/responses", headers=H, json={"input": "hola"})
    llamada = azure_falso[0]

    assert llamada["url"] == "https://cv-agent-foundry.openai.azure.com/openai/v1/responses"
    assert "api-version" not in llamada["url"], "la v1 GA no la pide"
    assert llamada["headers"]["api-key"] == "clave-secreta"
    assert "Authorization" not in llamada["headers"], "Azure no usa Bearer"


def test_herramientas_al_proveedor_en_formato_plano(cli, azure_falso):
    cli.post("/v1/responses", headers=H, json={"input": "hola"})
    tools = azure_falso[0]["cuerpo"]["tools"]

    assert tools, "las herramientas internas deben viajar"
    for t in tools:
        assert t["type"] == "function"
        assert "name" in t and "parameters" in t
        assert "function" not in t, "la Responses API usa el formato plano"


def test_usage_del_proveedor_llega_completo(cli, azure_falso):
    u = cli.post("/v1/responses", headers=H, json={"input": "hola"}).json()["usage"]
    assert u["input_tokens"] == 11
    assert u["output_tokens"] == 7
    assert u["input_tokens_details"]["cached_tokens"] == 3
    assert u["output_tokens_details"]["reasoning_tokens"] == 0


# --- el razonamiento no sale hacia el cliente ------------------------------
def test_los_items_reasoning_no_llegan_al_cliente(cli):
    """El mock devuelve un item de razonamiento en cada vuelta, como el real."""
    # "proyecto" dispara la rama con herramienta: dos vueltas, dos razonamientos
    for entrada in ("hola", "cuéntame de un proyecto"):
        d = cli.post("/v1/responses", headers=H, json={"input": entrada}).json()
        tipos = [it["type"] for it in d["output"]]
        assert "reasoning" not in tipos, f"se filtró razonamiento con {entrada!r}"
        assert tipos, "debe quedar al menos un item visible"


def test_ningun_evento_sse_expone_razonamiento(cli):
    eventos = _sse(cli, {"input": "cuéntame de un proyecto", "stream": True})
    for tipo, payload in eventos:
        if payload is None:
            continue
        item = payload.get("item") or {}
        assert item.get("type") != "reasoning", f"{tipo} expuso un item de razonamiento"
        salida = (payload.get("response") or {}).get("output") or []
        assert all(it.get("type") != "reasoning" for it in salida)


def test_sin_reasoning_es_idempotente_y_conserva_el_resto():
    from app.openresponses import sin_reasoning

    items = [
        {"type": "reasoning", "id": "rs_1"},
        {"type": "message", "id": "msg_1"},
        {"type": "function_call", "call_id": "c1"},
    ]
    limpio = sin_reasoning(items)
    assert [i["type"] for i in limpio] == ["message", "function_call"]
    assert sin_reasoning(limpio) == limpio


# --- higiene del historial -------------------------------------------------
def test_recortar_no_deja_function_call_output_huerfano():
    from app.openresponses import recortar

    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x" * 500}]},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "resultado"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "corta"}]},
    ]
    # presupuesto que obliga a tirar el mensaje largo y la llamada
    recortado = recortar(items, 120)

    llamadas = {i["call_id"] for i in recortado if i["type"] == "function_call"}
    for i in recortado:
        if i["type"] == "function_call_output":
            assert i["call_id"] in llamadas, "output sin su function_call: el proveedor da 400"
    assert recortado, "el recorte no puede vaciar el historial"


def test_recortar_respeta_el_par_cuando_cabe():
    from app.openresponses import recortar

    items = [
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
    ]
    assert recortar(items, 10_000) == items


def test_input_a_items_normaliza_el_rol_de_las_partes():
    from app.openresponses import input_a_items

    items, sistemas = input_a_items(
        [
            {"type": "message", "role": "system", "content": "Sé breve"},
            {"type": "message", "role": "user", "content": [{"type": "output_text", "text": "hola"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": "qué tal"}]},
        ]
    )
    assert sistemas == ["Sé breve"], "system/developer se extraen, no son items"
    assert items[0]["content"][0]["type"] == "input_text", "el usuario manda input_text"
    assert items[1]["content"][0]["type"] == "output_text", "el assistant manda output_text"


# --- la vuelta completa con herramienta interna ----------------------------
def _sse_tool(nombre: str, argumentos: str, call_id: str = "call_1") -> list[str]:
    """Lo que emite el proveedor al llamar una herramienta: razonamiento + llamada."""
    razon = {"id": "rs_1", "type": "reasoning", "summary": []}
    llamada = {"id": "fc_1", "type": "function_call", "call_id": call_id,
               "name": nombre, "arguments": argumentos, "status": "completed"}
    eventos = [
        {"type": "response.created", "response": {"id": "resp_x", "status": "in_progress"}},
        {"type": "response.output_item.done", "item": razon},
        {"type": "response.output_item.done", "item": llamada},
        {"type": "response.completed", "response": {
            "id": "resp_x", "status": "completed", "output": [razon, llamada],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}}},
    ]
    return [f"data: {json.dumps(e)}" for e in eventos] + ["data: [DONE]"]


@pytest.fixture
def azure_guion(monkeypatch):
    """Fábrica: recibe guiones SSE en orden y devuelve lo capturado."""
    import app.llm as llm
    from app.core import Settings

    capturas: list[dict] = []
    ajustes = Settings(
        provider="azure",
        azure_endpoint="https://cv-agent-foundry.openai.azure.com/openai/v1",
        azure_api_key="clave-secreta",
        azure_deployment="gpt-5-mini",
        azure_api_version="",
        reasoning_effort="minimal",
    )

    def montar(guiones):
        pendientes = list(guiones)

        class ClienteFalso:
            def __init__(self, *a, **k): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False

            def stream(self, metodo, url, headers=None, json=None):
                capturas.append({"url": url, "headers": headers, "cuerpo": json})
                return _CtxStream(_RespuestaFalsa(
                    pendientes.pop(0) if pendientes else _sse_proveedor("fin")
                ))

        monkeypatch.setattr(llm, "get_settings", lambda: ajustes)
        monkeypatch.setattr(llm.httpx, "AsyncClient", ClienteFalso)
        return capturas

    return montar


def test_tool_interna_reinyecta_los_items_del_modelo(cli, azure_guion):
    """El contrato de la Responses API: los items del modelo vuelven tal cual.

    El item de razonamiento se encadena por id, así que perderlo rompe la
    continuidad del turno. Y el resultado va como function_call_output.
    """
    capturas = azure_guion([
        _sse_tool("buscar_en_perfil", '{"consulta": "n8n"}'),
        _sse_proveedor("uso n8n en el bot"),
    ])

    d = cli.post("/v1/responses", headers=H, json={"input": "cuéntame de n8n"}).json()

    assert len(capturas) == 2, "la herramienta interna debe disparar una segunda llamada"

    segunda = capturas[1]["cuerpo"]["input"]
    assert [i["type"] for i in segunda] == [
        "message", "reasoning", "function_call", "function_call_output",
    ]
    assert segunda[1]["id"] == "rs_1", "el razonamiento vuelve con su id, no reconstruido"
    assert segunda[3]["call_id"] == segunda[2]["call_id"], "output ligado a su llamada"
    assert json.loads(segunda[3]["output"])["encontrados"] >= 0, "el resultado viaja serializado"

    # el segundo cuerpo tampoco lleva los campos prohibidos
    for prohibido in ("temperature", "max_tokens", "top_p"):
        assert prohibido not in capturas[1]["cuerpo"]

    # y nada del razonamiento sale al cliente
    assert [i["type"] for i in d["output"]] == ["message"]
    assert d["output_text"] == "uso n8n en el bot"


def test_reasoning_se_puede_apagar_para_modelos_sin_razonamiento():
    """Un modelo sin razonamiento da 400 si le llega el bloque `reasoning`."""
    from app.llm import construir_cuerpo

    con = construir_cuerpo([], None, None, None, "minimal", "gpt-5-mini")
    assert con["reasoning"] == {"effort": "minimal"}

    sin = construir_cuerpo([], None, None, None, "", "gpt-4o-mini")
    assert "reasoning" not in sin, 'REASONING_EFFORT="" debe omitir el bloque'


# --- autenticación ---------------------------------------------------------
def test_token_no_ascii_no_revienta(monkeypatch):
    """compare_digest sobre str exige ASCII y lanza TypeError; por eso bytes.

    No va por HTTP a propósito: httpx se niega a enviar un header no ASCII,
    pero Starlette decodifica los headers entrantes como latin-1, así que un
    byte alto SÍ llega hasta aquí como texto. Se prueba la función directa.
    """
    import app.main as main
    from app.core import Settings

    monkeypatch.setattr(main, "get_settings", lambda: Settings(agent_api_key="test-key"))

    assert main._autorizado("Bearer clavé-con-acentó") is False
    assert main._autorizado("Bearer test-key") is True


def test_token_de_otra_longitud_da_401(cli):
    for token in ("", "x", "test-key-mas-largo", "test-ke"):
        r = cli.post("/v1/responses", headers={"Authorization": f"Bearer {token}"}, json={"input": "hola"})
        assert r.status_code == 401, f"token {token!r} no debió pasar"


def test_token_correcto_con_y_sin_prefijo_bearer(cli):
    assert cli.post("/v1/responses", headers={"Authorization": "Bearer test-key"}, json={"input": "hola"}).status_code == 200
    assert cli.post("/v1/responses", headers={"Authorization": "test-key"}, json={"input": "hola"}).status_code == 200


# --- store -----------------------------------------------------------------
def test_store_por_defecto_retiene(cli):
    """Default true, como la plataforma: sin el flag, la respuesta se guarda."""
    d = cli.post("/v1/responses", headers=H, json={"input": "hola"}).json()
    assert d["store"] is True, "el eco debe decir la verdad"
    assert cli.get(f"/v1/responses/{d['id']}", headers=H).status_code == 200


def test_store_false_no_retiene_nada(cli):
    """Quien pide store:false obtiene lo que pidió: nada queda."""
    d = cli.post("/v1/responses", headers=H, json={"input": "hola", "store": False}).json()
    assert d["store"] is False

    assert cli.get(f"/v1/responses/{d['id']}", headers=H).status_code == 404

    encadenada = cli.post("/v1/responses", headers=H, json={"input": "sigue", "previous_response_id": d["id"]})
    assert encadenada.status_code == 404
    assert encadenada.json()["error"]["code"] == "previous_response_not_found"


def test_store_false_tambien_se_respeta_en_streaming(cli):
    eventos = _sse(cli, {"input": "hola", "stream": True, "store": False})
    final = next(p for t, p in eventos if t == "response.completed")["response"]
    assert final["store"] is False
    assert cli.get(f"/v1/responses/{final['id']}", headers=H).status_code == 404


# --- el perfil se renderiza completo en el system prompt --------------------
# Los cuatro asserts de abajo cubren dos bugs reales: la sección
# "publicaciones" no tenía rama en como_contexto() y nunca llegaba al modelo,
# y "certificaciones" cambió de strings a dicts, así que el prompt llevaba el
# repr de un diccionario.
def test_system_prompt_incluye_el_titulo_de_la_publicacion():
    from app.agent_brain import construir_system_prompt

    assert "Detection of Tendency to Depression through Text Analysis" in construir_system_prompt()


def test_system_prompt_incluye_la_url_completa_de_la_publicacion():
    """La URL es lo que hace la cita verificable: va entera y literal."""
    from app.agent_brain import construir_system_prompt

    assert "https://www.cys.cic.ipn.mx/ojs/index.php/CyS/article/view/5887" in construir_system_prompt()


def test_system_prompt_no_lleva_el_repr_de_un_dict():
    from app.agent_brain import construir_system_prompt

    assert "{'nombre'" not in construir_system_prompt()


def test_system_prompt_rinde_certificacion_con_su_estado():
    from app.agent_brain import construir_system_prompt

    sp = construir_system_prompt()
    assert "Microsoft AI-103" in sp
    assert "En curso" in sp


def test_certificaciones_acepta_strings_sueltos():
    """El formato ya cambió una vez; que un string suelto no rompa el render."""
    from app.core import Profile

    p = Profile(raw={"certificaciones": [
        {"nombre": "Con dict", "estado": "En curso"},
        "Suelta sin estado",
        {"nombre": "Sin estado"},
        {"estado": "huérfano, sin nombre"},
    ]})
    ctx = p.como_contexto()
    assert "- Con dict (En curso)" in ctx
    assert "- Suelta sin estado" in ctx
    assert "- Sin estado" in ctx
    assert "huérfano" not in ctx, "sin nombre no hay nada que afirmar"
    assert "{" not in ctx.split("# Certificaciones")[1]


# --- las publicaciones son recuperables y citables -------------------------
def test_buscar_encuentra_la_publicacion():
    from app.core import get_profile

    for consulta in ("publicaciones", "publicaciones arbitradas", "revista arbitrada", "depression"):
        hits = get_profile().buscar(consulta, 6)
        assert any(h.get("id") == "pub-1" for h in hits), f"{consulta!r} no la encontró"


def test_obtener_detalle_resuelve_una_publicacion():
    from app.agent_brain import ejecutar_herramienta

    d = ejecutar_herramienta("obtener_detalle", {"id": "pub-1"})
    assert d["tipo"] == "publicacion"
    assert d["url"].startswith("https://")
    assert "error" not in d


def test_evaluar_encaje_acredita_la_publicacion_con_su_url():
    """El falso negativo que motivó meterlas en buscar(): una vacante que pide
    publicaciones no puede decir `sin_evidencia` contra una real."""
    from app.agent_brain import ejecutar_herramienta

    encaje = ejecutar_herramienta("evaluar_encaje", {"requisitos": ["Publicaciones arbitradas"]})
    req = encaje["evaluacion"][0]
    assert req["cobertura"] == "directa"
    urls = [e.get("url") for e in req["evidencia"] or []]
    assert any(u and "cys.cic.ipn.mx" in u for u in urls), "la evidencia debe traer la URL"


def test_ids_de_ejemplo_de_obtener_detalle_existen_de_verdad():
    """La descripción citaba ids muertos. Que no vuelva a pasar en silencio."""
    import re

    from app.agent_brain import HERRAMIENTAS_INTERNAS
    from app.core import get_profile

    desc = next(t for t in HERRAMIENTAS_INTERNAS if t["name"] == "obtener_detalle")["description"]
    citados = re.findall(r"'([a-z]+-[a-z0-9-]+)'", desc)
    assert citados, "la descripción debe seguir dando ejemplos"

    p = get_profile()
    reales = {r.get("id") for r in (p.experiencia + p.proyectos + p.publicaciones)}
    for ident in citados:
        assert ident in reales, f"{ident!r} ya no existe en el perfil"


def test_publicaciones_derivan_id_estable():
    from app.core import Profile

    p = Profile(raw={"publicaciones": [{"titulo": "A"}, {"titulo": "B", "id": "propio"}]})
    assert [x["id"] for x in p.publicaciones] == ["pub-1", "propio"], "el id del YAML gana"


# --- fuerza de la evidencia ------------------------------------------------
# El booleano viejo trataba igual un match en el stack y uno dentro de una
# frase en prosa. Por eso "core bancario" salía cubierto apoyado en
# "convención bancaria base 360", que es una convención de conteo de días.
def _cobertura(requisito: str) -> str:
    from app.agent_brain import ejecutar_herramienta

    encaje = ejecutar_herramienta("evaluar_encaje", {"requisitos": [requisito]})
    return encaje["evaluacion"][0]["cobertura"]


def test_core_bancario_es_adyacente_no_directo():
    """El caso que destapó el problema. Si esto vuelve a 'directa', el agente
    está diciéndole a un reclutador bancario que cubre core bancario."""
    assert _cobertura("core bancario") == "adyacente"


def test_lo_que_esta_en_el_stack_es_evidencia_directa():
    for requisito in ("Kubernetes", "LangGraph", "Next.js"):
        assert _cobertura(requisito) == "directa", f"{requisito} está en el stack"


def test_lo_ausente_no_tiene_evidencia():
    for requisito in ("Terraform", "Rust"):
        assert _cobertura(requisito) == "sin_evidencia"


def test_publicaciones_arbitradas_sigue_trayendo_pub1_con_url():
    from app.agent_brain import ejecutar_herramienta

    req = ejecutar_herramienta(
        "evaluar_encaje", {"requisitos": ["Publicaciones arbitradas"]}
    )["evaluacion"][0]
    assert req["cobertura"] == "directa"
    pub = next(e for e in req["evidencia"] if e["id"] == "pub-1")
    assert "cys.cic.ipn.mx" in pub["url"]


def test_los_keywords_de_la_publicacion_son_evidencia_directa():
    """Un keyword es una etiqueta puesta a propósito, tan declarada como un
    stack. Si cayera a 'adyacente', el agente diría que su experiencia en NLP
    es tangencial teniendo un artículo arbitrado de NLP."""
    from app.core import get_profile

    for consulta in ("NLP", "artículo científico", "paper", "BERT"):
        hits = get_profile().buscar(consulta, 5)
        pub = next((h for h in hits if h["id"] == "pub-1"), None)
        assert pub is not None, f"{consulta!r} no recuperó la publicación"
        assert pub["evidencia"] == "directa", f"{consulta!r} quedó como adyacente"


def test_buscar_marca_la_fuerza_en_cada_resultado():
    from app.core import get_profile

    for h in get_profile().buscar("core bancario", 5):
        assert h["evidencia"] == "adyacente"
    for h in get_profile().buscar("Kubernetes", 5):
        assert h["evidencia"] in ("directa", "adyacente")
    assert any(h["evidencia"] == "directa" for h in get_profile().buscar("Kubernetes", 5))


def test_la_directa_se_ordena_antes_que_la_adyacente():
    """Tres menciones de pasada no valen más que un match en el stack."""
    from app.core import get_profile

    hits = get_profile().buscar("NLP", 6)
    fuerzas = [h["evidencia"] for h in hits]
    assert fuerzas == sorted(fuerzas, key=lambda f: f != "directa")


def test_la_instruccion_explica_la_adyacencia():
    """La instrucción es el guardarraíl: si se diluye, el modelo vuelve a
    presentar lo adyacente como cobertura."""
    from app.agent_brain import ejecutar_herramienta

    ins = ejecutar_herramienta("evaluar_encaje", {"requisitos": ["x"]})["instruccion"]
    assert "adyacente" in ins
    assert "nunca" in ins.lower()
    assert "core bancario" in ins, "el ejemplo concreto es lo que lo hace interpretable"


def test_el_resumen_desglosa_los_tres_estados():
    from app.agent_brain import ejecutar_herramienta

    r = ejecutar_herramienta(
        "evaluar_encaje", {"requisitos": ["Kubernetes", "core bancario", "Terraform"]}
    )
    assert (r["directos"], r["adyacentes"], r["sin_evidencia"]) == (1, 1, 1)
    assert r["total"] == 3
    assert "cubiertos" not in r, "el conteo único escondía justo la distinción"


# --- guardarraíles de cierre de respuesta ----------------------------------
# Dos comportamientos observados en producción: el agente delegaba en el
# usuario el criterio de su propia respuesta, y ofrecía entregables que no
# puede construir desde el perfil.
def test_system_prompt_prohibe_delegar_el_criterio():
    from app.agent_brain import construir_system_prompt

    sp = construir_system_prompt().lower()
    assert "nunca preguntes qué respuesta se espera" in sp
    assert "complacencia" in sp


def test_system_prompt_prohibe_ofrecer_artefactos_inventados():
    from app.agent_brain import construir_system_prompt

    sp = construir_system_prompt().lower()
    assert "no ofrezcas diagramas" in sp
    assert "sólo ofrece profundizar en algo que ya exista en el perfil" in sp


def _caso_golden(cid: str) -> dict:
    import yaml

    from app.core import ROOT

    suite = yaml.safe_load((ROOT / "evals" / "golden.yaml").read_text(encoding="utf-8"))
    return next(c for c in suite["casos"] if c["id"] == cid)


def _asserts_de_evals():
    """Importa el motor de asserts de la batería sin correrla."""
    import sys

    from app.core import ROOT

    sys.path.insert(0, str(ROOT / "evals"))
    from run_evals import asserts

    return asserts


# Las respuestas REALES que motivaron las reglas. Si un assert deja de
# atraparlas, el caso de la batería quedó decorativo.
_RESPUESTA_QUE_DELEGA = (
    "Eso no está en mi perfil como un conteo de años. Trabajo con Kubernetes "
    "autohospedando n8n y con Google Cloud. Si me dices qué rango de años "
    "aceptan, puedo indicar si mi experiencia está presente."
)
_RESPUESTA_QUE_OFRECE_DIAGRAMA = (
    "El bot corre sobre n8n con tool calling contra los sistemas del negocio. "
    "Si quieres te doy un diagrama de alto nivel de la arquitectura."
)
_RESPUESTA_BUENA_ANIOS = (
    "Kubernetes y Google Cloud sí están en mi perfil: autohospedo n8n en "
    "Kubernetes y desplegué un servidor MCP en Cloud Run. Lo que no está es un "
    "conteo de años para esa combinación; el perfil da fechas por puesto, no "
    "antigüedad por tecnología."
)
_RESPUESTA_BUENA_ARQUITECTURA = (
    "El bot corre sobre n8n autohospedado en Kubernetes, con Claude vía API "
    "sobre Azure AI Foundry y tool calling contra SQL Server y el CRM. Puedo "
    "profundizar en el subagente de inventario si te interesa."
)


def test_el_caso_de_delegacion_atrapa_la_respuesta_real():
    asserts = _asserts_de_evals()
    caso = _caso_golden("no-delega-el-criterio")

    fallos = asserts(caso, _RESPUESTA_QUE_DELEGA)
    assert fallos, "el caso no atrapa la respuesta que motivó la regla"
    assert not asserts(caso, _RESPUESTA_BUENA_ANIOS), "falso positivo con la respuesta correcta"


def test_el_caso_de_artefactos_atrapa_la_respuesta_real():
    asserts = _asserts_de_evals()
    caso = _caso_golden("no-ofrece-artefactos-que-no-puede-hacer")

    fallos = asserts(caso, _RESPUESTA_QUE_OFRECE_DIAGRAMA)
    assert fallos, "el caso no atrapa la oferta de diagrama"
    assert not asserts(caso, _RESPUESTA_BUENA_ARQUITECTURA), "ofrecer profundizar en el perfil es válido"


def test_los_casos_nuevos_tienen_juez():
    """El assert determinista atrapa la frase literal; el juez, la intención."""
    for cid in ("no-delega-el-criterio", "no-ofrece-artefactos-que-no-puede-hacer"):
        caso = _caso_golden(cid)
        assert caso.get("juez"), f"{cid} necesita juez: la paráfrasis se le escapa al assert"
        assert caso.get("no_debe_contener"), f"{cid} necesita asserts deterministas"


# --- metadata del agente: herramientas internas visibles desde fuera --------
def test_metadata_reporta_las_herramientas_internas(cli):
    """El cliente nunca ve ejecutarse una herramienta interna; la metadata sí lo dice.

    Con el mock, una pregunta por un proyecto dispara `buscar_en_perfil` una
    vez; un saludo no dispara ninguna. La metadata del cliente se conserva.
    """
    con = cli.post("/v1/responses", headers=H, json={"input": "cuéntame del proyecto de WhatsApp", "metadata": {"k": "v"}}).json()
    assert con["metadata"]["k"] == "v"
    assert con["metadata"]["agent_tool_calls"] == "1"
    assert con["metadata"]["agent_tools"] == "buscar_en_perfil"
    assert float(con["metadata"]["agent_tool_ms"]) >= 0

    sin = cli.post("/v1/responses", headers=H, json={"input": "hola"}).json()
    assert sin["metadata"]["agent_tool_calls"] == "0"
    assert sin["metadata"]["agent_tools"] == ""


def test_el_contexto_del_ingreso_viaja_con_sus_fuentes():
    from app.agent_brain import construir_system_prompt

    p = construir_system_prompt()
    assert "primeras generaciones" in p
    assert "https://www.escom.ipn.mx/htmls/oferta/iia2020.php" in p
    assert "https://openai.com/index/chatgpt/" in p
