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
    assert d["metadata"] == {"origen": "prueba"}
    assert d["truncation"] == "auto"


def test_modelo_opcional(cli):
    assert cli.post("/v1/responses", headers=H, json={"input": "hola"}).json()["model"]


# --- herramientas del cliente ---------------------------------------------
def test_function_tool_del_cliente_cede_control(cli, monkeypatch):
    async def fake(messages, tools=None, temperature=None, max_tokens=None):
        yield {"t": "tool", "index": 0, "id": "call_1", "name": "get_weather", "args_delta": '{"city":"GDL"}'}
        yield {"t": "done", "finish_reason": "tool_calls", "usage": None}

    import app.main as main

    monkeypatch.setattr(main, "stream_chat", fake)
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


# --- descubrimiento --------------------------------------------------------
def test_healthz(cli):
    assert cli.get("/healthz").json()["status"] == "ok"


def test_agent_card(cli):
    card = cli.get("/.well-known/agent-card.json").json()
    for campo in ("name", "description", "version", "url", "capabilities", "skills"):
        assert campo in card
    assert card["capabilities"]["streaming"] is True
    assert len(card["skills"]) >= 1


# --- herramientas internas -------------------------------------------------
def test_herramientas_internas_no_alucinan():
    from app.agent_brain import ejecutar_herramienta

    vacio = ejecutar_herramienta("buscar_en_perfil", {"consulta": "kubernetes terraform"})
    assert vacio["encontrados"] == 0 and vacio["nota"]

    hit = ejecutar_herramienta("buscar_en_perfil", {"consulta": "n8n WhatsApp"})
    assert hit["encontrados"] > 0

    encaje = ejecutar_herramienta("evaluar_encaje", {"requisitos": ["Kubernetes", "Firebase"]})
    por_req = {e["requisito"]: e["cubierto"] for e in encaje["evaluacion"]}
    assert por_req["Kubernetes"] is False, "debe reportar honestamente lo que no cubre"
    assert por_req["Firebase"] is True

    assert ejecutar_herramienta("obtener_detalle", {"id": "nope"})["error"] == "not_found"


def test_contacto_no_expone_datos_privados():
    from app.agent_brain import ejecutar_herramienta

    contacto = ejecutar_herramienta("obtener_contacto", {})["contacto"]
    assert "telefono" not in contacto and "direccion" not in contacto


def test_system_prompt_incluye_el_perfil():
    from app.agent_brain import construir_system_prompt

    p = construir_system_prompt()
    assert "PERFIL" in p and "fundamentación" in p.lower()
