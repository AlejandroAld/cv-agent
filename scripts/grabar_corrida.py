#!/usr/bin/env python3
"""Graba una corrida real del agente desplegado, para que el sitio la reproduzca.

El portafolio cuenta la página como la ejecución de este agente respondiendo
"¿Quién es Alex y por qué debería contratarlo?". No corre nada en vivo: sólo
reproduce lo que este script grabó, y cada número que muestra sale de aquí.

    AGENT_API_KEY=... python scripts/grabar_corrida.py --lang es \\
        --out ../personal-portfolio/src/content/runs/es.json
    AGENT_API_KEY=... python scripts/grabar_corrida.py --lang en \\
        --out ../personal-portfolio/src/content/runs/en.json

Qué graba, todo con marca de tiempo relativa al envío de la petición:
  - cada evento SSE de POST /v1/responses (los deltas con su texto);
  - el objeto Response final: id, modelo, usage (entrada, salida, razonamiento)
    y la metadata agent_* con las llamadas a herramientas internas;
  - los tiempos por etapa: created, primer token, último token, completed;
  - el contexto que este mismo código mete al prompt (como_contexto), bloque
    por bloque, contado con el tokenizador o200k_base. Ese conteo es un
    cálculo reproducible sobre el texto exacto; el total de entrada que manda
    es el `usage` del proveedor, que también se guarda.

Requisitos: `pip install tiktoken` además de requirements.txt. El script se
corre desde la raíz del repo, en el commit que está desplegado: el SHA que
guarda es el de HEAD, y el sitio lo cita.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from app.agent_brain import HERRAMIENTAS_INTERNAS, PLANTILLA  # noqa: E402
from app.core import get_profile  # noqa: E402

PREGUNTAS = {
    "es": "¿Quién es Alex y por qué debería contratarlo?",
    "en": "Who is Alex and why should I hire him?",
}
URL_POR_DEFECTO = "https://cv-agent.mangopond-59d644ac.eastus2.azurecontainerapps.io"


def _tokenizador():
    try:
        import tiktoken
    except ImportError:
        sys.exit("Falta tiktoken: pip install tiktoken")
    return tiktoken.get_encoding("o200k_base")


def _bloques_de_contexto(enc) -> dict:
    """El contexto exacto que construir_system_prompt mete al prompt, por bloque."""
    p = get_profile()
    contexto = p.como_contexto()
    bloques = []
    for trozo in contexto.split("\n\n# "):
        trozo = trozo if trozo.startswith("# ") else "# " + trozo
        titulo = trozo.split("\n", 1)[0].lstrip("# ").strip()
        bloques.append({"title": titulo, "chars": len(trozo), "tokens": len(enc.encode(trozo))})
    alias = p.persona.get("alias") or p.nombre.split()[0]
    plantilla = PLANTILLA.format(nombre=p.nombre, alias=alias, perfil="")
    return {
        "tokenizer": "o200k_base",
        "blocks": bloques,
        "profile_tokens": sum(b["tokens"] for b in bloques),
        "template_tokens": len(enc.encode(plantilla)),
        "tools_schema_tokens": len(enc.encode(json.dumps(HERRAMIENTAS_INTERNAS, ensure_ascii=False))),
        "tool_names": [t["name"] for t in HERRAMIENTAS_INTERNAS],
    }


def _sha() -> tuple[str, bool]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True, text=True, check=True).stdout.strip()
    sucio = bool(subprocess.run(["git", "status", "--porcelain"], cwd=RAIZ, capture_output=True, text=True).stdout.strip())
    return sha, sucio


def grabar(url: str, pregunta: str, api_key: str, timeout_s: float) -> dict:
    eventos: list[dict] = []
    respuesta_final: dict | None = None
    texto = ""
    t0 = time.perf_counter()
    marca = lambda: round((time.perf_counter() - t0) * 1000, 1)  # noqa: E731

    cuerpo = {"input": pregunta, "stream": True}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "text/event-stream"}

    with httpx.Client(timeout=timeout_s) as cli:
        with cli.stream("POST", f"{url.rstrip('/')}/v1/responses", headers=headers, json=cuerpo) as r:
            if r.status_code != 200:
                sys.exit(f"El agente respondió {r.status_code}: {r.read().decode('utf-8', 'replace')[:300]}")
            evento_actual: str | None = None
            for linea in r.iter_lines():
                if linea.startswith("event:"):
                    evento_actual = linea[6:].strip()
                    continue
                if not linea.startswith("data:"):
                    continue
                data = linea[5:].strip()
                t = marca()
                if data == "[DONE]":
                    eventos.append({"t_ms": t, "type": "[DONE]"})
                    break
                ev = json.loads(data)
                tipo = ev.get("type") or evento_actual
                registro = {"t_ms": t, "type": tipo, "seq": ev.get("sequence_number")}
                if tipo == "response.output_text.delta":
                    registro["delta"] = ev.get("delta", "")
                    texto += ev.get("delta", "")
                elif tipo in ("response.output_item.added", "response.output_item.done"):
                    item = ev.get("item") or {}
                    registro["item_type"] = item.get("type")
                    if item.get("type") == "function_call":
                        registro["name"] = item.get("name")
                elif tipo in ("response.completed", "response.failed", "response.incomplete"):
                    respuesta_final = ev.get("response")
                elif tipo == "error":
                    registro["message"] = ev.get("message")
                eventos.append(registro)

    if not respuesta_final or respuesta_final.get("status") != "completed":
        sys.exit(f"La corrida no terminó en completed: {json.dumps(respuesta_final, ensure_ascii=False)[:400]}")

    def primero(tipo: str):
        return next((e["t_ms"] for e in eventos if e["type"] == tipo), None)

    def ultimo(tipo: str):
        return next((e["t_ms"] for e in reversed(eventos) if e["type"] == tipo), None)

    meta = respuesta_final.get("metadata") or {}
    reportado = "agent_tool_calls" in meta
    return {
        "response": {
            "id": respuesta_final["id"],
            "model": respuesta_final.get("model"),
            "status": respuesta_final.get("status"),
            "created_at": respuesta_final.get("created_at"),
            "completed_at": respuesta_final.get("completed_at"),
            "usage": respuesta_final.get("usage"),
            "metadata": meta,
        },
        "timeline_ms": {
            "created": primero("response.created"),
            "first_token": primero("response.output_text.delta"),
            "last_token": ultimo("response.output_text.delta"),
            "completed": primero("response.completed"),
        },
        "tool_calls": {
            "reported": reportado,
            "count": int(meta.get("agent_tool_calls", 0)) if reportado else None,
            "names": [n for n in meta.get("agent_tools", "").split(",") if n] if reportado else None,
            "ms": [float(m) for m in meta.get("agent_tool_ms", "").split(",") if m] if reportado else None,
        },
        "events": eventos,
        "output_text": texto,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=sorted(PREGUNTAS), required=True)
    ap.add_argument("--out", required=True, help="ruta del JSON de salida (en el repo del sitio)")
    ap.add_argument("--url", default=os.getenv("AGENT_URL", URL_POR_DEFECTO))
    ap.add_argument("--question", default=None, help="por defecto, la pregunta del idioma")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    api_key = os.getenv("AGENT_API_KEY", "")
    if not api_key:
        sys.exit("Falta AGENT_API_KEY en el entorno (nunca va en el repo del sitio).")

    pregunta = args.question or PREGUNTAS[args.lang]
    sha, sucio = _sha()
    if sucio:
        print("AVISO: el árbol de cv-agent tiene cambios sin commit; el SHA grabado es HEAD.", file=sys.stderr)

    enc = _tokenizador()
    contexto = _bloques_de_contexto(enc)
    contexto["question_tokens"] = len(enc.encode(pregunta))

    print(f"==> POST {args.url}/v1/responses  [{args.lang}] {pregunta}", file=sys.stderr)
    corrida = grabar(args.url, pregunta, api_key, args.timeout)

    salida = {
        "schema": 1,
        "status": "recorded",
        "language": args.lang,
        "question": pregunta,
        "recorded_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "cv_agent_sha": sha,
        "endpoint": f"{args.url.rstrip('/')}/v1/responses",
        "recorder": "scripts/grabar_corrida.py",
        **corrida,
        "context": contexto,
    }

    destino = Path(args.out)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    u = salida["response"]["usage"] or {}
    tl = salida["timeline_ms"]
    print(
        f"\nGrabado en {destino}\n"
        f"  id {salida['response']['id']} · modelo {salida['response']['model']} · cv-agent {sha[:7]}\n"
        f"  tokens: entrada {u.get('input_tokens')} · salida {u.get('output_tokens')} · "
        f"razonamiento {(u.get('output_tokens_details') or {}).get('reasoning_tokens')}\n"
        f"  tiempos: created {tl['created']} ms · primer token {tl['first_token']} ms · completed {tl['completed']} ms\n"
        f"  herramientas: {salida['tool_calls']}\n"
        f"  contexto: {contexto['profile_tokens']} tokens de perfil en {len(contexto['blocks'])} bloques "
        f"(+{contexto['template_tokens']} de plantilla, +{contexto['tools_schema_tokens']} de herramientas)\n"
        f"\n--- respuesta grabada (apruébala antes de publicarla) ---\n{salida['output_text']}\n",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
