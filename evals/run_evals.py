#!/usr/bin/env python3
"""Corre la batería de evaluación contra un endpoint Open Responses.

    python evals/run_evals.py --base-url http://localhost:8080/v1 --api-key XXX

Dos capas de juicio:
  1. Asserts deterministas -> deciden el exit code. Son los que rompen el build.
  2. Juez LLM sobre fundamentación y tono -> se reporta siempre; sólo rompe el
     build con --strict, porque un juez puede equivocarse y no quiero un CI
     que falle por ruido.

Salida: tabla en consola + evals/reports/ultimo.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import httpx
import yaml

RAIZ = Path(__file__).resolve().parent.parent
REPORTES = RAIZ / "evals" / "reports"


def fold(t: str) -> str:
    n = unicodedata.normalize("NFD", t.lower())
    return "".join(c for c in n if unicodedata.category(c) != "Mn")


# ---------------------------------------------------------------------------
async def preguntar(
    client: httpx.AsyncClient, base: str, key: str, caso: dict[str, Any]
) -> tuple[str, float, str | None]:
    entrada: list[dict[str, Any]] = []
    for turno in caso.get("historial") or []:
        entrada.append(
            {
                "type": "message",
                "role": turno["rol"],
                "content": [
                    {
                        "type": "output_text" if turno["rol"] == "assistant" else "input_text",
                        "text": turno["texto"],
                    }
                ],
            }
        )
    entrada.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": caso["pregunta"]}]})

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    t0 = time.perf_counter()
    try:
        r = await client.post(
            f"{base.rstrip('/')}/responses",
            headers=headers,
            json={"model": "cv-agent", "input": entrada},
            timeout=90,
        )
    except Exception as exc:
        return "", (time.perf_counter() - t0) * 1000, f"red: {exc!r}"

    ms = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        return "", ms, f"HTTP {r.status_code}: {r.text[:200]}"

    data = r.json()
    texto = data.get("output_text") or "".join(
        c.get("text", "")
        for it in data.get("output", [])
        if it.get("type") == "message"
        for c in it.get("content", [])
        if c.get("type") == "output_text"
    )
    return texto, ms, None


def asserts(caso: dict[str, Any], texto: str) -> list[str]:
    fallos: list[str] = []
    t = fold(texto)

    incluye = caso.get("debe_contener_alguno")
    if incluye and not any(fold(x) in t for x in incluye):
        fallos.append(f"no menciona ninguno de {incluye}")

    for prohibido in caso.get("no_debe_contener") or []:
        if fold(prohibido) in t:
            fallos.append(f"contiene texto prohibido: {prohibido!r}")

    palabras = len(texto.split())
    if caso.get("min_palabras") and palabras < caso["min_palabras"]:
        fallos.append(f"muy corta: {palabras} palabras (min {caso['min_palabras']})")
    if caso.get("max_palabras") and palabras > caso["max_palabras"]:
        fallos.append(f"muy larga: {palabras} palabras (max {caso['max_palabras']})")
    if not texto.strip():
        fallos.append("respuesta vacía")
    return fallos


async def juzgar(client: httpx.AsyncClient, criterio: str, pregunta: str, respuesta: str) -> tuple[bool | None, str]:
    """Juez LLM. Devuelve (aprueba, razon). None si no hay credenciales."""
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "sin JUDGE_API_KEY"
    modelo = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

    prompt = (
        "Evalúas la respuesta de un agente de CV. Sé estricto pero justo.\n\n"
        f"PREGUNTA:\n{pregunta}\n\nRESPUESTA:\n{respuesta}\n\n"
        f"CRITERIO:\n{criterio}\n\n"
        'Contesta SOLO con JSON: {"aprueba": true|false, "razon": "una frase"}'
    )
    try:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": modelo,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        cuerpo = json.loads(r.json()["choices"][0]["message"]["content"])
        return bool(cuerpo.get("aprueba")), str(cuerpo.get("razon", ""))[:160]
    except Exception as exc:
        return None, f"juez falló: {exc!r}"


# ---------------------------------------------------------------------------
async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.getenv("EVAL_BASE_URL", "http://localhost:8080/v1"))
    ap.add_argument("--api-key", default=os.getenv("EVAL_API_KEY", os.getenv("AGENT_API_KEY", "")))
    ap.add_argument("--suite", default=str(RAIZ / "evals" / "golden.yaml"))
    ap.add_argument("--strict", action="store_true", help="el juez LLM también rompe el build")
    ap.add_argument("--concurrencia", type=int, default=4)
    args = ap.parse_args()

    casos = (yaml.safe_load(Path(args.suite).read_text(encoding="utf-8")) or {}).get("casos", [])
    print(f"Corriendo {len(casos)} casos contra {args.base_url}\n")

    sem = asyncio.Semaphore(args.concurrencia)
    resultados: list[dict[str, Any]] = []

    async with httpx.AsyncClient() as client:

        async def correr(caso: dict[str, Any]) -> None:
            async with sem:
                texto, ms, err = await preguntar(client, args.base_url, args.api_key, caso)
                fallos = [err] if err else asserts(caso, texto)
                veredicto_juez, razon = (None, "")
                if caso.get("juez") and not err:
                    veredicto_juez, razon = await juzgar(client, caso["juez"], caso["pregunta"], texto)
                resultados.append(
                    {
                        "id": caso["id"],
                        "categoria": caso.get("categoria", "-"),
                        "ms": round(ms),
                        "duro_ok": not fallos,
                        "fallos": fallos,
                        "juez": veredicto_juez,
                        "razon": razon,
                        "respuesta": texto,
                    }
                )

        await asyncio.gather(*(correr(c) for c in casos))

    orden = {c["id"]: i for i, c in enumerate(casos)}
    resultados.sort(key=lambda r: orden.get(r["id"], 999))

    duros_mal = [r for r in resultados if not r["duro_ok"]]
    juez_mal = [r for r in resultados if r["juez"] is False]
    juez_ok = sum(1 for r in resultados if r["juez"] is True)
    juez_total = sum(1 for r in resultados if r["juez"] is not None)
    latencias = sorted(r["ms"] for r in resultados)
    p50 = latencias[len(latencias) // 2] if latencias else 0
    p95 = latencias[int(len(latencias) * 0.95) - 1] if len(latencias) > 1 else p50

    for r in resultados:
        marca = "PASS" if r["duro_ok"] and r["juez"] is not False else "FAIL"
        print(f"[{marca}] {r['id']:<24} {r['categoria']:<15} {r['ms']:>6} ms")
        for f in r["fallos"]:
            print(f"         assert: {f}")
        if r["juez"] is False:
            print(f"         juez:   {r['razon']}")

    print(
        f"\nAsserts: {len(resultados) - len(duros_mal)}/{len(resultados)}"
        f" | Juez: {juez_ok}/{juez_total}"
        f" | Latencia p50 {p50} ms, p95 {p95} ms"
    )

    # Reporte markdown
    REPORTES.mkdir(parents=True, exist_ok=True)
    lineas = [
        "# Reporte de evaluación",
        "",
        f"Endpoint: `{args.base_url}`  ",
        f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Asserts: **{len(resultados)-len(duros_mal)}/{len(resultados)}** · "
        f"Juez LLM: **{juez_ok}/{juez_total}** · Latencia p50 **{p50} ms**, p95 **{p95} ms**",
        "",
        "| Caso | Categoría | ms | Asserts | Juez | Nota |",
        "|---|---|---:|:--:|:--:|---|",
    ]
    for r in resultados:
        juez_txt = {True: "ok", False: "falla", None: "-"}[r["juez"]]
        nota = "; ".join(r["fallos"]) or r["razon"] or ""
        lineas.append(
            f"| `{r['id']}` | {r['categoria']} | {r['ms']} | "
            f"{'ok' if r['duro_ok'] else 'falla'} | {juez_txt} | {nota[:110]} |"
        )
    (REPORTES / "ultimo.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"\nReporte: {REPORTES / 'ultimo.md'}")

    if duros_mal:
        return 1
    if args.strict and juez_mal:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
