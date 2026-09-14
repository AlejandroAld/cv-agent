"""Configuración, logging estructurado y acceso al perfil.

El perfil se carga una vez al arranque. Es pequeño (unos pocos miles de
tokens), así que vive completo en memoria: no hay base vectorial, y esa
decisión está documentada en el README.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    provider: str = os.getenv("LLM_PROVIDER", "mock").lower()
    model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    azure_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    azure_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    compat_base_url: str = os.getenv("LLM_BASE_URL", "").rstrip("/")
    compat_api_key: str = os.getenv("LLM_API_KEY", "")

    agent_api_key: str = os.getenv("AGENT_API_KEY", "")
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "http://localhost:8080/v1")

    max_tool_iterations: int = int(os.getenv("MAX_TOOL_ITERATIONS", "4"))
    request_timeout_s: float = float(os.getenv("REQUEST_TIMEOUT_S", "60"))
    max_input_chars: int = int(os.getenv("MAX_INPUT_CHARS", "24000"))
    profile_path: str = os.getenv("PROFILE_PATH", str(ROOT / "data" / "perfil.yaml"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# --------------------------------------------------------------------------
# Logging estructurado (una línea JSON por evento -> Cloud Logging lo parsea)
# --------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "severity": record.levelname,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("cv-agent")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    return logger


log = setup_logging()


def log_event(message: str, **fields: Any) -> None:
    log.info(message, extra={"extra_fields": fields})


# --------------------------------------------------------------------------
# Perfil
# --------------------------------------------------------------------------
def _fold(text: str) -> str:
    """Minúsculas sin acentos, para que 'valuacion' encuentre 'valuación'."""
    norm = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in norm if unicodedata.category(c) != "Mn")


@dataclass
class Profile:
    raw: dict[str, Any] = field(default_factory=dict)

    # ---- accesores -------------------------------------------------------
    @property
    def persona(self) -> dict[str, Any]:
        return self.raw.get("persona", {}) or {}

    @property
    def nombre(self) -> str:
        return self.persona.get("nombre", "el candidato")

    @property
    def experiencia(self) -> list[dict[str, Any]]:
        return self.raw.get("experiencia", []) or []

    @property
    def proyectos(self) -> list[dict[str, Any]]:
        return self.raw.get("proyectos", []) or []

    @property
    def habilidades(self) -> list[dict[str, Any]]:
        return self.raw.get("habilidades", []) or []

    @property
    def faq(self) -> list[dict[str, Any]]:
        return self.raw.get("faq", []) or []

    def todas_las_skills(self) -> list[str]:
        out: list[str] = []
        for grupo in self.habilidades:
            out.extend(grupo.get("items", []) or [])
        return out

    # ---- render para el system prompt ------------------------------------
    def como_contexto(self) -> str:
        """El perfil completo, aplanado a texto. Esto es el 'contexto' del agente."""
        p = self.persona
        partes: list[str] = []

        partes.append(
            f"# Identidad\n"
            f"Nombre: {p.get('nombre','')}\n"
            f"Titular: {p.get('titular','')}\n"
            f"Ubicación: {p.get('ubicacion','')}\n"
            f"Disponibilidad: {p.get('disponibilidad','')}\n"
            f"Idiomas: "
            + ", ".join(
                f"{i.get('idioma')} ({i.get('nivel')})" for i in p.get("idiomas", []) or []
            )
        )

        contacto = p.get("contacto", {}) or {}
        publicos = {k: v for k, v in contacto.items() if v}
        if publicos:
            partes.append(
                "# Contacto público\n"
                + "\n".join(f"{k}: {v}" for k, v in publicos.items())
            )

        if self.raw.get("resumen"):
            partes.append("# Resumen\n" + str(self.raw["resumen"]).strip())

        if self.experiencia:
            bloques = []
            for e in self.experiencia:
                encabezado = f"[{e.get('id')}] {e.get('puesto','')}"
                if e.get("empresa"):
                    encabezado += f" — {e['empresa']}"
                if e.get("cliente"):
                    encabezado += f" (cliente: {e['cliente']})"
                encabezado += f" | {e.get('inicio','?')} – {e.get('fin','?')}"
                cuerpo = [encabezado]
                if e.get("resumen"):
                    cuerpo.append(str(e["resumen"]).strip())
                for l in e.get("logros", []) or []:
                    cuerpo.append(f"  - {l}")
                if e.get("stack"):
                    cuerpo.append("  Stack: " + ", ".join(e["stack"]))
                bloques.append("\n".join(cuerpo))
            partes.append("# Experiencia\n" + "\n\n".join(bloques))

        if self.proyectos:
            bloques = []
            for pr in self.proyectos:
                cuerpo = [f"[{pr.get('id')}] {pr.get('nombre','')}"]
                if pr.get("resumen"):
                    cuerpo.append(str(pr["resumen"]).strip())
                if pr.get("detalle"):
                    cuerpo.append(str(pr["detalle"]).strip())
                if pr.get("stack"):
                    cuerpo.append("  Stack: " + ", ".join(pr["stack"]))
                if pr.get("repo"):
                    cuerpo.append(f"  Repo: {pr['repo']}")
                bloques.append("\n".join(cuerpo))
            partes.append("# Proyectos\n" + "\n\n".join(bloques))

        if self.habilidades:
            partes.append(
                "# Habilidades\n"
                + "\n".join(
                    f"{g.get('categoria','')}: " + ", ".join(g.get("items", []) or [])
                    for g in self.habilidades
                )
            )

        edu = [e for e in (self.raw.get("educacion") or []) if e.get("titulo")]
        if edu:
            partes.append(
                "# Educación\n"
                + "\n".join(
                    f"{e.get('titulo')} — {e.get('institucion','')} ({e.get('periodo','')})"
                    for e in edu
                )
            )

        certs = self.raw.get("certificaciones") or []
        if certs:
            partes.append("# Certificaciones\n" + "\n".join(f"- {c}" for c in certs))

        if self.faq:
            partes.append(
                "# Respuestas preparadas (úsalas casi literales cuando apliquen)\n"
                + "\n\n".join(
                    f"P: {f.get('pregunta')}\nR: {str(f.get('respuesta','')).strip()}"
                    for f in self.faq
                )
            )

        privado = p.get("privado", []) or []
        if privado:
            partes.append(
                "# Datos que NO debes revelar\n" + ", ".join(privado)
            )

        return "\n\n".join(partes)

    # ---- búsqueda determinista (usada por las herramientas) --------------
    def buscar(self, consulta: str, limite: int = 5) -> list[dict[str, Any]]:
        """Scoring léxico simple sobre experiencia + proyectos.

        A propósito NO es embeddings: el corpus son decenas de registros, el
        vocabulario es técnico y literal, y un match léxico es explicable,
        instantáneo y no necesita infraestructura extra.
        """
        terminos = [t for t in _fold(consulta).split() if len(t) > 2]
        if not terminos:
            return []

        candidatos: list[tuple[float, dict[str, Any]]] = []
        for e in self.experiencia:
            texto = _fold(json.dumps(e, ensure_ascii=False))
            score = sum(3.0 if t in _fold(e.get("puesto", "") + " " + " ".join(e.get("stack", []) or [])) else (1.0 if t in texto else 0.0) for t in terminos)
            if score > 0:
                candidatos.append((score, {"tipo": "experiencia", **e}))
        for pr in self.proyectos:
            texto = _fold(json.dumps(pr, ensure_ascii=False))
            score = sum(3.0 if t in _fold(pr.get("nombre", "") + " " + " ".join(pr.get("stack", []) or [])) else (1.0 if t in texto else 0.0) for t in terminos)
            if score > 0:
                candidatos.append((score, {"tipo": "proyecto", **pr}))

        candidatos.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in candidatos[:limite]]


@lru_cache(maxsize=1)
def get_profile() -> Profile:
    path = Path(get_settings().profile_path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    log_event("profile_loaded", path=str(path), experiencias=len(data.get("experiencia") or []))
    return Profile(raw=data)
