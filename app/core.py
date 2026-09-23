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
    model: str = os.getenv("LLM_MODEL", "gpt-5-mini")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    azure_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    # La v1 GA no pide api-version. Se deja vacío a propósito; si el recurso
    # llegara a responder 400 pidiéndola, se define AZURE_OPENAI_API_VERSION=preview.
    azure_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "")

    compat_base_url: str = os.getenv("LLM_BASE_URL", "").rstrip("/")
    compat_api_key: str = os.getenv("LLM_API_KEY", "")

    agent_api_key: str = os.getenv("AGENT_API_KEY", "")
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "http://localhost:8080/v1")

    max_tool_iterations: int = int(os.getenv("MAX_TOOL_ITERATIONS", "4"))
    # Los modelos de razonamiento gastan tokens pensando antes de responder.
    # "minimal" es lo que este agente necesita: el perfil ya va completo en el
    # prompt, así que no hay nada que deducir, sólo que citar bien.
    reasoning_effort: str = os.getenv("REASONING_EFFORT", "minimal")
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


def _variantes(termino: str) -> tuple[str, ...]:
    """El término y su singular, para que 'proyectos' encuentre 'proyecto'.

    El match es por subcadena, así que sin esto 'arbitradas' no encuentra
    'arbitrada' y una vacante en plural da falso negativo. Sólo se recorta a
    partir de cinco caracteres: 'has' o 'api' recortados generan ruido.
    """
    if len(termino) <= 4:
        return (termino,)
    if termino.endswith("es"):
        return (termino, termino[:-1], termino[:-2])
    return (termino, termino[:-1])


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
    def publicaciones(self) -> list[dict[str, Any]]:
        """Las publicaciones, con un id derivado para poder citarlas.

        Esta sección del YAML no trae id propio. Se deriva por posición para
        que `obtener_detalle` pueda resolverla igual que una experiencia o un
        proyecto: sin id no hay cita verificable, que es para lo que existen
        las herramientas. Si el YAML llega a traer uno, ese gana.
        """
        salida: list[dict[str, Any]] = []
        for i, pub in enumerate(self.raw.get("publicaciones", []) or [], start=1):
            if isinstance(pub, dict):
                salida.append({"id": f"pub-{i}", **pub})
        return salida

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
            lineas_edu = []
            for e in edu:
                lineas_edu.append(f"{e.get('titulo')} — {e.get('institucion','')} ({e.get('periodo','')})")
                # El contexto del ingreso viaja con sus fuentes: es un dato que
                # el agente puede afirmar, y la URL es lo que lo hace verificable.
                if e.get("contexto"):
                    lineas_edu.append("  " + str(e["contexto"]).strip())
                for f in e.get("fuentes") or []:
                    if isinstance(f, dict) and f.get("url"):
                        lineas_edu.append(f"  Fuente: {f.get('descripcion', '')} — {f['url']}")
            partes.append("# Educación\n" + "\n".join(lineas_edu))

        if self.publicaciones:
            bloques = []
            for pub in self.publicaciones:
                cuerpo = [f"[{pub.get('id')}] {pub.get('titulo','')}"]
                if pub.get("medio"):
                    cuerpo.append(f"  Medio: {pub['medio']}")
                if pub.get("idioma"):
                    cuerpo.append(f"  Idioma: {pub['idioma']}")
                # La URL va completa y literal: es lo que hace la cita
                # verificable, y el agente debe poder darla si se la piden.
                if pub.get("url"):
                    cuerpo.append(f"  URL: {pub['url']}")
                bloques.append("\n".join(cuerpo))
            partes.append("# Publicaciones\n" + "\n\n".join(bloques))

        # Las certificaciones son dicts {nombre, estado}, pero se acepta un
        # string suelto: el formato de esta sección ya cambió una vez.
        lineas_cert = []
        for c in self.raw.get("certificaciones") or []:
            if isinstance(c, dict):
                nombre = str(c.get("nombre") or "").strip()
                if not nombre:
                    continue  # sin nombre no hay nada que afirmar
                estado = str(c.get("estado") or "").strip()
                lineas_cert.append(f"- {nombre} ({estado})" if estado else f"- {nombre}")
            elif c:
                lineas_cert.append(f"- {c}")
        if lineas_cert:
            partes.append("# Certificaciones\n" + "\n".join(lineas_cert))

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
    @staticmethod
    def _puntuar(terminos: list[str], destacado: str, texto: str) -> tuple[float, str]:
        """Devuelve (score, fuerza de la evidencia).

        La distinción es el punto: un término que pega en lo DESTACADO —título,
        puesto, nombre de proyecto, stack, keywords— es experiencia declarada.
        Uno que sólo pega dentro de la prosa del registro es, a lo mucho, un
        tema que el perfil roza.

        Los dos sumaban al mismo booleano, y por eso un requisito de "core
        bancario" salía cubierto apoyado en "convención bancaria base 360", que
        es una convención de conteo de días, no integración con un core
        bancario. Ese estiramiento se detecta en la primera entrevista.
        """
        score = 0.0
        directa = False
        for t in terminos:
            variantes = _variantes(t)
            if any(v in destacado for v in variantes):
                score += 3.0
                directa = True
            elif any(v in texto for v in variantes):
                score += 1.0
        if score <= 0:
            return 0.0, "sin_evidencia"
        return score, "directa" if directa else "adyacente"

    def buscar(self, consulta: str, limite: int = 5) -> list[dict[str, Any]]:
        """Scoring léxico sobre experiencia + proyectos + publicaciones.

        A propósito NO es embeddings: el corpus son decenas de registros, el
        vocabulario es técnico y literal, y un match léxico es explicable,
        instantáneo y no necesita infraestructura extra.

        Cada resultado viaja con su `evidencia`: "directa" o "adyacente". Sin
        eso, quien consume la búsqueda no puede distinguir un match en el stack
        de uno dentro de una frase en prosa, y `evaluar_encaje` termina
        presentando lo segundo como cobertura.

        Las publicaciones se recorren aquí y no en una herramienta aparte
        porque `evaluar_encaje` se apoya en esta función: si no estuvieran, una
        vacante que pidiera publicaciones daría sin evidencia contra una
        publicación arbitrada que sí existe.

        La categoría del registro entra al texto buscable porque es un dato
        real del perfil, no un sinónimo inventado: sin ella, buscar
        "publicaciones" no encuentra la publicación, porque esa palabra no
        aparece dentro del registro.
        """
        terminos = [t for t in _fold(consulta).split() if len(t) > 2]
        if not terminos:
            return []

        candidatos: list[tuple[float, bool, dict[str, Any]]] = []

        def considerar(registro: dict[str, Any], tipo: str, destacado: str, texto: str) -> None:
            score, fuerza = self._puntuar(terminos, destacado, texto)
            if score > 0:
                # tipo y evidencia se calculan aquí: mandan sobre el registro.
                candidatos.append(
                    (score, fuerza == "directa", {**registro, "tipo": tipo, "evidencia": fuerza})
                )

        for e in self.experiencia:
            considerar(
                e,
                "experiencia",
                _fold(f"{e.get('puesto', '')} {' '.join(e.get('stack', []) or [])}"),
                _fold("experiencia " + json.dumps(e, ensure_ascii=False)),
            )

        for pr in self.proyectos:
            considerar(
                pr,
                "proyecto",
                _fold(f"{pr.get('nombre', '')} {' '.join(pr.get('stack', []) or [])}"),
                _fold("proyecto " + json.dumps(pr, ensure_ascii=False)),
            )

        for pub in self.publicaciones:
            # Los keywords van a lo destacado, no a la prosa: son etiquetas que
            # alguien puso a propósito, tan declaradas como un stack. Dejarlos
            # en prosa marcaría "NLP" como adyacente contra un artículo
            # arbitrado de NLP, que es el falso negativo al revés.
            considerar(
                pub,
                "publicacion",
                _fold(
                    f"{pub.get('titulo', '')} {pub.get('medio', '')} "
                    + " ".join(pub.get("keywords", []) or [])
                ),
                _fold("publicacion " + json.dumps(pub, ensure_ascii=False)),
            )

        # La evidencia directa gana sobre la adyacente aunque sume menos puntos:
        # tres menciones de pasada no valen más que un match en el stack.
        candidatos.sort(key=lambda c: (c[1], c[0]), reverse=True)
        return [registro for _, _, registro in candidatos[:limite]]


@lru_cache(maxsize=1)
def get_profile() -> Profile:
    path = Path(get_settings().profile_path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    log_event("profile_loaded", path=str(path), experiencias=len(data.get("experiencia") or []))
    return Profile(raw=data)
