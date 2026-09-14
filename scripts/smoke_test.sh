#!/usr/bin/env bash
# Prueba de humo contra un endpoint desplegado.
#   ./scripts/smoke_test.sh https://mi-agente.run.app/v1 MI_API_KEY
set -euo pipefail
BASE="${1:-http://localhost:8080/v1}"
KEY="${2:-}"
AUTH=(); [[ -n "$KEY" ]] && AUTH=(-H "Authorization: Bearer $KEY")

echo "== 1. health =="
curl -sf "${BASE%/v1}/healthz"; echo

echo "== 2. agent card =="
curl -sf "${BASE%/v1}/.well-known/agent-card.json" | head -c 300; echo

echo "== 3. respuesta simple =="
curl -sf -X POST "$BASE/responses" "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d '{"model":"cv-agent","input":"¿Cuál es tu experiencia con agentes de IA?"}' | head -c 800; echo

echo "== 4. streaming =="
curl -sfN -X POST "$BASE/responses" "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d '{"model":"cv-agent","stream":true,"input":"Resume tu perfil en una frase"}' | head -25

echo "== 5. sin credencial (debe dar 401) =="
curl -s -o /dev/null -w '%{http_code}\n' -X POST "$BASE/responses" \
  -H 'Content-Type: application/json' -d '{"input":"hola"}'
