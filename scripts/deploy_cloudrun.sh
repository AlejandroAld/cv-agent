#!/usr/bin/env bash
# Despliegue a Google Cloud Run desde el codigo fuente (no requiere Docker local).
#
#   export PROJECT_ID=mi-proyecto
#   export OPENAI_API_KEY=sk-...
#   export AGENT_API_KEY="$(openssl rand -hex 24)"
#   ./scripts/deploy_cloudrun.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?define PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-cv-agent}"
AGENT_API_KEY="${AGENT_API_KEY:?define AGENT_API_KEY}"
OPENAI_API_KEY="${OPENAI_API_KEY:?define OPENAI_API_KEY}"
LLM_MODEL="${LLM_MODEL:-gpt-5-mini}"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com

# Secretos: nunca en variables de entorno planas
for par in "cv-agent-llm-key:$OPENAI_API_KEY" "cv-agent-api-key:$AGENT_API_KEY"; do
  nombre="${par%%:*}"; valor="${par#*:}"
  gcloud secrets describe "$nombre" >/dev/null 2>&1 || gcloud secrets create "$nombre" --replication-policy=automatic
  printf '%s' "$valor" | gcloud secrets versions add "$nombre" --data-file=-
done

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --min-instances 1 \
  --max-instances 5 \
  --cpu 1 --memory 512Mi \
  --concurrency 20 \
  --timeout 120 \
  --set-env-vars "LLM_PROVIDER=openai,LLM_MODEL=${LLM_MODEL},LOG_LEVEL=INFO" \
  --set-secrets "OPENAI_API_KEY=cv-agent-llm-key:latest,AGENT_API_KEY=cv-agent-api-key:latest"

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
# La tarjeta de agente necesita conocer su propia URL publica
gcloud run services update "$SERVICE" --region "$REGION" \
  --update-env-vars "PUBLIC_BASE_URL=${URL}/v1" >/dev/null

echo
echo "Desplegado."
echo "  URL base para la plataforma:  ${URL}/v1"
echo "  Clave de API:                 ${AGENT_API_KEY}"
echo "  Tarjeta de agente:            ${URL}/.well-known/agent-card.json"
echo
./scripts/smoke_test.sh "${URL}/v1" "${AGENT_API_KEY}"
