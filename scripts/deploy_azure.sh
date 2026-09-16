#!/usr/bin/env bash
# Despliegue a Azure Container Apps desde codigo fuente.
# Corre en Azure Cloud Shell o en Codespaces. No requiere Docker local.
#
#   export AZURE_OPENAI_ENDPOINT="https://<recurso>.openai.azure.com/openai/v1"
#   export AZURE_OPENAI_API_KEY="..."
#   export AZURE_OPENAI_DEPLOYMENT="gpt-5-mini"
#   export AGENT_API_KEY="$(openssl rand -hex 24)"
#   ./scripts/deploy_azure.sh
#
# El endpoint es el que da el portal de Foundry, con el sufijo /openai/v1 (se
# acepta sin el sufijo tambien). La v1 GA no pide api-version: solo si el
# recurso llegara a responder 400 pidiendola, se define
# AZURE_OPENAI_API_VERSION=preview.
set -euo pipefail

RG="${RG:-rg-cv-agent}"
LOCATION="${LOCATION:-eastus}"
APP="${APP:-cv-agent}"
ENVNAME="${ENVNAME:-env-cv-agent}"

: "${AZURE_OPENAI_ENDPOINT:?define AZURE_OPENAI_ENDPOINT}"
: "${AZURE_OPENAI_API_KEY:?define AZURE_OPENAI_API_KEY}"
: "${AZURE_OPENAI_DEPLOYMENT:?define AZURE_OPENAI_DEPLOYMENT}"
: "${AGENT_API_KEY:?define AGENT_API_KEY}"

echo "==> Registrando proveedores y extension"
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az extension add --name containerapp --upgrade --allow-preview true -y

az group create -n "$RG" -l "$LOCATION" -o none

echo "==> Desplegando (la primera vez tarda ~5 min: crea el entorno y compila la imagen)"
az containerapp up \
  --name "$APP" \
  --resource-group "$RG" \
  --location "$LOCATION" \
  --environment "$ENVNAME" \
  --source . \
  --ingress external \
  --target-port 8080

echo "==> Configurando secretos, variables y escalado"
az containerapp secret set -n "$APP" -g "$RG" --secrets \
  "aoai-key=${AZURE_OPENAI_API_KEY}" \
  "agent-key=${AGENT_API_KEY}" -o none

FQDN="$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"

az containerapp update -n "$APP" -g "$RG" \
  --min-replicas 1 --max-replicas 3 \
  --set-env-vars \
    "LLM_PROVIDER=azure" \
    "AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}" \
    "AZURE_OPENAI_DEPLOYMENT=${AZURE_OPENAI_DEPLOYMENT}" \
    "AZURE_OPENAI_API_KEY=secretref:aoai-key" \
    "AGENT_API_KEY=secretref:agent-key" \
    "PUBLIC_BASE_URL=https://${FQDN}/v1" \
    "LOG_LEVEL=INFO" -o none

echo
echo "===================================================================="
echo " URL base para la plataforma:  https://${FQDN}/v1"
echo " Clave de API:                 ${AGENT_API_KEY}"
echo " Tarjeta de agente:            https://${FQDN}/.well-known/agent-card.json"
echo "===================================================================="
echo
sleep 20
./scripts/smoke_test.sh "https://${FQDN}/v1" "${AGENT_API_KEY}" || \
  echo "Si fallo, espera 30s mas y reintenta: ./scripts/smoke_test.sh https://${FQDN}/v1 ${AGENT_API_KEY}"
