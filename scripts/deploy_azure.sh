#!/usr/bin/env bash
# Despliegue a Azure Container Apps desde la imagen publicada en ghcr.io.
#
#   export AZURE_OPENAI_ENDPOINT="https://<recurso>.openai.azure.com/openai/v1"
#   export AZURE_OPENAI_API_KEY="..."
#   export AZURE_OPENAI_DEPLOYMENT="gpt-5-mini"
#   export AGENT_API_KEY="$(openssl rand -hex 24)"
#   ./scripts/deploy_azure.sh
#
# La imagen la compila y publica GitHub Actions (.github/workflows/imagen.yml)
# en ghcr.io/alejandroald/cv-agent. Este script NO compila.
#
# Antes usaba `az containerapp up --source .`, que compila dentro de Azure y
# para eso crea un Azure Container Registry: ~5 USD/mes de costo fijo sólo por
# hospedar la imagen de un repo publico. ghcr.io es gratis para repos publicos,
# asi que el paquete se deja publico y el pull no necesita credenciales.
#
# El endpoint es el que da el portal de Foundry, con el sufijo /openai/v1 (se
# acepta sin el sufijo tambien). La v1 GA no pide api-version: solo si el
# recurso llegara a responder 400 pidiendola, se define
# AZURE_OPENAI_API_VERSION=preview.
set -euo pipefail

RG="${RG:-rg-cv-agent}"
LOCATION="${LOCATION:-eastus2}"
APP="${APP:-cv-agent}"
ENVNAME="${ENVNAME:-env-cv-agent}"
IMAGE="${IMAGE:-ghcr.io/alejandroald/cv-agent:latest}"

: "${AZURE_OPENAI_ENDPOINT:?define AZURE_OPENAI_ENDPOINT}"
: "${AZURE_OPENAI_API_KEY:?define AZURE_OPENAI_API_KEY}"
: "${AZURE_OPENAI_DEPLOYMENT:?define AZURE_OPENAI_DEPLOYMENT}"
: "${AGENT_API_KEY:?define AGENT_API_KEY}"

# Microsoft.ContainerRegistry ya no se registra: nada compila ni hospeda
# imagenes en Azure. Si alguna vez vuelve `--source .`, hay que registrarlo.
echo "==> Registrando proveedores y extension"
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az extension add --name containerapp --upgrade --allow-preview true -y

az group create -n "$RG" -l "$LOCATION" -o none

# Crear o actualizar. `az containerapp up` con --image no crea registro.
if az containerapp show -n "$APP" -g "$RG" -o none 2>/dev/null; then
  echo "==> Actualizando la imagen de $APP"
  az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" -o none
else
  echo "==> Creando $APP (la primera vez tarda ~5 min: crea el entorno)"
  az containerapp up \
    --name "$APP" \
    --resource-group "$RG" \
    --location "$LOCATION" \
    --environment "$ENVNAME" \
    --image "$IMAGE" \
    --ingress external \
    --target-port 8080
fi

echo "==> Configurando secretos, variables y escalado"
az containerapp secret set -n "$APP" -g "$RG" --secrets \
  "aoai-key=${AZURE_OPENAI_API_KEY}" \
  "agent-key=${AGENT_API_KEY}" -o none

FQDN="$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"

az containerapp update -n "$APP" -g "$RG" \
  --min-replicas 0 --max-replicas 3 \
  --cpu 0.25 --memory 0.5Gi \
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
echo " Imagen desplegada:            ${IMAGE}"
echo " URL base para la plataforma:  https://${FQDN}/v1"
echo " Clave de API:                 ${AGENT_API_KEY}"
echo " Tarjeta de agente:            https://${FQDN}/.well-known/agent-card.json"
echo "===================================================================="
echo
# min-replicas 0 significa que la primera peticion paga arranque en frio.
sleep 20
./scripts/smoke_test.sh "https://${FQDN}/v1" "${AGENT_API_KEY}" || \
  echo "Si fallo, espera 30s mas y reintenta: ./scripts/smoke_test.sh https://${FQDN}/v1 ${AGENT_API_KEY}"
