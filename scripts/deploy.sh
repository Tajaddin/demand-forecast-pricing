#!/usr/bin/env bash
# Fly.io: an MLflow app (registry) and the API app. Needs flyctl logged in. Exports:
#   MLFLOW_USER MLFLOW_PASSWORD (basic auth for the MLflow app)  DFP_API_KEYS (keys for the API)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MLFLOW_USER:?set MLFLOW_USER}" "${MLFLOW_PASSWORD:?set MLFLOW_PASSWORD}" "${DFP_API_KEYS:?set DFP_API_KEYS}"
API_APP="${API_APP:-dfp-api}"
MLFLOW_APP="${MLFLOW_APP:-dfp-mlflow}"
ORG="${FLY_ORG:-personal}"
command -v fly >/dev/null || { echo "flyctl not found: https://fly.io/docs/flyctl/install/"; exit 1; }

for app in "$MLFLOW_APP" "$API_APP"; do
  fly apps list --json | grep -q "\"Name\": *\"$app\"" || fly apps create "$app" --org "$ORG"
done

fly volumes list -a "$MLFLOW_APP" --json | grep -q mlflow_data || fly volumes create mlflow_data -a "$MLFLOW_APP" --region mia --size 1 --yes
fly deploy -c fly.mlflow.toml -a "$MLFLOW_APP"
MLFLOW_URL="https://${MLFLOW_APP}.fly.dev"
echo "MLflow basic-auth users are managed with the mlflow auth CLI inside the app; see docs/DEPLOY.md"

# MLflow reads basic-auth credentials from these variables, so they never sit in the URI, which error messages quote.
fly secrets set --stage -a "$API_APP" "MLFLOW_TRACKING_URI=$MLFLOW_URL" "MLFLOW_TRACKING_USERNAME=$MLFLOW_USER" \
  "MLFLOW_TRACKING_PASSWORD=$MLFLOW_PASSWORD" "DFP_API_KEYS=$DFP_API_KEYS" >/dev/null
fly deploy -c fly.toml -a "$API_APP" --remote-only
echo "api:    https://${API_APP}.fly.dev/docs"
echo "mlflow: https://${MLFLOW_APP}.fly.dev"
echo "then, from this machine: MLFLOW_TRACKING_URI=$MLFLOW_URL MLFLOW_TRACKING_USERNAME=\$MLFLOW_USER MLFLOW_TRACKING_PASSWORD=\$MLFLOW_PASSWORD make backtest train"
