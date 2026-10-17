#!/usr/bin/env bash
# Removes the Fly apps (and their volume) and, if applied, the Terraform-managed AWS resources.
set -uo pipefail
cd "$(dirname "$0")/.."
for app in "${API_APP:-dfp-api}" "${MLFLOW_APP:-dfp-mlflow}"; do
  if command -v fly >/dev/null; then
    fly apps destroy "$app" --yes 2>/dev/null && echo "destroyed fly app $app" || echo "fly app $app not found"
  fi
done
if [ -f infra/terraform/terraform.tfstate ]; then
  (cd infra/terraform && terraform destroy -auto-approve) && echo "terraform resources destroyed"
else
  echo "no terraform state; nothing to destroy"
fi
