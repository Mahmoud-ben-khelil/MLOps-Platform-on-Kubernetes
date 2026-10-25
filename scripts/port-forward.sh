#!/usr/bin/env bash
# port-forward.sh — re-expose the KFP UI after a minikube start.
# Run this every time you restart Minikube without running setup.sh.
#
# Port note: port 8080 is occupied by Oracle TNS Listener (TNSLSNR.exe),
# so we use 8082 for the KFP UI instead.

set -euo pipefail

echo "==> Forwarding KFP UI -> http://localhost:8082"
echo "    (press Ctrl-C to stop)"
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8082:80
