#!/usr/bin/env bash
# setup.sh — build the KFP standalone cluster from scratch.
# Run once on a cold machine; takes ~5 min for images to pull.
#
# Port note: port 8080 is occupied by Oracle TNS Listener (TNSLSNR.exe),
# so we use 8082 for the KFP UI port-forward instead.

set -euo pipefail

PIPELINE_VERSION="2.16.1"

echo "==> Starting Minikube (4 CPU / 8 GB / 40 GB)..."
minikube start --cpus=4 --memory=8192 --disk-size=40g --driver=docker

echo ""
echo "==> Applying KFP ${PIPELINE_VERSION} cluster-scoped resources (CRDs, ClusterRoles)..."
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${PIPELINE_VERSION}"

echo ""
echo "==> Waiting for Application CRD to establish..."
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

echo ""
echo "==> Applying KFP ${PIPELINE_VERSION} platform-agnostic workloads..."
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${PIPELINE_VERSION}"

echo ""
echo "==> Waiting for all KFP pods to be Running (timeout 10 min)..."
kubectl wait --for=condition=available --timeout=600s \
  deployment/cache-deployer-deployment \
  deployment/cache-server \
  deployment/metadata-envoy-deployment \
  deployment/metadata-grpc-deployment \
  deployment/metadata-writer \
  deployment/ml-pipeline \
  deployment/ml-pipeline-persistenceagent \
  deployment/ml-pipeline-scheduledworkflow \
  deployment/ml-pipeline-ui \
  deployment/ml-pipeline-viewer-crd \
  deployment/ml-pipeline-visualizationserver \
  deployment/mysql \
  deployment/seaweedfs \
  deployment/workflow-controller \
  -n kubeflow

echo ""
kubectl get pods -n kubeflow

echo ""
echo "==> Starting port-forward on http://localhost:8082 ..."
echo "    (press Ctrl-C to stop)"
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8082:80
