# setup.ps1 - build the KFP standalone cluster from scratch.
# Run once on a cold machine; takes ~5 min for images to pull.
#
# Port note: port 8080 is occupied by Oracle TNS Listener (TNSLSNR.exe),
# so we use 8082 for the KFP UI port-forward instead.

$ErrorActionPreference = "Stop"

$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")
$env:PIPELINE_VERSION = "2.16.1"

Write-Host "==> Starting Minikube (4 CPU / 8 GB / 40 GB)..."
minikube start --cpus=4 --memory=8192 --disk-size=40g --driver=docker

Write-Host "`n==> Applying KFP $env:PIPELINE_VERSION cluster-scoped resources (CRDs, ClusterRoles)..."
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$env:PIPELINE_VERSION"

Write-Host "`n==> Waiting for Application CRD to establish..."
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

Write-Host "`n==> Applying KFP $env:PIPELINE_VERSION platform-agnostic workloads..."
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$env:PIPELINE_VERSION"

Write-Host "`n==> Waiting for all KFP deployments to become available (timeout 10 min)..."
$deployments = @(
    "cache-deployer-deployment",
    "cache-server",
    "metadata-envoy-deployment",
    "metadata-grpc-deployment",
    "metadata-writer",
    "ml-pipeline",
    "ml-pipeline-persistenceagent",
    "ml-pipeline-scheduledworkflow",
    "ml-pipeline-ui",
    "ml-pipeline-viewer-crd",
    "ml-pipeline-visualizationserver",
    "mysql",
    "seaweedfs",
    "workflow-controller"
)
kubectl wait --for=condition=available --timeout=600s ($deployments | ForEach-Object { "deployment/$_" }) -n kubeflow

Write-Host ""
kubectl get pods -n kubeflow

Write-Host "`n==> Starting port-forward on http://localhost:8082 ..."
Write-Host "    (press Ctrl-C to stop)"
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8082:80
