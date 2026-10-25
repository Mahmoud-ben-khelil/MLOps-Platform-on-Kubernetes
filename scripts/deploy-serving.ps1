# deploy-serving.ps1 - build the Titanic serving image and deploy it to Minikube.
# Run from anywhere after setup.ps1 (Minikube must be running).
#
# Steps: train serving/model.pkl if missing -> docker build -> minikube image load
#        -> create namespace "serving" -> kubectl apply -> port-forward on :8001

$ErrorActionPreference = "Stop"

$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")

# Native commands don't trigger $ErrorActionPreference, so check exit codes.
function Invoke-Step([string]$What, [scriptblock]$Cmd) {
    & $Cmd
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Image = "titanic-serving:latest"

if (-not (Test-Path "serving/model.pkl")) {
    Write-Host "==> serving/model.pkl not found - training it with serving/train_model.py..."
    Invoke-Step "Training" { python serving/train_model.py }
} else {
    Write-Host "==> Using existing serving/model.pkl (delete it to retrain)"
}

Write-Host "`n==> Building image $Image..."
Invoke-Step "docker build" { docker build -t $Image serving/ }

Write-Host "`n==> Loading image into Minikube..."
Invoke-Step "minikube image load" { minikube image load $Image }

Write-Host "`n==> Creating namespace 'serving' (if missing)..."
kubectl get namespace serving *> $null
if ($LASTEXITCODE -ne 0) {
    Invoke-Step "kubectl create namespace" { kubectl create namespace serving }
}

Write-Host "`n==> Applying serving/k8s manifests..."
Invoke-Step "kubectl apply" { kubectl apply -f serving/k8s/ }

# The tag stays "latest" with imagePullPolicy: Never, so restart to pick up a rebuilt image.
Write-Host "`n==> Restarting deployment and waiting for it to become ready..."
Invoke-Step "kubectl rollout restart" { kubectl rollout restart deployment/titanic-serving -n serving }
Invoke-Step "kubectl rollout status" { kubectl rollout status deployment/titanic-serving -n serving --timeout=180s }

Write-Host "`n==> Forwarding serving API -> http://localhost:8001"
Write-Host "    (press Ctrl-C to stop)"
kubectl port-forward -n serving svc/titanic-serving 8001:80
