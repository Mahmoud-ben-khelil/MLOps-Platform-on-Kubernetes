# port-forward.ps1 - re-expose the KFP UI after a minikube start.
# Run this every time you restart Minikube without running setup.ps1.
#
# Port note: port 8080 is occupied by Oracle TNS Listener (TNSLSNR.exe),
# so we use 8082 for the KFP UI instead.

$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")

Write-Host "==> Forwarding KFP UI -> http://localhost:8082"
Write-Host "    (press Ctrl-C to stop)"
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8082:80
