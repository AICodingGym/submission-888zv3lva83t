param(
    [string]$HostName = "connect.nma1.seetacloud.com",
    [int]$Port = 42103,
    [string]$RemoteRoot = "/root/autodl-tmp/essay-scoring",
    [int]$TimeoutMinutes = 720
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$keyPath = Join-Path $env:USERPROFILE ".ssh\id_rsa"
$sshBase = @("-o", "BatchMode=yes", "-i", $keyPath, "-p", $Port, "root@$HostName")
$started = Get-Date
$statusPath = Join-Path $projectRoot "outputs\gpu_fetch_status.txt"

while (((Get-Date) - $started).TotalMinutes -lt $TimeoutMinutes) {
    $ready = & ssh @sshBase "test -f $RemoteRoot/outputs/gpu_ensemble/metrics.json && test -f $RemoteRoot/submission_gpu_ensemble.csv" 2>$null
    if ($LASTEXITCODE -eq 0) {
        New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot "outputs\deberta_v3_large") | Out-Null
        New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot "outputs\gpu_ensemble") | Out-Null
        & scp -o BatchMode=yes -i $keyPath -P $Port "root@${HostName}:$RemoteRoot/submission_gpu_ensemble.csv" (Join-Path $projectRoot "submission_gpu_ensemble.csv")
        & scp -o BatchMode=yes -i $keyPath -P $Port "root@${HostName}:$RemoteRoot/outputs/deberta_v3_large/metrics.json" (Join-Path $projectRoot "outputs\deberta_v3_large\metrics.json")
        & scp -o BatchMode=yes -i $keyPath -P $Port "root@${HostName}:$RemoteRoot/outputs/deberta_v3_large/oof_predictions.csv" (Join-Path $projectRoot "outputs\deberta_v3_large\oof_predictions.csv")
        & scp -o BatchMode=yes -i $keyPath -P $Port "root@${HostName}:$RemoteRoot/outputs/deberta_v3_large/test_predictions.csv" (Join-Path $projectRoot "outputs\deberta_v3_large\test_predictions.csv")
        & scp -o BatchMode=yes -i $keyPath -P $Port "root@${HostName}:$RemoteRoot/outputs/gpu_ensemble/metrics.json" (Join-Path $projectRoot "outputs\gpu_ensemble\metrics.json")
        "Completed and downloaded at $(Get-Date -Format o)" | Set-Content -Encoding utf8 $statusPath
        exit 0
    }
    "Waiting for remote GPU training at $(Get-Date -Format o)" | Set-Content -Encoding utf8 $statusPath
    Start-Sleep -Seconds 60
}

"Timed out at $(Get-Date -Format o) without a completed remote GPU result." | Set-Content -Encoding utf8 $statusPath
exit 1
