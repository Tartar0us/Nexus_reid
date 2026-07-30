param(
    [Parameter(Mandatory=$true)]
    [string]$DataRoot,

    [Parameter(Mandatory=$true)]
    [string]$MeanCheckpoint,

    [Parameter(Mandatory=$true)]
    [string]$SemanticCheckpoint,

    [Parameter(Mandatory=$true)]
    [string]$QualityCheckpoint,

    [string]$OutputDir = "quality_aware_reid\models\ablation_metrics",
    [int]$SeqLen = 8,
    [int]$NumClasses = 625,
    [int]$BatchSize = 32,
    [string]$Device = "cuda",
    [double]$Severity = 0.5
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Push-Location $ProjectRoot

try {
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$models = @(
    @{ Name = "mean_pooling"; Checkpoint = $MeanCheckpoint },
    @{ Name = "semantic_attention"; Checkpoint = $SemanticCheckpoint },
    @{ Name = "quality_aware"; Checkpoint = $QualityCheckpoint }
)

foreach ($model in $models) {
    $name = $model.Name
    $checkpoint = $model.Checkpoint

    python quality_aware_reid\eval_mars_official.py `
        --data-root $DataRoot `
        --checkpoint $checkpoint `
        --model-type $name `
        --num-classes $NumClasses `
        --seq-len $SeqLen `
        --batch-size $BatchSize `
        --device $Device `
        --output-json (Join-Path $OutputDir "clean_$name.json")

    python quality_aware_reid\eval_mars_degraded.py `
        --data-root $DataRoot `
        --checkpoint $checkpoint `
        --model-type $name `
        --degradation mixed `
        --severity $Severity `
        --num-classes $NumClasses `
        --seq-len $SeqLen `
        --batch-size $BatchSize `
        --device $Device `
        --output-json (Join-Path $OutputDir "mixed_${Severity}_$name.json")
}

python quality_aware_reid\summarize_metrics.py `
    (Join-Path $OutputDir "*.json") `
    --csv (Join-Path $OutputDir "summary.csv") `
    --markdown (Join-Path $OutputDir "summary.md")
}
finally {
    Pop-Location
}
