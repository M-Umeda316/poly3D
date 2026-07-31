# PolyOmics MAIN build = full 22-class dataset build + split (ASCII only, PS 5.1).
# Stage 0 of the main run. CPU only (no GPU). Detached + idempotent via status flags.
#   1. build_polyomics_dataset.py: data/*.tar.gz (22 classes) -> data/polyomics_all.lmdb
#      (precompute_topology so train.py needs no per-batch BFS).
#   2. split_lmdb.py: polyomics_all.lmdb -> polyomics_all_train.lmdb / polyomics_all_val.lmdb
#
# IDEMPOTENT: each stage is skipped when its output lmdb already exists on disk, so a
# relaunch after an interrupt resumes at the first unfinished stage. tar.gz archives are
# expected to already be placed under data/ (HF yhayashi1986/PolyOmics MD_snapshot_JSON).
#
# LAUNCH (OS-detached):
#   $env:POLY3D_PY = "C:/Users/shanu/anaconda3/envs/polygen/python.exe"
#   Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','<repo>/scripts/run_main_build.ps1' -WindowStyle Hidden -PassThru
# -----------------------------------------------------------------------------

param(
    [int]$PerCellStride = 1,
    [int]$MaxAtoms = 288,
    [int]$MapSizeGb = 250
)

if ($env:POLY3D_PY) { $py = $env:POLY3D_PY } else { $py = "python" }
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:256"
Set-Location $repo

$out       = "runs/polyomics_main_build"
$status    = "$out/status.txt"
$dataDir   = "data/"
$allLmdb   = "data/polyomics_all.lmdb"
$trainLmdb = "data/polyomics_all_train.lmdb"
$valLmdb   = "data/polyomics_all_val.lmdb"

function Done($tag) {
    return (Test-Path $status) -and (Select-String -Path $status -Pattern $tag -Quiet)
}

if (-not (Test-Path $out)) { New-Item -ItemType Directory -Force -Path $out | Out-Null }
"START $(Get-Date -Format o) per_cell_stride=$PerCellStride max_atoms=$MaxAtoms map_size_gb=$MapSizeGb" |
    Out-File -Append -Encoding ascii $status

# ---- Stage 1: BUILD (data/*.tar.gz -> polyomics_all.lmdb) ---------------------
if (-not (Done "BUILD_DONE")) {
    if (Test-Path $allLmdb) {
        "BUILD_DONE (reuse existing $allLmdb) $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
    } else {
        "BUILD_START $(Get-Date -Format o) data_dir=$dataDir out=$allLmdb" | Out-File -Append -Encoding ascii $status
        & $py "scripts/build_polyomics_dataset.py" `
            --data_dir $dataDir --out_path $allLmdb `
            --precompute_topology --per_cell_stride $PerCellStride `
            --max_atoms $MaxAtoms --map_size_gb $MapSizeGb `
            1> "$out/build_out.log" 2> "$out/build_err.log"
        if ($LASTEXITCODE -ne 0) {
            "BUILD_FAILED exit=$LASTEXITCODE $(Get-Date -Format o) - relaunch to resume" |
                Out-File -Append -Encoding ascii $status
            exit $LASTEXITCODE
        }
        "BUILD_DONE exit=0 $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
    }
}

# ---- Stage 2: SPLIT (polyomics_all.lmdb -> train/val) ------------------------
if (-not (Done "SPLIT_DONE")) {
    if ((Test-Path $trainLmdb) -and (Test-Path $valLmdb)) {
        "SPLIT_DONE (reuse existing $trainLmdb / $valLmdb) $(Get-Date -Format o)" |
            Out-File -Append -Encoding ascii $status
    } else {
        "SPLIT_START $(Get-Date -Format o) src=$allLmdb" | Out-File -Append -Encoding ascii $status
        & $py "scripts/split_lmdb.py" `
            --src $allLmdb --train_out $trainLmdb --val_out $valLmdb `
            1> "$out/split_out.log" 2> "$out/split_err.log"
        if ($LASTEXITCODE -ne 0) {
            "SPLIT_FAILED exit=$LASTEXITCODE $(Get-Date -Format o) - relaunch to resume" |
                Out-File -Append -Encoding ascii $status
            exit $LASTEXITCODE
        }
        "SPLIT_DONE exit=0 $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
    }
}

"ALL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
Write-Host ""
Write-Host "Build+split done: $trainLmdb / $valLmdb. Next: run_main_vae.ps1"
