# PolyOmics MAIN DiT = Stage 2 latent flow-matching on the main VAE (ASCII, PS 5.1).
# Built on runs/polyomics_main_vae/vae_best.pt over the full 22-class data. Four stages,
# all detached + idempotent via status flags:
#   1. precompute train latents (main VAE encodes all_train -> cond/e_cond/z0/dist cache)
#   2. precompute val latents
#   3. DiT flow-matching train -> runs/polyomics_main_dit
#   4. eval --mode dit (SMILES -> N(0,I) -> ODE -> latent -> main-VAE decode -> 3D)
#
# COND DIM: the main VAE has cond_dim = hidden_dim = 256, and build_dit uses
# cond_dim = args.hidden_dim, so the DiT CLI MUST pass --hidden_dim 256. The VAE arch
# is rebuilt from the checkpoint's own stored args, so no VAE arch flags are needed here.
#
# LAUNCH (OS-detached):
#   $env:POLY3D_PY = "C:/Users/shanu/anaconda3/envs/polygen/python.exe"
#   Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','<repo>/scripts/run_main_dit.ps1' -WindowStyle Hidden -PassThru
#   Foreground live: ... -File scripts/run_main_dit.ps1 -Live
# Idempotent: each stage guarded by a status flag; DiT train resumes from latest ckpt.
# -----------------------------------------------------------------------------

param(
    [switch]$Live,
    [int]$BatchSize = 256,
    [int]$Epochs = 200,
    [double]$Lr = 3e-4,
    [string]$VaeRun = "polyomics_main_vae"
)

if ($env:POLY3D_PY) { $py = $env:POLY3D_PY } else { $py = "python" }
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$env:PYTHONUTF8 = "1"
$prevOutEnc = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:256"
Set-Location $repo

$out       = "runs/polyomics_main_dit"
$status    = "$out/status.txt"
$vae       = "runs/$VaeRun/vae_best.pt"
$trainLmdb = "data/polyomics_all_train.lmdb"
$valLmdb   = "data/polyomics_all_val.lmdb"
$latTrain  = "data/polyomics_all_latents_train.lmdb"
$latVal    = "data/polyomics_all_latents_val.lmdb"

function Done($tag) {
    return (Test-Path $status) -and (Select-String -Path $status -Pattern $tag -Quiet)
}
function LatestCkpt($dir) {
    if (-not (Test-Path $dir)) { return $null }
    $f = Get-ChildItem -Path $dir -Filter "dit_epoch*.pt" -ErrorAction SilentlyContinue |
         Sort-Object Name | Select-Object -Last 1
    if ($f) { return $f.FullName } else { return $null }
}
function Fail($tag, $ec) {
    "$tag`_FAILED exit=$ec $(Get-Date -Format o) - relaunch to resume" |
        Out-File -Append -Encoding ascii $status
    [Console]::OutputEncoding = $prevOutEnc
    exit $ec
}

if (-not (Test-Path $out)) { New-Item -ItemType Directory -Force -Path $out | Out-Null }
"START $(Get-Date -Format o) vae=$vae batch=$BatchSize epochs=$Epochs lr=$Lr" |
    Out-File -Append -Encoding ascii $status

# ---- Stage 1: precompute train latents ---------------------------------------
if (-not (Done "PRECOMP_TRAIN_DONE")) {
    "PRECOMP_TRAIN_START $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
    & $py "scripts/precompute_latents.py" --vae_checkpoint $vae `
        --src_lmdb $trainLmdb --out_lmdb $latTrain `
        --batch_size 256 --num_workers 16 --max_atoms 288 --map_size_gb 40 `
        1> "$out/precomp_train.log" 2> "$out/precomp_train.err"
    if ($LASTEXITCODE -ne 0) { Fail "PRECOMP_TRAIN" $LASTEXITCODE }
    "PRECOMP_TRAIN_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

# ---- Stage 2: precompute val latents -----------------------------------------
if (-not (Done "PRECOMP_VAL_DONE")) {
    "PRECOMP_VAL_START $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
    & $py "scripts/precompute_latents.py" --vae_checkpoint $vae `
        --src_lmdb $valLmdb --out_lmdb $latVal `
        --batch_size 256 --num_workers 16 --max_atoms 288 --map_size_gb 6 `
        1> "$out/precomp_val.log" 2> "$out/precomp_val.err"
    if ($LASTEXITCODE -ne 0) { Fail "PRECOMP_VAL" $LASTEXITCODE }
    "PRECOMP_VAL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

# ---- Stage 3: DiT flow-matching train ----------------------------------------
if (-not (Done "TRAIN_DONE")) {
    $a = @(
        "--stage","dit",
        "--train_lmdb",$trainLmdb,"--val_lmdb",$valLmdb,
        "--vae_checkpoint",$vae,
        "--latent_lmdb",$latTrain,"--latent_val_lmdb",$latVal,
        "--hidden_dim","256","--latent_dim","16",
        "--dit_hidden_dim","256","--dit_n_heads","8","--dit_n_layers","6",
        "--time_dim","64","--t_max","0.9","--p_selfcond","0.5",
        "--batch_size",[string]$BatchSize,"--epochs",[string]$Epochs,
        "--lr",[string]$Lr,"--lr_min","3e-5",
        "--weight_decay","1e-5","--grad_clip","1.0",
        "--warmup_steps","200","--warmup_start_factor","0.01",
        "--gnorm_log_every","50","--empty_cache_every","500",
        "--max_atoms","288","--num_workers","16",
        "--save_every","1","--seed","42",
        "--out_dir",$out
    )
    $ck = LatestCkpt $out
    if ($ck) {
        $a += @("--resume",$ck)
        "TRAIN_START $(Get-Date -Format o) resume=$ck" | Out-File -Append -Encoding ascii $status
    } else {
        "TRAIN_START $(Get-Date -Format o) fresh" | Out-File -Append -Encoding ascii $status
    }
    if ($Live) {
        & $py "scripts/train.py" @a 1> "$out/dit_out.log"
    } else {
        & $py "scripts/train.py" @a 1> "$out/dit_out.log" 2> "$out/dit_err.log"
    }
    if ($LASTEXITCODE -ne 0) { Fail "TRAIN" $LASTEXITCODE }
    "TRAIN_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

# ---- Stage 4: eval --mode dit (true generation) ------------------------------
if (-not (Done "EVAL_DONE")) {
    $ditCk = "$out/dit_best.pt"
    "EVAL_START $(Get-Date -Format o) vae=$vae dit=$ditCk" | Out-File -Append -Encoding ascii $status
    & $py "scripts/eval_ensemble.py" --checkpoint $vae --dit_checkpoint $ditCk `
        --val_lmdb $valLmdb --mode dit --max_atoms 288 --batch_size 32 --n_steps 100 `
        --out "$out/eval_dit.json" 1> "$out/eval_out.log" 2> "$out/eval_err.log"
    if ($LASTEXITCODE -ne 0) { Fail "EVAL" $LASTEXITCODE }
    "EVAL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

"ALL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
Write-Host ""
Write-Host "Report: $out/eval_dit.json, $out/dit_log.csv. Next: run_main_ditcons.ps1"

[Console]::OutputEncoding = $prevOutEnc
