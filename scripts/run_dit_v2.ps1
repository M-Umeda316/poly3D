# PolyOmics DiT v2 = end-to-end generative pipeline on the v3c decoder (ASCII, PS 5.1).
# Stage 2 of the pipeline, rebuilt on the clash-fine-tuned VAE v3c (recon validity
# 0.70 vs v2's 0.28). Three stages, all detached + idempotent via status flags:
#   1. precompute latents (v3c VAE encodes PG train/val -> cond/e_cond/z0/dist_mat cache)
#   2. DiT flow-matching train on those latents  -> runs/polyomics_dit_v2
#   3. eval --mode dit (SMILES -> N(0,I) -> ODE -> latent -> v3c decode -> 3D)
#
# WHY v3c latents are NEW (cannot reuse dit_v1's) --------------------------------
# dit_v1 trained on v2-VAE latents (hidden128). v3c is hidden256 + clash-polished, so
# its cond/z0 distribution is different -> latents must be recomputed. cond_dim=256 now,
# so the DiT CLI MUST pass --hidden_dim 256 (build_dit uses cond_dim=args.hidden_dim).
# The VAE itself is rebuilt from the checkpoint's OWN stored args (train.py:931), so no
# VAE arch flags are needed here -- only the DiT arch + cond_dim.
#
# THE QUESTION ------------------------------------------------------------------
# dit_v1 (on v2 decoder) generated at torsion-JS 0.285 but validity only 0.16 (it
# inherited v2's weak decoder). v3c's decoder is 0.70 valid. Does the generative
# pipeline now inherit that? SUCCESS = eval_dit validity_pass_rate >> 0.16, ideally
# approaching v3c recon's 0.70, while torsion-JS stays ~0.28 (<< prior).
#
# LAUNCH (OS-detached): set POLY3D_PY then
#   Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','<repo>/scripts/run_dit_v2.ps1' -WindowStyle Hidden -PassThru
# Idempotent: each stage guarded by a status flag; DiT train resumes from latest ckpt.
# Rough time: precompute ~1-2h (GPU), DiT ~8 min/ep x 150ep ~20h, eval ~0.5h.
# -----------------------------------------------------------------------------

param(
    [switch]$Live,
    [int]$Epochs = 150,
    [string]$Vae = "runs/polyomics_vae_v3c/vae_best.pt"
)

if ($env:POLY3D_PY) { $py = $env:POLY3D_PY } else { $py = "python" }
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$env:PYTHONUTF8 = "1"
$prevOutEnc = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:256"
Set-Location $repo

$out       = "runs/polyomics_dit_v2"
$status    = "$out/status.txt"
$trainLmdb = "data/polyomics_PG_train.lmdb"
$valLmdb   = "data/polyomics_PG_val.lmdb"
$latTrain  = "data/polyomics_PG_v3c_latents_train.lmdb"
$latVal    = "data/polyomics_PG_v3c_latents_val.lmdb"

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
"START $(Get-Date -Format o) vae=$Vae epochs=$Epochs" | Out-File -Append -Encoding ascii $status

# ---- Stage 1: precompute train latents ---------------------------------------
if (-not (Done "PRECOMP_TRAIN_DONE")) {
    "PRECOMP_TRAIN_START $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
    & $py "scripts/precompute_latents.py" --vae_checkpoint $Vae `
        --src_lmdb $trainLmdb --out_lmdb $latTrain `
        --batch_size 256 --num_workers 8 --max_atoms 288 --map_size_gb 24 `
        1> "$out/precomp_train.log" 2> "$out/precomp_train.err"
    if ($LASTEXITCODE -ne 0) { Fail "PRECOMP_TRAIN" $LASTEXITCODE }
    "PRECOMP_TRAIN_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

# ---- Stage 2: precompute val latents -----------------------------------------
if (-not (Done "PRECOMP_VAL_DONE")) {
    "PRECOMP_VAL_START $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
    & $py "scripts/precompute_latents.py" --vae_checkpoint $Vae `
        --src_lmdb $valLmdb --out_lmdb $latVal `
        --batch_size 256 --num_workers 8 --max_atoms 288 --map_size_gb 3 `
        1> "$out/precomp_val.log" 2> "$out/precomp_val.err"
    if ($LASTEXITCODE -ne 0) { Fail "PRECOMP_VAL" $LASTEXITCODE }
    "PRECOMP_VAL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

# ---- Stage 3: DiT flow-matching train ----------------------------------------
if (-not (Done "DIT_TRAIN_DONE")) {
    $a = @(
        "--stage","dit",
        "--train_lmdb",$trainLmdb,"--val_lmdb",$valLmdb,
        "--vae_checkpoint",$Vae,
        "--latent_lmdb",$latTrain,"--latent_val_lmdb",$latVal,
        "--hidden_dim","256","--latent_dim","16",   # cond_dim=hidden_dim=256 (v3c). REQUIRED.
        "--dit_hidden_dim","256","--dit_n_heads","8","--dit_n_layers","6",
        "--time_dim","64","--t_max","0.9","--p_selfcond","0.5",
        "--batch_size","256","--epochs",[string]$Epochs,
        "--lr","3e-4","--lr_min","3e-5",
        "--weight_decay","1e-5","--grad_clip","1.0",
        "--warmup_steps","200","--warmup_start_factor","0.01",
        "--gnorm_log_every","50","--empty_cache_every","500",
        "--max_atoms","288","--num_workers","8",
        "--save_every","1","--seed","42",
        "--out_dir",$out
    )
    $ck = LatestCkpt $out
    if ($ck) {
        $a += @("--resume",$ck)
        "DIT_TRAIN_START $(Get-Date -Format o) resume=$ck" | Out-File -Append -Encoding ascii $status
    } else {
        "DIT_TRAIN_START $(Get-Date -Format o) fresh" | Out-File -Append -Encoding ascii $status
    }
    if ($Live) {
        & $py "scripts/train.py" @a 1> "$out/dit_out.log"
    } else {
        & $py "scripts/train.py" @a 1> "$out/dit_out.log" 2> "$out/dit_err.log"
    }
    if ($LASTEXITCODE -ne 0) { Fail "DIT_TRAIN" $LASTEXITCODE }
    "DIT_TRAIN_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

# ---- Stage 4: eval --mode dit (true generation) ------------------------------
if (-not (Done "EVAL_DONE")) {
    $ditCk = "$out/dit_best.pt"
    "EVAL_START $(Get-Date -Format o) vae=$Vae dit=$ditCk" | Out-File -Append -Encoding ascii $status
    & $py "scripts/eval_ensemble.py" --checkpoint $Vae --dit_checkpoint $ditCk `
        --val_lmdb $valLmdb --mode dit --max_atoms 288 --batch_size 32 --n_steps 100 `
        --out "$out/eval_dit.json" 1> "$out/eval_out.log" 2> "$out/eval_err.log"
    if ($LASTEXITCODE -ne 0) { Fail "EVAL" $LASTEXITCODE }
    "EVAL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

"ALL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
Write-Host ""
Write-Host "Report: $out/eval_dit.json (validity_pass_rate vs dit_v1 0.16 / v3c recon 0.70), $out/dit_log.csv"

[Console]::OutputEncoding = $prevOutEnc
