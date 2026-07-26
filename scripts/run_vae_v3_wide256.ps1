# PolyOmics VAE v3 = CAPACITY lever (ASCII only for PS 5.1).
# ONE variable vs the accepted v2 (runs/polyomics_vae_v2, ep23, beta_end 0.1):
#   width 128 -> 256  (hidden_dim + vae_hidden_dim 128->256, edge_dim 64->128).
# Everything else == v2: EGT enc+dec 2&2, enc/dec 4, latent 16, kabsch,
#   beta 0->0.1 over 20ep, lr 3e-4 (warmup 200 optstep) lr_min 3e-5, 50ep,
#   val_subset 0.3, empty_cache_every 500, seed 42, max_atoms 288.
#
# WHY -------------------------------------------------------------------------
# v2 fixed posterior collapse (beta_end 0.1): latent works, DiT v1 then succeeded
# at the DISTRIBUTION level (torsion-JS 0.285 ~= recon 0.276 << prior 0.400,
# COV-R 1.0). The ONE remaining bottleneck is VAE DECODER geometric validity on
# 11+ heavy-atom units: validity pass-rate median only 0.16-0.28 (<=10 atoms is
# 0.86). That is decoder capacity/training, not latent and not DiT (DiT inherits
# the same decoder). Width 256 is the cheapest lever to test.
#
# VRAM (RTX 4060 Ti, 16GB usable) ---------------------------------------------
# v2 was hidden128 / bs128 and peaked reserved ~11GB. Doubling width ~2x the
# activations -> bs128 would risk spill+freeze (Windows has no clean OOM). So we
# keep EFFECTIVE batch 128 but split it bs64 x grad_accum2, halving per-step
# activation memory to offset the width doubling. warmup_steps is in OPTIMIZER
# steps, and the effective batch is unchanged, so the warmup danger-zone is
# identical to v2 -- the comparison stays clean. --empty_cache_every 500 +
# PYTORCH_CUDA_ALLOC_CONF handle fragmentation. If it still OOMs, drop to
# bs32 x ga4 (effective 128 unchanged) via -BatchSize 32 -GradAccum 4.
#
# WHAT TO READ AFTER ----------------------------------------------------------
# Auto-eval writes runs/polyomics_vae_v3/{eval_recon.json, eval_prior.json}.
# The metric that must MOVE vs v2 is the VALIDITY pass-rate on 11+ atom bands.
#   v2 baseline (recon): torsion-JS 0.276 < prior 0.400, validity ~0.28.
#   Success for v3 = recon validity pass-rate up on the 11-15 / 16+ bands while
#   recon still beats prior on torsion-JS (latent stays informative). If validity
#   climbs -> precompute_latents(v3) -> retrain DiT -> eval --mode dit for a full
#   loop. If flat -> width is not the lever; escalate to all-22-class data or a
#   validity loss term (bond/angle weight up + clash penalty).
#
# LAUNCH ----------------------------------------------------------------------
#   Set env POLY3D_PY to the polygen python if `python` is not it:
#     $env:POLY3D_PY = "C:/Users/shanu/anaconda3/envs/polygen/python.exe"
#   Foreground, watch tqdm live:
#     powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_vae_v3_wide256.ps1 -Live
#   OS-detached (harness idle-kill resistant, no console):
#     Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#       '-File','<repo>/scripts/run_vae_v3_wide256.ps1' -WindowStyle Hidden -PassThru
#   Idempotent: re-launch resumes from the newest vae_epoch*.pt (save_every 1).
#   ~21 min/epoch at hidden128 (dataloader-bound); width256 somewhat slower.
#   Do NOT run other GPU jobs alongside it (past OOM/freeze lesson).
# -----------------------------------------------------------------------------

param(
    [switch]$Live,                       # -Live: show tqdm bars (stderr unredirected)
    [int]$Width = 256,                   # hidden_dim + vae_hidden_dim. edge_dim = Width/2.
    [int]$BatchSize = 64,                # bs64 x ga2 = effective 128 (== v2 effective batch).
    [int]$GradAccum = 2,                 # drop to bs32 x ga4 if width256 still OOMs.
    [double]$Lr = 3e-4,                  # v3 used 3e-4 (bad basin). v3b: 1.5e-4.
    [int]$WarmupSteps = 200,             # optimizer-step LR warmup. v3b: 500 (longer).
    [string]$OutName = "polyomics_vae_v3"  # v3b lives in a FRESH dir (no bad-basin resume).
)

if ($env:POLY3D_PY) { $py = $env:POLY3D_PY } else { $py = "python" }
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$env:PYTHONUTF8 = "1"
$prevOutEnc = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# Windows has no expandable_segments; mitigate fragmentation at width256.
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:256"
Set-Location $repo

$edge    = [int]($Width / 2)                 # v2 used edge 64 for width 128.
$out     = "runs/$OutName"
$status  = "$out/status.txt"
$train   = "data/polyomics_PG_train.lmdb"
$val     = "data/polyomics_PG_val.lmdb"

function Done($tag) {
    return (Test-Path $status) -and (Select-String -Path $status -Pattern $tag -Quiet)
}
function LatestCkpt($dir) {
    if (-not (Test-Path $dir)) { return $null }
    $f = Get-ChildItem -Path $dir -Filter "vae_epoch*.pt" -ErrorAction SilentlyContinue |
         Sort-Object Name | Select-Object -Last 1
    if ($f) { return $f.FullName } else { return $null }
}

if (-not (Test-Path $out)) { New-Item -ItemType Directory -Force -Path $out | Out-Null }
"START $(Get-Date -Format o) width=$Width bs=$BatchSize ga=$GradAccum lr=$Lr" |
    Out-File -Append -Encoding ascii $status

# ---- Train: v2 config + width256 --------------------------------------------
if (-not (Done "TRAIN_DONE")) {
    $a = @(
        "--stage","vae",
        "--train_lmdb",$train,
        "--val_lmdb",$val,
        "--out_dir",$out,
        "--hidden_dim",[string]$Width,"--edge_dim",[string]$edge,"--vae_hidden_dim",[string]$Width,
        "--cond_layers","4","--latent_dim","16",
        "--enc_layers","4","--dec_layers","4",
        "--egt_every","2","--enc_egt_every","2",
        "--beta_start","0.0","--beta_end","0.1","--beta_warmup_epochs","20",
        "--w_pos","1.0","--w_bond","1.0","--w_angle","0.5","--w_dihedral","0.1",
        "--pos_loss_type","kabsch",
        "--batch_size",[string]$BatchSize,"--grad_accum",[string]$GradAccum,
        "--epochs","50",
        "--lr",[string]$Lr,"--lr_min","3e-5",
        "--weight_decay","1e-5","--grad_clip","1.0",
        "--warmup_steps",[string]$WarmupSteps,"--warmup_start_factor","0.01",
        "--gnorm_log_every","50",
        "--max_atoms","288",
        "--val_subset_ratio","0.3",
        "--empty_cache_every","500",
        "--num_workers","8","--prefetch_factor","4",
        "--save_every","1","--seed","42"
    )
    $ck = LatestCkpt $out
    if ($ck) { $a += @("--resume",$ck) }
    "TRAIN_START $(Get-Date -Format o) resume=$ck" | Out-File -Append -Encoding ascii $status
    if ($Live) {
        & $py "scripts/train.py" @a 1> "$out/train_out.log"
    } else {
        & $py "scripts/train.py" @a 1> "$out/train_out.log" 2> "$out/train_err.log"
    }
    $ec = $LASTEXITCODE
    # OOM fail-fast raises (exit != 0). Do NOT mark DONE so relaunch resumes.
    if ($ec -ne 0) {
        "TRAIN_FAILED exit=$ec $(Get-Date -Format o) - relaunch to resume from latest ckpt" |
            Out-File -Append -Encoding ascii $status
        exit $ec
    }
    "TRAIN_DONE exit=0 $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

# ---- Auto-eval: recon + prior on PG_val (validity is the thing to compare) ---
if (-not (Done "EVAL_DONE")) {
    $ckpt = "$out/vae_best.pt"
    $bestEp = & $py -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" $ckpt 2>$null
    "EVAL_START $(Get-Date -Format o) ckpt=$ckpt best_epoch=$bestEp" |
        Out-File -Append -Encoding ascii $status

    & $py "scripts/eval_ensemble.py" --checkpoint $ckpt --val_lmdb $val `
        --mode recon --max_atoms 288 --batch_size 32 `
        --out "$out/eval_recon.json" 1> "$out/eval_out.log" 2> "$out/eval_err.log"

    & $py "scripts/eval_ensemble.py" --checkpoint $ckpt --val_lmdb $val `
        --mode prior --max_atoms 288 --batch_size 32 `
        --out "$out/eval_prior.json" 1>> "$out/eval_out.log" 2>> "$out/eval_err.log"

    "EVAL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

"ALL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
Write-Host ""
Write-Host "Report: $out/vae_log.csv (per-epoch), $out/eval_recon.json + eval_prior.json"
Write-Host "Compare recon VALIDITY pass-rate on 11+ atom bands vs v2 (was ~0.28)."

[Console]::OutputEncoding = $prevOutEnc
