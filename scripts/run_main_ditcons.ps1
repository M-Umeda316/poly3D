# PolyOmics MAIN ditcons = Stage 2b decoder finishing (ASCII only, PS 5.1).
# Hardens the main VAE DECODER on the REAL main-DiT latents to close the gen->recon gap,
# on top of the multiscale recon decoder. Detached + idempotent via status flags.
#
# WHY --------------------------------------------------------------------------
# The multiscale decoder reaches a high recon ceiling, but generation decodes the DiT
# latents, which do not reach that ceiling by themselves. Re-applying the ditcons lever
# (train the decoder on the actual main-DiT latents, plus noise-robustness) pulls
# generation up toward the recon ceiling. The ENCODER is FROZEN, so the precomputed
# dit-latent pool stays valid and decoder-independent, and w_ditcons is kept MODERATE
# (3.0) so DiT-robustness does not erode the multiscale recon win.
#
# STATE MACHINE: PRECOMPUTE (skip if pool exists) -> TRAIN -> EVAL(recon+dit).
# AUTO-EVAL uses the FINAL-epoch ckpt, NOT best-val: val has no robust/ditcons term, so
# best-val would mis-select. Idempotent: resume own vae_epoch*.pt, else init from main VAE.
#
# LAUNCH (OS-detached):
#   $env:POLY3D_PY = "C:/Users/shanu/anaconda3/envs/polygen/python.exe"
#   Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','<repo>/scripts/run_main_ditcons.ps1' -WindowStyle Hidden -PassThru
#   Foreground live: ... -File scripts/run_main_ditcons.ps1 -Live
# -----------------------------------------------------------------------------
#
# BUDGET NOTE (the same epoch-scale trap that already bit Stage 1 and Stage 2) ---
# -Epochs 15 is the PG-pilot number, where train was 340k records = 5,300 loader
# batches per epoch at bs64, so the whole fine-tune was ~80k batches. The full build
# has ~5.3M train records, where ONE epoch is ~83,000 batches -- the entire PG budget
# per epoch, 1.24M batches over 15 epochs. --save_every 1 then puts the resume
# granularity at one full epoch as well, which is days.
# -StepsPerEpoch cuts an epoch to N loader batches (virtual epoch; the loader reshuffles
# every epoch, so no data is discarded). The default 8,000 x 15ep = 120k batches is
# ~1.5x the PG budget for ~15x the data, matching the ratio run_main_dit.ps1 uses.
# Pass 0 for whole passes (not advised here). Note grad_accum is 2, so optimizer steps
# per epoch = N / 2.
#
# PRECOMPUTE COST: the pool runs an -NSteps ODE per record. precompute_dit_latents.py
# measured ~2.8h for PG's 340k records at -NSteps 100, so the full build extrapolates to
# ~40h+. -NSteps 20 cuts that ~5x and is the setting the original ditcons win (PG v3e,
# runtime sampling at ditcons_steps 20) was actually produced with.
# DO NOT try to shrink this by capping the record count instead. The pool is keyed by
# record index and collate_fn only activates z_dit when EVERY record in the batch has one
# ("has_zdit = all(...)" in dataset.py). A partial pool therefore yields mixed batches
# that silently fall back to z_dit = None -- and because --dit_latent_lmdb suppresses the
# runtime DiT load, there is nothing to fall back TO, so w_ditcons quietly stops firing
# and the stage degrades into a plain recon fine-tune with no error. On top of that,
# split_lmdb.py writes records in source order, so a leading slice of the index range is
# the first few classes, not a sample of the data.
param(
    [switch]$Live,
    [int]$Width = 256,
    [int]$Epochs = 15,
    [int]$StepsPerEpoch = 8000,
    [double]$Lr = 5e-5,
    [double]$WRobust = 3.0,
    [double]$RobustNoiseStd = 0.35,
    [double]$WDitcons = 3.0,
    [int]$NSteps = 100,
    [int]$NSamples = 1,
    [double]$DitLatentMapGb = 40,
    [string]$VaeRun = "polyomics_main_vae"
)

if ($env:POLY3D_PY) { $py = $env:POLY3D_PY } else { $py = "python" }
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$env:PYTHONUTF8 = "1"
$prevOutEnc = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:256"
Set-Location $repo

$edge       = [int]($Width / 2)
$out        = "runs/polyomics_main_ditcons"
$status     = "$out/status.txt"
$train      = "data/polyomics_all_train.lmdb"
$val        = "data/polyomics_all_val.lmdb"
$vae        = "runs/$VaeRun/vae_best.pt"
$ditCkpt    = "runs/polyomics_main_dit/dit_best.pt"
# The pool is only valid for the encoder it was made with, and the PRECOMPUTE step below
# reuses any pool that already exists. Keying the filename on -VaeRun keeps a pool built
# for a different (e.g. the collapsed first-attempt) VAE from being silently reused.
$ditLatents = "data/polyomics_all_ditlatents_$VaeRun.lmdb"

function Done($tag) {
    return (Test-Path $status) -and (Select-String -Path $status -Pattern $tag -Quiet)
}
function LatestCkpt($dir) {
    if (-not (Test-Path $dir)) { return $null }
    $f = Get-ChildItem -Path $dir -Filter "vae_epoch*.pt" -ErrorAction SilentlyContinue |
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
"START $(Get-Date -Format o) width=$Width epochs=$Epochs steps_per_epoch=$StepsPerEpoch lr=$Lr w_robust=$WRobust w_ditcons=$WDitcons n_steps=$NSteps pos_loss=multiscale_distmat init=$vae freeze_encoder=1 beta=0.1" |
    Out-File -Append -Encoding ascii $status

# ---- PRECOMPUTE dit-latents: reuse existing pool; make only if missing --------
if (-not (Done "PRECOMPUTE_DONE")) {
    if (Test-Path $ditLatents) {
        "PRECOMPUTE_DONE (reuse existing pool $ditLatents) $(Get-Date -Format o)" |
            Out-File -Append -Encoding ascii $status
    } else {
        "PRECOMPUTE_START $(Get-Date -Format o) src=$train out=$ditLatents vae=$vae dit=$ditCkpt" |
            Out-File -Append -Encoding ascii $status
        & $py "scripts/precompute_dit_latents.py" `
            --vae_checkpoint $vae --dit_checkpoint $ditCkpt `
            --src_lmdb $train --out_lmdb $ditLatents `
            --n_steps $NSteps --n_samples $NSamples `
            --batch_size 64 --max_atoms 288 --map_size_gb $DitLatentMapGb --num_workers 8 --log_every 100 `
            1> "$out/precompute_out.log" 2> "$out/precompute_err.log"
        if ($LASTEXITCODE -ne 0) { Fail "PRECOMPUTE" $LASTEXITCODE }
        "PRECOMPUTE_DONE exit=0 $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
    }
}

# ---- TRAIN: decoder-only (frozen encoder), multiscale recon + ditcons ---------
if (-not (Done "TRAIN_DONE")) {
    $a = @(
        "--stage","vae","--train_lmdb",$train,"--val_lmdb",$val,"--out_dir",$out,
        "--hidden_dim",[string]$Width,"--edge_dim",[string]$edge,"--vae_hidden_dim",[string]$Width,
        "--cond_layers","4","--latent_dim","16","--enc_layers","4","--dec_layers","4",
        "--egt_every","2","--enc_egt_every","2",
        "--beta_start","0.1","--beta_end","0.1","--beta_warmup_epochs","1",
        "--pos_loss_type","multiscale_distmat","--w_local","1.0","--w_global","1.0",
        "--w_pos","1.0","--w_bond","1.0","--w_angle","0.5","--w_dihedral","0.1",
        "--w_clash","5.0","--clash_factor","0.6","--clash_min_graph_dist","3","--clash_max_pairs","512",
        "--w_robust",[string]$WRobust,"--robust_noise_std",[string]$RobustNoiseStd,
        "--w_ditcons",[string]$WDitcons,"--dit_latent_lmdb",$ditLatents,
        "--freeze_encoder",
        "--batch_size","64","--grad_accum","2","--epochs",[string]$Epochs,
        "--steps_per_epoch",[string]$StepsPerEpoch,
        "--lr",[string]$Lr,"--lr_min","1e-5","--weight_decay","1e-5","--grad_clip","5.0",
        "--warmup_steps","200","--warmup_start_factor","0.05","--gnorm_log_every","100",
        "--max_atoms","288","--val_subset_ratio","0.3","--empty_cache_every","500",
        "--num_workers","16","--prefetch_factor","4","--save_every","1","--seed","42"
    )
    $ck = LatestCkpt $out
    if ($ck) {
        $a += @("--resume",$ck)
        "TRAIN_START resume=$ck $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
    } else {
        $a += @("--init_weights",$vae)
        "TRAIN_START init=$vae $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
    }
    if ($Live) {
        & $py "scripts/train.py" @a 1> "$out/train_out.log"
    } else {
        & $py "scripts/train.py" @a 1> "$out/train_out.log" 2> "$out/train_err.log"
    }
    if ($LASTEXITCODE -ne 0) { Fail "TRAIN" $LASTEXITCODE }
    "TRAIN_DONE exit=0 $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

# ---- EVAL: FINAL-EPOCH ckpt (NOT best-val), recon + dit -----------------------
if (-not (Done "EVAL_DONE")) {
    $ckpt = LatestCkpt $out
    if (-not $ckpt) { Fail "EVAL" 1 }
    "EVAL_START $(Get-Date -Format o) ckpt=$ckpt" | Out-File -Append -Encoding ascii $status
    & $py "scripts/eval_ensemble.py" --checkpoint $ckpt --val_lmdb $val `
        --mode recon --max_ref 100 --n_gen 50 --max_atoms 288 --batch_size 32 `
        --out "$out/eval_recon_final.json" 1> "$out/eval_out.log" 2> "$out/eval_err.log"
    & $py "scripts/eval_ensemble.py" --checkpoint $ckpt --val_lmdb $val `
        --mode dit --dit_checkpoint $ditCkpt --max_ref 100 --n_gen 50 --max_atoms 288 --batch_size 32 --n_steps 100 `
        --out "$out/eval_dit_final.json" 1>> "$out/eval_out.log" 2>> "$out/eval_err.log"
    "EVAL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

"ALL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
Write-Host ""
Write-Host "Report: $out/eval_recon_final.json, $out/eval_dit_final.json (FINAL-epoch ckpt)"

[Console]::OutputEncoding = $prevOutEnc
