# PolyOmics MAIN VAE = Stage 1 Structural VAE, FROM-SCRATCH on the full 22-class data.
# ASCII only (PS 5.1). Detached + idempotent via status flags.
#
# WHY FROM-SCRATCH ------------------------------------------------------------
# The PG-only pilot proved the pipeline; this is the full-class production VAE with NO
# init_weights and NO freeze (the whole VAE learns the 22-class latent space) and NO
# oversampling (small PolyOmics units removed the large-molecule rarity problem). beta
# uses a 0->0.1 warmup: 0.1 is the collapse-free ceiling (1.0 collapsed the posterior),
# and the from-scratch encoder needs the warmup to avoid an early KL over-penalty.
#
# ARCH (FIXED recipe): width256, edge128, vae_hidden256, cond_layers4, latent16,
# enc_layers4, dec_layers4, EGT enc+dec every 2. RECON loss: multiscale_distmat with
# w_local/w_global 1.0, plus clash guard (w_clash 5.0, factor 0.6, min_graph_dist 3).
#
# GRAD NORM NOTE: on a from-scratch run the gnorm is driven mainly by the angle and
# dihedral geometry terms. If training is unstable (gnorm spikes / NaN), lower
# --grad_clip, or reduce --w_angle / --w_dihedral, rather than dropping the recon terms.
#
# AUTO-EVAL: vae_best.pt (Stage 1 is pure recon with NO robust/ditcons term, so the
# best-val checkpoint is the correct pick here) -> eval_ensemble --mode recon.
#
# LAUNCH (OS-detached):
#   $env:POLY3D_PY = "C:/Users/shanu/anaconda3/envs/polygen/python.exe"
#   Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','<repo>/scripts/run_main_vae.ps1' -WindowStyle Hidden -PassThru
#   Foreground live: ... -File scripts/run_main_vae.ps1 -Live
# Idempotent: resume own vae_epoch*.pt if present, else start from scratch.
# -----------------------------------------------------------------------------

# BUDGET NOTE: the defaults below are the PG-pilot scale (train 340k entries). On the
# full 22-class build the train set is millions of entries, so 1 epoch is already tens of
# thousands of steps. Three knobs MUST be rescaled together there, because all three are
# epoch-based, not step-based:
#   -Epochs            CosineAnnealingLR uses T_max=epochs, so this IS the LR decay length.
#   -BetaWarmupEpochs  beta ramps as (epoch-1)/warmup; a warmup longer than the run means
#                      beta never reaches 0.1.
#   -SaveEvery         resume granularity; 1 epoch of the full set is hours.
# Run scripts/suggest_vae_budget.py against the built lmdb to get the values (it reads the
# real entry count and atom-size distribution and prints the launch line).
#
# On the full build a real epoch is ~100k steps, so honouring the step budget with whole
# passes leaves only ~3 epochs = a 3-point staircase for LR, beta, val and ckpt selection.
# -StepsPerEpoch cuts one epoch down to N batches (reshuffled every epoch, so no data is
# discarded) which restores the resolution of every epoch-based schedule.
#
# BATCH SIZE: VRAM is batch_size x (largest unit in the batch)^2 because the EGT global
# attention is a dense (B, N_max, N_max) tensor -- a single 288-atom unit pads the whole
# batch to 288. bs128 needs ~34GB in that case and does not fit; bs64 needs ~17GB and does.
# Prefer -BatchSize 64 -GradAccum 2 over -BatchSize 128: the effective batch stays 128, so
# the lr and grad_clip keep the calibration they were tuned at on the PG pilot.
# Note -StepsPerEpoch counts LOADER batches, so optimizer steps per epoch = N / GradAccum.
#
# LR WARNING (learned the hard way, twice) ------------------------------------
# At width256 from-scratch, lr 3e-4 lands in a bad basin AND collapses the posterior.
#   PG v3  (width256, from-scratch, lr 3e-4)   -> val_kl 0.0059, val_pos stalled. ABANDONED.
#   PG v3b (width256, from-scratch, lr 1.5e-4, warmup 500) -> val_kl settles ~0.137. GOOD.
#   MAIN 1st attempt (22-class, lr 3e-4, 80ep) -> val_kl 0.0009, recon validity 0.00. DEAD.
# The default is therefore 1.5e-4, which is the only lr that has ever produced a healthy
# width256 from-scratch VAE here. Do not raise it without a collapse-free result to point at.
# Healthy-run tripwire: val_kl must stay above ~0.1 once beta reaches 0.1. If val_kl drops
# below ~0.05 the posterior is collapsing -- kill the run, do not wait for it to finish.

param(
    [switch]$Live,
    [int]$Width = 256,
    [int]$BatchSize = 128,
    [int]$Epochs = 300,
    [double]$Lr = 1.5e-4,
    [string]$OutName = "polyomics_main_vae",
    [int]$BetaWarmupEpochs = 20,
    [double]$ValSubsetRatio = 0.3,
    [int]$SaveEvery = 5,
    [int]$OomMaxSkips = 0,
    [int]$StepsPerEpoch = 0,
    [int]$GradAccum = 1
)

if ($env:POLY3D_PY) { $py = $env:POLY3D_PY } else { $py = "python" }
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$env:PYTHONUTF8 = "1"
$prevOutEnc = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:256"
Set-Location $repo

$edge   = [int]($Width / 2)
$out    = "runs/$OutName"
$status = "$out/status.txt"
$train  = "data/polyomics_all_train.lmdb"
$val    = "data/polyomics_all_val.lmdb"

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
"START $(Get-Date -Format o) width=$Width batch=$BatchSize accum=$GradAccum steps_per_epoch=$StepsPerEpoch epochs=$Epochs lr=$Lr pos_loss=multiscale_distmat beta=0->0.1(warmup$BetaWarmupEpochs) val_ratio=$ValSubsetRatio save_every=$SaveEvery oom_max_skips=$OomMaxSkips freeze_encoder=0 init=NONE (FROM-SCRATCH)" |
    Out-File -Append -Encoding ascii $status

# ---- TRAIN: full VAE from scratch, multiscale recon + clash guard ------------
if (-not (Done "TRAIN_DONE")) {
    $a = @(
        "--stage","vae","--train_lmdb",$train,"--val_lmdb",$val,"--out_dir",$out,
        "--hidden_dim",[string]$Width,"--edge_dim",[string]$edge,"--vae_hidden_dim",[string]$Width,
        "--cond_layers","4","--latent_dim","16","--enc_layers","4","--dec_layers","4",
        "--egt_every","2","--enc_egt_every","2",
        "--beta_start","0","--beta_end","0.1","--beta_warmup_epochs",[string]$BetaWarmupEpochs,
        "--pos_loss_type","multiscale_distmat","--w_local","1.0","--w_global","1.0",
        "--w_pos","1.0","--w_bond","1.0","--w_angle","0.5","--w_dihedral","0.1",
        "--w_clash","5.0","--clash_factor","0.6","--clash_min_graph_dist","3","--clash_max_pairs","512",
        "--batch_size",[string]$BatchSize,"--grad_accum",[string]$GradAccum,
        "--epochs",[string]$Epochs,
        "--lr",[string]$Lr,"--lr_min","1e-5","--warmup_steps","500","--grad_clip","1.0",
        "--weight_decay","1e-5","--max_atoms","288",
        "--val_subset_ratio",[string]$ValSubsetRatio,
        "--empty_cache_every","500","--num_workers","16","--prefetch_factor","4",
        "--save_every",[string]$SaveEvery,"--oom_max_skips",[string]$OomMaxSkips,
        "--steps_per_epoch",[string]$StepsPerEpoch,
        "--seed","42","--gnorm_log_every","100"
    )
    $ck = LatestCkpt $out
    if ($ck) {
        $a += @("--resume",$ck)
        "TRAIN_START $(Get-Date -Format o) resume=$ck" | Out-File -Append -Encoding ascii $status
    } else {
        "TRAIN_START $(Get-Date -Format o) from-scratch" | Out-File -Append -Encoding ascii $status
    }
    if ($Live) {
        & $py "scripts/train.py" @a 1> "$out/train_out.log"
    } else {
        & $py "scripts/train.py" @a 1> "$out/train_out.log" 2> "$out/train_err.log"
    }
    if ($LASTEXITCODE -ne 0) {
        "TRAIN_FAILED exit=$LASTEXITCODE $(Get-Date -Format o) - relaunch to resume from latest ckpt" |
            Out-File -Append -Encoding ascii $status
        [Console]::OutputEncoding = $prevOutEnc
        exit $LASTEXITCODE
    }
    "TRAIN_DONE exit=0 $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

# ---- AUTO-EVAL: vae_best.pt, --mode recon (pure recon -> best-val is valid) ---
if (-not (Done "EVAL_DONE")) {
    $ckpt = "$out/vae_best.pt"
    if (-not (Test-Path $ckpt)) { $ckpt = LatestCkpt $out }
    "EVAL_START $(Get-Date -Format o) ckpt=$ckpt mode=recon" | Out-File -Append -Encoding ascii $status
    & $py "scripts/eval_ensemble.py" --checkpoint $ckpt --val_lmdb $val `
        --mode recon --max_ref 100 --n_gen 50 --max_atoms 288 --batch_size 32 `
        --out "$out/eval_recon.json" 1> "$out/eval_out.log" 2> "$out/eval_err.log"
    if ($LASTEXITCODE -ne 0) {
        "EVAL_FAILED exit=$LASTEXITCODE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
        [Console]::OutputEncoding = $prevOutEnc
        exit $LASTEXITCODE
    }
    "EVAL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

"ALL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
Write-Host ""
Write-Host "Report: $out/vae_log.csv, $out/eval_recon.json. Next: run_main_dit.ps1"

[Console]::OutputEncoding = $prevOutEnc
