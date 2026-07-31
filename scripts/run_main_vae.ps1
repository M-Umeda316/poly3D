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

param(
    [switch]$Live,
    [int]$Width = 256,
    [int]$BatchSize = 128,
    [int]$Epochs = 300,
    [double]$Lr = 3e-4
)

if ($env:POLY3D_PY) { $py = $env:POLY3D_PY } else { $py = "python" }
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$env:PYTHONUTF8 = "1"
$prevOutEnc = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:256"
Set-Location $repo

$edge   = [int]($Width / 2)
$out    = "runs/polyomics_main_vae"
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
"START $(Get-Date -Format o) width=$Width batch=$BatchSize epochs=$Epochs lr=$Lr pos_loss=multiscale_distmat beta=0->0.1(warmup20) freeze_encoder=0 init=NONE (FROM-SCRATCH)" |
    Out-File -Append -Encoding ascii $status

# ---- TRAIN: full VAE from scratch, multiscale recon + clash guard ------------
if (-not (Done "TRAIN_DONE")) {
    $a = @(
        "--stage","vae","--train_lmdb",$train,"--val_lmdb",$val,"--out_dir",$out,
        "--hidden_dim",[string]$Width,"--edge_dim",[string]$edge,"--vae_hidden_dim",[string]$Width,
        "--cond_layers","4","--latent_dim","16","--enc_layers","4","--dec_layers","4",
        "--egt_every","2","--enc_egt_every","2",
        "--beta_start","0","--beta_end","0.1","--beta_warmup_epochs","20",
        "--pos_loss_type","multiscale_distmat","--w_local","1.0","--w_global","1.0",
        "--w_pos","1.0","--w_bond","1.0","--w_angle","0.5","--w_dihedral","0.1",
        "--w_clash","5.0","--clash_factor","0.6","--clash_min_graph_dist","3","--clash_max_pairs","512",
        "--batch_size",[string]$BatchSize,"--grad_accum","1","--epochs",[string]$Epochs,
        "--lr",[string]$Lr,"--lr_min","1e-5","--warmup_steps","500","--grad_clip","1.0",
        "--weight_decay","1e-5","--max_atoms","288","--val_subset_ratio","0.3",
        "--empty_cache_every","500","--num_workers","16","--prefetch_factor","4",
        "--save_every","5","--seed","42","--gnorm_log_every","100"
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
