# PolyOmics VAE v3g = SIZE-OVERSAMPLING lever (ASCII only for PS 5.1).
# Fine-tune the accepted v3c FULL VAE while OVERSAMPLING large repeat units, so the
# rare big molecules get many more gradient steps. Architecture is UNCHANGED.
#
# WHY --------------------------------------------------------------------------
# Overfit diagnosis proved the width256 arch can memorize+reconstruct big units
# (16-27 heavy atoms) perfectly (recon validity 1.00) => capacity is NOT the limit.
# The real cause: only ~11.4% of PG_train (340k units) are large (16+ heavy atoms),
# so they are too rare to be GENERALIZED. Fix: WeightedRandomSampler with weight
# w = max(1, size)^power concentrates gradient steps on the tail. power=1.5 lifts
# the large-unit share from ~11.8% to ~33% per epoch (measured on the size index).
#
# FULL VAE (NO FREEZE) ---------------------------------------------------------
# Unlike v3d, the encoder is NOT frozen: the encoder itself must learn a better
# representation of large units, so the whole VAE trains. (This means dit_v2 becomes
# stale w.r.t. the new latent space, so this run only self-evaluates via recon.)
#
# ARCH == v3c (width256, EGT enc+dec 2&2, enc/dec4, latent16) so v3c weights load
# cleanly via --init_weights. beta HELD at 0.1 (no re-warmup) and a GENTLE lr (5e-5
# with 200-step warmup) because an lr=1e-4 probe caused PG forgetting.
#
# READ AFTER -------------------------------------------------------------------
# Auto-eval writes runs/polyomics_vae_v3g/eval_recon_final.json from the FINAL-epoch
# ckpt, --mode recon on PG val. That recon validity/torsion on the large-unit tail
# IS the probe metric for whether oversampling helped generalization.
#
# LAUNCH (OS-detached, harness idle-kill resistant):
#   $env:POLY3D_PY = "C:/Users/shanu/anaconda3/envs/polygen/python.exe"
#   Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','<repo>/scripts/run_vae_v3g_oversample.ps1' -WindowStyle Hidden -PassThru
#   Foreground live: ... -File scripts/run_vae_v3g_oversample.ps1 -Live
# Idempotent: if v3g has its own vae_epoch*.pt it --resume's that; else it
#   --init_weights from v3c. The size index is built once (SIZEIDX_DONE), then reused.
# -----------------------------------------------------------------------------

param(
    [switch]$Live,
    [int]$Width = 256,               # MUST match v3c for the weight load to succeed.
    [int]$Epochs = 15,
    [double]$Lr = 5e-5,              # gentle fine-tune lr (probe forgot PG at 1e-4).
    [double]$OversamplePower = 1.5,  # weight = max(1,size)^power. 1.5 -> ~33% large tail.
    [double]$WClash = 5.0,           # keep v3c geometry pressure on the recon output.
    [string]$InitFrom = "runs/polyomics_vae_v3c/vae_best.pt",
    [string]$OutName = "polyomics_vae_v3g"
)

if ($env:POLY3D_PY) { $py = $env:POLY3D_PY } else { $py = "python" }
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$env:PYTHONUTF8 = "1"
$prevOutEnc = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:256"
Set-Location $repo

$edge      = [int]($Width / 2)
$out       = "runs/$OutName"
$status    = "$out/status.txt"
$train     = "data/polyomics_PG_train.lmdb"
$val       = "data/polyomics_PG_val.lmdb"
$sizeIndex = "$train.sizes.npy"

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
"START $(Get-Date -Format o) width=$Width epochs=$Epochs lr=$Lr oversample_power=$OversamplePower w_clash=$WClash freeze_encoder=0 (FULL VAE)" |
    Out-File -Append -Encoding ascii $status

# ---- Build the per-record size index once (reused across relaunches) ----------
if (-not (Done "SIZEIDX_DONE")) {
    if (Test-Path $sizeIndex) {
        "SIZEIDX_DONE $(Get-Date -Format o) exists=$sizeIndex" | Out-File -Append -Encoding ascii $status
    } else {
        "SIZEIDX_START $(Get-Date -Format o) src=$train" | Out-File -Append -Encoding ascii $status
        & $py "scripts/build_size_index.py" --src $train 1>> "$out/sizeidx_out.log" 2>> "$out/sizeidx_err.log"
        $ec = $LASTEXITCODE
        if (($ec -ne 0) -or (-not (Test-Path $sizeIndex))) {
            "SIZEIDX_FAILED exit=$ec $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
            exit 1
        }
        "SIZEIDX_DONE $(Get-Date -Format o) out=$sizeIndex" | Out-File -Append -Encoding ascii $status
    }
}

# ---- Fine-tune from v3c (or resume v3g's own ckpt), FULL VAE + oversampling ----
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
        "--beta_start","0.1","--beta_end","0.1","--beta_warmup_epochs","1",  # HOLD beta=0.1 (collapse回避, v3c-f同様)
        "--w_pos","1.0","--w_bond","1.0","--w_angle","0.5","--w_dihedral","0.1",
        "--w_clash",[string]$WClash,"--clash_factor","0.6","--clash_min_graph_dist","3","--clash_max_pairs","512",
        # SIZE OVERSAMPLING (full VAE -> encoder also learns large-unit representation)
        "--size_index",$sizeIndex,"--oversample_power",[string]$OversamplePower,
        "--pos_loss_type","kabsch",
        "--batch_size","64","--grad_accum","2",
        "--epochs",[string]$Epochs,
        "--lr",[string]$Lr,"--lr_min","1e-5",
        "--weight_decay","1e-5","--grad_clip","5.0",
        "--warmup_steps","200","--warmup_start_factor","0.05",
        "--gnorm_log_every","50",
        "--max_atoms","288",
        "--val_subset_ratio","0.3",
        "--empty_cache_every","500",
        "--num_workers","8","--prefetch_factor","4",
        "--save_every","1","--seed","42"
    )
    # Idempotency: prefer v3g's own latest ckpt (resume); only init from v3c on a fresh start.
    $ck = LatestCkpt $out
    if ($ck) {
        $a += @("--resume",$ck)
        "TRAIN_START $(Get-Date -Format o) resume=$ck" | Out-File -Append -Encoding ascii $status
    } else {
        $a += @("--init_weights",$InitFrom)
        "TRAIN_START $(Get-Date -Format o) init_weights=$InitFrom" | Out-File -Append -Encoding ascii $status
    }
    if ($Live) {
        & $py "scripts/train.py" @a 1> "$out/train_out.log"
    } else {
        & $py "scripts/train.py" @a 1> "$out/train_out.log" 2> "$out/train_err.log"
    }
    $ec = $LASTEXITCODE
    if ($ec -ne 0) {
        "TRAIN_FAILED exit=$ec $(Get-Date -Format o) - relaunch to resume from latest ckpt" |
            Out-File -Append -Encoding ascii $status
        exit $ec
    }
    "TRAIN_DONE exit=0 $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

# ---- Auto-eval: recon on the FINAL-epoch ckpt (probe for oversampling) --------
# Encoder changed => the existing dit_v2 latents are stale, so we run recon ONLY.
if (-not (Done "EVAL_DONE")) {
    $ckpt = LatestCkpt $out
    if (-not $ckpt) { $ckpt = "$out/vae_best.pt" }
    "EVAL_START $(Get-Date -Format o) ckpt=$ckpt mode=recon" | Out-File -Append -Encoding ascii $status

    & $py "scripts/eval_ensemble.py" --checkpoint $ckpt --val_lmdb $val `
        --mode recon --max_ref 100 --n_gen 50 --max_atoms 288 --batch_size 32 `
        --out "$out/eval_recon_final.json" 1> "$out/eval_out.log" 2> "$out/eval_err.log"
    $ec = $LASTEXITCODE
    if ($ec -ne 0) {
        "EVAL_FAILED exit=$ec $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
        exit $ec
    }
    "EVAL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

"ALL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
Write-Host ""
Write-Host "Report: $out/vae_log.csv (val_pos not drifting up?), $out/eval_recon_final.json"
Write-Host "SUCCESS = recon validity/torsion on the large-unit tail improves over v3c WITHOUT small-unit recon regressing."

[Console]::OutputEncoding = $prevOutEnc
