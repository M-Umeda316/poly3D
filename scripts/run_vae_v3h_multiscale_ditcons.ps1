# PolyOmics VAE v3h = multiscale recon (the big win) + DiT-consistency (close the
# gen->recon gap). ASCII only (PS 5.1).
#
# WHY --------------------------------------------------------------------------
# The multiscale_distmat ablation lifted large-unit RECON 0.47->0.85 (near the
# memorization ceiling). But generation stayed at 0.36 because the SAME latents
# (v3c encoder, frozen) decoded via dit_v2 do not reach the decoder's new ceiling
# -> the DiT latent quality is now the large-unit generation bottleneck.
# Fix: re-apply the ditcons lever (harden the decoder on the REAL dit_v2 latents)
# ON TOP of the multiscale decoder, so generation climbs toward the 0.85 ceiling.
#
# CORRECTNESS ------------------------------------------------------------------
# Encoder FROZEN (== v3c), so the precomputed dit-latent pool
# (data/polyomics_PG_train_ditlatents.lmdb, made from v3c cond + dit_v2) is STILL
# VALID and decoder-independent -> reuse it directly (no re-precompute cost).
# Recon loss stays multiscale_distmat to KEEP the 0.85 recon gain; w_ditcons is
# kept MODERATE (3.0) so the DiT-robustness does not erode that recon win.
#
# STATE MACHINE: (PRECOMPUTE skipped, pool exists) -> TRAIN -> EVAL(recon+dit).
# LAUNCH (OS-detached): Start-Process powershell -File this.ps1 -WindowStyle Hidden
# -----------------------------------------------------------------------------

param(
    [switch]$Live,
    [int]$Width = 256,
    [int]$Epochs = 15,
    [double]$Lr = 5e-5,
    [double]$WClash = 5.0,
    [double]$WBond = 1.0,
    [double]$WAngle = 0.5,
    [double]$WRobust = 3.0,
    [double]$RobustNoiseStd = 0.35,
    [double]$WDitcons = 3.0,       # moderate: preserve the multiscale recon 0.85 win.
    [double]$WLocal = 1.0,
    [double]$WGlobal = 1.0,
    [int]$NSteps = 100,
    [int]$NSamples = 1,
    [string]$InitFrom = "runs/abl_multiscale/vae_epoch0012.pt",
    [string]$VaeForCond = "runs/polyomics_vae_v3c/vae_best.pt",
    [string]$OutName = "polyomics_vae_v3h",
    [string]$DitCkpt = "runs/polyomics_dit_v2/dit_best.pt",
    [string]$DitLatents = "data/polyomics_PG_train_ditlatents.lmdb"
)

if ($env:POLY3D_PY) { $py = $env:POLY3D_PY } else { $py = "python" }
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:256"
Set-Location $repo

$edge   = [int]($Width / 2)
$out    = "runs/$OutName"
$status = "$out/status.txt"
$train  = "data/polyomics_PG_train.lmdb"
$val    = "data/polyomics_PG_val.lmdb"

function Done($tag) { return (Test-Path $status) -and (Select-String -Path $status -Pattern $tag -Quiet) }
function LatestCkpt($dir) {
    if (-not (Test-Path $dir)) { return $null }
    $f = Get-ChildItem -Path $dir -Filter "vae_epoch*.pt" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
    if ($f) { return $f.FullName } else { return $null }
}

if (-not (Test-Path $out)) { New-Item -ItemType Directory -Force -Path $out | Out-Null }
"START $(Get-Date -Format o) width=$Width epochs=$Epochs lr=$Lr w_clash=$WClash w_robust=$WRobust w_ditcons=$WDitcons pos_loss=multiscale_distmat w_local=$WLocal w_global=$WGlobal init=$InitFrom freeze_encoder=1" | Out-File -Append -Encoding ascii $status

# ---- PRECOMPUTE: reuse existing pool; regenerate only if missing (idempotent) ----
if (-not (Done "PRECOMPUTE_DONE")) {
    if (Test-Path $DitLatents) {
        "PRECOMPUTE_DONE (reuse existing pool $DitLatents) $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
    } else {
        "PRECOMPUTE_START $(Get-Date -Format o) src=$train out=$DitLatents vae=$VaeForCond dit=$DitCkpt" | Out-File -Append -Encoding ascii $status
        $pa = @("--vae_checkpoint",$VaeForCond,"--dit_checkpoint",$DitCkpt,"--src_lmdb",$train,"--out_lmdb",$DitLatents,"--n_steps",[string]$NSteps,"--n_samples",[string]$NSamples,"--batch_size","64","--max_atoms","288","--map_size_gb","6","--num_workers","4","--log_every","100")
        & $py "scripts/precompute_dit_latents.py" @pa 1> "$out/precompute_out.log" 2> "$out/precompute_err.log"
        if ($LASTEXITCODE -ne 0) { "PRECOMPUTE_FAILED exit=$LASTEXITCODE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status; exit $LASTEXITCODE }
        "PRECOMPUTE_DONE exit=0 $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
    }
}

# ---- TRAIN: decoder-only, multiscale recon + ditcons (precomputed latents) -------
if (-not (Done "TRAIN_DONE")) {
    $a = @(
        "--stage","vae","--train_lmdb",$train,"--val_lmdb",$val,"--out_dir",$out,
        "--hidden_dim",[string]$Width,"--edge_dim",[string]$edge,"--vae_hidden_dim",[string]$Width,
        "--cond_layers","4","--latent_dim","16","--enc_layers","4","--dec_layers","4",
        "--egt_every","2","--enc_egt_every","2",
        "--beta_start","0.1","--beta_end","0.1","--beta_warmup_epochs","1",
        "--pos_loss_type","multiscale_distmat","--w_local",[string]$WLocal,"--w_global",[string]$WGlobal,
        "--w_pos","1.0","--w_bond",[string]$WBond,"--w_angle",[string]$WAngle,"--w_dihedral","0.1",
        "--w_clash",[string]$WClash,"--clash_factor","0.6","--clash_min_graph_dist","3","--clash_max_pairs","512",
        "--w_robust",[string]$WRobust,"--robust_noise_std",[string]$RobustNoiseStd,
        "--w_ditcons",[string]$WDitcons,"--dit_latent_lmdb",$DitLatents,
        "--freeze_encoder",
        "--batch_size","64","--grad_accum","2","--epochs",[string]$Epochs,
        "--lr",[string]$Lr,"--lr_min","1e-5","--weight_decay","1e-5","--grad_clip","5.0",
        "--warmup_steps","200","--warmup_start_factor","0.05","--gnorm_log_every","100",
        "--max_atoms","288","--val_subset_ratio","0.3","--empty_cache_every","500",
        "--num_workers","8","--prefetch_factor","4","--save_every","1","--seed","42"
    )
    $ck = LatestCkpt $out
    if ($ck) { $a += @("--resume",$ck); "TRAIN_START resume=$ck $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status }
    else { $a += @("--init_weights",$InitFrom); "TRAIN_START init=$InitFrom $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status }
    & $py "scripts/train.py" @a 1> "$out/train_out.log" 2> "$out/train_err.log"
    if ($LASTEXITCODE -ne 0) { "TRAIN_FAILED exit=$LASTEXITCODE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status; exit $LASTEXITCODE }
    "TRAIN_DONE exit=0 $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

# ---- EVAL: FINAL-EPOCH ckpt, recon + dit ----
if (-not (Done "EVAL_DONE")) {
    $ckpt = LatestCkpt $out
    if (-not $ckpt) { "EVAL_FAILED no ckpt $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status; exit 1 }
    "EVAL_START $(Get-Date -Format o) ckpt=$ckpt" | Out-File -Append -Encoding ascii $status
    & $py "scripts/eval_ensemble.py" --checkpoint $ckpt --val_lmdb $val --mode recon --max_ref 100 --n_gen 50 --out "$out/eval_recon_final.json" 1> "$out/eval_out.log" 2> "$out/eval_err.log"
    & $py "scripts/eval_ensemble.py" --checkpoint $ckpt --val_lmdb $val --mode dit --dit_checkpoint $DitCkpt --max_ref 100 --n_gen 50 --out "$out/eval_dit_final.json" 1>> "$out/eval_out.log" 2>> "$out/eval_err.log"
    "EVAL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}
"ALL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
