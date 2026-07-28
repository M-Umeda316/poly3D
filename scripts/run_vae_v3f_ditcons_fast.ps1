# PolyOmics VAE v3f = DiT-CONSISTENCY lever, FAST variant (ASCII only for PS 5.1).
# Same idea as v3e (harden the decoder on the REAL DiT latents), but the per-batch
# ODE sampling that made v3e slow (~64 min/epoch) is PRECOMPUTED once into an LMDB
# and then merely READ during training (~30 min/epoch).
#
# WHY THIS IS CORRECT ----------------------------------------------------------
# cond_encoder + vae.encoder are FROZEN, so cond is deterministic and the DiT
# latent does NOT depend on the decoder. Each training record's z_dit is generated
# ONCE, in that record's OWN atom order, and stored. Training reads it back with no
# remap (structurally correct, immune to PolyOmics per-conformer relabeling). The
# saved-latent decode is the SAME generator as v3e's runtime sample -> identical
# distribution, just amortized.
#
# STATE MACHINE: PRECOMPUTE -> TRAIN -> EVAL (idempotent, crash-resumable) -------
#   PRECOMPUTE: build data/polyomics_PG_train_ditlatents.lmdb from the v3c cond +
#               dit_v2 (n_steps 100, K=1). Skipped if the LMDB already exists.
#   TRAIN     : decoder-only fine-tune, encoder frozen, arch == v3c so weights load
#               via --init_weights from v3e ep10 (continue the gains). The LEVER is
#               --w_ditcons 6.0 (v3e used 3.0) fed by --dit_latent_lmdb (NO runtime
#               DiT loaded). isotropic --w_robust kept ON to preserve v3d gains.
#   EVAL      : FINAL-EPOCH ckpt (NOT vae_best.pt; val loss lacks robust/ditcons
#               terms so early epochs mis-pick), recon + dit modes.
#
# LAUNCH (OS-detached, harness idle-kill resistant):
#   $env:POLY3D_PY = "C:/Users/shanu/anaconda3/envs/polygen/python.exe"
#   Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','<repo>/scripts/run_vae_v3f_ditcons_fast.ps1' -WindowStyle Hidden -PassThru
#   Foreground live: ... -File scripts/run_vae_v3f_ditcons_fast.ps1 -Live
# Idempotent: PRECOMPUTE resumes existing keys; TRAIN resumes v3f's own ckpt else
#   --init_weights from v3e ep10; EVAL skipped once EVAL_DONE. Relaunch is safe.
# SUCCESS = eval_dit_final validity/torsion beats v3c/v3d/v3e dit WITHOUT eval_recon
#   regressing, at ~half the wall-clock of v3e.
# -----------------------------------------------------------------------------

param(
    [switch]$Live,
    [int]$Width = 256,            # MUST match v3c for the weight load to succeed.
    [int]$Epochs = 20,            # more epochs are affordable now that we are fast.
    [double]$Lr = 5e-5,           # gentle fine-tune lr.
    [double]$WClash = 5.0,        # keep v3c geometry pressure on the recon output.
    [double]$WBond = 1.5,         # v3c: 1.0 -> 1.5
    [double]$WAngle = 0.75,       # v3c: 0.5 -> 0.75
    [double]$WRobust = 3.0,       # isotropic off-manifold guardrail (keep v3d gains).
    [double]$RobustNoiseStd = 0.35,
    [double]$WDitcons = 6.0,      # THE LEVER: v3e 3.0 -> 6.0 (harder DiT-consistency).
    [int]$NSteps = 100,           # ODE steps for the PRECOMPUTE (eval uses 100 too).
    [int]$NSamples = 1,           # K latents per record.
    [string]$InitFrom = "runs/polyomics_vae_v3e/vae_epoch0010.pt",
    [string]$VaeForCond = "runs/polyomics_vae_v3c/vae_best.pt",
    [string]$OutName = "polyomics_vae_v3f",
    [string]$DitCkpt = "runs/polyomics_dit_v2/dit_best.pt",
    [string]$DitLatents = "data/polyomics_PG_train_ditlatents.lmdb"
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
$train  = "data/polyomics_PG_train.lmdb"
$val    = "data/polyomics_PG_val.lmdb"

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
"START $(Get-Date -Format o) width=$Width epochs=$Epochs lr=$Lr w_clash=$WClash w_bond=$WBond w_angle=$WAngle w_robust=$WRobust robust_noise_std=$RobustNoiseStd w_ditcons=$WDitcons n_steps=$NSteps n_samples=$NSamples dit=$DitCkpt ditlatents=$DitLatents freeze_encoder=1" |
    Out-File -Append -Encoding ascii $status

# ---- PRECOMPUTE: build the DiT-latent LMDB once (idempotent, resumable) -------
# The precompute script itself skips existing keys, so a relaunch continues it.
# We treat "LMDB file exists AND PRECOMPUTE_DONE marker" as the skip condition; if
# the marker is missing (previous run died mid-precompute), we re-run and let the
# script resume from existing keys.
if (-not (Done "PRECOMPUTE_DONE")) {
    "PRECOMPUTE_START $(Get-Date -Format o) src=$train out=$DitLatents vae=$VaeForCond dit=$DitCkpt" |
        Out-File -Append -Encoding ascii $status
    $pa = @(
        "--vae_checkpoint",$VaeForCond,
        "--dit_checkpoint",$DitCkpt,
        "--src_lmdb",$train,
        "--out_lmdb",$DitLatents,
        "--n_steps",[string]$NSteps,
        "--n_samples",[string]$NSamples,
        "--batch_size","64",
        "--max_atoms","288",
        "--map_size_gb","6",
        "--num_workers","4",
        "--log_every","100"
    )
    if ($Live) {
        & $py "scripts/precompute_dit_latents.py" @pa 1> "$out/precompute_out.log"
    } else {
        & $py "scripts/precompute_dit_latents.py" @pa 1> "$out/precompute_out.log" 2> "$out/precompute_err.log"
    }
    $ec = $LASTEXITCODE
    if ($ec -ne 0) {
        "PRECOMPUTE_FAILED exit=$ec $(Get-Date -Format o) - relaunch to resume from existing keys" |
            Out-File -Append -Encoding ascii $status
        exit $ec
    }
    "PRECOMPUTE_DONE exit=0 $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

# ---- TRAIN: decoder-only fine-tune, reading z_dit from the precomputed LMDB ----
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
        "--beta_start","0.1","--beta_end","0.1","--beta_warmup_epochs","1",  # HOLD beta=0.1
        "--w_pos","1.0","--w_bond",[string]$WBond,"--w_angle",[string]$WAngle,"--w_dihedral","0.1",
        "--w_clash",[string]$WClash,"--clash_factor","0.6","--clash_min_graph_dist","3","--clash_max_pairs","512",
        # isotropic off-manifold robustness (keep v3d gains)
        "--w_robust",[string]$WRobust,"--robust_noise_std",[string]$RobustNoiseStd,
        # THE LEVER (fast): DiT-consistency guardrail fed by the PRECOMPUTED latents.
        # No --vae_dit_checkpoint here: --dit_latent_lmdb makes train.py skip the
        # runtime DiT load entirely and read z_dit from the LMDB instead.
        "--w_ditcons",[string]$WDitcons,"--dit_latent_lmdb",$DitLatents,
        # decoder-only, encoder frozen -> latent stays v3c
        "--freeze_encoder",
        "--pos_loss_type","kabsch",
        "--batch_size","64","--grad_accum","2",
        "--epochs",[string]$Epochs,
        "--lr",[string]$Lr,"--lr_min","1e-5",
        "--weight_decay","1e-5","--grad_clip","1.0",
        "--warmup_steps","100","--warmup_start_factor","0.05",
        "--gnorm_log_every","50",
        "--max_atoms","288",
        "--val_subset_ratio","0.3",
        "--empty_cache_every","500",
        "--num_workers","8","--prefetch_factor","4",
        "--save_every","1","--seed","42"
    )
    # Idempotency: prefer v3f's own latest ckpt (resume); only init from v3e ep10 on a fresh start.
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

# ---- Auto-eval: recon + dit, using the FINAL-EPOCH ckpt (NOT vae_best.pt) ----
if (-not (Done "EVAL_DONE")) {
    $ckpt = LatestCkpt $out
    if (-not $ckpt) {
        "EVAL_FAILED $(Get-Date -Format o) - no vae_epoch*.pt found" |
            Out-File -Append -Encoding ascii $status
        exit 1
    }
    $finalEp = & $py -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" $ckpt 2>$null
    "EVAL_START $(Get-Date -Format o) ckpt=$ckpt final_epoch=$finalEp dit=$DitCkpt" |
        Out-File -Append -Encoding ascii $status

    & $py "scripts/eval_ensemble.py" --checkpoint $ckpt --val_lmdb $val `
        --mode recon --max_atoms 288 --batch_size 32 --max_ref 100 --n_gen 50 `
        --out "$out/eval_recon_final.json" 1> "$out/eval_out.log" 2> "$out/eval_err.log"

    # KEY measurement: new decoder driven by the EXISTING dit_v2 (real off-manifold latents).
    & $py "scripts/eval_ensemble.py" --checkpoint $ckpt --val_lmdb $val `
        --mode dit --dit_checkpoint $DitCkpt --max_atoms 288 --batch_size 32 --max_ref 100 --n_gen 50 `
        --out "$out/eval_dit_final.json" 1>> "$out/eval_out.log" 2>> "$out/eval_err.log"

    "EVAL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

"ALL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
Write-Host ""
Write-Host "Report: $out/vae_log.csv (val_pos not drifting up? ditcons falling?), $out/eval_dit_final.json"
Write-Host "SUCCESS = eval_dit_final validity/torsion beats v3c/v3d/v3e dit WITHOUT eval_recon regressing, at ~half v3e wall-clock."

[Console]::OutputEncoding = $prevOutEnc
