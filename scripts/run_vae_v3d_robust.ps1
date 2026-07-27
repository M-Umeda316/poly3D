# PolyOmics VAE v3d = OFF-MANIFOLD ROBUSTNESS lever (ASCII only for PS 5.1).
# Fine-tune the accepted v3c decoder so it stays valid on OFF-MANIFOLD latents.
#
# WHY --------------------------------------------------------------------------
# DiT builds latents by N(0,I)->ODE; those land OUTSIDE the encoder posterior
# manifold (measured per-atom latent norm ~1.76 vs posterior ~1.12). The SAME
# decoder that reconstructs posterior latents well produces LESS valid geometry
# on those off-manifold latents. Fix: during fine-tune, in addition to the normal
# recon (decode of posterior z), also decode a perturbed latent z' = z + std*randn
# and put ONLY GT-free guardrail losses (clash + bond_range) on that output. The
# decoder learns "a slightly-off latent must not blow up".
#
# ENCODER IS FROZEN ------------------------------------------------------------
# We freeze cond_encoder + vae.encoder and train the DECODER ONLY. This keeps the
# latent space identical to v3c, so the EXISTING dit_v2 (width256/hidden256) can be
# reused with NO DiT retrain and NO latent recompute -- the --mode dit eval below
# directly measures the new decoder's generative validity against that same DiT.
#
# ARCH == v3c (width256, EGT enc+dec 2&2, enc/dec4, latent16) so v3c weights load
# cleanly via --init_weights. beta HELD at 0.1 (no re-warmup): we polish the
# decoder, not relearn the latent.
#
# READ AFTER -------------------------------------------------------------------
# Auto-eval writes runs/polyomics_vae_v3d/{eval_recon,eval_prior,eval_dit}.json.
# SUCCESS = eval_dit validity/torsion improves over v3c's dit numbers WITHOUT
# eval_recon regressing (recon must not pay for robustness). train_out.log logs the
# 'robust' loss term per epoch (should fall toward 0 as the decoder hardens).
#
# LAUNCH (OS-detached, harness idle-kill resistant):
#   $env:POLY3D_PY = "C:/Users/shanu/anaconda3/envs/polygen/python.exe"
#   Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','<repo>/scripts/run_vae_v3d_robust.ps1' -WindowStyle Hidden -PassThru
#   Foreground live: ... -File scripts/run_vae_v3d_robust.ps1 -Live
# Idempotent: if v3d has its own vae_epoch*.pt it --resume's that; else it
#   --init_weights from v3c. A relaunch after a crash continues v3d, never re-inits.
# -----------------------------------------------------------------------------

param(
    [switch]$Live,
    [int]$Width = 256,            # MUST match v3c for the weight load to succeed.
    [int]$Epochs = 15,
    [double]$Lr = 5e-5,           # gentle fine-tune lr.
    [double]$WClash = 5.0,        # keep v3c geometry pressure on the recon output.
    [double]$WBond = 1.5,         # v3c: 1.0 -> 1.5
    [double]$WAngle = 0.75,       # v3c: 0.5 -> 0.75
    [double]$WRobust = 3.0,       # off-manifold guardrail (clash+bond_range on z') weight.
    [double]$RobustNoiseStd = 0.35, # std of z' = z + std*randn (approximates DiT off-manifold gap).
    [string]$InitFrom = "runs/polyomics_vae_v3c/vae_best.pt",
    [string]$OutName = "polyomics_vae_v3d",
    [string]$DitCkpt = "runs/polyomics_dit_v2/dit_best.pt"
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
"START $(Get-Date -Format o) width=$Width epochs=$Epochs lr=$Lr w_clash=$WClash w_bond=$WBond w_angle=$WAngle w_robust=$WRobust robust_noise_std=$RobustNoiseStd freeze_encoder=1" |
    Out-File -Append -Encoding ascii $status

# ---- Fine-tune from v3c (or resume v3d's own ckpt), decoder-only -------------
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
        # off-manifold robustness (decoder-only, encoder frozen -> latent stays v3c)
        "--w_robust",[string]$WRobust,"--robust_noise_std",[string]$RobustNoiseStd,"--freeze_encoder",
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
    # Idempotency: prefer v3d's own latest ckpt (resume); only init from v3c on a fresh start.
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

# ---- Auto-eval: recon + prior (VAE only) + dit (new decoder x existing DiT) --
if (-not (Done "EVAL_DONE")) {
    $ckpt = "$out/vae_best.pt"
    $bestEp = & $py -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" $ckpt 2>$null
    "EVAL_START $(Get-Date -Format o) ckpt=$ckpt best_epoch=$bestEp dit=$DitCkpt" |
        Out-File -Append -Encoding ascii $status

    & $py "scripts/eval_ensemble.py" --checkpoint $ckpt --val_lmdb $val `
        --mode recon --max_atoms 288 --batch_size 32 `
        --out "$out/eval_recon.json" 1> "$out/eval_out.log" 2> "$out/eval_err.log"

    & $py "scripts/eval_ensemble.py" --checkpoint $ckpt --val_lmdb $val `
        --mode prior --max_atoms 288 --batch_size 32 `
        --out "$out/eval_prior.json" 1>> "$out/eval_out.log" 2>> "$out/eval_err.log"

    # KEY measurement: new decoder driven by the EXISTING dit_v2 (off-manifold latents).
    & $py "scripts/eval_ensemble.py" --checkpoint $ckpt --val_lmdb $val `
        --mode dit --dit_checkpoint $DitCkpt --max_atoms 288 --batch_size 32 `
        --out "$out/eval_dit.json" 1>> "$out/eval_out.log" 2>> "$out/eval_err.log"

    "EVAL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

"ALL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
Write-Host ""
Write-Host "Report: $out/vae_log.csv (val_pos not drifting up? robust falling?), $out/eval_dit.json"
Write-Host "SUCCESS = eval_dit validity/torsion beats v3c's dit WITHOUT eval_recon regressing."

[Console]::OutputEncoding = $prevOutEnc
