# PolyOmics VAE v3e = DiT-CONSISTENCY lever (ASCII only for PS 5.1).
# Fine-tune the accepted v3c decoder so it stays valid on the REAL DiT latents.
#
# WHY --------------------------------------------------------------------------
# v3d hardened the decoder against ISOTROPIC noise (z' = z + 0.35*randn) and it
# WORKED for isotropic (large-unit iso validity 0.33 -> 0.50). But the REAL DiT
# latents are structurally DIFFERENT from isotropic noise and stayed far worse
# (large-unit dit 0.03 -> 0.10, nowhere near the iso 0.50). Diagnosis: the DiT
# off-manifold error is CORRELATED across atoms (a structural distortion), which
# isotropic noise cannot cover. Fix: during fine-tune, ALSO decode the ACTUAL DiT
# latent (built by N(0,I) -> ODE, GT-free) and put ONLY guardrail losses
# (clash + bond_range) on that output. The decoder learns to produce valid
# geometry even for the structurally-broken latents the DiT actually emits.
# GOAL: push large-unit dit validity 0.10 -> toward the recon ceiling ~0.50.
#
# ENCODER IS FROZEN ------------------------------------------------------------
# We freeze cond_encoder + vae.encoder and train the DECODER ONLY. The latent
# space stays identical to v3c, so the EXISTING dit_v2 (width256/hidden256) is
# reused with NO DiT retrain and NO latent recompute -- and the same dit_v2 both
# (a) supplies the consistency latents during training and (b) is the --mode dit
# eval target below. isotropic robustness (--w_robust) is kept ON alongside the
# DiT-consistency term so we do not lose the iso gains v3d already secured.
#
# ARCH == v3c (width256, EGT enc+dec 2&2, enc/dec4, latent16) so v3c weights load
# cleanly via --init_weights. beta HELD at 0.1 (no re-warmup): polish the decoder.
#
# READ AFTER -------------------------------------------------------------------
# Auto-eval uses the FINAL-EPOCH ckpt (NOT vae_best.pt): the v3d lesson was that
# val loss carries no robust/ditcons term, so an under-trained early epoch got
# mis-picked as "best". It writes runs/polyomics_vae_v3e/eval_{recon,dit}_final.json.
# SUCCESS = eval_dit validity/torsion improves over v3c/v3d dit numbers WITHOUT
# eval_recon regressing. train_out.log logs the 'ditcons' loss per epoch
# (should fall toward 0 as the decoder hardens to the real DiT latents).
#
# LAUNCH (OS-detached, harness idle-kill resistant):
#   $env:POLY3D_PY = "C:/Users/shanu/anaconda3/envs/polygen/python.exe"
#   Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','<repo>/scripts/run_vae_v3e_ditcons.ps1' -WindowStyle Hidden -PassThru
#   Foreground live: ... -File scripts/run_vae_v3e_ditcons.ps1 -Live
# Idempotent: if v3e has its own vae_epoch*.pt it --resume's that; else it
#   --init_weights from v3c. A relaunch after a crash continues v3e, never re-inits.
# -----------------------------------------------------------------------------

param(
    [switch]$Live,
    [int]$Width = 256,            # MUST match v3c for the weight load to succeed.
    [int]$Epochs = 10,
    [double]$Lr = 5e-5,           # gentle fine-tune lr.
    [double]$WClash = 5.0,        # keep v3c geometry pressure on the recon output.
    [double]$WBond = 1.5,         # v3c: 1.0 -> 1.5
    [double]$WAngle = 0.75,       # v3c: 0.5 -> 0.75
    [double]$WRobust = 3.0,       # isotropic off-manifold guardrail (keep v3d gains).
    [double]$RobustNoiseStd = 0.35,
    [double]$WDitcons = 3.0,      # THE LEVER: guardrail on REAL DiT latent decode.
    [int]$DitconsSteps = 20,      # ODE steps for the per-batch DiT latent sampling.
    [string]$InitFrom = "runs/polyomics_vae_v3c/vae_best.pt",
    [string]$OutName = "polyomics_vae_v3e",
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
"START $(Get-Date -Format o) width=$Width epochs=$Epochs lr=$Lr w_clash=$WClash w_bond=$WBond w_angle=$WAngle w_robust=$WRobust robust_noise_std=$RobustNoiseStd w_ditcons=$WDitcons ditcons_steps=$DitconsSteps dit=$DitCkpt freeze_encoder=1" |
    Out-File -Append -Encoding ascii $status

# ---- Fine-tune from v3c (or resume v3e's own ckpt), decoder-only -------------
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
        # THE LEVER: DiT-consistency guardrail on the real DiT latent decode
        "--w_ditcons",[string]$WDitcons,"--ditcons_steps",[string]$DitconsSteps,"--vae_dit_checkpoint",$DitCkpt,
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
    # Idempotency: prefer v3e's own latest ckpt (resume); only init from v3c on a fresh start.
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
# v3d lesson: val loss lacks robust/ditcons terms, so an under-trained early epoch
# gets mis-picked as best. Use the highest-numbered vae_epoch*.pt instead.
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
Write-Host "SUCCESS = eval_dit_final validity/torsion beats v3c/v3d dit WITHOUT eval_recon regressing."

[Console]::OutputEncoding = $prevOutEnc
