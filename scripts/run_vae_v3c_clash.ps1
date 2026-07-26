# PolyOmics VAE v3c = LOSS lever (ASCII only for PS 5.1).
# Fine-tune the accepted v3b decoder with a STRONGER geometry loss that directly
# targets what the eval validity gate checks:
#   + clash guardrail (graph-dist>=3 pairs closer than (rvdw_i+rvdw_j)*0.6 are penalized),
#   + heavier bond / angle weights (1.0->1.5 / 0.5->0.75).
# Everything architectural == v3b (width256, EGT enc+dec 2&2, enc/dec4, latent16), so
# v3b weights load cleanly. beta is HELD at 0.1 (no re-warmup) to keep the latent v3b
# already made informative -- we polish geometry, not relearn the latent.
#
# WHY FINE-TUNE, NOT FROM SCRATCH ---------------------------------------------
# The clash term is a POLISH on v3b's residual validity failures (recon pass-rate
# 0.35: most conformers still trip clash/bond). Starting from v3b keeps the clash
# loss small at t=0 (a good model barely clashes) -> stable, no init explosion (the
# width256-from-scratch trap). And it is the cleanest attribution: same weights in,
# add loss pressure, measure whether recon validity rises above v3b's 0.35.
# Cheap: ~15 epochs (~7h) vs a 22h fresh run.
#
# LATENT / RECON SAFETY -------------------------------------------------------
# The dominant pos/bond/angle terms anchor to the (clash-free) GT, and clash only
# pushes in the SAME direction (GT has no clashes), so reconstruction is preserved.
# Watch vae_log.csv val_pos stays ~0.19 (not drifting up) and val_kl stays >0.1.
#
# READ AFTER ------------------------------------------------------------------
# Auto-eval writes runs/polyomics_vae_v3c/{eval_recon.json, eval_prior.json}.
# SUCCESS = recon validity_pass_rate median rises above v3b's 0.35 WITHOUT val_pos
# regressing. train_out.log logs the 'clash' loss term per epoch (should fall toward 0).
# If validity climbs -> adopt v3c, precompute_latents -> DiT. If flat -> loss lever
# is exhausted; next is all-22-class data.
#
# LAUNCH (OS-detached, harness idle-kill resistant):
#   $env:POLY3D_PY = "C:/Users/shanu/anaconda3/envs/polygen/python.exe"
#   Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','<repo>/scripts/run_vae_v3c_clash.ps1' -WindowStyle Hidden -PassThru
#   Foreground live: ... -File scripts/run_vae_v3c_clash.ps1 -Live
# Idempotent: if v3c has its own vae_epoch*.pt it --resume's that; else it --init_weights
#   from v3b. So a relaunch after a crash continues v3c, never re-inits from v3b.
# -----------------------------------------------------------------------------

param(
    [switch]$Live,
    [int]$Width = 256,            # MUST match v3b for the weight load to succeed.
    [int]$Epochs = 15,
    [double]$Lr = 5e-5,           # gentle fine-tune lr (v3b ended ~3e-5).
    [double]$WClash = 5.0,        # clash guardrail weight (hinge, self-limits to ~0).
    [double]$WBond = 1.5,         # 1.0 -> 1.5
    [double]$WAngle = 0.75,       # 0.5 -> 0.75
    [string]$InitFrom = "runs/polyomics_vae_v3b/vae_best.pt",
    [string]$OutName = "polyomics_vae_v3c"
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
"START $(Get-Date -Format o) width=$Width epochs=$Epochs lr=$Lr w_clash=$WClash w_bond=$WBond w_angle=$WAngle" |
    Out-File -Append -Encoding ascii $status

# ---- Fine-tune from v3b (or resume v3c's own ckpt) ---------------------------
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
    # Idempotency: prefer v3c's own latest ckpt (resume); only init from v3b on a fresh start.
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

# ---- Auto-eval: recon + prior on PG_val -------------------------------------
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
Write-Host "Report: $out/vae_log.csv (val_pos ~0.19? val_kl >0.1? clash falling?), $out/eval_recon.json"
Write-Host "SUCCESS = recon validity_pass_rate > v3b 0.35 without val_pos regressing."

[Console]::OutputEncoding = $prevOutEnc
