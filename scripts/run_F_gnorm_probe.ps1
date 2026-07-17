# Gradient-norm probe (ASCII only for PS 5.1). SHORT: ~10 min total, no long training.
#
# WHY --------------------------------------------------------------------------
# Step3 (run_E_wide256, hidden 256) came out WORSE than run_C (hidden 128) on
# EVERY band (large rmsd 2.51->3.61 / 3.12->4.56 / 5.05->6.68). But that is NOT a
# capacity verdict: E's *train* loss was also far worse (C 1.417 -> E 2.937). A
# wider net strictly contains the narrower net's function class, so a capacity
# limit cannot make the train loss worse. E is an OPTIMIZATION failure => the run
# is invalid and must be re-done once the cause is fixed.
#
# Prime suspect: --grad_clip 1.0 (train.py, VAETrainer) clips the GLOBAL grad norm
# over cond_encoder+vae on every optimizer step. If ||g|| >> 1.0 essentially
# always, the update degenerates to
#       delta = -lr * g/||g||            (normalized gradient descent)
# whose step LENGTH is exactly lr regardless of the gradient. Spreading a fixed
# step length over 4x more parameters (hidden 128->256 ~ 4x weights) shrinks the
# per-parameter movement by ~2x => E was really "width x2 AND effective LR /2",
# not the intended one-variable width test. Measured on a toy (tiny16, hidden 32):
# mean ||g|| = 8.65 with clip=1.0, clipped=100% of steps -- clipping is ALWAYS on.
#
# This probe measures ||g|| directly for both configs, in both regimes:
#   *_conv : resume the CONVERGED C/E checkpoints  <- the decision-relevant one.
#            E plateaued from ep16-40; this reads ||g|| in exactly that regime.
#   *_init : fresh init                            <- explains why E was already
#            2.4x behind C at epoch 1 (val_pos 4.59 vs 1.95).
#
# READ THE RESULT --------------------------------------------------------------
#   E clipped ~100% AND mean ||g|| ~2x C's  -> mechanism CONFIRMED. Re-run BOTH C
#       and E with a loose clip (e.g. --grad_clip 10.0). Both must be retaken: C
#       sat under the same throttle, so changing only E breaks comparability.
#   both clipped only a few %              -> clip is innocent. E's stall has a
#       different cause (e.g. lr 3e-4 too hot for width 256); probe LR instead.
# Beyond the width question: "optimization/step-count is the only lever that ever
# moves anything" has been this project's recurring finding (bs16->bs8->bs4 gave
# 0.42->0.39->0.13; 240+ froze the moment LR decayed). Permanent clipping is the
# signature of exactly that -- total distance travelled = n_steps * lr, and the
# loss landscape barely matters. So this probe may be worth far more than Step3.
#
# COST: subset_ratio 0.001 (~4.8k mol) => ~150 optimizer steps per run, ~1-3 min
# each. Nothing here trains anything; the checkpoints written under F_probe/ are
# throwaway. The real C_egt_long40/ and E_wide256/ dirs are NEVER touched.
#
# ---- PORTABILITY (same as run_C_long40.ps1 / run_E_wide256.ps1) ---------------
#   $repo auto-derived from scripts/ -> repo root. Datasets at:
#       <repo>/data/train.lmdb   and   <repo>/data/val.lmdb
#   Set env POLY3D_PY to the polygen env python, e.g.
#       $env:POLY3D_PY = "C:/path/to/envs/polygen/python.exe"
#
# Launch (foreground, watch it live):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_F_gnorm_probe.ps1 -Live
# Launch (quiet, everything to logs):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_F_gnorm_probe.ps1
#
# DO NOT run this while another GPU job is going (past OOM lesson).
#
# REPORTING: paste back runs/gen_v1/F_gnorm.log  (the auto-built summary; it is
# ASCII and already contains only the [GNORM] lines that matter).
# ------------------------------------------------------------------------------

param([switch]$Live)   # -Live: also echo each run's stdout to the console

if ($env:POLY3D_PY) { $py = $env:POLY3D_PY } else { $py = "python" }
# repo root = two levels up from scripts/run_F_gnorm_probe.ps1
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$env:PYTHONUTF8 = "1"
# PS 5.1 decodes a native process's stdout with [Console]::OutputEncoding (cp932 on
# a Japanese Windows) -> UTF-8 bytes garble. Force UTF-8; restore at the end.
$prevOutEnc = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:256"
Set-Location $repo

$base   = "runs/gen_v1"
$probe  = "$base/F_probe"
$status = "$base/status_F.txt"
$summary= "$base/F_gnorm.log"
$ckC    = "$base/C_egt_long40/vae_best.pt"   # hidden 128 converged
$ckE    = "$base/E_wide256/vae_best.pt"      # hidden 256 converged

function Done($tag) {
    return (Test-Path $status) -and (Select-String -Path $status -Pattern $tag -Quiet)
}
# train.py --resume starts at ckpt['epoch']+1 and loops range(start, epochs+1),
# so epochs must be ckpt_epoch+1 to get exactly ONE probe epoch.
function CkptEpoch($ck) {
    $e = & $py -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" $ck
    if ($LASTEXITCODE -ne 0) { return -1 }
    return [int]($e | Select-Object -Last 1)
}

if (-not (Test-Path $probe)) { New-Item -ItemType Directory -Force -Path $probe | Out-Null }
"START $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status

# Everything below is identical to run_C_long40 / run_E_wide256 EXCEPT the tiny
# subset (so one epoch is minutes, not half an hour) and gnorm_log_every.
# batch_size 8 x grad_accum 4 = eff 32 is kept so ||g|| is measured on exactly the
# same effective batch the real runs used -- ||g|| depends on it.
$common = @(
    "--stage","vae",
    "--train_lmdb","data/train.lmdb",
    "--val_lmdb","data/val.lmdb",
    "--lr","3e-4","--lr_min","3e-5",
    "--max_atoms","288",
    "--subset_ratio","0.001","--val_subset_ratio","0.002",
    "--pos_loss_type","kabsch",
    "--batch_size","8","--grad_accum","4",
    "--egt_every","2","--enc_egt_every","2",
    "--empty_cache_every","500",
    "--num_workers","8","--seed","42",
    "--gnorm_log_every","10",
    # The last epoch always saves (train.py:752 "or epoch == args.epochs"), so a
    # throwaway ckpt does land in F_probe/<tag>/. save_every 1000 only keeps the
    # rolling-delete path (train.py:506) from firing. F_probe/ is disposable.
    "--save_every","1000"
)
# The ONLY difference between the two conditions (same as run_C vs run_E).
$dimsC = @("--hidden_dim","128","--edge_dim","64","--vae_hidden_dim","128",
           "--enc_layers","4","--dec_layers","4","--latent_dim","16")
$dimsE = @("--hidden_dim","256","--edge_dim","128","--vae_hidden_dim","256",
           "--enc_layers","4","--dec_layers","4","--latent_dim","16")

function Probe($tag, $dims, $ck) {
    if (Done "${tag}_DONE") { "SKIP $tag (already done)" | Write-Host; return }

    $a = @() + $common + $dims + @("--out_dir","$probe/$tag")
    if ($ck) {
        if (-not (Test-Path $ck)) {
            "${tag}_SKIP missing ckpt $ck $(Get-Date -Format o)" |
                Out-File -Append -Encoding ascii $status
            return
        }
        $ep = CkptEpoch $ck
        if ($ep -lt 1) {
            "${tag}_SKIP unreadable ckpt $ck $(Get-Date -Format o)" |
                Out-File -Append -Encoding ascii $status
            return
        }
        $a += @("--resume",$ck,"--epochs",[string]($ep + 1))
        "${tag}_START $(Get-Date -Format o) resume=$ck ckpt_epoch=$ep" |
            Out-File -Append -Encoding ascii $status
    } else {
        $a += @("--epochs","1")
        "${tag}_START $(Get-Date -Format o) fresh" |
            Out-File -Append -Encoding ascii $status
    }

    if ($Live) {
        & $py "scripts/train.py" @a 2> "$probe/$tag.err" | Tee-Object -FilePath "$probe/$tag.log"
    } else {
        & $py "scripts/train.py" @a 1> "$probe/$tag.log" 2> "$probe/$tag.err"
    }
    $ec = $LASTEXITCODE
    if ($ec -ne 0) {
        "${tag}_FAILED exit=$ec $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
        return
    }
    "${tag}_DONE exit=0 $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

# Converged regime first: this is the one that decides the question.
Probe "c_conv" $dimsC $ckC
Probe "e_conv" $dimsE $ckE
# Fresh init: explains the epoch-1 gap (E val_pos 4.59 vs C 1.95).
Probe "c_init" $dimsC $null
Probe "e_init" $dimsE $null

# ---- Summary (ASCII, this is the only file that needs to be reported) ---------
# Per-run logs are UTF-16LE (PS 5.1 redirection); Select-String decodes them fine.
"=== GNORM PROBE $(Get-Date -Format o) ===" | Out-File -Encoding ascii $summary
"grad_clip is the clip threshold; 'clipped' = %% of optimizer steps that hit it." |
    Out-File -Append -Encoding ascii $summary
"c_* = hidden128 (run_C config), e_* = hidden256 (run_E config)" |
    Out-File -Append -Encoding ascii $summary
"*_conv = resumed from the converged ckpt, *_init = fresh init" |
    Out-File -Append -Encoding ascii $summary
"" | Out-File -Append -Encoding ascii $summary
foreach ($tag in @("c_conv","e_conv","c_init","e_init")) {
    $f = "$probe/$tag.log"
    if (-not (Test-Path $f)) { "--- $tag : NOT RUN" | Out-File -Append -Encoding ascii $summary; continue }
    "--- $tag" | Out-File -Append -Encoding ascii $summary
    # every [GNORM] line: the per-step trace plus the epoch aggregate
    $hits = Select-String -Path $f -Pattern '\[GNORM\]' | ForEach-Object { $_.Line.Trim() }
    if ($hits) { $hits | Out-File -Append -Encoding ascii $summary }
    else { "(no [GNORM] lines -- is train.py at commit 2a294ae or later?)" |
           Out-File -Append -Encoding ascii $summary }
    "" | Out-File -Append -Encoding ascii $summary
}
"ALL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
Get-Content $summary | Write-Host
Write-Host ""
Write-Host "Report this file: $summary"

[Console]::OutputEncoding = $prevOutEnc
