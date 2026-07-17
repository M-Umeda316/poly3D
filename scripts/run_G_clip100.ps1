# Step4 experiment for the 32GB-VRAM Windows machine (ASCII only for PS 5.1).
# GRADIENT-CLIP lever. One variable vs run_C_long40: --grad_clip 1.0 -> 100.
#
# WHY --------------------------------------------------------------------------
# The F probe (run_F_gnorm_probe.ps1) measured the clip actually firing on 100.0%
# of optimizer steps, in EVERY config and EVERY regime:
#     c(hidden128): mean ||g|| ~5 at init -> 47.4 converged   clipped 100%
#     e(hidden256): mean ||g|| ~63 (spikes 97.6) -> 73.4       clipped 100%
# grad_clip defaults to 1.0 (train.py), so EVERY run this project has ever done --
# including run_C, the baseline every verdict was measured against -- fed AdamW
# nothing but the unit vector g/||g||, for all 40 epochs.
#
# What that costs. AdamW's m/(sqrt(v)+eps) is invariant to rescaling g by a
# CONSTANT, so this is NOT "the effective LR shrank" (that intuition is SGD's, and
# it is wrong here). The damage is that the clip factor 1/||g_t|| VARIES per step,
# which erases the relative size ACROSS steps: a heavy batch carrying a 240+ giant
# (||g|| ~97 measured) enters Adam's m/v as the very same "one unit" as an easy
# small-molecule batch (~3). The rare hard examples never get the larger pull they
# earn. And the 1.0 norm budget is zero-sum across the batch, so pushing giants
# necessarily starves the main distribution.
#
# That last sentence is run_D's result verbatim. D (data 0.10 + oversample a=2.0)
# was read as "capacity-limited: pushing giants starves small molecules" -- but a
# fixed per-step norm budget MANUFACTURES exactly that trade-off. Same for the
# project's oldest mystery: "240+ never moves, whatever we throw at it" (budget,
# depth, width, data, oversampling all rejected). If the giants' gradient signal
# was being flattened every single step, none of those levers could have worked.
# So this run re-opens Step1's and Step2's verdicts, not just Step3's.
#
# WHY 100 AND NOT 10 -----------------------------------------------------------
# ||g|| GROWS through training (C: ~5 -> 47.4), so a clip that is loose at init
# goes tight later: --grad_clip 10 would be silent early and then clip ~100% again
# by convergence -- i.e. it would re-run the same broken experiment. 100 sits above
# C's converged mean (47.4) and above the heaviest batch seen anywhere (97.6), so
# typical AND heavy steps pass through untouched while a genuine explosion is still
# caught (keeping a net matters: E showed real spikes). Override with -Clip if the
# [GNORM] lines below show it is still biting.
#
# VERIFY THE INTERVENTION TOOK -------------------------------------------------
# This is the whole point, so the run logs it: [GNORM] ep<N> ... clipped=<pct>%.
#   clipped ~0-20%  -> intervention worked; the comparison vs C is meaningful.
#   clipped still ~100% -> 100 was too low, nothing was tested. Re-run with a
#                          bigger -Clip. DO NOT read the eval in that case.
#
# ---- ONE VARIABLE vs run_C_long40 --------------------------------------------
# Identical: EGT enc+dec 2&2, 40ep, lr 3e-4 sustained lr_min 3e-5, eff bs32
#   (bs8 x ga4), data 0.03 UNIFORM (no oversampling), kabsch, max_atoms 288,
#   hidden 128 / edge 64 / vae_hidden 128 (train.py defaults = what C used),
#   enc/dec 4, latent 16, seed 42, same large_eval set.
# Differs ONLY by: --grad_clip 1.0 -> 100  (+ --gnorm_log_every, a log line, and
#   --empty_cache_every, both provably compute-neutral).
# So cmp_G_large vs cmp_C_large isolates the clip.
#
# ---- PORTABILITY (same as run_C_long40.ps1 / run_E_wide256.ps1) --------------
#   $repo auto-derived from scripts/ -> repo root. Datasets at:
#       <repo>/data/train.lmdb   and   <repo>/data/val.lmdb
#   Set env POLY3D_PY to the polygen env python, e.g.
#       $env:POLY3D_PY = "C:/path/to/envs/polygen/python.exe"
#   Outputs under runs/gen_v1/ (gitignored, local to this machine).
#
# Launch (foreground, WATCH the tqdm bar live on console):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_G_clip100.ps1 -Live
# Launch (foreground, quiet - everything to logs):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_G_clip100.ps1
# Launch (OS-detached, harness idle-kill resistant; no console, so no -Live):
#   Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',
#     '<repo>/scripts/run_G_clip100.ps1' -WindowStyle Hidden -PassThru
# Custom clip:  ... -File scripts/run_G_clip100.ps1 -Clip 300
#
# ~10h (hidden128 is ~2x faster per epoch than E's hidden256). Do not run other
# GPU jobs alongside it (past OOM lesson).
#
# REPORTING (no need to copy files back): paste back
#   1) runs/gen_v1/G_gnorm.log             (auto-built ASCII summary: the per-epoch
#      [GNORM] lines. CHECK THIS FIRST - if clipped is still ~100%, ignore the rest)
#   2) runs/gen_v1/G_clip100/vae_log.csv   (per-epoch train/val loss)
#   3) runs/gen_v1/G_eval.log              (size-binned + endpoint tables, BOTH the
#      "LARGE-MOLECULE EVAL" and the "VAL SAMPLE EVAL" section = main distribution)
# --------------------------------------------------------------------------------

param(
    [switch]$Live,          # -Live: show tqdm bars on console (stderr unredirected)
    [double]$Clip = 100     # grad_clip. See "WHY 100 AND NOT 10" above.
)

if ($env:POLY3D_PY) { $py = $env:POLY3D_PY } else { $py = "python" }
# repo root = two levels up from scripts/run_G_clip100.ps1
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$env:PYTHONUTF8 = "1"
# PS 5.1 decodes a native process's stdout using [Console]::OutputEncoding (cp932 on
# a Japanese Windows) -> UTF-8 bytes get mis-decoded and the log garbles. Force UTF-8
# so the bytes round-trip; restore at the end so the parent console is undisturbed.
$prevOutEnc = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# VRAM fragmentation mitigation (Windows has no expandable_segments). Compute-neutral.
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:256"
Set-Location $repo

$base   = "runs/gen_v1"
$status = "$base/status_G.txt"
$outG   = "$base/G_clip100"
$summary= "$base/G_gnorm.log"
$largeG = "runs/overfit/large_eval.lmdb"   # regenerated from val.lmdb if absent

function Done($tag) {
    return (Test-Path $status) -and (Select-String -Path $status -Pattern $tag -Quiet)
}
function LatestCkpt($dir) {
    if (-not (Test-Path $dir)) { return $null }
    $f = Get-ChildItem -Path $dir -Filter "vae_epoch*.pt" -ErrorAction SilentlyContinue |
         Sort-Object Name | Select-Object -Last 1
    if ($f) { return $f.FullName } else { return $null }
}

# runs/gen_v1 is gitignored; create it on a fresh clone before writing status.
if (-not (Test-Path $base)) { New-Item -ItemType Directory -Force -Path $base | Out-Null }

"START $(Get-Date -Format o) clip=$Clip" | Out-File -Append -Encoding ascii $status

# ---- Train: C config with the clip released ---------------------------------------
if (-not (Done "G_DONE")) {
    $argsG = @(
        "--stage","vae",
        "--train_lmdb","data/train.lmdb",
        "--val_lmdb","data/val.lmdb",
        "--epochs","40",
        "--lr","3e-4","--lr_min","3e-5",
        "--max_atoms","288",
        "--subset_ratio","0.03","--val_subset_ratio","0.02",
        "--pos_loss_type","kabsch",
        # hidden/edge/vae_hidden/enc/dec/latent are left at train.py defaults
        # (128/64/128/4/4/16) = exactly what run_C used. Do not add --hidden_dim
        # here: width is Step3's variable, not this run's.
        "--enc_layers","4","--dec_layers","4","--latent_dim","16",
        "--batch_size","8","--grad_accum","4",
        "--egt_every","2","--enc_egt_every","2",
        "--grad_clip",[string]$Clip,      # <<-- THE ONLY REAL DIFFERENCE vs run_C
        "--gnorm_log_every","200",
        "--empty_cache_every","500",
        "--num_workers","8","--save_every","1","--seed","42",
        "--out_dir",$outG
    )
    $ck = LatestCkpt $outG
    if ($ck) { $argsG += @("--resume",$ck) }
    "G_START $(Get-Date -Format o) resume=$ck grad_clip=$Clip" | Out-File -Append -Encoding ascii $status
    if ($Live) {
        & $py "scripts/train.py" @argsG 1> "$base/G_out.log"
    } else {
        & $py "scripts/train.py" @argsG 1> "$base/G_out.log" 2> "$base/G_err.log"
    }
    $ec = $LASTEXITCODE
    # Only mark G_DONE on SUCCESS. On failure (e.g. OOM), write G_FAILED and exit
    # WITHOUT G_DONE so relaunching re-enters this block and LatestCkpt resumes from
    # the newest vae_epoch*.pt (save_every 1). (run_C had the reverse bug.)
    if ($ec -ne 0) {
        "G_FAILED exit=$ec $(Get-Date -Format o) - relaunch to resume from latest ckpt" |
            Out-File -Append -Encoding ascii $status
        exit $ec
    }
    "G_DONE exit=0 $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

# ---- Did the intervention actually take? ------------------------------------------
# Built before the eval on purpose: if clipped is still ~100% the eval is moot.
# G_out.log is UTF-16LE (PS 5.1 redirection); Select-String decodes it fine.
"=== GNORM (run_G, grad_clip=$Clip) $(Get-Date -Format o) ===" | Out-File -Encoding ascii $summary
"clipped ~0-20%     -> clip released, cmp_G vs cmp_C is a valid one-variable test." |
    Out-File -Append -Encoding ascii $summary
"clipped still ~100% -> Clip too low, nothing was tested. Re-run with a bigger -Clip." |
    Out-File -Append -Encoding ascii $summary
"(run_C baseline for reference: mean ||g|| ~5 at init -> 47.4 converged, clipped 100%)" |
    Out-File -Append -Encoding ascii $summary
"" | Out-File -Append -Encoding ascii $summary
if (Test-Path "$base/G_out.log") {
    $hits = Select-String -Path "$base/G_out.log" -Pattern '\[GNORM\] ep' |
            ForEach-Object { $_.Line.Trim() }
    if ($hits) { $hits | Out-File -Append -Encoding ascii $summary }
    else { "(no [GNORM] ep lines -- is train.py at commit 2a294ae or later?)" |
           Out-File -Append -Encoding ascii $summary }
}
Get-Content $summary | Write-Host

# ---- Auto-eval (GPU free now; no concurrent processes) ----------------------------
# Reuses the SAME large_eval.lmdb C/D/E built, so cmp_G_large is directly comparable
# to cmp_C_large (identical molecules, first 64 == tiny_large).
if (-not (Done "EVAL_DONE")) {
    $ckG = "$outG/vae_best.pt"

    if (-not (Test-Path $largeG)) {
        "MAKE_LARGE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
        & $py "scripts/make_tiny_lmdb.py" --src "data/val.lmdb" --dst $largeG `
            --n 300 --min_atoms 130 --max_atoms 300 1>> "$base/G_eval.log" 2>> "$base/G_err.log"
    }

    "EVAL_START $(Get-Date -Format o) ckpt=$ckG" | Out-File -Append -Encoding ascii $status
    "=== LARGE-MOLECULE EVAL (val-derived, >=130 atoms) ===" | Out-File -Append -Encoding ascii "$base/G_eval.log"
    & $py "scripts/eval_by_size.py" --checkpoint $ckG --val_lmdb $largeG `
        --max_atoms 300 --batch_size 8 --num_workers 4 `
        --out "$base/cmp_G_large.json" 1>> "$base/G_eval.log" 2>> "$base/G_err.log"

    "`n=== VAL SAMPLE EVAL (data/val.lmdb, first 640 mol) ===" | Out-File -Append -Encoding ascii "$base/G_eval.log"
    & $py "scripts/eval_by_size.py" --checkpoint $ckG --val_lmdb "data/val.lmdb" `
        --max_atoms 288 --batch_size 8 --num_workers 4 --max_batches 80 `
        --out "$base/cmp_G_valgen.json" 1>> "$base/G_eval.log" 2>> "$base/G_err.log"

    "EVAL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

"ALL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
Write-Host ""
Write-Host "Report: $summary (check clipped% FIRST), $outG/vae_log.csv, $base/G_eval.log"

# Restore the parent console's output encoding (no-op if run detached).
[Console]::OutputEncoding = $prevOutEnc
