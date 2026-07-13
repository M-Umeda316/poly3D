# Step 2 experiment for the 32GB-VRAM Windows machine (ASCII only for PS 5.1).
# Step 1 (run_C_long40.ps1) verdict: doubling budget (20->40ep) + sustained LR
# (lr_min 3e-5, not floored) did NOT crack the 240+ internal RMSD (still ~5A,
# 0% success<1A, p90 17.8). So the wall is NOT optimization budget -> it is
# data exposure / capacity. This Step 2 attacks the data-exposure side:
#   (1) subset_ratio 0.03 -> 0.10  (more molecules overall, incl. more 240+)
#   (2) atom-count-weighted LARGE-MOLECULE OVERSAMPLING (--oversample_alpha)
#       so the rare 240+ band (<1% of val dist) actually gets gradient.
# Everything else identical to Step 1 (EGT enc+dec, 40ep, sustained LR, eff bs32)
# so C vs D is a clean data-lever comparison on the SAME large_eval set.
#
# ---- PORTABILITY (same as run_C_long40.ps1) -----------------------------------
#   $repo auto-derived from scripts/ -> repo root. Datasets at:
#       <repo>/data/train.lmdb   and   <repo>/data/val.lmdb
#   Set env POLY3D_PY to the polygen env python, e.g.
#       $env:POLY3D_PY = "C:/path/to/envs/polygen/python.exe"
#   Outputs under runs/gen_v1/ (gitignored, local to this machine).
# --------------------------------------------------------------------------------
#
# ---- THE ONE KNOB TO CONSIDER: $alpha ------------------------------------------
#   weight_i = natoms_i ^ alpha  (WeightedRandomSampler, with replacement).
#   alpha=0  -> uniform (no oversampling). alpha up -> big molecules drawn more.
#   alpha=2.0 (default) is a moderate boost that protects the main <130 dist
#   (the deployment target) while giving 240+ real exposure. On launch the log
#   prints "oversampling: alpha=.. exp_large=.." = expected fraction of draws
#   that are >=130 atoms. If exp_large is tiny (<~10%) raise alpha; if it
#   dominates (>~35%, starving the main dist) lower it.
# --------------------------------------------------------------------------------
#
# Launch (foreground, WATCH the tqdm bar live on console):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_D_over.ps1 -Live
# Launch (foreground, quiet - everything to logs):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_D_over.ps1
# Launch (OS-detached, harness idle-kill resistant; no console, so no -Live):
#   Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',
#     '<repo>/scripts/run_D_over.ps1' -WindowStyle Hidden -PassThru
#
# -Live: streams stderr (the tqdm progress bar + any traceback) to the console so
#   you can watch progress. stdout summaries + vae_log.csv + status are STILL
#   written to files, so reporting is unaffected. Without -Live, stderr goes to
#   D_err.log (and train.py auto-hides the bar on non-TTY = no CR spam in the log).
#
# REPORTING (no need to copy files back): paste back
#   1) runs/gen_v1/D_egt_over/vae_log.csv   (per-epoch train/val loss)
#   2) runs/gen_v1/D_eval.log               (size-binned + endpoint tables)
#      plus the "oversampling: alpha=.. exp_large=.." line from D_out.log

param([switch]$Live)   # -Live: show tqdm bars on console (stderr unredirected)

if ($env:POLY3D_PY) { $py = $env:POLY3D_PY } else { $py = "python" }
# repo root = two levels up from scripts/run_D_over.ps1
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$env:PYTHONUTF8 = "1"
# train.py etc. emit UTF-8 (PYTHONUTF8 + stdout.reconfigure). PS 5.1 decodes a
# native process's stdout using [Console]::OutputEncoding, which defaults to cp932
# on a Japanese Windows -> UTF-8 bytes get mis-decoded as Shift-JIS -> the log is
# garbled (then written as UTF-16 by '>'). Force UTF-8 on the PS decode side so the
# bytes round-trip. Restored at the end so we don't disturb the parent console.
$prevOutEnc = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# VRAM fragmentation mitigation (Windows has no expandable_segments). Harmless on 32GB.
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:256"
Set-Location $repo

$alpha  = "2.0"                            # <-- the one knob (see header)
$base   = "runs/gen_v1"
$status = "$base/status_D.txt"
$outD   = "$base/D_egt_over"
$largeE = "runs/overfit/large_eval.lmdb"   # regenerated from val.lmdb if absent
$sizes  = "data/train.lmdb.sizes.npy"

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

"START $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status

# ---- Prereq: size index for train.lmdb (one full scan, idempotent) ----------------
# --oversample_alpha needs <train_lmdb>.sizes.npy. Skips if already present.
if (-not (Test-Path $sizes)) {
    "SIZEIDX_START $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
    if ($Live) {
        & $py "scripts/build_size_index.py" --src "data/train.lmdb"
    } else {
        & $py "scripts/build_size_index.py" --src "data/train.lmdb" 1> "$base/D_sizeidx.log" 2>&1
    }
    "SIZEIDX_DONE exit=$LASTEXITCODE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
    if (-not (Test-Path $sizes)) {
        "SIZEIDX_FAILED - aborting" | Out-File -Append -Encoding ascii $status; exit 1
    }
}

# ---- Train: enc+dec EGT, 40ep sustained LR, data 0.10 + large oversampling --------
# eff batch 32 (bs8 x ga4) = identical to Step 1 (run_C_long40) => clean C vs D
# data-lever comparison. Single-GPU only (oversampling is DDP-incompatible).
# LESSON (2026-07-12): bs16 fit a FRESH giant batch (~6.4GB) but OOM'd in epoch 1
# on the 16GB box - oversampling packs heavy large-mol batches and Windows lacks
# expandable_segments, so VRAM fragments over ~1000 steps until a heavy batch can't
# allocate, then the allocator wedges (empty_cache fails). bs8 keeps the per-batch
# peak ~5GB with headroom; run_C proved bs8 x ga4 completes 40ep on the 32GB box.
# If it still OOMs late, drop to bs4 x ga8 (gen_v1 B reached epoch 13 at bs4).
if (-not (Done "D_DONE")) {
    $argsD = @(
        "--stage","vae",
        "--train_lmdb","data/train.lmdb",
        "--val_lmdb","data/val.lmdb",
        "--epochs","40",
        "--lr","3e-4","--lr_min","3e-5",
        "--max_atoms","288",
        "--subset_ratio","0.10","--val_subset_ratio","0.02",
        "--oversample_alpha",$alpha,
        "--pos_loss_type","kabsch",
        "--batch_size","8","--grad_accum","4",
        "--egt_every","2","--enc_egt_every","2",
        "--num_workers","8","--save_every","1","--seed","42",
        "--out_dir",$outD
    )
    $ck = LatestCkpt $outD
    if ($ck) { $argsD += @("--resume",$ck) }
    "D_START $(Get-Date -Format o) resume=$ck alpha=$alpha" | Out-File -Append -Encoding ascii $status
    if ($Live) {
        # stdout summaries + vae_log.csv still logged; stderr (tqdm bar + any
        # traceback) streams to the console so you can watch it live.
        & $py "scripts/train.py" @argsD 1> "$base/D_out.log"
    } else {
        & $py "scripts/train.py" @argsD 1> "$base/D_out.log" 2> "$base/D_err.log"
    }
    $ec = $LASTEXITCODE
    # Only mark D_DONE on SUCCESS. On failure (e.g. OOM), write D_FAILED and exit
    # WITHOUT D_DONE, so relaunching re-enters this block and LatestCkpt resumes
    # from the newest vae_epoch*.pt (save_every 1). Writing D_DONE on failure would
    # make Done("D_DONE") skip training on relaunch and jump to eval a missing ckpt.
    if ($ec -ne 0) {
        "D_FAILED exit=$ec $(Get-Date -Format o) - relaunch to resume from latest ckpt" |
            Out-File -Append -Encoding ascii $status
        exit $ec
    }
    "D_DONE exit=0 $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

# ---- Auto-eval (GPU free now; no concurrent processes) ----------------------------
# Reuses the SAME large_eval.lmdb Step 1 built, so cmp_D_large is directly
# comparable to cmp_C_large (identical molecules, first 64 == tiny_large).
if (-not (Done "EVAL_DONE")) {
    $ckD = "$outD/vae_best.pt"

    if (-not (Test-Path $largeE)) {
        "MAKE_LARGE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
        & $py "scripts/make_tiny_lmdb.py" --src "data/val.lmdb" --dst $largeE `
            --n 300 --min_atoms 130 --max_atoms 300 1>> "$base/D_eval.log" 2>> "$base/D_err.log"
    }

    "EVAL_START $(Get-Date -Format o) ckpt=$ckD" | Out-File -Append -Encoding ascii $status
    "=== LARGE-MOLECULE EVAL (val-derived, >=130 atoms) ===" | Out-File -Append -Encoding ascii "$base/D_eval.log"
    & $py "scripts/eval_by_size.py" --checkpoint $ckD --val_lmdb $largeE `
        --max_atoms 300 --batch_size 8 --num_workers 4 `
        --out "$base/cmp_D_large.json" 1>> "$base/D_eval.log" 2>> "$base/D_err.log"

    "`n=== VAL SAMPLE EVAL (data/val.lmdb, first 640 mol) ===" | Out-File -Append -Encoding ascii "$base/D_eval.log"
    & $py "scripts/eval_by_size.py" --checkpoint $ckD --val_lmdb "data/val.lmdb" `
        --max_atoms 288 --batch_size 8 --num_workers 4 --max_batches 80 `
        --out "$base/cmp_D_valgen.json" 1>> "$base/D_eval.log" 2>> "$base/D_err.log"

    "EVAL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

"ALL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status

# Restore the parent console's output encoding (no-op if run detached).
[Console]::OutputEncoding = $prevOutEnc
