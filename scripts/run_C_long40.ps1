# Step 1 experiment for the 32GB-VRAM Windows machine (ASCII only for PS 5.1).
# Fresh B(EGT) with a LONGER, sustained-LR schedule to decide whether the 240+
# internal rmsd is optimization-budget limited or data/capacity saturated.
# Same 3% data as gen_v1 B, but epochs 20->40 and lr_min 1.5e-5 -> 3e-5 (LR not
# floored early). Fresh out_dir (do NOT resume the 16GB B: its scheduler is floored).
# Idempotent: relaunch resumes from the latest checkpoint. Sequential single-GPU only.
#
# ---- PORTABILITY (edit these for the other machine) ----------------------------
#   $repo is auto-derived from this script's location (scripts/ -> repo root),
#   so just clone the repo and place the datasets at:
#       <repo>/data/train.lmdb   and   <repo>/data/val.lmdb
#   Python: set env POLY3D_PY to the polygen env python, e.g.
#       $env:POLY3D_PY = "C:/path/to/envs/polygen/python.exe"
#   otherwise it falls back to "python" (activate the polygen env first).
#   num_workers 8 is safe; raise to 16 if the machine has many CPU cores.
#   Outputs (checkpoints, logs, eval json) are written under runs/gen_v1/ which is
#   gitignored - they stay local to whichever machine runs this.
# --------------------------------------------------------------------------------
#
# Launch (foreground):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_C_long40.ps1
# Launch (OS-detached, harness idle-kill resistant):
#   Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',
#     '<repo>/scripts/run_C_long40.ps1' -WindowStyle Hidden -PassThru
#
# REPORTING (no need to copy files back): paste back
#   1) runs/gen_v1/C_egt_long40/vae_log.csv          (per-epoch train/val loss)
#   2) runs/gen_v1/C_eval.log                         (final size-binned + endpoint tables)

if ($env:POLY3D_PY) { $py = $env:POLY3D_PY } else { $py = "python" }
# repo root = two levels up from scripts/run_C_long40.ps1
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$env:PYTHONUTF8 = "1"
# VRAM fragmentation mitigation (Windows has no expandable_segments). Harmless on 32GB.
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:256"
Set-Location $repo

$base   = "runs/gen_v1"
$status = "$base/status_C.txt"
$outC   = "$base/C_egt_long40"
$largeE = "runs/overfit/large_eval.lmdb"   # regenerated from val.lmdb if absent

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

# ---- Train: enc+dec EGT, 40ep sustained LR, bs8 x ga4 = eff 32 (32GB headroom) ----
# eff batch kept at 32 for LR comparability with gen_v1 B. On 32GB you may try
# bs16 x ga2 (also eff 32, fewer accum steps = faster) if VRAM stays under budget.
if (-not (Done "C_DONE")) {
    $argsC = @(
        "--stage","vae",
        "--train_lmdb","data/train.lmdb",
        "--val_lmdb","data/val.lmdb",
        "--epochs","40",
        "--lr","3e-4","--lr_min","3e-5",
        "--max_atoms","288",
        "--subset_ratio","0.03","--val_subset_ratio","0.02",
        "--pos_loss_type","kabsch",
        "--batch_size","8","--grad_accum","4",
        "--egt_every","2","--enc_egt_every","2",
        "--num_workers","8","--save_every","1","--seed","42",
        "--out_dir",$outC
    )
    $ck = LatestCkpt $outC
    if ($ck) { $argsC += @("--resume",$ck) }
    "C_START $(Get-Date -Format o) resume=$ck" | Out-File -Append -Encoding ascii $status
    & $py "scripts/train.py" @argsC 1> "$base/C_out.log" 2> "$base/C_err.log"
    $ec = $LASTEXITCODE
    "C_DONE exit=$ec $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
    if ($ec -ne 0) { "TRAIN_FAILED - skipping eval" | Out-File -Append -Encoding ascii $status; exit $ec }
}

# ---- Auto-eval (GPU free now; no concurrent processes) ----------------------------
if (-not (Done "EVAL_DONE")) {
    $ckC = "$outC/vae_best.pt"

    # (a) Large-molecule yardstick. Regenerate from val.lmdb if not present so the
    #     240+ signal is solid regardless of which files came over. --min_atoms 130
    #     --n 64 reproduces the original tiny_large (first-64 >=130 atoms, deterministic);
    #     we take n=300 for better large-band statistics (its first 64 == tiny_large).
    if (-not (Test-Path $largeE)) {
        "MAKE_LARGE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
        & $py "scripts/make_tiny_lmdb.py" --src "data/val.lmdb" --dst $largeE `
            --n 300 --min_atoms 130 --max_atoms 300 1>> "$base/C_eval.log" 2>> "$base/C_err.log"
    }

    "EVAL_START $(Get-Date -Format o) ckpt=$ckC" | Out-File -Append -Encoding ascii $status
    "=== LARGE-MOLECULE EVAL (val-derived, >=130 atoms) ===" | Out-File -Append -Encoding ascii "$base/C_eval.log"
    & $py "scripts/eval_by_size.py" --checkpoint $ckC --val_lmdb $largeE `
        --max_atoms 300 --batch_size 8 --num_workers 4 `
        --out "$base/cmp_C_large.json" 1>> "$base/C_eval.log" 2>> "$base/C_err.log"

    "`n=== VAL SAMPLE EVAL (data/val.lmdb, first 640 mol) ===" | Out-File -Append -Encoding ascii "$base/C_eval.log"
    & $py "scripts/eval_by_size.py" --checkpoint $ckC --val_lmdb "data/val.lmdb" `
        --max_atoms 288 --batch_size 8 --num_workers 4 --max_batches 80 `
        --out "$base/cmp_C_valgen.json" 1>> "$base/C_eval.log" 2>> "$base/C_err.log"

    "EVAL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

"ALL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
