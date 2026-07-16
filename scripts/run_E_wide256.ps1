# Step 3 experiment for the 32GB-VRAM Windows machine (ASCII only for PS 5.1).
# CAPACITY lever. Step1 (run_C: budget) and Step2 (run_D: data exposure +
# oversampling) both FAILED to crack the 240+ internal RMSD, and run_D showed the
# tell-tale of a capacity bottleneck: oversampling the giant tail cannibalized the
# main distribution (val_pos 0.965 -> 1.22, all large bands' median rmsd regressed;
# only the 240+ p90 outliers dropped). So the model can't hold both the hard tail
# AND the main dist at hidden=128 -> widen it.
#
# This is a CLEAN one-variable comparison vs run_C_long40: EVERYTHING identical
#   (EGT enc+dec 2&2, 40ep, sustained LR lr_min 3e-5, eff bs32, data 0.03 UNIFORM,
#    kabsch, max_atoms 288, seed 42, same large_eval set)
# except the width:
#   --hidden_dim     128 -> 256
#   --edge_dim        64 -> 128
#   --vae_hidden_dim 128 -> 256   (A2_wide / A4_large "wide256" config)
# enc/dec layers stay 4, latent_dim stays 16. NO oversampling (run_D proved it hurts
# the deployment target). So cmp_E_large vs cmp_C_large isolates the capacity effect.
#
# ---- PORTABILITY (same as run_C_long40.ps1 / run_D_over.ps1) -------------------
#   $repo auto-derived from scripts/ -> repo root. Datasets at:
#       <repo>/data/train.lmdb   and   <repo>/data/val.lmdb
#   Set env POLY3D_PY to the polygen env python, e.g.
#       $env:POLY3D_PY = "C:/path/to/envs/polygen/python.exe"
#   Outputs under runs/gen_v1/ (gitignored, local to this machine).
# --------------------------------------------------------------------------------
#
# Launch (foreground, WATCH the tqdm bar live on console):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_E_wide256.ps1 -Live
# Launch (foreground, quiet - everything to logs):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_E_wide256.ps1
# Launch (OS-detached, harness idle-kill resistant; no console, so no -Live):
#   Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',
#     '<repo>/scripts/run_E_wide256.ps1' -WindowStyle Hidden -PassThru
#
# REPORTING (no need to copy files back): paste back
#   1) runs/gen_v1/E_wide256/vae_log.csv   (per-epoch train/val loss)
#   2) runs/gen_v1/E_eval.log              (size-binned + endpoint tables, BOTH the
#      "LARGE-MOLECULE EVAL" and the "VAL SAMPLE EVAL" section = main distribution)

param([switch]$Live)   # -Live: show tqdm bars on console (stderr unredirected)

if ($env:POLY3D_PY) { $py = $env:POLY3D_PY } else { $py = "python" }
# repo root = two levels up from scripts/run_E_wide256.ps1
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$env:PYTHONUTF8 = "1"
# PS 5.1 decodes a native process's stdout using [Console]::OutputEncoding (cp932 on
# a Japanese Windows) -> UTF-8 bytes get mis-decoded and the log garbles. Force UTF-8
# so the bytes round-trip; restore at the end so the parent console is undisturbed.
$prevOutEnc = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# VRAM fragmentation mitigation (Windows has no expandable_segments). Harmless on 32GB.
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:256"
Set-Location $repo

$base   = "runs/gen_v1"
$status = "$base/status_E.txt"
$outE   = "$base/E_wide256"
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

# ---- Train: wide256 + enc+dec EGT, 40ep sustained LR, data 0.03 uniform ------------
# eff batch 32 (bs8 x ga4) = identical to run_C so C vs E isolates the width change.
# wide256 ~2x the activations of hidden128, but MEASURED real usage is tiny: on the
# 32GB box the training peak was alloc=4.13GB. The problem was RESERVED memory: it
# ballooned to 18.66GB in epoch 1 and kept climbing each epoch = pure fragmentation
# (Windows has no expandable_segments, and the EGT dist_bias buffer changes size with
# every batch's max_n so the caching allocator never reuses blocks). Fix = periodic
# --empty_cache_every 500 (releases the fragmented cache back; harmless to the math
# since only ~4GB is ever live). If reserved still creeps, lower it to 200.
# bs8 x ga4 stays identical to run_C so C vs E isolates the width change. bs4 x ga8
# or bs16 x ga2 are also eff 32 if ever needed, but with empty_cache bs8 fits easily.
if (-not (Done "E_DONE")) {
    $argsE = @(
        "--stage","vae",
        "--train_lmdb","data/train.lmdb",
        "--val_lmdb","data/val.lmdb",
        "--epochs","40",
        "--lr","3e-4","--lr_min","3e-5",
        "--max_atoms","288",
        "--subset_ratio","0.03","--val_subset_ratio","0.02",
        "--pos_loss_type","kabsch",
        "--hidden_dim","256","--edge_dim","128","--vae_hidden_dim","256",
        "--enc_layers","4","--dec_layers","4","--latent_dim","16",
        "--batch_size","8","--grad_accum","4",
        "--egt_every","2","--enc_egt_every","2",
        "--empty_cache_every","500",
        "--num_workers","8","--save_every","1","--seed","42",
        "--out_dir",$outE
    )
    $ck = LatestCkpt $outE
    if ($ck) { $argsE += @("--resume",$ck) }
    "E_START $(Get-Date -Format o) resume=$ck width=256" | Out-File -Append -Encoding ascii $status
    if ($Live) {
        & $py "scripts/train.py" @argsE 1> "$base/E_out.log"
    } else {
        & $py "scripts/train.py" @argsE 1> "$base/E_out.log" 2> "$base/E_err.log"
    }
    $ec = $LASTEXITCODE
    # Only mark E_DONE on SUCCESS. On failure (e.g. OOM), write E_FAILED and exit
    # WITHOUT E_DONE so relaunching re-enters this block and LatestCkpt resumes from
    # the newest vae_epoch*.pt (save_every 1). (run_C had the reverse bug; fixed here
    # as in run_D_over.ps1.)
    if ($ec -ne 0) {
        "E_FAILED exit=$ec $(Get-Date -Format o) - relaunch to resume from latest ckpt" |
            Out-File -Append -Encoding ascii $status
        exit $ec
    }
    "E_DONE exit=0 $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

# ---- Auto-eval (GPU free now; no concurrent processes) ----------------------------
# Reuses the SAME large_eval.lmdb C/D built, so cmp_E_large is directly comparable to
# cmp_C_large / cmp_D_large (identical molecules, first 64 == tiny_large).
if (-not (Done "EVAL_DONE")) {
    $ckE = "$outE/vae_best.pt"

    if (-not (Test-Path $largeE)) {
        "MAKE_LARGE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
        & $py "scripts/make_tiny_lmdb.py" --src "data/val.lmdb" --dst $largeE `
            --n 300 --min_atoms 130 --max_atoms 300 1>> "$base/E_eval.log" 2>> "$base/E_err.log"
    }

    "EVAL_START $(Get-Date -Format o) ckpt=$ckE" | Out-File -Append -Encoding ascii $status
    "=== LARGE-MOLECULE EVAL (val-derived, >=130 atoms) ===" | Out-File -Append -Encoding ascii "$base/E_eval.log"
    & $py "scripts/eval_by_size.py" --checkpoint $ckE --val_lmdb $largeE `
        --max_atoms 300 --batch_size 8 --num_workers 4 `
        --out "$base/cmp_E_large.json" 1>> "$base/E_eval.log" 2>> "$base/E_err.log"

    "`n=== VAL SAMPLE EVAL (data/val.lmdb, first 640 mol) ===" | Out-File -Append -Encoding ascii "$base/E_eval.log"
    & $py "scripts/eval_by_size.py" --checkpoint $ckE --val_lmdb "data/val.lmdb" `
        --max_atoms 288 --batch_size 8 --num_workers 4 --max_batches 80 `
        --out "$base/cmp_E_valgen.json" 1>> "$base/E_eval.log" 2>> "$base/E_err.log"

    "EVAL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

"ALL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status

# Restore the parent console's output encoding (no-op if run detached).
[Console]::OutputEncoding = $prevOutEnc
