# Step5 experiment for the 32GB-VRAM Windows machine (ASCII only for PS 5.1).
# CAPACITY lever, RE-RUN with a stable init. One variable vs run_E_wide256:
#   add --warmup_steps (LR warmup). Everything else == run_E == run_C + width256.
#
# WHY --------------------------------------------------------------------------
# Step3 (run_E_wide256, hidden 256) was declared INVALID, not a capacity verdict.
# The [GNORM] probe (run_F) measured E's gradient EXPLODING at init: mean ||g||
# went 6 -> 97.6 over the first ~40 optimizer steps (10x run_C's init, which stayed
# ~5). E's val_pos hit its best at epoch 7 (1.900), then diverged and flat-lined --
# it fell into a bad basin in the first few dozen steps and never climbed out. So E
# never tested "does width 256 help"; it tested "does width 256 x lr 3e-4 blow up at
# init" (yes). The width/capacity lever is therefore still UNJUDGED.
#
# Step4 (run_G) separately closed the clip question: releasing grad_clip 1.0 -> 100
# (clipped 100% -> 0.7%) did NOT move the 240+ band (5.048 -> 5.302, still 0%
# success). So clip was not the bottleneck, and Step1(budget)/Step2(data) verdicts
# stand. That leaves WIDTH as the one live lever we have never cleanly tested.
#
# THE FIX: LR warmup. The explosion is confined to the first ~40 optimizer steps, so
# we ramp lr from lr*warmup_start_factor (default 0.01 = 3e-6) linearly up to the
# rated lr over --warmup_steps OPTIMIZER steps (not epochs -- 1 epoch is ~4.5k steps,
# far too coarse). With lr tiny through the danger zone the giant-batch gradients
# can't run away, and by the time lr reaches 3e-4 the parameters are past the fragile
# init. This is train.py's verified warmup (cosine body untouched; warmup_steps=0 is
# bit-identical to the old path, so run_C/E remain comparable).
#
# VERIFY THE INTERVENTION TOOK -------------------------------------------------
# This is the whole point, so the run logs it via --gnorm_log_every: the early
# [GNORM] optstep lines carry norm=.. and lr=.. .
#   norm stays bounded early (say < ~30) while lr ramps  -> explosion tamed, the
#       width-256 comparison below is meaningful.
#   norm still spikes toward ~97 in the first tens of steps -> warmup too short or lr
#       too high. Re-run with a bigger -WarmupSteps, or drop -Lr to 1.5e-4. In that
#       case the width verdict is NOT yet valid (same trap as E). See H_gnorm.log.
#
# ---- ONE VARIABLE vs run_E_wide256 -------------------------------------------
# Identical to run_E: EGT enc+dec 2&2, 40ep, lr 3e-4 sustained lr_min 3e-5, eff bs32
#   (bs8 x ga4), data 0.03 UNIFORM (no oversampling), kabsch, max_atoms 288,
#   hidden 256 / edge 128 / vae_hidden 256, enc/dec 4, latent 16, seed 42,
#   same large_eval set, --empty_cache_every 500.
# Differs ONLY by: + --warmup_steps 200 (+ --gnorm_log_every, a log line). So
#   cmp_H_large vs cmp_E_large isolates "did warmup rescue the width-256 run".
#
# ---- THE CAPACITY QUESTION (H vs C) ------------------------------------------
# The real question is width: does hidden 256 (H, stable init) beat hidden 128 (run_C,
# cmp_C_large [130,170) 2.514 / [170,240) 3.118 / [240,+) 5.048, all 0% success)?
# NB warmup is a minor confound here (C had none) -- but C's init did NOT explode
# (||g|| ~5), so warmup would be a near no-op for width 128. If H lands AMBIGUOUSLY
# between C and E, settle it cheaply with the matched control: this SAME script with
#   -Width 128   (width-128 + identical warmup = clean one-variable vs H).
#
# ---- PORTABILITY (same as run_C_long40.ps1 / run_E_wide256.ps1) --------------
#   $repo auto-derived from scripts/ -> repo root. Datasets at:
#       <repo>/data/train.lmdb   and   <repo>/data/val.lmdb
#   Set env POLY3D_PY to the polygen env python, e.g.
#       $env:POLY3D_PY = "C:/path/to/envs/polygen/python.exe"
#   Outputs under runs/gen_v1/ (gitignored, local to this machine).
#
# Launch (foreground, WATCH the tqdm bar live on console):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_H_wide256_warmup.ps1 -Live
# Launch (foreground, quiet - everything to logs):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_H_wide256_warmup.ps1
# Launch (OS-detached, harness idle-kill resistant; no console, so no -Live):
#   Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',
#     '<repo>/scripts/run_H_wide256_warmup.ps1' -WindowStyle Hidden -PassThru
# Tune:  ... -WarmupSteps 400   |   ... -Lr 1.5e-4   |   ... -Width 128  (control)
#
# ~2 days (hidden256 is ~2x slower per epoch than hidden128). Do not run other GPU
# jobs alongside it (past OOM lesson).
#
# REPORTING (nothing to copy back, only a few lines to hand-type / photograph):
#   1) runs/gen_v1/H_gnorm.log   <- CHECK FIRST. Auto-built ASCII summary: the early
#      [GNORM] optstep lines (did warmup tame the init explosion?), which epoch
#      vae_best.pt is (<30 => diverged, like E did), and the [OOM] count.
#      If the explosion is NOT tamed, or [OOM] > 0, the eval below is meaningless.
#   2) runs/gen_v1/H_eval.log  -- the LARGE-MOLECULE EVAL table ([130,170)/[170,240)/
#      [240,+): rmsd median + success<1A) and the VAL SAMPLE EVAL (main distribution).
#      Compare the 240+ row to run_C (5.048, 0%) and run_E (5.64, 0%) by hand.
#   3) runs/gen_v1/H_wide256_wu/vae_log.csv   (per-epoch train/val loss)
# --------------------------------------------------------------------------------

param(
    [switch]$Live,              # -Live: show tqdm bars on console (stderr unredirected)
    [int]$WarmupSteps = 200,    # LR warmup length in OPTIMIZER steps. See "THE FIX".
    [double]$Lr = 3e-4,         # rated lr (== run_C/E). Drop to 1.5e-4 if warmup alone
                                # does not tame the init explosion (see H_gnorm.log).
    [int]$Width = 256           # hidden/vae_hidden. 256 = capacity run. 128 = matched
                                # control (width-128 + same warmup) to isolate width.
)

if ($env:POLY3D_PY) { $py = $env:POLY3D_PY } else { $py = "python" }
# repo root = two levels up from scripts/run_H_wide256_warmup.ps1
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

$edge   = [int]($Width / 2)                 # run_E used edge 128 for width 256.
$base   = "runs/gen_v1"
$status = "$base/status_H.txt"
$outH   = "$base/H_wide256_wu"
$summary= "$base/H_gnorm.log"
$largeH = "runs/overfit/large_eval.lmdb"    # regenerated from val.lmdb if absent

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

"START $(Get-Date -Format o) width=$Width warmup=$WarmupSteps lr=$Lr" |
    Out-File -Append -Encoding ascii $status

# ---- Train: run_E config + LR warmup ----------------------------------------------
if (-not (Done "H_DONE")) {
    $argsH = @(
        "--stage","vae",
        "--train_lmdb","data/train.lmdb",
        "--val_lmdb","data/val.lmdb",
        "--epochs","40",
        "--lr",[string]$Lr,"--lr_min","3e-5",
        "--max_atoms","288",
        "--subset_ratio","0.03","--val_subset_ratio","0.02",
        "--pos_loss_type","kabsch",
        "--hidden_dim",[string]$Width,"--edge_dim",[string]$edge,"--vae_hidden_dim",[string]$Width,
        "--enc_layers","4","--dec_layers","4","--latent_dim","16",
        "--batch_size","8","--grad_accum","4",
        "--egt_every","2","--enc_egt_every","2",
        "--warmup_steps",[string]$WarmupSteps,     # <<-- THE ONLY REAL DIFFERENCE vs run_E
        "--gnorm_log_every","50",                  # see lr ramp + norm early (verify the fix)
        "--empty_cache_every","500",
        "--num_workers","8","--save_every","1","--seed","42",
        "--out_dir",$outH
    )
    $ck = LatestCkpt $outH
    if ($ck) { $argsH += @("--resume",$ck) }
    "H_START $(Get-Date -Format o) resume=$ck width=$Width warmup=$WarmupSteps lr=$Lr" |
        Out-File -Append -Encoding ascii $status
    if ($Live) {
        & $py "scripts/train.py" @argsH 1> "$base/H_out.log"
    } else {
        & $py "scripts/train.py" @argsH 1> "$base/H_out.log" 2> "$base/H_err.log"
    }
    $ec = $LASTEXITCODE
    # Only mark H_DONE on SUCCESS. On failure (OOM fail-fast raises now, exit != 0),
    # write H_FAILED and exit WITHOUT H_DONE so relaunching re-enters this block and
    # LatestCkpt resumes from the newest vae_epoch*.pt (save_every 1).
    if ($ec -ne 0) {
        "H_FAILED exit=$ec $(Get-Date -Format o) - relaunch to resume from latest ckpt" |
            Out-File -Append -Encoding ascii $status
        exit $ec
    }
    "H_DONE exit=0 $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

# ---- Did the intervention actually take? (built before eval on purpose) -----------
"=== WARMUP CHECK (run_H, width=$Width warmup=$WarmupSteps lr=$Lr) $(Get-Date -Format o) ===" |
    Out-File -Encoding ascii $summary
"Early [GNORM] optstep lines below. Want: norm stays bounded (< ~30) while lr ramps." |
    Out-File -Append -Encoding ascii $summary
"If norm still spikes toward ~97 in the first tens of steps, warmup was too short --" |
    Out-File -Append -Encoding ascii $summary
"raise -WarmupSteps or drop -Lr to 1.5e-4, and DO NOT trust the eval (E's trap)." |
    Out-File -Append -Encoding ascii $summary
"(run_E baseline, NO warmup: mean ||g|| 6 -> 97.6 over first ~40 steps, best=ep7)" |
    Out-File -Append -Encoding ascii $summary
"" | Out-File -Append -Encoding ascii $summary
if (Test-Path "$base/H_out.log") {
    # First ~20 optstep lines = the danger zone where E exploded.
    $early = Select-String -Path "$base/H_out.log" -Pattern '\[GNORM\] optstep' |
             ForEach-Object { $_.Line.Trim() } | Select-Object -First 20
    if ($early) { $early | Out-File -Append -Encoding ascii $summary }
    else { "(no [GNORM] optstep lines -- is --gnorm_log_every set / train.py current?)" |
           Out-File -Append -Encoding ascii $summary }

    # --- Which epoch is vae_best.pt? (E's trap: best=ep7, then 33 ep of no gain) ---
    "" | Out-File -Append -Encoding ascii $summary
    $bestEp = & $py -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" "$outH/vae_best.pt" 2>$null
    if ($LASTEXITCODE -eq 0) {
        "vae_best.pt = epoch $bestEp of 40  (eval below uses THIS checkpoint)" |
            Out-File -Append -Encoding ascii $summary
        if ([int]($bestEp | Select-Object -Last 1) -lt 30) {
            "  ^ WARNING: best is early => likely diverged and never recovered (E did this)" |
                Out-File -Append -Encoding ascii $summary
        }
    }

    # --- Were any batches silently dropped? (fail-fast should have RAISED, not skipped) ---
    $oom = @(Select-String -Path "$base/H_out.log" -Pattern '\[OOM\]')
    "" | Out-File -Append -Encoding ascii $summary
    if ($oom.Count -gt 0) {
        "*** [OOM] skipped batches: $($oom.Count) -- RESULT IS CONTAMINATED ***" |
            Out-File -Append -Encoding ascii $summary
        "    Dropped batches are the heaviest = the 240+ giants = what we measure." |
            Out-File -Append -Encoding ascii $summary
        "    With fail-fast (--oom_max_skips 0) the run should RAISE on the 1st OOM;" |
            Out-File -Append -Encoding ascii $summary
        "    if you see many, an older train.py without fail-fast was used. Re-run" |
            Out-File -Append -Encoding ascii $summary
        "    with bs4 x ga8 (eff 32 unchanged) or a box with more VRAM headroom." |
            Out-File -Append -Encoding ascii $summary
    } else {
        "[OOM] skipped batches: 0  (clean)" | Out-File -Append -Encoding ascii $summary
    }
}
Get-Content $summary | Write-Host

# ---- Auto-eval (GPU free now; no concurrent processes) ----------------------------
# Reuses the SAME large_eval.lmdb C/D/E/G built, so cmp_H_large is directly comparable
# to cmp_C_large / cmp_E_large (identical molecules, first 64 == tiny_large).
if (-not (Done "EVAL_DONE")) {
    $ckH = "$outH/vae_best.pt"

    if (-not (Test-Path $largeH)) {
        "MAKE_LARGE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
        & $py "scripts/make_tiny_lmdb.py" --src "data/val.lmdb" --dst $largeH `
            --n 300 --min_atoms 130 --max_atoms 300 1>> "$base/H_eval.log" 2>> "$base/H_err.log"
    }

    "EVAL_START $(Get-Date -Format o) ckpt=$ckH" | Out-File -Append -Encoding ascii $status
    "=== LARGE-MOLECULE EVAL (val-derived, >=130 atoms) ===" | Out-File -Append -Encoding ascii "$base/H_eval.log"
    & $py "scripts/eval_by_size.py" --checkpoint $ckH --val_lmdb $largeH `
        --max_atoms 300 --batch_size 8 --num_workers 4 `
        --out "$base/cmp_H_large.json" 1>> "$base/H_eval.log" 2>> "$base/H_err.log"

    "`n=== VAL SAMPLE EVAL (data/val.lmdb, first 640 mol) ===" | Out-File -Append -Encoding ascii "$base/H_eval.log"
    & $py "scripts/eval_by_size.py" --checkpoint $ckH --val_lmdb "data/val.lmdb" `
        --max_atoms 288 --batch_size 8 --num_workers 4 --max_batches 80 `
        --out "$base/cmp_H_valgen.json" 1>> "$base/H_eval.log" 2>> "$base/H_err.log"

    "EVAL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

"ALL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
Write-Host ""
Write-Host "Report: $summary (check the early norm/lr FIRST), $outH/vae_log.csv, $base/H_eval.log"

# Restore the parent console's output encoding (no-op if run detached).
[Console]::OutputEncoding = $prevOutEnc
