# Loss ablation: does multiscale_distmat (smooth loss for global folding) improve
# large-unit generalization? Decoder-only (freeze_encoder) fine-tune from v3c.
# Compare vs v3c(kabsch, large-unit recon 0.50). ASCII only (PS 5.1).
param(
    [string]$OutName = "abl_multiscale",
    [int]$Epochs = 12
)
if ($env:POLY3D_PY) { $py = $env:POLY3D_PY } else { $py = "python" }
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:256"
Set-Location $repo
$out = "runs/$OutName"
$status = "$out/status.txt"
$val = "data/polyomics_PG_val.lmdb"
if (-not (Test-Path $out)) { New-Item -ItemType Directory -Force -Path $out | Out-Null }

function Done($tag) { return (Test-Path $status) -and (Select-String -Path $status -Pattern $tag -Quiet) }
function LatestCkpt($dir) {
    $f = Get-ChildItem -Path $dir -Filter "vae_epoch*.pt" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
    if ($f) { return $f.FullName } else { return $null }
}

"START $(Get-Date -Format o) multiscale_distmat decoder-only(freeze_encoder) epochs=$Epochs" | Out-File -Append -Encoding ascii $status

if (-not (Done "TRAIN_DONE")) {
    $a = @(
        "--stage","vae","--train_lmdb","data/polyomics_PG_train.lmdb","--val_lmdb",$val,
        "--out_dir",$out,
        "--hidden_dim","256","--edge_dim","128","--vae_hidden_dim","256",
        "--cond_layers","4","--latent_dim","16","--enc_layers","4","--dec_layers","4",
        "--egt_every","2","--enc_egt_every","2",
        "--beta_start","0.1","--beta_end","0.1","--beta_warmup_epochs","1",
        "--pos_loss_type","multiscale_distmat","--w_local","1.0","--w_global","1.0",
        "--w_pos","1.0","--w_bond","1.0","--w_angle","0.5","--w_dihedral","0.1",
        "--w_clash","5.0","--clash_factor","0.6","--clash_min_graph_dist","3","--clash_max_pairs","512",
        "--freeze_encoder",
        "--batch_size","64","--grad_accum","2","--epochs",[string]$Epochs,
        "--lr","5e-5","--lr_min","1e-5","--weight_decay","1e-5","--grad_clip","5.0",
        "--warmup_steps","200","--warmup_start_factor","0.05","--gnorm_log_every","100",
        "--max_atoms","288","--val_subset_ratio","0.3","--empty_cache_every","500",
        "--num_workers","8","--prefetch_factor","4","--save_every","1","--seed","42"
    )
    $ck = LatestCkpt $out
    if ($ck) { $a += @("--resume",$ck); "TRAIN_START resume=$ck $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status }
    else { $a += @("--init_weights","runs/polyomics_vae_v3c/vae_best.pt"); "TRAIN_START init=v3c $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status }
    & $py "scripts/train.py" @a 1> "$out/train_out.log" 2> "$out/train_err.log"
    if ($LASTEXITCODE -ne 0) { "TRAIN_FAILED exit=$LASTEXITCODE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status; exit $LASTEXITCODE }
    "TRAIN_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}

if (-not (Done "EVAL_DONE")) {
    $ckpt = LatestCkpt $out
    "EVAL_START $(Get-Date -Format o) ckpt=$ckpt" | Out-File -Append -Encoding ascii $status
    & $py "scripts/eval_ensemble.py" --checkpoint $ckpt --val_lmdb $val --mode recon `
        --max_ref 100 --n_gen 50 --out "$out/eval_recon_final.json" 1> "$out/eval_out.log" 2> "$out/eval_err.log"
    "EVAL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
}
"ALL_DONE $(Get-Date -Format o)" | Out-File -Append -Encoding ascii $status
