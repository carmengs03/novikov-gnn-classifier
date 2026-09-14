#!/usr/bin/env bash
# Orchestrator: runs every experiment in this folder end-to-end on a remote GPU host.
# Steps are independent: if one fails, the others continue.
# Run with:  bash run_all.sh 2>&1 | tee run_all.log

set -u
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

TS() { date +"%Y-%m-%d %H:%M:%S"; }
LOG_DIR=logs
mkdir -p "$LOG_DIR"

run_step() {
  local label="$1"; shift
  local logfile="$LOG_DIR/${label}.log"
  echo "[$(TS)] START $label -> $logfile"
  if python -u "$@" >"$logfile" 2>&1; then
    echo "[$(TS)] OK    $label"
  else
    echo "[$(TS)] FAIL  $label (see $logfile)"
  fi
}

echo "[$(TS)] Batch start"
python -c "import torch; print('cuda:', torch.cuda.is_available(),
'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# ---------------------------------------------------------------------
# Quick step first: 04a residuals (no GPU, just the verifier)
# ---------------------------------------------------------------------
run_step "04a_residuos" 04a_residuos.py --output continuidad_4a.json

# ---------------------------------------------------------------------
# Step 03: 10 seeds of the joint model + dim=3 adversarial evaluation.
# ---------------------------------------------------------------------
run_step "03_joint_10seeds" 03_joint_10seeds.py \
    --n_seeds 10 --n_epocas 100 --output semillas_10_joint.json

# ---------------------------------------------------------------------
# Step 03 dim=2: 10 seeds of the dedicated dim=2 model + adversarial dim=2.
# ---------------------------------------------------------------------
run_step "03_dim2_10seeds" 03_dim2_10seeds.py \
    --n_seeds 10 --n_epocas 100 --output semillas_10_dim2.json

# ---------------------------------------------------------------------
# Step 04c: GNN accuracy vs sigma curve.
# ---------------------------------------------------------------------
run_step "04c_curva_sigma" 04c_curva_sigma.py \
    --sigmas 1e-5,1e-4,1e-3,1e-2,1e-1,1 \
    --n_positivos 800 \
    --output continuidad_4c.json

# ---------------------------------------------------------------------
# Step 04b: tangent perturbation.
# ---------------------------------------------------------------------
run_step "04b_tangente" 04b_tangente.py \
    --n_positivos 500 --sigma 1e-3 \
    --output continuidad_4b.json

# ---------------------------------------------------------------------
# Step 01: leave-one-out over the 40 topologies.
# Start with 1 seed x 80 epochs for the first pass.
# ---------------------------------------------------------------------
run_step "01_leaveoneout" 01_ood_leaveoneout.py \
    --n_seeds 1 --n_epocas 80 --output ood_leaveoneout.json

echo "[$(TS)] Batch complete. Results:"
ls -la *.json 2>/dev/null
echo "Per-step logs in $LOG_DIR/"
