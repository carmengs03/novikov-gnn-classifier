# experiments — Standalone experiment scripts

Self-contained bundle with the scripts that produced the JSON files in
[`../results/from_experiments/`](../results/from_experiments/). Every script in
this folder imports only from within the folder itself
(`sys.path.insert(0, str(HERE))`), so the directory can be zipped and
deployed to a remote GPU host without additional setup.

## Contents

```
experiments/
├── README.md                       this file
├── run_all.sh                      orchestrator with per-step logging
│
├── verificador.py                  exact Novikov verifier + residual computation
├── enriquecer_datos.py             six-feature node encoding
├── identificar_topologia.py        catalogue lookup from edge_index + mask
│
├── generadores_dim2/               nine dimension-two config generators
├── generadores_dim3/               thirty-one dimension-three config generators
│
├── 03_joint_10seeds.py             ten seeds of the joint model + noisy dim=3 eval
├── 03_dim2_10seeds.py              ten seeds of the dedicated dim=2 model + noisy dim=2 eval
├── 01_ood_leaveoneout.py           leave-one-out over the 9+31 topologies
├── 04a_residuos.py                 distribution of ||R||_inf on noisy negatives
├── 04c_curva_sigma.py              graph-classifier accuracy vs perturbation magnitude
└── 04b_tangente.py                 tangent perturbation of positives
```

## Datasets and checkpoints

The scripts expect the following files next to them (same directory):

- `dataset_combinado_dim2_dim3_dificiles.pt` (joint model training set)
- `dataset_enriquecido_dim2_50000_DATOS_dificiles.pt` (dim=2 training set)
- `dataset_noisy_topologico_dim2_enriquecido.pt` (dim=2 noisy eval)
- `dataset_noisy_topologico_dim3_enriquecido.pt` (dim=3 noisy eval)
- `modelo_conjunto_dim23_dificil.pt` (published joint checkpoint)
- `modelo_final_dim2_dificil.pt` (published dim=2 checkpoint)

They live in [`../data/`](../data/) and [`../models/`](../models/) in
the main repository. To run this bundle on a remote GPU, copy those
`.pt` files into this directory alongside the scripts.

## Running on a remote GPU host

**Recommended GPU:** NVIDIA A40 with CUDA 12.4 or 12.8. Avoid Blackwell
architectures (RTX Pro 4000/5000, sm_120), which are not supported by
PyTorch 2.4 stable.

```bash
# Quick GPU sanity check (~30 s):
python -c "import torch; from torch_geometric.nn import global_mean_pool; \
x = torch.randn(100,8).cuda(); b = torch.zeros(100, dtype=torch.long).cuda(); \
print('OK', torch.cuda.get_device_name(0), global_mean_pool(x,b).shape)"

# Launch inside tmux so the batch survives disconnection:
tmux new -s experiments
bash run_all.sh 2>&1 | tee run_all.log
# Ctrl-b then d to detach
```

## Reproducibility conventions

All scripts follow the same conventions as the training scripts in
`../src/training/`:

- **Fixed seeds:** `torch.manual_seed(42+i)` for splits,
  `torch.manual_seed(100+i)` for network initialization, with
  `i = 0, ..., N_SEEDS-1`. `cudnn.deterministic = True` throughout.
- **70/15/15 train/val/test split** per seed via `random_split`.
- **Best checkpoint by val_loss:** during training the model state is
  saved when `val_loss` reaches its minimum; test accuracy is measured
  on that checkpoint (not on the final epoch).
- **Architecture:** `RedNovikov` with `MAX` aggregation, `MEAN` pool,
  three layers, residual connections, `dim_oculta=64`.
- **Joint model hyperparameters** (from the published checkpoint):
  `lr=1.194e-3`, `dim_oculta=64`, `batch=64`.
- **Dedicated dim=2 model hyperparameters:**
  `lr=9.72e-4`, `dim_oculta=64`, `batch=16`.

## Outputs

Each step writes a JSON in the current directory with incremental saving:

- `semillas_10_joint.json` — joint model, ten seeds, noisy dim=3
- `semillas_10_dim2.json` — dedicated dim=2 model, ten seeds, noisy dim=2
- `continuidad_4a.json` — distribution of residuals
- `continuidad_4c.json` — graph-classifier accuracy vs sigma
- `continuidad_4b.json` — original positives / tangent positives / noisy mix
- `ood_leaveoneout.json` — zero-shot accuracy per held-out topology
