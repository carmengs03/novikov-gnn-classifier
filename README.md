# Novikov GNN Classifier

Code, datasets and trained models accompanying the manuscript

> **Coefficient-level and graph-level classifiers for Novikov algebra recognition**
> Carmen Gutiérrez Silva, Universidad Loyola Andalucía.
> Submitted to *Journal of Computational and Applied Mathematics* (Elsevier).

---

## Repository layout

```
novikov-gnn-classifier/
├── LICENSE
├── README.md
├── requirements.txt
│
├── data/                           Active datasets used in the paper
│   ├── dataset_combinado_dim2_dim3_dificiles.pt         (joint model training)
│   ├── dataset_enriquecido_dim2_50000_DATOS_dificiles.pt (dim=2 training)
│   ├── dataset_noisy_topologico_dim2_enriquecido.pt (noisy-eval dim=2)
│   └── dataset_noisy_topologico_dim3_enriquecido.pt (noisy-eval dim=3)
│
├── models/                         Published checkpoints
│   ├── modelo_conjunto_dim23_dificil.pt     (joint model, 90.83% ± 0.13% global)
│   └── modelo_final_dim2_dificil.pt         (dim=2 dedicated, 82.38% ± 0.40%)
│
├── src/
│   ├── core/         Architecture, exact verifier, feature engineering, generators
│   ├── generation/   Scripts that produced the datasets
│   ├── training/     Scripts that trained the published models
│   ├── evaluation/   Baselines, noisy and OOD evaluation, basis-change control
│   └── analysis/     Latent space, dim2→dim3 transition, figure generation
│
├── experiments/                    Standalone scripts that produced the
│                                   JSONs in results/from_experiments/
│   (self-contained: sys.path.insert(0, HERE), ready to zip and deploy
│   to a remote GPU host)
│
└── results/
    ├── from_paper/     JSONs produced by scripts in src/
    ├── from_experiments/  JSONs produced by scripts in experiments/
    └── figures/        Figures used in the paper
```

## Reproducibility

Every accuracy reported in the manuscript can be reproduced from the
published checkpoints and datasets. Global seeds are fixed:

- `torch.manual_seed(42 + i)` for the train/val/test split
- `torch.manual_seed(100 + i)` for network initialization

with `i = 0, ..., 9`.

Minimal example (chance-level noisy accuracy of the graph classifier
in dimension 3):

```python
import torch
from src.core.modelo_mpnn import RedNovikov
from torch_geometric.loader import DataLoader

ckpt = torch.load("models/modelo_conjunto_dim23_dificil.pt", weights_only=False)
modelo = RedNovikov(dim_nodos=6, dim_aristas=3,
                    dim_oculta=ckpt["hp"]["dim_oculta"])
modelo.load_state_dict(ckpt["state_dict"])
modelo.eval()

ds = torch.load("data/dataset_noisy_topologico_dim3_enriquecido.pt",
                weights_only=False)
loader = DataLoader(ds, batch_size=128)
# evaluation loop ...
```

Scripts in `src/` were designed to be run from the repository root with the
`.pt` files at the top level of `data/`. Scripts in `experiments/` are
self-contained (`sys.path.insert(0, str(HERE))`) and can be packaged and
deployed to a remote GPU host as-is.

## License

MIT. See [`LICENSE`](LICENSE).
