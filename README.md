# MNIST Optimization Playground

## Why the Target is 99.4% in ≤15 Epochs and <8000 Parameters

- Practical benchmark: 99.4% on MNIST is a recognized strong threshold indicating the model has learned nearly all patterns without resorting to very large capacity.
- Time/compute constraint: ≤15 epochs ensures efficient convergence and reproducibility on modest hardware (Apple Silicon).
- Model efficiency: <8000 parameters enforces architectural discipline (GAP heads, judicious channels) and improves generalization.

These constraints jointly encourage modern, efficient design rather than brute-force capacity.

---

## Repository Structure

- `model1.py` – Baseline CNN
- `model2.py` – CNN + BatchNorm + Dropout
- `model3.py` – Optimized small CNN (GAP; tuned channels) [current mainline]
- `model3a.py` – MobileNet-inspired depthwise-separable baseline (experimental)
- `utils.py` – Training utilities, loaders, TTA, EMA/SWA helpers
- `experiment_runner.py` – Reproducible experiments with CLI
  - `python experiment_runner.py 3` – run Experiment 3
  - `python experiment_runner.py sweep` – sweep selected hyperparameters
  - (Optional) `python experiment_runner.py 3a` – depthwise-separable experiment (if wired)
- Archives in `archive_v1/` capture earlier best results and analyses

---

## Experiments (Targets, Results, Analysis)

### Experiment 1 – Baseline CNN (model1.py)

- Target: ≥98.5% test accuracy; ≤15 epochs; <8000 params.
- Architecture: multiple 3×3 convs, 1×1 bottlenecks, single MaxPool, GAP head.
- Parameters: ~5,904
- Receptive field (approx): 20×20 at head (3×3 stacks, one pooling stage).

Results (archived):

- Best test accuracy: ~98.58%
- Last 3 epochs: stable around 98.3–98.6%

Analysis:

- GAP plus slender channels delivers strong baseline without dense layers.
- Minimal overfitting; room to improve stability and convergence.

Links:

- `model1.py`
- `experiment1_results.json`, `experiment1_analysis.txt`

---

### Experiment 2 – BN + Dropout (model2.py)

- Target: ≥99.0% test accuracy; ≤15 epochs; <8000 params.
- Architecture: same as E1 with BatchNorm after each conv and modest Dropout (~0.1).
- Parameters: ~6,056
- Receptive field: ~22×22 (same topology, marginally different placements).

Results (archived):

- Best test accuracy: ~99.17%
- Last 3 epochs: ~98.95–99.15%

Analysis:

- BN stabilizes training, allowing higher LR and faster convergence.
- Dropout curbs overfitting; good generalization with small budget.

Links:

- `model2.py`
- `experiment2_results.json`, `experiment2_analysis.txt`

---

### Experiment 3 – Optimized small CNN (model3.py)

- Target: ≥99.4% test accuracy consistently in last 3 epochs; ≤15 epochs; <8000 params.
- Architecture:
  - Channels: 1→10→16→12→14→14→12, GAP head
  - 3×3 kernels with padding=1, single MaxPool(2×2)
  - ReLU → BatchNorm ordering (empirically best here)
  - No dense layers; GAP + 1×1 classifier
- Parameters: ~7,602
- Receptive field (approx):
  - After conv2: 7×7
  - After pool: 8×8
  - After conv6: ~20×20

Current best (latest stable run):

- Best test accuracy: ~99.19%
- Last 3 epochs: e.g., ~99.08, 99.19, 99.18 (top configs)

Analysis:

- Under strong constraints, 7.6k-parameter CNN with GAP is near the 99.2–99.3% band.
- Consistency ≥99.4% requires extremely tight training—margins are small. Best levers were OneCycleLR tuning, label smoothing=0.02, and removing late-phase destabilizers (mid-epoch SWA eval, SAM, over-augmentation).

Links:

- `model3.py`
- `experiment3_results.json`, `experiment3_analysis.txt`

---

### Experiment 3a – Depthwise-Separable (model3a.py)

- Target: same as E3 (≥99.4% consistent; ≤15 epochs; <8000 params).
- Architecture: MobileNet-inspired depthwise-separable blocks; GAP head.
- Parameters: ~4,560
- Receptive field: similar progression to E3 due to 3×3 depthwise kernels and two pooling stages.

Status: baseline created; intended as a strong efficiency baseline for future tuning.

Links:

- `model3a.py`
- `experiment3a_results.json` (when run)

---

## Training and Optimization Playbook (E3)

- Stable recipe:

  - OneCycleLR (epochs=15, max_lr=0.028, pct_start=0.25)
  - SGD(momentum=0.9, nesterov=True), weight_decay=1e-4
  - Label smoothing = 0.02 (constant)
  - Train aug: rotation ±7°; no-aug in last 2–3 epochs
  - TTA: angles [-3°, 0°, +3°]
  - SWA evaluated only after training (update_bn before final eval)

- Lessons:
  - Mid-epoch SWA eval without updating BN will tank accuracy; avoid until end.
  - Over-augmentation late in training or stacking SAM + SWA can depress peak.
  - Dense layers break the <8k budget; GAP heads are key.

---

## How to Reproduce

- Single run (E3):

```
python experiment_runner.py 3
```

- Hyperparameter sweep (E3):

```
python experiment_runner.py sweep
```

Outputs:

- Results JSONs: `experiment*_results.json`
- Analyses: `experiment*_analysis.txt`
- Best checkpoints: `model*_best.pth`

---

## Future Work

- Carefully tuned depthwise-separable variant (3a) to match 7–8k params and push consistency.
- CosineAnnealing + SWA tail (only end-of-training SWA) with small eta_min.
- Micro TTA refinement: rotations ±2°/±4° only.
