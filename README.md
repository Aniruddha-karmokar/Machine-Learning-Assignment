# Machine Learning: Assignment 2 — Bagging, Boosting and Bias-Variance on Face Recognition

- **Student Name:** Aniruddha Karmokar
- **Roll Number:** `P26CS0001`
- **Random Seed:** `260001` (all numeric characters of `P26CS0001` in order, via `numpy.random.default_rng(260001)`)
- **Repository Link:** `https://github.com/aniruddha-karmokar/P26CS0001-ml-assignment2`

---

## Repository Structure

```text
.
├── README.md                         # Exact reproduction commands & project overview
├── environment.yml                   # Conda environment specification
├── requirements.txt                  # Python pip dependencies
├── configs/
│   └── config_P26CS0001.yaml         # Hyperparameter grids and frozen model configurations
├── data/
│   ├── olivetti_faces.npz            # Cached Olivetti Faces dataset (400 x 64 x 64 float32)
│   ├── split_P26CS0001.npz           # Saved deterministic 6/2/2 per-identity split (seed 260001)
│   └── split_P26CS0001.json          # Human-readable JSON index record of train/val/test split
├── src/
│   ├── __init__.py
│   ├── dataset.py                    # Dataset loading, seed extraction, split & 16x16 masking
│   ├── bagging.py                    # Bagging from scratch, OOB, pairwise disagreement, vote margin
│   ├── boosting.py                   # Multiclass SAMME AdaBoost from scratch & 3-class toy round
│   ├── bias_variance.py              # Empirical Bias^2 and Variance evaluator across 30 refits
│   └── fast_cart.c                   # Optional OpenMP C-accelerated exact CART Gini solver
├── scripts/
│   ├── run_all.py                    # Master script executing Q1, Q2, and Q3 end-to-end
│   └── generate_figures.py           # Script generating all 7 publication-quality PDF/PNG figures
├── logs/
│   ├── experiment_results.json       # Complete structured output of all Q1, Q2, and Q3 experiments
│   └── final_run.log                 # Console execution log
└── figures/
    ├── fig1_dataset_and_mask.pdf     # Clean vs. 16x16 masked test faces
    ├── fig2_q1a_tree_depth_analysis.pdf
    ├── fig3_q1bc_bagging_curves.pdf
    ├── fig4_q1d_case_studies.pdf
    ├── fig5_q2b_samme_dynamics.pdf
    ├── fig6_q2cd_hard_face_and_mask.pdf
    └── fig7_q3_bias_variance.pdf
```

---

## Exact Reproduction Commands

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate ml-assign2-p26cs0001
# Or via pip:
pip install -r requirements.txt
```

### 2. Optional C CART Acceleration Build

```bash
gcc -O3 -march=native -fopenmp -shared -fPIC src/fast_cart.c -o src/libfastcart.so -lm
```

### 3. Reproduce Split and All Experiments (Q1, Q2, Q3)

```bash
python3 scripts/run_all.py | tee logs/final_run.log
```

### 4. Reproduce All Report Figures

```bash
python3 scripts/generate_figures.py
```
