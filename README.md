# ADI: Adversarial Dominating Inputs in Vertical Federated Learning Systems

Code for **"ADI: Adversarial Dominating Inputs in Vertical Federated Learning
Systems"** (Qi Pang, Yuanyuan Yuan, Shuai Wang, Wenting Zheng; IEEE S&P 2023;
extended version on [arXiv:2201.02775](https://arxiv.org/abs/2201.02775)). The extended version
included as `paper.pdf` with detailed proofs and discussions in appendix.

An **Adversarial Dominating Input (ADI)** is an input crafted by a malicious VFL
participant *A* that *dominates* the joint inference: regardless of what the benign
participant *B* feeds in, the model's prediction is pinned to a target label, so *B*'s
contribution is nullified. This repo trains the VFL models and runs the main gradient-based
ADI synthesis attack algorithm in **whitebox** and **blackbox** settings, with **random**
and **bounded** mutation.

> The implementation to use is `vfl_adi/`.

## Datasets

| Dataset  | Protocol | Partition (A / B)              | Source              |
|----------|----------|--------------------------------|---------------------|
| credit   | HeteroLR | 13 / 10 features (binary)      | UCI (auto-download) |
| vehicle  | HeteroLR | 9 / 9 features (4 classes)     | UCI (auto-download) |
| mnist    | SplitNN  | 28×14 / 28×14 image halves     | torchvision (auto)  |
| cifar10  | SplitNN  | VGG16 embeddings of L/R halves | torchvision (auto)  |
| nuswide  | SplitNN  | 634 image / 1000 tag features  | manual (see below)  |

## Setup

```bash
conda create -y -n vfl-adi python=3.10
conda activate vfl-adi
# install PyTorch for your CUDA (example: CUDA 12.4)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Runs on CPU, but a GPU is recommended.

## Run

```bash
# everything for one dataset: prepare -> train -> baseline -> attack
python -m vfl_adi.main all --dataset credit

# or step by step
python -m vfl_adi.main prepare  --dataset credit
python -m vfl_adi.main train    --dataset credit
python -m vfl_adi.main baseline --dataset credit
python -m vfl_adi.main attack   --dataset credit --box white --mode random  --threshold 95
python -m vfl_adi.main attack   --dataset credit --box black --mode bounded --threshold 99

# all datasets at once, writes results/summary.md
python scripts/reproduce.py --datasets credit vehicle mnist cifar10
```

`--dataset` is one of `credit vehicle mnist cifar10 nuswide`; `--box` is `white|black`;
`--mode` is `random|bounded`; `--threshold` is `95|99`. Processed data is cached under
`data/processed/` and checkpoints under `checkpoints/` (set `VFL_DATA_ROOT` to relocate
the data directory).

## A note on the numbers

Exact success rates might slightly differ from the paper — the trained checkpoints, feature
splits, preprocessing, optimizers, and hardware are not identical, and the blackbox
attack uses query-based gradient estimation. This repo targets the qualitative trend,
not exact parity: gradient-based ADI synthesis achieves high (random mutation) to
non-trivial (bounded mutation) dominating success across datasets, and ADIs suppress the
benign party's contribution. Read the printed numbers as evidence of that trend.

## NUS-WIDE

NUS-WIDE must be obtained manually from the
[official site](https://lms.comp.nus.edu.sg/research/NUS-WIDE.htm) (its low-level
features are not openly mirrored). Place them under `data/raw/NUS_WIDE/`:

```
data/raw/NUS_WIDE/
  Low_Level_Features/{Train,Test}_Normalized_*.dat
  NUS_WID_Tags/{Train,Test}_Tags1k.dat
  Groundtruth/AllLabels/Labels_*.txt
  Groundtruth/TrainTestLabels/Labels_<concept>_{Train,Test}.txt
```

Then `python -m vfl_adi.main all --dataset nuswide`.

## Layout

```
vfl_adi/             models.py, datasets.py, train.py, attack.py, main.py
scripts/reproduce.py runs all datasets and writes results/summary.md
paper.pdf            the extended paper
```
