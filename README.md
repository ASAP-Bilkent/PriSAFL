# Pri-SAFL Reproduction Package

Code and processed data for the "Pri-SAFL: Privacy-Preserving Survival Analysis via Federated Learning Under Homomorphic Encryption" study.

**Docker is the only supported environment.** The image pins the package versions and builds OpenFHE 1.3.0 with its Python
binding from source.

## Contents

```
PriSAFL/
├── baseline/        # Original PiDeeL reproduction (Adam + ReLU + bias, CoxPH 4-layer)
├── ablation/        # HE-ready plaintext ablations (SGD + poly activation, no bias)
│   ├── make_tables.py   # prints the supplementary c-index tables from logs/*.csv
│   ├── fig_boxplots.py  # regenerates the federated boxplot figure from logs/*.csv
│   └── logs/            # 15-run CV result CSVs (inputs to the tables/figures)
├── he/              # Fully encrypted CKKS training pipeline
│   ├── tests/           # correctness, level-budget and communication tests
│   └── e2e/             # fully encrypted K=10 run on one fold
├── data/            # Processed model inputs (see Data)
├── docker_prisafl/  # Dockerfile + build/run/verify scripts
└── README.md
```

## Requirements

- **Docker** (Bare metal or VM, see below).
- One-time image build takes **~30 min** based on system (compiles OpenFHE from source).
- **~32+ GB RAM** for one-client bootstrap at `--ring-dim-exp 15` (~4x that at the 128-bit
  2^17 ring). K>1 at 2^16 needs 10-16 GB. Plaintext baseline/ablation need only a few GB.
- *Optional* GPU for plaintext runs and the e2e initial weights (HE is CPU): install
  `nvidia-container-toolkit` (`sudo docker_prisafl/install-nvidia-toolkit.sh` on Debian/Ubuntu),
  then prefix commands with `GPU=1`.

## Quickstart

Please see `docker_prisafl/README.md` for all build-related details.

```bash
cd docker_prisafl
./build.sh        # build the `prisafl` image (~30 min, OpenFHE compile)
./verify.sh       # end-to-end proof:
                  #   1. openfhe + torch import
                  #   2. HE correctness: encrypted vs plaintext
                  #   3. plaintext parity: Pri-SAFL Cox-CC 2L median, training
                  #      from scratch
                  #   4. HE runtime + per-phase timing
```

**Build daemon: Linux vs macOS/Windows.** `build.sh` defaults to the **native
Docker engine** (recommended on Linux, fastest). On
**macOS/Windows**, or to use Docker Desktop on Linux, run `DESKTOP=1 ./build.sh`.
Docker Desktop builds in a Linux VM.

`-march=native` is the fast default and is fine on server CPUs (Haswell/Xeon),
but on some CPUs the native OpenFHE codegen is numerically unstable (we add a context-validation test that retries until clean, so it still works,
just with wasted rebuilds). For a deterministic, portable build use
`GENERIC=1 ./build.sh` (`WITH_NATIVEOPT=OFF` + `-march=x86-64`): fully stable, at
~20% slower HE. Details in `docker_prisafl/README.md`.

## Running

Every command runs inside the container via `run.sh`, which binds this
folder at `/work`, so any output CSVs land back here on the host:

```bash
cd docker_prisafl

# encrypted pipeline (CPU): 2-layer Cox-CC, 3 samples, 10 epochs, bootstrap + timing
./run.sh "python -m he.main --n-samples 3 --epochs 10 --bootstrap --timing --dims 37 138 1 --seed 42 --ring-dim-exp 15"

# HE correctness test (asserts encrypted vs plaintext < 1e-3)
./run.sh "python -m he.tests.test_pipeline --stage multi_iter --n-samples 3 --n-iters 1 --bootstrap --dims 37 138 1 --seed 42 --ring-dim-exp 15"

# level consumption per training iteration (K=1, then K=2)
./run.sh "python -m he.tests.test_levels"
./run.sh "python -m he.tests.test_levels --n-clients 2 --max-rss-gb 15"

# message sizes and IntMPBoot timings (K=2)
./run.sh "python -m he.tests.measure_comm --stage multi_iter --bootstrap --n-clients 2 --n-samples 4 --n-iters 2"

# plaintext baseline / ablation
./run.sh "python baseline/main.py --output baseline_result"
./run.sh "python ablation/main.py --loss cox-cc --n-layers 2 --approx-cox-cc 3-15cT4 --lr 0.045 --output boxplot_coxcc2_k1"

# print the supplementary c-index tables from ablation/logs/*.csv
./run.sh "python ablation/make_tables.py"
```

Add `GPU=1` to attach a GPU (plaintext runs and e2e_fold). **It is advised** to run HE commands one at a time, since they are very RAM intensive.

Without `--ring-dim-exp`, HE uses the 128-bit rings of the manuscript (2^17 for one client,
2^16 for `--n-clients` > 1). `--ring-dim-exp 15` fits the one-client commands in ~32 GB and is
not 128-bit secure. `test_levels` always uses the 128-bit rings and skips `--n-clients 2` if
less than `--max-rss-gb` is free.

## Reproducing the tables and figures

Shared settings: SGD with Nesterov momentum 0.9, L2 = 0, 100 epochs, full batch, polynomial
activation (deg 3, interval ±15), seeded weight init (seeds 6, 35, 81 × 5 folds). Cox-CC uses
the constrained degree-3 sigmoid `3-15cT4` and CoxPH the 21-level exp/inverse gradient
`7-2-m-l8-u8-b16-r2.5-n4-s7`. Learning rates depend on loss, depth and split. The
`ablation/main.py` defaults differ, so pass the flags below. Each `--output` name matches a
shipped CSV under `ablation/logs/`, and a rerun overwrites it. `make_tables.py` reads them all.

```bash
cd docker_prisafl

# baseline vs HE-ready C-index (K=1, 50) and federated boxplots (K=1, 10, 50)
./run.sh "python baseline/main.py --output baseline_result"
./run.sh 'for k in 1 10 50; do python ablation/main.py --loss cox-cc --n-layers 2 --approx-cox-cc 3-15cT4 --lr 0.045 --n-clients $k --output boxplot_coxcc2_k$k; done'
./run.sh 'for k in 1 10 50; do python ablation/main.py --loss coxph --n-layers 4 --approx-coxph 7-2-m-l8-u8-b16-r2.5-n4-s7 --lr 0.01 --n-clients $k --output boxplot_coxph4_k$k; done'
./run.sh 'for k in 1 10 50; do python ablation/main.py --loss cox-cc --n-layers 2 --approx-cox-cc exact --lr 0.045 --n-clients $k --output boxplot_coxcc2_exact_k$k; done'
./run.sh 'for k in 1 10 50; do python ablation/main.py --loss coxph --n-layers 4 --approx-coxph "" --lr 0.01 --n-clients $k --output boxplot_coxph4_exact_k$k; done'
./run.sh "python ablation/main.py --loss coxph --n-layers 2 --approx-coxph 7-2-m-l8-u8-b16-r2.5-n4-s7 --lr 0.0075 --output boxplot_coxph2_k1"
./run.sh "python ablation/main.py --loss cox-cc --n-layers 4 --approx-cox-cc 3-15cT4 --lr 0.025 --output boxplot_coxcc4_k1"
./run.sh "python ablation/main.py --loss cox-cc --n-layers 4 --approx-cox-cc 3-15cT4 --lr 0.01 --n-clients 50 --output boxplot_coxcc4_k50"

# polynomial activation sweep (CoxPH 4L, K=1). Unlisted cells diverge and show as --
./run.sh 'for iv in 5 10 15; do python ablation/main.py --loss coxph --n-layers 4 --approx-coxph 7-2-m-l8-u8-b16-r2.5-n4-s7 --lr 0.01 --activation 3-$iv --output act_3_$iv; done'
./run.sh 'for d in 5 7 9; do for iv in 10 15; do python ablation/main.py --loss coxph --n-layers 4 --approx-coxph 7-2-m-l8-u8-b16-r2.5-n4-s7 --lr 0.01 --activation $d-$iv --output act_${d}_$iv; done; done'

# network depth, 3 layers (the 2- and 4-layer cells reuse the runs above)
./run.sh "python ablation/main.py --loss cox-cc --n-layers 3 --approx-cox-cc 3-15cT4 --lr 0.035 --output boxplot_coxcc3_k1"
./run.sh "python ablation/main.py --loss coxph --n-layers 3 --approx-coxph 7-2-m-l8-u8-b16-r2.5-n4-s7 --lr 0.0075 --output boxplot_coxph3_k1"

# patient-level splits on the 339 verified spectra
./run.sh "python baseline/main.py --patient-grouped --verified-only --output baseline_ver"
./run.sh 'for k in 1 10 50; do python ablation/main.py --loss cox-cc --n-layers 2 --approx-cox-cc 3-15cT4 --lr 0.02 --n-clients $k --patient-grouped --verified-only --output patient_coxcc2_k$k; done'
./run.sh 'for k in 1 10 50; do python ablation/main.py --loss coxph --n-layers 4 --approx-coxph 7-2-m-l8-u8-b16-r2.5-n4-s7 --lr 0.0035 --n-clients $k --patient-grouped --verified-only --output patient_coxph4_k$k; done'

# label-skewed partitions, Dirichlet beta (the stratified row reuses the K=10 and K=50 runs above)
./run.sh 'for k in 10 50; do for b in 5.0 1.0 0.5; do python ablation/main.py --loss cox-cc --n-layers 2 --approx-cox-cc 3-15cT4 --lr 0.045 --n-clients $k --dirichlet-beta $b --output noniid_coxcc2_b${b}_k$k; done; done'
./run.sh 'for k in 10 50; do for b in 5.0 1.0 0.5; do python ablation/main.py --loss coxph --n-layers 4 --approx-coxph 7-2-m-l8-u8-b16-r2.5-n4-s7 --lr 0.01 --n-clients $k --dirichlet-beta $b --output noniid_coxph4_b${b}_k$k; done; done'

# print all tables
./run.sh "python ablation/make_tables.py"

# regenerate the federated boxplot figure (ablation/figures/)
./run.sh "python ablation/fig_boxplots.py"

# fully encrypted K=10 run on one fold (seed 81, fold 2), ~16 h, ~16 GB RAM
GPU=1 ./run.sh "python -m he.e2e.e2e_fold --epochs 100 --ckpt-every 1 --out he/e2e/e2e_rerun"
./run.sh "python he/e2e/plot_e2e.py he/e2e/e2e_rerun"
```

## Data

`data/` holds the processed inputs the pipeline consumes:

- `predicted_quant.pickle`: metabolite quantifications (model input)
- `survival.pickle`, `censorship.pickle`: survival times and event indicators
- `result.csv`: pathway mask (37 metabolites × 138 pathways)
- `patient_group.csv`: patient group and verified flag per spectrum, from earlier public
  releases of the cohort (read by `--patient-grouped` and `--verified-only`)

These derive from the HRMAS NMR brain-tumor dataset of the original
[PiDeeL study](https://doi.org/10.1093/bioinformatics/btad684). See the
manuscript for the raw-data source.
