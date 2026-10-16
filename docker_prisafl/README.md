# Docker reproduction environment

Self-contained image to run the reproduction package (baseline + ablation +
fully encrypted `he/`). It pins PyTorch 2.7.1 and OpenFHE 1.3.0.
This is the only supported environment.

## What's inside

- **One Python 3.11 env**
- **OpenFHE 1.3.0**

## Which Docker — native engine vs Docker Desktop

`build.sh` defaults to the **native Docker engine** (`/var/run/docker.sock`),
falling back to `sudo docker` if you're not in the `docker` group.

- **Linux → native engine (recommended).** Fastest. Just `./build.sh`.
- **macOS / Windows, or Linux Docker Desktop → `DESKTOP=1 ./build.sh`.** Docker
  Desktop runs a Linux VM: everything works, just slower on HE, and on a
  flaky VM network a build may retry downloads automatically (see Troubleshooting).

## Quick start

```bash
# 1. build the image (compiles OpenFHE from source). ~30 min.
./build.sh                 # DESKTOP=1 ./build.sh to build on Docker Desktop
#                            GENERIC=1 ./build.sh -> portable build if native is flaky (see Notes)

# 2. verify correctness + result parity
./verify.sh                # HE encrypted-vs-plaintext < 1e-3; plaintext Pri-SAFL parity (CPU)
GPU=1 ./verify.sh          # run the plaintext ablation on a GPU (optional)
```

## Running

```bash
./run.sh

# encrypted pipeline
./run.sh "python -m he.tests.test_pipeline --stage multi_iter --n-samples 3 --n-iters 1 --bootstrap --dims 37 138 1 --seed 42 --ring-dim-exp 15"
./run.sh "python -m he.main --n-samples 3 --epochs 10 --bootstrap --timing --dims 37 138 1 --seed 42 --ring-dim-exp 15"

# plaintext ablation / baseline (add GPU=1 to use a GPU)
./run.sh "python ablation/main.py --loss cox-cc --n-layers 2 --approx-cox-cc 3-15cT4 --lr 0.045 --output boxplot_coxcc2_k1"
./run.sh "python baseline/main.py --output baseline_result"
./run.sh "python ablation/make_tables.py"
```

**Run HE commands one at a time.** Bootstrap needs ~32+ GB RAM.

Without `--ring-dim-exp`, HE uses the 128-bit rings of the manuscript (2^17 for one client,
2^16 for `--n-clients` > 1). The commands above pass `--ring-dim-exp 15` (not 128-bit secure)
to fit in ~32 GB. 2^17 needs ~4x that RAM.

## Environment variables

**Run-time** (`run.sh`):

| var | default | meaning |
|-----|---------|---------|
| `GPU` | `0` | `1` → `--gpus all` on the system dockerd (needs nvidia-container-toolkit) |
| `SYS` | `0` | `1` → run on the system/bare-metal dockerd (`/var/run/docker.sock`) instead of the current `docker` context |
| `CPUSET` | all cores | pin to specific cores, e.g. `0-15` (full-speed OpenMP) |
| `IMAGE` | `prisafl` | image tag to run |

**Build-time** (`build.sh`):

| var | default | meaning |
|-----|---------|---------|
| `GENERIC` | `0` | `1` → portable/safe build (`WITH_NATIVEOPT=OFF` + `-march=x86-64`) vs default max-opt `native`. Use where native is unstable — e.g. Ryzen znver3 (see Notes) |
| `DESKTOP` | `0` | `1` → build on Docker Desktop instead of the native engine |
| `MINIFORGE_VERSION` | `25.11.0-1` | conda installer pin |
| `OPENFHE_DEV_TAG` | `v1.3.0` | OpenFHE C++ source tag |
| `OPENFHE_PY_TAG` | `v1.3.0.0` | OpenFHE python-binding source tag |
| `IMAGE` | `prisafl` | output image tag |

## Notes

- **Native-build stability caveat.** `-march=native` is fastest, but on some CPUs
  the native OpenFHE codegen is numerically unstable. **Haswell/Xeon and most server
  CPUs are unaffected.** If you encounter such an issue (or want a deterministic build), use
  `GENERIC=1 ./build.sh` → `WITH_NATIVEOPT=OFF` + `-march=x86-64`: fully stable,
  at ~20% slower HE (bootstrap-bound).
- On Linux, Docker Desktop runs containers in a VM (slower on the HE
  workload). Prefer a native daemon, or just rebuild on the target.
- **Build fails with a corrupted download** — "THESE PACKAGES DO NOT MATCH THE
  HASHES", or a `ZIP bad CRC` / `Error reading from archive` from the miniforge
  installer. A package arrived mangled: a CDN/mirror hiccup (happens on bare
  metal too) or Docker Desktop's VM NAT. The Dockerfile wraps every download in a
  retry loop that re-fetches automatically, so it usually self-heals; if a build
  still dies, just re-run `./build.sh`.
- GPU affects the plaintext runs and the e2e initial weights. HE itself is CPU-only.
