---
name: launch-rccl-tests-slurm
description: >-
  Run rccl-tests inside a SLURM allocation or queue jobs via sbatch: quick sanity,
  single-node stress, multi-node stress, soak loops, randomized alltoallv /
  P2P-channel trials, and PAT-style all_gather_perf (1 rank per node, -g 1). Use
  when the user wants to drive *_perf binaries with srun (or mpirun) from a
  non-interactive shell, without editing host lists by hand.
disable-model-invocation: true
---

# Launch rccl-tests in a SLURM allocation

Scripts live under `.cursor/skills/launch-rccl-tests-slurm/scripts/`. They are
generic: they do not enable or document any checksum-specific RCCL options.
Export normal `NCCL_*` / `RCCL_*` tuning yourself before launching.

## Prerequisites

- **SLURM allocation** for multi-node runs (`salloc` / interactive `srun --pty`),
  **or** use [scripts/submit_rccl_tests_slurm.sh](scripts/submit_rccl_tests_slurm.sh)
  to **queue via `sbatch`** (no shell on compute nodes required). The runners
  recover the nodelist from `SLURM_JOB_ID` when the shell only inherited the job
  id (e.g. Cursor agent terminal).
- **rccl-tests** `*_perf` binaries (default path `~/rocm-systems/projects/rccl-tests/build`).
- **RCCL** shared library on `LD_LIBRARY_PATH` (default `~/rocm-systems/projects/rccl/build/release` and `.../lib` when present, via
  `RCCL_BUILD_DIR`).
- **MPI**: MPICH-style `mpirun` for non-SLURM fallback (default
  `~/mpich/install/bin/mpirun`). Inside an allocation, **`srun --mpi=pmi2`** is
  preferred so ranks start without ssh.

## Scripts

| Script | Purpose |
|--------|---------|
| [scripts/build_rccl.sh](scripts/build_rccl.sh) | Configure + build RCCL into `~/rocm-systems/projects/rccl/build/release` (or `RCCL_BUILD_DIR`). Pass extra CMake flags as argv. |
| [scripts/run_rccl_tests_quick.sh](scripts/run_rccl_tests_quick.sh) | One short correctness run (default `all_reduce_perf`, 64 MB). |
| [scripts/run_rccl_tests_stress_multi_node.sh](scripts/run_rccl_tests_stress_multi_node.sh) | Full sweep of `*_perf` on all allocation nodes; logs + `SUMMARY.txt`. |
| [scripts/run_rccl_tests_stress_single_node.sh](scripts/run_rccl_tests_stress_single_node.sh) | Same sweep on **one** node (`srun -N1` in an alloc, or local `mpirun`). |
| [scripts/run_rccl_tests_stress_soak.sh](scripts/run_rccl_tests_stress_soak.sh) | Run the multi-node stress script `CYCLES` times; writes `AGGREGATE.txt`. |
| [scripts/run_rccl_tests_alltoallv_p2p_channels_random.sh](scripts/run_rccl_tests_alltoallv_p2p_channels_random.sh) | **`alltoallv_perf`**: `COMBOS` trials (default 100) with random `NP` in `1..allocation` and random `NCCL_MAX_P2P_NCHANNELS` ∈ {1,2,4,8,16,32,64}; `NCCL_MIN_P2P_NCHANNELS=1`. |
| [scripts/run_rccl_tests_pat_all_gather_1gpu_per_node.sh](scripts/run_rccl_tests_pat_all_gather_1gpu_per_node.sh) | **`all_gather_perf`**: **every allocated node**, one MPI rank per node (**`GPUS_PER_NODE=1`**, ignores preset **`NP`**), **`srun --nodes=$NNODES`**, **`-g 1`**; defaults **`NCCL_DEBUG=INFO`**, **`NCCL_DEBUG_SUBSYS=INIT,TUNING`** (algorithm lines), **`NCCL_PAT_ENABLE=1`**, **2 MiB** sweep (`-b 2M -e 2M`); override via **`PAT_ALLGATHER_ARGS`** / **`NCCL_PAT_ENABLE`**. |
| [scripts/common_rccl_tests_slurm.sh](scripts/common_rccl_tests_slurm.sh) | Shared bash helpers (sourced by the runners). |
| [scripts/submit_rccl_tests_slurm.sh](scripts/submit_rccl_tests_slurm.sh) | **`sbatch`** wrapper: queue any runner on `amd-rccl` (defaults match `srun --pty -N 2 -C block3\|block4 --ntasks-per-node=64 --gres=gpu:8 -p amd-rccl -t 4:00:00 --qos=urgent`). |

## Step 1: Allocation and paths

```bash
squeue -u "$USER"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"

# Optional: pin RCCL / tests / MPI locations
export RCCL_BUILD_DIR="${RCCL_BUILD_DIR:-$HOME/rocm-systems/projects/rccl/build/release}"
export RCCL_TESTS_BIN_DIR="${RCCL_TESTS_BIN_DIR:-$HOME/rocm-systems/projects/rccl-tests/build}"
```

Request an allocation if needed, for example:

```bash
salloc -N 2 --ntasks-per-node=8 --gpus-per-node=8 -t 4:00:00
# or interactive (this site):
srun --pty -N 2 -C "block3|block4" --ntasks-per-node=64 --gres=gpu:8 \
  -p amd-rccl -t 4:00:00 --qos=urgent bash
```

**Or queue a batch job** (no allocation in your shell):

```bash
# Quick all_reduce on 2 nodes (default)
./.cursor/skills/launch-rccl-tests-slurm/scripts/submit_rccl_tests_slurm.sh

# PAT all_gather, 8 nodes, wait for completion
SLURM_NNODES=8 \
  RUN_SCRIPT=run_rccl_tests_pat_all_gather_1gpu_per_node.sh \
  PAT_ALLGATHER_ARGS="-b 2M -e 128M -f 2 -g 1 -d bfloat16 -w 2 -n 5 -c 0" \
  NCCL_PAT_ENABLE=1 NCCL_ALGO=PAT \
  WAIT=1 \
  ./.cursor/skills/launch-rccl-tests-slurm/scripts/submit_rccl_tests_slurm.sh

# Multi-node stress sweep on 4 nodes
SLURM_NNODES=4 RUN_SCRIPT=run_rccl_tests_stress_multi_node.sh \
  ./.cursor/skills/launch-rccl-tests-slurm/scripts/submit_rccl_tests_slurm.sh
```

`submit_rccl_tests_slurm.sh` writes sbatch logs under `~/rccl_slurm_jobs/` by
default. Override `SLURM_PARTITION`, `SLURM_QOS`, `SLURM_CONSTRAINT`, `SLURM_TIME`,
etc. to match your account. Export `NCCL_*` / `RCCL_*` before submit — they are
forwarded into the batch step.

## Step 2: Cluster tuning (your responsibility)

Set **NIC / topology** variables your site requires before running, for example:

```bash
export NCCL_IB_HCA=mlx5_0,mlx5_1   # example only
export NCCL_IB_TC=41               # example only
```

The runners set mild defaults (`NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS`, affinity
knobs). They do **not** hard-code a vendor-specific HCA list.

## Step 3: Run

From the repo root (or any cwd); use absolute paths to the scripts if you prefer:

```bash
# Build RCCL (optional extra CMake flags at end)
./.cursor/skills/launch-rccl-tests-slurm/scripts/build_rccl.sh

# Quick cross-node check (uses full allocation)
./.cursor/skills/launch-rccl-tests-slurm/scripts/run_rccl_tests_quick.sh

# Multi-node matrix (auto log dir under ~/rccl_tests_stress_<n>/)
ONLY="all_reduce_perf alltoall_perf" \
  ./.cursor/skills/launch-rccl-tests-slurm/scripts/run_rccl_tests_stress_multi_node.sh

# Single-node saturation inside the same allocation
./.cursor/skills/launch-rccl-tests-slurm/scripts/run_rccl_tests_stress_single_node.sh

# Soak: repeat multi-node stress (default 5 cycles)
CYCLES=10 ./.cursor/skills/launch-rccl-tests-slurm/scripts/run_rccl_tests_stress_soak.sh

# Random alltoallv vs NCCL_MAX_P2P_NCHANNELS (100 trials, default size sweep to 256M)
./.cursor/skills/launch-rccl-tests-slurm/scripts/run_rccl_tests_alltoallv_p2p_channels_random.sh
```

### Launcher selection

- `LAUNCHER=auto` (default): **`srun`** if `SLURM_JOB_ID` is set, else **`mpirun`**.
- `LAUNCHER=srun` / `LAUNCHER=mpirun`: force one launcher.
- Multi-node **`mpirun`** uses `--hosts "${HOSTS}"`** and needs working ssh to compute nodes.

### Host list

If `HOSTS` is unset, it is built from the allocation nodelist and
`GPUS_PER_NODE` (from `SLURM_GPUS_ON_NODE`, `SLURM_GPUS_PER_NODE`,
`SLURM_NTASKS_PER_NODE`, or **8**). Outside SLURM you must set **`HOSTS`**
explicitly for multi-node scripts.

### Stress parameters (env)

| Variable | Typical use |
|----------|-------------|
| `ONLY` / `SKIP` | Subset of `*_perf` binaries |
| `MIN_BYTES` / `MAX_BYTES` / `STEP_FACTOR` | Size sweep (`-b` / `-e` / `-f`) |
| `WARMUP` / `ITERS` / `GPUS_PER_THREAD` | `-w` / `-n` / `-g` |
| `DTYPE` | rccl-tests `-d` (default `bfloat16`) |
| `NP` | Total ranks (default: sum of `:gpu` counts in `HOSTS`) |

Each run writes per-collective logs and **`SUMMARY.txt`** with exit status,
aggregated `#wrong` from rccl-tests tables, and counts of `NCCL`/`RCCL`
`WARN`/`ERROR` lines.

## Copying into `tools/` (optional)

To match a `tools/` workflow, symlink or copy these scripts next to your other
helpers; behavior is identical.
