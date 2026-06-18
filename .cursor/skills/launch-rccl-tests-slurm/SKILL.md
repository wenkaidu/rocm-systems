---
name: launch-rccl-tests-slurm
description: >-
  Launch rccl-tests inside a SLURM allocation to exercise the IB net-checksum
  feature on this branch. Use when the user wants to run rccl-tests, the csum
  stress/debug/single/soak sweeps, or any multi-node RCCL collective run under
  sbatch/salloc/srun on this cluster.
disable-model-invocation: true
---

# Launch rccl-tests in a SLURM allocation

The `tools/run_csum_*.sh` scripts run rccl-tests against the debug RCCL build to
stress the IB net-checksum path. They are SLURM-aware: inside an allocation they
auto-derive the host list and launch with `srun --mpi=pmi2` (no ssh/TTY needed),
which is what makes them work from a Cursor/agent terminal.

Reference copies of these scripts are bundled under `scripts/` next to this
file (kept in sync with `tools/` in the repo root). Prefer running the repo-root
`tools/` copies; read the `scripts/` copies as worked examples of the launch
pattern, env wiring, and result-parsing logic:

- [scripts/build_csum_debug.sh](scripts/build_csum_debug.sh) — debug build with the checksum gate ON.
- [scripts/build_csum_off.sh](scripts/build_csum_off.sh) — build with the checksum feature OFF (baseline).
- [scripts/run_csum_debug.sh](scripts/run_csum_debug.sh) — single cross-node AllReduce sanity check.
- [scripts/run_csum_stress.sh](scripts/run_csum_stress.sh) — full multi-node stress sweep (srun launcher).
- [scripts/run_csum_single.sh](scripts/run_csum_single.sh) — single-node intra-net sweep.
- [scripts/run_csum_stress_soak.sh](scripts/run_csum_stress_soak.sh) — repeat the stress sweep across cap configs.

## Prerequisites

- A debug RCCL build at `~/rccl/build/release` (build with `tools/build_csum_debug.sh`).
- rccl-tests `*_perf` binaries at `~/rocm-systems/projects/rccl-tests/build`.
- An active SLURM allocation (the scripts need `SLURM_JOB_ID` in the env).

## Step 1: Confirm / get an allocation

Check for an existing allocation first:

```bash
squeue -u "$USER"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
```

If `SLURM_JOB_ID` is set, reuse it (the scripts pass `--jobid` + `--overlap` so
they attach to the existing alloc). If not, request one, e.g.:

```bash
salloc -p amd-rccl -N 2 --ntasks-per-node=8 --gpus-per-node=8 -t 60:00
```

The Cursor terminal may only inherit `SLURM_JOB_ID` (not the nodelist). That is
fine — the scripts recover the nodelist via `squeue -h -j "$SLURM_JOB_ID" -o '%N'`
and `scontrol show hostnames`, so do NOT hand-edit host lists when in an alloc.

## Step 2: Pick the right script

| Script | Use it for |
|--------|-----------|
| `tools/run_csum_debug.sh` | One quick cross-node AllReduce (64M) — fastest sanity check. |
| `tools/run_csum_stress.sh` | Full multi-node sweep over every `*_perf` collective, `-b 8 -e 1G`, with validation. |
| `tools/run_csum_single.sh` | Single-node sweep (flips `RCCL_ENABLE_INTRANET=1` + `RCCL_P2P_NET_DISABLE=0` so intra-node pairs route over IB). |
| `tools/run_csum_stress_soak.sh` | Repeat the stress sweep N cycles per `RCCL_IB_RDMA_CHECKSUM_BYTES` cap to catch intermittent failures. |

## Step 3: Launch

Run from the repo root. Host list, node count, and tasks-per-node are derived
from the allocation automatically:

```bash
# quick sanity check
tools/run_csum_debug.sh

# full multi-node stress sweep (auto log dir ~/csum_stress_<N>/)
tools/run_csum_stress.sh

# restrict collectives
ONLY="all_reduce_perf alltoall_perf" tools/run_csum_stress.sh
```

These can run long. Start them in the background and monitor the log dir rather
than blocking the turn.

## Step 4: Read results

The stress/single scripts write per-collective logs plus a `SUMMARY.txt` to the
log dir (printed at startup). Each row reports `exit / #wrong / csum_mm / warn`,
and any nonzero is flagged `FAIL`:

```bash
grep -E '^(#|FAIL| )' ~/csum_stress_<N>/SUMMARY.txt
grep -rn "net recv csum mismatch" ~/csum_stress_<N>/   # kernel mismatch detail
```

`overall: PASS/FAIL` is the last summary line; the script exit code mirrors it.

## Key environment overrides

All are optional; defaults live at the top of each script.

- `HOSTS=host1:8,host2:8` — explicit host list (always wins over SLURM derivation).
- `NP=16` — total ranks (defaults to the sum of `:N` GPU counts in `HOSTS`).
- `GPUS_PER_NODE=8` — per-node GPU count for SLURM-derived hosts.
- `LAUNCHER=auto|srun|mpirun` — `auto` picks `srun` when `SLURM_JOB_ID` is set.
- `RCCL_IB_RDMA_CHECKSUM=1` — gate the kernel XOR + IB IMM checksum (the feature under test).
- `RCCL_IB_RDMA_CHECKSUM_BYTES=0` — per-slot byte cap (0 = no cap).
- `DTYPE / MIN_BYTES / MAX_BYTES / STEP_FACTOR / WARMUP / ITERS` — rccl-tests `-d/-b/-e/-f/-w/-n`.
- `RCCL_TESTS_BIN_DIR`, `RCCL_BUILD_DIR`, `MPIRUN_BIN` — override binary/library paths.

## Why srun (not mpirun) inside an allocation

mpich `mpirun` launches remote ranks over ssh, which stalls from a non-TTY
agent terminal. `srun --mpi=pmi2` goes through slurmd/PMI and needs no ssh or
controlling TTY, so it works from any shell. The `auto` launcher uses `srun`
whenever `SLURM_JOB_ID` is present; force it with `LAUNCHER=srun` if needed.
