---
name: launch-rccl-tests-slurm
description: >-
  Launch rccl-tests inside a SLURM allocation to exercise the IB net-checksum
  feature on this branch, or to benchmark collective latency across env-var
  combinations and node counts. Use when the user wants to run rccl-tests, the
  csum stress/debug/single/soak sweeps, an env-var on/off combination latency
  sweep, or any multi-node RCCL collective run under sbatch/salloc/srun on this
  cluster.
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
- [scripts/run_env_combo_sweep.sh](scripts/run_env_combo_sweep.sh) — benchmark a collective across every ON/OFF combination of a set of env vars, node counts, and repeat cycles (data collection).
- [scripts/parse_env_combo_latency.py](scripts/parse_env_combo_latency.py) — parse that sweep into per-node-count median-latency Markdown tables (parsing).

## Prerequisites

- A debug RCCL build at `~/rccl/build/release` (build with `tools/build_csum_debug.sh`).
- rccl-tests `*_perf` binaries at `~/rocm-systems/projects/rccl-tests/build`.
- An active SLURM allocation (the scripts need `SLURM_JOB_ID` in the env).

> **Always build RCCL from clean.** RCCL's device-kernel link step (`device.elf`)
> does not list the per-kernel object files as make prerequisites — they are
> passed to the linker via a response file — so an incremental `make` can report
> "Built target" while silently re-linking **stale device code**. Edits to device
> headers/sources (`prims_*.h`, `sendrecv.h`, etc.) then never reach the running
> `librccl.so`, and you end up debugging a binary that does not match your source
> (e.g. a "fixed" hang that still reproduces). A full clean RCCL build only takes
> ~2 minutes, so always build from clean rather than incrementally. If you must
> build incrementally, force the device relink by removing the stale products
> first, e.g. `rm -f build/release/device_build/-*/device.elf
> build/release/device_build/device.hipfb build/release/device_build/common.o
> build/release/librccl.so.1.0` before `make`.

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

## Ruby cluster (bnxt_re NICs)

The `ruby` cluster (login node `ruby-slurmlogin01`, partitions `meta64` /
`interactive`, nodes `cv350-rck-*`) uses **Broadcom `bnxt_re` RoCE NICs**, not
the Mellanox `mlx5` HCAs the `run_csum_*.sh` defaults assume. Multi-node runs
will silently hang during connection setup unless you override the fabric env.
Each node has 8 GPUs; a 2-node run is `-N 2 --ntasks-per-node=8` (16 ranks).

Required per-rank env on ruby (pass via `srun`'s inherited env, or `-x`/`-env`
with mpirun):

```bash
# ruby bnxt_re RoCE fabric
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_IB_HCA=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7
export NCCL_SOCKET_IFNAME=fenic0,enp49s0f0np0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=104
# rccl-tests baseline env
export HSA_NO_SCRATCH_RECLAIM=1
export RCCL_MSCCL_ENABLE=0
export RCCL_IB_QPS_PER_P2P=1
export RCCL_P2P_BATCH_ENABLE=0
export NCCL_DEBUG=VERSION
export RSMI_MUTEX_THREAD_ONLY=1
export NCCL_IB_QPS_PER_CONNECTION=4
```

With `run_csum_stress.sh`, override the defaults inline, e.g.:

```bash
NCCL_IB_HCA=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7 \
NCCL_IB_TC=104 \
tools/run_csum_stress.sh
```

(`NCCL_SOCKET_IFNAME` and `NCCL_IB_GID_INDEX` are ruby-specific and are not in
the script's env list — export them in the shell before launching so they
propagate through `srun --export=ALL`.)

Verified working with a 2-node `reduce_scatter_perf -d bfloat16 -b 8 -e 1G -f 2`
(16 ranks, srun `--mpi=pmi2`): 0 `#wrong`, ~335 GB/s busbw at 1 GB.

## Env-var combination latency sweeps

To A/B (or N-way) compare RCCL/NCCL env-var settings on a collective's latency
across scales, use the generic sweep + parse pair. It runs one full size sweep
of a collective for **every ON/OFF (0/1) combination** of the named env vars, at
each node count, repeated over several cycles, then reports the **per-size
median** latency (median rejects run-to-run outliers).

`run_env_combo_sweep.sh` is configured entirely through the environment:

| Var | Default | Meaning |
|-----|---------|---------|
| `COLL` | `alltoall_perf` | any rccl-tests `*_perf` binary |
| `SWEEP_VARS` | `NCCL_ALLOC_P2P_NET_LL_BUFFERS RCCL_GFX9_CHEAP_FENCE_OFF` | space-separated env vars, each toggled 0/1 → 2^k combos |
| `NODE_COUNTS` | `1 2 4 8 16` | node counts to test (all must fit in the allocation) |
| `CYCLES` | `5` | repeats per combo (outer loop, spread in time) |
| `FLAGS` | `-b 8 -e 1G -f 2 -g 1` | rccl-tests size-sweep flags |
| `RCCL_LIB_DIR` | `~/rccl_libs/sel` | dir containing the `librccl.so.1` under test |
| `RCCL_TESTS_BIN_DIR` | `~/rccl-tests/build` | where `$COLL` lives |
| `OUT` | `~/logs/env_combo_sweep` | output dir (logs + `sweep_meta.env`) |

Cost is `len(NODE_COUNTS) × 2^k × CYCLES` runs — e.g. the defaults are
`5 × 4 × 5 = 100` runs (~30 min on 16 nodes). Add a var and it doubles.

**Pin the library under test** by pointing `librccl.so.1` at a specific build.
rccl-tests links `librccl.so.1`, so front-load a dir that resolves it:

```bash
mkdir -p ~/rccl_libs/sel
ln -sf ~/rccl_libs/librccl.so.1.0.<hash> ~/rccl_libs/sel/librccl.so.1
```

Launch inside an allocation sized to the largest node count. `meta64`
interactive allocations are capped at **240 minutes** (use `sbatch` for longer):

```bash
# example: sweep two vars for alltoall on 1/2/4/8/16 nodes, 5 cycles
salloc -p meta64 -N 16 --ntasks-per-node=8 -t 3:59:00 \
  bash .cursor/skills/launch-rccl-tests-slurm/scripts/run_env_combo_sweep.sh
```

These are long; start them in the background (`nohup ... &`) and monitor the
run count (`ls $OUT/att__*.log | wc -l`) or `grep RUN $OUT/driver.log | tail`.

Then build the tables (one per node count; columns = combos labelled `A0B1…`
with a legend mapping each letter to its env var):

```bash
python3 .cursor/skills/launch-rccl-tests-slurm/scripts/parse_env_combo_latency.py \
  ~/logs/env_combo_sweep            # writes latency_by_size.md in that dir
```

Latency is the out-of-place `time` (µs) column; sub-rank-count sizes that round
to 0 B are dropped. Env effects are usually strongly size- and scale-dependent,
so read the per-size tables rather than a single sweep-averaged number — a knob
that helps small messages at 16 nodes may regress the mid-size band and be pure
noise on 1 node.

## Why srun (not mpirun) inside an allocation

mpich `mpirun` launches remote ranks over ssh, which stalls from a non-TTY
agent terminal. `srun --mpi=pmi2` goes through slurmd/PMI and needs no ssh or
controlling TTY, so it works from any shell. The `auto` launcher uses `srun`
whenever `SLURM_JOB_ID` is present; force it with `LAUNCHER=srun` if needed.
