---
name: launch-rccl-tests-slurm
description: >-
  Launch rccl-tests inside a SLURM allocation to exercise the IB net-checksum
  feature on this branch, benchmark collective latency across env-var
  combinations and node counts, or build/test RCCL against specific ROCm
  module versions on the single-node MI300A `lockhart` cluster. Use when the
  user wants to run rccl-tests, the csum stress/debug/single/soak sweeps, an
  env-var on/off combination latency sweep, a single-node gfx942/MI300A
  build-and-test pass (e.g. ROCm version compatibility checks), or any RCCL
  collective run under sbatch/salloc/srun on this cluster.
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

> **Always drive long / unattended sweeps with `sbatch`, never `nohup … salloc … &`.**
> A `salloc` (or a `nohup`-backgrounded driver that owns a `salloc`) is tied to
> the login-session process tree. When that session is cleaned up — logout, an
> agent/terminal teardown, an idle-session reaper, or a stray `scancel -n bash` —
> the **entire driver + `salloc` + `srun` tree is killed**, and the underlying
> SLURM allocations are left **orphaned** (still holding nodes, doing nothing,
> while the multi-`N` driver loop silently dies partway through). `sbatch` jobs
> are owned by `slurmctld`, not your shell, so they survive session cleanup and
> restart-safely record their own state. **Submit one `sbatch` job per node
> count** so they are independent and can even run in parallel:
>
> ```bash
> for N in 1 2 4 8 16; do
>   sbatch --parsable -p meta64 -N "$N" --ntasks-per-node=8 -t 3:00:00 \
>          -J sweep_N$N -o ~/logs/sweep_N${N}.out \
>          --wrap "bash ~/logs/<inner_sweep>.sh"   # inner reads SLURM_NNODES
> done
> squeue -u "$USER" -o '%.8i %.9j %.2t %.6M %.5D %R'   # monitor
> ```
>
> Only use `salloc` for short, interactive, actively-watched runs (a quick
> sanity check you will not walk away from). If you ever see running jobs in
> `squeue` with **no** matching `salloc`/`srun`/driver process in `ps -u "$USER"`,
> they are orphans — `scancel` them and resubmit via `sbatch`.

## Step 1: Confirm / get an allocation

Check for an existing allocation first:

```bash
squeue -u "$USER"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
```

If `SLURM_JOB_ID` is set, reuse it (the scripts pass `--jobid` + `--overlap` so
they attach to the existing alloc). If not, request one. For a **short,
interactive** run you will actively watch, `salloc` is fine:

```bash
salloc -p amd-rccl -N 2 --ntasks-per-node=8 --gpus-per-node=8 -t 60:00
```

For anything **long or unattended, use `sbatch` instead** (see the callout above)
— a `salloc`/`nohup` driver dies with your login session and orphans its nodes.

The Cursor terminal may only inherit `SLURM_JOB_ID` (not the nodelist). That is
fine — the scripts recover the nodelist via `squeue -h -j "$SLURM_JOB_ID" -o '%N'`
and `scontrol show hostnames`, so do NOT hand-edit host lists when in an alloc.

## Step 2: Pick the right script

| Script | Use it for |
|--------|-----------|
| `tools/run_csum_debug.sh` | One quick cross-node AllReduce (64M) — fastest sanity check. |
| `tools/run_csum_stress.sh` | Full multi-node sweep over every `*_perf` collective, `-b 8 -e 1G`, with validation. |
| `tools/run_csum_single.sh` | Single-node sweep (flips `RCCL_ENABLE_INTRANET=1` + `RCCL_P2P_NET_DISABLE=0` so intra-node pairs route over IB). **Verify the NET path was actually taken** — `RCCL_ENABLE_INTRANET=1` alone routes `via P2P/IPC`; see "Forcing intra-node send/recv over the network" below. |
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

## Lockhart cluster (single-node MI300A, gfx942)

The `lockhart` cluster (partition `MI300A_A1`, node names like
`x1001c0s1b1n0`) is a **single node with 4 MI300A GPUs** (gfx942), fed by
Cray MPICH (`/opt/cray/pe/mpich/<ver>/ofi/crayclang/<ver>`) instead of
OpenMPI. Multiple ROCm versions are available as modules (`module avail
rocm`) under `/opt/COE_modules/rocm/rocm-<ver>/`; use `module show
rocm/<ver>` to read off `ROCM_PATH` when building/testing against a
specific version (e.g. to check backward compatibility of a change).

### Allocation

Request the node with `-G 4` (4 GPUs) — without it the allocation has no
GPUs bound:

```bash
srun -N 1 -n 64 -G 4 -p MI300A_A1_COS_OK --pty bash -i
```

Once on the allocated node, the ROCm modulefiles live under a non-default
path, so add it before loading a version:

```bash
module use /opt/COE_modules/modulefiles/rocm
module load rocm/7.13.0-gfx94x
```

### Build

```bash
# RCCL — target gfx942 only, both to save build time and to avoid pulling
# in unrelated-arch failures (see gotcha below)
ROCM_PATH=/opt/COE_modules/rocm/rocm-<ver> ./install.sh --amdgpu_targets gfx942 -j 96

# rccl-tests — the Makefile reads ROCM_PATH, NOT HIP_HOME
cd projects/rccl-tests
make MPI=1 MPI_HOME=/opt/cray/pe/mpich/9.1.0/ofi/crayclang/20.0 \
     ROCM_PATH=/opt/COE_modules/rocm/rocm-<ver> \
     RCCL_HOME=<rccl_repo>/build/release NCCL_HOME=<rccl_repo>/build/release \
     GPU_TARGETS=gfx942 -j 32
```

**Gotcha — `GPU_TARGETS`:** the default builds for `gfx906 gfx908 gfx90a
gfx942 gfx950 gfx1030 gfx1100 …`. Compiling for the unrelated `gfx1030`
target can fail on headers unrelated to your change (e.g. `nccl_device`
template errors) and wastes build time. Always pass `GPU_TARGETS=gfx942`
when you only need to test on this node.

**Gotcha — `HIP_HOME` vs `ROCM_PATH`:** rccl-tests' `src/Makefile` derives
the compiler as `$(ROCM_PATH)/llvm/bin/amdclang++`. Passing `HIP_HOME=`
instead does nothing — the build silently falls back to whatever `hipcc` is
first on `PATH` (check `hipcc --version`'s `InstalledDir` if you get symbol
errors that don't match the ROCm version you intended).

### Run

The Cursor/agent shell attached to an allocation typically does **not**
have `SLURM_JOB_ID` set (the login/agent shell is not the job's own shell),
so a bare `srun` will silently schedule a brand-new job — possibly on a
different, GPU-less node/partition. Always check `squeue --me` for your job
id first, then:

```bash
# single process, all 4 GPUs — no srun needed, just run the binary
LD_LIBRARY_PATH=<rccl_repo>/build/release:$ROCM_PATH/lib:$LD_LIBRARY_PATH \
  ./build/all_reduce_perf -b 8 -e 256M -f 2 -g 4

# 1 GPU per MPI rank — explicit --jobid + --overlap; the interactive shell
# already occupies step .0 on that job, and --overlap lets a new step share it
LD_LIBRARY_PATH=<rccl_repo>/build/release:$ROCM_PATH/lib:$LD_LIBRARY_PATH \
  srun --jobid=<job_id> --overlap -N1 -n4 ./build/all_reduce_perf -b 8 -e 256M -f 2 -g 1
```

**Gotcha — do not add GPU-binding flags** (`--gpus-per-task=1`,
`--gpu-bind=…`) to the `srun` above. rccl-tests picks each rank's device as
`localRank*nGpus + i` (`src/util.cu`), assuming **all** GPUs stay visible to
every rank; restricting visibility per task makes ranks >0 see only 1
device and fail with `Invalid number of GPUs: N requested but only 1 were
found`. Leave GPU visibility unrestricted and let the app index into the
full set.

Verify GPUs are idle/healthy before/after a run with `rocm-smi` (0% GPU%,
low power draw, no stray `all_reduce_perf`/`srun` processes in `ps aux`).

### Checking ROCm version compatibility

To verify a change doesn't regress against older ROCm releases, repeat the
build+run above once per `module avail rocm` version (rebuild RCCL **and**
rccl-tests from clean each time — `rm -rf build` — since headers/ABI differ
across versions). Known results for the `HIP_HOST_UNCACHED_MEMORY` /
uncached-host-alloc change (PR #8352, commit `faa9111f85`):

| ROCm version | Result |
|---|---|
| 6.4.3 | ❌ fails — pre-existing, unrelated bug: `src/misc/rocmwrap.cc` sets `prop.requestedHandleTypes` (plural) unconditionally, but 6.4.3's `hipMemAllocationProp` only has the singular `requestedHandleType`. Needs a `#if NCCL_CUMEM_VERSION_SUPPORTED(HIP_VERSION)` guard (see `src/include/rocmwrap.h`) to fix; not caused by this PR. |
| 7.0.0 | ✅ pass (0 wrong) — missing `hipMemcpyBatchAsync`/`hipMemLocationTypeHostNuma` natively, but `nccl_device` already falls back to `CU_MEM_LOCATION_TYPE_HOST_NUMA`, so it's just a benign `-Wtautological-constant-out-of-range-compare` warning, not a build error. |
| 7.1.0 | ✅ pass (0 wrong) — has all symbols the PR touches. |
| 7.2.0 | ✅ pass (0 wrong) — has all symbols the PR touches. |
| 7.13.0-gfx94x | ✅ pass (0 wrong) — reference/primary build target. |

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

These sweeps are long, so **submit them with `sbatch`, not `salloc`+`nohup`**
(see the top-of-file callout — a `salloc` driver dies with your login session and
orphans its nodes). Submit one job per node count so they are independent and can
run in parallel:

```bash
# example: sweep two vars for alltoall on 1/2/4/8/16 nodes, 5 cycles
for N in 1 2 4 8 16; do
  sbatch --parsable -p meta64 -N "$N" --ntasks-per-node=8 -t 3:59:00 \
         -J envsweep_N$N -o ~/logs/envsweep_N${N}.out \
         --wrap "NODE_COUNTS=$N bash .cursor/skills/launch-rccl-tests-slurm/scripts/run_env_combo_sweep.sh"
done
```

(`meta64` interactive allocations are capped at **240 minutes**, another reason
to prefer batch jobs.) Monitor with `squeue -u "$USER"`, the run count
(`ls $OUT/att__*.log | wc -l`), or `grep RUN $OUT/driver.log | tail`.

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

## Chained sbatch jobs (build → test) that outlive your interactive allocation

Long unit-test or benchmark runs (`rccl-UnitTests`, multi-hour soaks) will
outlast a `salloc`/interactive `srun` that is reclaimed when your session ends.
Split the work into **two `sbatch` jobs — a build job and a run job chained with
`--dependency=afterok`** — so the run only starts if the build succeeds and both
survive session cleanup (they are owned by `slurmctld`, not your shell):

```bash
cd ~/logs
JID_BUILD=$(sbatch --parsable ut_build.sbatch)                        # build librccl + tests
JID_RUN=$(sbatch --parsable --dependency=afterok:$JID_BUILD ut_run.sbatch)
echo "$JID_BUILD $JID_RUN" > ut_jobids.txt                            # persist ids
```

Each script carries its own `#SBATCH` header (this site: `--partition=meta64
--account=vip --qos=normal`, 8 GPUs/node — read them off a running job with
`scontrol show job <id> | grep -oE 'Account=[^ ]+|Partition=[^ ]+|QOS=[^ ]+'`)
and writes to a **persistent log under `~/logs`** (`--output=…%j.log`) plus a
final results file, so the verdict is readable after your allocation is gone.
The run script should `tee` the full output and end with a summary + exit code:

```bash
stdbuf -oL -eL "$BUILD/test/rccl-UnitTests" 2>&1 | tee -a "$RESULT"
rc=${PIPESTATUS[0]}; echo "rc=$rc"; grep -E '\[  (PASSED|FAILED)  \]' "$RESULT" | tail
```

Watch the build finish **before** you end your turn (an `afterok` dependency
cancels the run job if the build fails). Check status later with
`squeue -j <run_jid>` (empty = finished) and `tail ~/logs/<results>.txt`.

Gotchas:

- **Never `exec` a shared library** to print its version — running
  `"$BUILD/librccl.so"` directly segfaults. Print the RCCL banner via a test
  binary with `NCCL_DEBUG=VERSION` instead.
- `rccl-UnitTests` is a **single task** (`--ntasks-per-node=1`) that forks its
  own child processes for the multi-process (MP) cases and uses all 8 GPUs on the
  node. It needs `librccl.so` **and** the OMPI runtime on `LD_LIBRARY_PATH`
  (`/opt/sre-tools/ompi/lib`, for `libmpi.so.40`); rccl-tests `*_perf` binaries
  additionally need `srun --mpi=pmix` to launch ranks.

## Forcing intra-node send/recv over the network (single node)

To exercise **network-transport** code paths on a single node — e.g. the P2P
LL/LL128 net staging buffers governed by `NCCL_ALLOC_P2P_NET_LL_BUFFERS`, or the
IB checksum feature — you must make intra-node peers use the `NET` transport
instead of the faster on-node transports. `RCCL_ENABLE_INTRANET=1` alone is
**not** enough: it only keeps the NIC in the single-node topology (prevents
trimming); the transport selector still prefers P2P/IPC, then SHM. Empirically
(`NCCL_DEBUG=INFO` `via …` connection lines):

| Env on a single node | Transport actually used |
|----------------------|-------------------------|
| `RCCL_ENABLE_INTRANET=1` only | `via P2P/IPC` — net **not** used |
| `+ NCCL_P2P_DISABLE=1` | `via SHM` — net **not** used |
| `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1` | `via NET/IB/*/GDRDMA` (incl. `/Shared`) |

So set **`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`** (RCCL_ENABLE_INTRANET is not
even required once both are disabled). **Always verify** the path was actually
taken rather than assuming — a test that silently falls back to P2P/IPC makes any
net-only env var a no-op (a false pass):

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT <binary> 2>&1 \
  | grep -oE 'via (P2P/IPC|SHM|NET[^ ]*)' | sort | uniq -c
# want: only 'via NET/...' lines (incl. the '/Shared' P2P-net connections that
# carry shared!=0 send/recv, which is where NCCL_ALLOC_P2P_NET_LL_BUFFERS applies)
```

## Why srun (not mpirun) inside an allocation

mpich `mpirun` launches remote ranks over ssh, which stalls from a non-TTY
agent terminal. `srun --mpi=pmi2` goes through slurmd/PMI and needs no ssh or
controlling TTY, so it works from any shell. The `auto` launcher uses `srun`
whenever `SLURM_JOB_ID` is present; force it with `LAUNCHER=srun` if needed.
