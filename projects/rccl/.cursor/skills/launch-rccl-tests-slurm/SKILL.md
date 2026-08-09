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

> **For long / unattended / multi-`N` sweeps, always use `sbatch`, never a
> `nohup … salloc … &` driver.** A `salloc` (and any `nohup`-backgrounded loop
> that owns it) belongs to the login-session process tree, so a session cleanup
> — logout, agent/terminal teardown, idle reaper, or stray `scancel -n bash` —
> kills the whole driver + `salloc` + `srun` tree and leaves the SLURM
> allocations **orphaned** (holding nodes, doing nothing) with the sweep dead
> partway through. `sbatch` jobs are owned by `slurmctld` and survive session
> cleanup. **Submit one job per node count** (they run independently, often in
> parallel): `for N in 1 2 4 8 16; do sbatch -N "$N" --ntasks-per-node=8 -t 3:00:00 -o ~/logs/sw_N$N.out --wrap "bash <inner>.sh"; done`.
> If `squeue` shows running jobs with **no** matching `salloc`/`srun`/driver in
> `ps -u "$USER"`, they are orphans — `scancel` and resubmit via `sbatch`. Use
> `salloc` only for short interactive runs you actively watch.

## Prerequisites

- **SLURM allocation** for multi-node runs (`salloc` / interactive `srun --pty`),
  **or** use [scripts/submit_rccl_tests_slurm.sh](scripts/submit_rccl_tests_slurm.sh)
  to **queue via `sbatch`** (no shell on compute nodes required — and the
  recommended path for long/unattended sweeps, see callout above). The runners
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
| [scripts/run_ll128_coherence_soak.sh](scripts/run_ll128_coherence_soak.sh) | **Coherence soak**: force `NCCL_PROTO=LL128` and hammer a collective (default `all_reduce`) at **small sizes**, **high rank counts**, **many iterations**, over **exact-integer datatypes** (`int8..uint64`, `sum` → bit-exact so any `#wrong>0` is a real coherence bug), sweeping **both `-R 0` non-registered and `-R 1` registered** user buffers (the two LL128 kernel paths PR #9347 splits). Auto-creates a `librccl.so.1` link dir for `RCCL_LIB`; pins OOB bootstrap to `OOB_IFACE` (default `eth0`); **marker-kills** each run at its completion line so post-result teardown hangs don't stall/false-fail. Content-based pass/fail (`total_wrong`), not exit code. Env: `RCCL_LIB`, `DTYPES`, `REGS`, `OP`, `MIN_BYTES`/`MAX_BYTES`, `ITERS`, `CHECK` (`-c`), `RUN_CYCLES` (`-N`), `CYCLES`, `SRUN_TIMEOUT`. |
| [scripts/submit_ll128_coherence_soak.sbatch](scripts/submit_ll128_coherence_soak.sbatch) | **`sbatch`** wrapper for the coherence soak; sets site RoCE tuning + `OUT_DIR`. Submit per scale: `sbatch -N 16 --exclude=<bad nodes> --export=ALL,RCCL_LIB=... submit_ll128_coherence_soak.sbatch`. |
| [scripts/run_rccl_tests_alltoallv_p2p_channels_random.sh](scripts/run_rccl_tests_alltoallv_p2p_channels_random.sh) | **`alltoallv_perf`**: `COMBOS` trials (default 100) with random `NP` in `1..allocation` and random `NCCL_MAX_P2P_NCHANNELS` ∈ {1,2,4,8,16,32,64}; `NCCL_MIN_P2P_NCHANNELS=1`. |
| [scripts/run_rccl_tests_pat_all_gather_1gpu_per_node.sh](scripts/run_rccl_tests_pat_all_gather_1gpu_per_node.sh) | **`all_gather_perf`**: **every allocated node**, one MPI rank per node (**`GPUS_PER_NODE=1`**, ignores preset **`NP`**), **`srun --nodes=$NNODES`**, **`-g 1`**; defaults **`NCCL_DEBUG=INFO`**, **`NCCL_DEBUG_SUBSYS=INIT,TUNING`** (algorithm lines), **`NCCL_PAT_ENABLE=1`**, **2 MiB** sweep (`-b 2M -e 2M`); override via **`PAT_ALLGATHER_ARGS`** / **`NCCL_PAT_ENABLE`**. |
| [scripts/run_rccl_tests_multi_lib.sh](scripts/run_rccl_tests_multi_lib.sh) | **Compare multiple RCCL libs in ONE allocation**, interleaved across cycles (`outer=cycle, mid=collective, inner=lib`) to share thermal / run-order bias. Captures per-size `algo/proto/#channels` via **`-A 1`**; writes `results.csv`. Env: **`RCCL_LIBS`** (paths or commit suffixes; first = baseline), **`COLLECTIVES`** (or `all`), **`CYCLES`**, **`RCCL_DIRECT_ALLGATHER_DISABLE`**, **`RCCL_DIRECT_REDUCE_SCATTER_DISABLE`**, **`EXTRA_ENV`** (`VAR=VAL ...` A/B knobs, e.g. `RCCL_GFX9_CHEAP_FENCE_OFF=0`), **`SRUN_EXTRA`**, sweep knobs. |
| [scripts/report_multi_lib.py](scripts/report_multi_lib.py) | Reduce a multi-lib `results.csv` to per-size **median oop** tables + a per-collective / per-protocol **geomean delta% vs a reference lib** (isolates LL / LL128 / SIMPLE effects). Args: `[results.csv] [--ref COMMIT] [--labels c=name,...]`. |
| [scripts/report_multi_lib_ab.py](scripts/report_multi_lib_ab.py) | **A/B two multi-lib sweeps** (two `results.csv` from differing `EXTRA_ENV`) to isolate one knob per lib. `--mode delta` prints `(B-A)/A%` per lib (neg => B faster); `--mode combined` prints each lib's delta% vs a `--ref` for both runs side by side. Args: `A B [--libs ..] [--mode delta\|combined] [--ref COMMIT]` (A/B may be a dir or its `results.csv`). |
| [scripts/run_static_vs_dyn.sh](scripts/run_static_vs_dyn.sh) | **Compare a statically-linked rccl-tests (rccl baked into the binary from a `librccl*.a`) against the SAME source linked dynamically** across several `librccl.so` builds, interleaved per cycle. Env: **`STATIC_BIN_DIR`** (empty to skip static), **`STATIC_TAG`**, **`STATIC_ALGO_PROXY`** (same-era `.so` for the static binary's `-M` algo query), **`DYN_BIN_DIR`**, **`DYN_LIBS`** (abs paths or `RCCL_LIB_DIR` suffixes), **`COLLECTIVES`**, **`CYCLES`**, **`DTYPE`**, sweep knobs. Writes `results.csv` with per-size `algo/proto/#channels`. |
| [scripts/report_static_vs_dyn.py](scripts/report_static_vs_dyn.py) | Summarize a `run_static_vs_dyn.sh` `results.csv`: per-collective median-oop tables with **delta% vs baseline** (default `static`), a geomean line, and an algo/proto:#channels table to confirm all configs took the same path. Args: `[results.csv] [baseline_tag]`. |
| [scripts/submit_static_vs_dyn.sbatch](scripts/submit_static_vs_dyn.sbatch) | **`sbatch`** wrapper for `run_static_vs_dyn.sh`; `OUT_DIR` gets a `_<N>n` suffix. Submit per node count: `sbatch -N <n> --export=ALL,DYN_LIBS=..,OUT_DIR_BASE=.. submit_static_vs_dyn.sbatch`. |
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

**Or queue a batch job** (no allocation in your shell). Run from the **rccl repo
root**; the submit script path is:

`./.cursor/skills/launch-rccl-tests-slurm/scripts/submit_rccl_tests_slurm.sh`

**bash / sh** (inline env vars):

```bash
# Quick all_reduce on 2 nodes (default)
./.cursor/skills/launch-rccl-tests-slurm/scripts/submit_rccl_tests_slurm.sh

# PAT all_gather, 16 nodes (1 rank per node, -g 1)
env RUN_SCRIPT=run_rccl_tests_pat_all_gather_1gpu_per_node.sh SLURM_NNODES=16 \
  ./.cursor/skills/launch-rccl-tests-slurm/scripts/submit_rccl_tests_slurm.sh

# PAT all_gather, 8 nodes, wait for completion
env RUN_SCRIPT=run_rccl_tests_pat_all_gather_1gpu_per_node.sh SLURM_NNODES=8 \
  PAT_ALLGATHER_ARGS="-b 2M -e 128M -f 2 -g 1 -d bfloat16 -w 2 -n 5 -c 0" \
  NCCL_PAT_ENABLE=1 NCCL_ALGO=PAT WAIT=1 \
  ./.cursor/skills/launch-rccl-tests-slurm/scripts/submit_rccl_tests_slurm.sh

# Multi-node stress sweep on 4 nodes
env RUN_SCRIPT=run_rccl_tests_stress_multi_node.sh SLURM_NNODES=4 \
  ./.cursor/skills/launch-rccl-tests-slurm/scripts/submit_rccl_tests_slurm.sh
```

**tcsh / csh** (default login shell on some clusters — do **not** use
`VAR=value command`; that syntax is bash-only):

```tcsh
setenv RUN_SCRIPT run_rccl_tests_pat_all_gather_1gpu_per_node.sh
setenv SLURM_NNODES 16
./.cursor/skills/launch-rccl-tests-slurm/scripts/submit_rccl_tests_slurm.sh
```

Or from tcsh, call through bash:

```tcsh
bash -c 'env RUN_SCRIPT=run_rccl_tests_pat_all_gather_1gpu_per_node.sh SLURM_NNODES=16 \
  ./.cursor/skills/launch-rccl-tests-slurm/scripts/submit_rccl_tests_slurm.sh'
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

## Comparing multiple RCCL libraries (interleaved, per-protocol)

To A/B/… several `librccl.so` builds fairly, run them **all inside one
allocation, interleaved**, so every lib sees the same thermal state and network
conditions within each cycle. This removes the biggest source of noise when
comparing builds: separate allocations (or back-to-back blocks per lib) attribute
drift to whichever lib happened to run during it.

Stage the libs (any naming works, but the runner resolves bare commit/tag
suffixes against `RCCL_LIB_DIR`):

```bash
ls ~/rccl_libs/librccl.so.1.0.*      # e.g. ...0ff9140, ...08f97e1e, ...
```

Then, **inside a multi-node allocation** (16 nodes shown), interleave the libs
over 5 cycles for all collectives, forcing all_gather onto the standard
RING/LL/LL128/SIMPLE path so the protocol comparison is apples-to-apples:

```bash
RCCL_LIBS="0ff9140 08f97e1e 2e0cbb95 acbde506 c79e0ee8" \
COLLECTIVES=all CYCLES=5 RCCL_DIRECT_ALLGATHER_DISABLE=1 \
OUT_DIR=~/rccl_mn_perf/final_do4 \
  ./.cursor/skills/launch-rccl-tests-slurm/scripts/run_rccl_tests_multi_lib.sh
```

Or queue it via the sbatch wrapper (no interactive shell needed). Env is
forwarded into the batch step; size the walltime for `libs × collectives ×
cycles` srun runs:

```bash
env RUN_SCRIPT=run_rccl_tests_multi_lib.sh SLURM_NNODES=16 SLURM_TIME=09:00:00 \
  RCCL_LIBS="0ff9140 08f97e1e 2e0cbb95 acbde506 c79e0ee8" \
  COLLECTIVES=all CYCLES=5 RCCL_DIRECT_ALLGATHER_DISABLE=1 \
  OUT_DIR="$HOME/rccl_mn_perf/final_do4" \
  ./.cursor/skills/launch-rccl-tests-slurm/scripts/submit_rccl_tests_slurm.sh
```

When `${OUT_DIR}/DONE` appears, report per-protocol results (first lib in
`RCCL_LIBS` is the baseline unless you pass `--ref`):

```bash
./.cursor/skills/launch-rccl-tests-slurm/scripts/report_multi_lib.py \
  ~/rccl_mn_perf/final_do4/results.csv --ref 0ff9140 \
  --labels "0ff9140=baseline,08f97e1e=2.29.7+split,c79e0ee8=2.30.4 HEAD"
```

The report prints, per collective, a per-size table of **median out-of-place
latency** with the dominant `algo/proto`, then a per-collective / per-protocol
**geomean delta% vs the reference**. Read it by protocol band: **LL** (small
sizes), **LL128** (mid–large), **SIMPLE** (largest). This is how the
`ll128-reg-split` LL128 win and the 2.30.4-base SIMPLE regression were isolated
without confounding thermal drift.

Notes:

- `-A 1` (default `CAPTURE_ALGO=1`) needs the tested lib discoverable as plain
  `librccl.so`; the runner auto-creates a per-lib `.link_<commit>/` symlink dir
  and prepends it to `LD_LIBRARY_PATH`, so the `-A` `dlopen("librccl.so")`
  resolves to the lib under test (not `LD_PRELOAD`).
- p2p-style collectives (`alltoall*`, `sendrecv`, `gather`, `scatter`,
  `hypercube`) report `N/A` for algo/proto but latency is still captured.
- Each `srun` is wrapped in `timeout ${SRUN_TIMEOUT}` (default 400s) so one
  hanging collective cannot stall the whole interleaved sweep.

## Static vs dynamic linkage (baked-in `librccl.a` vs `librccl.so`)

To check whether **static linking** changes performance, or to use a
statically-linked `dev` archive as a stable baseline for several `librccl.so`
builds, compare a rccl-tests binary with rccl **baked in** against the **same
source** linked dynamically. Keeping one source tree + one header set means any
gap is pure kernel/linkage, not harness differences.

### Build the two binary trees

Teach `rccl-tests/src/Makefile` an `RCCL_STATIC_LIB` knob so an archive can be
linked in place of `-lrccl`:

```make
ifneq ($(RCCL_STATIC_LIB),)
# --start-group/--end-group (not --whole-archive): the dev archive bundles aux
# objects with their own main() (client.cc), which --whole-archive would force in.
HIPLDFLAGS += -Wl,--start-group $(RCCL_STATIC_LIB) -Wl,--end-group
# rccl-tests ships its own copies of a few rccl helpers (e.g. IsArchMatch);
# let the test's definition win, as it shadows the .so in the dynamic build.
HIPLDFLAGS += -Wl,--allow-multiple-definition
# A static rccl needs these named explicitly (the .so normally pulls them in):
HIPLDFLAGS += -lrocm_smi64 -lnuma -ldrm -ldrm_amdgpu -L/opt/amdgpu/lib/x86_64-linux-gnu
else
LIBRARIES += rccl dl
endif
HIPLDFLAGS += $(LIBRARIES:%=-l%)
# Also add -Wl,--export-dynamic so the static binary exposes its baked-in rccl*
# symbols to the -M dlopen(NULL) algo query.
```

Build the **static** tree (rccl from the archive) and a **dynamic** counterpart
from the identical source + headers. Use develop headers that match the archive
ABI (`NCCL_HOME=.../build_rel/include`), **not** a stale `/opt/rocm` header:

```bash
cd ~/rccl-tests-static
ALLCOLL="all_reduce all_gather reduce_scatter"   # BIN_FILES_LIST subset
# static: rccl baked in
make -j"$(nproc)" MPI=1 MPI_HOME=$HOME/mpich/install \
  NCCL_HOME=$HOME/rccl_build_wt/<devwt>/build_rel/include \
  RCCL_STATIC_LIB=$HOME/rccl_libs/librccl-dev.a \
  GPU_TARGETS=gfx942 BIN_FILES_LIST="$ALLCOLL"
# dynamic counterpart: same source, -lrccl chosen at runtime
make -j"$(nproc)" BUILDDIR=build_dyn MPI=1 MPI_HOME=$HOME/mpich/install \
  NCCL_HOME=$HOME/rccl_build_wt/<devwt>/build_rel/include \
  CUSTOM_RCCL_LIB=$HOME/rccl_build_wt/<devwt>/build_rel \
  GPU_TARGETS=gfx942 BIN_FILES_LIST="$ALLCOLL"
# confirm the static binary has no librccl.so dependency:
ldd build/all_reduce_perf | grep -i rccl || echo "statically linked"
```

### Run and report

Interleave the static config plus any number of dynamic libs in one allocation.
`STATIC_ALGO_PROXY` is a same-era develop `.so` used **only** for the static
binary's `-M` algo/proto query (its collective still runs the baked-in rccl):

```bash
DYN_LIBS="0ff9140 pr8451f2af4e8 92be5e5new" \
STATIC_TAG=static_dev0717 \
STATIC_ALGO_PROXY=$HOME/rccl_build_wt/2976ade3/build_rel/librccl.so.1.0 \
COLLECTIVES="all_reduce_perf all_gather_perf reduce_scatter_perf" \
CYCLES=5 DTYPE=bfloat16 OUT_DIR=~/rccl_mn_perf/static_vs_dyn_1n \
  ./.cursor/skills/launch-rccl-tests-slurm/scripts/run_static_vs_dyn.sh

./.cursor/skills/launch-rccl-tests-slurm/scripts/report_static_vs_dyn.py \
  ~/rccl_mn_perf/static_vs_dyn_1n/results.csv static_dev0717
```

Queue per node count via the sbatch wrapper (report each as it lands, since
larger allocations take longer to schedule):

```bash
for n in 1 2 4; do
  sbatch -N $n --export=ALL,DYN_LIBS="0ff9140 pr8451f2af4e8 92be5e5new",\
STATIC_TAG=static_dev0717,\
STATIC_ALGO_PROXY=$HOME/rccl_build_wt/2976ade3/build_rel/librccl.so.1.0,\
COLLECTIVES="all_reduce_perf all_gather_perf reduce_scatter_perf",\
OUT_DIR_BASE=$HOME/rccl_mn_perf/static_vs_dyn,CYCLES=5 \
    ./.cursor/skills/launch-rccl-tests-slurm/scripts/submit_static_vs_dyn.sbatch
done
```

Read the report by protocol band (the algo/proto table confirms every config
picked the same path, so deltas are kernel/linkage only): typically there is
**no systematic static-vs-dynamic penalty** — gaps track the underlying lib
differences (e.g. an LL128 mid-range win, or a small-message LL regression).
Single-cycle blow-ups at small all_gather sizes are fabric noise; add cycles to
average them out.

## LL128 coherence soak (forced protocol, exact-integer validation)

To stress a **specific protocol's correctness at scale** — e.g. validating the
`ll128-reg-split` work (PR #9347) that emits separate **registered vs
non-registered** user-buffer LL128 kernels — force the protocol and check
results with datatypes whose reduction is **bit-exact**, so any mismatch is a
genuine coherence bug rather than float rounding.

```bash
# Inside a healthy N-node allocation (SLURM_JOB_ID set), or via the sbatch wrapper.
RCCL_LIB=$HOME/rccl_libs/librccl.so.1.0.pr9347 \
DTYPES="int8 uint8 int32 uint32 int64 uint64" REGS="0 1" \
MIN_BYTES=8 MAX_BYTES=64K ITERS=30 CHECK=5 CYCLES=4 SRUN_TIMEOUT=180 \
OUT_DIR=$HOME/rccl_ll128_soak_16n \
  ./.cursor/skills/launch-rccl-tests-slurm/scripts/run_ll128_coherence_soak.sh
```

The runner forces `NCCL_PROTO=LL128`, runs `-o sum -c ${CHECK}` per size, and
reports `PASS (0 wrong)` only if **every** run validated with `total_wrong=0`.
`SUMMARY.txt` legend: `OK` clean exit · `OK*` validated 0-wrong but killed on a
teardown hang · `WRONG` coherence mismatch · `HANG` no sweep completed. Aggregate
across raw logs to double-check:

```bash
awk '/^[[:space:]]*[0-9]+[[:space:]]/{r++; if(NF>=13)w+=$9+$13} END{printf "rows=%d wrong=%d\n",r,w+0}' "$OUT_DIR"/c*.log
grep -rh 'Out of bounds values' "$OUT_DIR"/c*.log | grep -vc ': 0 OK'   # want 0
```

Confirm the protocol was actually honored (not just requested) with a short
`NCCL_DEBUG_SUBSYS=ENV` run: look for `NCCL_PROTO set by environment to LL128`.

### Multi-node bring-up gotchas (learned the hard way)

These bit us going from 4 → 16 nodes; check them **before** blaming the lib:

- **SONAME resolution.** rccl-tests links against `librccl.so.1`. A directory
  that only contains `librccl.so.1.0.<tag>` files does **not** satisfy the
  loader — it silently falls back to `/opt/rocm/lib/librccl.so.1` (you test the
  wrong lib). Always expose a `librccl.so.1` symlink dir on `LD_LIBRARY_PATH`
  (`run_ll128_coherence_soak.sh` and `run_rccl_tests_multi_lib.sh` do this
  automatically). Verify with `ldd .../all_reduce_perf | grep rccl`.
- **Out-of-band bootstrap must use the routable management NIC**, not the RoCE
  `rdma*` devices. At 16 nodes MPICH/UCX `MPI_Init` (and RCCL's own socket
  bootstrap) can fail with `UCX ... Destination is unreachable` / DC
  `ibv_create_ah ... Connection timed out` unless you pin them to `eth0`:
  `NCCL_SOCKET_IFNAME=eth0 UCX_NET_DEVICES=eth0 UCX_TLS=tcp,sm,self`
  (the runner sets these from `OOB_IFACE`). Small allocations often "work"
  on defaults and mask this.
- **Dead RoCE links = specific bad nodes.** `ibv_modify_qp failed with 110
  Connection timed out ... INIT -> RTR ... dev mlx5_X` during RCCL init means a
  node pair can't establish QPs (data path), independent of your config. It
  reproduces on the same hosts. Bisect (2 → 8 → 16 nodes) to find them, then
  `--exclude=<nodelist>` and re-request until you get a healthy island. A clean
  16-node run prints the data table with `# Out of bounds values : 0 OK`.
- **Benign teardown hangs.** Some stacks finish the test (print `Avg bus
  bandwidth`) then hang in `ncclCommDestroy`/`MPI_Finalize`. Don't treat the
  resulting non-zero exit as a failure — key on log content. The soak runner
  launches `srun` in the background and stops it at the completion marker, so a
  60 s test doesn't eat a 180 s timeout on every run.

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
