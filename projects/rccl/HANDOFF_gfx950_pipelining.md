# Handoff: gfx950 ReduceScatter pipelining — 4-node test on ruby

Branch: `wenkai/rccl/gfx950-allgather-pipelining` (based directly on the NCCL
2.30.7 sync `59442207fd`, which uses **256 threads/block** on gfx950). The
temporary 512-thread revert `96f6d9f` has been **dropped** from this branch, so
this is the production 256-thread config. Pushed to remote `wenkai`
(github.com/wenkaidu/rocm-systems).

This continues a chat that couldn't run on ruby (the dev box `smci355` has no
SLURM client and can't resolve `ruby-slurmlogin01`). Everything needed to build,
run, and interpret the 4-node ReduceScatter experiment is below.

---

## 1. Goal

Test whether **software pipelining** helps a **256-thread** block on
**multi-node ReduceScatter (bf16)**, by overlapping next-hunk loads with the
current-hunk reduce+store (double-buffering) instead of relying on a 2nd wave
per SIMD to hide IB/load latency. The A/B is **pipelining ON vs OFF**, both at
256 threads/block (same build, runtime toggle).

## 2. Background findings (from the prior investigation)

- The two commits `96f6d9f` (512 threads/block on gfx950) vs `59442207fd`
  (256) differ only in `NCCL_MAX_NTHREADS` (`device.h`) and
  `RCCL_GFX950_MAX_NTHREADS` (`rccl_common.h`). This branch stays on **256**
  (the 2.30.7 default) so we measure pipelining benefit at the shipping config.
- **Single-node** collectives are hard-capped to 256 threads anyway
  (`rccl_wrap.cc`, `rcclOptThreadBlockSize`, `nNodes==1`). The thread-count macro
  only bites **multi-node** (`nNodes>1` uses `RCCL_GFX950_MAX_NTHREADS`), so the
  interesting regime is multi-node.
- Chunk size / #chunks / #steps are set on the host (`calcCollChunking`) from
  buffer size, protocol, channels, message size, ring `nranks` / tree depth —
  **independent of thread count**. Thread count only changes intra-chunk
  parallelism (MLP), warp-role splits, and occupancy/LDS.
- Latency hiding has three levers: TLP (waves), ILP (`COLL_UNROLL`), and
  **software pipelining** (`reduceCopyPacksPipelined`, double-buffered
  `acc1`/`acc2`). Pipelining is the most direct substitute for the 2nd wave in
  latency-bound regimes (small/mid sizes, multi-node) — but it was **disabled on
  gfx950**. This branch enables it. ReduceScatter (and Reduce/AllReduce) are the
  natural fit because they run the double-buffered reduce+store path.

## 3. Code changes on this branch (also in `gfx950-pipelining.patch`)

- `src/device/generate.py`
  - `calc_unroll_and_pipeline_for_local_arch()` gfx950 branch: return
    `(["1","2"], all_pipelines)` instead of forcing pipeline `["0"]`. This emits
    the pipeline=1 kernels for the collectives already marked `all_pipelines`
    (ReduceScatter, Reduce, AllReduce). `pipelines_of_coll["ReduceScatter"]` is
    already `all_pipelines` upstream — no change needed there.
- `src/rccl_wrap.cc` `rcclSetPipelining`
  - Drop the `gfx950` early-return; gate on `isGfx942 || isGfx950`.
  - `case ncclFuncReduceScatter` / `ncclFuncReduce` already set
    `info->pipeline = 1` — now reachable on gfx950.

Notes:
- Pipelined kernels are generated **only for bf16** (`pipelined_types`) and
  **SIMPLE** proto. So **test with `-d bfloat16`**.
- generate.py changed → **full device rebuild required** (regenerates + recompiles
  gfx950 kernels; ~tens of minutes cold).

## 4. Build on ruby

```bash
cd ~/rocm-systems/projects/rccl        # ensure this branch is checked out
git fetch wenkai && git checkout wenkai/rccl/gfx950-allgather-pipelining
# clean device build so generate.py changes take effect
find build/release -path "*rccl_device*" -name "*.o" -delete 2>/dev/null
./install.sh -l   # or the skill's tools/build_csum_debug.sh; produces build/release/librccl.so
```
The skill expects the lib at `~/rccl/build/release`; if you use that layout,
build there (or symlink). rccl-tests `*_perf` at
`~/rocm-systems/projects/rccl-tests/build`.

Confirm pipelined ReduceScatter kernels exist:
```bash
ls build/release/**/gensrc/**/*reduce_scatter*pipeline* 2>/dev/null | head
# or grep the generated function table for a ReduceScatter entry with pipeline idx 1
```

## 5. Run: 4-node ReduceScatter on ruby (bnxt_re fabric)

Get an allocation (4 nodes × 8 GPUs = 32 ranks):
```bash
salloc -p meta64 -N 4 --ntasks-per-node=8 --gpus-per-node=8 -t 60:00
# reuse SLURM_JOB_ID if already allocated
```

Ruby fabric env (bnxt_re RoCE — export before srun so --export=ALL propagates):
```bash
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_IB_HCA=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7
export NCCL_SOCKET_IFNAME=fenic0,enp49s0f0np0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=104
export NCCL_GRAPH_REGISTER=0
export NCCL_NET_SHARED_BUFFERS=0
export LD_LIBRARY_PATH=$HOME/rccl/build/release:$LD_LIBRARY_PATH   # or projects/rccl/build/release
```

**A/B on the SAME build** (kernels identical; only the host `info->pipeline`
flag differs) via the runtime toggle — this is the clean comparison:

```bash
RS=~/rocm-systems/projects/rccl-tests/build/reduce_scatter_perf
FLAGS="-d bfloat16 -b 8K -e 1G -f 2 -g 1 -n 100 -w 50"

# pipelining ON (this branch's default on gfx950)
srun --mpi=pmi2 --export=ALL -N 4 --ntasks-per-node=8 $RS $FLAGS | tee rs_pipe_on.log

# pipelining OFF (baseline) — runtime disable, no rebuild
RCCL_DISABLE_REDUCE_COPY_PIPELINING=1 \
  srun --mpi=pmi2 --export=ALL -N 4 --ntasks-per-node=8 $RS $FLAGS | tee rs_pipe_off.log
```

Compare busbw per size between the two logs (focus on small/mid sizes where
latency hiding matters; large sizes are bandwidth-bound and should match).

Alternative: if you merge the skill branch's `tools/`, you can use
`ONLY=reduce_scatter_perf DTYPE=bfloat16 tools/run_csum_stress.sh` with the
bnxt_re overrides from the skill.

## 6. Verify pipelining is actually engaged

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL,INIT \
  srun --mpi=pmi2 --export=ALL -N 4 --ntasks-per-node=8 $RS -d bfloat16 -b 1M -e 1M -g 1 -n 2 -w 1 2>&1 | grep -iE "pipeline|ReduceScatter|unsupported"
```
If you see `unsupported architecture ... Pipeline=1`, the pipeline=1 kernel
wasn't generated — re-check the generate.py change and do a clean device rebuild.

## 7. Gotchas (learned on the dev box)

- Use the matched rccl-tests at `~/rocm-systems/projects/rccl-tests/build`; a
  stale rccl-tests built against an older RCCL crashes (illegal memory access) on
  multi-GPU.
- `HSA_NO_SCRATCH_RECLAIM=1` is required on gfx950.
- Inside an allocation use `srun --mpi=pmi2`, not `mpirun` (ssh stalls from a
  non-TTY agent terminal).
- Pipeline benefit is expected in the **latency-bound** regime; at large,
  bandwidth-saturated sizes ON≈OFF is the expected (correct) result.
