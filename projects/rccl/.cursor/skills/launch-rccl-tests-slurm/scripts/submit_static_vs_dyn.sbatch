#!/usr/bin/env bash
#SBATCH --partition=amd-rccl
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=08:00:00
# Node count set at submit time:  sbatch -N <n> submit_static_vs_dyn.sbatch
# Everything else is driven by env (forward with --export=ALL,VAR=..,VAR=..):
#   DYN_LIBS, STATIC_BIN_DIR, STATIC_ALGO_PROXY, DYN_BIN_DIR, COLLECTIVES,
#   CYCLES, DTYPE, OUT_DIR_BASE, sweep knobs, NCCL_*/RCCL_* tuning.
# OUT_DIR gets a _<N>n suffix so per-node-count runs don't collide.
set -uo pipefail
# Runner: honor an explicit RUNNER, else the copy in this skill's scripts dir.
SKILL_SCRIPTS="${HOME}/rocm-systems/projects/rccl/.cursor/skills/launch-rccl-tests-slurm/scripts"
RUNNER="${RUNNER:-${SKILL_SCRIPTS}/run_static_vs_dyn.sh}"

export STATIC_BIN_DIR="${STATIC_BIN_DIR-${HOME}/rccl-tests-static/build}"
export DYN_BIN_DIR="${DYN_BIN_DIR:-${HOME}/rccl-tests-static/build_dyn}"
export RCCL_LIB_DIR="${RCCL_LIB_DIR:-${HOME}/rccl_libs}"
export MPI_LIB="${MPI_LIB:-${HOME}/mpich/install/lib}"
export COLLECTIVES="${COLLECTIVES:-all_reduce_perf all_gather_perf reduce_scatter_perf}"
export CYCLES="${CYCLES:-5}"
export DTYPE="${DTYPE:-bfloat16}"
export MIN_BYTES="${MIN_BYTES:-8}" MAX_BYTES="${MAX_BYTES:-1G}" STEP_FACTOR="${STEP_FACTOR:-2}"
export WARMUP="${WARMUP:-5}" ITERS="${ITERS:-15}"
export OUT_DIR="${OUT_DIR_BASE:-${HOME}/rccl_mn_perf/static_vs_dyn}_${SLURM_JOB_NUM_NODES}n"

# Site NIC/topology tuning (override for your cluster).
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
export NCCL_IB_TC="${NCCL_IB_TC:-41}"
export NCCL_IGNORE_CPU_AFFINITY=1
export HSA_NO_SCRATCH_RECLAIM=1

echo "[cmp] job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} runner=${RUNNER} out=${OUT_DIR} start=$(date -Is)"
bash "${RUNNER}"
echo "[cmp] exit=$? end=$(date -Is)"
