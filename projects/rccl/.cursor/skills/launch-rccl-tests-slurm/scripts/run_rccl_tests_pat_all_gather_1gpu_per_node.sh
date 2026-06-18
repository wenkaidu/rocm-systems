#!/usr/bin/env bash
# Run all_gather_perf with exactly one MPI rank per node and -g 1 (one GPU per
# node). Intended for PAT / multi-node sanity where each node contributes a
# single device to the collective.
#
# Usage:
#   ./run_rccl_tests_pat_all_gather_1gpu_per_node.sh
#   ./run_rccl_tests_pat_all_gather_1gpu_per_node.sh ~/pat_all_gather.log
#
# Env: HOSTS, RCCL_BUILD_DIR, RCCL_TESTS_BIN_DIR, MPIRUN_BIN, LAUNCHER,
#      PAT_ALLGATHER_ARGS (space-separated perf flags; default 2 MiB scan for PAT),
#      SLURM_*. Forces GPUS_PER_NODE=1 when resolving HOSTS/NP.
#      Defaults NCCL_DEBUG=INFO, NCCL_DEBUG_SUBSYS=INIT,TUNING, NCCL_PAT_ENABLE=1 (set NCCL_PAT_ENABLE=0 to turn off).
#
# Always one MPI rank per *allocated* node: HOSTS is built with :1 per host and
# NP is recomputed from HOSTS (a preset NP in the environment is ignored so we
# do not pack multiple ranks per node). srun uses --nodes=NNODES so the step
# spans the full allocation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_rccl_tests_slurm.sh
source "${SCRIPT_DIR}/common_rccl_tests_slurm.sh"

if [[ -n "${1:-}" ]]; then
  LOG_PATH="$1"
elif [[ -n "${LOG_PATH:-}" ]]; then
  :
else
  next=1
  shopt -s nullglob
  for f in "${HOME}"/rccl_tests_pat_all_gather_[0-9]*.log; do
    base="${f##*/rccl_tests_pat_all_gather_}"
    num="${base%.log}"
    if [[ "${num}" =~ ^[0-9]+$ ]] && (( num + 1 > next )); then
      next=$(( num + 1 ))
    fi
  done
  shopt -u nullglob
  LOG_PATH="${HOME}/rccl_tests_pat_all_gather_${next}.log"
fi

export GPUS_PER_NODE=1
# Do not inherit NP from the shell (e.g. total GPUs in the job); PAT needs
# NP == number of hosts after :1 resolution so PPN stays 1 on every node.
unset NP 2>/dev/null || true
rccl_tests_resolve_hosts_np || exit 2
rccl_tests_pick_launcher
: "${SRUN_BIN:=srun}"
: "${SRUN_MPI:=pmi2}"
rccl_tests_srun_layout

: "${RCCL_TESTS_BIN_DIR:=${HOME}/rocm-systems/projects/rccl-tests/build}"
: "${RCCL_BUILD_DIR:=${HOME}/rocm-systems/projects/rccl/build/release}"
: "${MPIRUN_BIN:=${HOME}/mpich/install/bin/mpirun}"
MPI_LIB_DIR="${MPI_LIB_DIR:-${HOME}/mpich/install/lib}"
MPI_BIN_DIR="${MPI_BIN_DIR:-${HOME}/mpich/install/bin}"

: "${NCCL_DEBUG:=INFO}"
# Include TUNING so logs show "AllGather: ... -> Algo PAT ..." (override if needed).
: "${NCCL_DEBUG_SUBSYS:=INIT,TUNING}"

ENV_KV=(
  "PATH=${MPI_BIN_DIR}:${PATH}"
  "LD_LIBRARY_PATH=$(rccl_tests_rccl_ld_prefix):${MPI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  "NCCL_DEBUG=${NCCL_DEBUG}"
  "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
  "NCCL_PAT_ENABLE=${NCCL_PAT_ENABLE:-1}"
)
[[ -n "${NCCL_IB_HCA:-}" ]] && ENV_KV+=( "NCCL_IB_HCA=${NCCL_IB_HCA}" )
[[ -n "${NCCL_IB_TC:-}" ]] && ENV_KV+=( "NCCL_IB_TC=${NCCL_IB_TC}" )
[[ -n "${NCCL_IGNORE_CPU_AFFINITY:-}" ]] && ENV_KV+=( "NCCL_IGNORE_CPU_AFFINITY=${NCCL_IGNORE_CPU_AFFINITY}" )
[[ -n "${HSA_NO_SCRATCH_RECLAIM:-}" ]] && ENV_KV+=( "HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM}" )

MPIRUN_ENV=()
_mpirun_env_from_kv() {
  local _kv
  for _kv in "${ENV_KV[@]}"; do MPIRUN_ENV+=( -env "${_kv%%=*}" "${_kv#*=}" ); done
}
_mpirun_env_from_kv

bin="${RCCL_TESTS_BIN_DIR}/all_gather_perf"
[[ -x "${bin}" ]] || { echo "[pat-all_gather] missing binary: ${bin}" >&2; exit 2; }

read -r -a PAT_ALLGATHER_ARGS <<< "${PAT_ALLGATHER_ARGS:--b 2M -e 2M -f 2 -g 1 -d bfloat16 -w 2 -n 5 -c 1}"

echo "[pat-all_gather] LOG_PATH=${LOG_PATH}"
echo "[pat-all_gather] HOSTS=${HOSTS} NP=${NP} PPN=${PPN} NNODES=${NNODES} LAUNCHER=${LAUNCHER}"

if [[ "${LAUNCHER}" == "srun" ]]; then
  cmd=(
    env "${ENV_KV[@]}"
    "${SRUN_BIN}"
      ${SLURM_JOB_ID:+--jobid=${SLURM_JOB_ID}} --overlap
      --nodes="${NNODES}"
      --ntasks="${NP}" --ntasks-per-node="${PPN}"
      --mpi="${SRUN_MPI}" --cpu-bind=none
      "${bin}"
        "${PAT_ALLGATHER_ARGS[@]}"
  )
else
  cmd=(
    "${MPIRUN_BIN}" -np "${NP}"
    --hosts "${HOSTS}"
    --bind-to numa
    "${MPIRUN_ENV[@]}"
    "${bin}"
      "${PAT_ALLGATHER_ARGS[@]}"
  )
fi

{
  printf '# cmd:'; printf ' %q' "${cmd[@]}"; printf ' </dev/null\n\n'
  "${cmd[@]}" </dev/null
} > "${LOG_PATH}" 2>&1
