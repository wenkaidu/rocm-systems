#!/usr/bin/env bash
# One-shot rccl-tests sanity (default: all_reduce_perf) across the allocation.
# Uses srun inside SLURM (recommended for agent/non-TTY shells) or mpirun with --hosts.
#
# Usage:
#   ./run_rccl_tests_quick.sh
#   ./run_rccl_tests_quick.sh ~/rccl_quick_3.log
#   QUICK_BIN=all_gather_perf ./run_rccl_tests_quick.sh
#
# Env: HOSTS, NP, RCCL_BUILD_DIR, RCCL_TESTS_BIN_DIR, MPIRUN_BIN, LAUNCHER, SLURM_*

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
  for f in "${HOME}"/rccl_tests_quick_[0-9]*.log; do
    base="${f##*/rccl_tests_quick_}"
    num="${base%.log}"
    if [[ "${num}" =~ ^[0-9]+$ ]] && (( num + 1 > next )); then
      next=$(( num + 1 ))
    fi
  done
  shopt -u nullglob
  LOG_PATH="${HOME}/rccl_tests_quick_${next}.log"
fi

rccl_tests_resolve_hosts_np || exit 2
rccl_tests_pick_launcher
: "${SRUN_BIN:=srun}"
: "${SRUN_MPI:=pmi2}"
rccl_tests_srun_layout

: "${RCCL_TESTS_BIN_DIR:=${HOME}/rocm-systems/projects/rccl-tests/build}"
: "${RCCL_BUILD_DIR:=${HOME}/rocm-systems/projects/rccl/build/release}"
: "${MPIRUN_BIN:=${HOME}/mpich/install/bin/mpirun}"
: "${QUICK_BIN:=all_reduce_perf}"
MPI_LIB_DIR="${MPI_LIB_DIR:-${HOME}/mpich/install/lib}"
MPI_BIN_DIR="${MPI_BIN_DIR:-${HOME}/mpich/install/bin}"

: "${NCCL_DEBUG:=VERSION}"
: "${NCCL_DEBUG_SUBSYS:=INIT}"

ENV_KV=(
  "PATH=${MPI_BIN_DIR}:${PATH}"
  "LD_LIBRARY_PATH=$(rccl_tests_rccl_ld_prefix):${MPI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  "NCCL_DEBUG=${NCCL_DEBUG}"
  "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
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

bin="${RCCL_TESTS_BIN_DIR}/${QUICK_BIN}"
[[ -x "${bin}" ]] || { echo "[quick] missing binary: ${bin}" >&2; exit 2; }

echo "[quick] LOG_PATH=${LOG_PATH}"
echo "[quick] HOSTS=${HOSTS} NP=${NP} LAUNCHER=${LAUNCHER}"

if [[ "${LAUNCHER}" == "srun" ]]; then
  cmd=(
    env "${ENV_KV[@]}"
    "${SRUN_BIN}"
      ${SLURM_JOB_ID:+--jobid=${SLURM_JOB_ID}} --overlap
      --ntasks="${NP}" --ntasks-per-node="${PPN}"
      --mpi="${SRUN_MPI}" --cpu-bind=none
      "${bin}"
        -b 64M -e 64M -f 2 -g 1 -d bfloat16 -w 2 -n 5 -c 1
  )
else
  cmd=(
    "${MPIRUN_BIN}" -np "${NP}"
    --hosts "${HOSTS}"
    --bind-to numa
    "${MPIRUN_ENV[@]}"
    "${bin}"
      -b 64M -e 64M -f 2 -g 1 -d bfloat16 -w 2 -n 5 -c 1
  )
fi

# Write command line and all launcher output to LOG_PATH. (Avoid `| tee`:
# rccl-tests / MPI often fully-buffer stdout when stdout is a pipe, so logs
# can appear empty aside from the # cmd line.)
{
  printf '# cmd:'; printf ' %q' "${cmd[@]}"; printf ' </dev/null\n\n'
  "${cmd[@]}" </dev/null
} > "${LOG_PATH}" 2>&1
