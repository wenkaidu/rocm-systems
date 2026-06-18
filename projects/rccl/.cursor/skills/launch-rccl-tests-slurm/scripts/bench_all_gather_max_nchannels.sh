#!/usr/bin/env bash
# Compare all_gather_perf bus bandwidth for NCCL_MAX_NCHANNELS caps (e.g. 2 vs 4 vs 8).
#
# Usage (single node / mpirun):
#   NP=4 HOSTS=localhost LAUNCHER=mpirun ./bench_all_gather_max_nchannels.sh
#
# Usage (SLURM, one rank per node — same layout as PAT script):
#   GPUS_PER_NODE=1 unset NP
#   SLURM_JOB_ID=37423 LAUNCHER=srun ./bench_all_gather_max_nchannels.sh
#
# LAUNCHER=auto uses srun when SLURM_JOB_ID is set, else mpirun.
#
# Env:
#   RCCL_BUILD_DIR, RCCL_TESTS_BIN_DIR, MPIRUN_BIN, MPI_LIB_DIR
#   NP, HOSTS — mpirun only; SLURM: resolve from nodelist when HOSTS unset
#   GPUS_PER_NODE — for SLURM hostlist :N suffix (default 1 when LAUNCHER=srun)
#   SRUN_BIN, SRUN_MPI — srun options (default pmi2, --overlap like PAT)
#   PAT_ALLGATHER_ARGS — perf flags (default: 2 MiB bfloat16)
#   BENCH_CAPS — space-separated list; include "unset" for no cap
#   NCCL_PAT_ENABLE (default 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_rccl_tests_slurm.sh
source "${SCRIPT_DIR}/common_rccl_tests_slurm.sh"

: "${RCCL_BUILD_DIR:=${HOME}/rocm-systems/projects/rccl/build/release}"
: "${RCCL_TESTS_BIN_DIR:=${HOME}/rocm-systems/projects/rccl-tests/build}"
: "${MPIRUN_BIN:=${HOME}/mpich/install/bin/mpirun}"
MPI_LIB_DIR="${MPI_LIB_DIR:-${HOME}/mpich/install/lib}}"
MPI_BIN_DIR="${MPI_BIN_DIR:-${HOME}/mpich/install/bin}"
: "${SRUN_BIN:=srun}"
: "${SRUN_MPI:=pmi2}"

: "${BENCH_CAPS:=2 4 8 unset}"
: "${NCCL_PAT_ENABLE:=1}"

read -r -a PERF_ARGS <<< "${PAT_ALLGATHER_ARGS:--b 2M -e 2M -f 2 -g 1 -d bfloat16 -w 8 -n 25 -c 0}"

bin="${RCCL_TESTS_BIN_DIR}/all_gather_perf"
[[ -x "${bin}" ]] || { echo "[bench_max_nchannels] missing ${bin}" >&2; exit 2; }

LD_PREFIX="$(rccl_tests_rccl_ld_prefix)"

rccl_tests_pick_launcher
USE_SRUN=0
if [[ "${LAUNCHER}" == "srun" ]]; then
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "[bench_max_nchannels] LAUNCHER=srun requires SLURM_JOB_ID (or run inside a Slurm allocation)." >&2
    exit 2
  fi
  : "${GPUS_PER_NODE:=1}"
  unset NP 2>/dev/null || true
  rccl_tests_resolve_hosts_np || exit 2
  rccl_tests_srun_layout
  USE_SRUN=1
elif [[ -z "${HOSTS:-}" ]] && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  rccl_tests_resolve_hosts_np || exit 2
  : "${NP:=}"
  if [[ -z "${NP:-}" ]]; then
    echo "[bench_max_nchannels] resolve_hosts_np did not set NP" >&2
    exit 2
  fi
elif [[ -z "${HOSTS:-}" ]]; then
  : "${NP:=4}"
  : "${HOSTS:=localhost}"
fi

: "${NP:?NP must be set (or use SLURM + resolve)}"
: "${HOSTS:?HOSTS must be set (or use SLURM + resolve)}"

base_env_kv() {
  local cap="$1"
  echo "PATH=${MPI_BIN_DIR}:${PATH}"
  echo "LD_LIBRARY_PATH=${LD_PREFIX}:${MPI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  echo "HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}"
  echo "NCCL_DEBUG=INFO"
  echo "NCCL_DEBUG_SUBSYS=INIT"
  echo "NCCL_PAT_ENABLE=${NCCL_PAT_ENABLE}"
  [[ -n "${NCCL_IB_HCA:-}" ]] && echo "NCCL_IB_HCA=${NCCL_IB_HCA}"
  [[ -n "${NCCL_IB_TC:-}" ]] && echo "NCCL_IB_TC=${NCCL_IB_TC}"
  [[ -n "${NCCL_IGNORE_CPU_AFFINITY:-}" ]] && echo "NCCL_IGNORE_CPU_AFFINITY=${NCCL_IGNORE_CPU_AFFINITY}"
  if [[ "${cap}" != "unset" ]]; then
    echo "NCCL_MAX_NCHANNELS=${cap}"
  fi
}

run_one() {
  local cap="$1"
  local log
  log="$(mktemp)"

  if [[ "${USE_SRUN}" == 1 ]]; then
    local -a env_args=()
    local line
    while IFS= read -r line; do
      [[ -z "${line}" ]] && continue
      env_args+=( "$line" )
    done < <(base_env_kv "${cap}")

    local -a cmd=(
      env "${env_args[@]}"
      "${SRUN_BIN}"
      ${SLURM_JOB_ID:+--jobid="${SLURM_JOB_ID}"} --overlap
      --nodes="${NNODES}"
      --ntasks="${NP}"
      --ntasks-per-node="${PPN}"
      --mpi="${SRUN_MPI}"
      --cpu-bind=none
      "${bin}"
      "${PERF_ARGS[@]}"
    )
    "${cmd[@]}" >"${log}" 2>&1 || { cat "${log}"; rm -f "${log}"; return 1; }
  else
    local -a mpie=(
      "${MPIRUN_BIN}" -np "${NP}" --hosts "${HOSTS}"
      -env PATH "${MPI_BIN_DIR}:${PATH}"
      -env LD_LIBRARY_PATH "${LD_PREFIX}:${MPI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
      -env HSA_NO_SCRATCH_RECLAIM "${HSA_NO_SCRATCH_RECLAIM:-1}"
      -env NCCL_DEBUG INFO
      -env NCCL_DEBUG_SUBSYS INIT
      -env NCCL_PAT_ENABLE "${NCCL_PAT_ENABLE}"
    )
    if [[ "${cap}" != "unset" ]]; then
      mpie+=( -env NCCL_MAX_NCHANNELS "${cap}" )
    fi
    mpie+=( "${bin}" "${PERF_ARGS[@]}" )
    "${mpie[@]}" >"${log}" 2>&1 || { cat "${log}"; rm -f "${log}"; return 1; }
  fi

  local ip avg ch
  ip="$(awk '/^[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+/ && $3 ~ /^[a-z]/ {print $(NF-1); exit}' "${log}")"
  avg="$(awk '/^# Avg bus bandwidth/ {print $NF; exit}' "${log}")"
  ch="$(awk -F'coll channels:' '/coll channels:/ {sub(/collnet.*/,""); gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print $2; exit}' "${log}")"
  rm -f "${log}"
  printf '%s\t%s\t%s\t%s\n' "${cap}" "${ip}" "${avg}" "${ch}"
}

echo "# bench_all_gather_max_nchannels LAUNCHER=${LAUNCHER} USE_SRUN=${USE_SRUN} SLURM_JOB_ID=${SLURM_JOB_ID:-} NP=${NP} HOSTS=${HOSTS:-} NNODES=${NNODES:-na} PPN=${PPN:-na} PERF=${PERF_ARGS[*]}"
echo -e "NCCL_MAX_NCHANNELS\tin_place_busbw_GB/s\tavg_busbw_GB/s\tcoll_channels(rank0_log)"
for cap in ${BENCH_CAPS}; do
  run_one "${cap}"
done
