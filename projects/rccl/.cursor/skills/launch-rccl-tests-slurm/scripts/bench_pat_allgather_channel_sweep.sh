#!/usr/bin/env bash
# Sweep all_gather_perf in-place busbw vs message size (up to 128MiB) for PAT,
# comparing NCCL_MAX_NCHANNELS caps (default 2, 4, 8).
#
# Forces NCCL_ALGO=PAT so results are PAT kernel/proxy path (not ring fallback).
#
# Usage:
#   SLURM_JOB_ID=... LAUNCHER=srun GPUS_PER_NODE=1 ./bench_pat_allgather_channel_sweep.sh
#   NP=4 HOSTS=localhost LAUNCHER=mpirun ./bench_pat_allgather_channel_sweep.sh
#
# Env:
#   PAT_SWEEP_ARGS — passed to all_gather_perf (default: -b 64K -e 128M -f 2 ... bfloat16)
#   BENCH_CAPS — default "2 4 8"
#   NCCL_ALGO — default PAT (override to compare without forcing)
#   RCCL_BUILD_DIR, RCCL_TESTS_BIN_DIR, MPIRUN_BIN, SLURM_JOB_ID, etc. (see bench_all_gather_max_nchannels.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_rccl_tests_slurm.sh
source "${SCRIPT_DIR}/common_rccl_tests_slurm.sh"

: "${RCCL_BUILD_DIR:=${HOME}/rocm-systems/projects/rccl/build/release}"
: "${RCCL_TESTS_BIN_DIR:=${HOME}/rocm-systems/projects/rccl-tests/build}"
: "${MPIRUN_BIN:=${HOME}/mpich/install/bin/mpirun}"
MPI_LIB_DIR="${MPI_LIB_DIR:-${HOME}/mpich/install/lib}}"
MPI_BIN_DIR="${MPI_BIN_DIR:-${HOME}/mpich/install/bin}}"
: "${SRUN_BIN:=srun}"
: "${SRUN_MPI:=pmi2}"

: "${BENCH_CAPS:=2 4 8}"
: "${NCCL_PAT_ENABLE:=1}"
: "${NCCL_ALGO:=PAT}"

read -r -a PERF_ARGS <<< "${PAT_SWEEP_ARGS:--b 64K -e 128M -f 2 -g 1 -d bfloat16 -w 2 -n 5 -c 0}"

bin="${RCCL_TESTS_BIN_DIR}/all_gather_perf"
[[ -x "${bin}" ]] || { echo "[pat_sweep] missing ${bin}" >&2; exit 2; }

LD_PREFIX="$(rccl_tests_rccl_ld_prefix)"

rccl_tests_pick_launcher
USE_SRUN=0
if [[ "${LAUNCHER}" == "srun" ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]] || { echo "[pat_sweep] LAUNCHER=srun needs SLURM_JOB_ID" >&2; exit 2; }
  : "${GPUS_PER_NODE:=1}"
  unset NP 2>/dev/null || true
  rccl_tests_resolve_hosts_np || exit 2
  rccl_tests_srun_layout
  USE_SRUN=1
elif [[ -z "${HOSTS:-}" ]] && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  rccl_tests_resolve_hosts_np || exit 2
  : "${NP:=}"
  [[ -n "${NP:-}" ]] || { echo "[pat_sweep] NP unset after resolve" >&2; exit 2; }
elif [[ -z "${HOSTS:-}" ]]; then
  : "${NP:=4}"
  : "${HOSTS:=localhost}"
fi

: "${NP:?}"
: "${HOSTS:?}"

base_env_kv() {
  local cap="$1"
  echo "PATH=${MPI_BIN_DIR}:${PATH}"
  echo "LD_LIBRARY_PATH=${LD_PREFIX}:${MPI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  echo "HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}"
  echo "NCCL_DEBUG=WARN"
  echo "NCCL_PAT_ENABLE=${NCCL_PAT_ENABLE}"
  echo "NCCL_ALGO=${NCCL_ALGO}"
  [[ -n "${NCCL_PROTO:-}" ]] && echo "NCCL_PROTO=${NCCL_PROTO}"
  [[ -n "${NCCL_IB_HCA:-}" ]] && echo "NCCL_IB_HCA=${NCCL_IB_HCA}"
  [[ -n "${NCCL_IB_TC:-}" ]] && echo "NCCL_IB_TC=${NCCL_IB_TC}"
  echo "NCCL_MAX_NCHANNELS=${cap}"
}

run_sweep_to_tsv() {
  local cap="$1"
  local out="$2"
  local raw="${tmpdir}/raw_${cap}.log"

  if [[ "${USE_SRUN}" == 1 ]]; then
    local -a env_args=()
    local line
    while IFS= read -r line; do
      [[ -z "${line}" ]] && continue
      env_args+=( "$line" )
    done < <(base_env_kv "${cap}")

    env "${env_args[@]}" "${SRUN_BIN}" ${SLURM_JOB_ID:+--jobid="${SLURM_JOB_ID}"} --overlap \
      --nodes="${NNODES}" --ntasks="${NP}" --ntasks-per-node="${PPN}" \
      --mpi="${SRUN_MPI}" --cpu-bind=none \
      "${bin}" "${PERF_ARGS[@]}" >"${raw}" 2>&1 || { echo "[pat_sweep] cap=${cap} failed:" >&2; tail -40 "${raw}"; return 1; }
  else
    local -a mpie=(
      "${MPIRUN_BIN}" -np "${NP}" --hosts "${HOSTS}"
      -env PATH "${MPI_BIN_DIR}:${PATH}"
      -env LD_LIBRARY_PATH "${LD_PREFIX}:${MPI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
      -env HSA_NO_SCRATCH_RECLAIM "${HSA_NO_SCRATCH_RECLAIM:-1}"
      -env NCCL_DEBUG WARN
      -env NCCL_PAT_ENABLE "${NCCL_PAT_ENABLE}"
      -env NCCL_ALGO "${NCCL_ALGO}"
    )
    [[ -n "${NCCL_PROTO:-}" ]] && mpie+=( -env NCCL_PROTO "${NCCL_PROTO}" )
    mpie+=( -env NCCL_MAX_NCHANNELS "${cap}" "${bin}" "${PERF_ARGS[@]}" )
    "${mpie[@]}" >"${raw}" 2>&1 || { echo "[pat_sweep] cap=${cap} failed:" >&2; tail -40 "${raw}"; return 1; }
  fi

  # Per-size in-place busbw (column before trailing N/A on data rows).
  awk '/^[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+/ && $3 ~ /^bfloat16|^float|^half|^int|^double|^nccl/ {
    print $1 "\t" $(NF-1)
  }' "${raw}" >"${out}"
}

echo "# pat_channel_sweep LAUNCHER=${LAUNCHER} SLURM_JOB_ID=${SLURM_JOB_ID:-} NP=${NP} NNODES=${NNODES:-na} NCCL_ALGO=${NCCL_ALGO} PERF=${PERF_ARGS[*]}"

declare -A CAP_FILE=()
tmpdir=$(mktemp -d)
trap 'rm -rf "${tmpdir}"' EXIT

for cap in ${BENCH_CAPS}; do
  f="${tmpdir}/cap_${cap}.tsv"
  echo "[pat_sweep] running NCCL_MAX_NCHANNELS=${cap} ..." >&2
  run_sweep_to_tsv "${cap}" "${f}"
  CAP_FILE[${cap}]="${f}"
done

# Join on size (assume identical size list for each cap)
read -r first_cap _ <<< "${BENCH_CAPS}"
[[ -n "${first_cap}" ]] || exit 1

echo -ne "size_B"
for cap in ${BENCH_CAPS}; do
  echo -ne "\tch${cap}_in_place_busbw_GB_s"
done
echo

while read -r sz _; do
  [[ -z "${sz}" ]] && continue
  echo -ne "${sz}"
  for cap in ${BENCH_CAPS}; do
    ip=$(awk -v s="${sz}" -F'\t' '$1==s {print $2; exit}' "${CAP_FILE[${cap}]}")
    [[ -z "${ip}" ]] && ip="NA"
    echo -ne "\t${ip}"
  done
  echo
done < <(cut -f1 "${CAP_FILE[${first_cap}]}")

echo "#"
echo -n "# run_avg_busbw_GB_s (single reported average for full sweep)"
for cap in ${BENCH_CAPS}; do
  av=$(awk '/^# Avg bus bandwidth/ {v=$NF} END{print v+0}' "${tmpdir}/raw_${cap}.log")
  echo -ne "\t${av}"
done
echo
