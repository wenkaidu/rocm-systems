#!/usr/bin/env bash
# Multi-node rccl-tests sweep: every *_perf binary (or ONLY= subset), size sweep,
# validation (-c 1). Intended for stress under a SLURM allocation using srun
# (no ssh/TTY required).
#
# Usage:
#   ./run_rccl_tests_stress_multi_node.sh
#   ./run_rccl_tests_stress_multi_node.sh ~/my_logs
#   ONLY="all_reduce_perf alltoall_perf" ./run_rccl_tests_stress_multi_node.sh
#   SKIP="hypercube_perf" ./run_rccl_tests_stress_multi_node.sh
#
# Env: HOSTS, NP, DTYPE, MIN_BYTES, MAX_BYTES, STEP_FACTOR, WARMUP, ITERS,
#   GPUS_PER_THREAD, RCCL_TESTS_BIN_DIR, RCCL_BUILD_DIR, MPIRUN_BIN, LAUNCHER,
#   SLURM_*, NCCL_*, RCCL_* (export before launch; included in ENV_KV for mpirun)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_rccl_tests_slurm.sh
source "${SCRIPT_DIR}/common_rccl_tests_slurm.sh"

if [[ -n "${1:-}" ]]; then
  LOG_DIR="$1"
elif [[ -n "${LOG_DIR:-}" ]]; then
  :
else
  next=1
  shopt -s nullglob
  for d in "${HOME}"/rccl_tests_stress_[0-9]*; do
    [[ -d "${d}" ]] || continue
    base="${d##*/rccl_tests_stress_}"
    if [[ "${base}" =~ ^[0-9]+$ ]] && (( base + 1 > next )); then
      next=$(( base + 1 ))
    fi
  done
  shopt -u nullglob
  LOG_DIR="${HOME}/rccl_tests_stress_${next}"
fi
if [[ -e "${LOG_DIR}" ]]; then
  n=1
  while [[ -e "${LOG_DIR}_${n}" ]]; do n=$(( n + 1 )); done
  LOG_DIR="${LOG_DIR}_${n}"
fi
mkdir -p "${LOG_DIR}"

rccl_tests_resolve_hosts_np || exit 2
rccl_tests_pick_launcher
: "${SRUN_BIN:=srun}"
: "${SRUN_MPI:=pmi2}"
rccl_tests_srun_layout

: "${DTYPE:=bfloat16}"
: "${MIN_BYTES:=8}"
: "${MAX_BYTES:=1G}"
: "${STEP_FACTOR:=2}"
: "${WARMUP:=5}"
: "${ITERS:=20}"
: "${GPUS_PER_THREAD:=1}"
: "${RCCL_TESTS_BIN_DIR:=${HOME}/rocm-systems/projects/rccl-tests/build}"
: "${RCCL_BUILD_DIR:=${HOME}/rocm-systems/projects/rccl/build/release}"
: "${MPIRUN_BIN:=${HOME}/mpich/install/bin/mpirun}"
MPI_LIB_DIR="${MPI_LIB_DIR:-${HOME}/mpich/install/lib}"
MPI_BIN_DIR="${MPI_BIN_DIR:-${HOME}/mpich/install/bin}"

: "${NCCL_DEBUG:=VERSION}"
: "${NCCL_DEBUG_SUBSYS:=INIT}"
: "${NCCL_IGNORE_CPU_AFFINITY:=1}"
: "${HSA_NO_SCRATCH_RECLAIM:=1}"

ENV_KV=(
  "PATH=${MPI_BIN_DIR}:${PATH}"
  "LD_LIBRARY_PATH=$(rccl_tests_rccl_ld_prefix):${MPI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  "NCCL_DEBUG=${NCCL_DEBUG}"
  "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
  "NCCL_IGNORE_CPU_AFFINITY=${NCCL_IGNORE_CPU_AFFINITY}"
  "HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM}"
)
[[ -n "${NCCL_IB_HCA:-}" ]] && ENV_KV+=( "NCCL_IB_HCA=${NCCL_IB_HCA}" )
[[ -n "${NCCL_IB_TC:-}" ]] && ENV_KV+=( "NCCL_IB_TC=${NCCL_IB_TC}" )
[[ -n "${NCCL_NET_PLUGIN:-}" ]] && ENV_KV+=( "NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN}" )

MPIRUN_ENV=()
for _kv in "${ENV_KV[@]}"; do MPIRUN_ENV+=( -env "${_kv%%=*}" "${_kv#*=}" ); done

if [[ -n "${ONLY:-}" ]]; then
  read -r -a COLLECTIVES <<< "${ONLY}"
else
  COLLECTIVES=()
  shopt -s nullglob
  for f in "${RCCL_TESTS_BIN_DIR}"/*_perf; do
    [[ -x "${f}" ]] || continue
    COLLECTIVES+=( "$(basename "${f}")" )
  done
  shopt -u nullglob
fi
if [[ -n "${SKIP:-}" ]]; then
  read -r -a SKIP_ARR <<< "${SKIP}"
  KEEP=()
  for c in "${COLLECTIVES[@]}"; do
    drop=0
    for s in "${SKIP_ARR[@]}"; do [[ "${c}" == "${s}" ]] && drop=1; done
    (( drop == 0 )) && KEEP+=( "${c}" )
  done
  COLLECTIVES=( "${KEEP[@]}" )
fi

if (( ${#COLLECTIVES[@]} == 0 )); then
  echo "[stress-multi] no collectives (RCCL_TESTS_BIN_DIR=${RCCL_TESTS_BIN_DIR})" >&2
  exit 2
fi

SUMMARY="${LOG_DIR}/SUMMARY.txt"
{
  echo "# run_rccl_tests_stress_multi_node @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# log_dir         : ${LOG_DIR}"
  if [[ "${LAUNCHER}" == "srun" ]]; then
    echo "# launcher        : srun (--mpi=${SRUN_MPI}) jobid=${SLURM_JOB_ID:-<none>} nodes=${NNODES} ppn=${PPN}"
  else
    echo "# launcher        : mpirun hosts=${HOSTS}"
  fi
  echo "# hosts           : ${HOSTS}  np=${NP}"
  echo "# rccl_build_dir  : ${RCCL_BUILD_DIR}"
  echo "# rccl_tests_bin  : ${RCCL_TESTS_BIN_DIR}"
  echo "# sweep           : -d ${DTYPE} -b ${MIN_BYTES} -e ${MAX_BYTES} -f ${STEP_FACTOR} -w ${WARMUP} -n ${ITERS} -g ${GPUS_PER_THREAD}"
  echo "# collectives     : ${COLLECTIVES[*]}"
  echo "#"
  printf "# %-22s %6s %8s %8s %4s\n" "collective" "exit" "#wrong" "warn" "ok?"
} | tee "${SUMMARY}"

overall_fail=0
for col in "${COLLECTIVES[@]}"; do
  bin="${RCCL_TESTS_BIN_DIR}/${col}"
  log="${LOG_DIR}/${col}.log"
  if [[ ! -x "${bin}" ]]; then
    printf "  %-22s %6s %8s %8s %4s\n" "${col}" "SKIP" "-" "-" "miss" | tee -a "${SUMMARY}"
    continue
  fi
  echo "[stress-multi] === ${col} ===" >&2
  if [[ "${LAUNCHER}" == "srun" ]]; then
    cmd=(
      env "${ENV_KV[@]}"
      "${SRUN_BIN}"
        ${SLURM_JOB_ID:+--jobid=${SLURM_JOB_ID}} --overlap
        --ntasks="${NP}" --ntasks-per-node="${PPN}"
        --mpi="${SRUN_MPI}" --cpu-bind=none
        "${bin}"
          -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${STEP_FACTOR}"
          -g "${GPUS_PER_THREAD}" -d "${DTYPE}"
          -w "${WARMUP}" -n "${ITERS}"
          -c 1
    )
  else
    cmd=(
      "${MPIRUN_BIN}" -np "${NP}"
      --hosts "${HOSTS}"
      --bind-to numa
      "${MPIRUN_ENV[@]}"
      "${bin}"
        -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${STEP_FACTOR}"
        -g "${GPUS_PER_THREAD}" -d "${DTYPE}"
        -w "${WARMUP}" -n "${ITERS}"
        -c 1
    )
  fi
  { printf '# cmd:'; printf ' %q' "${cmd[@]}"; printf ' </dev/null\n'; } > "${log}"
  set +e
  "${cmd[@]}" >> "${log}" 2>&1 </dev/null
  rc=$?
  set -e
  read -r wrong warn <<< "$(rccl_tests_parse_log_summary "${log}")"
  status="OK"
  if (( rc != 0 || wrong != 0 )); then
    status="FAIL"
    overall_fail=1
  fi
  printf "  %-22s %6d %8d %8d %4s\n" "${col}" "${rc}" "${wrong}" "${warn}" "${status}" | tee -a "${SUMMARY}"
done

echo "# done -> ${LOG_DIR}" | tee -a "${SUMMARY}"
echo "# overall: $([[ ${overall_fail} -eq 0 ]] && echo PASS || echo FAIL)" | tee -a "${SUMMARY}"

exit "${overall_fail}"
