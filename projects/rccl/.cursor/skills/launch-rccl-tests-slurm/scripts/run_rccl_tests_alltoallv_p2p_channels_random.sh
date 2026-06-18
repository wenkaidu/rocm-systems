#!/usr/bin/env bash
# Randomized stress: alltoallv_perf with random world size (1 .. allocation GPUs)
# and random NCCL_MAX_P2P_NCHANNELS (power of two in [1, 64]). Runs a fixed number
# of sampled combinations (default 100).
#
# Requires a SLURM allocation (or explicit HOSTS + use mpirun branch in LAUNCHER=mpirun).
#
# Usage:
#   ./run_rccl_tests_alltoallv_p2p_channels_random.sh
#   ./run_rccl_tests_alltoallv_p2p_channels_random.sh ~/my_a2av_rand
#   COMBOS=200 SEED=12345 ./run_rccl_tests_alltoallv_p2p_channels_random.sh
#
# Env:
#   COMBOS=100           number of random (np, channels) trials
#   SEED                 if set, seeds bash RANDOM for reproducible draws
#   ALLOC_NP             max ranks (default: from SLURM/HOSTS via common helper)
#   GPUS_PER_NODE        caps per-node rank packing for srun (default: SLURM hints or 8)
#   MIN_BYTES / MAX_BYTES / STEP_FACTOR / WARMUP / ITERS / DTYPE / GPUS_PER_THREAD
#   RCCL_TESTS_BIN_DIR, RCCL_BUILD_DIR, MPI_BIN_DIR, MPI_LIB_DIR
#   RCCL_P2P_BATCH_ENABLE if set (e.g. 1), forwarded to every trial via env
#   NCCL_* / RCCL_*      other tuning: set before launch; IB HCA/TC are merged when set

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_rccl_tests_slurm.sh
source "${SCRIPT_DIR}/common_rccl_tests_slurm.sh"

# Powers of two in [1, 64] for NCCL_MAX_P2P_NCHANNELS
POW2_P2P_CH=(1 2 4 8 16 32 64)

if [[ -n "${1:-}" ]]; then
  LOG_DIR="$1"
elif [[ -n "${LOG_DIR:-}" ]]; then
  :
else
  next=1
  shopt -s nullglob
  for d in "${HOME}"/rccl_tests_a2av_p2p_rand_[0-9]*; do
    [[ -d "${d}" ]] || continue
    base="${d##*/rccl_tests_a2av_p2p_rand_}"
    if [[ "${base}" =~ ^[0-9]+$ ]] && (( base + 1 > next )); then
      next=$(( base + 1 ))
    fi
  done
  shopt -u nullglob
  LOG_DIR="${HOME}/rccl_tests_a2av_p2p_rand_${next}"
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

: "${ALLOC_NP:=${NP}}"
(( ALLOC_NP >= 1 )) || { echo "[a2av-rand] invalid ALLOC_NP=${ALLOC_NP}" >&2; exit 2; }

: "${GPUS_PER_NODE:=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-8}}}}"
: "${COMBOS:=100}"
: "${DTYPE:=bfloat16}"
: "${MIN_BYTES:=8}"
: "${MAX_BYTES:=256M}"
: "${STEP_FACTOR:=2}"
: "${WARMUP:=3}"
: "${ITERS:=10}"
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

if [[ -n "${SEED:-}" ]]; then
  RANDOM="${SEED}"
fi

bin="${RCCL_TESTS_BIN_DIR}/alltoallv_perf"
[[ -x "${bin}" ]] || { echo "[a2av-rand] missing ${bin}" >&2; exit 2; }

SUMMARY="${LOG_DIR}/SUMMARY.txt"
{
  echo "# run_rccl_tests_alltoallv_p2p_channels_random @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# log_dir       : ${LOG_DIR}"
  echo "# combos        : ${COMBOS}"
  echo "# alloc_np max  : ${ALLOC_NP}"
  echo "# gpus_per_node : ${GPUS_PER_NODE}"
  echo "# launcher      : ${LAUNCHER}"
  echo "# hosts         : ${HOSTS:-<n/a>}"
  echo "# sweep         : -d ${DTYPE} -b ${MIN_BYTES} -e ${MAX_BYTES} -f ${STEP_FACTOR} -w ${WARMUP} -n ${ITERS} -g ${GPUS_PER_THREAD} -c 1"
  echo "# p2p_ch choices: ${POW2_P2P_CH[*]}"
  echo "# RCCL_P2P_BATCH_ENABLE: ${RCCL_P2P_BATCH_ENABLE:-<unset>}"
  echo "#"
  printf "# %4s %4s %4s %6s %8s %8s %4s %s\n" "iter" "np" "p2p" "exit" "#wrong" "warn" "ok?" "log"
} | tee "${SUMMARY}"

overall_fail=0

pick_random_ch() {
  # uniform from POW2_P2P_CH
  local idx=$(( RANDOM % ${#POW2_P2P_CH[@]} ))
  printf '%s' "${POW2_P2P_CH[$idx]}"
}

pick_random_np() {
  # uniform 1 .. ALLOC_NP
  echo $(( 1 + RANDOM % ALLOC_NP ))
}

build_env_kv() {
  # Use a non-generic local name: a global `max_ch` (e.g. from the environment)
  # would make `local max_ch=$1` error on the second call under bash.
  local _p2p_max=$1
  ENV_KV=(
    "PATH=${MPI_BIN_DIR}:${PATH}"
    "LD_LIBRARY_PATH=$(rccl_tests_rccl_ld_prefix):${MPI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    "NCCL_DEBUG=${NCCL_DEBUG}"
    "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
    "NCCL_IGNORE_CPU_AFFINITY=${NCCL_IGNORE_CPU_AFFINITY}"
    "HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM}"
    "NCCL_MIN_P2P_NCHANNELS=1"
    "NCCL_MAX_P2P_NCHANNELS=${_p2p_max}"
  )
  [[ -n "${NCCL_IB_HCA:-}" ]] && ENV_KV+=( "NCCL_IB_HCA=${NCCL_IB_HCA}" )
  [[ -n "${NCCL_IB_TC:-}" ]] && ENV_KV+=( "NCCL_IB_TC=${NCCL_IB_TC}" )
  [[ -n "${RCCL_P2P_BATCH_ENABLE:-}" ]] && ENV_KV+=( "RCCL_P2P_BATCH_ENABLE=${RCCL_P2P_BATCH_ENABLE}" )
  # Last line must succeed: a lone `[[ ... ]] && ...` that skips the RHS leaves
  # status 1 and breaks callers running with `set -e`.
  return 0
}

MPIRUN_ENV=()
build_mpirun_env() {
  MPIRUN_ENV=()
  local _kv
  for _kv in "${ENV_KV[@]}"; do MPIRUN_ENV+=( -env "${_kv%%=*}" "${_kv#*=}" ); done
}

for ((iter = 1; iter <= COMBOS; iter++)); do
  trial_np=$(pick_random_np)
  trial_ch=$(pick_random_ch)
  iterlog="${LOG_DIR}/iter_$(printf '%03d' "${iter}")_np${trial_np}_p2pmax${trial_ch}.log"

  build_env_kv "${trial_ch}"
  build_mpirun_env

  set +e
  {
    printf '# iter=%d np=%d NCCL_MAX_P2P_NCHANNELS=%s\n' "${iter}" "${trial_np}" "${trial_ch}"
    printf '# cmd:'
    if [[ "${LAUNCHER}" == "srun" ]]; then
      printf ' %q' env "${ENV_KV[@]}" "${SRUN_BIN}" \
        ${SLURM_JOB_ID:+--jobid=${SLURM_JOB_ID}} --overlap \
        --ntasks="${trial_np}" --ntasks-per-node="${GPUS_PER_NODE}" \
        --mpi="${SRUN_MPI}" --cpu-bind=none \
        "${bin}" \
        -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${STEP_FACTOR}" \
        -g "${GPUS_PER_THREAD}" -d "${DTYPE}" \
        -w "${WARMUP}" -n "${ITERS}" -c 1
    else
      printf ' %q' "${MPIRUN_BIN}" -np "${trial_np}" --hosts "${HOSTS}" --bind-to numa \
        "${MPIRUN_ENV[@]}" "${bin}" \
        -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${STEP_FACTOR}" \
        -g "${GPUS_PER_THREAD}" -d "${DTYPE}" \
        -w "${WARMUP}" -n "${ITERS}" -c 1
    fi
    printf '\n\n'
    if [[ "${LAUNCHER}" == "srun" ]]; then
      env "${ENV_KV[@]}" "${SRUN_BIN}" \
        ${SLURM_JOB_ID:+--jobid=${SLURM_JOB_ID}} --overlap \
        --ntasks="${trial_np}" --ntasks-per-node="${GPUS_PER_NODE}" \
        --mpi="${SRUN_MPI}" --cpu-bind=none \
        "${bin}" \
          -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${STEP_FACTOR}" \
          -g "${GPUS_PER_THREAD}" -d "${DTYPE}" \
          -w "${WARMUP}" -n "${ITERS}" -c 1
    else
      "${MPIRUN_BIN}" -np "${trial_np}" --hosts "${HOSTS}" --bind-to numa \
        "${MPIRUN_ENV[@]}" "${bin}" \
          -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${STEP_FACTOR}" \
          -g "${GPUS_PER_THREAD}" -d "${DTYPE}" \
          -w "${WARMUP}" -n "${ITERS}" -c 1
    fi
  } > "${iterlog}" 2>&1 </dev/null
  rc=$?
  set -e

  wrong=0
  warn=0
  read -r wrong warn <<< "$(rccl_tests_parse_log_summary "${iterlog}")" || true
  : "${wrong:=0}" "${warn:=0}"
  st="OK"
  if (( rc != 0 || wrong != 0 )); then
    st="FAIL"
    overall_fail=1
  fi
  printf "  %4d %4d %4d %6d %8d %8d %4s %s\n" \
    "${iter}" "${trial_np}" "${trial_ch}" "${rc}" "${wrong}" "${warn}" "${st}" "$(basename "${iterlog}")" \
    | tee -a "${SUMMARY}" || true
done

{
  echo "# done -> ${LOG_DIR}"
  echo "# overall: $([[ ${overall_fail} -eq 0 ]] && echo PASS || echo FAIL)"
} | tee -a "${SUMMARY}"

exit "${overall_fail}"
