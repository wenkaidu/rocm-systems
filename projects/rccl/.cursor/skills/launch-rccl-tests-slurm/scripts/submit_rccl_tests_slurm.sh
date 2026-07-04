#!/usr/bin/env bash
# Queue an rccl-tests runner via sbatch (no interactive salloc / srun --pty needed).
#
# Submits a batch job with cluster defaults matching a typical amd-rccl allocation:
#   srun --pty -N 2 -C "block3|block4" --ntasks-per-node=64 --gres=gpu:8 \
#     -p amd-rccl -t 4:00:00 --qos=urgent bash
#
# Inside the job, invokes an existing launcher from this directory (same helpers as
# the allocation-based scripts). SLURM_JOB_ID is set automatically; runners use srun.
#
# Usage:
#   ./submit_rccl_tests_slurm.sh
#   RUN_SCRIPT=run_rccl_tests_pat_all_gather_1gpu_per_node.sh SLURM_NNODES=8 ./submit_rccl_tests_slurm.sh
#   RUN_SCRIPT=run_rccl_tests_stress_multi_node.sh SLURM_NNODES=4 WAIT=1 ./submit_rccl_tests_slurm.sh
#   ./submit_rccl_tests_slurm.sh run_rccl_tests_quick.sh ~/rccl_quick_batch.log
#
# Env (Slurm request):
#   SLURM_NNODES            nodes (default: 2)
#   SLURM_PARTITION         partition (default: amd-rccl)
#   SLURM_QOS               qos (default: urgent)
#   SLURM_CONSTRAINT        feature constraint (default: block3|block4)
#   SLURM_TIME              wall time (default: 4:00:00)
#   SLURM_NTASKS_PER_NODE    ntasks-per-node on the allocation (default: 64)
#   SLURM_GPUS_PER_NODE      gpus per node (default: 8)
#   SLURM_JOB_NAME          sbatch job name (default: derived from RUN_SCRIPT)
#   SLURM_ACCOUNT           optional account
#   SBATCH_EXTRA            extra flags passed verbatim to sbatch (quoted string)
#   SBATCH_LOG_DIR          where sbatch -o/-e files go (default: ~/rccl_slurm_jobs)
#
# Env (runner):
#   RUN_SCRIPT              script under scripts/ (default: run_rccl_tests_quick.sh)
#   WAIT                    if 1, block until the job finishes (default: 0)
#   RCCL_BUILD_DIR, RCCL_TESTS_BIN_DIR, NCCL_*, RCCL_*, PAT_ALLGATHER_ARGS, etc.
#     are forwarded into the batch step when set in the submitting shell.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_rccl_tests_slurm.sh
source "${SCRIPT_DIR}/common_rccl_tests_slurm.sh"

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

RUN_LOG_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    -N|--nodes)
      SLURM_NNODES="$2"
      shift 2
      ;;
    -t|--time)
      SLURM_TIME="$2"
      shift 2
      ;;
    -p|--partition)
      SLURM_PARTITION="$2"
      shift 2
      ;;
    --qos)
      SLURM_QOS="$2"
      shift 2
      ;;
    -C|--constraint)
      SLURM_CONSTRAINT="$2"
      shift 2
      ;;
    --run|--runner)
      RUN_SCRIPT="$2"
      shift 2
      ;;
    *)
      if [[ -z "${RUN_SCRIPT:-}" && -f "${SCRIPT_DIR}/$1" && "$1" == *.sh ]]; then
        RUN_SCRIPT="$1"
      elif [[ -z "${RUN_LOG_PATH}" ]]; then
        RUN_LOG_PATH="$1"
      else
        echo "[submit] unexpected argument: $1" >&2
        usage 1
      fi
      shift
      ;;
  esac
done

: "${RUN_SCRIPT:=run_rccl_tests_quick.sh}"
: "${SLURM_NNODES:=2}"
: "${SLURM_PARTITION:=amd-rccl}"
: "${SLURM_QOS:=urgent}"
: "${SLURM_CONSTRAINT:=block3|block4}"
: "${SLURM_TIME:=4:00:00}"
: "${SLURM_NTASKS_PER_NODE:=64}"
: "${SLURM_GPUS_PER_NODE:=8}"
: "${WAIT:=0}"
: "${SBATCH_LOG_DIR:=${HOME}/rccl_slurm_jobs}"

RUNNER="${SCRIPT_DIR}/${RUN_SCRIPT}"
[[ -f "${RUNNER}" ]] || { echo "[submit] missing runner: ${RUNNER}" >&2; exit 2; }

job_tag="${RUN_SCRIPT%.sh}"
job_tag="${job_tag#run_rccl_tests_}"
: "${SLURM_JOB_NAME:=rccl-${job_tag}-${SLURM_NNODES}n}"

mkdir -p "${SBATCH_LOG_DIR}"
ts="$(date +%Y%m%d_%H%M%S)"
batch_log="${SBATCH_LOG_DIR}/${SLURM_JOB_NAME}_${ts}_%j.out"
batch_script="$(mktemp "${SBATCH_LOG_DIR}/.sbatch_${SLURM_JOB_NAME}_${ts}_XXXXXX.sh")"
trap 'rm -f "${batch_script}"' EXIT

# Forward tuning / path env vars set by the caller into the batch step.
FORWARD_VARS=(
  RCCL_BUILD_DIR RCCL_TESTS_BIN_DIR MPIRUN_BIN MPI_LIB_DIR MPI_BIN_DIR
  LAUNCHER SRUN_BIN SRUN_MPI
  NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_PAT_ENABLE NCCL_ALGO NCCL_IB_HCA NCCL_IB_TC
  NCCL_IGNORE_CPU_AFFINITY NCCL_NET_PLUGIN NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
  RCCL_MAX_PAT_NCHANNELS RCCL_DIRECT_ALLGATHER_DISABLE RCCL_HIERARCHICAL_ALLGATHER
  HSA_NO_SCRATCH_RECLAIM
  PAT_ALLGATHER_ARGS PAT_SWEEP_ARGS BENCH_CAPS LOG_DIR
  ONLY SKIP DTYPE MIN_BYTES MAX_BYTES STEP_FACTOR WARMUP ITERS GPUS_PER_THREAD
  QUICK_BIN CYCLES COMBOS GPUS_PER_NODE
)
forward_exports=()
for v in "${FORWARD_VARS[@]}"; do
  if [[ -n "${!v+x}" ]]; then
    forward_exports+=( "export ${v}=$(printf '%q' "${!v}")" )
  fi
done

if [[ -z "${RUN_LOG_PATH}" ]]; then
  RUN_LOG_PATH="${SBATCH_LOG_DIR}/${SLURM_JOB_NAME}_${ts}_%j.run.log"
fi

{
  echo '#!/usr/bin/env bash'
  echo '#SBATCH --job-name='"${SLURM_JOB_NAME}"
  echo '#SBATCH --nodes='"${SLURM_NNODES}"
  echo '#SBATCH --ntasks-per-node='"${SLURM_NTASKS_PER_NODE}"
  echo '#SBATCH --gres=gpu:'"${SLURM_GPUS_PER_NODE}"
  echo '#SBATCH --partition='"${SLURM_PARTITION}"
  echo '#SBATCH --qos='"${SLURM_QOS}"
  echo '#SBATCH --constraint='"${SLURM_CONSTRAINT}"
  echo '#SBATCH --time='"${SLURM_TIME}"
  echo '#SBATCH --output='"${batch_log}"
  echo '#SBATCH --error='"${batch_log}"
  [[ -n "${SLURM_ACCOUNT:-}" ]] && echo '#SBATCH --account='"${SLURM_ACCOUNT}"
  [[ -n "${SBATCH_EXTRA:-}" ]] && echo "#SBATCH ${SBATCH_EXTRA}"
  echo 'set -euo pipefail'
  echo 'echo "[batch] job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} host=$(hostname) start=$(date -Is)"'
  printf '%s\n' "${forward_exports[@]}"
  echo 'export LAUNCHER=srun'
  echo 'cd "$(dirname "'"${RUNNER}"'")"'
  echo 'bash "'"${RUNNER}"'" "'"${RUN_LOG_PATH//%j/\$SLURM_JOB_ID}"'"'
  echo 'rc=$?'
  echo 'echo "[batch] runner exit=${rc} end=$(date -Is)"'
  echo 'exit "${rc}"'
} > "${batch_script}"

echo "[submit] runner=${RUN_SCRIPT}"
echo "[submit] nodes=${SLURM_NNODES} partition=${SLURM_PARTITION} qos=${SLURM_QOS} time=${SLURM_TIME}"
echo "[submit] constraint=${SLURM_CONSTRAINT} gres=gpu:${SLURM_GPUS_PER_NODE} ntasks-per-node=${SLURM_NTASKS_PER_NODE}"
echo "[submit] batch script: ${batch_script}"
echo "[submit] sbatch log pattern: ${batch_log}"

submit_out="$(sbatch "${batch_script}")"
echo "${submit_out}"
job_id="${submit_out##* }"
resolved_sbatch_log="${batch_log//%j/${job_id}}"
resolved_run_log="${RUN_LOG_PATH//%j/${job_id}}"

echo "[submit] job_id=${job_id}"
echo "[submit] sbatch log: ${resolved_sbatch_log}"
echo "[submit] runner log: ${resolved_run_log}"

if [[ "${WAIT}" == 1 ]]; then
  echo "[submit] waiting for job ${job_id} ..."
  while squeue -h -j "${job_id}" 2>/dev/null | grep -q .; do
    sleep 10
  done
  echo "[submit] finished; tail sbatch log:"
  tail -30 "${resolved_sbatch_log}" 2>/dev/null || true
fi
