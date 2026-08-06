#!/usr/bin/env bash
# sbatch wrapper for run_ll128_coherence_soak.sh: force NCCL_PROTO=LL128 and
# hammer all_reduce at small sizes / high rank counts / many iterations over
# exact-integer datatypes (both -R 0 and -R 1), with data validation.
#
# Submit (size -N to the rank count you want; exclude known-bad nodes):
#   sbatch -N 16 --exclude=useocpm2m-097-[132,135,136,142] \
#     --export=ALL,RCCL_LIB=$HOME/rccl_libs/librccl.so.1.0.pr9347 \
#     submit_ll128_coherence_soak.sbatch
#
# Key env (forwarded to the runner; see run_ll128_coherence_soak.sh for all):
#   RCCL_LIB   librccl.so* under test (exposed via a librccl.so.1 link dir).
#   OOB_IFACE  routable management NIC for MPI/UCX + RCCL bootstrap (default eth0).
#   NCCL_IB_HCA / NCCL_IB_TC   RoCE data-path tuning for this site.
#   DTYPES REGS MIN_BYTES MAX_BYTES ITERS CHECK CYCLES SRUN_TIMEOUT OUT_DIR
#SBATCH --job-name=ll128_soak
#SBATCH --partition=amd-rccl
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
set -uo pipefail

SKILL_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
RUNNER="${SKILL_SCRIPTS}/run_ll128_coherence_soak.sh"
RCCL_LIB="${RCCL_LIB:-${HOME}/rccl_libs/librccl.so.1.0.pr9347}"
[[ -s "${RCCL_LIB}" ]] || { echo "[ll128-soak] RCCL_LIB not found: ${RCCL_LIB}" >&2; exit 3; }

# Site RoCE data-path tuning (override to match your fabric).
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
export NCCL_IB_TC="${NCCL_IB_TC:-41}"

export RCCL_LIB
export OUT_DIR="${OUT_DIR:-${HOME}/rccl_ll128_soak_${SLURM_JOB_NUM_NODES}n_${SLURM_JOB_ID}}"
echo "[ll128-soak] job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} lib=${RCCL_LIB} -> ${OUT_DIR}"
bash "${RUNNER}"
