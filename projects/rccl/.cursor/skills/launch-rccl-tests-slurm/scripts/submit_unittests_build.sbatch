#!/bin/bash
# Build job for the chained "build -> run rccl-UnitTests" pipeline.
# Submit chained with the run job so results survive your interactive allocation:
#   JID_BUILD=$(sbatch --parsable submit_unittests_build.sbatch)
#   JID_RUN=$(sbatch --parsable --dependency=afterok:$JID_BUILD submit_unittests_run.sbatch)
# Override RCCL / account / partition via --export or by editing the header.
#SBATCH --job-name=ut_build
#SBATCH --partition=meta64
#SBATCH --account=vip
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.log
set -eo pipefail

RCCL="${RCCL:-$HOME/rocm-systems/projects/rccl}"
BUILD="${RCCL_BUILD_DIR:-$RCCL/build/release}"

echo "=== $SLURM_JOB_NAME starting $(date) on $(hostname) ==="
git -C "$RCCL" rev-parse HEAD 2>/dev/null || true

echo "--- incremental build: librccl + rccl-UnitTests ---"
cmake --build "$BUILD" --target rccl rccl-UnitTests -j "${JOBS:-32}"

test -f "$BUILD/librccl.so"            && echo "OK librccl.so"
test -x "$BUILD/test/rccl-UnitTests"   && echo "OK rccl-UnitTests"
echo "=== $SLURM_JOB_NAME DONE $(date) ==="
