#!/bin/bash
# Run job for the chained "build -> run rccl-UnitTests" pipeline (see
# submit_unittests_build.sbatch). Runs the FULL rccl-UnitTests suite (override
# with GTEST_FILTER) as a single task that forks its own MP child processes and
# uses all GPUs on the node. Writes a persistent result file so the verdict is
# readable after your interactive allocation is gone.
#SBATCH --job-name=ut_run
#SBATCH --partition=meta64
#SBATCH --account=vip
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.log
set -uo pipefail

RCCL="${RCCL:-$HOME/rocm-systems/projects/rccl}"
BUILD="${RCCL_BUILD_DIR:-$RCCL/build/release}"
OMPI="${OMPI_LIB:-/opt/sre-tools/ompi/lib}"          # libmpi.so.40 lives here
UT="$BUILD/test/rccl-UnitTests"
RESULT="${RESULT:-$HOME/logs/ut_full_result.txt}"

export LD_LIBRARY_PATH="$BUILD:$OMPI:${LD_LIBRARY_PATH:-}"
export NCCL_IGNORE_CPU_AFFINITY=1
export HSA_NO_SCRATCH_RECLAIM=1
export RCCL_MSCCL_ENABLE=0
export RSMI_MUTEX_THREAD_ONLY=1

echo "=== $SLURM_JOB_NAME starting $(date) on $(hostname) ===" | tee "$RESULT"
git -C "$RCCL" rev-parse HEAD 2>/dev/null | tee -a "$RESULT" || true

echo "--- running rccl-UnitTests (${GTEST_FILTER:-full suite}) ---" | tee -a "$RESULT"
SECONDS=0
stdbuf -oL -eL "$UT" ${GTEST_FILTER:+--gtest_filter="$GTEST_FILTER"} 2>&1 | tee -a "$RESULT"
rc=${PIPESTATUS[0]}
echo "=== $SLURM_JOB_NAME FINISHED rc=$rc after ${SECONDS}s $(date) ===" | tee -a "$RESULT"
grep -E '\[  (PASSED|FAILED)  \]|FAILED TEST|tests? ran' "$RESULT" | tail -40 | tee -a "$RESULT"
exit $rc
