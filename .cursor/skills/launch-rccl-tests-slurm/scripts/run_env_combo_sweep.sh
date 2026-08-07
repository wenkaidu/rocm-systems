#!/bin/bash
# Generic env-var combination latency sweep for an rccl-tests collective.
#
# Runs one full message-size sweep of $COLL for every ON/OFF (0/1) combination
# of the env vars named in $SWEEP_VARS, across every node count in $NODE_COUNTS,
# repeated $CYCLES times. Cycle is the OUTER loop so the repeats of each combo
# are spread out in time (more robust for the per-size median taken by
# parse_env_combo_latency.py).
#
# Designed for the ruby cluster (cv350-* / bnxt_re RoCE, 8 GPUs/node). Launch
# inside an allocation whose node count is >= max($NODE_COUNTS), e.g.:
#   salloc -p meta64 -N 16 --ntasks-per-node=8 -t 3:59:00 \
#     bash .../scripts/run_env_combo_sweep.sh
# (meta64 interactive allocs are capped at 240 min; use sbatch for longer.)
#
# The srun launcher needs no ssh/TTY, so this works from a Cursor/agent shell.
set -u

# ---- config (all overridable from the environment) ----
COLL=${COLL:-alltoall_perf}                 # any rccl-tests *_perf binary
NODE_COUNTS=${NODE_COUNTS:-"1 2 4 8 16"}
CYCLES=${CYCLES:-5}
SWEEP_VARS=${SWEEP_VARS:-"NCCL_ALLOC_P2P_NET_LL_BUFFERS RCCL_GFX9_CHEAP_FENCE_OFF"}
FLAGS=${FLAGS:-"-b 8 -e 1G -f 2 -g 1"}      # rccl-tests size sweep flags
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
RCCL_LIB_DIR=${RCCL_LIB_DIR:-$HOME/rccl_libs/sel}   # dir containing librccl.so.1
OMPI_LIB_DIR=${OMPI_LIB_DIR:-/opt/sre-tools/ompi/lib}
RCCL_TESTS_BIN_DIR=${RCCL_TESTS_BIN_DIR:-$HOME/rccl-tests/build}
OUT=${OUT:-$HOME/logs/env_combo_sweep}
BIN="$RCCL_TESTS_BIN_DIR/$COLL"
mkdir -p "$OUT"

# ---- ruby bnxt_re RoCE fabric + rccl-tests baseline env ----
# (swept vars are set per-combo below and override anything exported here)
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_IB_HCA=${NCCL_IB_HCA:-bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-fenic0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_IB_TC=${NCCL_IB_TC:-104}
export HSA_NO_SCRATCH_RECLAIM=1
export RCCL_MSCCL_ENABLE=0
export RCCL_IB_QPS_PER_P2P=1
export NCCL_DEBUG=VERSION
export RSMI_MUTEX_THREAD_ONLY=1
export NCCL_IB_QPS_PER_CONNECTION=4
export LD_LIBRARY_PATH=$RCCL_LIB_DIR:$OMPI_LIB_DIR:${LD_LIBRARY_PATH:-}

read -r -a VARS <<< "$SWEEP_VARS"
NVARS=${#VARS[@]}
NCOMBO=$(( 1 << NVARS ))

# record what was run so the parser can label columns without guessing
{
  echo "COLL=$COLL"
  echo "NODE_COUNTS=$NODE_COUNTS"
  echo "CYCLES=$CYCLES"
  echo "SWEEP_VARS=$SWEEP_VARS"
  echo "FLAGS=$FLAGS"
  echo "LIB=$(readlink -f "$RCCL_LIB_DIR/librccl.so.1" 2>/dev/null)"
} > "$OUT/sweep_meta.env"

echo "=== alloc: ${SLURM_JOB_NODELIST:-?} (${SLURM_NNODES:-?} nodes) ==="
echo "=== coll: $COLL | lib: $(readlink -f "$RCCL_LIB_DIR/librccl.so.1") ==="
echo "=== sweep vars: $SWEEP_VARS ($NCOMBO combos) | flags: $FLAGS ==="

for CYCLE in $(seq 1 "$CYCLES"); do
  echo "########## CYCLE $CYCLE/$CYCLES ($(date +%H:%M:%S)) ##########"
  for N in $NODE_COUNTS; do
    for (( c = 0; c < NCOMBO; c++ )); do
      # build the per-combo env assignments + filename tag (VAR=VAL segments,
      # joined by '__' so the parser can split unambiguously even though var
      # names contain underscores).
      assign=(); tag="att__N${N}"
      for (( j = 0; j < NVARS; j++ )); do
        v=$(( (c >> j) & 1 ))
        assign+=( "${VARS[j]}=$v" )
        tag+="__${VARS[j]}=$v"
      done
      log="$OUT/${tag}__c${CYCLE}.log"
      echo "=== RUN N$N ${assign[*]} c$CYCLE ($(date +%H:%M:%S)) ==="
      env "${assign[@]}" \
        srun --mpi=pmix --export=ALL -N "$N" --ntasks-per-node="$GPUS_PER_NODE" \
        "$BIN" $FLAGS > "$log" 2>&1
    done
  done
done

echo "=== DONE $(date +%H:%M:%S) -> $OUT ==="
echo "Parse with: python3 $(dirname "$0")/parse_env_combo_latency.py $OUT"
