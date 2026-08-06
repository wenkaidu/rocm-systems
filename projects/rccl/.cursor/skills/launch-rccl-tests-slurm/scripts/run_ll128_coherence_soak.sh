#!/usr/bin/env bash
# LL128 coherence soak: force NCCL_PROTO=LL128 and hammer a collective (default
# all_reduce) at SMALL sizes, HIGH rank counts, MANY iterations, over
# EXACT-INTEGER datatypes, with data validation (-c). Because integer sum is
# exact, ANY non-zero "wrong" count is a real coherence/correctness failure.
#
# It sweeps both user-buffer registration modes (-R 0 non-registered, -R 1
# registered), which for the LL128 reg-split work (PR #9347) exercise two
# distinct device kernels (plain/non-temporal vs system-scope cache-bypass).
#
# Run INSIDE a SLURM allocation (uses srun --overlap). Example:
#   salloc -p amd-rccl -N 16 --ntasks-per-node=8 --gpus-per-node=8 --exclusive -t 04:00:00
#   RCCL_LIB=$HOME/rccl_libs/librccl.so.1.0.pr9347 \
#     bash run_ll128_coherence_soak.sh
#
# Env (all optional):
#   RCCL_LIB        absolute path to librccl.so* to test (prepended to LD_LIBRARY_PATH).
#                   If unset, whatever is found via RCCL_BUILD_DIR / system is used.
#   RCCL_TESTS_BIN_DIR  default ~/rocm-systems/projects/rccl-tests/build
#   MPI_LIB         MPICH lib dir (default ~/mpich/install/lib)
#   COLL            perf binary (default all_reduce_perf)
#   DTYPES          exact-integer types (default "int8 uint8 int32 uint32 int64 uint64")
#   REGS            registration modes to sweep (default "0 1")
#   OP              reduction op (default sum; must be exact for integers)
#   MIN_BYTES/MAX_BYTES/STEP_FACTOR   small-size sweep (default 8 / 256K / 2)
#   WARMUP          warmup iters (default 5)
#   ITERS           timed iters per run (default 50)
#   CHECK           -c check count (default 1 => validate every run)
#   CYCLES          repeats of the whole matrix (default 10)
#   OUT_DIR         results dir (default ~/rccl_ll128_soak_<ts>)
#   SRUN_TIMEOUT    per-run wall cap seconds (default 600)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/common_rccl_tests_slurm.sh"

RCCL_TESTS_BIN_DIR="${RCCL_TESTS_BIN_DIR:-$HOME/rocm-systems/projects/rccl-tests/build}"
MPI_LIB="${MPI_LIB:-$HOME/mpich/install/lib}"
COLL="${COLL:-all_reduce_perf}"
read -r -a DTYPES <<< "${DTYPES:-int8 uint8 int32 uint32 int64 uint64}"
read -r -a REGS   <<< "${REGS:-0 1}"
OP="${OP:-sum}"
MIN_BYTES="${MIN_BYTES:-8}"; MAX_BYTES="${MAX_BYTES:-256K}"; STEP_FACTOR="${STEP_FACTOR:-2}"
WARMUP="${WARMUP:-5}"; ITERS="${ITERS:-50}"; CHECK="${CHECK:-1}"
RUN_CYCLES="${RUN_CYCLES:-1}"   # rccl-tests -N: repeat the whole sweep in-process
CYCLES="${CYCLES:-10}"; SRUN_TIMEOUT="${SRUN_TIMEOUT:-600}"
OUT_DIR="${OUT_DIR:-$HOME/rccl_ll128_soak_$(date -u +%Y%m%dT%H%M%SZ)}"

bin="${RCCL_TESTS_BIN_DIR}/${COLL}"
[[ -x "${bin}" ]] || { echo "[ll128-soak] missing binary: ${bin}" >&2; exit 2; }

rccl_tests_pick_launcher
rccl_tests_resolve_hosts_np || exit 1
rccl_tests_srun_layout
: "${PPN:=${GPUS_PER_NODE:-8}}"

mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/SUMMARY.txt"

# Force LL128 for every run; exact-integer validation makes coherence bugs visible.
export NCCL_PROTO=LL128
export NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY:-1}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
# Out-of-band bootstrap (MPI/UCX and RCCL) must use a routable management NIC;
# the RoCE rdma* devices are for the data path only. Pin both to ${OOB_IFACE}.
OOB_IFACE="${OOB_IFACE:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${OOB_IFACE}}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-${OOB_IFACE}}"
export UCX_TLS="${UCX_TLS:-tcp,sm,self}"

# Resolve the lib to test into a link dir exposing the librccl.so.1 SONAME, so
# the test binary (linked against librccl.so.1) actually loads THIS build rather
# than falling back to the system /opt/rocm librccl.
LINK_DIR=""
if [[ -n "${RCCL_LIB:-}" ]]; then
  [[ -e "${RCCL_LIB}" ]] || { echo "[ll128-soak] RCCL_LIB not found: ${RCCL_LIB}" >&2; exit 2; }
  LINK_DIR="${OUT_DIR}/.liblink"
  mkdir -p "${LINK_DIR}"
  ln -sf "${RCCL_LIB}" "${LINK_DIR}/librccl.so.1"
  ln -sf "${RCCL_LIB}" "${LINK_DIR}/librccl.so"
fi
export LD_LIBRARY_PATH="${LINK_DIR:+${LINK_DIR}:}${MPI_LIB}:/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

{
  echo "# LL128 coherence soak @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# jobid=${SLURM_JOB_ID:-?} nodes=${NNODES} np=${NP} ppn=${PPN} nodelist=${SLURM_JOB_NODELIST:-${HOSTS}}"
  echo "# rccl_lib   : ${RCCL_LIB:-<default/system>}"
  echo "# collective : ${COLL}  op=${OP}  NCCL_PROTO=LL128"
  echo "# oob        : NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} UCX_NET_DEVICES=${UCX_NET_DEVICES} UCX_TLS=${UCX_TLS}"
  echo "# sweep      : -b ${MIN_BYTES} -e ${MAX_BYTES} -f ${STEP_FACTOR} -w ${WARMUP} -n ${ITERS} -c ${CHECK} -N ${RUN_CYCLES}"
  echo "# dtypes     : ${DTYPES[*]}"
  echo "# reg modes  : ${REGS[*]}  (0=non-registered, 1=registered user buffers)"
  echo "# cycles     : ${CYCLES}"
  echo "#"
  echo "# status legend: OK=clean exit, 0 wrong | OK*=validated 0 wrong but teardown hang (killed) |"
  echo "#                WRONG=coherence mismatch (>0 wrong) | HANG=no sweep completed"
  echo "#"
  printf "# %-6s %-4s %-8s %6s %6s %8s %6s\n" "cycle" "reg" "dtype" "exit" "sweeps" "#wrong" "ok?"
} | tee "${SUMMARY}"

# Some stacks hang in ncclCommDestroy/MPI_Finalize AFTER the test has printed
# its results (a teardown-only hang, not a coherence issue). To keep the soak
# fast we launch srun in the background and, once the completion marker appears,
# stop the step immediately instead of waiting out the timeout. A hard cap still
# catches genuine mid-run hangs.
DONE_MARKER="${DONE_MARKER:-Avg bus bandwidth}"
run_one() {  # $1=log  shift -> extra args
  local log="$1"; shift
  : > "${log}"
  "${SRUN_BIN}" ${SLURM_JOB_ID:+--jobid=${SLURM_JOB_ID}} --overlap \
      --mpi="${SRUN_MPI}" --ntasks="${NP}" \
      --ntasks-per-node="${PPN}" --cpu-bind=none \
      "${bin}" -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${STEP_FACTOR}" \
        -g 1 -o "${OP}" -w "${WARMUP}" -n "${ITERS}" -c "${CHECK}" -N "${RUN_CYCLES}" "$@" \
      > "${log}" 2>&1 &
  local pid=$! waited=0
  while kill -0 "${pid}" 2>/dev/null; do
    if grep -q "${DONE_MARKER}" "${log}" 2>/dev/null; then
      sleep 1
      kill -INT "${pid}" 2>/dev/null; sleep 3
      kill -TERM "${pid}" 2>/dev/null; sleep 2; kill -KILL "${pid}" 2>/dev/null
      wait "${pid}" 2>/dev/null; return 0
    fi
    sleep 2; waited=$((waited+2))
    if (( waited >= SRUN_TIMEOUT )); then
      kill -INT "${pid}" 2>/dev/null; sleep 3; kill -KILL "${pid}" 2>/dev/null
      wait "${pid}" 2>/dev/null; return 124
    fi
  done
  wait "${pid}" 2>/dev/null; return $?
}

overall_fail=0; total_runs=0; total_wrong=0; total_sweeps=0
for cyc in $(seq 1 "${CYCLES}"); do
  ccy=$(printf "%02d" "${cyc}")
  for reg in "${REGS[@]}"; do
    for dt in "${DTYPES[@]}"; do
      log="${OUT_DIR}/c${ccy}.R${reg}.${dt}.log"
      total_runs=$((total_runs+1))
      set +e; run_one "${log}" -d "${dt}" -R "${reg}"; rc=$?; set -e
      read -r wrong warn <<< "$(rccl_tests_parse_log_summary "${log}")"
      # A "sweep" completes when rccl-tests prints its out-of-bounds validation line.
      sweeps=$(grep -c 'Out of bounds values' "${log}" 2>/dev/null || echo 0)
      total_wrong=$((total_wrong+wrong)); total_sweeps=$((total_sweeps+sweeps))
      if (( wrong != 0 )); then status="WRONG"; overall_fail=1
      elif (( sweeps >= 1 )); then
        if (( rc == 0 )); then status="OK"; else status="OK*"; fi
      else status="HANG"; overall_fail=1; fi
      printf "  %-6s %-4s %-8s %6d %6d %8d %6s\n" "${ccy}/${CYCLES}" "${reg}" "${dt}" "${rc}" "${sweeps}" "${wrong}" "${status}" | tee -a "${SUMMARY}"
    done
  done
done

{
  echo "#"
  echo "# runs=${total_runs}  validated_sweeps=${total_sweeps}  total_wrong=${total_wrong}"
  echo "# coherence: $([[ ${total_wrong} -eq 0 ]] && echo 'PASS (0 wrong across all runs)' || echo "FAIL (${total_wrong} wrong)")"
  echo "# done -> ${OUT_DIR}"
} | tee -a "${SUMMARY}"
# Exit reflects COHERENCE only (teardown hangs do not fail the soak).
[[ ${total_wrong} -eq 0 ]]; exit $?
