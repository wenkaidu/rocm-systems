#!/usr/bin/env bash
# Compare MULTIPLE RCCL libraries in ONE allocation, interleaved across cycles,
# capturing per-size algorithm/protocol/#channels (rccl-tests -A 1).
#
# Why interleave: running every lib back-to-back within each (cycle, collective)
# samples all libs under the same thermal / network state, so run-order and drift
# bias are shared rather than attributed to one library. Loop order is
#   outer = cycle, middle = collective, inner = library.
#
# Output: ${OUT_DIR}/results.csv  (commit,collective,cycle,order,size_bytes,
#         oop_us,ip_us,algo,proto,nchannels), plus run.log and a DONE marker.
# Feed results.csv to report_multi_lib.py for per-protocol median/geomean tables.
#
# Key env (all optional):
#   RCCL_LIBS       space/comma list of libs to compare. Each entry is either an
#                   absolute path to a librccl .so, or a commit/tag suffix that is
#                   resolved to ${RCCL_LIB_DIR}/librccl.so.1.0.<suffix>. The FIRST
#                   entry is treated as the baseline by the reporter.
#                   (default: every librccl.so.1.0.* under RCCL_LIB_DIR)
#   RCCL_LIB_DIR    dir holding prebuilt libs (default ~/rccl_libs)
#   COLLECTIVES     space list of *_perf binaries (default: the reduction/gather
#                   + p2p set below). Use COLLECTIVES=all for every *_perf.
#   CYCLES          repeats of the whole interleaved sweep (default 5)
#   MIN_BYTES MAX_BYTES STEP_FACTOR WARMUP ITERS   sweep knobs (8 1G 2 5 15)
#   CAPTURE_ALGO    1 → add -A 1 and expose a per-lib librccl.so symlink so the
#                   -A dlopen resolves to the tested lib (default 1)
#   RCCL_DIRECT_ALLGATHER_DISABLE   set to 1 to force all_gather onto the standard
#                   RING (LL/LL128/SIMPLE) path (exported as-is when set)
#   OUT_DIR         results dir (default ~/rccl_mn_perf/multi_lib)
#   SRUN_TIMEOUT    per-run wall cap in seconds (default 400)
#   EXTRA_ENV       space list of VAR=VAL exported to every run and logged, e.g.
#                   "RCCL_GFX9_CHEAP_FENCE_OFF=0 NCCL_PROTO=LL128" (for A/B knobs)
#   SRUN_EXTRA      extra srun flags, e.g. "--jobid=<id> --overlap" to run inside
#                   an existing interactive allocation alongside its shell step
#   RCCL_TESTS_BIN_DIR, MPI_LIB, GPUS_PER_NODE, NP, HOSTS, LAUNCHER  as usual
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/common_rccl_tests_slurm.sh"

RCCL_TESTS_BIN_DIR="${RCCL_TESTS_BIN_DIR:-$HOME/rocm-systems/projects/rccl-tests/build}"
RCCL_LIB_DIR="${RCCL_LIB_DIR:-$HOME/rccl_libs}"
MPI_LIB="${MPI_LIB:-$HOME/mpich/install/lib}"
OUT_DIR="${OUT_DIR:-$HOME/rccl_mn_perf/multi_lib}"
CYCLES="${CYCLES:-5}"
MIN_BYTES="${MIN_BYTES:-8}"; MAX_BYTES="${MAX_BYTES:-1G}"; STEP_FACTOR="${STEP_FACTOR:-2}"
WARMUP="${WARMUP:-5}"; ITERS="${ITERS:-15}"; SRUN_TIMEOUT="${SRUN_TIMEOUT:-400}"
CAPTURE_ALGO="${CAPTURE_ALGO:-1}"

# ----- collectives -------------------------------------------------------------
DEFAULT_COLLECTIVES="all_reduce_perf all_gather_perf reduce_scatter_perf \
broadcast_perf reduce_perf alltoall_perf sendrecv_perf"
if [[ "${COLLECTIVES:-}" == "all" ]]; then
  mapfile -t _colls < <(cd "${RCCL_TESTS_BIN_DIR}" && ls -1 ./*_perf 2>/dev/null | sed 's#^\./##')
else
  read -r -a _colls <<< "${COLLECTIVES:-${DEFAULT_COLLECTIVES}}"
fi
(( ${#_colls[@]} > 0 )) || { echo "[multi-lib] no collectives found in ${RCCL_TESTS_BIN_DIR}" >&2; exit 1; }

# ----- libraries ---------------------------------------------------------------
resolve_lib() {  # arg: path or commit suffix -> absolute .so path (or empty)
  local x="$1"
  if [[ -f "${x}" ]]; then printf '%s' "${x}"; return; fi
  local p="${RCCL_LIB_DIR}/librccl.so.1.0.${x}"
  [[ -f "${p}" ]] && printf '%s' "${p}"
}
LIBS=()
if [[ -n "${RCCL_LIBS:-}" ]]; then
  IFS=', ' read -r -a _want <<< "${RCCL_LIBS}"
  for w in "${_want[@]}"; do
    [[ -z "${w}" ]] && continue
    r="$(resolve_lib "${w}")"
    if [[ -n "${r}" ]]; then LIBS+=("${r}"); else echo "[multi-lib] WARN missing lib: ${w}" >&2; fi
  done
else
  mapfile -t LIBS < <(find "${RCCL_LIB_DIR}" -maxdepth 1 -type f -name 'librccl.so.1.0.*' \
                        ! -name '*amdgcn*' ! -name '*host-x86*' | sort)
fi
(( ${#LIBS[@]} > 0 )) || { echo "[multi-lib] no RCCL libs (set RCCL_LIBS or populate ${RCCL_LIB_DIR})" >&2; exit 1; }

# ----- launcher + layout -------------------------------------------------------
rccl_tests_pick_launcher
rccl_tests_resolve_hosts_np || exit 1
rccl_tests_srun_layout
: "${PPN:=${GPUS_PER_NODE:-8}}"

mkdir -p "${OUT_DIR}"
CSV="${OUT_DIR}/results.csv"
DONE="${OUT_DIR}/DONE"
rm -f "${DONE}"
trap 'touch "${DONE}"' EXIT TERM

export NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY:-1}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
[[ -n "${RCCL_DIRECT_ALLGATHER_DISABLE:-}" ]] && export RCCL_DIRECT_ALLGATHER_DISABLE
[[ -n "${RCCL_DIRECT_REDUCE_SCATTER_DISABLE:-}" ]] && export RCCL_DIRECT_REDUCE_SCATTER_DISABLE
# Arbitrary extra env (VAR=VAL ...) exported to every run so A/B tuning knobs
# (e.g. RCCL_GFX9_CHEAP_FENCE_OFF=0) reach the srun tasks and are captured below.
for _kv in ${EXTRA_ENV:-}; do [[ "${_kv}" == *=* ]] && export "${_kv?}"; done

# Pre-create per-lib symlink dirs so -A's dlopen("librccl.so") resolves to the
# tested lib (needed because we select libs via LD_LIBRARY_PATH, not LD_PRELOAD).
declare -A LINKOF
for lib in "${LIBS[@]}"; do
  commit="${lib##*.}"; link="${RCCL_LIB_DIR}/.link_${commit}"; mkdir -p "${link}"
  ln -sf "${lib}" "${link}/librccl.so"; ln -sf "${lib}" "${link}/librccl.so.1"
  LINKOF["${commit}"]="${link}"
done

AFLAG=(); [[ "${CAPTURE_ALGO}" == "1" ]] && AFLAG=(-A 1)

echo "commit,collective,cycle,order,size_bytes,oop_us,ip_us,algo,proto,nchannels" > "${CSV}"
{
  echo "# multi-lib interleaved  launcher=${LAUNCHER} nodes=${NNODES} np=${NP} ppn=${PPN}"
  echo "# jobid=${SLURM_JOB_ID:-?}  nodelist=${SLURM_JOB_NODELIST:-${HOSTS}}"
  echo "# libs: ${LIBS[*]}"
  echo "# collectives: ${_colls[*]}"
  echo "# cycles=${CYCLES} sweep -b ${MIN_BYTES} -e ${MAX_BYTES} -f ${STEP_FACTOR} -w ${WARMUP} -n ${ITERS} ${AFLAG[*]}"
  echo "# direct_allgather_disable=${RCCL_DIRECT_ALLGATHER_DISABLE:-<unset>} direct_reduce_scatter_disable=${RCCL_DIRECT_REDUCE_SCATTER_DISABLE:-<unset>}"
  echo "# extra_env=${EXTRA_ENV:-<none>}"
} | tee "${OUT_DIR}/run.log"

run_one() {  # $1=bin $2=logfile
  local bin="$1" log="$2"
  if [[ "${LAUNCHER}" == "srun" ]]; then
    # SRUN_EXTRA lets callers add flags such as "--jobid=<id> --overlap" to run
    # steps inside an existing (e.g. interactive) allocation alongside a shell.
    timeout --signal=TERM --kill-after=20 "${SRUN_TIMEOUT}" \
      "${SRUN_BIN}" --mpi="${SRUN_MPI}" --kill-on-bad-exit=1 --ntasks="${NP}" \
      --ntasks-per-node="${PPN}" --cpu-bind=none ${SRUN_EXTRA:-} \
      "${bin}" -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${STEP_FACTOR}" \
      -g 1 -w "${WARMUP}" -n "${ITERS}" -c 1 "${AFLAG[@]}" > "${log}" 2>&1
  else
    timeout --signal=TERM --kill-after=20 "${SRUN_TIMEOUT}" \
      "${MPIRUN_BIN:-$HOME/mpich/install/bin/mpirun}" -np "${NP}" --hosts "${HOSTS}" \
      "${bin}" -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${STEP_FACTOR}" \
      -g 1 -w "${WARMUP}" -n "${ITERS}" -c 1 "${AFLAG[@]}" > "${log}" 2>&1
  fi
}

ORDER=0
for cyc in $(seq 1 "${CYCLES}"); do
  for col in "${_colls[@]}"; do
    for lib in "${LIBS[@]}"; do
      commit="${lib##*.}"; bin="${RCCL_TESTS_BIN_DIR}/${col}"
      [[ -x "${bin}" ]] || { echo "skip missing ${bin}"; continue; }
      log="${OUT_DIR}/${commit}.${col}.c${cyc}.log"
      ORDER=$((ORDER+1))
      export LD_LIBRARY_PATH="${LINKOF[$commit]}:${MPI_LIB}:/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      run_one "${bin}" "${log}"; rc=$?
      # $6 = out-of-place time(us), $10 = in-place time(us); with -A the last
      # three columns are algo, proto, #channels (N/A for p2p collectives).
      awk -v cm="${commit}" -v col="${col}" -v cyc="${cyc}" -v ord="${ORDER}" '
        /^[[:space:]]*[0-9]+[[:space:]]/ && NF>=14 {
          print cm","col","cyc","ord","$1","$6","$10","$(NF-2)","$(NF-1)","$NF }' "${log}" >> "${CSV}"
      rows=$(grep -cE '^[[:space:]]*[0-9]+[[:space:]]' "${log}")
      echo "c${cyc} ord${ORDER} ${col} ${commit}: rc=${rc} rows=${rows}" | tee -a "${OUT_DIR}/run.log"
    done
  done
done
touch "${DONE}"
echo "DONE multi-lib -> ${CSV}" | tee -a "${OUT_DIR}/run.log"
