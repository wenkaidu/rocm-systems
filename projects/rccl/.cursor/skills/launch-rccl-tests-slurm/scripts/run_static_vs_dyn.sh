#!/usr/bin/env bash
# Compare a STATICALLY-linked rccl-tests (rccl baked into the binary from a
# librccl*.a archive) against the SAME rccl-tests source linked DYNAMICALLY,
# swapping several librccl.so builds via LD_LIBRARY_PATH. All configs use an
# identical rccl-tests source + headers; only the rccl library (and static-vs-
# dynamic linkage) differ, so any latency gap is pure kernel/linkage, not test
# harness or path-selection differences.
#
# Interleave order: outer=cycle, mid=collective, inner=config, so every config
# is sampled under the same thermal/network state within each cycle.
#
# Build the two binary trees first (see the SKILL.md "Static vs dynamic" section):
#   build/      -> statically linked (RCCL_STATIC_LIB=/path/librccl-dev.a)
#   build_dyn/  -> same source, dynamic (-lrccl), rccl chosen at runtime
#
# Config env:
#   STATIC_BIN_DIR    dir with statically-linked *_perf (default ~/rccl-tests-static/build)
#                     Set empty to skip the static config entirely.
#   STATIC_TAG        label for the static config           (default static)
#   STATIC_ALGO_PROXY .so exposed as librccl.so to the static binary's -M dlopen
#                     for the algo/proto query only (the static binary runs its
#                     baked-in rccl for the actual collective). Use a same-era
#                     develop build whose tuner matches the archive.
#   DYN_BIN_DIR       dir with dynamically-linked *_perf     (default ~/rccl-tests-static/build_dyn)
#   DYN_LIBS          space-separated librccl.so builds to compare dynamically.
#                     Each entry is an absolute path OR a suffix resolved to
#                     ${RCCL_LIB_DIR}/librccl.so.1.0.<suffix>.
#   RCCL_LIB_DIR      where suffix libs + .cmp_* link dirs live (default ~/rccl_libs)
#   MPI_LIB           MPICH lib dir for LD_LIBRARY_PATH      (default ~/mpich/install/lib)
#   COLLECTIVES       space-separated *_perf list            (default all_reduce_perf all_gather_perf)
#   CYCLES            repeats                                 (default 3)
#   DTYPE             rccl-tests -d                           (default float)
#   MIN_BYTES/MAX_BYTES/STEP_FACTOR/WARMUP/ITERS/SRUN_TIMEOUT sweep knobs
#   OUT_DIR           results dir                             (default ~/rccl_mn_perf/static_vs_dyn)
#
# Output: ${OUT_DIR}/results.csv
#   config,collective,cycle,order,size_bytes,oop_us,oop_busbw,ip_us,ip_busbw,algo,proto,nchannels
# Summarize with report_static_vs_dyn.py (baseline defaults to STATIC_TAG).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/common_rccl_tests_slurm.sh"

STATIC_BIN_DIR="${STATIC_BIN_DIR-${HOME}/rccl-tests-static/build}"
STATIC_TAG="${STATIC_TAG:-static}"
DYN_BIN_DIR="${DYN_BIN_DIR:-${HOME}/rccl-tests-static/build_dyn}"
RCCL_LIB_DIR="${RCCL_LIB_DIR:-${HOME}/rccl_libs}"
MPI_LIB="${MPI_LIB:-${HOME}/mpich/install/lib}"
OUT_DIR="${OUT_DIR:-${HOME}/rccl_mn_perf/static_vs_dyn}"
CYCLES="${CYCLES:-3}"
MIN_BYTES="${MIN_BYTES:-8}"; MAX_BYTES="${MAX_BYTES:-1G}"; STEP_FACTOR="${STEP_FACTOR:-2}"
WARMUP="${WARMUP:-5}"; ITERS="${ITERS:-15}"; SRUN_TIMEOUT="${SRUN_TIMEOUT:-400}"
DTYPE="${DTYPE:-float}"
STATIC_ALGO_PROXY="${STATIC_ALGO_PROXY:-${HOME}/rccl_build_wt/2976ade3/build_rel/librccl.so.1.0}"
DYN_LIBS="${DYN_LIBS:-}"

read -r -a COLLECTIVES <<< "${COLLECTIVES:-all_reduce_perf all_gather_perf}"

# Resolve a DYN_LIBS entry to an absolute .so path (suffix => RCCL_LIB_DIR lib).
resolve_lib() {
  local e="$1"
  if [[ "${e}" == /* || -f "${e}" ]]; then echo "${e}"; else
    echo "${RCCL_LIB_DIR}/librccl.so.1.0.${e}"; fi
}
# Short tag for a DYN_LIBS entry (basename suffix).
lib_tag() {
  local e="$1"; e="${e##*/}"; e="${e#librccl.so.1.0.}"; echo "dyn_${e}"
}

# config = "tag|bindir|dlopenlib|static?"
#   dlopenlib = .so exposed as librccl.so on LD_LIBRARY_PATH. For dynamic configs
#   this is BOTH the runtime rccl and the -M algo source. For the static config
#   it is only the -M algo source (proxy); the collective runs the baked-in rccl.
CONFIGS=()
if [[ -n "${STATIC_BIN_DIR}" ]]; then
  CONFIGS+=("${STATIC_TAG}|${STATIC_BIN_DIR}|${STATIC_ALGO_PROXY}|1")
fi
for e in ${DYN_LIBS}; do
  CONFIGS+=("$(lib_tag "${e}")|${DYN_BIN_DIR}|$(resolve_lib "${e}")|0")
done
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "ERROR: nothing to run. Set DYN_LIBS and/or STATIC_BIN_DIR." >&2; exit 2
fi

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

# Per-config link dirs (stable librccl.so/.so.1 names for the -M dlopen).
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r tag bindir lib isstatic <<< "${cfg}"
  [[ -z "${lib}" ]] && continue
  d="${RCCL_LIB_DIR}/.cmp_${tag}"; mkdir -p "${d}"
  ln -sf "${lib}" "${d}/librccl.so"; ln -sf "${lib}" "${d}/librccl.so.1"
done

AFLAG=(-M 1)  # capture per-size algo/proto/#channels

echo "config,collective,cycle,order,size_bytes,oop_us,oop_busbw,ip_us,ip_busbw,algo,proto,nchannels" > "${CSV}"
{
  echo "# static-vs-dyn  launcher=${LAUNCHER} nodes=${NNODES} np=${NP} ppn=${PPN}"
  echo "# jobid=${SLURM_JOB_ID:-?} nodelist=${SLURM_JOB_NODELIST:-${HOSTS}}"
  echo "# configs: ${CONFIGS[*]}"
  echo "# collectives: ${COLLECTIVES[*]}  cycles=${CYCLES}  dtype=${DTYPE}"
  echo "# sweep -b ${MIN_BYTES} -e ${MAX_BYTES} -f ${STEP_FACTOR} -d ${DTYPE} -w ${WARMUP} -n ${ITERS} -c 1"
} | tee "${OUT_DIR}/run.log"

run_one() {  # $1=bin $2=log
  local bin="$1" log="$2"
  timeout --signal=TERM --kill-after=20 "${SRUN_TIMEOUT}" \
    "${SRUN_BIN}" --mpi="${SRUN_MPI}" --kill-on-bad-exit=1 --ntasks="${NP}" \
    --ntasks-per-node="${PPN}" --cpu-bind=none ${SRUN_EXTRA:-} \
    "${bin}" -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${STEP_FACTOR}" \
    -d "${DTYPE}" -g 1 -w "${WARMUP}" -n "${ITERS}" -c 1 "${AFLAG[@]}" > "${log}" 2>&1
}

ORDER=0
for cyc in $(seq 1 "${CYCLES}"); do
  for col in "${COLLECTIVES[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
      IFS='|' read -r tag bindir lib isstatic <<< "${cfg}"
      bin="${bindir}/${col}"
      [[ -x "${bin}" ]] || { echo "skip missing ${bin}"; continue; }
      ORDER=$((ORDER+1))
      log="${OUT_DIR}/${tag}.${col}.c${cyc}.log"
      # link dir first so the -M dlopen("librccl.so") resolves to this config's
      # (proxy for static) lib; dynamic binaries also load it as runtime rccl.
      export LD_LIBRARY_PATH="${RCCL_LIB_DIR}/.cmp_${tag}:${MPI_LIB}:/opt/rocm/lib:/opt/amdgpu/lib/x86_64-linux-gnu"
      run_one "${bin}" "${log}"; rc=$?
      # With -M the last 3 cols are algo, proto, #channels.
      awk -v cf="${tag}" -v col="${col}" -v cyc="${cyc}" -v ord="${ORDER}" '
        /^[[:space:]]*[0-9]+[[:space:]]/ && NF>=15 {
          print cf","col","cyc","ord","$1","$6","$8","$10","$12","$(NF-2)","$(NF-1)","$NF }' "${log}" >> "${CSV}"
      rows=$(grep -cE '^[[:space:]]*[0-9]+[[:space:]]' "${log}")
      echo "c${cyc} ord${ORDER} ${col} ${tag}: rc=${rc} rows=${rows}" | tee -a "${OUT_DIR}/run.log"
    done
  done
done
touch "${DONE}"
echo "DONE static-vs-dyn -> ${CSV}" | tee -a "${OUT_DIR}/run.log"
