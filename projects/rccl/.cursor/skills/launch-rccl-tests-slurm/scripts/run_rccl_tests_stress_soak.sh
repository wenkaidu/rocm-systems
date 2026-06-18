#!/usr/bin/env bash
# Repeat the multi-node stress sweep several times (intermittent bug hunting).
#
# Usage:
#   ./run_rccl_tests_stress_soak.sh
#   CYCLES=20 ./run_rccl_tests_stress_soak.sh
#   STRESS_SCRIPT=/path/to/run_rccl_tests_stress_multi_node.sh ./run_rccl_tests_stress_soak.sh
#
# Env:
#   CYCLES=5                    runs of the stress script
#   SOAK_DIR=~/rccl_soak_<ts>   parent directory for cycle logs
#   STRESS_SCRIPT               default: sibling run_rccl_tests_stress_multi_node.sh
# Any HOSTS/NP/ONLY/SKIP/RCCL_* exported in the environment is inherited by each cycle.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_rccl_tests_slurm.sh
source "${SCRIPT_DIR}/common_rccl_tests_slurm.sh"

: "${STRESS_SCRIPT:=${SCRIPT_DIR}/run_rccl_tests_stress_multi_node.sh}"
: "${CYCLES:=5}"
: "${SOAK_DIR:=${HOME}/rccl_tests_stress_soak_$(date -u +%Y%m%dT%H%M%SZ)}"

[[ -x "${STRESS_SCRIPT}" ]] || { echo "[soak] not executable: ${STRESS_SCRIPT}" >&2; exit 2; }
mkdir -p "${SOAK_DIR}"

AGG="${SOAK_DIR}/AGGREGATE.txt"
{
  echo "# run_rccl_tests_stress_soak @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# soak_dir : ${SOAK_DIR}"
  echo "# cycles   : ${CYCLES}"
  echo "# script   : ${STRESS_SCRIPT}"
  echo "#"
  printf "# %-8s %-8s %10s %10s %s\n" "cycle" "result" "tot_wrong" "tot_warn" "log_dir"
} | tee "${AGG}"

overall_fail=0
for (( c=1; c<=CYCLES; c++ )); do
  cyc=$(printf "%02d" "${c}")
  log_dir="${SOAK_DIR}/cycle${cyc}"
  set +e
  "${STRESS_SCRIPT}" "${log_dir}" >/dev/null 2>&1
  rc=$?
  set -e

  tot_wrong=0
  tot_warn=0
  shopt -s nullglob
  for lf in "${log_dir}"/*.log; do
    [[ -f "${lf}" ]] || continue
    read -r w wa <<< "$(rccl_tests_parse_log_summary "${lf}")"
    tot_wrong=$(( tot_wrong + w ))
    tot_warn=$(( tot_warn + wa ))
  done
  shopt -u nullglob

  result="PASS"
  if (( rc != 0 || tot_wrong != 0 )); then
    result="FAIL"
    overall_fail=1
  fi
  printf "  %-8s %-8s %10d %10d %s\n" "${cyc}/${CYCLES}" "${result}" "${tot_wrong}" "${tot_warn}" "${log_dir}" | tee -a "${AGG}"
done

{
  echo "# overall: $([[ ${overall_fail} -eq 0 ]] && echo PASS || echo FAIL)"
  echo "# done -> ${SOAK_DIR}"
} | tee -a "${AGG}"

exit "${overall_fail}"
