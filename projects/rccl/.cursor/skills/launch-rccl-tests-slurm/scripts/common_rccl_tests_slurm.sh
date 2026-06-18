#!/usr/bin/env bash
# Shared helpers for rccl-tests launchers under SLURM (sourced, not executed).
# shellcheck shell=bash

# Repository root: .../.cursor/skills/launch-rccl-tests-slurm/scripts/thisfile
rccl_tests_repo_root() {
  (cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
}

# Populate HOSTS as "h1:N,h2:N,..." and NP as total GPU count when unset.
# Requires SLURM allocation or explicit HOSTS. GPUS_PER_NODE from SLURM when set.
rccl_tests_resolve_hosts_np() {
  : "${GPUS_PER_NODE:=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-8}}}}"
  if [[ -z "${HOSTS:-}" ]]; then
    local _nodelist="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}"
    if [[ -z "${_nodelist}" && -n "${SLURM_JOB_ID:-}" ]] && command -v squeue >/dev/null 2>&1; then
      _nodelist=$(squeue -h -j "${SLURM_JOB_ID}" -o '%N' 2>/dev/null || true)
    fi
    if [[ -n "${_nodelist}" ]] && command -v scontrol >/dev/null 2>&1; then
      HOSTS=$(scontrol show hostnames "${_nodelist}" \
        | sed "s/\$/:${GPUS_PER_NODE}/" | paste -sd, -)
    fi
  fi
  if [[ -z "${HOSTS:-}" ]]; then
    echo "[rccl-tests] Set HOSTS (e.g. host1:8,host2:8) or run inside a SLURM allocation (SLURM_JOB_ID)." >&2
    return 1
  fi
  if [[ -z "${NP:-}" ]]; then
    NP=0
    local _h _n
    IFS=, read -r -a _hostlist <<< "${HOSTS}"
    for _h in "${_hostlist[@]}"; do
      _n="${_h##*:}"
      if [[ "${_h}" == *:* && "${_n}" =~ ^[0-9]+$ ]]; then
        NP=$(( NP + _n ))
      else
        NP=$(( NP + 1 ))
      fi
    done
    (( NP > 0 )) || NP=16
  fi
  return 0
}

# NNODES / PPN for srun from HOSTS and NP.
rccl_tests_srun_layout() {
  NNODES=0
  local _h
  IFS=, read -r -a _hl <<< "${HOSTS}"
  for _h in "${_hl[@]}"; do [[ -n "${_h}" ]] && NNODES=$(( NNODES + 1 )); done
  (( NNODES > 0 )) || NNODES=1
  if (( NP % NNODES == 0 )); then PPN=$(( NP / NNODES )); else NNODES=1; PPN="${NP}"; fi
}

# LAUNCHER: auto | srun | mpirun
rccl_tests_pick_launcher() {
  : "${LAUNCHER:=auto}"
  : "${SRUN_BIN:=srun}"
  : "${SRUN_MPI:=pmi2}"
  if [[ "${LAUNCHER}" == "auto" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then LAUNCHER=srun; else LAUNCHER=mpirun; fi
  fi
}

# Echo a colon-separated prefix for LD_LIBRARY_PATH so librccl is found for
# both layouts: CMake default (${RCCL_BUILD_DIR}/lib/...) and flat trees that
# place librccl.so directly under ${RCCL_BUILD_DIR}/.
rccl_tests_rccl_ld_prefix() {
  local root="${RCCL_BUILD_DIR:-}"
  [[ -n "${root}" ]] || return 0
  if [[ -d "${root}/lib" ]]; then
    printf '%s:%s' "${root}/lib" "${root}"
  else
    printf '%s' "${root}"
  fi
}

# Summarize one log: wrong count, WARN/ERROR count (printed as two fields).
rccl_tests_parse_log_summary() {
  local log=$1
  local wrong
  wrong=$(awk '
    /^[[:space:]]*[0-9]+[[:space:]]/ {
      if (NF >= 13) { w += $9 + $13 } else if (NF >= 9) { w += $9 }
    }
    END { print w+0 }
  ' "${log}")
  local warn
  warn=$(grep -cE "NCCL (WARN|ERROR)|RCCL (WARN|ERROR)" "${log}" 2>/dev/null || true)
  wrong=${wrong:-0}
  warn=${warn:-0}
  printf '%s %s' "${wrong}" "${warn}"
}
