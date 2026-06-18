#!/usr/bin/env bash
# Build rccl-tests with MPICH=1 and MPI_HOME at the MPICH install prefix
# (default $HOME/mpich/install). Adds mpich/{include,lib} symlinks for Makefile.
#
# Env: RCCL_TESTS_DIR, RCCL_BUILD_DIR, MPI_HOME, ROCM_PATH, HIPCC, NJOBS,
#      GPU_TARGETS, ENABLE_DEVICE_API, NAME_SUFFIX

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts -> build-rccl-tests -> skills -> .cursor -> rccl repo root
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
: "${RCCL_TESTS_DIR:=$(cd "${REPO_ROOT}/.." && pwd)/rccl-tests}"
: "${RCCL_BUILD_DIR:=${HOME}/rocm-systems/projects/rccl/build/release}"
: "${MPI_HOME:=${HOME}/mpich/install}"
: "${ROCM_PATH:=/opt/rocm}"
: "${HIPCC:=${ROCM_PATH}/llvm/bin/amdclang++}"
: "${NJOBS:=$(nproc 2>/dev/null || echo 8)}"

MPI_HOME="$(cd "${MPI_HOME}" && pwd)"

if [[ ! -f "${MPI_HOME}/include/mpi.h" ]]; then
  echo "error: MPICH install missing mpi.h at ${MPI_HOME}/include" >&2
  echo "hint: build MPICH first (see .cursor/skills/build-install-mpich/)" >&2
  exit 1
fi

if [[ ! -f "${RCCL_TESTS_DIR}/Makefile" || ! -f "${RCCL_TESTS_DIR}/src/Makefile" ]]; then
  echo "error: RCCL_TESTS_DIR does not look like rccl-tests: ${RCCL_TESTS_DIR}" >&2
  exit 1
fi

if [[ ! -f "${RCCL_BUILD_DIR}/lib/librccl.so" ]]; then
  echo "error: RCCL build dir missing lib/librccl.so: ${RCCL_BUILD_DIR}" >&2
  exit 1
fi

# rccl-tests src/Makefile (MPICH=1) expects MPI_HOME/{include,lib,mpich/include,mpich/lib}.
# A vanilla MPICH prefix only has include/ and lib/; add Debian-style names under mpich/.
mkdir -p "${MPI_HOME}/mpich"
ln -sfn "../include" "${MPI_HOME}/mpich/include"
ln -sfn "../lib" "${MPI_HOME}/mpich/lib"

cd "${RCCL_TESTS_DIR}"

export ROCM_PATH HIPCC

MAKE_ARGS=(
  MPICH=1
  "MPI_HOME=${MPI_HOME}"
  "NCCL_HOME=${RCCL_BUILD_DIR}"
  "CUSTOM_RCCL_LIB=${RCCL_BUILD_DIR}/lib/librccl.so"
  "HIPCC=${HIPCC}"
)

if [[ -n "${GPU_TARGETS:-}" ]]; then
  MAKE_ARGS+=("GPU_TARGETS=${GPU_TARGETS}")
fi
if [[ -n "${ENABLE_DEVICE_API:-}" ]]; then
  MAKE_ARGS+=("ENABLE_DEVICE_API=${ENABLE_DEVICE_API}")
fi
if [[ -n "${NAME_SUFFIX:-}" ]]; then
  MAKE_ARGS+=("NAME_SUFFIX=${NAME_SUFFIX}")
fi

echo "# RCCL_TESTS_DIR=${RCCL_TESTS_DIR}"
echo "# MPI_HOME=${MPI_HOME}"
echo "# NCCL_HOME=${RCCL_BUILD_DIR}"
echo "# make ${MAKE_ARGS[*]} -j${NJOBS}"

make "${MAKE_ARGS[@]}" -j"${NJOBS}"

echo "[build_rccl_tests] OK: binaries under ${RCCL_TESTS_DIR}/build (BUILDDIR)"
