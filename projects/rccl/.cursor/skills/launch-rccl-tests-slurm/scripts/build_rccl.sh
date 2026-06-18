#!/usr/bin/env bash
# Generic RCCL CMake + build (no feature-specific toggles).
#
# Usage:
#   ./build_rccl.sh
#   CMAKE_BUILD_TYPE=Release ./build_rccl.sh
#   RCCL_BUILD_DIR=~/build/rccl-rel ./build_rccl.sh
#   ./build_rccl.sh -DENABLE_MSCCL_KERNEL=OFF
#
# Env:
#   RCCL_BUILD_DIR   output dir (default: ~/rocm-systems/projects/rccl/build/release)
#   CMAKE_BUILD_TYPE Debug|Release|RelWithDebInfo (default: RelWithDebInfo)
#   NJOBS            make -j (default: nproc)
#   CLEAN            if 1, rm -rf build dir contents before cmake
#   Remaining argv        extra arguments forwarded to cmake (e.g. -DENABLE_MSCCL_KERNEL=OFF)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts -> launch-rccl-tests-slurm -> skills -> .cursor -> repo
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
: "${RCCL_BUILD_DIR:=${HOME}/rocm-systems/projects/rccl/build/release}"
: "${CMAKE_BUILD_TYPE:=RelWithDebInfo}"
: "${NJOBS:=$(nproc 2>/dev/null || echo 8)}"
: "${CLEAN:=0}"
: "${CXX:=/opt/rocm/bin/hipcc}"

mkdir -p "${RCCL_BUILD_DIR}"
cd "${RCCL_BUILD_DIR}"

if [[ "${CLEAN}" == "1" ]]; then
  rm -rf -- "${RCCL_BUILD_DIR:?}/"*
fi

if [[ ! -f CMakeCache.txt ]]; then
  CXX="${CXX}" cmake \
    -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-/opt/rocm/}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    "$@" \
    "${REPO_ROOT}"
fi

make -j"${NJOBS}"

echo "[build_rccl] OK: ${RCCL_BUILD_DIR}"
