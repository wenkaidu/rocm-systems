#!/usr/bin/env bash
# Build and install MPICH from the official tarball into ${WORKDIR}/mpich/install.
# Requires: wget, a C/C++ toolchain.
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME}"
MPICH_VERSION="${MPICH_VERSION:-4.1.2}"
MPICH_BUILD_JOBS="${MPICH_BUILD_JOBS:-16}"

if ! command -v wget >/dev/null 2>&1; then
  echo "error: wget not found; install wget or extend this script to use curl" >&2
  exit 1
fi

mkdir -p "${WORKDIR}"
WORKDIR="$(cd "${WORKDIR}" && pwd)"

TARBALL="mpich-${MPICH_VERSION}.tar.gz"
URL="https://www.mpich.org/static/downloads/${MPICH_VERSION}/${TARBALL}"
SRC_DIR="${WORKDIR}/mpich"
BUILD_DIR="${SRC_DIR}/build"
PREFIX="${SRC_DIR}/install"

echo "# WORKDIR=${WORKDIR}"
echo "# MPICH_VERSION=${MPICH_VERSION}"
echo "# prefix=${PREFIX}"

cd "${WORKDIR}"
if [[ ! -f "${TARBALL}" ]]; then
  wget "${URL}"
fi

mkdir -p mpich
tar -zxf "${TARBALL}" -C mpich --strip-components=1

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

../configure --prefix="${PREFIX}" --disable-fortran --with-ucx=embedded --without-slurm

make -j "${MPICH_BUILD_JOBS}"
make install

echo "# done: MPICH installed to ${PREFIX}"
echo "# export PATH=\"${PREFIX}/bin:\${PATH}\""
