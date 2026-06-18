---
name: build-install-mpich
description: >-
  Download MPICH 4.1.x (default 4.1.2), configure with embedded UCX, no Fortran,
  no SLURM PMI, and install under ${WORKDIR}/mpich/install. Use when you need a
  local mpirun/mpicc tree for rccl-tests or development without the distro MPI.
disable-model-invocation: true
---

# Build and install MPICH (from source)

Script: [scripts/build_install_mpich.sh](scripts/build_install_mpich.sh).

## Prerequisites

- **Compiler toolchain** (C/C++): `gcc` / `g++` or compatible.
- **`wget`** (or install `wget`); the script fails fast if it is missing.
- **Disk**: several GB under `WORKDIR` for the tarball, build tree, and install prefix.

## Usage

Pick a parent directory `WORKDIR`. The tree will be:

- `WORKDIR/mpich/` — unpacked source (and `build/` under it)
- `WORKDIR/mpich/install/` — install prefix (`bin/mpirun`, `bin/mpicc`, …)

```bash
# Optional: override parent directory (default is $HOME)
# export WORKDIR=/path/to/parent
bash .cursor/skills/build-install-mpich/scripts/build_install_mpich.sh
```

The script defaults `WORKDIR` to **`$HOME`** when unset (install ends up at **`~/mpich/install`**).

Optional environment:

| Variable | Default | Meaning |
|----------|---------|---------|
| `WORKDIR` | `$HOME` | Parent path for `mpich/` source and `mpich/install/` prefix |
| `MPICH_VERSION` | `4.1.2` | Upstream tarball version |
| `MPICH_BUILD_JOBS` | `16` | `make -j` parallelism |

Re-running on the same `WORKDIR` after a failed or partial build: remove `WORKDIR/mpich` (and the tarball if you want a fresh download) before invoking again.

Configure flags match a typical **no Fortran**, **embedded UCX**, **no SLURM** build:

`--disable-fortran --with-ucx=embedded --without-slurm`

## After install

```bash
export PATH="${WORKDIR}/mpich/install/bin:${PATH}"
mpirun --version
```

This matches rccl-tests launch scripts that default to `~/mpich/install/bin/mpirun` when `WORKDIR` is left at **`$HOME`**.

To compile **rccl-tests** against this prefix (`MPICH=1`), use [build-rccl-tests](../build-rccl-tests/SKILL.md).
