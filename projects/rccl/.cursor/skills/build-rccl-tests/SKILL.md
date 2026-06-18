---
name: build-rccl-tests
description: >-
  Build rocm-systems rccl-tests with MPICH=1 and MPI_HOME at the MPICH install
  prefix (default ~/mpich/install). Adds mpich/include and mpich/lib symlinks
  so src/Makefile matches a user-built MPICH. Use with a custom RCCL tree.
disable-model-invocation: true
---

# Build rccl-tests (Makefile, MPICH)

Script: [scripts/build_rccl_tests.sh](scripts/build_rccl_tests.sh).

## Prerequisites

- **ROCm / HIP** at `ROCM_PATH` (default `/opt/rocm`).
- **RCCL** artifacts for `NCCL_HOME` + `CUSTOM_RCCL_LIB` (defaults point at the
  CMake build tree under `~/rocm-systems/projects/rccl/build/release`).
- **MPICH** install prefix (default **`~/mpich/install`**). See
  [build-install-mpich](../build-install-mpich/SKILL.md) if you need to build MPI first.

`src/Makefile` with **`MPICH=1`** expects **`MPI_HOME/mpich/include`** and
**`MPI_HOME/mpich/lib`** in addition to top-level **`include/`** and **`lib/`**.
A normal MPICH prefix only has the top-level dirs. The script adds
**`mpich/include` → `../include`** and **`mpich/lib` → `../lib`** under the same
**`MPI_HOME`** so the makefile resolves.

## Usage

From anywhere:

```bash
bash .cursor/skills/build-rccl-tests/scripts/build_rccl_tests.sh
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `RCCL_TESTS_DIR` | `<rccl repo>/../rccl-tests` | rccl-tests checkout (contains top-level `Makefile`) |
| `RCCL_BUILD_DIR` | `$HOME/rocm-systems/projects/rccl/build/release` | Passed as `NCCL_HOME`; `lib/librccl.so` used for `CUSTOM_RCCL_LIB` |
| `MPI_HOME` | `$HOME/mpich/install` | MPICH install prefix passed to `make` |
| `ROCM_PATH` | `/opt/rocm` | ROCm root |
| `HIPCC` | `$ROCM_PATH/llvm/bin/amdclang++` | HIP C++ compiler |
| `NJOBS` | `$(nproc)` | `make -j` |

Optional makefile knobs (forwarded when set): `GPU_TARGETS`, `ENABLE_DEVICE_API`, `NAME_SUFFIX`.

Equivalent manual recipe:

```bash
make MPICH=1 MPI_HOME="$HOME/mpich/install" NCCL_HOME=... CUSTOM_RCCL_LIB=... HIPCC=... -j"$(nproc)"
```

(after `mpich/include` and `mpich/lib` exist under `MPI_HOME`; the script creates them.)
