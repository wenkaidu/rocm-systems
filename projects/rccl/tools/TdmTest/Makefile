
ROCM_PATH ?= /opt/rocm
HIPCC     := $(ROCM_PATH)/bin/hipcc

# TDM tensor-descriptor builtins are gfx1250-only. The descriptor encoder
# (ck_tile/core/arch/amd_tdm_descriptor.hpp) ships in the ROCm toolchain include
# tree, so point at $(ROCM_PATH)/include. ck_tile requires C++17+.
OFFLOAD_ARCH ?= gfx1250

EXE := TdmTest

# amd_tdm.h lives in the RCCL device sources; ck_tile ships in the ROCm tree.
CXXFLAGS := --offload-arch=$(OFFLOAD_ARCH) -O3 -std=c++20 \
            -I$(ROCM_PATH)/include -I../../src/device

all: $(EXE)

$(EXE): TdmTest.cpp
	$(HIPCC) $(CXXFLAGS) TdmTest.cpp -o $(EXE)

clean:
	rm -f $(EXE)

.PHONY: all clean
