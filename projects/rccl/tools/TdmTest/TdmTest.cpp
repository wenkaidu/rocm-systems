/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// =====================================================================
// TdmTest - exercises the AMD Tensor Data Mover (TDM) *tensor-descriptor*
// path on gfx1250 via __builtin_amdgcn_tensor_load_to_lds /
// __builtin_amdgcn_tensor_store_from_lds.
//
// Unlike the per-lane async LDS-DMA path, a TDM tensor op is issued once per
// wave and moves an entire 2D tile (box) described by a hardware descriptor.
// The descriptor is built on-device with ck_tile::createTDMDescriptor (the
// authoritative encoder shipped in the ROCm toolchain), and its five chunks
// D0..D4 are fed to the builtins. TENSORcnt (s_wait_tensorcnt) tracks
// completion.
//
// The kernel copies one rows x cols float tile:
//     global(in) --tensor_load_to_lds--> LDS --tensor_store_from_lds--> global(out)
// and the host verifies out == in.
//
// TDM is gfx1250-only, so real work happens only when built with
// --offload-arch=gfx1250 AND run on a gfx1250 device; otherwise the kernel is
// a no-op stub and the host reports the feature is unavailable.
// =====================================================================

#include <hip/hip_runtime.h>

#if defined(__gfx1250__)
#include "ck_tile/core/arch/amd_tdm_descriptor.hpp"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CALL(cmd)                                                            \
    do {                                                                         \
        hipError_t error = (cmd);                                                \
        if (error != hipSuccess) {                                               \
            std::fprintf(stderr, "HIP error '%s' at %s:%d\n",                    \
                         hipGetErrorString(error), __FILE__, __LINE__);          \
            std::exit(-1);                                                       \
        }                                                                        \
    } while (0)

static constexpr int kWave = 32;   // gfx1250 is wave32; tensor DMA needs wave32

// ---------------------------------------------------------------------
// One wave copies a rows x cols tile: global(in) -> LDS -> global(out)
// using the TDM tensor-descriptor builtins.
// ---------------------------------------------------------------------
#if defined(__gfx1250__)
__global__ void tdmTensorRoundTrip(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int rows, int cols) {
  extern __shared__ float lds[];

  // Contiguous row-major rows x cols tensor; the whole tensor is one tile.
  uint32_t globalDim[2] = { static_cast<uint32_t>(rows), static_cast<uint32_t>(cols) };
  uint64_t globalStr[2] = { static_cast<uint64_t>(cols), 1ull };  // element strides
  uint16_t boxDim[2]    = { static_cast<uint16_t>(rows), static_cast<uint16_t>(cols) };

  ck_tile::TDMConfig cfg{};   // defaults: not-restore, no barrier/iterate/pad

  // ---- global(in) -> LDS ----
  {
    auto desc = ck_tile::createTDMDescriptor<float, 2>(
        static_cast<const void*>(in), static_cast<void*>(lds),
        globalDim, globalStr, boxDim, cfg);
    auto g = desc.getResourceDescriptorGroup();
    __builtin_amdgcn_tensor_load_to_lds(
        g.get(ck_tile::number<0>{}), g.get(ck_tile::number<1>{}),
        g.get(ck_tile::number<2>{}), g.get(ck_tile::number<3>{}),
        g.get(ck_tile::number<4>{}), /*cpol*/ 0);
  }
  __builtin_amdgcn_s_wait_tensorcnt(0);   // load complete: data in LDS
  __builtin_amdgcn_s_barrier();           // LDS visibility before store reads it

  // ---- LDS -> global(out) ----
  {
    auto desc = ck_tile::createTDMDescriptor<float, 2>(
        static_cast<const void*>(out), static_cast<void*>(lds),
        globalDim, globalStr, boxDim, cfg);
    auto g = desc.getResourceDescriptorGroup();
    __builtin_amdgcn_tensor_store_from_lds(
        g.get(ck_tile::number<0>{}), g.get(ck_tile::number<1>{}),
        g.get(ck_tile::number<2>{}), g.get(ck_tile::number<3>{}),
        g.get(ck_tile::number<4>{}), /*cpol*/ 0);
  }
  __builtin_amdgcn_s_wait_tensorcnt(0);   // store complete: data in global
}
#else
__global__ void tdmTensorRoundTrip(const float*, float*, int, int) {}
#endif

static bool deviceSupportsTdm() {
  hipDeviceProp_t prop{};
  HIP_CALL(hipGetDeviceProperties(&prop, 0));
  std::printf("Device 0: %s (gcnArch: %s)\n", prop.name, prop.gcnArchName);
  return std::strncmp(prop.gcnArchName, "gfx1250", 7) == 0;
}

int main(int argc, char** argv) {
  const int rows = (argc > 1 ? std::atoi(argv[1]) : 64);
  const int cols = (argc > 2 ? std::atoi(argv[2]) : 64);
  const int numElts  = rows * cols;
  const size_t bytes = static_cast<size_t>(numElts) * sizeof(float);

  std::printf("TdmTest (tensor-descriptor): %d x %d floats = %zu bytes (LDS tile)\n",
              rows, cols, bytes);

  if (!deviceSupportsTdm()) {
    std::printf("SKIP: TDM tensor DMA requires a gfx1250 device. "
                "Build with --offload-arch=gfx1250 and run on gfx1250.\n");
    return 0;
  }

  std::vector<float> hIn(numElts), hOut(numElts, -1.0f);
  for (int i = 0; i < numElts; ++i) hIn[i] = static_cast<float>(i) * 1.5f + 0.25f;

  float *dIn = nullptr, *dOut = nullptr;
  HIP_CALL(hipMalloc(&dIn, bytes));
  HIP_CALL(hipMalloc(&dOut, bytes));
  HIP_CALL(hipMemcpy(dIn, hIn.data(), bytes, hipMemcpyHostToDevice));
  HIP_CALL(hipMemset(dOut, 0, bytes));

  tdmTensorRoundTrip<<<1, kWave, bytes>>>(dIn, dOut, rows, cols);
  HIP_CALL(hipGetLastError());
  HIP_CALL(hipDeviceSynchronize());

  HIP_CALL(hipMemcpy(hOut.data(), dOut, bytes, hipMemcpyDeviceToHost));

  int mismatches = 0;
  for (int i = 0; i < numElts && mismatches < 10; ++i) {
    if (hOut[i] != hIn[i]) {
      std::printf("  mismatch at %d: got %f expected %f\n", i, hOut[i], hIn[i]);
      ++mismatches;
    }
  }

  HIP_CALL(hipFree(dIn));
  HIP_CALL(hipFree(dOut));

  if (mismatches == 0) {
    std::printf("PASS: tensor-descriptor round-trip global->LDS->global matches input\n");
    return 0;
  }
  std::printf("FAIL: %d mismatches detected\n", mismatches);
  return 1;
}
