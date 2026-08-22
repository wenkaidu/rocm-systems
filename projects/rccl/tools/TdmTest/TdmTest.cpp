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
// TdmTest - exercises the AMD Tensor Data Mover (TDM) through the
// cp.async.bulk-style amd_tdm.h API, which mirrors the CUDA / CCCL
// `cuda::ptx` bulk-copy intrinsics and lowers to
// __builtin_amdgcn_tensor_load_to_lds / tensor_store_from_lds.
//
// The kernel bulk-copies one contiguous buffer global(in) -> LDS -> global(out):
//     cp_async_bulk(space_shared, space_global, lds, in,  bytes)   // g->s load
//     cp_async_bulk_wait_group_read(n32_t<0>())                    // drain
//     cp_async_bulk(space_global, space_shared, out, lds, bytes)   // s->g store
//     cp_async_bulk_wait_group_read(n32_t<0>())                    // drain
// and the host verifies out == in.
//
// TDM is gfx1250-only, so real work happens only when built with
// --offload-arch=gfx1250 AND run on a gfx1250 device; otherwise the kernel is
// a no-op stub and the host reports the feature is unavailable.
// =====================================================================

#include <hip/hip_runtime.h>

#include "amd_tdm.h"

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
// One wave bulk-copies a contiguous buffer: global(in) -> LDS -> global(out)
// using the cp.async.bulk-style amd_tdm API.
// ---------------------------------------------------------------------
__global__ void tdmTensorRoundTrip(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int rows, int cols) {
  extern __shared__ char lds[];
  const uint32_t bytes = static_cast<uint32_t>(rows) * cols * sizeof(float);

  // ---- global(in) -> LDS ----
  amd_tdm::cp_async_bulk(amd_tdm::space_shared, amd_tdm::space_global,
                         lds, in, bytes);
  amd_tdm::cp_async_bulk_commit_group();
  amd_tdm::cp_async_bulk_wait_group_read(amd_tdm::n32_t<0>());  // data in LDS

#if defined(__gfx1250__)
  __builtin_amdgcn_s_barrier();     // LDS visibility before store reads it
#endif

  // ---- LDS -> global(out) ----
  amd_tdm::cp_async_bulk(amd_tdm::space_global, amd_tdm::space_shared,
                         out, lds, bytes);
  amd_tdm::cp_async_bulk_commit_group();
  amd_tdm::cp_async_bulk_wait_group_read(amd_tdm::n32_t<0>());  // data in global
}

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
