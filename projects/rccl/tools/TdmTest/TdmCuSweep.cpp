// Sweep workgroups (≈ CUs) and waves/block for TDM vs vector copies.
// Default is a local HBM copy. Pass "p2p" to copy device0 -> device1 so each
// warp submits its own TDM request against peer memory.

#include <hip/hip_runtime.h>

#include "amd_tdm.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CALL(cmd)                                                                            \
  do {                                                                                           \
    hipError_t error = (cmd);                                                                    \
    if (error != hipSuccess) {                                                                   \
      std::fprintf(stderr, "HIP error '%s' at %s:%d\n", hipGetErrorString(error), __FILE__,      \
                   __LINE__);                                                                    \
      std::exit(-1);                                                                             \
    }                                                                                            \
  } while (0)

static constexpr int kWave = 32;
// TDM 1-D extent is 16-bit (65535). gfx1250 LDS is 320 KiB; ping-pong uses
// 2 windows per warp, so the per-op tile is min(desc max, LDS/(2*warps)).
static constexpr int kDescMax = 65535;
static constexpr int kLdsBytes = 327680;

static int tileBytesForWarps(int warps) {
  const int byLds = (kLdsBytes / (2 * warps)) & ~255;
  const int t = byLds < kDescMax ? byLds : (kDescMax & ~255);
  return t < 256 ? 256 : t;
}

__device__ inline char* stageForWarp(int buf, int tile) {
  extern __shared__ char lds[];
  return lds + (threadIdx.x / kWave) * (2 * tile) + buf * tile;
}

// Each warp stripes tiles and keeps a two-deep TDM pipeline: store of tile i
// is issued before the load of tile i+1 waits, so two requests from that warp
// are outstanding.
__global__ void tdmCopy(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, size_t nBytes,
                        int tile) {
  const int nWarps = blockDim.x / kWave;
  const int warp = threadIdx.x / kWave;
  const int nAll = nWarps * gridDim.x;
  const int id = blockIdx.x * nWarps + warp;
  size_t off = (size_t)id * (size_t)tile;
  if (off >= nBytes) return;
  int buf = 0;
  uint32_t chunk = (uint32_t)(nBytes - off < (size_t)tile ? nBytes - off : tile);
  amd_tdm::cp_async_bulk(amd_tdm::space_shared, amd_tdm::space_global, stageForWarp(buf, tile),
                         in + off, chunk);
  amd_tdm::cp_async_bulk_wait_group_read(amd_tdm::n32_t<0>());
  while (true) {
    const size_t next = off + (size_t)nAll * (size_t)tile;
    amd_tdm::cp_async_bulk(amd_tdm::space_global, amd_tdm::space_shared, out + off,
                           stageForWarp(buf, tile), chunk);
    if (next >= nBytes) {
      amd_tdm::cp_async_bulk_wait_group_read(amd_tdm::n32_t<0>());
      break;
    }
    buf ^= 1;
    chunk = (uint32_t)(nBytes - next < (size_t)tile ? nBytes - next : tile);
    amd_tdm::cp_async_bulk(amd_tdm::space_shared, amd_tdm::space_global, stageForWarp(buf, tile),
                           in + next, chunk);
    amd_tdm::cp_async_bulk_wait_group_read(amd_tdm::n32_t<0>());
    off = next;
  }
}

__global__ void vecCopy(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, size_t nBytes) {
  const size_t n = nBytes / 16;
  const size_t stride = (size_t)blockDim.x * gridDim.x;
  const uint4* src = reinterpret_cast<const uint4*>(in);
  uint4* dst = reinterpret_cast<uint4*>(out);
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) dst[i] = src[i];
}

static float timeMs(void (*fn)(const uint8_t*, uint8_t*, size_t, int, dim3, dim3, size_t),
                    const uint8_t* in, uint8_t* out, size_t nBytes, int tile, dim3 grid, dim3 block,
                    size_t shmem, int iters) {
  hipEvent_t a, b;
  HIP_CALL(hipEventCreate(&a));
  HIP_CALL(hipEventCreate(&b));
  fn(in, out, nBytes, tile, grid, block, shmem);
  HIP_CALL(hipDeviceSynchronize());
  HIP_CALL(hipEventRecord(a));
  for (int i = 0; i < iters; ++i) fn(in, out, nBytes, tile, grid, block, shmem);
  HIP_CALL(hipEventRecord(b));
  HIP_CALL(hipEventSynchronize(b));
  float ms = 0;
  HIP_CALL(hipEventElapsedTime(&ms, a, b));
  HIP_CALL(hipEventDestroy(a));
  HIP_CALL(hipEventDestroy(b));
  return ms / (float)iters;
}

static void launchTdm(const uint8_t* in, uint8_t* out, size_t nBytes, int tile, dim3 grid, dim3 block,
                      size_t shmem) {
  tdmCopy<<<grid, block, shmem>>>(in, out, nBytes, tile);
}

static void launchVec(const uint8_t* in, uint8_t* out, size_t nBytes, int, dim3 grid, dim3 block,
                      size_t) {
  vecCopy<<<grid, block>>>(in, out, nBytes);
}

static void runSweep(const uint8_t* dIn, uint8_t* dOut, size_t nBytes, int nCUs, int iters) {
  const int blocksList[] = {1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96};
  const int warpsList[] = {1, 2, 4, 8};

  std::printf("%6s %6s %8s %8s %10s %10s %8s\n", "CUs", "warps", "threads", "tileB", "TDM_GBs",
              "VEC_GBs", "TDM/VEC");
  for (int warps : warpsList) {
    const int threads = warps * kWave;
    const int tile = tileBytesForWarps(warps);
    const size_t shmem = (size_t)warps * 2 * (size_t)tile;
    for (int blocks : blocksList) {
      if (blocks > nCUs) continue;
      dim3 grid(blocks), block(threads);
      const float tdmMs = timeMs(launchTdm, dIn, dOut, nBytes, tile, grid, block, shmem, iters);
      const float vecMs = timeMs(launchVec, dIn, dOut, nBytes, 0, grid, block, 0, iters);
      const double tdmGBs = (nBytes / 1e9) / (tdmMs / 1e3);
      const double vecGBs = (nBytes / 1e9) / (vecMs / 1e3);
      std::printf("%6d %6d %8d %8d %10.1f %10.1f %8.2f\n", blocks, warps, threads, tile, tdmGBs,
                  vecGBs, tdmGBs / vecGBs);
    }
    std::printf("\n");
  }
}

int main(int argc, char** argv) {
  const size_t nBytes = (argc > 1 ? (size_t)std::atoll(argv[1]) : 32ull << 20);
  const int iters = (argc > 2 ? std::atoi(argv[2]) : 20);
  const bool p2p = (argc > 3 && std::strcmp(argv[3], "p2p") == 0);

  int nDev = 0;
  HIP_CALL(hipGetDeviceCount(&nDev));
  hipDeviceProp_t prop{};
  HIP_CALL(hipGetDeviceProperties(&prop, 0));
  const int nCUs = prop.multiProcessorCount;
  std::printf("Device: %s (%s)  CUs=%d  copy=%zu bytes  iters=%d  mode=%s\n", prop.name,
              prop.gcnArchName, nCUs, nBytes, iters, p2p ? "p2p" : "local");
  if (std::strncmp(prop.gcnArchName, "gfx1250", 7) != 0) {
    std::printf("SKIP: gfx1250 required\n");
    return 0;
  }

  uint8_t *dIn = nullptr, *dOut = nullptr;
  if (p2p) {
    if (nDev < 2) {
      std::printf("SKIP: p2p mode needs 2 devices (have %d)\n", nDev);
      return 0;
    }
    int can01 = 0, can10 = 0;
    HIP_CALL(hipDeviceCanAccessPeer(&can01, 0, 1));
    HIP_CALL(hipDeviceCanAccessPeer(&can10, 1, 0));
    if (!can01 || !can10) {
      std::printf("SKIP: peer access not available (0->1=%d 1->0=%d)\n", can01, can10);
      return 0;
    }
    HIP_CALL(hipSetDevice(0));
    HIP_CALL(hipDeviceEnablePeerAccess(1, 0));
    HIP_CALL(hipMalloc(&dIn, nBytes));
    HIP_CALL(hipMemset(dIn, 0x5a, nBytes));
    HIP_CALL(hipSetDevice(1));
    HIP_CALL(hipDeviceEnablePeerAccess(0, 0));
    HIP_CALL(hipMalloc(&dOut, nBytes));
    HIP_CALL(hipMemset(dOut, 0, nBytes));
    HIP_CALL(hipSetDevice(0));
  } else {
    HIP_CALL(hipMalloc(&dIn, nBytes));
    HIP_CALL(hipMalloc(&dOut, nBytes));
    HIP_CALL(hipMemset(dIn, 0x5a, nBytes));
    HIP_CALL(hipMemset(dOut, 0, nBytes));
  }

  runSweep(dIn, dOut, nBytes, nCUs, iters);

  HIP_CALL(hipSetDevice(0));
  HIP_CALL(hipFree(dIn));
  if (p2p) HIP_CALL(hipSetDevice(1));
  HIP_CALL(hipFree(dOut));
  return 0;
}
