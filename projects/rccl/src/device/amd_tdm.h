/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef AMD_TDM_H_
#define AMD_TDM_H_

// =====================================================================
// AMD Tensor Data Mover (TDM) - a bulk async copy engine, exposed here
// through an API that mirrors the CUDA / CCCL `cuda::ptx` bulk-copy
// intrinsics so the two backends read the same way at the call site.
//
// CUDA (cp.async.bulk / TMA)                 AMD (this header, TDM)
// ------------------------------------------ ---------------------------
// ptx::cp_async_bulk(dst,src spaces,...)  -> amd_tdm::cp_async_bulk(...)
//   (space_shared, space_global) g->s load ->   __builtin_amdgcn_tensor_load_to_lds
//   (space_global, space_shared) s->g store->   __builtin_amdgcn_tensor_store_from_lds
// ptx::cp_async_bulk_commit_group()       -> amd_tdm::cp_async_bulk_commit_group()  (no-op)
// ptx::cp_async_bulk_wait_group_read(<N>) -> amd_tdm::cp_async_bulk_wait_group_read(n32_t<N>())
//                                              __builtin_amdgcn_s_wait_tensorcnt(N)
//
// Direction is chosen by the (dstSpace, srcSpace) tags, exactly like the
// PTX form. A TDM op is issued once per wave and moves an entire tile
// described by a hardware descriptor (encoded on-device by
// ck_tile::createTDMDescriptor, the authoritative encoder in the ROCm tree).
//
// Semantic notes vs CUDA:
//  * There is no "group": TDM's TENSORcnt increments once per issued op, so
//    cp_async_bulk_commit_group() is a no-op and cp_async_bulk_wait_group*(<N>)
//    maps directly to s_wait_tensorcnt(N) (wait until <=N ops remain in
//    flight; <0>/0 => drain all). For the common wait-for-all (N==0) case
//    this is exact.
//  * There is no mbarrier. The CUDA high-level memcpy_async_tx + _tx-barrier
//    path (which tracks bytes in flight) is replaced by TENSORcnt tracking;
//    memcpy_async() below is the direct equivalent for the global->shared load.
//  * cp.async.bulk is a linear byte copy; it maps to a 1-D TDM tile. The N-D
//    tile helpers (loadTileToLds / storeTileFromLds) are exposed too for the
//    strided / multi-dim cases (the CUDA cp.async.bulk.tensor analogue).
//
// TDM is gfx1250-only. On other targets every entry point compiles to a
// no-op so runtime-guarded code still builds for all archs.
// =====================================================================

#include <cstdint>

// Enable the real implementation only when compiling device code for gfx1250,
// where the tensor builtins and the ck_tile descriptor encoder are available.
#if defined(__gfx1250__)
#define RCCL_TDM_ENABLED 1
#else
#define RCCL_TDM_ENABLED 0
#endif

#if RCCL_TDM_ENABLED
#include "ck_tile/core/arch/amd_tdm_descriptor.hpp"
#endif

namespace amd_tdm {

// ---------------------------------------------------------------------
// Address-space tags and compile-time integer, mirroring cuda::ptx.
// ---------------------------------------------------------------------
struct space_global_t {};
struct space_shared_t {};
inline constexpr space_global_t space_global{};
inline constexpr space_shared_t space_shared{};

// Compile-time integer argument (mirrors cuda::ptx::n32_t<N>). Used so the
// wait count stays a constant, matching the PTX intrinsic's immediate.
template <int N>
struct n32_t {
  static constexpr int value = N;
  __host__ __device__ constexpr operator int() const { return N; }
};

// Runtime/compile query for dispatch code: is the TDM path usable here?
__host__ __device__ inline constexpr bool available() {
  return RCCL_TDM_ENABLED != 0;
}

#if RCCL_TDM_ENABLED

// Descriptor knobs. Defaults describe a plain (non-restore, no barrier, no
// iterate, no pad) contiguous move, which is all the copy path needs.
using Config = ck_tile::TDMConfig;

// ---------------------------------------------------------------------
// Low-level N-D tile primitives (the cp.async.bulk.tensor analogue).
// ---------------------------------------------------------------------

// Load one N-D tile from global memory into LDS.
//   global       - source base pointer in global memory
//   lds          - destination base pointer in LDS (shared memory)
//   globalDim    - N tensor extents (elements)
//   globalStride - N tensor strides (elements)
//   boxDim       - N tile extents to move (elements)
// cpol is the cache-policy immediate (compile-time constant).
template <typename T, int Rank, int cpol = 0>
__device__ inline void loadTileToLds(const void* global, void* lds,
                                     const uint32_t (&globalDim)[Rank],
                                     const uint64_t (&globalStride)[Rank],
                                     const uint16_t (&boxDim)[Rank],
                                     Config cfg = Config{}) {
  auto desc = ck_tile::createTDMDescriptor<T, Rank>(
      global, lds, globalDim, globalStride, boxDim, cfg);
  auto g = desc.getResourceDescriptorGroup();
  __builtin_amdgcn_tensor_load_to_lds(
      g.get(ck_tile::number<0>{}), g.get(ck_tile::number<1>{}),
      g.get(ck_tile::number<2>{}), g.get(ck_tile::number<3>{}),
      g.get(ck_tile::number<4>{}), cpol);
}

// Store one N-D tile from LDS back into global memory.
template <typename T, int Rank, int cpol = 0>
__device__ inline void storeTileFromLds(void* global, const void* lds,
                                        const uint32_t (&globalDim)[Rank],
                                        const uint64_t (&globalStride)[Rank],
                                        const uint16_t (&boxDim)[Rank],
                                        Config cfg = Config{}) {
  auto desc = ck_tile::createTDMDescriptor<T, Rank>(
      global, const_cast<void*>(lds), globalDim, globalStride, boxDim, cfg);
  auto g = desc.getResourceDescriptorGroup();
  __builtin_amdgcn_tensor_store_from_lds(
      g.get(ck_tile::number<0>{}), g.get(ck_tile::number<1>{}),
      g.get(ck_tile::number<2>{}), g.get(ck_tile::number<3>{}),
      g.get(ck_tile::number<4>{}), cpol);
}

namespace detail {
// A linear `bytes`-sized copy is a 1-D byte tile. The tile extent is a 16-bit
// field in the descriptor, so a single op moves up to 65535 bytes (>= the LDS
// staging tiles used by the copy path).
template <int cpol>
__device__ inline void bulkLoad(void* smem, const void* gmem, uint32_t bytes) {
  const uint32_t dim[1] = { bytes };
  const uint64_t str[1] = { 1ull };
  const uint16_t box[1] = { static_cast<uint16_t>(bytes) };
  loadTileToLds<uint8_t, 1, cpol>(gmem, smem, dim, str, box);
}
template <int cpol>
__device__ inline void bulkStore(void* gmem, const void* smem, uint32_t bytes) {
  const uint32_t dim[1] = { bytes };
  const uint64_t str[1] = { 1ull };
  const uint16_t box[1] = { static_cast<uint16_t>(bytes) };
  storeTileFromLds<uint8_t, 1, cpol>(gmem, smem, dim, str, box);
}
}  // namespace detail

// ---------------------------------------------------------------------
// cp.async.bulk-style linear copy. Direction from the (dst, src) space tags.
// ---------------------------------------------------------------------

// (space_shared, space_global): global -> shared load.
template <int cpol = 0>
__device__ inline void cp_async_bulk(space_shared_t, space_global_t,
                                     void* dst, const void* src, uint32_t size) {
  detail::bulkLoad<cpol>(dst, src, size);
}

// (space_global, space_shared): shared -> global store.
template <int cpol = 0>
__device__ inline void cp_async_bulk(space_global_t, space_shared_t,
                                     void* dst, const void* src, uint32_t size) {
  detail::bulkStore<cpol>(dst, const_cast<void*>(src), size);
}

// No group concept on TDM (TENSORcnt tracks each op); kept for API parity.
__device__ inline void cp_async_bulk_commit_group() {}

// Wait until at most N TDM ops remain in flight (<0>/0 => drain all).
template <int N>
__device__ inline void cp_async_bulk_wait_group(n32_t<N>) {
  __builtin_amdgcn_s_wait_tensorcnt(N < 0 ? 0 : N);
}

// _read variant: on CUDA this only guarantees the source is reusable. TDM has
// no separate read/write completion, so it maps to the same TENSORcnt wait
// (which is at least as strong).
template <int N>
__device__ inline void cp_async_bulk_wait_group_read(n32_t<N>) {
  __builtin_amdgcn_s_wait_tensorcnt(N < 0 ? 0 : N);
}

// High-level global->shared load (cuda::device::memcpy_async_tx analogue).
// AMD has no mbarrier; completion is via cp_async_bulk_wait_group*().
template <int cpol = 0>
__device__ inline void memcpy_async(space_shared_t, space_global_t,
                                    void* dst, const void* src, uint32_t size) {
  detail::bulkLoad<cpol>(dst, src, size);
}

#else  // !RCCL_TDM_ENABLED -----------------------------------------------------

// Non-gfx1250 fallbacks: never reached at runtime (guarded by available()),
// but must compile for host and other device targets.
struct Config {};

template <typename T, int Rank, int cpol = 0>
__device__ inline void loadTileToLds(const void*, void*,
                                     const uint32_t (&)[Rank],
                                     const uint64_t (&)[Rank],
                                     const uint16_t (&)[Rank],
                                     Config = Config{}) {}

template <typename T, int Rank, int cpol = 0>
__device__ inline void storeTileFromLds(void*, const void*,
                                        const uint32_t (&)[Rank],
                                        const uint64_t (&)[Rank],
                                        const uint16_t (&)[Rank],
                                        Config = Config{}) {}

template <int cpol = 0>
__device__ inline void cp_async_bulk(space_shared_t, space_global_t,
                                     void*, const void*, uint32_t) {}
template <int cpol = 0>
__device__ inline void cp_async_bulk(space_global_t, space_shared_t,
                                     void*, const void*, uint32_t) {}

__device__ inline void cp_async_bulk_commit_group() {}

template <int N>
__device__ inline void cp_async_bulk_wait_group(n32_t<N>) {}
template <int N>
__device__ inline void cp_async_bulk_wait_group_read(n32_t<N>) {}

template <int cpol = 0>
__device__ inline void memcpy_async(space_shared_t, space_global_t,
                                    void*, const void*, uint32_t) {}

#endif  // RCCL_TDM_ENABLED

}  // namespace amd_tdm

#endif  // AMD_TDM_H_
