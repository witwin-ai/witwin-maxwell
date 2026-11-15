// Zero-initialised device allocator for the sparse direct-solver handles of the near-field LU.
//
// In deterministic mode the direct solver's forward-solve kernel (library 0.8.0)
// reads int workspace it never writes, i.e. it relies on fresh allocations being
// zero. nvmath's default async allocator draws from the device memory pool, whose
// release threshold keeps freed blocks; a later solver then receives the previous
// solver's dirty blocks and the solve indexes out of range ("Out Of Range Address").
// This handler allocates from the same pool on the call stream and clears every
// block with a stream-ordered memset before the library sees it.
#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace {

// Mirror of cudssDeviceMemHandler_t (cudss_data_types.h, public C ABI since cuDSS 0.4).
struct DeviceMemHandler {
  void* ctx;
  int (*device_alloc)(void* ctx, void** ptr, size_t size, cudaStream_t stream);
  int (*device_free)(void* ctx, void* ptr, size_t size, cudaStream_t stream);
  char name[64];
};
static_assert(sizeof(DeviceMemHandler) == 3 * sizeof(void*) + 64, "cudssDeviceMemHandler_t layout");

std::atomic<int64_t> allocations{0};
std::atomic<int64_t> zeroed_bytes{0};

int zeroed_alloc(void* ctx, void** ptr, size_t size, cudaStream_t stream) {
  cudaError_t status = cudaMallocFromPoolAsync(ptr, size, static_cast<cudaMemPool_t>(ctx), stream);
  if (status != cudaSuccess) return static_cast<int>(status);
  status = cudaMemsetAsync(*ptr, 0, size, stream);
  if (status != cudaSuccess) {
    cudaFreeAsync(*ptr, stream);
    *ptr = nullptr;
    return static_cast<int>(status);
  }
  allocations.fetch_add(1, std::memory_order_relaxed);
  zeroed_bytes.fetch_add(static_cast<int64_t>(size), std::memory_order_relaxed);
  return 0;
}

int stream_free(void*, void* ptr, size_t, cudaStream_t stream) {
  return static_cast<int>(cudaFreeAsync(ptr, stream));
}

}  // namespace

// Address of a process-lifetime handler for `pool` (a cudaMemPool_t), for cudssSetDeviceMemHandler.
int64_t layered_cudss_zeroed_allocator(int64_t pool) {
  static std::mutex mutex;
  static std::map<int64_t, DeviceMemHandler> handlers;
  std::lock_guard<std::mutex> lock(mutex);
  auto found = handlers.find(pool);
  if (found == handlers.end()) {
    DeviceMemHandler handler{};
    handler.ctx = reinterpret_cast<void*>(pool);
    handler.device_alloc = zeroed_alloc;
    handler.device_free = stream_free;
    std::strncpy(handler.name, "witwin_zeroed_mempool", sizeof(handler.name) - 1);
    found = handlers.emplace(pool, handler).first;
  }
  return reinterpret_cast<int64_t>(&found->second);
}

// (allocations, bytes) cleared by the handler since process start.
std::vector<int64_t> layered_cudss_zeroed_stats() {
  return {allocations.load(), zeroed_bytes.load()};
}
