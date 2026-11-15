#include <vector>
#include <mutex>
#include <unordered_map>

#include <torch/csrc/stable/ops.h>

#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

#if defined(_WIN32)
#define MW_ALLOC_EXPORT __declspec(dllexport)
#else
#define MW_ALLOC_EXPORT __attribute__((visibility("default")))
#endif

namespace {
struct BatchBudget {
  size_t limit, reserve, used = 0, peak = 0;
  int device;
  bool closed = false;
};
struct BatchAllocation { uint64_t owner; size_t bytes; };
struct BatchDeviceGuard {
  int previous = -1;
  cudaError_t status;
  explicit BatchDeviceGuard(int device) : status(cudaGetDevice(&previous)) {
    if (status == cudaSuccess && previous != device) status = cudaSetDevice(device);
  }
  ~BatchDeviceGuard() { if (previous >= 0) cudaSetDevice(previous); }
};
std::mutex batch_budget_mutex;
std::unordered_map<uint64_t, BatchBudget> batch_budgets;
std::unordered_map<void*, BatchAllocation> batch_allocations;
uint64_t next_batch_budget = 1;
thread_local uint64_t current_batch_budget = 0;
}

extern "C" MW_ALLOC_EXPORT uint64_t mw_batch_budget_create(size_t limit, size_t reserve, int device) {
  std::lock_guard<std::mutex> lock(batch_budget_mutex);
  const auto id = next_batch_budget++;
  batch_budgets.emplace(id, BatchBudget{limit, reserve, 0, 0, device, false});
  return id;
}

extern "C" MW_ALLOC_EXPORT uint64_t mw_batch_budget_exchange(uint64_t id) {
  const auto previous = current_batch_budget;
  current_batch_budget = id;
  return previous;
}

extern "C" MW_ALLOC_EXPORT size_t mw_batch_budget_bytes(uint64_t id, bool peak) {
  std::lock_guard<std::mutex> lock(batch_budget_mutex);
  const auto found = batch_budgets.find(id);
  return found == batch_budgets.end() ? 0 : (peak ? found->second.peak : found->second.used);
}

extern "C" MW_ALLOC_EXPORT void mw_batch_budget_close(uint64_t id) {
  std::lock_guard<std::mutex> lock(batch_budget_mutex);
  const auto found = batch_budgets.find(id);
  if (found == batch_budgets.end()) return;
  found->second.closed = true;
  if (!found->second.used) batch_budgets.erase(found);
}

extern "C" MW_ALLOC_EXPORT void* mw_batch_malloc(size_t bytes, int device, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(batch_budget_mutex);
  const auto found = batch_budgets.find(current_batch_budget);
  if (found == batch_budgets.end() || found->second.closed || found->second.device != device) return nullptr;
  auto& budget = found->second;
  if (bytes > budget.limit - budget.used) return nullptr;
  BatchDeviceGuard guard(device);
  if (guard.status != cudaSuccess) return nullptr;
  size_t free_bytes = 0, total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || free_bytes < budget.reserve ||
      bytes > free_bytes - budget.reserve) return nullptr;
  void* pointer = nullptr;
  if (cudaMalloc(&pointer, bytes) != cudaSuccess) return nullptr;
  budget.used += bytes;
  budget.peak = std::max(budget.peak, budget.used);
  batch_allocations.emplace(pointer, BatchAllocation{current_batch_budget, bytes});
  return pointer;
}

extern "C" MW_ALLOC_EXPORT void mw_batch_free(void* pointer, size_t bytes, int device, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(batch_budget_mutex);
  const auto allocation = batch_allocations.find(pointer);
  if (allocation == batch_allocations.end()) return;
  BatchDeviceGuard guard(device);
  if (guard.status != cudaSuccess) return;
  // The caching allocator calls this only after all stream uses have completed.
  if (cudaFree(pointer) != cudaSuccess) return;
  const auto owner = batch_budgets.find(allocation->second.owner);
  if (owner != batch_budgets.end()) {
    owner->second.used -= allocation->second.bytes;
    if (owner->second.closed && !owner->second.used) batch_budgets.erase(owner);
  }
  batch_allocations.erase(allocation);
}

namespace {

__global__ void noop_kernel() {}

}  // namespace

void synchronize_noop_cuda() {
  noop_kernel<<<1, 1, 0, current_cuda_stream()>>>();
  WITWIN_CUDA_CHECK();
}

