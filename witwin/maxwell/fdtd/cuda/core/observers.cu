#include "observers.cuh"
#include <c10/cuda/CUDAException.h>

namespace {
__device__ __forceinline__ float accumulate(float value, float weight, float previous) {
  return __fmaf_rn(value, weight, previous);
}

__device__ __forceinline__ double accumulate(float value, double weight, double previous) {
  return __fma_rn(static_cast<double>(value), weight, previous);
}

template <typename AccumT>
__global__ void fused_observer_gather(CoreObserverBatch<AccumT> batch) {
  int64_t q = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= batch.prefix[batch.group_count]) return;
  int index = 0;
  while (q >= batch.prefix[index + 1]) ++index;
  const auto& group = batch.groups[index];
  q -= batch.prefix[index];
  const int64_t row = q / group.points, lane = row / group.frequencies;
  if (!batch.active[lane]) return;
  const int64_t offset = group.offsets[lane * group.points + q % group.points];
  const int64_t f = group.frequency_map[row];
  if (offset < 0 || offset >= group.field_size || f < 0 || f >= batch.weight_count) return;
  const float value = group.field[offset];
  group.real[q] = accumulate(value, batch.weights[4 * f + group.retard], group.real[q]);
  group.imag[q] = accumulate(value, batch.weights[4 * f + group.retard + 1], group.imag[q]);
}

template <typename AccumT>
void launch(const CoreObserverBatch<AccumT>& batch, cudaStream_t stream) {
  const int64_t total = batch.prefix[batch.group_count];
  fused_observer_gather<<<(total + 255) / 256, 256, 0, stream>>>(batch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

void launch_observer_gather(const CoreObserverBatch<float>& batch, cudaStream_t stream) {
  launch(batch, stream);
}

void launch_observer_gather(const CoreObserverBatch<double>& batch, cudaStream_t stream) {
  launch(batch, stream);
}
