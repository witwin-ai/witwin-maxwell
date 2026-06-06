#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <vector>

#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

namespace {

__global__ void noop_kernel() {}

__global__ void debug_linear_indices_kernel(
    int64_t total,
    int64_t size_y,
    int64_t size_z,
    int64_t* linear,
    int64_t* i_index,
    int64_t* j_index,
    int64_t* k_index) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const Index3D coord = unflatten3d(
      static_cast<unsigned int>(index),
      static_cast<unsigned int>(size_y),
      static_cast<unsigned int>(size_z));
  linear[index] = index;
  i_index[index] = static_cast<int64_t>(coord.i);
  j_index[index] = static_cast<int64_t>(coord.j);
  k_index[index] = static_cast<int64_t>(coord.k);
}

}  // namespace

void synchronize_noop_cuda() {
  noop_kernel<<<1, 1, 0, current_cuda_stream()>>>();
  WITWIN_CUDA_CHECK();
}

std::vector<at::Tensor> debug_linear_indices_cuda(std::vector<int64_t> shape) {
  TORCH_CHECK(shape.size() == 3, "shape must contain exactly three dimensions");
  TORCH_CHECK(shape[0] > 0 && shape[1] > 0 && shape[2] > 0, "shape dimensions must be positive");
  const int64_t total = shape[0] * shape[1] * shape[2];
  auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
  auto linear = at::empty(shape, options);
  auto i_index = at::empty(shape, options);
  auto j_index = at::empty(shape, options);
  auto k_index = at::empty(shape, options);
  c10::cuda::CUDAGuard guard(linear.device());
  debug_linear_indices_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      shape[1],
      shape[2],
      linear.data_ptr<int64_t>(),
      i_index.data_ptr<int64_t>(),
      j_index.data_ptr<int64_t>(),
      k_index.data_ptr<int64_t>());
  WITWIN_CUDA_CHECK();
  return {linear, i_index, j_index, k_index};
}
