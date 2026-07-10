#include <tuple>
#include <vector>

#include <torch/csrc/stable/ops.h>

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

std::tuple<
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor>
debug_linear_indices_cuda(int64_t size_x, int64_t size_y, int64_t size_z) {
  STD_TORCH_CHECK(size_x > 0 && size_y > 0 && size_z > 0, "shape dimensions must be positive");
  const std::vector<int64_t> shape{size_x, size_y, size_z};
  const int64_t total = shape[0] * shape[1] * shape[2];
  const torch::stable::Device device(
      torch::headeronly::DeviceType::CUDA,
      torch::stable::accelerator::getCurrentDeviceIndex());
  auto linear = torch::stable::empty(
      shape, torch::headeronly::ScalarType::Long, std::nullopt, device);
  auto i_index = torch::stable::empty(
      shape, torch::headeronly::ScalarType::Long, std::nullopt, device);
  auto j_index = torch::stable::empty(
      shape, torch::headeronly::ScalarType::Long, std::nullopt, device);
  auto k_index = torch::stable::empty(
      shape, torch::headeronly::ScalarType::Long, std::nullopt, device);
  torch::stable::accelerator::DeviceGuard guard(linear.get_device_index());
  debug_linear_indices_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      shape[1],
      shape[2],
      linear.mutable_data_ptr<int64_t>(),
      i_index.mutable_data_ptr<int64_t>(),
      j_index.mutable_data_ptr<int64_t>(),
      k_index.mutable_data_ptr<int64_t>());
  WITWIN_CUDA_CHECK();
  return std::make_tuple(linear, i_index, j_index, k_index);
}
