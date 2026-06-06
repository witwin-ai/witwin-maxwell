#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

namespace {

__global__ void accumulate_point_observer_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* field,
    const int64_t* point_i,
    const int64_t* point_j,
    const int64_t* point_k,
    double weighted_cos,
    double weighted_sin,
    float* real_accum,
    float* imag_accum) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const long long field_offset = offset3d(
      static_cast<unsigned int>(point_i[index]),
      static_cast<unsigned int>(point_j[index]),
      static_cast<unsigned int>(point_k[index]),
      ny,
      nz);
  const float value = field[field_offset];
  real_accum[index] += value * static_cast<float>(weighted_cos);
  imag_accum[index] += value * static_cast<float>(weighted_sin);
}

__global__ void accumulate_plane_observer_kernel(
    int64_t total,
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* field,
    int axis,
    int plane_index,
    double weighted_cos,
    double weighted_sin,
    float* real_accum,
    float* imag_accum) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  unsigned int i = 0;
  unsigned int j = 0;
  unsigned int k = 0;
  if (axis == 0) {
    j = static_cast<unsigned int>(linear / nz);
    k = static_cast<unsigned int>(linear - static_cast<int64_t>(j) * nz);
    i = static_cast<unsigned int>(plane_index);
  } else if (axis == 1) {
    i = static_cast<unsigned int>(linear / nz);
    k = static_cast<unsigned int>(linear - static_cast<int64_t>(i) * nz);
    j = static_cast<unsigned int>(plane_index);
  } else {
    i = static_cast<unsigned int>(linear / ny);
    j = static_cast<unsigned int>(linear - static_cast<int64_t>(i) * ny);
    k = static_cast<unsigned int>(plane_index);
  }
  const float value = field[offset3d(i, j, k, ny, nz)];
  real_accum[linear] += value * static_cast<float>(weighted_cos);
  imag_accum[linear] += value * static_cast<float>(weighted_sin);
}

}  // namespace

void accumulate_point_observers_cuda(
    const at::Tensor& field,
    const at::Tensor& point_i,
    const at::Tensor& point_j,
    const at::Tensor& point_k,
    at::Tensor real_accum,
    at::Tensor imag_accum,
    double weighted_cos,
    double weighted_sin) {
  check_float32_tensor(field, "field");
  check_float32_tensor(real_accum, "real_accum");
  check_float32_tensor(imag_accum, "imag_accum");
  check_contiguous_tensor(field, "field");
  check_contiguous_tensor(real_accum, "real_accum");
  check_contiguous_tensor(imag_accum, "imag_accum");
  TORCH_CHECK(field.dim() == 3, "field must be rank 3");
  TORCH_CHECK(point_i.scalar_type() == at::kLong, "point_i must be int64");
  TORCH_CHECK(point_j.scalar_type() == at::kLong, "point_j must be int64");
  TORCH_CHECK(point_k.scalar_type() == at::kLong, "point_k must be int64");
  TORCH_CHECK(real_accum.sizes() == imag_accum.sizes(), "observer accumulators must match");
  TORCH_CHECK(point_i.numel() == real_accum.numel(), "point_i size must match accumulators");
  TORCH_CHECK(point_j.numel() == real_accum.numel(), "point_j size must match accumulators");
  TORCH_CHECK(point_k.numel() == real_accum.numel(), "point_k size must match accumulators");
  c10::cuda::CUDAGuard guard(field.device());
  const auto sizes = field.sizes();
  accumulate_point_observer_kernel<<<linear_grid(real_accum.numel()), 256, 0, current_cuda_stream()>>>(
      real_accum.numel(),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      field.data_ptr<float>(),
      point_i.data_ptr<int64_t>(),
      point_j.data_ptr<int64_t>(),
      point_k.data_ptr<int64_t>(),
      weighted_cos,
      weighted_sin,
      real_accum.data_ptr<float>(),
      imag_accum.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void accumulate_plane_observer_cuda(
    const at::Tensor& field,
    at::Tensor real_accum,
    at::Tensor imag_accum,
    int64_t axis,
    int64_t plane_index,
    double weighted_cos,
    double weighted_sin) {
  check_float32_tensor(field, "field");
  check_float32_tensor(real_accum, "real_accum");
  check_float32_tensor(imag_accum, "imag_accum");
  check_contiguous_tensor(field, "field");
  check_contiguous_tensor(real_accum, "real_accum");
  check_contiguous_tensor(imag_accum, "imag_accum");
  TORCH_CHECK(field.dim() == 3, "field must be rank 3");
  TORCH_CHECK(axis >= 0 && axis <= 2, "axis must be 0, 1, or 2");
  TORCH_CHECK(real_accum.sizes() == imag_accum.sizes(), "observer accumulators must match");
  c10::cuda::CUDAGuard guard(field.device());
  const auto sizes = field.sizes();
  accumulate_plane_observer_kernel<<<linear_grid(real_accum.numel()), 256, 0, current_cuda_stream()>>>(
      real_accum.numel(),
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      field.data_ptr<float>(),
      static_cast<int>(axis),
      static_cast<int>(plane_index),
      weighted_cos,
      weighted_sin,
      real_accum.data_ptr<float>(),
      imag_accum.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}
