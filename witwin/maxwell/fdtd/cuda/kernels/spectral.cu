#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

namespace {

__global__ void accumulate_dft_kernel(
    int64_t total,
    int64_t field_numel,
    const float* field,
    const float* weighted_cos,
    const float* weighted_sin,
    float* real_accum,
    float* imag_accum) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int64_t freq = linear / field_numel;
  const int64_t index = linear - freq * field_numel;
  const float value = field[index];
  real_accum[linear] += value * weighted_cos[freq];
  imag_accum[linear] += value * weighted_sin[freq];
}

void check_dft_component(
    const at::Tensor& field,
    const at::Tensor& real_accum,
    const at::Tensor& imag_accum,
    int64_t frequency_count,
    const char* name) {
  check_float32_tensor(field, name);
  check_float32_tensor(real_accum, "real_accum");
  check_float32_tensor(imag_accum, "imag_accum");
  check_contiguous_tensor(field, name);
  check_contiguous_tensor(real_accum, "real_accum");
  check_contiguous_tensor(imag_accum, "imag_accum");
  TORCH_CHECK(real_accum.dim() == field.dim() + 1, "real_accum rank must be field rank + 1");
  TORCH_CHECK(imag_accum.sizes() == real_accum.sizes(), "imag_accum must match real_accum shape");
  TORCH_CHECK(real_accum.size(0) == frequency_count, "accum frequency dimension mismatch");
  for (int64_t dim = 0; dim < field.dim(); ++dim) {
    TORCH_CHECK(real_accum.size(dim + 1) == field.size(dim), "accum spatial shape must match field");
  }
}

void launch_dft_component(
    const at::Tensor& field,
    at::Tensor real_accum,
    at::Tensor imag_accum,
    const at::Tensor& weighted_cos,
    const at::Tensor& weighted_sin) {
  const int64_t total = real_accum.numel();
  accumulate_dft_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      field.numel(),
      field.data_ptr<float>(),
      weighted_cos.data_ptr<float>(),
      weighted_sin.data_ptr<float>(),
      real_accum.data_ptr<float>(),
      imag_accum.data_ptr<float>());
}

}  // namespace

void accumulate_dft_batched_cuda(
    const at::Tensor& ex,
    const at::Tensor& ey,
    const at::Tensor& ez,
    at::Tensor ex_real,
    at::Tensor ex_imag,
    at::Tensor ey_real,
    at::Tensor ey_imag,
    at::Tensor ez_real,
    at::Tensor ez_imag,
    const at::Tensor& weighted_cos,
    const at::Tensor& weighted_sin) {
  check_float32_tensor(weighted_cos, "weighted_cos");
  check_float32_tensor(weighted_sin, "weighted_sin");
  check_contiguous_tensor(weighted_cos, "weighted_cos");
  check_contiguous_tensor(weighted_sin, "weighted_sin");
  TORCH_CHECK(weighted_cos.dim() == 1, "weighted_cos must be rank 1");
  TORCH_CHECK(weighted_sin.sizes() == weighted_cos.sizes(), "weighted_sin must match weighted_cos");
  const int64_t frequency_count = weighted_cos.size(0);
  check_dft_component(ex, ex_real, ex_imag, frequency_count, "ex");
  check_dft_component(ey, ey_real, ey_imag, frequency_count, "ey");
  check_dft_component(ez, ez_real, ez_imag, frequency_count, "ez");
  c10::cuda::CUDAGuard guard(ex.device());
  launch_dft_component(ex, ex_real, ex_imag, weighted_cos, weighted_sin);
  launch_dft_component(ey, ey_real, ey_imag, weighted_cos, weighted_sin);
  launch_dft_component(ez, ez_real, ez_imag, weighted_cos, weighted_sin);
  WITWIN_CUDA_CHECK();
}
