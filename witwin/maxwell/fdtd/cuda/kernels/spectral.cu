#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

namespace {

__global__ void accumulate_dft_batched_kernel(
    int64_t ex_numel,
    int64_t ey_numel,
    int64_t ez_numel,
    const float* __restrict__ ex,
    const float* __restrict__ ey,
    const float* __restrict__ ez,
    const float* __restrict__ weighted_cos,
    const float* __restrict__ weighted_sin,
    float* __restrict__ ex_real,
    float* __restrict__ ex_imag,
    float* __restrict__ ey_real,
    float* __restrict__ ey_imag,
    float* __restrict__ ez_real,
    float* __restrict__ ez_imag) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t freq = static_cast<int64_t>(blockIdx.y);
  const float cos_value = weighted_cos[freq];
  const float sin_value = weighted_sin[freq];
  if (index < ex_numel) {
    const int64_t linear = freq * ex_numel + index;
    const float value = ex[index];
    ex_real[linear] += value * cos_value;
    ex_imag[linear] += value * sin_value;
  }
  if (index < ey_numel) {
    const int64_t linear = freq * ey_numel + index;
    const float value = ey[index];
    ey_real[linear] += value * cos_value;
    ey_imag[linear] += value * sin_value;
  }
  if (index < ez_numel) {
    const int64_t linear = freq * ez_numel + index;
    const float value = ez[index];
    ez_real[linear] += value * cos_value;
    ez_imag[linear] += value * sin_value;
  }
}

__global__ void accumulate_dft_single_frequency_kernel(
    int64_t ex_numel,
    int64_t ey_numel,
    int64_t ez_numel,
    const float* __restrict__ ex,
    const float* __restrict__ ey,
    const float* __restrict__ ez,
    const float* __restrict__ weighted_cos,
    const float* __restrict__ weighted_sin,
    float* __restrict__ ex_real,
    float* __restrict__ ex_imag,
    float* __restrict__ ey_real,
    float* __restrict__ ey_imag,
    float* __restrict__ ez_real,
    float* __restrict__ ez_imag) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const float cos_value = weighted_cos[0];
  const float sin_value = weighted_sin[0];
  if (index < ex_numel) {
    const float value = ex[index];
    ex_real[index] += value * cos_value;
    ex_imag[index] += value * sin_value;
  }
  if (index < ey_numel) {
    const float value = ey[index];
    ey_real[index] += value * cos_value;
    ey_imag[index] += value * sin_value;
  }
  if (index < ez_numel) {
    const float value = ez[index];
    ez_real[index] += value * cos_value;
    ez_imag[index] += value * sin_value;
  }
}

__global__ void accumulate_dft_frequency_coarsened_kernel(
    int64_t max_field_numel,
    int64_t frequency_count,
    int64_t ex_numel,
    int64_t ey_numel,
    int64_t ez_numel,
    const float* __restrict__ ex,
    const float* __restrict__ ey,
    const float* __restrict__ ez,
    const float* __restrict__ weighted_cos,
    const float* __restrict__ weighted_sin,
    float* __restrict__ ex_real,
    float* __restrict__ ex_imag,
    float* __restrict__ ey_real,
    float* __restrict__ ey_imag,
    float* __restrict__ ez_real,
    float* __restrict__ ez_imag) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= max_field_numel) {
    return;
  }
  if (index < ex_numel) {
    const float value = ex[index];
    for (int64_t freq = 0; freq < frequency_count; ++freq) {
      const int64_t linear = freq * ex_numel + index;
      ex_real[linear] += value * weighted_cos[freq];
      ex_imag[linear] += value * weighted_sin[freq];
    }
  }
  if (index < ey_numel) {
    const float value = ey[index];
    for (int64_t freq = 0; freq < frequency_count; ++freq) {
      const int64_t linear = freq * ey_numel + index;
      ey_real[linear] += value * weighted_cos[freq];
      ey_imag[linear] += value * weighted_sin[freq];
    }
  }
  if (index < ez_numel) {
    const float value = ez[index];
    for (int64_t freq = 0; freq < frequency_count; ++freq) {
      const int64_t linear = freq * ez_numel + index;
      ez_real[linear] += value * weighted_cos[freq];
      ez_imag[linear] += value * weighted_sin[freq];
    }
  }
}

__global__ void accumulate_dft_component_kernel(
    int64_t field_numel,
    const float* __restrict__ field,
    const float* __restrict__ weighted_cos,
    const float* __restrict__ weighted_sin,
    float* __restrict__ real_accum,
    float* __restrict__ imag_accum) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= field_numel) {
    return;
  }
  const int64_t freq = static_cast<int64_t>(blockIdx.y);
  const int64_t linear = freq * field_numel + index;
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
  check_same_cuda_device(field, real_accum, "real_accum");
  check_same_cuda_device(field, imag_accum, "imag_accum");
  TORCH_CHECK(real_accum.dim() == field.dim() + 1, "real_accum rank must be field rank + 1");
  TORCH_CHECK(imag_accum.sizes() == real_accum.sizes(), "imag_accum must match real_accum shape");
  TORCH_CHECK(real_accum.size(0) == frequency_count, "accum frequency dimension mismatch");
  for (int64_t dim = 0; dim < field.dim(); ++dim) {
    TORCH_CHECK(real_accum.size(dim + 1) == field.size(dim), "accum spatial shape must match field");
  }
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
  check_same_cuda_device(ex, ey, "ey");
  check_same_cuda_device(ex, ez, "ez");
  check_same_cuda_device(ex, weighted_cos, "weighted_cos");
  check_same_cuda_device(ex, weighted_sin, "weighted_sin");
  check_same_cuda_device(ex, ey_real, "ey_real");
  check_same_cuda_device(ex, ey_imag, "ey_imag");
  check_same_cuda_device(ex, ez_real, "ez_real");
  check_same_cuda_device(ex, ez_imag, "ez_imag");
  c10::cuda::CUDAGuard guard(ex.device());
  const int64_t max_field_numel = std::max(ex.numel(), std::max(ey.numel(), ez.numel()));
  constexpr int block_size = 256;
  if (frequency_count == 1) {
    accumulate_dft_single_frequency_kernel<<<linear_grid(max_field_numel, block_size), block_size, 0, current_cuda_stream()>>>(
        ex.numel(),
        ey.numel(),
        ez.numel(),
        ex.data_ptr<float>(),
        ey.data_ptr<float>(),
        ez.data_ptr<float>(),
        weighted_cos.data_ptr<float>(),
        weighted_sin.data_ptr<float>(),
        ex_real.data_ptr<float>(),
        ex_imag.data_ptr<float>(),
        ey_real.data_ptr<float>(),
        ey_imag.data_ptr<float>(),
        ez_real.data_ptr<float>(),
        ez_imag.data_ptr<float>());
  } else if (
      (frequency_count == 4 && max_field_numel < 1000000) ||
      (frequency_count == 3 && max_field_numel >= 400000 && max_field_numel < 1000000)) {
    accumulate_dft_frequency_coarsened_kernel<<<linear_grid(max_field_numel, block_size), block_size, 0, current_cuda_stream()>>>(
        max_field_numel,
        frequency_count,
        ex.numel(),
        ey.numel(),
        ez.numel(),
        ex.data_ptr<float>(),
        ey.data_ptr<float>(),
        ez.data_ptr<float>(),
        weighted_cos.data_ptr<float>(),
        weighted_sin.data_ptr<float>(),
        ex_real.data_ptr<float>(),
        ex_imag.data_ptr<float>(),
        ey_real.data_ptr<float>(),
        ey_imag.data_ptr<float>(),
        ez_real.data_ptr<float>(),
        ez_imag.data_ptr<float>());
  } else if (frequency_count >= 4 && max_field_numel >= 1000000) {
    const dim3 ex_grid(
        static_cast<unsigned int>((ex.numel() + block_size - 1) / block_size),
        static_cast<unsigned int>(frequency_count),
        1);
    const dim3 ey_grid(
        static_cast<unsigned int>((ey.numel() + block_size - 1) / block_size),
        static_cast<unsigned int>(frequency_count),
        1);
    const dim3 ez_grid(
        static_cast<unsigned int>((ez.numel() + block_size - 1) / block_size),
        static_cast<unsigned int>(frequency_count),
        1);
    accumulate_dft_component_kernel<<<ex_grid, block_size, 0, current_cuda_stream()>>>(
        ex.numel(),
        ex.data_ptr<float>(),
        weighted_cos.data_ptr<float>(),
        weighted_sin.data_ptr<float>(),
        ex_real.data_ptr<float>(),
        ex_imag.data_ptr<float>());
    accumulate_dft_component_kernel<<<ey_grid, block_size, 0, current_cuda_stream()>>>(
        ey.numel(),
        ey.data_ptr<float>(),
        weighted_cos.data_ptr<float>(),
        weighted_sin.data_ptr<float>(),
        ey_real.data_ptr<float>(),
        ey_imag.data_ptr<float>());
    accumulate_dft_component_kernel<<<ez_grid, block_size, 0, current_cuda_stream()>>>(
        ez.numel(),
        ez.data_ptr<float>(),
        weighted_cos.data_ptr<float>(),
        weighted_sin.data_ptr<float>(),
        ez_real.data_ptr<float>(),
        ez_imag.data_ptr<float>());
  } else {
    const dim3 grid(
        static_cast<unsigned int>((max_field_numel + block_size - 1) / block_size),
        static_cast<unsigned int>(frequency_count),
        1);
    accumulate_dft_batched_kernel<<<grid, block_size, 0, current_cuda_stream()>>>(
        ex.numel(),
        ey.numel(),
        ez.numel(),
        ex.data_ptr<float>(),
        ey.data_ptr<float>(),
        ez.data_ptr<float>(),
        weighted_cos.data_ptr<float>(),
        weighted_sin.data_ptr<float>(),
        ex_real.data_ptr<float>(),
        ex_imag.data_ptr<float>(),
        ey_real.data_ptr<float>(),
        ey_imag.data_ptr<float>(),
        ez_real.data_ptr<float>(),
        ez_imag.data_ptr<float>());
  }
  WITWIN_CUDA_CHECK();
}
