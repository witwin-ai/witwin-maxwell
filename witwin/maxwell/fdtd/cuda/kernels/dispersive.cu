#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

namespace {

__global__ void update_debye_kernel(
    int64_t total,
    const float* electric,
    const float* drive,
    double decay,
    double dt,
    float* polarization,
    float* current) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const float previous = polarization[index];
  const float next = static_cast<float>(decay) * previous + drive[index] * electric[index];
  polarization[index] = next;
  current[index] = (next - previous) / static_cast<float>(dt);
}

__global__ void update_drude_kernel(
    int64_t total,
    const float* electric,
    const float* drive,
    double decay,
    float* current) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  current[index] = static_cast<float>(decay) * current[index] + drive[index] * electric[index];
}

__global__ void update_lorentz_kernel(
    int64_t total,
    const float* electric,
    const float* drive,
    double decay,
    double restoring,
    double dt,
    float* polarization,
    float* current) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const float next_current =
      static_cast<float>(decay) * current[index] -
      static_cast<float>(restoring) * polarization[index] +
      drive[index] * electric[index];
  current[index] = next_current;
  polarization[index] += static_cast<float>(dt) * next_current;
}

__global__ void apply_polarization_kernel(
    int64_t total,
    const float* current,
    const float* inv_permittivity,
    double dt,
    float* electric) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  electric[index] -= static_cast<float>(dt) * current[index] * inv_permittivity[index];
}

__device__ inline int clamp_index(int index, int size) {
  if (index <= 0) {
    return 0;
  }
  const int max_index = size - 1;
  if (index > max_index) {
    return max_index;
  }
  return index;
}

__device__ inline float sample_clamped(
    const float* field,
    int i,
    int j,
    int k,
    int size_x,
    int size_y,
    int size_z) {
  return field[offset3d(
      static_cast<unsigned int>(clamp_index(i, size_x)),
      static_cast<unsigned int>(clamp_index(j, size_y)),
      static_cast<unsigned int>(clamp_index(k, size_z)),
      static_cast<unsigned int>(size_y),
      static_cast<unsigned int>(size_z))];
}

__global__ void update_kerr_curl_kernel(
    int64_t total,
    int component,
    int dynamic_y,
    int dynamic_z,
    int ex_x,
    int ex_y,
    int ex_z,
    int ey_x,
    int ey_y,
    int ey_z,
    int ez_x,
    int ez_y,
    int ez_z,
    const float* ex,
    const float* ey,
    const float* ez,
    const float* linear_permittivity,
    const float* decay,
    const float* chi3,
    double dt,
    double eps0,
    float* dynamic_curl) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }

  const Index3D index = unflatten3d(
      static_cast<unsigned int>(linear),
      static_cast<unsigned int>(dynamic_y),
      static_cast<unsigned int>(dynamic_z));
  const int i = static_cast<int>(index.i);
  const int j = static_cast<int>(index.j);
  const int k = static_cast<int>(index.k);

  float ex_value;
  float ey_value;
  float ez_value;
  if (component == 0) {
    ex_value = ex[linear];
    ey_value = 0.25f * (
        sample_clamped(ey, i, j - 1, k, ey_x, ey_y, ey_z) +
        sample_clamped(ey, i, j, k, ey_x, ey_y, ey_z) +
        sample_clamped(ey, i + 1, j - 1, k, ey_x, ey_y, ey_z) +
        sample_clamped(ey, i + 1, j, k, ey_x, ey_y, ey_z));
    ez_value = 0.25f * (
        sample_clamped(ez, i, j, k - 1, ez_x, ez_y, ez_z) +
        sample_clamped(ez, i, j, k, ez_x, ez_y, ez_z) +
        sample_clamped(ez, i + 1, j, k - 1, ez_x, ez_y, ez_z) +
        sample_clamped(ez, i + 1, j, k, ez_x, ez_y, ez_z));
  } else if (component == 1) {
    ex_value = 0.25f * (
        sample_clamped(ex, i - 1, j, k, ex_x, ex_y, ex_z) +
        sample_clamped(ex, i, j, k, ex_x, ex_y, ex_z) +
        sample_clamped(ex, i - 1, j + 1, k, ex_x, ex_y, ex_z) +
        sample_clamped(ex, i, j + 1, k, ex_x, ex_y, ex_z));
    ey_value = ey[linear];
    ez_value = 0.25f * (
        sample_clamped(ez, i, j, k - 1, ez_x, ez_y, ez_z) +
        sample_clamped(ez, i, j, k, ez_x, ez_y, ez_z) +
        sample_clamped(ez, i, j + 1, k - 1, ez_x, ez_y, ez_z) +
        sample_clamped(ez, i, j + 1, k, ez_x, ez_y, ez_z));
  } else {
    ex_value = 0.25f * (
        sample_clamped(ex, i - 1, j, k, ex_x, ex_y, ex_z) +
        sample_clamped(ex, i, j, k, ex_x, ex_y, ex_z) +
        sample_clamped(ex, i - 1, j, k + 1, ex_x, ex_y, ex_z) +
        sample_clamped(ex, i, j, k + 1, ex_x, ex_y, ex_z));
    ey_value = 0.25f * (
        sample_clamped(ey, i, j - 1, k, ey_x, ey_y, ey_z) +
        sample_clamped(ey, i, j, k, ey_x, ey_y, ey_z) +
        sample_clamped(ey, i, j - 1, k + 1, ey_x, ey_y, ey_z) +
        sample_clamped(ey, i, j, k + 1, ey_x, ey_y, ey_z));
    ez_value = ez[linear];
  }

  float effective =
      linear_permittivity[linear] +
      static_cast<float>(eps0) * chi3[linear] * (ex_value * ex_value + ey_value * ey_value + ez_value * ez_value);
  const float floor = 1.0e-12f * static_cast<float>(eps0);
  if (effective < floor) {
    effective = floor;
  }
  dynamic_curl[linear] = (static_cast<float>(dt) / effective) * decay[linear];
}

void check_matching_field(const at::Tensor& reference, const at::Tensor& value, const char* name) {
  check_float32_tensor(value, name);
  check_contiguous_tensor(value, name);
  TORCH_CHECK(value.sizes() == reference.sizes(), name, " must match field shape");
}

void check_field3d(const at::Tensor& value, const char* name) {
  check_float32_tensor(value, name);
  check_contiguous_tensor(value, name);
  TORCH_CHECK(value.dim() == 3, name, " must be a 3D tensor");
}

void launch_kerr_curl(
    int component,
    at::Tensor dynamic_curl,
    const at::Tensor& ex,
    const at::Tensor& ey,
    const at::Tensor& ez,
    const at::Tensor& linear_permittivity,
    const at::Tensor& decay,
    const at::Tensor& chi3,
    double dt,
    double eps0) {
  check_field3d(dynamic_curl, "dynamic_curl");
  check_field3d(ex, "ex");
  check_field3d(ey, "ey");
  check_field3d(ez, "ez");
  check_matching_field(dynamic_curl, linear_permittivity, "linear_permittivity");
  check_matching_field(dynamic_curl, decay, "decay");
  check_matching_field(dynamic_curl, chi3, "chi3");
  if (component == 0) {
    TORCH_CHECK(ex.sizes() == dynamic_curl.sizes(), "ex must match dynamic_curl shape");
  } else if (component == 1) {
    TORCH_CHECK(ey.sizes() == dynamic_curl.sizes(), "ey must match dynamic_curl shape");
  } else {
    TORCH_CHECK(ez.sizes() == dynamic_curl.sizes(), "ez must match dynamic_curl shape");
  }
  c10::cuda::CUDAGuard guard(dynamic_curl.device());
  update_kerr_curl_kernel<<<linear_grid(dynamic_curl.numel()), 256, 0, current_cuda_stream()>>>(
      dynamic_curl.numel(),
      component,
      dynamic_curl.size(1),
      dynamic_curl.size(2),
      ex.size(0),
      ex.size(1),
      ex.size(2),
      ey.size(0),
      ey.size(1),
      ey.size(2),
      ez.size(0),
      ez.size(1),
      ez.size(2),
      ex.data_ptr<float>(),
      ey.data_ptr<float>(),
      ez.data_ptr<float>(),
      linear_permittivity.data_ptr<float>(),
      decay.data_ptr<float>(),
      chi3.data_ptr<float>(),
      dt,
      eps0,
      dynamic_curl.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

}  // namespace

void update_debye_current_cuda(
    const at::Tensor& electric,
    at::Tensor polarization,
    at::Tensor current,
    const at::Tensor& drive,
    double decay,
    double dt) {
  check_float32_tensor(electric, "electric");
  check_contiguous_tensor(electric, "electric");
  check_matching_field(electric, polarization, "polarization");
  check_matching_field(electric, current, "current");
  check_matching_field(electric, drive, "drive");
  c10::cuda::CUDAGuard guard(electric.device());
  update_debye_kernel<<<linear_grid(electric.numel()), 256, 0, current_cuda_stream()>>>(
      electric.numel(),
      electric.data_ptr<float>(),
      drive.data_ptr<float>(),
      decay,
      dt,
      polarization.data_ptr<float>(),
      current.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_drude_current_cuda(
    const at::Tensor& electric,
    at::Tensor current,
    const at::Tensor& drive,
    double decay) {
  check_float32_tensor(electric, "electric");
  check_contiguous_tensor(electric, "electric");
  check_matching_field(electric, current, "current");
  check_matching_field(electric, drive, "drive");
  c10::cuda::CUDAGuard guard(electric.device());
  update_drude_kernel<<<linear_grid(electric.numel()), 256, 0, current_cuda_stream()>>>(
      electric.numel(),
      electric.data_ptr<float>(),
      drive.data_ptr<float>(),
      decay,
      current.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_lorentz_current_cuda(
    const at::Tensor& electric,
    at::Tensor polarization,
    at::Tensor current,
    const at::Tensor& drive,
    double decay,
    double restoring,
    double dt) {
  check_float32_tensor(electric, "electric");
  check_contiguous_tensor(electric, "electric");
  check_matching_field(electric, polarization, "polarization");
  check_matching_field(electric, current, "current");
  check_matching_field(electric, drive, "drive");
  c10::cuda::CUDAGuard guard(electric.device());
  update_lorentz_kernel<<<linear_grid(electric.numel()), 256, 0, current_cuda_stream()>>>(
      electric.numel(),
      electric.data_ptr<float>(),
      drive.data_ptr<float>(),
      decay,
      restoring,
      dt,
      polarization.data_ptr<float>(),
      current.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void apply_polarization_current_cuda(
    at::Tensor electric,
    const at::Tensor& current,
    const at::Tensor& inv_permittivity,
    double dt) {
  check_float32_tensor(electric, "electric");
  check_contiguous_tensor(electric, "electric");
  check_matching_field(electric, current, "current");
  check_matching_field(electric, inv_permittivity, "inv_permittivity");
  c10::cuda::CUDAGuard guard(electric.device());
  apply_polarization_kernel<<<linear_grid(electric.numel()), 256, 0, current_cuda_stream()>>>(
      electric.numel(),
      current.data_ptr<float>(),
      inv_permittivity.data_ptr<float>(),
      dt,
      electric.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_kerr_ex_curl_cuda(
    at::Tensor dynamic_curl,
    const at::Tensor& ex,
    const at::Tensor& ey,
    const at::Tensor& ez,
    const at::Tensor& linear_permittivity,
    const at::Tensor& ex_decay,
    const at::Tensor& chi3,
    double dt,
    double eps0) {
  launch_kerr_curl(0, dynamic_curl, ex, ey, ez, linear_permittivity, ex_decay, chi3, dt, eps0);
}

void update_kerr_ey_curl_cuda(
    at::Tensor dynamic_curl,
    const at::Tensor& ex,
    const at::Tensor& ey,
    const at::Tensor& ez,
    const at::Tensor& linear_permittivity,
    const at::Tensor& ey_decay,
    const at::Tensor& chi3,
    double dt,
    double eps0) {
  launch_kerr_curl(1, dynamic_curl, ex, ey, ez, linear_permittivity, ey_decay, chi3, dt, eps0);
}

void update_kerr_ez_curl_cuda(
    at::Tensor dynamic_curl,
    const at::Tensor& ex,
    const at::Tensor& ey,
    const at::Tensor& ez,
    const at::Tensor& linear_permittivity,
    const at::Tensor& ez_decay,
    const at::Tensor& chi3,
    double dt,
    double eps0) {
  launch_kerr_curl(2, dynamic_curl, ex, ey, ez, linear_permittivity, ez_decay, chi3, dt, eps0);
}
