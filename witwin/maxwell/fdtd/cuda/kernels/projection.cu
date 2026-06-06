#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

namespace {

void check_field3d(const at::Tensor& field, const char* name) {
  check_float32_tensor(field, name);
  check_contiguous_tensor(field, name);
  TORCH_CHECK(field.dim() == 3, name, " must be a contiguous 3D float32 CUDA tensor");
}

__device__ inline void face_indices(
    int64_t linear,
    int axis,
    int nx,
    int ny,
    int nz,
    int& low,
    int& high) {
  if (axis == 0) {
    const int j = static_cast<int>(linear / nz);
    const int k = static_cast<int>(linear - static_cast<int64_t>(j) * nz);
    low = static_cast<int>(offset3d(0, j, k, ny, nz));
    high = static_cast<int>(offset3d(nx - 1, j, k, ny, nz));
    return;
  }
  if (axis == 1) {
    const int i = static_cast<int>(linear / nz);
    const int k = static_cast<int>(linear - static_cast<int64_t>(i) * nz);
    low = static_cast<int>(offset3d(i, 0, k, ny, nz));
    high = static_cast<int>(offset3d(i, ny - 1, k, ny, nz));
    return;
  }
  const int i = static_cast<int>(linear / ny);
  const int j = static_cast<int>(linear - static_cast<int64_t>(i) * ny);
  low = static_cast<int>(offset3d(i, j, 0, ny, nz));
  high = static_cast<int>(offset3d(i, j, nz - 1, ny, nz));
}

__global__ void project_periodic_boundary_kernel(
    int64_t face_total,
    int axis,
    int nx,
    int ny,
    int nz,
    float* field) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= face_total) {
    return;
  }
  int low = 0;
  int high = 0;
  face_indices(linear, axis, nx, ny, nz, low, high);
  const float average = 0.5f * (field[low] + field[high]);
  field[low] = average;
  field[high] = average;
}

__global__ void project_bloch_boundary_kernel(
    int64_t face_total,
    int axis,
    int nx,
    int ny,
    int nz,
    double phase_cos,
    double phase_sin,
    float* real_field,
    float* imag_field) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= face_total) {
    return;
  }
  int low = 0;
  int high = 0;
  face_indices(linear, axis, nx, ny, nz, low, high);

  const float low_r = real_field[low];
  const float low_i = imag_field[low];
  const float high_r = real_field[high];
  const float high_i = imag_field[high];
  const float c = static_cast<float>(phase_cos);
  const float s = static_cast<float>(phase_sin);

  const float projected_low_r = 0.5f * (low_r + c * high_r + s * high_i);
  const float projected_low_i = 0.5f * (low_i + c * high_i - s * high_r);
  const float projected_high_r = c * projected_low_r - s * projected_low_i;
  const float projected_high_i = s * projected_low_r + c * projected_low_i;

  real_field[low] = projected_low_r;
  imag_field[low] = projected_low_i;
  real_field[high] = projected_high_r;
  imag_field[high] = projected_high_i;
}

int64_t face_elements(const at::Tensor& field, int64_t axis) {
  if (axis == 0) {
    return field.size(1) * field.size(2);
  }
  if (axis == 1) {
    return field.size(0) * field.size(2);
  }
  return field.size(0) * field.size(1);
}

}  // namespace

void project_periodic_boundary_cuda(at::Tensor field, int64_t axis) {
  check_field3d(field, "field");
  TORCH_CHECK(axis >= 0 && axis < 3, "axis must be in [0, 3)");
  const c10::cuda::CUDAGuard device_guard(field.device());
  const int64_t total = face_elements(field, axis);
  project_periodic_boundary_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(axis),
      static_cast<int>(field.size(0)),
      static_cast<int>(field.size(1)),
      static_cast<int>(field.size(2)),
      field.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void project_bloch_boundary_cuda(
    at::Tensor field_real,
    at::Tensor field_imag,
    int64_t axis,
    double phase_cos,
    double phase_sin) {
  check_field3d(field_real, "field_real");
  check_field3d(field_imag, "field_imag");
  TORCH_CHECK(field_real.sizes() == field_imag.sizes(), "field_imag must match field_real shape");
  TORCH_CHECK(axis >= 0 && axis < 3, "axis must be in [0, 3)");
  const c10::cuda::CUDAGuard device_guard(field_real.device());
  const int64_t total = face_elements(field_real, axis);
  project_bloch_boundary_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(axis),
      static_cast<int>(field_real.size(0)),
      static_cast<int>(field_real.size(1)),
      static_cast<int>(field_real.size(2)),
      phase_cos,
      phase_sin,
      field_real.data_ptr<float>(),
      field_imag.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}
