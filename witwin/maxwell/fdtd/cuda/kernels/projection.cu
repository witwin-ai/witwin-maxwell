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

template <int Axis>
__device__ __forceinline__ void face_indices(
    int coord_a,
    int coord_b,
    int nx,
    int ny,
    int nz,
    long long& low,
    long long& high) {
  if constexpr (Axis == 0) {
    const int j = coord_a;
    const int k = coord_b;
    low = offset3d(0, j, k, ny, nz);
    high = low + static_cast<long long>(nx - 1) * ny * nz;
    return;
  }
  if constexpr (Axis == 1) {
    const int i = coord_a;
    const int k = coord_b;
    low = offset3d(i, 0, k, ny, nz);
    high = low + static_cast<long long>(ny - 1) * nz;
    return;
  }
  const int i = coord_a;
  const int j = coord_b;
  low = offset3d(i, j, 0, ny, nz);
  high = low + nz - 1;
}

template <int Axis>
__global__ void project_periodic_boundary_kernel(
    int nx,
    int ny,
    int nz,
    float* __restrict__ field) {
  const int coord_b = blockIdx.x * blockDim.x + threadIdx.x;
  const int coord_a = blockIdx.y * blockDim.y + threadIdx.y;
  const int dim_b = Axis == 2 ? ny : nz;
  const int dim_a = Axis == 0 ? ny : nx;
  if (coord_a >= dim_a || coord_b >= dim_b) {
    return;
  }
  long long low = 0;
  long long high = 0;
  face_indices<Axis>(coord_a, coord_b, nx, ny, nz, low, high);
  const float average = 0.5f * (field[low] + field[high]);
  field[low] = average;
  field[high] = average;
}

template <int Axis, bool RealPhase>
__global__ void project_bloch_boundary_kernel(
    int nx,
    int ny,
    int nz,
    float phase_cos,
    float phase_sin,
    float* __restrict__ real_field,
    float* __restrict__ imag_field) {
  const int coord_b = blockIdx.x * blockDim.x + threadIdx.x;
  const int coord_a = blockIdx.y * blockDim.y + threadIdx.y;
  const int dim_b = Axis == 2 ? ny : nz;
  const int dim_a = Axis == 0 ? ny : nx;
  if (coord_a >= dim_a || coord_b >= dim_b) {
    return;
  }
  long long low = 0;
  long long high = 0;
  face_indices<Axis>(coord_a, coord_b, nx, ny, nz, low, high);

  const float low_r = real_field[low];
  const float low_i = imag_field[low];
  const float high_r = real_field[high];
  const float high_i = imag_field[high];
  const float c = phase_cos;

  float projected_low_r;
  float projected_low_i;
  float projected_high_r;
  float projected_high_i;
  if constexpr (RealPhase) {
    projected_low_r = 0.5f * (low_r + c * high_r);
    projected_low_i = 0.5f * (low_i + c * high_i);
    projected_high_r = c * projected_low_r;
    projected_high_i = c * projected_low_i;
  } else {
    const float s = phase_sin;
    projected_low_r = 0.5f * (low_r + c * high_r + s * high_i);
    projected_low_i = 0.5f * (low_i + c * high_i - s * high_r);
    projected_high_r = c * projected_low_r - s * projected_low_i;
    projected_high_i = s * projected_low_r + c * projected_low_i;
  }

  real_field[low] = projected_low_r;
  imag_field[low] = projected_low_i;
  real_field[high] = projected_high_r;
  imag_field[high] = projected_high_i;
}

template <int Axis>
void launch_periodic_projection(int nx, int ny, int nz, float* __restrict__ field) {
  const dim3 block(16, 16, 1);
  const unsigned int dim_b = Axis == 2 ? static_cast<unsigned int>(ny) : static_cast<unsigned int>(nz);
  const unsigned int dim_a = Axis == 0 ? static_cast<unsigned int>(ny) : static_cast<unsigned int>(nx);
  const dim3 grid((dim_b + block.x - 1) / block.x, (dim_a + block.y - 1) / block.y, 1);
  project_periodic_boundary_kernel<Axis><<<grid, block, 0, current_cuda_stream()>>>(
      nx,
      ny,
      nz,
      field);
}

template <int Axis, bool RealPhase>
void launch_bloch_projection(
    int nx,
    int ny,
    int nz,
    float phase_cos,
    float phase_sin,
    float* __restrict__ field_real,
    float* __restrict__ field_imag) {
  const dim3 block(16, 16, 1);
  const unsigned int dim_b = Axis == 2 ? static_cast<unsigned int>(ny) : static_cast<unsigned int>(nz);
  const unsigned int dim_a = Axis == 0 ? static_cast<unsigned int>(ny) : static_cast<unsigned int>(nx);
  const dim3 grid((dim_b + block.x - 1) / block.x, (dim_a + block.y - 1) / block.y, 1);
  project_bloch_boundary_kernel<Axis, RealPhase><<<grid, block, 0, current_cuda_stream()>>>(
      nx,
      ny,
      nz,
      phase_cos,
      phase_sin,
      field_real,
      field_imag);
}

}  // namespace

void project_periodic_boundary_cuda(at::Tensor field, int64_t axis) {
  check_field3d(field, "field");
  TORCH_CHECK(axis >= 0 && axis < 3, "axis must be in [0, 3)");
  const c10::cuda::CUDAGuard device_guard(field.device());
  const int nx = static_cast<int>(field.size(0));
  const int ny = static_cast<int>(field.size(1));
  const int nz = static_cast<int>(field.size(2));
  if (axis == 0) {
    launch_periodic_projection<0>(nx, ny, nz, field.data_ptr<float>());
  } else if (axis == 1) {
    launch_periodic_projection<1>(nx, ny, nz, field.data_ptr<float>());
  } else {
    launch_periodic_projection<2>(nx, ny, nz, field.data_ptr<float>());
  }
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
  check_same_cuda_device(field_real, field_imag, "field_imag");
  TORCH_CHECK(field_real.sizes() == field_imag.sizes(), "field_imag must match field_real shape");
  TORCH_CHECK(axis >= 0 && axis < 3, "axis must be in [0, 3)");
  const c10::cuda::CUDAGuard device_guard(field_real.device());
  const int nx = static_cast<int>(field_real.size(0));
  const int ny = static_cast<int>(field_real.size(1));
  const int nz = static_cast<int>(field_real.size(2));
  const float phase_cos_f = static_cast<float>(phase_cos);
  const float phase_sin_f = static_cast<float>(phase_sin);
  const bool real_phase = phase_sin_f == 0.0f;
  if (axis == 0) {
    if (real_phase) {
      launch_bloch_projection<0, true>(
          nx, ny, nz, phase_cos_f, phase_sin_f, field_real.data_ptr<float>(), field_imag.data_ptr<float>());
    } else {
      launch_bloch_projection<0, false>(
          nx, ny, nz, phase_cos_f, phase_sin_f, field_real.data_ptr<float>(), field_imag.data_ptr<float>());
    }
  } else if (axis == 1) {
    if (real_phase) {
      launch_bloch_projection<1, true>(
          nx, ny, nz, phase_cos_f, phase_sin_f, field_real.data_ptr<float>(), field_imag.data_ptr<float>());
    } else {
      launch_bloch_projection<1, false>(
          nx, ny, nz, phase_cos_f, phase_sin_f, field_real.data_ptr<float>(), field_imag.data_ptr<float>());
    }
  } else {
    if (real_phase) {
      launch_bloch_projection<2, true>(
          nx, ny, nz, phase_cos_f, phase_sin_f, field_real.data_ptr<float>(), field_imag.data_ptr<float>());
    } else {
      launch_bloch_projection<2, false>(
          nx, ny, nz, phase_cos_f, phase_sin_f, field_real.data_ptr<float>(), field_imag.data_ptr<float>());
    }
  }
  WITWIN_CUDA_CHECK();
}
