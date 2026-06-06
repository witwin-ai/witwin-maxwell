#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

namespace {

__global__ void update_magnetic_hx_standard_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* ey,
    const float* ez,
    const float* decay,
    const float* curl_coeff,
    float inv_dy,
    float inv_dz,
    float* hx) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const long long ez_hi = offset3d(coord.i, coord.j + 1, coord.k, ny + 1, nz);
  const long long ez_lo = offset3d(coord.i, coord.j, coord.k, ny + 1, nz);
  const long long ey_hi = offset3d(coord.i, coord.j, coord.k + 1, ny, nz + 1);
  const long long ey_lo = offset3d(coord.i, coord.j, coord.k, ny, nz + 1);
  const float curl = (ez[ez_hi] - ez[ez_lo]) * inv_dy - (ey[ey_hi] - ey[ey_lo]) * inv_dz;
  hx[linear] = hx[linear] * decay[linear] - curl_coeff[linear] * curl;
}

__global__ void update_magnetic_hy_standard_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* ex,
    const float* ez,
    const float* decay,
    const float* curl_coeff,
    float inv_dx,
    float inv_dz,
    float* hy) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const long long ex_hi = offset3d(coord.i, coord.j, coord.k + 1, ny, nz + 1);
  const long long ex_lo = offset3d(coord.i, coord.j, coord.k, ny, nz + 1);
  const long long ez_hi = offset3d(coord.i + 1, coord.j, coord.k, ny, nz);
  const long long ez_lo = offset3d(coord.i, coord.j, coord.k, ny, nz);
  const float curl = (ex[ex_hi] - ex[ex_lo]) * inv_dz - (ez[ez_hi] - ez[ez_lo]) * inv_dx;
  hy[linear] = hy[linear] * decay[linear] - curl_coeff[linear] * curl;
}

__global__ void update_magnetic_hz_standard_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* ex,
    const float* ey,
    const float* decay,
    const float* curl_coeff,
    float inv_dx,
    float inv_dy,
    float* hz) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const long long ey_hi = offset3d(coord.i + 1, coord.j, coord.k, ny, nz);
  const long long ey_lo = offset3d(coord.i, coord.j, coord.k, ny, nz);
  const long long ex_hi = offset3d(coord.i, coord.j + 1, coord.k, ny + 1, nz);
  const long long ex_lo = offset3d(coord.i, coord.j, coord.k, ny + 1, nz);
  const float curl = (ey[ey_hi] - ey[ey_lo]) * inv_dx - (ex[ex_hi] - ex[ex_lo]) * inv_dy;
  hz[linear] = hz[linear] * decay[linear] - curl_coeff[linear] * curl;
}

__global__ void update_magnetic_hx_cpml_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* ey,
    const float* ez,
    const float* decay,
    const float* curl_coeff,
    const float* inv_kappa_y,
    const float* b_y,
    const float* c_y,
    const float* inv_kappa_z,
    const float* b_z,
    const float* c_z,
    float inv_dy,
    float inv_dz,
    float* psi_y,
    float* psi_z,
    float* hx) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const long long ez_hi = offset3d(coord.i, coord.j + 1, coord.k, ny + 1, nz);
  const long long ez_lo = offset3d(coord.i, coord.j, coord.k, ny + 1, nz);
  const long long ey_hi = offset3d(coord.i, coord.j, coord.k + 1, ny, nz + 1);
  const long long ey_lo = offset3d(coord.i, coord.j, coord.k, ny, nz + 1);
  const float d_y = (ez[ez_hi] - ez[ez_lo]) * inv_dy;
  const float d_z = (ey[ey_hi] - ey[ey_lo]) * inv_dz;
  const float psi_y_value = b_y[coord.j] * psi_y[linear] + c_y[coord.j] * d_y;
  const float psi_z_value = b_z[coord.k] * psi_z[linear] + c_z[coord.k] * d_z;
  psi_y[linear] = psi_y_value;
  psi_z[linear] = psi_z_value;
  const float curl = d_y * inv_kappa_y[coord.j] + psi_y_value - d_z * inv_kappa_z[coord.k] - psi_z_value;
  hx[linear] = hx[linear] * decay[linear] - curl_coeff[linear] * curl;
}

__global__ void update_magnetic_hy_cpml_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* ex,
    const float* ez,
    const float* decay,
    const float* curl_coeff,
    const float* inv_kappa_x,
    const float* b_x,
    const float* c_x,
    const float* inv_kappa_z,
    const float* b_z,
    const float* c_z,
    float inv_dx,
    float inv_dz,
    float* psi_x,
    float* psi_z,
    float* hy) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const long long ex_hi = offset3d(coord.i, coord.j, coord.k + 1, ny, nz + 1);
  const long long ex_lo = offset3d(coord.i, coord.j, coord.k, ny, nz + 1);
  const long long ez_hi = offset3d(coord.i + 1, coord.j, coord.k, ny, nz);
  const long long ez_lo = offset3d(coord.i, coord.j, coord.k, ny, nz);
  const float d_z = (ex[ex_hi] - ex[ex_lo]) * inv_dz;
  const float d_x = (ez[ez_hi] - ez[ez_lo]) * inv_dx;
  const float psi_x_value = b_x[coord.i] * psi_x[linear] + c_x[coord.i] * d_x;
  const float psi_z_value = b_z[coord.k] * psi_z[linear] + c_z[coord.k] * d_z;
  psi_x[linear] = psi_x_value;
  psi_z[linear] = psi_z_value;
  const float curl = d_z * inv_kappa_z[coord.k] + psi_z_value - d_x * inv_kappa_x[coord.i] - psi_x_value;
  hy[linear] = hy[linear] * decay[linear] - curl_coeff[linear] * curl;
}

__global__ void update_magnetic_hz_cpml_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* ex,
    const float* ey,
    const float* decay,
    const float* curl_coeff,
    const float* inv_kappa_x,
    const float* b_x,
    const float* c_x,
    const float* inv_kappa_y,
    const float* b_y,
    const float* c_y,
    float inv_dx,
    float inv_dy,
    float* psi_x,
    float* psi_y,
    float* hz) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const long long ey_hi = offset3d(coord.i + 1, coord.j, coord.k, ny, nz);
  const long long ey_lo = offset3d(coord.i, coord.j, coord.k, ny, nz);
  const long long ex_hi = offset3d(coord.i, coord.j + 1, coord.k, ny + 1, nz);
  const long long ex_lo = offset3d(coord.i, coord.j, coord.k, ny + 1, nz);
  const float d_x = (ey[ey_hi] - ey[ey_lo]) * inv_dx;
  const float d_y = (ex[ex_hi] - ex[ex_lo]) * inv_dy;
  const float psi_x_value = b_x[coord.i] * psi_x[linear] + c_x[coord.i] * d_x;
  const float psi_y_value = b_y[coord.j] * psi_y[linear] + c_y[coord.j] * d_y;
  psi_x[linear] = psi_x_value;
  psi_y[linear] = psi_y_value;
  const float curl = d_x * inv_kappa_x[coord.i] + psi_x_value - d_y * inv_kappa_y[coord.j] - psi_y_value;
  hz[linear] = hz[linear] * decay[linear] - curl_coeff[linear] * curl;
}

__device__ inline float update_compact_magnetic_psi(
    float* psi,
    const float* b,
    const float* c,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    unsigned int size_y,
    unsigned int size_z,
    int axis,
    unsigned int axis_coord,
    int low_length,
    int high_start,
    int high_length,
    float derivative) {
  const int local = compact_local_index(axis_coord, low_length, high_start, high_length);
  if (local < 0) {
    return 0.0f;
  }
  const unsigned int compact_length = static_cast<unsigned int>(low_length + high_length);
  const long long psi_offset = compact_offset3d(axis, i, j, k, size_y, size_z, local, compact_length);
  const float value = b[axis_coord] * psi[psi_offset] + c[axis_coord] * derivative;
  psi[psi_offset] = value;
  return value;
}

__global__ void update_magnetic_hx_cpml_compressed_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* ey,
    const float* ez,
    const float* decay,
    const float* curl_coeff,
    const float* inv_kappa_y,
    const float* b_y,
    const float* c_y,
    const float* inv_kappa_z,
    const float* b_z,
    const float* c_z,
    float inv_dy,
    float inv_dz,
    int y_low_length,
    int y_high_start,
    int y_high_length,
    int z_low_length,
    int z_high_start,
    int z_high_length,
    float* psi_y,
    float* psi_z,
    float* hx) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const long long ez_hi = offset3d(coord.i, coord.j + 1, coord.k, ny + 1, nz);
  const long long ez_lo = offset3d(coord.i, coord.j, coord.k, ny + 1, nz);
  const long long ey_hi = offset3d(coord.i, coord.j, coord.k + 1, ny, nz + 1);
  const long long ey_lo = offset3d(coord.i, coord.j, coord.k, ny, nz + 1);
  const float d_y = (ez[ez_hi] - ez[ez_lo]) * inv_dy;
  const float d_z = (ey[ey_hi] - ey[ey_lo]) * inv_dz;
  const float psi_y_value = update_compact_magnetic_psi(
      psi_y, b_y, c_y, coord.i, coord.j, coord.k, ny, nz, 1, coord.j, y_low_length, y_high_start, y_high_length, d_y);
  const float psi_z_value = update_compact_magnetic_psi(
      psi_z, b_z, c_z, coord.i, coord.j, coord.k, ny, nz, 2, coord.k, z_low_length, z_high_start, z_high_length, d_z);
  const float curl = d_y * inv_kappa_y[coord.j] + psi_y_value - d_z * inv_kappa_z[coord.k] - psi_z_value;
  hx[linear] = hx[linear] * decay[linear] - curl_coeff[linear] * curl;
}

__global__ void update_magnetic_hy_cpml_compressed_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* ex,
    const float* ez,
    const float* decay,
    const float* curl_coeff,
    const float* inv_kappa_x,
    const float* b_x,
    const float* c_x,
    const float* inv_kappa_z,
    const float* b_z,
    const float* c_z,
    float inv_dx,
    float inv_dz,
    int x_low_length,
    int x_high_start,
    int x_high_length,
    int z_low_length,
    int z_high_start,
    int z_high_length,
    float* psi_x,
    float* psi_z,
    float* hy) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const long long ex_hi = offset3d(coord.i, coord.j, coord.k + 1, ny, nz + 1);
  const long long ex_lo = offset3d(coord.i, coord.j, coord.k, ny, nz + 1);
  const long long ez_hi = offset3d(coord.i + 1, coord.j, coord.k, ny, nz);
  const long long ez_lo = offset3d(coord.i, coord.j, coord.k, ny, nz);
  const float d_z = (ex[ex_hi] - ex[ex_lo]) * inv_dz;
  const float d_x = (ez[ez_hi] - ez[ez_lo]) * inv_dx;
  const float psi_x_value = update_compact_magnetic_psi(
      psi_x, b_x, c_x, coord.i, coord.j, coord.k, ny, nz, 0, coord.i, x_low_length, x_high_start, x_high_length, d_x);
  const float psi_z_value = update_compact_magnetic_psi(
      psi_z, b_z, c_z, coord.i, coord.j, coord.k, ny, nz, 2, coord.k, z_low_length, z_high_start, z_high_length, d_z);
  const float curl = d_z * inv_kappa_z[coord.k] + psi_z_value - d_x * inv_kappa_x[coord.i] - psi_x_value;
  hy[linear] = hy[linear] * decay[linear] - curl_coeff[linear] * curl;
}

__global__ void update_magnetic_hz_cpml_compressed_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* ex,
    const float* ey,
    const float* decay,
    const float* curl_coeff,
    const float* inv_kappa_x,
    const float* b_x,
    const float* c_x,
    const float* inv_kappa_y,
    const float* b_y,
    const float* c_y,
    float inv_dx,
    float inv_dy,
    int x_low_length,
    int x_high_start,
    int x_high_length,
    int y_low_length,
    int y_high_start,
    int y_high_length,
    float* psi_x,
    float* psi_y,
    float* hz) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const long long ey_hi = offset3d(coord.i + 1, coord.j, coord.k, ny, nz);
  const long long ey_lo = offset3d(coord.i, coord.j, coord.k, ny, nz);
  const long long ex_hi = offset3d(coord.i, coord.j + 1, coord.k, ny + 1, nz);
  const long long ex_lo = offset3d(coord.i, coord.j, coord.k, ny + 1, nz);
  const float d_x = (ey[ey_hi] - ey[ey_lo]) * inv_dx;
  const float d_y = (ex[ex_hi] - ex[ex_lo]) * inv_dy;
  const float psi_x_value = update_compact_magnetic_psi(
      psi_x, b_x, c_x, coord.i, coord.j, coord.k, ny, nz, 0, coord.i, x_low_length, x_high_start, x_high_length, d_x);
  const float psi_y_value = update_compact_magnetic_psi(
      psi_y, b_y, c_y, coord.i, coord.j, coord.k, ny, nz, 1, coord.j, y_low_length, y_high_start, y_high_length, d_y);
  const float curl = d_x * inv_kappa_x[coord.i] + psi_x_value - d_y * inv_kappa_y[coord.j] - psi_y_value;
  hz[linear] = hz[linear] * decay[linear] - curl_coeff[linear] * curl;
}

void check_magnetic_inputs(
    const at::Tensor& field,
    const at::Tensor& first,
    const at::Tensor& second,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const char* name) {
  check_float32_tensor(field, name);
  check_float32_tensor(first, "first");
  check_float32_tensor(second, "second");
  check_float32_tensor(decay, "decay");
  check_float32_tensor(curl, "curl");
  check_contiguous_tensor(field, name);
  check_contiguous_tensor(first, "first");
  check_contiguous_tensor(second, "second");
  check_contiguous_tensor(decay, "decay");
  check_contiguous_tensor(curl, "curl");
  TORCH_CHECK(field.dim() == 3, name, " must be rank 3");
  TORCH_CHECK(decay.sizes() == field.sizes(), "decay must match field shape");
  TORCH_CHECK(curl.sizes() == field.sizes(), "curl must match field shape");
}

void check_vector_input(const at::Tensor& tensor, int64_t length, const char* name) {
  check_float32_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  TORCH_CHECK(tensor.dim() == 1, name, " must be rank 1");
  TORCH_CHECK(tensor.size(0) == length, name, " length must match CPML field axis");
}

void check_magnetic_cpml_inputs(
    const at::Tensor& field,
    const at::Tensor& first,
    const at::Tensor& second,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const at::Tensor& psi_first,
    const at::Tensor& psi_second,
    const at::Tensor& inv_kappa_first,
    const at::Tensor& b_first,
    const at::Tensor& c_first,
    const at::Tensor& inv_kappa_second,
    const at::Tensor& b_second,
    const at::Tensor& c_second,
    int64_t first_axis,
    int64_t second_axis,
    const char* name) {
  check_magnetic_inputs(field, first, second, decay, curl, name);
  check_float32_tensor(psi_first, "psi_first");
  check_float32_tensor(psi_second, "psi_second");
  check_contiguous_tensor(psi_first, "psi_first");
  check_contiguous_tensor(psi_second, "psi_second");
  TORCH_CHECK(psi_first.sizes() == field.sizes(), "psi_first must match field shape");
  TORCH_CHECK(psi_second.sizes() == field.sizes(), "psi_second must match field shape");
  check_vector_input(inv_kappa_first, field.size(first_axis), "inv_kappa_first");
  check_vector_input(b_first, field.size(first_axis), "b_first");
  check_vector_input(c_first, field.size(first_axis), "c_first");
  check_vector_input(inv_kappa_second, field.size(second_axis), "inv_kappa_second");
  check_vector_input(b_second, field.size(second_axis), "b_second");
  check_vector_input(c_second, field.size(second_axis), "c_second");
}

void check_compact_psi_input(
    const at::Tensor& psi,
    const at::Tensor& field,
    int64_t axis,
    int64_t low_length,
    int64_t high_length,
    const char* name) {
  check_float32_tensor(psi, name);
  check_contiguous_tensor(psi, name);
  TORCH_CHECK(psi.dim() == 3, name, " must be rank 3");
  for (int64_t dim = 0; dim < 3; ++dim) {
    const int64_t expected = dim == axis ? low_length + high_length : field.size(dim);
    TORCH_CHECK(psi.size(dim) == expected, name, " shape does not match compact CPML layout");
  }
}

void check_magnetic_cpml_compressed_inputs(
    const at::Tensor& field,
    const at::Tensor& first,
    const at::Tensor& second,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const at::Tensor& psi_first,
    const at::Tensor& psi_second,
    const at::Tensor& inv_kappa_first,
    const at::Tensor& b_first,
    const at::Tensor& c_first,
    const at::Tensor& inv_kappa_second,
    const at::Tensor& b_second,
    const at::Tensor& c_second,
    int64_t first_axis,
    int64_t second_axis,
    int64_t first_low_length,
    int64_t first_high_length,
    int64_t second_low_length,
    int64_t second_high_length,
    const char* name) {
  check_magnetic_inputs(field, first, second, decay, curl, name);
  check_compact_psi_input(psi_first, field, first_axis, first_low_length, first_high_length, "psi_first");
  check_compact_psi_input(psi_second, field, second_axis, second_low_length, second_high_length, "psi_second");
  check_vector_input(inv_kappa_first, field.size(first_axis), "inv_kappa_first");
  check_vector_input(b_first, field.size(first_axis), "b_first");
  check_vector_input(c_first, field.size(first_axis), "c_first");
  check_vector_input(inv_kappa_second, field.size(second_axis), "inv_kappa_second");
  check_vector_input(b_second, field.size(second_axis), "b_second");
  check_vector_input(c_second, field.size(second_axis), "c_second");
}

}  // namespace

void update_magnetic_hx_standard_cuda(
    at::Tensor hx,
    const at::Tensor& ey,
    const at::Tensor& ez,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double inv_dy,
    double inv_dz) {
  check_magnetic_inputs(hx, ey, ez, decay, curl, "hx");
  c10::cuda::CUDAGuard guard(hx.device());
  const auto sizes = hx.sizes();
  const int64_t total = hx.numel();
  update_magnetic_hx_standard_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      ey.data_ptr<float>(),
      ez.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>(),
      static_cast<float>(inv_dy),
      static_cast<float>(inv_dz),
      hx.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_magnetic_hy_standard_cuda(
    at::Tensor hy,
    const at::Tensor& ex,
    const at::Tensor& ez,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double inv_dx,
    double inv_dz) {
  check_magnetic_inputs(hy, ex, ez, decay, curl, "hy");
  c10::cuda::CUDAGuard guard(hy.device());
  const auto sizes = hy.sizes();
  const int64_t total = hy.numel();
  update_magnetic_hy_standard_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      ex.data_ptr<float>(),
      ez.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>(),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dz),
      hy.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_magnetic_hz_standard_cuda(
    at::Tensor hz,
    const at::Tensor& ex,
    const at::Tensor& ey,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double inv_dx,
    double inv_dy) {
  check_magnetic_inputs(hz, ex, ey, decay, curl, "hz");
  c10::cuda::CUDAGuard guard(hz.device());
  const auto sizes = hz.sizes();
  const int64_t total = hz.numel();
  update_magnetic_hz_standard_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      ex.data_ptr<float>(),
      ey.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>(),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dy),
      hz.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_magnetic_hx_cpml_cuda(
    at::Tensor hx,
    const at::Tensor& ey,
    const at::Tensor& ez,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_y,
    at::Tensor psi_z,
    const at::Tensor& inv_kappa_y,
    const at::Tensor& b_y,
    const at::Tensor& c_y,
    const at::Tensor& inv_kappa_z,
    const at::Tensor& b_z,
    const at::Tensor& c_z,
    double inv_dy,
    double inv_dz) {
  check_magnetic_cpml_inputs(
      hx, ey, ez, decay, curl, psi_y, psi_z, inv_kappa_y, b_y, c_y, inv_kappa_z, b_z, c_z, 1, 2, "hx");
  c10::cuda::CUDAGuard guard(hx.device());
  const auto sizes = hx.sizes();
  const int64_t total = hx.numel();
  update_magnetic_hx_cpml_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      ey.data_ptr<float>(),
      ez.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>(),
      inv_kappa_y.data_ptr<float>(),
      b_y.data_ptr<float>(),
      c_y.data_ptr<float>(),
      inv_kappa_z.data_ptr<float>(),
      b_z.data_ptr<float>(),
      c_z.data_ptr<float>(),
      static_cast<float>(inv_dy),
      static_cast<float>(inv_dz),
      psi_y.data_ptr<float>(),
      psi_z.data_ptr<float>(),
      hx.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_magnetic_hy_cpml_cuda(
    at::Tensor hy,
    const at::Tensor& ex,
    const at::Tensor& ez,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_x,
    at::Tensor psi_z,
    const at::Tensor& inv_kappa_x,
    const at::Tensor& b_x,
    const at::Tensor& c_x,
    const at::Tensor& inv_kappa_z,
    const at::Tensor& b_z,
    const at::Tensor& c_z,
    double inv_dx,
    double inv_dz) {
  check_magnetic_cpml_inputs(
      hy, ex, ez, decay, curl, psi_x, psi_z, inv_kappa_x, b_x, c_x, inv_kappa_z, b_z, c_z, 0, 2, "hy");
  c10::cuda::CUDAGuard guard(hy.device());
  const auto sizes = hy.sizes();
  const int64_t total = hy.numel();
  update_magnetic_hy_cpml_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      ex.data_ptr<float>(),
      ez.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>(),
      inv_kappa_x.data_ptr<float>(),
      b_x.data_ptr<float>(),
      c_x.data_ptr<float>(),
      inv_kappa_z.data_ptr<float>(),
      b_z.data_ptr<float>(),
      c_z.data_ptr<float>(),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dz),
      psi_x.data_ptr<float>(),
      psi_z.data_ptr<float>(),
      hy.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_magnetic_hz_cpml_cuda(
    at::Tensor hz,
    const at::Tensor& ex,
    const at::Tensor& ey,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_x,
    at::Tensor psi_y,
    const at::Tensor& inv_kappa_x,
    const at::Tensor& b_x,
    const at::Tensor& c_x,
    const at::Tensor& inv_kappa_y,
    const at::Tensor& b_y,
    const at::Tensor& c_y,
    double inv_dx,
    double inv_dy) {
  check_magnetic_cpml_inputs(
      hz, ex, ey, decay, curl, psi_x, psi_y, inv_kappa_x, b_x, c_x, inv_kappa_y, b_y, c_y, 0, 1, "hz");
  c10::cuda::CUDAGuard guard(hz.device());
  const auto sizes = hz.sizes();
  const int64_t total = hz.numel();
  update_magnetic_hz_cpml_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      ex.data_ptr<float>(),
      ey.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>(),
      inv_kappa_x.data_ptr<float>(),
      b_x.data_ptr<float>(),
      c_x.data_ptr<float>(),
      inv_kappa_y.data_ptr<float>(),
      b_y.data_ptr<float>(),
      c_y.data_ptr<float>(),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dy),
      psi_x.data_ptr<float>(),
      psi_y.data_ptr<float>(),
      hz.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_magnetic_hx_cpml_compressed_cuda(
    at::Tensor hx,
    const at::Tensor& ey,
    const at::Tensor& ez,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_y,
    at::Tensor psi_z,
    const at::Tensor& inv_kappa_y,
    const at::Tensor& b_y,
    const at::Tensor& c_y,
    const at::Tensor& inv_kappa_z,
    const at::Tensor& b_z,
    const at::Tensor& c_z,
    double inv_dy,
    double inv_dz,
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length) {
  check_magnetic_cpml_compressed_inputs(
      hx, ey, ez, decay, curl, psi_y, psi_z, inv_kappa_y, b_y, c_y, inv_kappa_z, b_z, c_z,
      1, 2, y_low_length, y_high_length, z_low_length, z_high_length, "hx");
  c10::cuda::CUDAGuard guard(hx.device());
  const auto sizes = hx.sizes();
  const int64_t total = hx.numel();
  update_magnetic_hx_cpml_compressed_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      ey.data_ptr<float>(),
      ez.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>(),
      inv_kappa_y.data_ptr<float>(),
      b_y.data_ptr<float>(),
      c_y.data_ptr<float>(),
      inv_kappa_z.data_ptr<float>(),
      b_z.data_ptr<float>(),
      c_z.data_ptr<float>(),
      static_cast<float>(inv_dy),
      static_cast<float>(inv_dz),
      static_cast<int>(y_low_length),
      static_cast<int>(y_high_start),
      static_cast<int>(y_high_length),
      static_cast<int>(z_low_length),
      static_cast<int>(z_high_start),
      static_cast<int>(z_high_length),
      psi_y.data_ptr<float>(),
      psi_z.data_ptr<float>(),
      hx.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_magnetic_hy_cpml_compressed_cuda(
    at::Tensor hy,
    const at::Tensor& ex,
    const at::Tensor& ez,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_x,
    at::Tensor psi_z,
    const at::Tensor& inv_kappa_x,
    const at::Tensor& b_x,
    const at::Tensor& c_x,
    const at::Tensor& inv_kappa_z,
    const at::Tensor& b_z,
    const at::Tensor& c_z,
    double inv_dx,
    double inv_dz,
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length) {
  check_magnetic_cpml_compressed_inputs(
      hy, ex, ez, decay, curl, psi_x, psi_z, inv_kappa_x, b_x, c_x, inv_kappa_z, b_z, c_z,
      0, 2, x_low_length, x_high_length, z_low_length, z_high_length, "hy");
  c10::cuda::CUDAGuard guard(hy.device());
  const auto sizes = hy.sizes();
  const int64_t total = hy.numel();
  update_magnetic_hy_cpml_compressed_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      ex.data_ptr<float>(),
      ez.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>(),
      inv_kappa_x.data_ptr<float>(),
      b_x.data_ptr<float>(),
      c_x.data_ptr<float>(),
      inv_kappa_z.data_ptr<float>(),
      b_z.data_ptr<float>(),
      c_z.data_ptr<float>(),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dz),
      static_cast<int>(x_low_length),
      static_cast<int>(x_high_start),
      static_cast<int>(x_high_length),
      static_cast<int>(z_low_length),
      static_cast<int>(z_high_start),
      static_cast<int>(z_high_length),
      psi_x.data_ptr<float>(),
      psi_z.data_ptr<float>(),
      hy.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_magnetic_hz_cpml_compressed_cuda(
    at::Tensor hz,
    const at::Tensor& ex,
    const at::Tensor& ey,
    const at::Tensor& decay,
    const at::Tensor& curl,
    at::Tensor psi_x,
    at::Tensor psi_y,
    const at::Tensor& inv_kappa_x,
    const at::Tensor& b_x,
    const at::Tensor& c_x,
    const at::Tensor& inv_kappa_y,
    const at::Tensor& b_y,
    const at::Tensor& c_y,
    double inv_dx,
    double inv_dy,
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length) {
  check_magnetic_cpml_compressed_inputs(
      hz, ex, ey, decay, curl, psi_x, psi_y, inv_kappa_x, b_x, c_x, inv_kappa_y, b_y, c_y,
      0, 1, x_low_length, x_high_length, y_low_length, y_high_length, "hz");
  c10::cuda::CUDAGuard guard(hz.device());
  const auto sizes = hz.sizes();
  const int64_t total = hz.numel();
  update_magnetic_hz_cpml_compressed_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      ex.data_ptr<float>(),
      ey.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>(),
      inv_kappa_x.data_ptr<float>(),
      b_x.data_ptr<float>(),
      c_x.data_ptr<float>(),
      inv_kappa_y.data_ptr<float>(),
      b_y.data_ptr<float>(),
      c_y.data_ptr<float>(),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dy),
      static_cast<int>(x_low_length),
      static_cast<int>(x_high_start),
      static_cast<int>(x_high_length),
      static_cast<int>(y_low_length),
      static_cast<int>(y_high_start),
      static_cast<int>(y_high_length),
      psi_x.data_ptr<float>(),
      psi_y.data_ptr<float>(),
      hz.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}
