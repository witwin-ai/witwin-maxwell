#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

namespace {

dim3 field_block3d() {
  return dim3(32, 4, 2);
}

dim3 field_grid3d(int64_t nx, int64_t ny, int64_t nz, dim3 block) {
  return dim3(
      static_cast<unsigned int>((nz + block.x - 1) / block.x),
      static_cast<unsigned int>((ny + block.y - 1) / block.y),
      static_cast<unsigned int>((nx + block.z - 1) / block.z));
}

__global__ void update_magnetic_hx_standard_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ ey,
    const float* __restrict__ ez,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    float inv_dy,
    float inv_dz,
    float* __restrict__ hx) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const long long ez_hi = offset3d(i, j + 1, k, ny + 1, nz);
  const long long ez_lo = offset3d(i, j, k, ny + 1, nz);
  const long long ey_hi = offset3d(i, j, k + 1, ny, nz + 1);
  const long long ey_lo = offset3d(i, j, k, ny, nz + 1);
  const float curl = (ez[ez_hi] - ez[ez_lo]) * inv_dy - (ey[ey_hi] - ey[ey_lo]) * inv_dz;
  hx[linear] = hx[linear] * decay[linear] - curl_coeff[linear] * curl;
}

__global__ void update_magnetic_hy_standard_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ ex,
    const float* __restrict__ ez,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    float inv_dx,
    float inv_dz,
    float* __restrict__ hy) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const long long ex_hi = offset3d(i, j, k + 1, ny, nz + 1);
  const long long ex_lo = offset3d(i, j, k, ny, nz + 1);
  const long long ez_hi = offset3d(i + 1, j, k, ny, nz);
  const long long ez_lo = offset3d(i, j, k, ny, nz);
  const float curl = (ex[ex_hi] - ex[ex_lo]) * inv_dz - (ez[ez_hi] - ez[ez_lo]) * inv_dx;
  hy[linear] = hy[linear] * decay[linear] - curl_coeff[linear] * curl;
}

__global__ void update_magnetic_hz_standard_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ ex,
    const float* __restrict__ ey,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    float inv_dx,
    float inv_dy,
    float* __restrict__ hz) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const long long ey_hi = offset3d(i + 1, j, k, ny, nz);
  const long long ey_lo = offset3d(i, j, k, ny, nz);
  const long long ex_hi = offset3d(i, j + 1, k, ny + 1, nz);
  const long long ex_lo = offset3d(i, j, k, ny + 1, nz);
  const float curl = (ey[ey_hi] - ey[ey_lo]) * inv_dx - (ex[ex_hi] - ex[ex_lo]) * inv_dy;
  hz[linear] = hz[linear] * decay[linear] - curl_coeff[linear] * curl;
}

__global__ void update_magnetic_hx_cpml_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ ey,
    const float* __restrict__ ez,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ b_y,
    const float* __restrict__ c_y,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ b_z,
    const float* __restrict__ c_z,
    float inv_dy,
    float inv_dz,
    float* __restrict__ psi_y,
    float* __restrict__ psi_z,
    float* __restrict__ hx) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const long long ez_lo = (static_cast<long long>(i) * (ny + 1u) + j) * nz + k;
  const long long ez_hi = ez_lo + nz;
  const long long ey_lo = (static_cast<long long>(i) * ny + j) * (nz + 1u) + k;
  const long long ey_hi = ey_lo + 1;
  const float d_y = (ez[ez_hi] - ez[ez_lo]) * inv_dy;
  const float d_z = (ey[ey_hi] - ey[ey_lo]) * inv_dz;
  const float psi_y_value = b_y[j] * psi_y[linear] + c_y[j] * d_y;
  const float psi_z_value = b_z[k] * psi_z[linear] + c_z[k] * d_z;
  psi_y[linear] = psi_y_value;
  psi_z[linear] = psi_z_value;
  const float curl = d_y * inv_kappa_y[j] + psi_y_value - d_z * inv_kappa_z[k] - psi_z_value;
  hx[linear] = hx[linear] * decay[linear] - curl_coeff[linear] * curl;
}

__global__ void update_magnetic_hy_cpml_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ ex,
    const float* __restrict__ ez,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ b_x,
    const float* __restrict__ c_x,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ b_z,
    const float* __restrict__ c_z,
    float inv_dx,
    float inv_dz,
    float* __restrict__ psi_x,
    float* __restrict__ psi_z,
    float* __restrict__ hy) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const long long ex_lo = (static_cast<long long>(i) * ny + j) * (nz + 1u) + k;
  const long long ex_hi = ex_lo + 1;
  const long long ez_lo = linear;
  const long long ez_hi = ez_lo + static_cast<long long>(ny) * nz;
  const float d_z = (ex[ex_hi] - ex[ex_lo]) * inv_dz;
  const float d_x = (ez[ez_hi] - ez[ez_lo]) * inv_dx;
  const float psi_x_value = b_x[i] * psi_x[linear] + c_x[i] * d_x;
  const float psi_z_value = b_z[k] * psi_z[linear] + c_z[k] * d_z;
  psi_x[linear] = psi_x_value;
  psi_z[linear] = psi_z_value;
  const float curl = d_z * inv_kappa_z[k] + psi_z_value - d_x * inv_kappa_x[i] - psi_x_value;
  hy[linear] = hy[linear] * decay[linear] - curl_coeff[linear] * curl;
}

__global__ void update_magnetic_hz_cpml_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ ex,
    const float* __restrict__ ey,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ b_x,
    const float* __restrict__ c_x,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ b_y,
    const float* __restrict__ c_y,
    float inv_dx,
    float inv_dy,
    float* __restrict__ psi_x,
    float* __restrict__ psi_y,
    float* __restrict__ hz) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const long long ey_lo = linear;
  const long long ey_hi = ey_lo + static_cast<long long>(ny) * nz;
  const long long ex_lo = (static_cast<long long>(i) * (ny + 1u) + j) * nz + k;
  const long long ex_hi = ex_lo + nz;
  const float d_x = (ey[ey_hi] - ey[ey_lo]) * inv_dx;
  const float d_y = (ex[ex_hi] - ex[ex_lo]) * inv_dy;
  const float psi_x_value = b_x[i] * psi_x[linear] + c_x[i] * d_x;
  const float psi_y_value = b_y[j] * psi_y[linear] + c_y[j] * d_y;
  psi_x[linear] = psi_x_value;
  psi_y[linear] = psi_y_value;
  const float curl = d_x * inv_kappa_x[i] + psi_x_value - d_y * inv_kappa_y[j] - psi_y_value;
  hz[linear] = hz[linear] * decay[linear] - curl_coeff[linear] * curl;
}

template <int Axis>
__device__ __forceinline__ float update_compact_magnetic_psi(
    float* __restrict__ psi,
    const float* __restrict__ b,
    const float* __restrict__ c,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    unsigned int size_y,
    unsigned int size_z,
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
  const long long psi_offset = compact_offset3d_axis<Axis>(i, j, k, size_y, size_z, local, compact_length);
  const float value = b[axis_coord] * psi[psi_offset] + c[axis_coord] * derivative;
  psi[psi_offset] = value;
  return value;
}

__global__ void update_magnetic_hx_cpml_compressed_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ ey,
    const float* __restrict__ ez,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ b_y,
    const float* __restrict__ c_y,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ b_z,
    const float* __restrict__ c_z,
    float inv_dy,
    float inv_dz,
    int y_low_length,
    int y_high_start,
    int y_high_length,
    int z_low_length,
    int z_high_start,
    int z_high_length,
    float* __restrict__ psi_y,
    float* __restrict__ psi_z,
    float* __restrict__ hx) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const long long ez_hi = offset3d(i, j + 1, k, ny + 1, nz);
  const long long ez_lo = offset3d(i, j, k, ny + 1, nz);
  const long long ey_hi = offset3d(i, j, k + 1, ny, nz + 1);
  const long long ey_lo = offset3d(i, j, k, ny, nz + 1);
  const float d_y = (ez[ez_hi] - ez[ez_lo]) * inv_dy;
  const float d_z = (ey[ey_hi] - ey[ey_lo]) * inv_dz;
  const float psi_y_value = update_compact_magnetic_psi<1>(
      psi_y, b_y, c_y, i, j, k, ny, nz, j, y_low_length, y_high_start, y_high_length, d_y);
  const float psi_z_value = update_compact_magnetic_psi<2>(
      psi_z, b_z, c_z, i, j, k, ny, nz, k, z_low_length, z_high_start, z_high_length, d_z);
  const float curl = d_y * inv_kappa_y[j] + psi_y_value - d_z * inv_kappa_z[k] - psi_z_value;
  hx[linear] = hx[linear] * decay[linear] - curl_coeff[linear] * curl;
}

__global__ void update_magnetic_hy_cpml_compressed_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ ex,
    const float* __restrict__ ez,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ b_x,
    const float* __restrict__ c_x,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ b_z,
    const float* __restrict__ c_z,
    float inv_dx,
    float inv_dz,
    int x_low_length,
    int x_high_start,
    int x_high_length,
    int z_low_length,
    int z_high_start,
    int z_high_length,
    float* __restrict__ psi_x,
    float* __restrict__ psi_z,
    float* __restrict__ hy) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const long long ex_hi = offset3d(i, j, k + 1, ny, nz + 1);
  const long long ex_lo = offset3d(i, j, k, ny, nz + 1);
  const long long ez_hi = offset3d(i + 1, j, k, ny, nz);
  const long long ez_lo = offset3d(i, j, k, ny, nz);
  const float d_z = (ex[ex_hi] - ex[ex_lo]) * inv_dz;
  const float d_x = (ez[ez_hi] - ez[ez_lo]) * inv_dx;
  const float psi_x_value = update_compact_magnetic_psi<0>(
      psi_x, b_x, c_x, i, j, k, ny, nz, i, x_low_length, x_high_start, x_high_length, d_x);
  const float psi_z_value = update_compact_magnetic_psi<2>(
      psi_z, b_z, c_z, i, j, k, ny, nz, k, z_low_length, z_high_start, z_high_length, d_z);
  const float curl = d_z * inv_kappa_z[k] + psi_z_value - d_x * inv_kappa_x[i] - psi_x_value;
  hy[linear] = hy[linear] * decay[linear] - curl_coeff[linear] * curl;
}

__global__ void update_magnetic_hz_cpml_compressed_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ ex,
    const float* __restrict__ ey,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ b_x,
    const float* __restrict__ c_x,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ b_y,
    const float* __restrict__ c_y,
    float inv_dx,
    float inv_dy,
    int x_low_length,
    int x_high_start,
    int x_high_length,
    int y_low_length,
    int y_high_start,
    int y_high_length,
    float* __restrict__ psi_x,
    float* __restrict__ psi_y,
    float* __restrict__ hz) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const long long ey_hi = offset3d(i + 1, j, k, ny, nz);
  const long long ey_lo = offset3d(i, j, k, ny, nz);
  const long long ex_hi = offset3d(i, j + 1, k, ny + 1, nz);
  const long long ex_lo = offset3d(i, j, k, ny + 1, nz);
  const float d_x = (ey[ey_hi] - ey[ey_lo]) * inv_dx;
  const float d_y = (ex[ex_hi] - ex[ex_lo]) * inv_dy;
  const float psi_x_value = update_compact_magnetic_psi<0>(
      psi_x, b_x, c_x, i, j, k, ny, nz, i, x_low_length, x_high_start, x_high_length, d_x);
  const float psi_y_value = update_compact_magnetic_psi<1>(
      psi_y, b_y, c_y, i, j, k, ny, nz, j, y_low_length, y_high_start, y_high_length, d_y);
  const float curl = d_x * inv_kappa_x[i] + psi_x_value - d_y * inv_kappa_y[j] - psi_y_value;
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
  check_same_cuda_device(field, first, "first");
  check_same_cuda_device(field, second, "second");
  check_same_cuda_device(field, decay, "decay");
  check_same_cuda_device(field, curl, "curl");
  check_contiguous_tensor(field, name);
  check_contiguous_tensor(first, "first");
  check_contiguous_tensor(second, "second");
  check_contiguous_tensor(decay, "decay");
  check_contiguous_tensor(curl, "curl");
  TORCH_CHECK(field.dim() == 3, name, " must be rank 3");
  TORCH_CHECK(first.dim() == 3, "first must be rank 3");
  TORCH_CHECK(second.dim() == 3, "second must be rank 3");
  TORCH_CHECK(decay.sizes() == field.sizes(), "decay must match field shape");
  TORCH_CHECK(curl.sizes() == field.sizes(), "curl must match field shape");
}

void check_rank3_shape(
    const at::Tensor& tensor,
    const char* name,
    int64_t x,
    int64_t y,
    int64_t z) {
  TORCH_CHECK(tensor.dim() == 3, name, " must be rank 3");
  TORCH_CHECK(
      tensor.size(0) == x && tensor.size(1) == y && tensor.size(2) == z,
      name,
      " has an incompatible Yee-grid shape");
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
  check_same_cuda_device(field, psi_first, "psi_first");
  check_same_cuda_device(field, psi_second, "psi_second");
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
  check_same_cuda_device(field, inv_kappa_first, "inv_kappa_first");
  check_same_cuda_device(field, b_first, "b_first");
  check_same_cuda_device(field, c_first, "c_first");
  check_same_cuda_device(field, inv_kappa_second, "inv_kappa_second");
  check_same_cuda_device(field, b_second, "b_second");
  check_same_cuda_device(field, c_second, "c_second");
}

void check_compact_psi_input(
    const at::Tensor& psi,
    const at::Tensor& field,
    int64_t axis,
    int64_t low_length,
    int64_t high_length,
    const char* name) {
  check_float32_tensor(psi, name);
  check_same_cuda_device(field, psi, name);
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
  check_same_cuda_device(field, inv_kappa_first, "inv_kappa_first");
  check_same_cuda_device(field, b_first, "b_first");
  check_same_cuda_device(field, c_first, "c_first");
  check_same_cuda_device(field, inv_kappa_second, "inv_kappa_second");
  check_same_cuda_device(field, b_second, "b_second");
  check_same_cuda_device(field, c_second, "c_second");
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
  check_rank3_shape(ey, "ey", hx.size(0), hx.size(1), hx.size(2) + 1);
  check_rank3_shape(ez, "ez", hx.size(0), hx.size(1) + 1, hx.size(2));
  c10::cuda::CUDAGuard guard(hx.device());
  const auto sizes = hx.sizes();
  const dim3 block = field_block3d();
  update_magnetic_hx_standard_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
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
  check_rank3_shape(ex, "ex", hy.size(0), hy.size(1), hy.size(2) + 1);
  check_rank3_shape(ez, "ez", hy.size(0) + 1, hy.size(1), hy.size(2));
  c10::cuda::CUDAGuard guard(hy.device());
  const auto sizes = hy.sizes();
  const dim3 block = field_block3d();
  update_magnetic_hy_standard_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
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
  check_rank3_shape(ex, "ex", hz.size(0), hz.size(1) + 1, hz.size(2));
  check_rank3_shape(ey, "ey", hz.size(0) + 1, hz.size(1), hz.size(2));
  c10::cuda::CUDAGuard guard(hz.device());
  const auto sizes = hz.sizes();
  const dim3 block = field_block3d();
  update_magnetic_hz_standard_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
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
  check_rank3_shape(ey, "ey", hx.size(0), hx.size(1), hx.size(2) + 1);
  check_rank3_shape(ez, "ez", hx.size(0), hx.size(1) + 1, hx.size(2));
  c10::cuda::CUDAGuard guard(hx.device());
  const auto sizes = hx.sizes();
  const dim3 block = field_block3d();
  update_magnetic_hx_cpml_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
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
  check_rank3_shape(ex, "ex", hy.size(0), hy.size(1), hy.size(2) + 1);
  check_rank3_shape(ez, "ez", hy.size(0) + 1, hy.size(1), hy.size(2));
  c10::cuda::CUDAGuard guard(hy.device());
  const auto sizes = hy.sizes();
  const dim3 block = field_block3d();
  update_magnetic_hy_cpml_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
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
  check_rank3_shape(ex, "ex", hz.size(0), hz.size(1) + 1, hz.size(2));
  check_rank3_shape(ey, "ey", hz.size(0) + 1, hz.size(1), hz.size(2));
  c10::cuda::CUDAGuard guard(hz.device());
  const auto sizes = hz.sizes();
  const dim3 block = field_block3d();
  update_magnetic_hz_cpml_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
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
  check_rank3_shape(ey, "ey", hx.size(0), hx.size(1), hx.size(2) + 1);
  check_rank3_shape(ez, "ez", hx.size(0), hx.size(1) + 1, hx.size(2));
  c10::cuda::CUDAGuard guard(hx.device());
  const auto sizes = hx.sizes();
  const dim3 block = field_block3d();
  update_magnetic_hx_cpml_compressed_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
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
  check_rank3_shape(ex, "ex", hy.size(0), hy.size(1), hy.size(2) + 1);
  check_rank3_shape(ez, "ez", hy.size(0) + 1, hy.size(1), hy.size(2));
  c10::cuda::CUDAGuard guard(hy.device());
  const auto sizes = hy.sizes();
  const dim3 block = field_block3d();
  update_magnetic_hy_cpml_compressed_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
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
  check_rank3_shape(ex, "ex", hz.size(0), hz.size(1) + 1, hz.size(2));
  check_rank3_shape(ey, "ey", hz.size(0) + 1, hz.size(1), hz.size(2));
  c10::cuda::CUDAGuard guard(hz.device());
  const auto sizes = hz.sizes();
  const dim3 block = field_block3d();
  update_magnetic_hz_cpml_compressed_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
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
