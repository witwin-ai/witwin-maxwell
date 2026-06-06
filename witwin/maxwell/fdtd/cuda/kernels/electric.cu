#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

namespace {

__device__ inline float boundary_diff_low(float low_value, float high_value, int low_mode, float inv_delta) {
  if (low_mode == BOUNDARY_PERIODIC) {
    return (low_value - high_value) * inv_delta;
  }
  if (low_mode == BOUNDARY_PMC) {
    return 2.0f * low_value * inv_delta;
  }
  return 0.0f;
}

__device__ inline float boundary_diff_high(float low_value, float high_value, int high_mode, float inv_delta) {
  if (high_mode == BOUNDARY_PERIODIC) {
    return (low_value - high_value) * inv_delta;
  }
  if (high_mode == BOUNDARY_PMC) {
    return -2.0f * high_value * inv_delta;
  }
  return 0.0f;
}

__device__ inline bool boundary_pec(unsigned int coord, unsigned int size, int low_mode, int high_mode) {
  return (coord == 0 && low_mode == BOUNDARY_PEC) || (coord + 1 == size && high_mode == BOUNDARY_PEC);
}

__device__ inline bool boundary_inactive(unsigned int coord, unsigned int size, int low_mode, int high_mode) {
  const bool low = coord == 0 && (low_mode == BOUNDARY_NONE || low_mode == BOUNDARY_PML);
  const bool high = coord + 1 == size && (high_mode == BOUNDARY_NONE || high_mode == BOUNDARY_PML);
  return low || high;
}

__device__ inline float backward_diff_axis0(
    const float* source,
    unsigned int target_x,
    unsigned int source_y,
    unsigned int source_z,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    int low_mode,
    int high_mode,
    float inv_delta) {
  const unsigned int source_x = target_x - 1;
  const float low_value = source[offset3d(0, j, k, source_y, source_z)];
  const float high_value = source[offset3d(source_x - 1, j, k, source_y, source_z)];
  if (i == 0) {
    return boundary_diff_low(low_value, high_value, low_mode, inv_delta);
  }
  if (i + 1 == target_x) {
    return boundary_diff_high(low_value, high_value, high_mode, inv_delta);
  }
  return (source[offset3d(i, j, k, source_y, source_z)] - source[offset3d(i - 1, j, k, source_y, source_z)]) * inv_delta;
}

__device__ inline float backward_diff_axis1(
    const float* source,
    unsigned int source_y,
    unsigned int source_z,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    int low_mode,
    int high_mode,
    float inv_delta) {
  const float low_value = source[offset3d(i, 0, k, source_y, source_z)];
  const float high_value = source[offset3d(i, source_y - 1, k, source_y, source_z)];
  if (j == 0) {
    return boundary_diff_low(low_value, high_value, low_mode, inv_delta);
  }
  if (j == source_y) {
    return boundary_diff_high(low_value, high_value, high_mode, inv_delta);
  }
  return (source[offset3d(i, j, k, source_y, source_z)] - source[offset3d(i, j - 1, k, source_y, source_z)]) * inv_delta;
}

__device__ inline float backward_diff_axis2(
    const float* source,
    unsigned int source_y,
    unsigned int source_z,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    int low_mode,
    int high_mode,
    float inv_delta) {
  const float low_value = source[offset3d(i, j, 0, source_y, source_z)];
  const float high_value = source[offset3d(i, j, source_z - 1, source_y, source_z)];
  if (k == 0) {
    return boundary_diff_low(low_value, high_value, low_mode, inv_delta);
  }
  if (k == source_z) {
    return boundary_diff_high(low_value, high_value, high_mode, inv_delta);
  }
  return (source[offset3d(i, j, k, source_y, source_z)] - source[offset3d(i, j, k - 1, source_y, source_z)]) * inv_delta;
}

struct ComplexValue {
  float real;
  float imag;
};

__device__ inline ComplexValue phase_positive(float real, float imag, float phase_cos, float phase_sin) {
  return {phase_cos * real - phase_sin * imag, phase_sin * real + phase_cos * imag};
}

__device__ inline ComplexValue phase_negative(float real, float imag, float phase_cos, float phase_sin) {
  return {phase_cos * real + phase_sin * imag, phase_cos * imag - phase_sin * real};
}

__device__ inline ComplexValue bloch_backward_diff_axis0(
    const float* real,
    const float* imag,
    unsigned int target_x,
    unsigned int source_x,
    unsigned int source_y,
    unsigned int source_z,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    float phase_cos,
    float phase_sin,
    float inv_delta) {
  const long long low = offset3d(0, j, k, source_y, source_z);
  const long long high = offset3d(source_x - 1, j, k, source_y, source_z);
  if (i == 0) {
    const ComplexValue shifted = phase_negative(real[high], imag[high], phase_cos, phase_sin);
    return {(real[low] - shifted.real) * inv_delta, (imag[low] - shifted.imag) * inv_delta};
  }
  if (i + 1 == target_x) {
    const ComplexValue shifted = phase_positive(real[low], imag[low], phase_cos, phase_sin);
    return {(shifted.real - real[high]) * inv_delta, (shifted.imag - imag[high]) * inv_delta};
  }
  const long long current = offset3d(i, j, k, source_y, source_z);
  const long long previous = offset3d(i - 1, j, k, source_y, source_z);
  return {(real[current] - real[previous]) * inv_delta, (imag[current] - imag[previous]) * inv_delta};
}

__device__ inline ComplexValue bloch_backward_diff_axis1(
    const float* real,
    const float* imag,
    unsigned int target_y,
    unsigned int source_y,
    unsigned int source_z,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    float phase_cos,
    float phase_sin,
    float inv_delta) {
  const long long low = offset3d(i, 0, k, source_y, source_z);
  const long long high = offset3d(i, source_y - 1, k, source_y, source_z);
  if (j == 0) {
    const ComplexValue shifted = phase_negative(real[high], imag[high], phase_cos, phase_sin);
    return {(real[low] - shifted.real) * inv_delta, (imag[low] - shifted.imag) * inv_delta};
  }
  if (j + 1 == target_y) {
    const ComplexValue shifted = phase_positive(real[low], imag[low], phase_cos, phase_sin);
    return {(shifted.real - real[high]) * inv_delta, (shifted.imag - imag[high]) * inv_delta};
  }
  const long long current = offset3d(i, j, k, source_y, source_z);
  const long long previous = offset3d(i, j - 1, k, source_y, source_z);
  return {(real[current] - real[previous]) * inv_delta, (imag[current] - imag[previous]) * inv_delta};
}

__device__ inline ComplexValue bloch_backward_diff_axis2(
    const float* real,
    const float* imag,
    unsigned int target_z,
    unsigned int source_y,
    unsigned int source_z,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    float phase_cos,
    float phase_sin,
    float inv_delta) {
  const long long low = offset3d(i, j, 0, source_y, source_z);
  const long long high = offset3d(i, j, source_z - 1, source_y, source_z);
  if (k == 0) {
    const ComplexValue shifted = phase_negative(real[high], imag[high], phase_cos, phase_sin);
    return {(real[low] - shifted.real) * inv_delta, (imag[low] - shifted.imag) * inv_delta};
  }
  if (k + 1 == target_z) {
    const ComplexValue shifted = phase_positive(real[low], imag[low], phase_cos, phase_sin);
    return {(shifted.real - real[high]) * inv_delta, (shifted.imag - imag[high]) * inv_delta};
  }
  const long long current = offset3d(i, j, k, source_y, source_z);
  const long long previous = offset3d(i, j, k - 1, source_y, source_z);
  return {(real[current] - real[previous]) * inv_delta, (imag[current] - imag[previous]) * inv_delta};
}

__global__ void update_electric_ex_standard_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* hy,
    const float* hz,
    const float* decay,
    const float* curl_coeff,
    float inv_dy,
    float inv_dz,
    int y_low_mode,
    int y_high_mode,
    int z_low_mode,
    int z_high_mode,
    float* ex) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  if (boundary_pec(coord.j, ny, y_low_mode, y_high_mode) || boundary_pec(coord.k, nz, z_low_mode, z_high_mode)) {
    ex[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(coord.j, ny, y_low_mode, y_high_mode) ||
      boundary_inactive(coord.k, nz, z_low_mode, z_high_mode)) {
    return;
  }
  const float d_y = backward_diff_axis1(hz, ny - 1, nz, coord.i, coord.j, coord.k, y_low_mode, y_high_mode, inv_dy);
  const float d_z = backward_diff_axis2(hy, ny, nz - 1, coord.i, coord.j, coord.k, z_low_mode, z_high_mode, inv_dz);
  ex[linear] = ex[linear] * decay[linear] + curl_coeff[linear] * (d_y - d_z);
}

__global__ void update_electric_ey_standard_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* hx,
    const float* hz,
    const float* decay,
    const float* curl_coeff,
    float inv_dx,
    float inv_dz,
    int x_low_mode,
    int x_high_mode,
    int z_low_mode,
    int z_high_mode,
    float* ey) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const unsigned int nx = static_cast<unsigned int>(total / (ny * nz));
  if (boundary_pec(coord.i, nx, x_low_mode, x_high_mode) || boundary_pec(coord.k, nz, z_low_mode, z_high_mode)) {
    ey[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(coord.i, nx, x_low_mode, x_high_mode) ||
      boundary_inactive(coord.k, nz, z_low_mode, z_high_mode)) {
    return;
  }
  const float d_z = backward_diff_axis2(hx, ny, nz - 1, coord.i, coord.j, coord.k, z_low_mode, z_high_mode, inv_dz);
  const float d_x = backward_diff_axis0(hz, nx, ny, nz, coord.i, coord.j, coord.k, x_low_mode, x_high_mode, inv_dx);
  ey[linear] = ey[linear] * decay[linear] + curl_coeff[linear] * (d_z - d_x);
}

__global__ void update_electric_ez_standard_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* hx,
    const float* hy,
    const float* decay,
    const float* curl_coeff,
    float inv_dx,
    float inv_dy,
    int x_low_mode,
    int x_high_mode,
    int y_low_mode,
    int y_high_mode,
    float* ez) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const unsigned int nx = static_cast<unsigned int>(total / (ny * nz));
  if (boundary_pec(coord.i, nx, x_low_mode, x_high_mode) || boundary_pec(coord.j, ny, y_low_mode, y_high_mode)) {
    ez[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(coord.i, nx, x_low_mode, x_high_mode) ||
      boundary_inactive(coord.j, ny, y_low_mode, y_high_mode)) {
    return;
  }
  const float d_x = backward_diff_axis0(hy, nx, ny, nz, coord.i, coord.j, coord.k, x_low_mode, x_high_mode, inv_dx);
  const float d_y = backward_diff_axis1(hx, ny - 1, nz, coord.i, coord.j, coord.k, y_low_mode, y_high_mode, inv_dy);
  ez[linear] = ez[linear] * decay[linear] + curl_coeff[linear] * (d_x - d_y);
}

__global__ void update_electric_ex_bloch_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* hy_real,
    const float* hy_imag,
    const float* hz_real,
    const float* hz_imag,
    const float* decay,
    const float* curl_coeff,
    float phase_cos_y,
    float phase_sin_y,
    float phase_cos_z,
    float phase_sin_z,
    float inv_dy,
    float inv_dz,
    float* ex_real,
    float* ex_imag) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const ComplexValue d_y = bloch_backward_diff_axis1(
      hz_real, hz_imag, ny, ny - 1, nz, coord.i, coord.j, coord.k, phase_cos_y, phase_sin_y, inv_dy);
  const ComplexValue d_z = bloch_backward_diff_axis2(
      hy_real, hy_imag, nz, ny, nz - 1, coord.i, coord.j, coord.k, phase_cos_z, phase_sin_z, inv_dz);
  ex_real[linear] = ex_real[linear] * decay[linear] + curl_coeff[linear] * (d_y.real - d_z.real);
  ex_imag[linear] = ex_imag[linear] * decay[linear] + curl_coeff[linear] * (d_y.imag - d_z.imag);
}

__global__ void update_electric_ey_bloch_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* hx_real,
    const float* hx_imag,
    const float* hz_real,
    const float* hz_imag,
    const float* decay,
    const float* curl_coeff,
    float phase_cos_x,
    float phase_sin_x,
    float phase_cos_z,
    float phase_sin_z,
    float inv_dx,
    float inv_dz,
    float* ey_real,
    float* ey_imag) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const unsigned int nx = static_cast<unsigned int>(total / (ny * nz));
  const ComplexValue d_z = bloch_backward_diff_axis2(
      hx_real, hx_imag, nz, ny, nz - 1, coord.i, coord.j, coord.k, phase_cos_z, phase_sin_z, inv_dz);
  const ComplexValue d_x = bloch_backward_diff_axis0(
      hz_real, hz_imag, nx, nx - 1, ny, nz, coord.i, coord.j, coord.k, phase_cos_x, phase_sin_x, inv_dx);
  ey_real[linear] = ey_real[linear] * decay[linear] + curl_coeff[linear] * (d_z.real - d_x.real);
  ey_imag[linear] = ey_imag[linear] * decay[linear] + curl_coeff[linear] * (d_z.imag - d_x.imag);
}

__global__ void update_electric_ez_bloch_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* hx_real,
    const float* hx_imag,
    const float* hy_real,
    const float* hy_imag,
    const float* decay,
    const float* curl_coeff,
    float phase_cos_x,
    float phase_sin_x,
    float phase_cos_y,
    float phase_sin_y,
    float inv_dx,
    float inv_dy,
    float* ez_real,
    float* ez_imag) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const unsigned int nx = static_cast<unsigned int>(total / (ny * nz));
  const ComplexValue d_x = bloch_backward_diff_axis0(
      hy_real, hy_imag, nx, nx - 1, ny, nz, coord.i, coord.j, coord.k, phase_cos_x, phase_sin_x, inv_dx);
  const ComplexValue d_y = bloch_backward_diff_axis1(
      hx_real, hx_imag, ny, ny - 1, nz, coord.i, coord.j, coord.k, phase_cos_y, phase_sin_y, inv_dy);
  ez_real[linear] = ez_real[linear] * decay[linear] + curl_coeff[linear] * (d_x.real - d_y.real);
  ez_imag[linear] = ez_imag[linear] * decay[linear] + curl_coeff[linear] * (d_x.imag - d_y.imag);
}

__global__ void update_electric_ex_cpml_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* hy,
    const float* hz,
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
    int y_low_mode,
    int y_high_mode,
    int z_low_mode,
    int z_high_mode,
    float* psi_y,
    float* psi_z,
    float* ex) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  if (boundary_pec(coord.j, ny, y_low_mode, y_high_mode) || boundary_pec(coord.k, nz, z_low_mode, z_high_mode)) {
    ex[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(coord.j, ny, y_low_mode, y_high_mode) ||
      boundary_inactive(coord.k, nz, z_low_mode, z_high_mode)) {
    return;
  }
  const float d_y = backward_diff_axis1(hz, ny - 1, nz, coord.i, coord.j, coord.k, y_low_mode, y_high_mode, inv_dy);
  const float d_z = backward_diff_axis2(hy, ny, nz - 1, coord.i, coord.j, coord.k, z_low_mode, z_high_mode, inv_dz);
  const float psi_y_value = b_y[coord.j] * psi_y[linear] + c_y[coord.j] * d_y;
  const float psi_z_value = b_z[coord.k] * psi_z[linear] + c_z[coord.k] * d_z;
  psi_y[linear] = psi_y_value;
  psi_z[linear] = psi_z_value;
  const float curl = d_y * inv_kappa_y[coord.j] + psi_y_value - d_z * inv_kappa_z[coord.k] - psi_z_value;
  ex[linear] = ex[linear] * decay[linear] + curl_coeff[linear] * curl;
}

__global__ void update_electric_ey_cpml_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* hx,
    const float* hz,
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
    int x_low_mode,
    int x_high_mode,
    int z_low_mode,
    int z_high_mode,
    float* psi_x,
    float* psi_z,
    float* ey) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const unsigned int nx = static_cast<unsigned int>(total / (ny * nz));
  if (boundary_pec(coord.i, nx, x_low_mode, x_high_mode) || boundary_pec(coord.k, nz, z_low_mode, z_high_mode)) {
    ey[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(coord.i, nx, x_low_mode, x_high_mode) ||
      boundary_inactive(coord.k, nz, z_low_mode, z_high_mode)) {
    return;
  }
  const float d_z = backward_diff_axis2(hx, ny, nz - 1, coord.i, coord.j, coord.k, z_low_mode, z_high_mode, inv_dz);
  const float d_x = backward_diff_axis0(hz, nx, ny, nz, coord.i, coord.j, coord.k, x_low_mode, x_high_mode, inv_dx);
  const float psi_x_value = b_x[coord.i] * psi_x[linear] + c_x[coord.i] * d_x;
  const float psi_z_value = b_z[coord.k] * psi_z[linear] + c_z[coord.k] * d_z;
  psi_x[linear] = psi_x_value;
  psi_z[linear] = psi_z_value;
  const float curl = d_z * inv_kappa_z[coord.k] + psi_z_value - d_x * inv_kappa_x[coord.i] - psi_x_value;
  ey[linear] = ey[linear] * decay[linear] + curl_coeff[linear] * curl;
}

__global__ void update_electric_ez_cpml_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* hx,
    const float* hy,
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
    int x_low_mode,
    int x_high_mode,
    int y_low_mode,
    int y_high_mode,
    float* psi_x,
    float* psi_y,
    float* ez) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const unsigned int nx = static_cast<unsigned int>(total / (ny * nz));
  if (boundary_pec(coord.i, nx, x_low_mode, x_high_mode) || boundary_pec(coord.j, ny, y_low_mode, y_high_mode)) {
    ez[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(coord.i, nx, x_low_mode, x_high_mode) ||
      boundary_inactive(coord.j, ny, y_low_mode, y_high_mode)) {
    return;
  }
  const float d_x = backward_diff_axis0(hy, nx, ny, nz, coord.i, coord.j, coord.k, x_low_mode, x_high_mode, inv_dx);
  const float d_y = backward_diff_axis1(hx, ny - 1, nz, coord.i, coord.j, coord.k, y_low_mode, y_high_mode, inv_dy);
  const float psi_x_value = b_x[coord.i] * psi_x[linear] + c_x[coord.i] * d_x;
  const float psi_y_value = b_y[coord.j] * psi_y[linear] + c_y[coord.j] * d_y;
  psi_x[linear] = psi_x_value;
  psi_y[linear] = psi_y_value;
  const float curl = d_x * inv_kappa_x[coord.i] + psi_x_value - d_y * inv_kappa_y[coord.j] - psi_y_value;
  ez[linear] = ez[linear] * decay[linear] + curl_coeff[linear] * curl;
}

__device__ inline float update_compact_electric_psi(
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
    float derivative,
    bool active) {
  const int local = compact_local_index(axis_coord, low_length, high_start, high_length);
  if (local < 0) {
    return 0.0f;
  }
  const unsigned int compact_length = static_cast<unsigned int>(low_length + high_length);
  const long long psi_offset = compact_offset3d(axis, i, j, k, size_y, size_z, local, compact_length);
  if (!active) {
    return psi[psi_offset];
  }
  const float value = b[axis_coord] * psi[psi_offset] + c[axis_coord] * derivative;
  psi[psi_offset] = value;
  return value;
}

__global__ void update_electric_ex_cpml_compressed_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* hy,
    const float* hz,
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
    int y_low_mode,
    int y_high_mode,
    int z_low_mode,
    int z_high_mode,
    int y_low_length,
    int y_high_start,
    int y_high_length,
    int z_low_length,
    int z_high_start,
    int z_high_length,
    float* psi_y,
    float* psi_z,
    float* ex) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  if (boundary_pec(coord.j, ny, y_low_mode, y_high_mode) || boundary_pec(coord.k, nz, z_low_mode, z_high_mode)) {
    ex[linear] = 0.0f;
    return;
  }
  const bool active = !(boundary_inactive(coord.j, ny, y_low_mode, y_high_mode) ||
                        boundary_inactive(coord.k, nz, z_low_mode, z_high_mode));
  const float d_y = backward_diff_axis1(hz, ny - 1, nz, coord.i, coord.j, coord.k, y_low_mode, y_high_mode, inv_dy);
  const float d_z = backward_diff_axis2(hy, ny, nz - 1, coord.i, coord.j, coord.k, z_low_mode, z_high_mode, inv_dz);
  const float psi_y_value = update_compact_electric_psi(
      psi_y, b_y, c_y, coord.i, coord.j, coord.k, ny, nz, 1, coord.j, y_low_length, y_high_start, y_high_length, d_y, active);
  const float psi_z_value = update_compact_electric_psi(
      psi_z, b_z, c_z, coord.i, coord.j, coord.k, ny, nz, 2, coord.k, z_low_length, z_high_start, z_high_length, d_z, active);
  if (!active) {
    return;
  }
  const float curl = d_y * inv_kappa_y[coord.j] + psi_y_value - d_z * inv_kappa_z[coord.k] - psi_z_value;
  ex[linear] = ex[linear] * decay[linear] + curl_coeff[linear] * curl;
}

__global__ void update_electric_ey_cpml_compressed_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* hx,
    const float* hz,
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
    int x_low_mode,
    int x_high_mode,
    int z_low_mode,
    int z_high_mode,
    int x_low_length,
    int x_high_start,
    int x_high_length,
    int z_low_length,
    int z_high_start,
    int z_high_length,
    float* psi_x,
    float* psi_z,
    float* ey) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const unsigned int nx = static_cast<unsigned int>(total / (ny * nz));
  if (boundary_pec(coord.i, nx, x_low_mode, x_high_mode) || boundary_pec(coord.k, nz, z_low_mode, z_high_mode)) {
    ey[linear] = 0.0f;
    return;
  }
  const bool active = !(boundary_inactive(coord.i, nx, x_low_mode, x_high_mode) ||
                        boundary_inactive(coord.k, nz, z_low_mode, z_high_mode));
  const float d_z = backward_diff_axis2(hx, ny, nz - 1, coord.i, coord.j, coord.k, z_low_mode, z_high_mode, inv_dz);
  const float d_x = backward_diff_axis0(hz, nx, ny, nz, coord.i, coord.j, coord.k, x_low_mode, x_high_mode, inv_dx);
  const float psi_x_value = update_compact_electric_psi(
      psi_x, b_x, c_x, coord.i, coord.j, coord.k, ny, nz, 0, coord.i, x_low_length, x_high_start, x_high_length, d_x, active);
  const float psi_z_value = update_compact_electric_psi(
      psi_z, b_z, c_z, coord.i, coord.j, coord.k, ny, nz, 2, coord.k, z_low_length, z_high_start, z_high_length, d_z, active);
  if (!active) {
    return;
  }
  const float curl = d_z * inv_kappa_z[coord.k] + psi_z_value - d_x * inv_kappa_x[coord.i] - psi_x_value;
  ey[linear] = ey[linear] * decay[linear] + curl_coeff[linear] * curl;
}

__global__ void update_electric_ez_cpml_compressed_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* hx,
    const float* hy,
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
    int x_low_mode,
    int x_high_mode,
    int y_low_mode,
    int y_high_mode,
    int x_low_length,
    int x_high_start,
    int x_high_length,
    int y_low_length,
    int y_high_start,
    int y_high_length,
    float* psi_x,
    float* psi_y,
    float* ez) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ny, nz);
  const unsigned int nx = static_cast<unsigned int>(total / (ny * nz));
  if (boundary_pec(coord.i, nx, x_low_mode, x_high_mode) || boundary_pec(coord.j, ny, y_low_mode, y_high_mode)) {
    ez[linear] = 0.0f;
    return;
  }
  const bool active = !(boundary_inactive(coord.i, nx, x_low_mode, x_high_mode) ||
                        boundary_inactive(coord.j, ny, y_low_mode, y_high_mode));
  const float d_x = backward_diff_axis0(hy, nx, ny, nz, coord.i, coord.j, coord.k, x_low_mode, x_high_mode, inv_dx);
  const float d_y = backward_diff_axis1(hx, ny - 1, nz, coord.i, coord.j, coord.k, y_low_mode, y_high_mode, inv_dy);
  const float psi_x_value = update_compact_electric_psi(
      psi_x, b_x, c_x, coord.i, coord.j, coord.k, ny, nz, 0, coord.i, x_low_length, x_high_start, x_high_length, d_x, active);
  const float psi_y_value = update_compact_electric_psi(
      psi_y, b_y, c_y, coord.i, coord.j, coord.k, ny, nz, 1, coord.j, y_low_length, y_high_start, y_high_length, d_y, active);
  if (!active) {
    return;
  }
  const float curl = d_x * inv_kappa_x[coord.i] + psi_x_value - d_y * inv_kappa_y[coord.j] - psi_y_value;
  ez[linear] = ez[linear] * decay[linear] + curl_coeff[linear] * curl;
}

void check_electric_inputs(
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

void check_electric_bloch_inputs(
    const at::Tensor& field_real,
    const at::Tensor& field_imag,
    const at::Tensor& first_real,
    const at::Tensor& first_imag,
    const at::Tensor& second_real,
    const at::Tensor& second_imag,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const char* name) {
  check_electric_inputs(field_real, first_real, second_real, decay, curl, name);
  check_float32_tensor(field_imag, "field_imag");
  check_float32_tensor(first_imag, "first_imag");
  check_float32_tensor(second_imag, "second_imag");
  check_contiguous_tensor(field_imag, "field_imag");
  check_contiguous_tensor(first_imag, "first_imag");
  check_contiguous_tensor(second_imag, "second_imag");
  TORCH_CHECK(field_imag.sizes() == field_real.sizes(), "field imaginary tensor must match real shape");
  TORCH_CHECK(first_imag.sizes() == first_real.sizes(), "first imaginary tensor must match real shape");
  TORCH_CHECK(second_imag.sizes() == second_real.sizes(), "second imaginary tensor must match real shape");
  TORCH_CHECK(first_real.dim() == 3, "first Bloch source tensor must be rank 3");
  TORCH_CHECK(second_real.dim() == 3, "second Bloch source tensor must be rank 3");
}

void check_vector_input(const at::Tensor& tensor, int64_t length, const char* name) {
  check_float32_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  TORCH_CHECK(tensor.dim() == 1, name, " must be rank 1");
  TORCH_CHECK(tensor.size(0) == length, name, " length must match CPML field axis");
}

void check_electric_cpml_inputs(
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
  check_electric_inputs(field, first, second, decay, curl, name);
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

void check_electric_cpml_compressed_inputs(
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
  check_electric_inputs(field, first, second, decay, curl, name);
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

void update_electric_ex_standard_cuda(
    at::Tensor ex,
    const at::Tensor& hy,
    const at::Tensor& hz,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double inv_dy,
    double inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  check_electric_inputs(ex, hy, hz, decay, curl, "ex");
  c10::cuda::CUDAGuard guard(ex.device());
  const auto sizes = ex.sizes();
  const int64_t total = ex.numel();
  update_electric_ex_standard_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hy.data_ptr<float>(),
      hz.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>(),
      static_cast<float>(inv_dy),
      static_cast<float>(inv_dz),
      static_cast<int>(y_low_mode),
      static_cast<int>(y_high_mode),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode),
      ex.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ey_standard_cuda(
    at::Tensor ey,
    const at::Tensor& hx,
    const at::Tensor& hz,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double inv_dx,
    double inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  check_electric_inputs(ey, hx, hz, decay, curl, "ey");
  c10::cuda::CUDAGuard guard(ey.device());
  const auto sizes = ey.sizes();
  const int64_t total = ey.numel();
  update_electric_ey_standard_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hx.data_ptr<float>(),
      hz.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>(),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dz),
      static_cast<int>(x_low_mode),
      static_cast<int>(x_high_mode),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode),
      ey.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ez_standard_cuda(
    at::Tensor ez,
    const at::Tensor& hx,
    const at::Tensor& hy,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double inv_dx,
    double inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode) {
  check_electric_inputs(ez, hx, hy, decay, curl, "ez");
  c10::cuda::CUDAGuard guard(ez.device());
  const auto sizes = ez.sizes();
  const int64_t total = ez.numel();
  update_electric_ez_standard_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hx.data_ptr<float>(),
      hy.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>(),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dy),
      static_cast<int>(x_low_mode),
      static_cast<int>(x_high_mode),
      static_cast<int>(y_low_mode),
      static_cast<int>(y_high_mode),
      ez.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ex_bloch_cuda(
    at::Tensor ex_real,
    at::Tensor ex_imag,
    const at::Tensor& hy_real,
    const at::Tensor& hy_imag,
    const at::Tensor& hz_real,
    const at::Tensor& hz_imag,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double phase_cos_y,
    double phase_sin_y,
    double phase_cos_z,
    double phase_sin_z,
    double inv_dy,
    double inv_dz) {
  check_electric_bloch_inputs(ex_real, ex_imag, hy_real, hy_imag, hz_real, hz_imag, decay, curl, "ex_real");
  c10::cuda::CUDAGuard guard(ex_real.device());
  const auto sizes = ex_real.sizes();
  const int64_t total = ex_real.numel();
  update_electric_ex_bloch_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hy_real.data_ptr<float>(),
      hy_imag.data_ptr<float>(),
      hz_real.data_ptr<float>(),
      hz_imag.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>(),
      static_cast<float>(phase_cos_y),
      static_cast<float>(phase_sin_y),
      static_cast<float>(phase_cos_z),
      static_cast<float>(phase_sin_z),
      static_cast<float>(inv_dy),
      static_cast<float>(inv_dz),
      ex_real.data_ptr<float>(),
      ex_imag.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ey_bloch_cuda(
    at::Tensor ey_real,
    at::Tensor ey_imag,
    const at::Tensor& hx_real,
    const at::Tensor& hx_imag,
    const at::Tensor& hz_real,
    const at::Tensor& hz_imag,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_z,
    double phase_sin_z,
    double inv_dx,
    double inv_dz) {
  check_electric_bloch_inputs(ey_real, ey_imag, hx_real, hx_imag, hz_real, hz_imag, decay, curl, "ey_real");
  c10::cuda::CUDAGuard guard(ey_real.device());
  const auto sizes = ey_real.sizes();
  const int64_t total = ey_real.numel();
  update_electric_ey_bloch_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hx_real.data_ptr<float>(),
      hx_imag.data_ptr<float>(),
      hz_real.data_ptr<float>(),
      hz_imag.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>(),
      static_cast<float>(phase_cos_x),
      static_cast<float>(phase_sin_x),
      static_cast<float>(phase_cos_z),
      static_cast<float>(phase_sin_z),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dz),
      ey_real.data_ptr<float>(),
      ey_imag.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ez_bloch_cuda(
    at::Tensor ez_real,
    at::Tensor ez_imag,
    const at::Tensor& hx_real,
    const at::Tensor& hx_imag,
    const at::Tensor& hy_real,
    const at::Tensor& hy_imag,
    const at::Tensor& decay,
    const at::Tensor& curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_y,
    double phase_sin_y,
    double inv_dx,
    double inv_dy) {
  check_electric_bloch_inputs(ez_real, ez_imag, hx_real, hx_imag, hy_real, hy_imag, decay, curl, "ez_real");
  c10::cuda::CUDAGuard guard(ez_real.device());
  const auto sizes = ez_real.sizes();
  const int64_t total = ez_real.numel();
  update_electric_ez_bloch_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hx_real.data_ptr<float>(),
      hx_imag.data_ptr<float>(),
      hy_real.data_ptr<float>(),
      hy_imag.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>(),
      static_cast<float>(phase_cos_x),
      static_cast<float>(phase_sin_x),
      static_cast<float>(phase_cos_y),
      static_cast<float>(phase_sin_y),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dy),
      ez_real.data_ptr<float>(),
      ez_imag.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ex_cpml_cuda(
    at::Tensor ex,
    const at::Tensor& hy,
    const at::Tensor& hz,
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
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  check_electric_cpml_inputs(
      ex, hy, hz, decay, curl, psi_y, psi_z, inv_kappa_y, b_y, c_y, inv_kappa_z, b_z, c_z, 1, 2, "ex");
  c10::cuda::CUDAGuard guard(ex.device());
  const auto sizes = ex.sizes();
  const int64_t total = ex.numel();
  update_electric_ex_cpml_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hy.data_ptr<float>(),
      hz.data_ptr<float>(),
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
      static_cast<int>(y_low_mode),
      static_cast<int>(y_high_mode),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode),
      psi_y.data_ptr<float>(),
      psi_z.data_ptr<float>(),
      ex.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ey_cpml_cuda(
    at::Tensor ey,
    const at::Tensor& hx,
    const at::Tensor& hz,
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
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  check_electric_cpml_inputs(
      ey, hx, hz, decay, curl, psi_x, psi_z, inv_kappa_x, b_x, c_x, inv_kappa_z, b_z, c_z, 0, 2, "ey");
  c10::cuda::CUDAGuard guard(ey.device());
  const auto sizes = ey.sizes();
  const int64_t total = ey.numel();
  update_electric_ey_cpml_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hx.data_ptr<float>(),
      hz.data_ptr<float>(),
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
      static_cast<int>(x_low_mode),
      static_cast<int>(x_high_mode),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode),
      psi_x.data_ptr<float>(),
      psi_z.data_ptr<float>(),
      ey.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ez_cpml_cuda(
    at::Tensor ez,
    const at::Tensor& hx,
    const at::Tensor& hy,
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
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode) {
  check_electric_cpml_inputs(
      ez, hx, hy, decay, curl, psi_x, psi_y, inv_kappa_x, b_x, c_x, inv_kappa_y, b_y, c_y, 0, 1, "ez");
  c10::cuda::CUDAGuard guard(ez.device());
  const auto sizes = ez.sizes();
  const int64_t total = ez.numel();
  update_electric_ez_cpml_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hx.data_ptr<float>(),
      hy.data_ptr<float>(),
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
      static_cast<int>(x_low_mode),
      static_cast<int>(x_high_mode),
      static_cast<int>(y_low_mode),
      static_cast<int>(y_high_mode),
      psi_x.data_ptr<float>(),
      psi_y.data_ptr<float>(),
      ez.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ex_cpml_compressed_cuda(
    at::Tensor ex,
    const at::Tensor& hy,
    const at::Tensor& hz,
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
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode,
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length) {
  check_electric_cpml_compressed_inputs(
      ex, hy, hz, decay, curl, psi_y, psi_z, inv_kappa_y, b_y, c_y, inv_kappa_z, b_z, c_z,
      1, 2, y_low_length, y_high_length, z_low_length, z_high_length, "ex");
  c10::cuda::CUDAGuard guard(ex.device());
  const auto sizes = ex.sizes();
  const int64_t total = ex.numel();
  update_electric_ex_cpml_compressed_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hy.data_ptr<float>(),
      hz.data_ptr<float>(),
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
      static_cast<int>(y_low_mode),
      static_cast<int>(y_high_mode),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode),
      static_cast<int>(y_low_length),
      static_cast<int>(y_high_start),
      static_cast<int>(y_high_length),
      static_cast<int>(z_low_length),
      static_cast<int>(z_high_start),
      static_cast<int>(z_high_length),
      psi_y.data_ptr<float>(),
      psi_z.data_ptr<float>(),
      ex.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ey_cpml_compressed_cuda(
    at::Tensor ey,
    const at::Tensor& hx,
    const at::Tensor& hz,
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
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode,
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length) {
  check_electric_cpml_compressed_inputs(
      ey, hx, hz, decay, curl, psi_x, psi_z, inv_kappa_x, b_x, c_x, inv_kappa_z, b_z, c_z,
      0, 2, x_low_length, x_high_length, z_low_length, z_high_length, "ey");
  c10::cuda::CUDAGuard guard(ey.device());
  const auto sizes = ey.sizes();
  const int64_t total = ey.numel();
  update_electric_ey_cpml_compressed_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hx.data_ptr<float>(),
      hz.data_ptr<float>(),
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
      static_cast<int>(x_low_mode),
      static_cast<int>(x_high_mode),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode),
      static_cast<int>(x_low_length),
      static_cast<int>(x_high_start),
      static_cast<int>(x_high_length),
      static_cast<int>(z_low_length),
      static_cast<int>(z_high_start),
      static_cast<int>(z_high_length),
      psi_x.data_ptr<float>(),
      psi_z.data_ptr<float>(),
      ey.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ez_cpml_compressed_cuda(
    at::Tensor ez,
    const at::Tensor& hx,
    const at::Tensor& hy,
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
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length) {
  check_electric_cpml_compressed_inputs(
      ez, hx, hy, decay, curl, psi_x, psi_y, inv_kappa_x, b_x, c_x, inv_kappa_y, b_y, c_y,
      0, 1, x_low_length, x_high_length, y_low_length, y_high_length, "ez");
  c10::cuda::CUDAGuard guard(ez.device());
  const auto sizes = ez.sizes();
  const int64_t total = ez.numel();
  update_electric_ez_cpml_compressed_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hx.data_ptr<float>(),
      hy.data_ptr<float>(),
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
      static_cast<int>(x_low_mode),
      static_cast<int>(x_high_mode),
      static_cast<int>(y_low_mode),
      static_cast<int>(y_high_mode),
      static_cast<int>(x_low_length),
      static_cast<int>(x_high_start),
      static_cast<int>(x_high_length),
      static_cast<int>(y_low_length),
      static_cast<int>(y_high_start),
      static_cast<int>(y_high_length),
      psi_x.data_ptr<float>(),
      psi_y.data_ptr<float>(),
      ez.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}
