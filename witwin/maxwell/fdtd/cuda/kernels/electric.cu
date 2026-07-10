#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

namespace {

dim3 field_block3d() {
  return dim3(128, 2, 1);
}

dim3 field_grid3d(int64_t nx, int64_t ny, int64_t nz, dim3 block) {
  return dim3(
      static_cast<unsigned int>((nz + block.x - 1) / block.x),
      static_cast<unsigned int>((ny + block.y - 1) / block.y),
      static_cast<unsigned int>((nx + block.z - 1) / block.z));
}

bool inactive_boundary_pair(int64_t low_mode, int64_t high_mode) {
  return (low_mode == BOUNDARY_NONE || low_mode == BOUNDARY_PML) &&
      (high_mode == BOUNDARY_NONE || high_mode == BOUNDARY_PML);
}

__device__ __forceinline__ float boundary_diff_low(float low_value, float high_value, int low_mode, float inv_delta) {
  if (low_mode == BOUNDARY_PERIODIC) {
    return (low_value - high_value) * inv_delta;
  }
  if (low_mode == BOUNDARY_PMC) {
    return 2.0f * low_value * inv_delta;
  }
  return 0.0f;
}

__device__ __forceinline__ float boundary_diff_high(float low_value, float high_value, int high_mode, float inv_delta) {
  if (high_mode == BOUNDARY_PERIODIC) {
    return (low_value - high_value) * inv_delta;
  }
  if (high_mode == BOUNDARY_PMC) {
    return -2.0f * high_value * inv_delta;
  }
  return 0.0f;
}

__device__ __forceinline__ bool boundary_pec(unsigned int coord, unsigned int size, int low_mode, int high_mode) {
  return (coord == 0 && low_mode == BOUNDARY_PEC) || (coord + 1 == size && high_mode == BOUNDARY_PEC);
}

__device__ __forceinline__ bool boundary_inactive(unsigned int coord, unsigned int size, int low_mode, int high_mode) {
  const bool low = coord == 0 && (low_mode == BOUNDARY_NONE || low_mode == BOUNDARY_PML);
  const bool high = coord + 1 == size && (high_mode == BOUNDARY_NONE || high_mode == BOUNDARY_PML);
  return low || high;
}

__device__ __forceinline__ float backward_diff_axis0(
    const float* __restrict__ source,
    unsigned int target_x,
    unsigned int source_y,
    unsigned int source_z,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    int low_mode,
  int high_mode,
  const float* __restrict__ inv_delta) {
  const float inv = inv_delta[i];
  const unsigned int source_x = target_x - 1;
  if (i == 0) {
    const float low_value = source[offset3d(0, j, k, source_y, source_z)];
    const float high_value = source[offset3d(source_x - 1, j, k, source_y, source_z)];
    return boundary_diff_low(low_value, high_value, low_mode, inv);
  }
  if (i + 1 == target_x) {
    const float low_value = source[offset3d(0, j, k, source_y, source_z)];
    const float high_value = source[offset3d(source_x - 1, j, k, source_y, source_z)];
    return boundary_diff_high(low_value, high_value, high_mode, inv);
  }
  return (source[offset3d(i, j, k, source_y, source_z)] - source[offset3d(i - 1, j, k, source_y, source_z)]) * inv;
}

__device__ __forceinline__ float backward_diff_axis1(
    const float* __restrict__ source,
    unsigned int source_y,
    unsigned int source_z,
    unsigned int i,
    unsigned int j,
    unsigned int k,
  int low_mode,
  int high_mode,
  const float* __restrict__ inv_delta) {
  const float inv = inv_delta[j];
  if (j == 0) {
    const float low_value = source[offset3d(i, 0, k, source_y, source_z)];
    const float high_value = source[offset3d(i, source_y - 1, k, source_y, source_z)];
    return boundary_diff_low(low_value, high_value, low_mode, inv);
  }
  if (j == source_y) {
    const float low_value = source[offset3d(i, 0, k, source_y, source_z)];
    const float high_value = source[offset3d(i, source_y - 1, k, source_y, source_z)];
    return boundary_diff_high(low_value, high_value, high_mode, inv);
  }
  return (source[offset3d(i, j, k, source_y, source_z)] - source[offset3d(i, j - 1, k, source_y, source_z)]) * inv;
}

__device__ __forceinline__ float backward_diff_axis2(
    const float* __restrict__ source,
    unsigned int source_y,
    unsigned int source_z,
    unsigned int i,
    unsigned int j,
    unsigned int k,
  int low_mode,
  int high_mode,
  const float* __restrict__ inv_delta) {
  const float inv = inv_delta[k];
  if (k == 0) {
    const float low_value = source[offset3d(i, j, 0, source_y, source_z)];
    const float high_value = source[offset3d(i, j, source_z - 1, source_y, source_z)];
    return boundary_diff_low(low_value, high_value, low_mode, inv);
  }
  if (k == source_z) {
    const float low_value = source[offset3d(i, j, 0, source_y, source_z)];
    const float high_value = source[offset3d(i, j, source_z - 1, source_y, source_z)];
    return boundary_diff_high(low_value, high_value, high_mode, inv);
  }
  return (source[offset3d(i, j, k, source_y, source_z)] - source[offset3d(i, j, k - 1, source_y, source_z)]) * inv;
}

struct ComplexValue {
  float real;
  float imag;
};

__device__ __forceinline__ ComplexValue phase_positive(float real, float imag, float phase_cos, float phase_sin) {
  return {phase_cos * real - phase_sin * imag, phase_sin * real + phase_cos * imag};
}

__device__ __forceinline__ ComplexValue phase_negative(float real, float imag, float phase_cos, float phase_sin) {
  return {phase_cos * real + phase_sin * imag, phase_cos * imag - phase_sin * real};
}

__device__ __forceinline__ ComplexValue bloch_backward_diff_axis0(
    const float* __restrict__ real,
    const float* __restrict__ imag,
    unsigned int target_x,
    unsigned int source_x,
    unsigned int source_y,
    unsigned int source_z,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    float phase_cos,
    float phase_sin,
    const float* __restrict__ inv_delta) {
  const float inv = inv_delta[i];
  if (i == 0) {
    const long long low = offset3d(0, j, k, source_y, source_z);
    const long long high = offset3d(source_x - 1, j, k, source_y, source_z);
    const ComplexValue shifted = phase_negative(real[high], imag[high], phase_cos, phase_sin);
    return {(real[low] - shifted.real) * inv, (imag[low] - shifted.imag) * inv};
  }
  if (i + 1 == target_x) {
    const long long low = offset3d(0, j, k, source_y, source_z);
    const long long high = offset3d(source_x - 1, j, k, source_y, source_z);
    const ComplexValue shifted = phase_positive(real[low], imag[low], phase_cos, phase_sin);
    return {(shifted.real - real[high]) * inv, (shifted.imag - imag[high]) * inv};
  }
  const long long current = offset3d(i, j, k, source_y, source_z);
  const long long previous = offset3d(i - 1, j, k, source_y, source_z);
  return {(real[current] - real[previous]) * inv, (imag[current] - imag[previous]) * inv};
}

__device__ __forceinline__ ComplexValue bloch_backward_diff_axis1(
    const float* __restrict__ real,
    const float* __restrict__ imag,
    unsigned int target_y,
    unsigned int source_y,
    unsigned int source_z,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    float phase_cos,
    float phase_sin,
    const float* __restrict__ inv_delta) {
  const float inv = inv_delta[j];
  if (j == 0) {
    const long long low = offset3d(i, 0, k, source_y, source_z);
    const long long high = offset3d(i, source_y - 1, k, source_y, source_z);
    const ComplexValue shifted = phase_negative(real[high], imag[high], phase_cos, phase_sin);
    return {(real[low] - shifted.real) * inv, (imag[low] - shifted.imag) * inv};
  }
  if (j + 1 == target_y) {
    const long long low = offset3d(i, 0, k, source_y, source_z);
    const long long high = offset3d(i, source_y - 1, k, source_y, source_z);
    const ComplexValue shifted = phase_positive(real[low], imag[low], phase_cos, phase_sin);
    return {(shifted.real - real[high]) * inv, (shifted.imag - imag[high]) * inv};
  }
  const long long current = offset3d(i, j, k, source_y, source_z);
  const long long previous = offset3d(i, j - 1, k, source_y, source_z);
  return {(real[current] - real[previous]) * inv, (imag[current] - imag[previous]) * inv};
}

__device__ __forceinline__ ComplexValue bloch_backward_diff_axis2(
    const float* __restrict__ real,
    const float* __restrict__ imag,
    unsigned int target_z,
    unsigned int source_y,
    unsigned int source_z,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    float phase_cos,
    float phase_sin,
    const float* __restrict__ inv_delta) {
  const float inv = inv_delta[k];
  if (k == 0) {
    const long long low = offset3d(i, j, 0, source_y, source_z);
    const long long high = offset3d(i, j, source_z - 1, source_y, source_z);
    const ComplexValue shifted = phase_negative(real[high], imag[high], phase_cos, phase_sin);
    return {(real[low] - shifted.real) * inv, (imag[low] - shifted.imag) * inv};
  }
  if (k + 1 == target_z) {
    const long long low = offset3d(i, j, 0, source_y, source_z);
    const long long high = offset3d(i, j, source_z - 1, source_y, source_z);
    const ComplexValue shifted = phase_positive(real[low], imag[low], phase_cos, phase_sin);
    return {(shifted.real - real[high]) * inv, (shifted.imag - imag[high]) * inv};
  }
  const long long current = offset3d(i, j, k, source_y, source_z);
  const long long previous = offset3d(i, j, k - 1, source_y, source_z);
  return {(real[current] - real[previous]) * inv, (imag[current] - imag[previous]) * inv};
}

__global__ void update_electric_ex_standard_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hy,
    const float* __restrict__ hz,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
    int y_low_mode,
    int y_high_mode,
    int z_low_mode,
    int z_high_mode,
    float* __restrict__ ex) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(j, ny, y_low_mode, y_high_mode) || boundary_pec(k, nz, z_low_mode, z_high_mode)) {
    ex[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(j, ny, y_low_mode, y_high_mode) ||
      boundary_inactive(k, nz, z_low_mode, z_high_mode)) {
    return;
  }
  const float d_y = backward_diff_axis1(hz, ny - 1, nz, i, j, k, y_low_mode, y_high_mode, inv_dy);
  const float d_z = backward_diff_axis2(hy, ny, nz - 1, i, j, k, z_low_mode, z_high_mode, inv_dz);
  ex[linear] = ex[linear] * decay[linear] + curl_coeff[linear] * (d_y - d_z);
}

__global__ void update_electric_ex_standard_interior_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hy,
    const float* __restrict__ hz,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
    float* __restrict__ ex) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j + 1 >= ny || k + 1 >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const long long hz_current = offset3d(i, j, k, ny - 1, nz);
  const long long hz_previous = offset3d(i, j - 1, k, ny - 1, nz);
  const long long hy_current = offset3d(i, j, k, ny, nz - 1);
  const long long hy_previous = offset3d(i, j, k - 1, ny, nz - 1);
  const float d_y = (hz[hz_current] - hz[hz_previous]) * inv_dy[j];
  const float d_z = (hy[hy_current] - hy[hy_previous]) * inv_dz[k];
  ex[linear] = ex[linear] * decay[linear] + curl_coeff[linear] * (d_y - d_z);
}

__global__ void update_electric_ey_standard_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hz,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dz,
    int x_low_mode,
    int x_high_mode,
    int z_low_mode,
    int z_high_mode,
    float* __restrict__ ey) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(i, nx, x_low_mode, x_high_mode) || boundary_pec(k, nz, z_low_mode, z_high_mode)) {
    ey[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(i, nx, x_low_mode, x_high_mode) ||
      boundary_inactive(k, nz, z_low_mode, z_high_mode)) {
    return;
  }
  const float d_z = backward_diff_axis2(hx, ny, nz - 1, i, j, k, z_low_mode, z_high_mode, inv_dz);
  const float d_x = backward_diff_axis0(hz, nx, ny, nz, i, j, k, x_low_mode, x_high_mode, inv_dx);
  ey[linear] = ey[linear] * decay[linear] + curl_coeff[linear] * (d_z - d_x);
}

__global__ void update_electric_ey_standard_interior_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hz,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dz,
    float* __restrict__ ey) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  if (i + 1 >= nx || j >= ny || k + 1 >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const long long hx_current = offset3d(i, j, k, ny, nz - 1);
  const long long hx_previous = offset3d(i, j, k - 1, ny, nz - 1);
  const long long hz_current = offset3d(i, j, k, ny, nz);
  const long long hz_previous = offset3d(i - 1, j, k, ny, nz);
  const float d_z = (hx[hx_current] - hx[hx_previous]) * inv_dz[k];
  const float d_x = (hz[hz_current] - hz[hz_previous]) * inv_dx[i];
  ey[linear] = ey[linear] * decay[linear] + curl_coeff[linear] * (d_z - d_x);
}

__global__ void update_electric_ez_standard_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hy,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy,
    int x_low_mode,
    int x_high_mode,
    int y_low_mode,
    int y_high_mode,
    float* __restrict__ ez) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(i, nx, x_low_mode, x_high_mode) || boundary_pec(j, ny, y_low_mode, y_high_mode)) {
    ez[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(i, nx, x_low_mode, x_high_mode) ||
      boundary_inactive(j, ny, y_low_mode, y_high_mode)) {
    return;
  }
  const float d_x = backward_diff_axis0(hy, nx, ny, nz, i, j, k, x_low_mode, x_high_mode, inv_dx);
  const float d_y = backward_diff_axis1(hx, ny - 1, nz, i, j, k, y_low_mode, y_high_mode, inv_dy);
  ez[linear] = ez[linear] * decay[linear] + curl_coeff[linear] * (d_x - d_y);
}

__global__ void update_electric_ez_standard_interior_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hy,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy,
    float* __restrict__ ez) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  if (i + 1 >= nx || j + 1 >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const long long hy_current = offset3d(i, j, k, ny, nz);
  const long long hy_previous = offset3d(i - 1, j, k, ny, nz);
  const long long hx_current = offset3d(i, j, k, ny - 1, nz);
  const long long hx_previous = offset3d(i, j - 1, k, ny - 1, nz);
  const float d_x = (hy[hy_current] - hy[hy_previous]) * inv_dx[i];
  const float d_y = (hx[hx_current] - hx[hx_previous]) * inv_dy[j];
  ez[linear] = ez[linear] * decay[linear] + curl_coeff[linear] * (d_x - d_y);
}

// Space-time modulated permittivity: eps(x, t) = eps_static(x) * m(x, t) with
//   m(x, t) = 1 + mod_cos(x) * cos(W*t) - mod_sin(x) * sin(W*t)
// where mod_cos = amplitude*cos(phase) and mod_sin = amplitude*sin(phase) are
// compiled once. The conservative update of d(eps E)/dt = curl H gives
//   E_new = decay * (m_prev / m_next) * E_old + (curl_coeff / m_next) * curl(H)
// which reduces bit-exactly to the static update where the modulation depth is
// zero (m_prev = m_next = 1). The host passes the (cos, sin) phase pairs of the
// previous and new E time instants as scalars.
__device__ __forceinline__ float modulation_factor(
    float mod_cos, float mod_sin, float phase_cos, float phase_sin) {
  return 1.0f + mod_cos * phase_cos - mod_sin * phase_sin;
}

// Per-cell modulation phase. The scene may hold several distinct modulation
// frequencies; each Yee edge carries its OWN angular frequency ``omega`` (0 where
// unmodulated), so the phase cos/sin of the previous and new E time instants are
// evaluated per cell here rather than passed as scene-wide host scalars. This is
// what lets disjoint structures modulate at independent frequencies in one run.
__device__ __forceinline__ void modulation_phase_from_omega(
    float omega, float t_prev, float t_next,
    float& cos_prev, float& sin_prev, float& cos_next, float& sin_next) {
  sincosf(omega * t_prev, &sin_prev, &cos_prev);
  sincosf(omega * t_next, &sin_next, &cos_next);
}

__global__ void update_electric_ex_modulated_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hy,
    const float* __restrict__ hz,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ mod_cos,
    const float* __restrict__ mod_sin,
    const float* __restrict__ mod_omega,
    float t_prev,
    float t_next,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
    int y_low_mode,
    int y_high_mode,
    int z_low_mode,
    int z_high_mode,
    float* __restrict__ ex) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(j, ny, y_low_mode, y_high_mode) || boundary_pec(k, nz, z_low_mode, z_high_mode)) {
    ex[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(j, ny, y_low_mode, y_high_mode) ||
      boundary_inactive(k, nz, z_low_mode, z_high_mode)) {
    return;
  }
  const float d_y = backward_diff_axis1(hz, ny - 1, nz, i, j, k, y_low_mode, y_high_mode, inv_dy);
  const float d_z = backward_diff_axis2(hy, ny, nz - 1, i, j, k, z_low_mode, z_high_mode, inv_dz);
  const float mc = mod_cos[linear];
  const float ms = mod_sin[linear];
  float cos_prev, sin_prev, cos_next, sin_next;
  modulation_phase_from_omega(mod_omega[linear], t_prev, t_next, cos_prev, sin_prev, cos_next, sin_next);
  const float m_prev = modulation_factor(mc, ms, cos_prev, sin_prev);
  const float inv_next = 1.0f / fmaxf(modulation_factor(mc, ms, cos_next, sin_next), 1.0e-6f);
  ex[linear] = ex[linear] * (decay[linear] * m_prev * inv_next) + (curl_coeff[linear] * inv_next) * (d_y - d_z);
}

__global__ void update_electric_ey_modulated_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hz,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ mod_cos,
    const float* __restrict__ mod_sin,
    const float* __restrict__ mod_omega,
    float t_prev,
    float t_next,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dz,
    int x_low_mode,
    int x_high_mode,
    int z_low_mode,
    int z_high_mode,
    float* __restrict__ ey) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(i, nx, x_low_mode, x_high_mode) || boundary_pec(k, nz, z_low_mode, z_high_mode)) {
    ey[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(i, nx, x_low_mode, x_high_mode) ||
      boundary_inactive(k, nz, z_low_mode, z_high_mode)) {
    return;
  }
  const float d_z = backward_diff_axis2(hx, ny, nz - 1, i, j, k, z_low_mode, z_high_mode, inv_dz);
  const float d_x = backward_diff_axis0(hz, nx, ny, nz, i, j, k, x_low_mode, x_high_mode, inv_dx);
  const float mc = mod_cos[linear];
  const float ms = mod_sin[linear];
  float cos_prev, sin_prev, cos_next, sin_next;
  modulation_phase_from_omega(mod_omega[linear], t_prev, t_next, cos_prev, sin_prev, cos_next, sin_next);
  const float m_prev = modulation_factor(mc, ms, cos_prev, sin_prev);
  const float inv_next = 1.0f / fmaxf(modulation_factor(mc, ms, cos_next, sin_next), 1.0e-6f);
  ey[linear] = ey[linear] * (decay[linear] * m_prev * inv_next) + (curl_coeff[linear] * inv_next) * (d_z - d_x);
}

__global__ void update_electric_ez_modulated_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hy,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ mod_cos,
    const float* __restrict__ mod_sin,
    const float* __restrict__ mod_omega,
    float t_prev,
    float t_next,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy,
    int x_low_mode,
    int x_high_mode,
    int y_low_mode,
    int y_high_mode,
    float* __restrict__ ez) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(i, nx, x_low_mode, x_high_mode) || boundary_pec(j, ny, y_low_mode, y_high_mode)) {
    ez[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(i, nx, x_low_mode, x_high_mode) ||
      boundary_inactive(j, ny, y_low_mode, y_high_mode)) {
    return;
  }
  const float d_x = backward_diff_axis0(hy, nx, ny, nz, i, j, k, x_low_mode, x_high_mode, inv_dx);
  const float d_y = backward_diff_axis1(hx, ny - 1, nz, i, j, k, y_low_mode, y_high_mode, inv_dy);
  const float mc = mod_cos[linear];
  const float ms = mod_sin[linear];
  float cos_prev, sin_prev, cos_next, sin_next;
  modulation_phase_from_omega(mod_omega[linear], t_prev, t_next, cos_prev, sin_prev, cos_next, sin_next);
  const float m_prev = modulation_factor(mc, ms, cos_prev, sin_prev);
  const float inv_next = 1.0f / fmaxf(modulation_factor(mc, ms, cos_next, sin_next), 1.0e-6f);
  ez[linear] = ez[linear] * (decay[linear] * m_prev * inv_next) + (curl_coeff[linear] * inv_next) * (d_x - d_y);
}

__global__ void update_electric_ex_bloch_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hy_real,
    const float* __restrict__ hy_imag,
    const float* __restrict__ hz_real,
    const float* __restrict__ hz_imag,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    float phase_cos_y,
    float phase_sin_y,
    float phase_cos_z,
    float phase_sin_z,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
    float* __restrict__ ex_real,
    float* __restrict__ ex_imag) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const ComplexValue d_y = bloch_backward_diff_axis1(
      hz_real, hz_imag, ny, ny - 1, nz, i, j, k, phase_cos_y, phase_sin_y, inv_dy);
  const ComplexValue d_z = bloch_backward_diff_axis2(
      hy_real, hy_imag, nz, ny, nz - 1, i, j, k, phase_cos_z, phase_sin_z, inv_dz);
  ex_real[linear] = ex_real[linear] * decay[linear] + curl_coeff[linear] * (d_y.real - d_z.real);
  ex_imag[linear] = ex_imag[linear] * decay[linear] + curl_coeff[linear] * (d_y.imag - d_z.imag);
}

__global__ void update_electric_ey_bloch_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx_real,
    const float* __restrict__ hx_imag,
    const float* __restrict__ hz_real,
    const float* __restrict__ hz_imag,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    float phase_cos_x,
    float phase_sin_x,
    float phase_cos_z,
    float phase_sin_z,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dz,
    float* __restrict__ ey_real,
    float* __restrict__ ey_imag) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const ComplexValue d_z = bloch_backward_diff_axis2(
      hx_real, hx_imag, nz, ny, nz - 1, i, j, k, phase_cos_z, phase_sin_z, inv_dz);
  const ComplexValue d_x = bloch_backward_diff_axis0(
      hz_real, hz_imag, nx, nx - 1, ny, nz, i, j, k, phase_cos_x, phase_sin_x, inv_dx);
  ey_real[linear] = ey_real[linear] * decay[linear] + curl_coeff[linear] * (d_z.real - d_x.real);
  ey_imag[linear] = ey_imag[linear] * decay[linear] + curl_coeff[linear] * (d_z.imag - d_x.imag);
}

__global__ void update_electric_ez_bloch_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx_real,
    const float* __restrict__ hx_imag,
    const float* __restrict__ hy_real,
    const float* __restrict__ hy_imag,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    float phase_cos_x,
    float phase_sin_x,
    float phase_cos_y,
    float phase_sin_y,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy,
    float* __restrict__ ez_real,
    float* __restrict__ ez_imag) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const ComplexValue d_x = bloch_backward_diff_axis0(
      hy_real, hy_imag, nx, nx - 1, ny, nz, i, j, k, phase_cos_x, phase_sin_x, inv_dx);
  const ComplexValue d_y = bloch_backward_diff_axis1(
      hx_real, hx_imag, ny, ny - 1, nz, i, j, k, phase_cos_y, phase_sin_y, inv_dy);
  ez_real[linear] = ez_real[linear] * decay[linear] + curl_coeff[linear] * (d_x.real - d_y.real);
  ez_imag[linear] = ez_imag[linear] * decay[linear] + curl_coeff[linear] * (d_x.imag - d_y.imag);
}

__global__ void update_electric_ex_cpml_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hy,
    const float* __restrict__ hz,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ b_y,
    const float* __restrict__ c_y,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ b_z,
    const float* __restrict__ c_z,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
    int y_low_mode,
    int y_high_mode,
    int z_low_mode,
    int z_high_mode,
    float* __restrict__ psi_y,
    float* __restrict__ psi_z,
    float* __restrict__ ex) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(j, ny, y_low_mode, y_high_mode) || boundary_pec(k, nz, z_low_mode, z_high_mode)) {
    ex[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(j, ny, y_low_mode, y_high_mode) ||
      boundary_inactive(k, nz, z_low_mode, z_high_mode)) {
    return;
  }
  const float d_y = backward_diff_axis1(hz, ny - 1, nz, i, j, k, y_low_mode, y_high_mode, inv_dy);
  const float d_z = backward_diff_axis2(hy, ny, nz - 1, i, j, k, z_low_mode, z_high_mode, inv_dz);
  const float psi_y_value = b_y[j] * psi_y[linear] + c_y[j] * d_y;
  const float psi_z_value = b_z[k] * psi_z[linear] + c_z[k] * d_z;
  psi_y[linear] = psi_y_value;
  psi_z[linear] = psi_z_value;
  const float curl = d_y * inv_kappa_y[j] + psi_y_value - d_z * inv_kappa_z[k] - psi_z_value;
  ex[linear] = ex[linear] * decay[linear] + curl_coeff[linear] * curl;
}

__global__ void update_electric_ex_cpml_interior_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hy,
    const float* __restrict__ hz,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ b_y,
    const float* __restrict__ c_y,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ b_z,
    const float* __restrict__ c_z,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
    float* __restrict__ psi_y,
    float* __restrict__ psi_z,
    float* __restrict__ ex) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j + 1 >= ny || k + 1 >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const long long hz_current = offset3d(i, j, k, ny - 1, nz);
  const long long hz_previous = offset3d(i, j - 1, k, ny - 1, nz);
  const long long hy_current = offset3d(i, j, k, ny, nz - 1);
  const long long hy_previous = offset3d(i, j, k - 1, ny, nz - 1);
  const float d_y = (hz[hz_current] - hz[hz_previous]) * inv_dy[j];
  const float d_z = (hy[hy_current] - hy[hy_previous]) * inv_dz[k];
  const float psi_y_value = b_y[j] * psi_y[linear] + c_y[j] * d_y;
  const float psi_z_value = b_z[k] * psi_z[linear] + c_z[k] * d_z;
  psi_y[linear] = psi_y_value;
  psi_z[linear] = psi_z_value;
  const float curl = d_y * inv_kappa_y[j] + psi_y_value - d_z * inv_kappa_z[k] - psi_z_value;
  ex[linear] = ex[linear] * decay[linear] + curl_coeff[linear] * curl;
}

__global__ void update_electric_ey_cpml_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hz,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ b_x,
    const float* __restrict__ c_x,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ b_z,
    const float* __restrict__ c_z,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dz,
    int x_low_mode,
    int x_high_mode,
    int z_low_mode,
    int z_high_mode,
    float* __restrict__ psi_x,
    float* __restrict__ psi_z,
    float* __restrict__ ey) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(i, nx, x_low_mode, x_high_mode) || boundary_pec(k, nz, z_low_mode, z_high_mode)) {
    ey[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(i, nx, x_low_mode, x_high_mode) ||
      boundary_inactive(k, nz, z_low_mode, z_high_mode)) {
    return;
  }
  const float d_z = backward_diff_axis2(hx, ny, nz - 1, i, j, k, z_low_mode, z_high_mode, inv_dz);
  const float d_x = backward_diff_axis0(hz, nx, ny, nz, i, j, k, x_low_mode, x_high_mode, inv_dx);
  const float psi_x_value = b_x[i] * psi_x[linear] + c_x[i] * d_x;
  const float psi_z_value = b_z[k] * psi_z[linear] + c_z[k] * d_z;
  psi_x[linear] = psi_x_value;
  psi_z[linear] = psi_z_value;
  const float curl = d_z * inv_kappa_z[k] + psi_z_value - d_x * inv_kappa_x[i] - psi_x_value;
  ey[linear] = ey[linear] * decay[linear] + curl_coeff[linear] * curl;
}

__global__ void update_electric_ey_cpml_interior_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hz,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ b_x,
    const float* __restrict__ c_x,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ b_z,
    const float* __restrict__ c_z,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dz,
    float* __restrict__ psi_x,
    float* __restrict__ psi_z,
    float* __restrict__ ey) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  if (i + 1 >= nx || j >= ny || k + 1 >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const long long hx_current = offset3d(i, j, k, ny, nz - 1);
  const long long hx_previous = offset3d(i, j, k - 1, ny, nz - 1);
  const long long hz_current = offset3d(i, j, k, ny, nz);
  const long long hz_previous = offset3d(i - 1, j, k, ny, nz);
  const float d_z = (hx[hx_current] - hx[hx_previous]) * inv_dz[k];
  const float d_x = (hz[hz_current] - hz[hz_previous]) * inv_dx[i];
  const float psi_x_value = b_x[i] * psi_x[linear] + c_x[i] * d_x;
  const float psi_z_value = b_z[k] * psi_z[linear] + c_z[k] * d_z;
  psi_x[linear] = psi_x_value;
  psi_z[linear] = psi_z_value;
  const float curl = d_z * inv_kappa_z[k] + psi_z_value - d_x * inv_kappa_x[i] - psi_x_value;
  ey[linear] = ey[linear] * decay[linear] + curl_coeff[linear] * curl;
}

__global__ void update_electric_ez_cpml_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hy,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ b_x,
    const float* __restrict__ c_x,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ b_y,
    const float* __restrict__ c_y,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy,
    int x_low_mode,
    int x_high_mode,
    int y_low_mode,
    int y_high_mode,
    float* __restrict__ psi_x,
    float* __restrict__ psi_y,
    float* __restrict__ ez) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(i, nx, x_low_mode, x_high_mode) || boundary_pec(j, ny, y_low_mode, y_high_mode)) {
    ez[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(i, nx, x_low_mode, x_high_mode) ||
      boundary_inactive(j, ny, y_low_mode, y_high_mode)) {
    return;
  }
  const float d_x = backward_diff_axis0(hy, nx, ny, nz, i, j, k, x_low_mode, x_high_mode, inv_dx);
  const float d_y = backward_diff_axis1(hx, ny - 1, nz, i, j, k, y_low_mode, y_high_mode, inv_dy);
  const float psi_x_value = b_x[i] * psi_x[linear] + c_x[i] * d_x;
  const float psi_y_value = b_y[j] * psi_y[linear] + c_y[j] * d_y;
  psi_x[linear] = psi_x_value;
  psi_y[linear] = psi_y_value;
  const float curl = d_x * inv_kappa_x[i] + psi_x_value - d_y * inv_kappa_y[j] - psi_y_value;
  ez[linear] = ez[linear] * decay[linear] + curl_coeff[linear] * curl;
}

__global__ void update_electric_ez_cpml_interior_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hy,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ b_x,
    const float* __restrict__ c_x,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ b_y,
    const float* __restrict__ c_y,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy,
    float* __restrict__ psi_x,
    float* __restrict__ psi_y,
    float* __restrict__ ez) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  if (i + 1 >= nx || j + 1 >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const long long hy_current = offset3d(i, j, k, ny, nz);
  const long long hy_previous = offset3d(i - 1, j, k, ny, nz);
  const long long hx_current = offset3d(i, j, k, ny - 1, nz);
  const long long hx_previous = offset3d(i, j - 1, k, ny - 1, nz);
  const float d_x = (hy[hy_current] - hy[hy_previous]) * inv_dx[i];
  const float d_y = (hx[hx_current] - hx[hx_previous]) * inv_dy[j];
  const float psi_x_value = b_x[i] * psi_x[linear] + c_x[i] * d_x;
  const float psi_y_value = b_y[j] * psi_y[linear] + c_y[j] * d_y;
  psi_x[linear] = psi_x_value;
  psi_y[linear] = psi_y_value;
  const float curl = d_x * inv_kappa_x[i] + psi_x_value - d_y * inv_kappa_y[j] - psi_y_value;
  ez[linear] = ez[linear] * decay[linear] + curl_coeff[linear] * curl;
}

// Dense-CPML variants of the modulated electric update. Identical psi/kappa
// handling to the base CPML kernels with the modulation factor applied to the
// decay/curl pair (see the modulated standard kernels above for the scheme).
__global__ void update_electric_ex_cpml_modulated_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hy,
    const float* __restrict__ hz,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ mod_cos,
    const float* __restrict__ mod_sin,
    const float* __restrict__ mod_omega,
    float t_prev,
    float t_next,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ b_y,
    const float* __restrict__ c_y,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ b_z,
    const float* __restrict__ c_z,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
    int y_low_mode,
    int y_high_mode,
    int z_low_mode,
    int z_high_mode,
    float* __restrict__ psi_y,
    float* __restrict__ psi_z,
    float* __restrict__ ex) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(j, ny, y_low_mode, y_high_mode) || boundary_pec(k, nz, z_low_mode, z_high_mode)) {
    ex[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(j, ny, y_low_mode, y_high_mode) ||
      boundary_inactive(k, nz, z_low_mode, z_high_mode)) {
    return;
  }
  const float d_y = backward_diff_axis1(hz, ny - 1, nz, i, j, k, y_low_mode, y_high_mode, inv_dy);
  const float d_z = backward_diff_axis2(hy, ny, nz - 1, i, j, k, z_low_mode, z_high_mode, inv_dz);
  const float psi_y_value = b_y[j] * psi_y[linear] + c_y[j] * d_y;
  const float psi_z_value = b_z[k] * psi_z[linear] + c_z[k] * d_z;
  psi_y[linear] = psi_y_value;
  psi_z[linear] = psi_z_value;
  const float curl = d_y * inv_kappa_y[j] + psi_y_value - d_z * inv_kappa_z[k] - psi_z_value;
  const float mc = mod_cos[linear];
  const float ms = mod_sin[linear];
  float cos_prev, sin_prev, cos_next, sin_next;
  modulation_phase_from_omega(mod_omega[linear], t_prev, t_next, cos_prev, sin_prev, cos_next, sin_next);
  const float m_prev = modulation_factor(mc, ms, cos_prev, sin_prev);
  const float inv_next = 1.0f / fmaxf(modulation_factor(mc, ms, cos_next, sin_next), 1.0e-6f);
  ex[linear] = ex[linear] * (decay[linear] * m_prev * inv_next) + (curl_coeff[linear] * inv_next) * curl;
}

__global__ void update_electric_ey_cpml_modulated_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hz,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ mod_cos,
    const float* __restrict__ mod_sin,
    const float* __restrict__ mod_omega,
    float t_prev,
    float t_next,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ b_x,
    const float* __restrict__ c_x,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ b_z,
    const float* __restrict__ c_z,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dz,
    int x_low_mode,
    int x_high_mode,
    int z_low_mode,
    int z_high_mode,
    float* __restrict__ psi_x,
    float* __restrict__ psi_z,
    float* __restrict__ ey) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(i, nx, x_low_mode, x_high_mode) || boundary_pec(k, nz, z_low_mode, z_high_mode)) {
    ey[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(i, nx, x_low_mode, x_high_mode) ||
      boundary_inactive(k, nz, z_low_mode, z_high_mode)) {
    return;
  }
  const float d_z = backward_diff_axis2(hx, ny, nz - 1, i, j, k, z_low_mode, z_high_mode, inv_dz);
  const float d_x = backward_diff_axis0(hz, nx, ny, nz, i, j, k, x_low_mode, x_high_mode, inv_dx);
  const float psi_x_value = b_x[i] * psi_x[linear] + c_x[i] * d_x;
  const float psi_z_value = b_z[k] * psi_z[linear] + c_z[k] * d_z;
  psi_x[linear] = psi_x_value;
  psi_z[linear] = psi_z_value;
  const float curl = d_z * inv_kappa_z[k] + psi_z_value - d_x * inv_kappa_x[i] - psi_x_value;
  const float mc = mod_cos[linear];
  const float ms = mod_sin[linear];
  float cos_prev, sin_prev, cos_next, sin_next;
  modulation_phase_from_omega(mod_omega[linear], t_prev, t_next, cos_prev, sin_prev, cos_next, sin_next);
  const float m_prev = modulation_factor(mc, ms, cos_prev, sin_prev);
  const float inv_next = 1.0f / fmaxf(modulation_factor(mc, ms, cos_next, sin_next), 1.0e-6f);
  ey[linear] = ey[linear] * (decay[linear] * m_prev * inv_next) + (curl_coeff[linear] * inv_next) * curl;
}

__global__ void update_electric_ez_cpml_modulated_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hy,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ mod_cos,
    const float* __restrict__ mod_sin,
    const float* __restrict__ mod_omega,
    float t_prev,
    float t_next,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ b_x,
    const float* __restrict__ c_x,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ b_y,
    const float* __restrict__ c_y,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy,
    int x_low_mode,
    int x_high_mode,
    int y_low_mode,
    int y_high_mode,
    float* __restrict__ psi_x,
    float* __restrict__ psi_y,
    float* __restrict__ ez) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(i, nx, x_low_mode, x_high_mode) || boundary_pec(j, ny, y_low_mode, y_high_mode)) {
    ez[linear] = 0.0f;
    return;
  }
  if (boundary_inactive(i, nx, x_low_mode, x_high_mode) ||
      boundary_inactive(j, ny, y_low_mode, y_high_mode)) {
    return;
  }
  const float d_x = backward_diff_axis0(hy, nx, ny, nz, i, j, k, x_low_mode, x_high_mode, inv_dx);
  const float d_y = backward_diff_axis1(hx, ny - 1, nz, i, j, k, y_low_mode, y_high_mode, inv_dy);
  const float psi_x_value = b_x[i] * psi_x[linear] + c_x[i] * d_x;
  const float psi_y_value = b_y[j] * psi_y[linear] + c_y[j] * d_y;
  psi_x[linear] = psi_x_value;
  psi_y[linear] = psi_y_value;
  const float curl = d_x * inv_kappa_x[i] + psi_x_value - d_y * inv_kappa_y[j] - psi_y_value;
  const float mc = mod_cos[linear];
  const float ms = mod_sin[linear];
  float cos_prev, sin_prev, cos_next, sin_next;
  modulation_phase_from_omega(mod_omega[linear], t_prev, t_next, cos_prev, sin_prev, cos_next, sin_next);
  const float m_prev = modulation_factor(mc, ms, cos_prev, sin_prev);
  const float inv_next = 1.0f / fmaxf(modulation_factor(mc, ms, cos_next, sin_next), 1.0e-6f);
  ez[linear] = ez[linear] * (decay[linear] * m_prev * inv_next) + (curl_coeff[linear] * inv_next) * curl;
}

template <int Axis>
__device__ __forceinline__ float update_compact_electric_psi(
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

template <bool UniformDecay, bool UniformCurl>
__global__ void update_electric_ex_cpml_compressed_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hy,
    const float* __restrict__ hz,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    float decay_value,
    float curl_value,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ b_y,
    const float* __restrict__ c_y,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ b_z,
    const float* __restrict__ c_z,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
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
    float* __restrict__ psi_y,
    float* __restrict__ psi_z,
    float* __restrict__ ex) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(j, ny, y_low_mode, y_high_mode) || boundary_pec(k, nz, z_low_mode, z_high_mode)) {
    ex[linear] = 0.0f;
    return;
  }
  const bool active = !(boundary_inactive(j, ny, y_low_mode, y_high_mode) ||
                        boundary_inactive(k, nz, z_low_mode, z_high_mode));
  if (!active) {
    return;
  }
  const float d_y = backward_diff_axis1(hz, ny - 1, nz, i, j, k, y_low_mode, y_high_mode, inv_dy);
  const float d_z = backward_diff_axis2(hy, ny, nz - 1, i, j, k, z_low_mode, z_high_mode, inv_dz);
  const float psi_y_value = update_compact_electric_psi<1>(
      psi_y, b_y, c_y, i, j, k, ny, nz, j, y_low_length, y_high_start, y_high_length, d_y);
  const float psi_z_value = update_compact_electric_psi<2>(
      psi_z, b_z, c_z, i, j, k, ny, nz, k, z_low_length, z_high_start, z_high_length, d_z);
  const float curl = d_y * inv_kappa_y[j] + psi_y_value - d_z * inv_kappa_z[k] - psi_z_value;
  const float decay_factor = UniformDecay ? decay_value : decay[linear];
  const float curl_factor = UniformCurl ? curl_value : curl_coeff[linear];
  ex[linear] = ex[linear] * decay_factor + curl_factor * curl;
}

template <bool UniformDecay, bool UniformCurl>
__global__ void update_electric_ey_cpml_compressed_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hz,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    float decay_value,
    float curl_value,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ b_x,
    const float* __restrict__ c_x,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ b_z,
    const float* __restrict__ c_z,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dz,
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
    float* __restrict__ psi_x,
    float* __restrict__ psi_z,
    float* __restrict__ ey) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(i, nx, x_low_mode, x_high_mode) || boundary_pec(k, nz, z_low_mode, z_high_mode)) {
    ey[linear] = 0.0f;
    return;
  }
  const bool active = !(boundary_inactive(i, nx, x_low_mode, x_high_mode) ||
                        boundary_inactive(k, nz, z_low_mode, z_high_mode));
  if (!active) {
    return;
  }
  const float d_z = backward_diff_axis2(hx, ny, nz - 1, i, j, k, z_low_mode, z_high_mode, inv_dz);
  const float d_x = backward_diff_axis0(hz, nx, ny, nz, i, j, k, x_low_mode, x_high_mode, inv_dx);
  const float psi_x_value = update_compact_electric_psi<0>(
      psi_x, b_x, c_x, i, j, k, ny, nz, i, x_low_length, x_high_start, x_high_length, d_x);
  const float psi_z_value = update_compact_electric_psi<2>(
      psi_z, b_z, c_z, i, j, k, ny, nz, k, z_low_length, z_high_start, z_high_length, d_z);
  const float curl = d_z * inv_kappa_z[k] + psi_z_value - d_x * inv_kappa_x[i] - psi_x_value;
  const float decay_factor = UniformDecay ? decay_value : decay[linear];
  const float curl_factor = UniformCurl ? curl_value : curl_coeff[linear];
  ey[linear] = ey[linear] * decay_factor + curl_factor * curl;
}

template <bool UniformDecay, bool UniformCurl>
__global__ void update_electric_ez_cpml_compressed_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hy,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    float decay_value,
    float curl_value,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ b_x,
    const float* __restrict__ c_x,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ b_y,
    const float* __restrict__ c_y,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy,
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
    float* __restrict__ psi_x,
    float* __restrict__ psi_y,
    float* __restrict__ ez) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(i, nx, x_low_mode, x_high_mode) || boundary_pec(j, ny, y_low_mode, y_high_mode)) {
    ez[linear] = 0.0f;
    return;
  }
  const bool active = !(boundary_inactive(i, nx, x_low_mode, x_high_mode) ||
                        boundary_inactive(j, ny, y_low_mode, y_high_mode));
  if (!active) {
    return;
  }
  const float d_x = backward_diff_axis0(hy, nx, ny, nz, i, j, k, x_low_mode, x_high_mode, inv_dx);
  const float d_y = backward_diff_axis1(hx, ny - 1, nz, i, j, k, y_low_mode, y_high_mode, inv_dy);
  const float psi_x_value = update_compact_electric_psi<0>(
      psi_x, b_x, c_x, i, j, k, ny, nz, i, x_low_length, x_high_start, x_high_length, d_x);
  const float psi_y_value = update_compact_electric_psi<1>(
      psi_y, b_y, c_y, i, j, k, ny, nz, j, y_low_length, y_high_start, y_high_length, d_y);
  const float curl = d_x * inv_kappa_x[i] + psi_x_value - d_y * inv_kappa_y[j] - psi_y_value;
  const float decay_factor = UniformDecay ? decay_value : decay[linear];
  const float curl_factor = UniformCurl ? curl_value : curl_coeff[linear];
  ez[linear] = ez[linear] * decay_factor + curl_factor * curl;
}

// Compressed-(slab-)CPML variants of the modulated electric update. The compact
// psi bookkeeping is identical to the plain compressed CPML kernels; the
// modulation factor is applied to the decay/curl pair exactly as in the dense
// modulated CPML kernels. The modulated path always carries per-cell decay/curl
// coefficients, so these kernels do not take the uniform-scalar fast path.
__global__ void update_electric_ex_cpml_modulated_compressed_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hy,
    const float* __restrict__ hz,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ mod_cos,
    const float* __restrict__ mod_sin,
    const float* __restrict__ mod_omega,
    float t_prev,
    float t_next,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ b_y,
    const float* __restrict__ c_y,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ b_z,
    const float* __restrict__ c_z,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
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
    float* __restrict__ psi_y,
    float* __restrict__ psi_z,
    float* __restrict__ ex) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(j, ny, y_low_mode, y_high_mode) || boundary_pec(k, nz, z_low_mode, z_high_mode)) {
    ex[linear] = 0.0f;
    return;
  }
  const bool active = !(boundary_inactive(j, ny, y_low_mode, y_high_mode) ||
                        boundary_inactive(k, nz, z_low_mode, z_high_mode));
  if (!active) {
    return;
  }
  const float d_y = backward_diff_axis1(hz, ny - 1, nz, i, j, k, y_low_mode, y_high_mode, inv_dy);
  const float d_z = backward_diff_axis2(hy, ny, nz - 1, i, j, k, z_low_mode, z_high_mode, inv_dz);
  const float psi_y_value = update_compact_electric_psi<1>(
      psi_y, b_y, c_y, i, j, k, ny, nz, j, y_low_length, y_high_start, y_high_length, d_y);
  const float psi_z_value = update_compact_electric_psi<2>(
      psi_z, b_z, c_z, i, j, k, ny, nz, k, z_low_length, z_high_start, z_high_length, d_z);
  const float curl = d_y * inv_kappa_y[j] + psi_y_value - d_z * inv_kappa_z[k] - psi_z_value;
  const float mc = mod_cos[linear];
  const float ms = mod_sin[linear];
  float cos_prev, sin_prev, cos_next, sin_next;
  modulation_phase_from_omega(mod_omega[linear], t_prev, t_next, cos_prev, sin_prev, cos_next, sin_next);
  const float m_prev = modulation_factor(mc, ms, cos_prev, sin_prev);
  const float inv_next = 1.0f / fmaxf(modulation_factor(mc, ms, cos_next, sin_next), 1.0e-6f);
  ex[linear] = ex[linear] * (decay[linear] * m_prev * inv_next) + (curl_coeff[linear] * inv_next) * curl;
}

__global__ void update_electric_ey_cpml_modulated_compressed_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hz,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ mod_cos,
    const float* __restrict__ mod_sin,
    const float* __restrict__ mod_omega,
    float t_prev,
    float t_next,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ b_x,
    const float* __restrict__ c_x,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ b_z,
    const float* __restrict__ c_z,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dz,
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
    float* __restrict__ psi_x,
    float* __restrict__ psi_z,
    float* __restrict__ ey) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(i, nx, x_low_mode, x_high_mode) || boundary_pec(k, nz, z_low_mode, z_high_mode)) {
    ey[linear] = 0.0f;
    return;
  }
  const bool active = !(boundary_inactive(i, nx, x_low_mode, x_high_mode) ||
                        boundary_inactive(k, nz, z_low_mode, z_high_mode));
  if (!active) {
    return;
  }
  const float d_z = backward_diff_axis2(hx, ny, nz - 1, i, j, k, z_low_mode, z_high_mode, inv_dz);
  const float d_x = backward_diff_axis0(hz, nx, ny, nz, i, j, k, x_low_mode, x_high_mode, inv_dx);
  const float psi_x_value = update_compact_electric_psi<0>(
      psi_x, b_x, c_x, i, j, k, ny, nz, i, x_low_length, x_high_start, x_high_length, d_x);
  const float psi_z_value = update_compact_electric_psi<2>(
      psi_z, b_z, c_z, i, j, k, ny, nz, k, z_low_length, z_high_start, z_high_length, d_z);
  const float curl = d_z * inv_kappa_z[k] + psi_z_value - d_x * inv_kappa_x[i] - psi_x_value;
  const float mc = mod_cos[linear];
  const float ms = mod_sin[linear];
  float cos_prev, sin_prev, cos_next, sin_next;
  modulation_phase_from_omega(mod_omega[linear], t_prev, t_next, cos_prev, sin_prev, cos_next, sin_next);
  const float m_prev = modulation_factor(mc, ms, cos_prev, sin_prev);
  const float inv_next = 1.0f / fmaxf(modulation_factor(mc, ms, cos_next, sin_next), 1.0e-6f);
  ey[linear] = ey[linear] * (decay[linear] * m_prev * inv_next) + (curl_coeff[linear] * inv_next) * curl;
}

__global__ void update_electric_ez_cpml_modulated_compressed_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hy,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ mod_cos,
    const float* __restrict__ mod_sin,
    const float* __restrict__ mod_omega,
    float t_prev,
    float t_next,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ b_x,
    const float* __restrict__ c_x,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ b_y,
    const float* __restrict__ c_y,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy,
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
    float* __restrict__ psi_x,
    float* __restrict__ psi_y,
    float* __restrict__ ez) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  if (boundary_pec(i, nx, x_low_mode, x_high_mode) || boundary_pec(j, ny, y_low_mode, y_high_mode)) {
    ez[linear] = 0.0f;
    return;
  }
  const bool active = !(boundary_inactive(i, nx, x_low_mode, x_high_mode) ||
                        boundary_inactive(j, ny, y_low_mode, y_high_mode));
  if (!active) {
    return;
  }
  const float d_x = backward_diff_axis0(hy, nx, ny, nz, i, j, k, x_low_mode, x_high_mode, inv_dx);
  const float d_y = backward_diff_axis1(hx, ny - 1, nz, i, j, k, y_low_mode, y_high_mode, inv_dy);
  const float psi_x_value = update_compact_electric_psi<0>(
      psi_x, b_x, c_x, i, j, k, ny, nz, i, x_low_length, x_high_start, x_high_length, d_x);
  const float psi_y_value = update_compact_electric_psi<1>(
      psi_y, b_y, c_y, i, j, k, ny, nz, j, y_low_length, y_high_start, y_high_length, d_y);
  const float curl = d_x * inv_kappa_x[i] + psi_x_value - d_y * inv_kappa_y[j] - psi_y_value;
  const float mc = mod_cos[linear];
  const float ms = mod_sin[linear];
  float cos_prev, sin_prev, cos_next, sin_next;
  modulation_phase_from_omega(mod_omega[linear], t_prev, t_next, cos_prev, sin_prev, cos_next, sin_next);
  const float m_prev = modulation_factor(mc, ms, cos_prev, sin_prev);
  const float inv_next = 1.0f / fmaxf(modulation_factor(mc, ms, cos_next, sin_next), 1.0e-6f);
  ez[linear] = ez[linear] * (decay[linear] * m_prev * inv_next) + (curl_coeff[linear] * inv_next) * curl;
}

void check_electric_inputs(
    const torch::stable::Tensor& field,
    const torch::stable::Tensor& first,
    const torch::stable::Tensor& second,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
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
  STD_TORCH_CHECK(field.dim() == 3, name, " must be rank 3");
  STD_TORCH_CHECK(first.dim() == 3, "first must be rank 3");
  STD_TORCH_CHECK(second.dim() == 3, "second must be rank 3");
  STD_TORCH_CHECK(decay.sizes().equals(field.sizes()), "decay must match field shape");
  STD_TORCH_CHECK(curl.sizes().equals(field.sizes()), "curl must match field shape");
}

void check_modulation_fields(
    const torch::stable::Tensor& field,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega) {
  check_float32_tensor(mod_cos, "mod_cos");
  check_float32_tensor(mod_sin, "mod_sin");
  check_float32_tensor(mod_omega, "mod_omega");
  check_same_cuda_device(field, mod_cos, "mod_cos");
  check_same_cuda_device(field, mod_sin, "mod_sin");
  check_same_cuda_device(field, mod_omega, "mod_omega");
  check_contiguous_tensor(mod_cos, "mod_cos");
  check_contiguous_tensor(mod_sin, "mod_sin");
  check_contiguous_tensor(mod_omega, "mod_omega");
  STD_TORCH_CHECK(mod_cos.sizes().equals(field.sizes()), "mod_cos must match field shape");
  STD_TORCH_CHECK(mod_sin.sizes().equals(field.sizes()), "mod_sin must match field shape");
  STD_TORCH_CHECK(mod_omega.sizes().equals(field.sizes()), "mod_omega must match field shape");
}

void check_rank3_shape(
    const torch::stable::Tensor& tensor,
    const char* name,
    int64_t x,
    int64_t y,
    int64_t z) {
  STD_TORCH_CHECK(tensor.dim() == 3, name, " must be rank 3");
  STD_TORCH_CHECK(
      tensor.size(0) == x && tensor.size(1) == y && tensor.size(2) == z,
      name,
      " has an incompatible Yee-grid shape");
}

void check_electric_bloch_inputs(
    const torch::stable::Tensor& field_real,
    const torch::stable::Tensor& field_imag,
    const torch::stable::Tensor& first_real,
    const torch::stable::Tensor& first_imag,
    const torch::stable::Tensor& second_real,
    const torch::stable::Tensor& second_imag,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const char* name) {
  check_electric_inputs(field_real, first_real, second_real, decay, curl, name);
  check_float32_tensor(field_imag, "field_imag");
  check_float32_tensor(first_imag, "first_imag");
  check_float32_tensor(second_imag, "second_imag");
  check_same_cuda_device(field_real, field_imag, "field_imag");
  check_same_cuda_device(field_real, first_imag, "first_imag");
  check_same_cuda_device(field_real, second_imag, "second_imag");
  check_contiguous_tensor(field_imag, "field_imag");
  check_contiguous_tensor(first_imag, "first_imag");
  check_contiguous_tensor(second_imag, "second_imag");
  STD_TORCH_CHECK(field_imag.sizes().equals(field_real.sizes()), "field imaginary tensor must match real shape");
  STD_TORCH_CHECK(first_imag.sizes().equals(first_real.sizes()), "first imaginary tensor must match real shape");
  STD_TORCH_CHECK(second_imag.sizes().equals(second_real.sizes()), "second imaginary tensor must match real shape");
  STD_TORCH_CHECK(first_real.dim() == 3, "first Bloch source tensor must be rank 3");
  STD_TORCH_CHECK(second_real.dim() == 3, "second Bloch source tensor must be rank 3");
}

void check_vector_input(const torch::stable::Tensor& tensor, int64_t length, const char* name) {
  check_float32_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  STD_TORCH_CHECK(tensor.dim() == 1, name, " must be rank 1");
  STD_TORCH_CHECK(tensor.size(0) == length, name, " length must match CPML field axis");
}

void check_spacing_vector(
    const torch::stable::Tensor& field,
    const torch::stable::Tensor& inv_delta,
    int64_t axis,
    const char* name) {
  check_vector_input(inv_delta, field.size(axis), name);
  check_same_cuda_device(field, inv_delta, name);
}

void check_electric_cpml_inputs(
    const torch::stable::Tensor& field,
    const torch::stable::Tensor& first,
    const torch::stable::Tensor& second,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& psi_first,
    const torch::stable::Tensor& psi_second,
    const torch::stable::Tensor& inv_kappa_first,
    const torch::stable::Tensor& b_first,
    const torch::stable::Tensor& c_first,
    const torch::stable::Tensor& inv_kappa_second,
    const torch::stable::Tensor& b_second,
    const torch::stable::Tensor& c_second,
    int64_t first_axis,
    int64_t second_axis,
    const char* name) {
  check_electric_inputs(field, first, second, decay, curl, name);
  check_float32_tensor(psi_first, "psi_first");
  check_float32_tensor(psi_second, "psi_second");
  check_same_cuda_device(field, psi_first, "psi_first");
  check_same_cuda_device(field, psi_second, "psi_second");
  check_contiguous_tensor(psi_first, "psi_first");
  check_contiguous_tensor(psi_second, "psi_second");
  STD_TORCH_CHECK(psi_first.sizes().equals(field.sizes()), "psi_first must match field shape");
  STD_TORCH_CHECK(psi_second.sizes().equals(field.sizes()), "psi_second must match field shape");
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
    const torch::stable::Tensor& psi,
    const torch::stable::Tensor& field,
    int64_t axis,
    int64_t low_length,
    int64_t high_length,
    const char* name) {
  check_float32_tensor(psi, name);
  check_same_cuda_device(field, psi, name);
  check_contiguous_tensor(psi, name);
  STD_TORCH_CHECK(psi.dim() == 3, name, " must be rank 3");
  for (int64_t dim = 0; dim < 3; ++dim) {
    const int64_t expected = dim == axis ? low_length + high_length : field.size(dim);
    STD_TORCH_CHECK(psi.size(dim) == expected, name, " shape does not match compact CPML layout");
  }
}

void check_electric_cpml_compressed_inputs(
    const torch::stable::Tensor& field,
    const torch::stable::Tensor& first,
    const torch::stable::Tensor& second,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& psi_first,
    const torch::stable::Tensor& psi_second,
    const torch::stable::Tensor& inv_kappa_first,
    const torch::stable::Tensor& b_first,
    const torch::stable::Tensor& c_first,
    const torch::stable::Tensor& inv_kappa_second,
    const torch::stable::Tensor& b_second,
    const torch::stable::Tensor& c_second,
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
  check_same_cuda_device(field, inv_kappa_first, "inv_kappa_first");
  check_same_cuda_device(field, b_first, "b_first");
  check_same_cuda_device(field, c_first, "c_first");
  check_same_cuda_device(field, inv_kappa_second, "inv_kappa_second");
  check_same_cuda_device(field, b_second, "b_second");
  check_same_cuda_device(field, c_second, "c_second");
}

__device__ __forceinline__ bool cpml_correction_active(
    unsigned int global_index, unsigned int full_size, int low_mode, int high_mode) {
  const bool low_inactive = (global_index == 0) &&
      (low_mode == BOUNDARY_NONE || low_mode == BOUNDARY_PEC || low_mode == BOUNDARY_PML);
  const bool high_inactive = (global_index + 1 == full_size) &&
      (high_mode == BOUNDARY_NONE || high_mode == BOUNDARY_PEC || high_mode == BOUNDARY_PML);
  return !(low_inactive || high_inactive);
}

// Mixed Bloch (transverse) + standard (z) electric update for the Bloch+CPML
// path. Ex: Bloch backward diff of Hz along y, standard backward diff of Hy
// along z.
__global__ void update_electric_ex_bloch_y_standard_z_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hy_real,
    const float* __restrict__ hy_imag,
    const float* __restrict__ hz_real,
    const float* __restrict__ hz_imag,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    float phase_cos_y,
    float phase_sin_y,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
    int z_low_mode,
    int z_high_mode,
    float* __restrict__ ex_real,
    float* __restrict__ ex_imag) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const ComplexValue d_y = bloch_backward_diff_axis1(
      hz_real, hz_imag, ny, ny - 1, nz, i, j, k, phase_cos_y, phase_sin_y, inv_dy);
  const float d_z_real = backward_diff_axis2(hy_real, ny, nz - 1, i, j, k, z_low_mode, z_high_mode, inv_dz);
  const float d_z_imag = backward_diff_axis2(hy_imag, ny, nz - 1, i, j, k, z_low_mode, z_high_mode, inv_dz);
  ex_real[linear] = ex_real[linear] * decay[linear] + curl_coeff[linear] * (d_y.real - d_z_real);
  ex_imag[linear] = ex_imag[linear] * decay[linear] + curl_coeff[linear] * (d_y.imag - d_z_imag);
}

// Ey: Bloch backward diff of Hz along x, standard backward diff of Hx along z.
__global__ void update_electric_ey_bloch_x_standard_z_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx_real,
    const float* __restrict__ hx_imag,
    const float* __restrict__ hz_real,
    const float* __restrict__ hz_imag,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    float phase_cos_x,
    float phase_sin_x,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dz,
    int z_low_mode,
    int z_high_mode,
    float* __restrict__ ey_real,
    float* __restrict__ ey_imag) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const float d_z_real = backward_diff_axis2(hx_real, ny, nz - 1, i, j, k, z_low_mode, z_high_mode, inv_dz);
  const float d_z_imag = backward_diff_axis2(hx_imag, ny, nz - 1, i, j, k, z_low_mode, z_high_mode, inv_dz);
  const ComplexValue d_x = bloch_backward_diff_axis0(
      hz_real, hz_imag, nx, nx - 1, ny, nz, i, j, k, phase_cos_x, phase_sin_x, inv_dx);
  ey_real[linear] = ey_real[linear] * decay[linear] + curl_coeff[linear] * (d_z_real - d_x.real);
  ey_imag[linear] = ey_imag[linear] * decay[linear] + curl_coeff[linear] * (d_z_imag - d_x.imag);
}

// CPML psi + kappa correction along z applied over a region narrowed along z,
// for the Bloch+CPML mixed electric update. Region-local (x, y, lz) map to
// global (offset_i + x, offset_j + y, offset_k + lz); psi/curl/field are
// region-shaped, the CPML coefficient arrays are indexed by global z.
__global__ void apply_electric_ex_cpml_z_correction_kernel(
    unsigned int sx,
    unsigned int sy,
    unsigned int sz,
    unsigned int hy_ny,
    unsigned int hy_nz,
    const float* __restrict__ hy,
    const float* __restrict__ curl,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ b_z,
    const float* __restrict__ c_z,
    const float* __restrict__ inv_dz,
    int offset_i,
    int offset_j,
    int offset_k,
    int y_low_mode,
    int y_high_mode,
    unsigned int full_size_y,
    int local_z_start,
    int local_z_stop,
    float* __restrict__ psi_z,
    float* __restrict__ ex) {
  const unsigned int lz = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int x = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= sx || y >= sy || lz >= sz) {
    return;
  }
  if (static_cast<int>(lz) < local_z_start || static_cast<int>(lz) >= local_z_stop) {
    return;
  }
  const unsigned int gy = static_cast<unsigned int>(offset_j) + y;
  if (!cpml_correction_active(gy, full_size_y, y_low_mode, y_high_mode)) {
    return;
  }
  const unsigned int gx = static_cast<unsigned int>(offset_i) + x;
  const unsigned int gz = static_cast<unsigned int>(offset_k) + lz;
  const long long region_lin = offset3d(x, y, lz, sy, sz);
  const long long hy_cur = offset3d(gx, gy, gz, hy_ny, hy_nz);
  const long long hy_prev = offset3d(gx, gy, gz - 1, hy_ny, hy_nz);
  const float derivative = (hy[hy_cur] - hy[hy_prev]) * inv_dz[gz];
  const float psi_new = b_z[gz] * psi_z[region_lin] + c_z[gz] * derivative;
  psi_z[region_lin] = psi_new;
  const float correction = derivative * (inv_kappa_z[gz] - 1.0f) + psi_new;
  ex[region_lin] = ex[region_lin] - curl[region_lin] * correction;
}

__global__ void apply_electric_ey_cpml_z_correction_kernel(
    unsigned int sx,
    unsigned int sy,
    unsigned int sz,
    unsigned int hx_ny,
    unsigned int hx_nz,
    const float* __restrict__ hx,
    const float* __restrict__ curl,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ b_z,
    const float* __restrict__ c_z,
    const float* __restrict__ inv_dz,
    int offset_i,
    int offset_j,
    int offset_k,
    int x_low_mode,
    int x_high_mode,
    unsigned int full_size_x,
    int local_z_start,
    int local_z_stop,
    float* __restrict__ psi_z,
    float* __restrict__ ey) {
  const unsigned int lz = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int x = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= sx || y >= sy || lz >= sz) {
    return;
  }
  if (static_cast<int>(lz) < local_z_start || static_cast<int>(lz) >= local_z_stop) {
    return;
  }
  const unsigned int gx = static_cast<unsigned int>(offset_i) + x;
  if (!cpml_correction_active(gx, full_size_x, x_low_mode, x_high_mode)) {
    return;
  }
  const unsigned int gy = static_cast<unsigned int>(offset_j) + y;
  const unsigned int gz = static_cast<unsigned int>(offset_k) + lz;
  const long long region_lin = offset3d(x, y, lz, sy, sz);
  const long long hx_cur = offset3d(gx, gy, gz, hx_ny, hx_nz);
  const long long hx_prev = offset3d(gx, gy, gz - 1, hx_ny, hx_nz);
  const float derivative = (hx[hx_cur] - hx[hx_prev]) * inv_dz[gz];
  const float psi_new = b_z[gz] * psi_z[region_lin] + c_z[gz] * derivative;
  psi_z[region_lin] = psi_new;
  const float correction = derivative * (inv_kappa_z[gz] - 1.0f) + psi_new;
  ey[region_lin] = ey[region_lin] + curl[region_lin] * correction;
}

}  // namespace

void update_electric_ex_standard_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  check_electric_inputs(ex, hy, hz, decay, curl, "ex");
  check_rank3_shape(hy, "hy", ex.size(0), ex.size(1), ex.size(2) - 1);
  check_rank3_shape(hz, "hz", ex.size(0), ex.size(1) - 1, ex.size(2));
  check_spacing_vector(ex, inv_dy, 1, "inv_dy");
  check_spacing_vector(ex, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ex.get_device_index());
  const auto sizes = ex.sizes();
  const dim3 block = field_block3d();
  if (inactive_boundary_pair(y_low_mode, y_high_mode) && inactive_boundary_pair(z_low_mode, z_high_mode)) {
    if (sizes[1] > 2 && sizes[2] > 2) {
      update_electric_ex_standard_interior_kernel<<<field_grid3d(sizes[0], sizes[1] - 2, sizes[2] - 2, block), block, 0, current_cuda_stream()>>>(
          static_cast<unsigned int>(sizes[0]),
          static_cast<unsigned int>(sizes[1]),
          static_cast<unsigned int>(sizes[2]),
          hy.mutable_data_ptr<float>(),
          hz.mutable_data_ptr<float>(),
          decay.mutable_data_ptr<float>(),
          curl.mutable_data_ptr<float>(),
          inv_dy.mutable_data_ptr<float>(),
          inv_dz.mutable_data_ptr<float>(),
          ex.mutable_data_ptr<float>());
    }
  } else {
    update_electric_ex_standard_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
        static_cast<unsigned int>(sizes[0]),
        static_cast<unsigned int>(sizes[1]),
        static_cast<unsigned int>(sizes[2]),
        hy.mutable_data_ptr<float>(),
        hz.mutable_data_ptr<float>(),
        decay.mutable_data_ptr<float>(),
        curl.mutable_data_ptr<float>(),
        inv_dy.mutable_data_ptr<float>(),
        inv_dz.mutable_data_ptr<float>(),
        static_cast<int>(y_low_mode),
        static_cast<int>(y_high_mode),
        static_cast<int>(z_low_mode),
        static_cast<int>(z_high_mode),
        ex.mutable_data_ptr<float>());
  }
  WITWIN_CUDA_CHECK();
}

void update_electric_ey_standard_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  check_electric_inputs(ey, hx, hz, decay, curl, "ey");
  check_rank3_shape(hx, "hx", ey.size(0), ey.size(1), ey.size(2) - 1);
  check_rank3_shape(hz, "hz", ey.size(0) - 1, ey.size(1), ey.size(2));
  check_spacing_vector(ey, inv_dx, 0, "inv_dx");
  check_spacing_vector(ey, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ey.get_device_index());
  const auto sizes = ey.sizes();
  const dim3 block = field_block3d();
  if (inactive_boundary_pair(x_low_mode, x_high_mode) && inactive_boundary_pair(z_low_mode, z_high_mode)) {
    if (sizes[0] > 2 && sizes[2] > 2) {
      update_electric_ey_standard_interior_kernel<<<field_grid3d(sizes[0] - 2, sizes[1], sizes[2] - 2, block), block, 0, current_cuda_stream()>>>(
          static_cast<unsigned int>(sizes[0]),
          static_cast<unsigned int>(sizes[1]),
          static_cast<unsigned int>(sizes[2]),
          hx.mutable_data_ptr<float>(),
          hz.mutable_data_ptr<float>(),
          decay.mutable_data_ptr<float>(),
          curl.mutable_data_ptr<float>(),
          inv_dx.mutable_data_ptr<float>(),
          inv_dz.mutable_data_ptr<float>(),
          ey.mutable_data_ptr<float>());
    }
  } else {
    update_electric_ey_standard_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
        static_cast<unsigned int>(sizes[0]),
        static_cast<unsigned int>(sizes[1]),
        static_cast<unsigned int>(sizes[2]),
        hx.mutable_data_ptr<float>(),
        hz.mutable_data_ptr<float>(),
        decay.mutable_data_ptr<float>(),
        curl.mutable_data_ptr<float>(),
        inv_dx.mutable_data_ptr<float>(),
        inv_dz.mutable_data_ptr<float>(),
        static_cast<int>(x_low_mode),
        static_cast<int>(x_high_mode),
        static_cast<int>(z_low_mode),
        static_cast<int>(z_high_mode),
        ey.mutable_data_ptr<float>());
  }
  WITWIN_CUDA_CHECK();
}

void update_electric_ez_standard_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode) {
  check_electric_inputs(ez, hx, hy, decay, curl, "ez");
  check_rank3_shape(hx, "hx", ez.size(0), ez.size(1) - 1, ez.size(2));
  check_rank3_shape(hy, "hy", ez.size(0) - 1, ez.size(1), ez.size(2));
  check_spacing_vector(ez, inv_dx, 0, "inv_dx");
  check_spacing_vector(ez, inv_dy, 1, "inv_dy");
  torch::stable::accelerator::DeviceGuard guard(ez.get_device_index());
  const auto sizes = ez.sizes();
  const dim3 block = field_block3d();
  if (inactive_boundary_pair(x_low_mode, x_high_mode) && inactive_boundary_pair(y_low_mode, y_high_mode)) {
    if (sizes[0] > 2 && sizes[1] > 2) {
      update_electric_ez_standard_interior_kernel<<<field_grid3d(sizes[0] - 2, sizes[1] - 2, sizes[2], block), block, 0, current_cuda_stream()>>>(
          static_cast<unsigned int>(sizes[0]),
          static_cast<unsigned int>(sizes[1]),
          static_cast<unsigned int>(sizes[2]),
          hx.mutable_data_ptr<float>(),
          hy.mutable_data_ptr<float>(),
          decay.mutable_data_ptr<float>(),
          curl.mutable_data_ptr<float>(),
          inv_dx.mutable_data_ptr<float>(),
          inv_dy.mutable_data_ptr<float>(),
          ez.mutable_data_ptr<float>());
    }
  } else {
    update_electric_ez_standard_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
        static_cast<unsigned int>(sizes[0]),
        static_cast<unsigned int>(sizes[1]),
        static_cast<unsigned int>(sizes[2]),
        hx.mutable_data_ptr<float>(),
        hy.mutable_data_ptr<float>(),
        decay.mutable_data_ptr<float>(),
        curl.mutable_data_ptr<float>(),
        inv_dx.mutable_data_ptr<float>(),
        inv_dy.mutable_data_ptr<float>(),
        static_cast<int>(x_low_mode),
        static_cast<int>(x_high_mode),
        static_cast<int>(y_low_mode),
        static_cast<int>(y_high_mode),
        ez.mutable_data_ptr<float>());
  }
  WITWIN_CUDA_CHECK();
}

void update_electric_ex_bloch_cuda(
    torch::stable::Tensor ex_real,
    torch::stable::Tensor ex_imag,
    const torch::stable::Tensor& hy_real,
    const torch::stable::Tensor& hy_imag,
    const torch::stable::Tensor& hz_real,
    const torch::stable::Tensor& hz_imag,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    double phase_cos_y,
    double phase_sin_y,
    double phase_cos_z,
    double phase_sin_z,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz) {
  check_electric_bloch_inputs(ex_real, ex_imag, hy_real, hy_imag, hz_real, hz_imag, decay, curl, "ex_real");
  check_rank3_shape(hy_real, "hy_real", ex_real.size(0), ex_real.size(1), ex_real.size(2) - 1);
  check_rank3_shape(hy_imag, "hy_imag", ex_real.size(0), ex_real.size(1), ex_real.size(2) - 1);
  check_rank3_shape(hz_real, "hz_real", ex_real.size(0), ex_real.size(1) - 1, ex_real.size(2));
  check_rank3_shape(hz_imag, "hz_imag", ex_real.size(0), ex_real.size(1) - 1, ex_real.size(2));
  check_spacing_vector(ex_real, inv_dy, 1, "inv_dy");
  check_spacing_vector(ex_real, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ex_real.get_device_index());
  const auto sizes = ex_real.sizes();
  const dim3 block = field_block3d();
  update_electric_ex_bloch_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hy_real.mutable_data_ptr<float>(),
      hy_imag.mutable_data_ptr<float>(),
      hz_real.mutable_data_ptr<float>(),
      hz_imag.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      static_cast<float>(phase_cos_y),
      static_cast<float>(phase_sin_y),
      static_cast<float>(phase_cos_z),
      static_cast<float>(phase_sin_z),
      inv_dy.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      ex_real.mutable_data_ptr<float>(),
      ex_imag.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ey_bloch_cuda(
    torch::stable::Tensor ey_real,
    torch::stable::Tensor ey_imag,
    const torch::stable::Tensor& hx_real,
    const torch::stable::Tensor& hx_imag,
    const torch::stable::Tensor& hz_real,
    const torch::stable::Tensor& hz_imag,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_z,
    double phase_sin_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz) {
  check_electric_bloch_inputs(ey_real, ey_imag, hx_real, hx_imag, hz_real, hz_imag, decay, curl, "ey_real");
  check_rank3_shape(hx_real, "hx_real", ey_real.size(0), ey_real.size(1), ey_real.size(2) - 1);
  check_rank3_shape(hx_imag, "hx_imag", ey_real.size(0), ey_real.size(1), ey_real.size(2) - 1);
  check_rank3_shape(hz_real, "hz_real", ey_real.size(0) - 1, ey_real.size(1), ey_real.size(2));
  check_rank3_shape(hz_imag, "hz_imag", ey_real.size(0) - 1, ey_real.size(1), ey_real.size(2));
  check_spacing_vector(ey_real, inv_dx, 0, "inv_dx");
  check_spacing_vector(ey_real, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ey_real.get_device_index());
  const auto sizes = ey_real.sizes();
  const dim3 block = field_block3d();
  update_electric_ey_bloch_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hx_real.mutable_data_ptr<float>(),
      hx_imag.mutable_data_ptr<float>(),
      hz_real.mutable_data_ptr<float>(),
      hz_imag.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      static_cast<float>(phase_cos_x),
      static_cast<float>(phase_sin_x),
      static_cast<float>(phase_cos_z),
      static_cast<float>(phase_sin_z),
      inv_dx.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      ey_real.mutable_data_ptr<float>(),
      ey_imag.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ez_bloch_cuda(
    torch::stable::Tensor ez_real,
    torch::stable::Tensor ez_imag,
    const torch::stable::Tensor& hx_real,
    const torch::stable::Tensor& hx_imag,
    const torch::stable::Tensor& hy_real,
    const torch::stable::Tensor& hy_imag,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_y,
    double phase_sin_y,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy) {
  check_electric_bloch_inputs(ez_real, ez_imag, hx_real, hx_imag, hy_real, hy_imag, decay, curl, "ez_real");
  check_rank3_shape(hx_real, "hx_real", ez_real.size(0), ez_real.size(1) - 1, ez_real.size(2));
  check_rank3_shape(hx_imag, "hx_imag", ez_real.size(0), ez_real.size(1) - 1, ez_real.size(2));
  check_rank3_shape(hy_real, "hy_real", ez_real.size(0) - 1, ez_real.size(1), ez_real.size(2));
  check_rank3_shape(hy_imag, "hy_imag", ez_real.size(0) - 1, ez_real.size(1), ez_real.size(2));
  check_spacing_vector(ez_real, inv_dx, 0, "inv_dx");
  check_spacing_vector(ez_real, inv_dy, 1, "inv_dy");
  torch::stable::accelerator::DeviceGuard guard(ez_real.get_device_index());
  const auto sizes = ez_real.sizes();
  const dim3 block = field_block3d();
  update_electric_ez_bloch_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hx_real.mutable_data_ptr<float>(),
      hx_imag.mutable_data_ptr<float>(),
      hy_real.mutable_data_ptr<float>(),
      hy_imag.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      static_cast<float>(phase_cos_x),
      static_cast<float>(phase_sin_x),
      static_cast<float>(phase_cos_y),
      static_cast<float>(phase_sin_y),
      inv_dx.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>(),
      ez_real.mutable_data_ptr<float>(),
      ez_imag.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ex_cpml_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_y,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  check_electric_cpml_inputs(
      ex, hy, hz, decay, curl, psi_y, psi_z, inv_kappa_y, b_y, c_y, inv_kappa_z, b_z, c_z, 1, 2, "ex");
  check_rank3_shape(hy, "hy", ex.size(0), ex.size(1), ex.size(2) - 1);
  check_rank3_shape(hz, "hz", ex.size(0), ex.size(1) - 1, ex.size(2));
  check_spacing_vector(ex, inv_dy, 1, "inv_dy");
  check_spacing_vector(ex, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ex.get_device_index());
  const auto sizes = ex.sizes();
  const dim3 block = field_block3d();
  if (inactive_boundary_pair(y_low_mode, y_high_mode) && inactive_boundary_pair(z_low_mode, z_high_mode)) {
    if (sizes[1] > 2 && sizes[2] > 2) {
      update_electric_ex_cpml_interior_kernel<<<field_grid3d(sizes[0], sizes[1] - 2, sizes[2] - 2, block), block, 0, current_cuda_stream()>>>(
          static_cast<unsigned int>(sizes[0]),
          static_cast<unsigned int>(sizes[1]),
          static_cast<unsigned int>(sizes[2]),
          hy.mutable_data_ptr<float>(),
          hz.mutable_data_ptr<float>(),
          decay.mutable_data_ptr<float>(),
          curl.mutable_data_ptr<float>(),
          inv_kappa_y.mutable_data_ptr<float>(),
          b_y.mutable_data_ptr<float>(),
          c_y.mutable_data_ptr<float>(),
          inv_kappa_z.mutable_data_ptr<float>(),
          b_z.mutable_data_ptr<float>(),
          c_z.mutable_data_ptr<float>(),
          inv_dy.mutable_data_ptr<float>(),
          inv_dz.mutable_data_ptr<float>(),
          psi_y.mutable_data_ptr<float>(),
          psi_z.mutable_data_ptr<float>(),
          ex.mutable_data_ptr<float>());
    }
  } else {
    update_electric_ex_cpml_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
        static_cast<unsigned int>(sizes[0]),
        static_cast<unsigned int>(sizes[1]),
        static_cast<unsigned int>(sizes[2]),
        hy.mutable_data_ptr<float>(),
        hz.mutable_data_ptr<float>(),
        decay.mutable_data_ptr<float>(),
        curl.mutable_data_ptr<float>(),
        inv_kappa_y.mutable_data_ptr<float>(),
        b_y.mutable_data_ptr<float>(),
        c_y.mutable_data_ptr<float>(),
        inv_kappa_z.mutable_data_ptr<float>(),
        b_z.mutable_data_ptr<float>(),
        c_z.mutable_data_ptr<float>(),
        inv_dy.mutable_data_ptr<float>(),
        inv_dz.mutable_data_ptr<float>(),
        static_cast<int>(y_low_mode),
        static_cast<int>(y_high_mode),
        static_cast<int>(z_low_mode),
        static_cast<int>(z_high_mode),
        psi_y.mutable_data_ptr<float>(),
        psi_z.mutable_data_ptr<float>(),
        ex.mutable_data_ptr<float>());
  }
  WITWIN_CUDA_CHECK();
}

void update_electric_ey_cpml_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  check_electric_cpml_inputs(
      ey, hx, hz, decay, curl, psi_x, psi_z, inv_kappa_x, b_x, c_x, inv_kappa_z, b_z, c_z, 0, 2, "ey");
  check_rank3_shape(hx, "hx", ey.size(0), ey.size(1), ey.size(2) - 1);
  check_rank3_shape(hz, "hz", ey.size(0) - 1, ey.size(1), ey.size(2));
  check_spacing_vector(ey, inv_dx, 0, "inv_dx");
  check_spacing_vector(ey, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ey.get_device_index());
  const auto sizes = ey.sizes();
  const dim3 block = field_block3d();
  if (inactive_boundary_pair(x_low_mode, x_high_mode) && inactive_boundary_pair(z_low_mode, z_high_mode)) {
    if (sizes[0] > 2 && sizes[2] > 2) {
      update_electric_ey_cpml_interior_kernel<<<field_grid3d(sizes[0] - 2, sizes[1], sizes[2] - 2, block), block, 0, current_cuda_stream()>>>(
          static_cast<unsigned int>(sizes[0]),
          static_cast<unsigned int>(sizes[1]),
          static_cast<unsigned int>(sizes[2]),
          hx.mutable_data_ptr<float>(),
          hz.mutable_data_ptr<float>(),
          decay.mutable_data_ptr<float>(),
          curl.mutable_data_ptr<float>(),
          inv_kappa_x.mutable_data_ptr<float>(),
          b_x.mutable_data_ptr<float>(),
          c_x.mutable_data_ptr<float>(),
          inv_kappa_z.mutable_data_ptr<float>(),
          b_z.mutable_data_ptr<float>(),
          c_z.mutable_data_ptr<float>(),
          inv_dx.mutable_data_ptr<float>(),
          inv_dz.mutable_data_ptr<float>(),
          psi_x.mutable_data_ptr<float>(),
          psi_z.mutable_data_ptr<float>(),
          ey.mutable_data_ptr<float>());
    }
  } else {
    update_electric_ey_cpml_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
        static_cast<unsigned int>(sizes[0]),
        static_cast<unsigned int>(sizes[1]),
        static_cast<unsigned int>(sizes[2]),
        hx.mutable_data_ptr<float>(),
        hz.mutable_data_ptr<float>(),
        decay.mutable_data_ptr<float>(),
        curl.mutable_data_ptr<float>(),
        inv_kappa_x.mutable_data_ptr<float>(),
        b_x.mutable_data_ptr<float>(),
        c_x.mutable_data_ptr<float>(),
        inv_kappa_z.mutable_data_ptr<float>(),
        b_z.mutable_data_ptr<float>(),
        c_z.mutable_data_ptr<float>(),
        inv_dx.mutable_data_ptr<float>(),
        inv_dz.mutable_data_ptr<float>(),
        static_cast<int>(x_low_mode),
        static_cast<int>(x_high_mode),
        static_cast<int>(z_low_mode),
        static_cast<int>(z_high_mode),
        psi_x.mutable_data_ptr<float>(),
        psi_z.mutable_data_ptr<float>(),
        ey.mutable_data_ptr<float>());
  }
  WITWIN_CUDA_CHECK();
}

void update_electric_ez_cpml_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_y,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode) {
  check_electric_cpml_inputs(
      ez, hx, hy, decay, curl, psi_x, psi_y, inv_kappa_x, b_x, c_x, inv_kappa_y, b_y, c_y, 0, 1, "ez");
  check_rank3_shape(hx, "hx", ez.size(0), ez.size(1) - 1, ez.size(2));
  check_rank3_shape(hy, "hy", ez.size(0) - 1, ez.size(1), ez.size(2));
  check_spacing_vector(ez, inv_dx, 0, "inv_dx");
  check_spacing_vector(ez, inv_dy, 1, "inv_dy");
  torch::stable::accelerator::DeviceGuard guard(ez.get_device_index());
  const auto sizes = ez.sizes();
  const dim3 block = field_block3d();
  if (inactive_boundary_pair(x_low_mode, x_high_mode) && inactive_boundary_pair(y_low_mode, y_high_mode)) {
    if (sizes[0] > 2 && sizes[1] > 2) {
      update_electric_ez_cpml_interior_kernel<<<field_grid3d(sizes[0] - 2, sizes[1] - 2, sizes[2], block), block, 0, current_cuda_stream()>>>(
          static_cast<unsigned int>(sizes[0]),
          static_cast<unsigned int>(sizes[1]),
          static_cast<unsigned int>(sizes[2]),
          hx.mutable_data_ptr<float>(),
          hy.mutable_data_ptr<float>(),
          decay.mutable_data_ptr<float>(),
          curl.mutable_data_ptr<float>(),
          inv_kappa_x.mutable_data_ptr<float>(),
          b_x.mutable_data_ptr<float>(),
          c_x.mutable_data_ptr<float>(),
          inv_kappa_y.mutable_data_ptr<float>(),
          b_y.mutable_data_ptr<float>(),
          c_y.mutable_data_ptr<float>(),
          inv_dx.mutable_data_ptr<float>(),
          inv_dy.mutable_data_ptr<float>(),
          psi_x.mutable_data_ptr<float>(),
          psi_y.mutable_data_ptr<float>(),
          ez.mutable_data_ptr<float>());
    }
  } else {
    update_electric_ez_cpml_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
        static_cast<unsigned int>(sizes[0]),
        static_cast<unsigned int>(sizes[1]),
        static_cast<unsigned int>(sizes[2]),
        hx.mutable_data_ptr<float>(),
        hy.mutable_data_ptr<float>(),
        decay.mutable_data_ptr<float>(),
        curl.mutable_data_ptr<float>(),
        inv_kappa_x.mutable_data_ptr<float>(),
        b_x.mutable_data_ptr<float>(),
        c_x.mutable_data_ptr<float>(),
        inv_kappa_y.mutable_data_ptr<float>(),
        b_y.mutable_data_ptr<float>(),
        c_y.mutable_data_ptr<float>(),
        inv_dx.mutable_data_ptr<float>(),
        inv_dy.mutable_data_ptr<float>(),
        static_cast<int>(x_low_mode),
        static_cast<int>(x_high_mode),
        static_cast<int>(y_low_mode),
        static_cast<int>(y_high_mode),
        psi_x.mutable_data_ptr<float>(),
        psi_y.mutable_data_ptr<float>(),
        ez.mutable_data_ptr<float>());
  }
  WITWIN_CUDA_CHECK();
}

void update_electric_ex_cpml_compressed_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_y,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode,
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length,
    std::optional<double> uniform_decay,
    std::optional<double> uniform_curl) {
  check_electric_cpml_compressed_inputs(
      ex, hy, hz, decay, curl, psi_y, psi_z, inv_kappa_y, b_y, c_y, inv_kappa_z, b_z, c_z,
      1, 2, y_low_length, y_high_length, z_low_length, z_high_length, "ex");
  check_rank3_shape(hy, "hy", ex.size(0), ex.size(1), ex.size(2) - 1);
  check_rank3_shape(hz, "hz", ex.size(0), ex.size(1) - 1, ex.size(2));
  check_spacing_vector(ex, inv_dy, 1, "inv_dy");
  check_spacing_vector(ex, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ex.get_device_index());
  const auto sizes = ex.sizes();
  const dim3 block = field_block3d();
  dispatch_uniform_coefficients(uniform_decay.has_value(), uniform_curl.has_value(), [&](auto u_decay, auto u_curl) {
    update_electric_ex_cpml_compressed_kernel<decltype(u_decay)::value, decltype(u_curl)::value><<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
        static_cast<unsigned int>(sizes[0]),
        static_cast<unsigned int>(sizes[1]),
        static_cast<unsigned int>(sizes[2]),
        hy.mutable_data_ptr<float>(),
        hz.mutable_data_ptr<float>(),
        decay.mutable_data_ptr<float>(),
        curl.mutable_data_ptr<float>(),
        static_cast<float>(uniform_decay.value_or(0.0)),
        static_cast<float>(uniform_curl.value_or(0.0)),
        inv_kappa_y.mutable_data_ptr<float>(),
        b_y.mutable_data_ptr<float>(),
        c_y.mutable_data_ptr<float>(),
        inv_kappa_z.mutable_data_ptr<float>(),
        b_z.mutable_data_ptr<float>(),
        c_z.mutable_data_ptr<float>(),
        inv_dy.mutable_data_ptr<float>(),
        inv_dz.mutable_data_ptr<float>(),
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
        psi_y.mutable_data_ptr<float>(),
        psi_z.mutable_data_ptr<float>(),
        ex.mutable_data_ptr<float>());
  });
  WITWIN_CUDA_CHECK();
}

void update_electric_ey_cpml_compressed_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode,
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length,
    std::optional<double> uniform_decay,
    std::optional<double> uniform_curl) {
  check_electric_cpml_compressed_inputs(
      ey, hx, hz, decay, curl, psi_x, psi_z, inv_kappa_x, b_x, c_x, inv_kappa_z, b_z, c_z,
      0, 2, x_low_length, x_high_length, z_low_length, z_high_length, "ey");
  check_rank3_shape(hx, "hx", ey.size(0), ey.size(1), ey.size(2) - 1);
  check_rank3_shape(hz, "hz", ey.size(0) - 1, ey.size(1), ey.size(2));
  check_spacing_vector(ey, inv_dx, 0, "inv_dx");
  check_spacing_vector(ey, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ey.get_device_index());
  const auto sizes = ey.sizes();
  const dim3 block = field_block3d();
  dispatch_uniform_coefficients(uniform_decay.has_value(), uniform_curl.has_value(), [&](auto u_decay, auto u_curl) {
    update_electric_ey_cpml_compressed_kernel<decltype(u_decay)::value, decltype(u_curl)::value><<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
        static_cast<unsigned int>(sizes[0]),
        static_cast<unsigned int>(sizes[1]),
        static_cast<unsigned int>(sizes[2]),
        hx.mutable_data_ptr<float>(),
        hz.mutable_data_ptr<float>(),
        decay.mutable_data_ptr<float>(),
        curl.mutable_data_ptr<float>(),
        static_cast<float>(uniform_decay.value_or(0.0)),
        static_cast<float>(uniform_curl.value_or(0.0)),
        inv_kappa_x.mutable_data_ptr<float>(),
        b_x.mutable_data_ptr<float>(),
        c_x.mutable_data_ptr<float>(),
        inv_kappa_z.mutable_data_ptr<float>(),
        b_z.mutable_data_ptr<float>(),
        c_z.mutable_data_ptr<float>(),
        inv_dx.mutable_data_ptr<float>(),
        inv_dz.mutable_data_ptr<float>(),
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
        psi_x.mutable_data_ptr<float>(),
        psi_z.mutable_data_ptr<float>(),
        ey.mutable_data_ptr<float>());
  });
  WITWIN_CUDA_CHECK();
}

void update_electric_ez_cpml_compressed_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_y,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length,
    std::optional<double> uniform_decay,
    std::optional<double> uniform_curl) {
  check_electric_cpml_compressed_inputs(
      ez, hx, hy, decay, curl, psi_x, psi_y, inv_kappa_x, b_x, c_x, inv_kappa_y, b_y, c_y,
      0, 1, x_low_length, x_high_length, y_low_length, y_high_length, "ez");
  check_rank3_shape(hx, "hx", ez.size(0), ez.size(1) - 1, ez.size(2));
  check_rank3_shape(hy, "hy", ez.size(0) - 1, ez.size(1), ez.size(2));
  check_spacing_vector(ez, inv_dx, 0, "inv_dx");
  check_spacing_vector(ez, inv_dy, 1, "inv_dy");
  torch::stable::accelerator::DeviceGuard guard(ez.get_device_index());
  const auto sizes = ez.sizes();
  const dim3 block = field_block3d();
  dispatch_uniform_coefficients(uniform_decay.has_value(), uniform_curl.has_value(), [&](auto u_decay, auto u_curl) {
    update_electric_ez_cpml_compressed_kernel<decltype(u_decay)::value, decltype(u_curl)::value><<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
        static_cast<unsigned int>(sizes[0]),
        static_cast<unsigned int>(sizes[1]),
        static_cast<unsigned int>(sizes[2]),
        hx.mutable_data_ptr<float>(),
        hy.mutable_data_ptr<float>(),
        decay.mutable_data_ptr<float>(),
        curl.mutable_data_ptr<float>(),
        static_cast<float>(uniform_decay.value_or(0.0)),
        static_cast<float>(uniform_curl.value_or(0.0)),
        inv_kappa_x.mutable_data_ptr<float>(),
        b_x.mutable_data_ptr<float>(),
        c_x.mutable_data_ptr<float>(),
        inv_kappa_y.mutable_data_ptr<float>(),
        b_y.mutable_data_ptr<float>(),
        c_y.mutable_data_ptr<float>(),
        inv_dx.mutable_data_ptr<float>(),
        inv_dy.mutable_data_ptr<float>(),
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
        psi_x.mutable_data_ptr<float>(),
        psi_y.mutable_data_ptr<float>(),
        ez.mutable_data_ptr<float>());
  });
  WITWIN_CUDA_CHECK();
}

void update_electric_ex_cpml_modulated_compressed_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    torch::stable::Tensor psi_y,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
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
  check_modulation_fields(ex, mod_cos, mod_sin, mod_omega);
  check_rank3_shape(hy, "hy", ex.size(0), ex.size(1), ex.size(2) - 1);
  check_rank3_shape(hz, "hz", ex.size(0), ex.size(1) - 1, ex.size(2));
  check_spacing_vector(ex, inv_dy, 1, "inv_dy");
  check_spacing_vector(ex, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ex.get_device_index());
  const auto sizes = ex.sizes();
  const dim3 block = field_block3d();
  update_electric_ex_cpml_modulated_compressed_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hy.mutable_data_ptr<float>(),
      hz.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      mod_cos.mutable_data_ptr<float>(),
      mod_sin.mutable_data_ptr<float>(),
      mod_omega.mutable_data_ptr<float>(),
      static_cast<float>(t_prev),
      static_cast<float>(t_next),
      inv_kappa_y.mutable_data_ptr<float>(),
      b_y.mutable_data_ptr<float>(),
      c_y.mutable_data_ptr<float>(),
      inv_kappa_z.mutable_data_ptr<float>(),
      b_z.mutable_data_ptr<float>(),
      c_z.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
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
      psi_y.mutable_data_ptr<float>(),
      psi_z.mutable_data_ptr<float>(),
      ex.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ey_cpml_modulated_compressed_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
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
  check_modulation_fields(ey, mod_cos, mod_sin, mod_omega);
  check_rank3_shape(hx, "hx", ey.size(0), ey.size(1), ey.size(2) - 1);
  check_rank3_shape(hz, "hz", ey.size(0) - 1, ey.size(1), ey.size(2));
  check_spacing_vector(ey, inv_dx, 0, "inv_dx");
  check_spacing_vector(ey, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ey.get_device_index());
  const auto sizes = ey.sizes();
  const dim3 block = field_block3d();
  update_electric_ey_cpml_modulated_compressed_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hx.mutable_data_ptr<float>(),
      hz.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      mod_cos.mutable_data_ptr<float>(),
      mod_sin.mutable_data_ptr<float>(),
      mod_omega.mutable_data_ptr<float>(),
      static_cast<float>(t_prev),
      static_cast<float>(t_next),
      inv_kappa_x.mutable_data_ptr<float>(),
      b_x.mutable_data_ptr<float>(),
      c_x.mutable_data_ptr<float>(),
      inv_kappa_z.mutable_data_ptr<float>(),
      b_z.mutable_data_ptr<float>(),
      c_z.mutable_data_ptr<float>(),
      inv_dx.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
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
      psi_x.mutable_data_ptr<float>(),
      psi_z.mutable_data_ptr<float>(),
      ey.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ez_cpml_modulated_compressed_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_y,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
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
  check_modulation_fields(ez, mod_cos, mod_sin, mod_omega);
  check_rank3_shape(hx, "hx", ez.size(0), ez.size(1) - 1, ez.size(2));
  check_rank3_shape(hy, "hy", ez.size(0) - 1, ez.size(1), ez.size(2));
  check_spacing_vector(ez, inv_dx, 0, "inv_dx");
  check_spacing_vector(ez, inv_dy, 1, "inv_dy");
  torch::stable::accelerator::DeviceGuard guard(ez.get_device_index());
  const auto sizes = ez.sizes();
  const dim3 block = field_block3d();
  update_electric_ez_cpml_modulated_compressed_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hx.mutable_data_ptr<float>(),
      hy.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      mod_cos.mutable_data_ptr<float>(),
      mod_sin.mutable_data_ptr<float>(),
      mod_omega.mutable_data_ptr<float>(),
      static_cast<float>(t_prev),
      static_cast<float>(t_next),
      inv_kappa_x.mutable_data_ptr<float>(),
      b_x.mutable_data_ptr<float>(),
      c_x.mutable_data_ptr<float>(),
      inv_kappa_y.mutable_data_ptr<float>(),
      b_y.mutable_data_ptr<float>(),
      c_y.mutable_data_ptr<float>(),
      inv_dx.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>(),
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
      psi_x.mutable_data_ptr<float>(),
      psi_y.mutable_data_ptr<float>(),
      ez.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ex_bloch_y_standard_z_cuda(
    torch::stable::Tensor ex_real,
    torch::stable::Tensor ex_imag,
    const torch::stable::Tensor& hy_real,
    const torch::stable::Tensor& hy_imag,
    const torch::stable::Tensor& hz_real,
    const torch::stable::Tensor& hz_imag,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    double phase_cos_y,
    double phase_sin_y,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  check_spacing_vector(ex_real, inv_dy, 1, "inv_dy");
  check_spacing_vector(ex_real, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ex_real.get_device_index());
  const auto sizes = ex_real.sizes();
  const dim3 block = field_block3d();
  update_electric_ex_bloch_y_standard_z_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hy_real.mutable_data_ptr<float>(),
      hy_imag.mutable_data_ptr<float>(),
      hz_real.mutable_data_ptr<float>(),
      hz_imag.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      static_cast<float>(phase_cos_y),
      static_cast<float>(phase_sin_y),
      inv_dy.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode),
      ex_real.mutable_data_ptr<float>(),
      ex_imag.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ey_bloch_x_standard_z_cuda(
    torch::stable::Tensor ey_real,
    torch::stable::Tensor ey_imag,
    const torch::stable::Tensor& hx_real,
    const torch::stable::Tensor& hx_imag,
    const torch::stable::Tensor& hz_real,
    const torch::stable::Tensor& hz_imag,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    double phase_cos_x,
    double phase_sin_x,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  check_spacing_vector(ey_real, inv_dx, 0, "inv_dx");
  check_spacing_vector(ey_real, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ey_real.get_device_index());
  const auto sizes = ey_real.sizes();
  const dim3 block = field_block3d();
  update_electric_ey_bloch_x_standard_z_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hx_real.mutable_data_ptr<float>(),
      hx_imag.mutable_data_ptr<float>(),
      hz_real.mutable_data_ptr<float>(),
      hz_imag.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      static_cast<float>(phase_cos_x),
      static_cast<float>(phase_sin_x),
      inv_dx.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode),
      ey_real.mutable_data_ptr<float>(),
      ey_imag.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void apply_electric_ex_cpml_z_correction_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dz,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t full_size_y,
    int64_t full_size_z) {
  const auto sizes = ex.sizes();
  const int sx = static_cast<int>(sizes[0]);
  const int sy = static_cast<int>(sizes[1]);
  const int sz = static_cast<int>(sizes[2]);
  const int start_candidate = 1 - static_cast<int>(offset_k);
  const int local_z_start = start_candidate > 0 ? start_candidate : 0;
  const int stop_candidate = static_cast<int>(full_size_z) - 1 - static_cast<int>(offset_k);
  const int local_z_stop = sz < stop_candidate ? sz : stop_candidate;
  if (local_z_stop <= local_z_start) {
    return;
  }
  check_vector_input(inv_dz, full_size_z, "inv_dz");
  check_same_cuda_device(ex, inv_dz, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ex.get_device_index());
  const dim3 block = field_block3d();
  apply_electric_ex_cpml_z_correction_kernel<<<field_grid3d(sx, sy, sz, block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sx),
      static_cast<unsigned int>(sy),
      static_cast<unsigned int>(sz),
      static_cast<unsigned int>(hy.size(1)),
      static_cast<unsigned int>(hy.size(2)),
      hy.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      inv_kappa_z.mutable_data_ptr<float>(),
      b_z.mutable_data_ptr<float>(),
      c_z.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      static_cast<int>(offset_i),
      static_cast<int>(offset_j),
      static_cast<int>(offset_k),
      static_cast<int>(y_low_mode),
      static_cast<int>(y_high_mode),
      static_cast<unsigned int>(full_size_y),
      local_z_start,
      local_z_stop,
      psi_z.mutable_data_ptr<float>(),
      ex.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void apply_electric_ey_cpml_z_correction_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dz,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t full_size_x,
    int64_t full_size_z) {
  const auto sizes = ey.sizes();
  const int sx = static_cast<int>(sizes[0]);
  const int sy = static_cast<int>(sizes[1]);
  const int sz = static_cast<int>(sizes[2]);
  const int start_candidate = 1 - static_cast<int>(offset_k);
  const int local_z_start = start_candidate > 0 ? start_candidate : 0;
  const int stop_candidate = static_cast<int>(full_size_z) - 1 - static_cast<int>(offset_k);
  const int local_z_stop = sz < stop_candidate ? sz : stop_candidate;
  if (local_z_stop <= local_z_start) {
    return;
  }
  check_vector_input(inv_dz, full_size_z, "inv_dz");
  check_same_cuda_device(ey, inv_dz, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ey.get_device_index());
  const dim3 block = field_block3d();
  apply_electric_ey_cpml_z_correction_kernel<<<field_grid3d(sx, sy, sz, block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sx),
      static_cast<unsigned int>(sy),
      static_cast<unsigned int>(sz),
      static_cast<unsigned int>(hx.size(1)),
      static_cast<unsigned int>(hx.size(2)),
      hx.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      inv_kappa_z.mutable_data_ptr<float>(),
      b_z.mutable_data_ptr<float>(),
      c_z.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      static_cast<int>(offset_i),
      static_cast<int>(offset_j),
      static_cast<int>(offset_k),
      static_cast<int>(x_low_mode),
      static_cast<int>(x_high_mode),
      static_cast<unsigned int>(full_size_x),
      local_z_start,
      local_z_stop,
      psi_z.mutable_data_ptr<float>(),
      ey.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ex_modulated_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  check_electric_inputs(ex, hy, hz, decay, curl, "ex");
  check_modulation_fields(ex, mod_cos, mod_sin, mod_omega);
  check_rank3_shape(hy, "hy", ex.size(0), ex.size(1), ex.size(2) - 1);
  check_rank3_shape(hz, "hz", ex.size(0), ex.size(1) - 1, ex.size(2));
  check_spacing_vector(ex, inv_dy, 1, "inv_dy");
  check_spacing_vector(ex, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ex.get_device_index());
  const auto sizes = ex.sizes();
  const dim3 block = field_block3d();
  update_electric_ex_modulated_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hy.mutable_data_ptr<float>(),
      hz.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      mod_cos.mutable_data_ptr<float>(),
      mod_sin.mutable_data_ptr<float>(),
      mod_omega.mutable_data_ptr<float>(),
      static_cast<float>(t_prev),
      static_cast<float>(t_next),
      inv_dy.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      static_cast<int>(y_low_mode),
      static_cast<int>(y_high_mode),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode),
      ex.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ey_modulated_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  check_electric_inputs(ey, hx, hz, decay, curl, "ey");
  check_modulation_fields(ey, mod_cos, mod_sin, mod_omega);
  check_rank3_shape(hx, "hx", ey.size(0), ey.size(1), ey.size(2) - 1);
  check_rank3_shape(hz, "hz", ey.size(0) - 1, ey.size(1), ey.size(2));
  check_spacing_vector(ey, inv_dx, 0, "inv_dx");
  check_spacing_vector(ey, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ey.get_device_index());
  const auto sizes = ey.sizes();
  const dim3 block = field_block3d();
  update_electric_ey_modulated_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hx.mutable_data_ptr<float>(),
      hz.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      mod_cos.mutable_data_ptr<float>(),
      mod_sin.mutable_data_ptr<float>(),
      mod_omega.mutable_data_ptr<float>(),
      static_cast<float>(t_prev),
      static_cast<float>(t_next),
      inv_dx.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      static_cast<int>(x_low_mode),
      static_cast<int>(x_high_mode),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode),
      ey.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ez_modulated_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode) {
  check_electric_inputs(ez, hx, hy, decay, curl, "ez");
  check_modulation_fields(ez, mod_cos, mod_sin, mod_omega);
  check_rank3_shape(hx, "hx", ez.size(0), ez.size(1) - 1, ez.size(2));
  check_rank3_shape(hy, "hy", ez.size(0) - 1, ez.size(1), ez.size(2));
  check_spacing_vector(ez, inv_dx, 0, "inv_dx");
  check_spacing_vector(ez, inv_dy, 1, "inv_dy");
  torch::stable::accelerator::DeviceGuard guard(ez.get_device_index());
  const auto sizes = ez.sizes();
  const dim3 block = field_block3d();
  update_electric_ez_modulated_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hx.mutable_data_ptr<float>(),
      hy.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      mod_cos.mutable_data_ptr<float>(),
      mod_sin.mutable_data_ptr<float>(),
      mod_omega.mutable_data_ptr<float>(),
      static_cast<float>(t_prev),
      static_cast<float>(t_next),
      inv_dx.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>(),
      static_cast<int>(x_low_mode),
      static_cast<int>(x_high_mode),
      static_cast<int>(y_low_mode),
      static_cast<int>(y_high_mode),
      ez.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ex_cpml_modulated_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    torch::stable::Tensor psi_y,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  check_electric_cpml_inputs(
      ex, hy, hz, decay, curl, psi_y, psi_z, inv_kappa_y, b_y, c_y, inv_kappa_z, b_z, c_z, 1, 2, "ex");
  check_modulation_fields(ex, mod_cos, mod_sin, mod_omega);
  check_rank3_shape(hy, "hy", ex.size(0), ex.size(1), ex.size(2) - 1);
  check_rank3_shape(hz, "hz", ex.size(0), ex.size(1) - 1, ex.size(2));
  check_spacing_vector(ex, inv_dy, 1, "inv_dy");
  check_spacing_vector(ex, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ex.get_device_index());
  const auto sizes = ex.sizes();
  const dim3 block = field_block3d();
  update_electric_ex_cpml_modulated_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hy.mutable_data_ptr<float>(),
      hz.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      mod_cos.mutable_data_ptr<float>(),
      mod_sin.mutable_data_ptr<float>(),
      mod_omega.mutable_data_ptr<float>(),
      static_cast<float>(t_prev),
      static_cast<float>(t_next),
      inv_kappa_y.mutable_data_ptr<float>(),
      b_y.mutable_data_ptr<float>(),
      c_y.mutable_data_ptr<float>(),
      inv_kappa_z.mutable_data_ptr<float>(),
      b_z.mutable_data_ptr<float>(),
      c_z.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      static_cast<int>(y_low_mode),
      static_cast<int>(y_high_mode),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode),
      psi_y.mutable_data_ptr<float>(),
      psi_z.mutable_data_ptr<float>(),
      ex.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ey_cpml_modulated_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  check_electric_cpml_inputs(
      ey, hx, hz, decay, curl, psi_x, psi_z, inv_kappa_x, b_x, c_x, inv_kappa_z, b_z, c_z, 0, 2, "ey");
  check_modulation_fields(ey, mod_cos, mod_sin, mod_omega);
  check_rank3_shape(hx, "hx", ey.size(0), ey.size(1), ey.size(2) - 1);
  check_rank3_shape(hz, "hz", ey.size(0) - 1, ey.size(1), ey.size(2));
  check_spacing_vector(ey, inv_dx, 0, "inv_dx");
  check_spacing_vector(ey, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(ey.get_device_index());
  const auto sizes = ey.sizes();
  const dim3 block = field_block3d();
  update_electric_ey_cpml_modulated_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hx.mutable_data_ptr<float>(),
      hz.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      mod_cos.mutable_data_ptr<float>(),
      mod_sin.mutable_data_ptr<float>(),
      mod_omega.mutable_data_ptr<float>(),
      static_cast<float>(t_prev),
      static_cast<float>(t_next),
      inv_kappa_x.mutable_data_ptr<float>(),
      b_x.mutable_data_ptr<float>(),
      c_x.mutable_data_ptr<float>(),
      inv_kappa_z.mutable_data_ptr<float>(),
      b_z.mutable_data_ptr<float>(),
      c_z.mutable_data_ptr<float>(),
      inv_dx.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      static_cast<int>(x_low_mode),
      static_cast<int>(x_high_mode),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode),
      psi_x.mutable_data_ptr<float>(),
      psi_z.mutable_data_ptr<float>(),
      ey.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ez_cpml_modulated_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_y,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode) {
  check_electric_cpml_inputs(
      ez, hx, hy, decay, curl, psi_x, psi_y, inv_kappa_x, b_x, c_x, inv_kappa_y, b_y, c_y, 0, 1, "ez");
  check_modulation_fields(ez, mod_cos, mod_sin, mod_omega);
  check_rank3_shape(hx, "hx", ez.size(0), ez.size(1) - 1, ez.size(2));
  check_rank3_shape(hy, "hy", ez.size(0) - 1, ez.size(1), ez.size(2));
  check_spacing_vector(ez, inv_dx, 0, "inv_dx");
  check_spacing_vector(ez, inv_dy, 1, "inv_dy");
  torch::stable::accelerator::DeviceGuard guard(ez.get_device_index());
  const auto sizes = ez.sizes();
  const dim3 block = field_block3d();
  update_electric_ez_cpml_modulated_kernel<<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      hx.mutable_data_ptr<float>(),
      hy.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      mod_cos.mutable_data_ptr<float>(),
      mod_sin.mutable_data_ptr<float>(),
      mod_omega.mutable_data_ptr<float>(),
      static_cast<float>(t_prev),
      static_cast<float>(t_next),
      inv_kappa_x.mutable_data_ptr<float>(),
      b_x.mutable_data_ptr<float>(),
      c_x.mutable_data_ptr<float>(),
      inv_kappa_y.mutable_data_ptr<float>(),
      b_y.mutable_data_ptr<float>(),
      c_y.mutable_data_ptr<float>(),
      inv_dx.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>(),
      static_cast<int>(x_low_mode),
      static_cast<int>(x_high_mode),
      static_cast<int>(y_low_mode),
      static_cast<int>(y_high_mode),
      psi_x.mutable_data_ptr<float>(),
      psi_y.mutable_data_ptr<float>(),
      ez.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}
