#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

namespace {

void check_field(const at::Tensor& tensor, const char* name) {
  check_float32_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  TORCH_CHECK(tensor.dim() == 3, name, " must be a contiguous 3D float32 CUDA tensor");
}

void check_matching_field(const at::Tensor& reference, const at::Tensor& tensor, const char* name) {
  check_field(tensor, name);
  TORCH_CHECK(tensor.sizes() == reference.sizes(), name, " must match reference shape");
}

void check_vector(const at::Tensor& tensor, const char* name) {
  check_float32_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  TORCH_CHECK(tensor.dim() == 1, name, " must be a contiguous 1D float32 CUDA tensor");
}

void check_matching_vector(const at::Tensor& reference, const at::Tensor& tensor, const char* name) {
  check_vector(tensor, name);
  TORCH_CHECK(tensor.sizes() == reference.sizes(), name, " must match reference shape");
}

void check_int32_vector(const at::Tensor& tensor, const char* name) {
  check_cuda_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be a contiguous int32 CUDA tensor");
  TORCH_CHECK(tensor.dim() == 1, name, " must be a contiguous 1D int32 CUDA tensor");
}

__device__ inline bool is_valid_index_3d(int i, int j, int k, int nx, int ny, int nz) {
  return i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz;
}

__device__ inline bool is_interior_coordinate(int coordinate, int size) {
  return coordinate > 0 && coordinate + 1 < size;
}

__device__ inline bool is_ex_active_index(int i, int j, int k, int nx, int ny, int nz) {
  return is_valid_index_3d(i, j, k, nx, ny, nz) && is_interior_coordinate(j, ny) && is_interior_coordinate(k, nz);
}

__device__ inline bool is_ey_active_index(int i, int j, int k, int nx, int ny, int nz) {
  return is_valid_index_3d(i, j, k, nx, ny, nz) && is_interior_coordinate(i, nx) && is_interior_coordinate(k, nz);
}

__device__ inline bool is_ez_active_index(int i, int j, int k, int nx, int ny, int nz) {
  return is_valid_index_3d(i, j, k, nx, ny, nz) && is_interior_coordinate(i, nx) && is_interior_coordinate(j, ny);
}

__device__ inline bool is_boundary_index(int coordinate, int size) {
  return coordinate == 0 || coordinate + 1 == size;
}

__device__ inline int select_boundary_mode(int low_mode, int high_mode, int coordinate, int size) {
  return coordinate == 0 ? low_mode : high_mode;
}

__device__ inline bool is_pec_boundary_mode(int mode) {
  return mode == BOUNDARY_PEC;
}

__device__ inline bool is_inactive_boundary_mode(int mode) {
  return mode == BOUNDARY_NONE || mode == BOUNDARY_PML;
}

struct ElectricCellStatus {
  bool active;
  bool inactive;
  bool pec;
};

__device__ inline ElectricCellStatus resolve_electric_cell_status(
    int coordinate_a,
    int size_a,
    int low_mode_a,
    int high_mode_a,
    int coordinate_b,
    int size_b,
    int low_mode_b,
    int high_mode_b) {
  bool axis_a_pec = false;
  bool axis_a_inactive = false;
  if (is_boundary_index(coordinate_a, size_a)) {
    const int mode_a = select_boundary_mode(low_mode_a, high_mode_a, coordinate_a, size_a);
    axis_a_pec = is_pec_boundary_mode(mode_a);
    axis_a_inactive = is_inactive_boundary_mode(mode_a);
  }

  bool axis_b_pec = false;
  bool axis_b_inactive = false;
  if (is_boundary_index(coordinate_b, size_b)) {
    const int mode_b = select_boundary_mode(low_mode_b, high_mode_b, coordinate_b, size_b);
    axis_b_pec = is_pec_boundary_mode(mode_b);
    axis_b_inactive = is_inactive_boundary_mode(mode_b);
  }

  ElectricCellStatus status;
  status.pec = axis_a_pec || axis_b_pec;
  status.inactive = !status.pec && (axis_a_inactive || axis_b_inactive);
  status.active = !status.pec && !status.inactive;
  return status;
}

__device__ inline bool diff_index_valid(int i, int j, int k, int nx, int ny, int nz) {
  return i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz;
}

__device__ inline float diff_grad_value(const float* diff_grad, int i, int j, int k, int ny, int nz) {
  return diff_grad[offset3d(
      static_cast<unsigned int>(i),
      static_cast<unsigned int>(j),
      static_cast<unsigned int>(k),
      static_cast<unsigned int>(ny),
      static_cast<unsigned int>(nz))];
}

__global__ void reverse_magnetic_decay_kernel(
    int64_t total,
    float* adj_prev,
    const float* adj_mid,
    const float* decay) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  adj_prev[index] = adj_mid[index] * decay[index];
}

__global__ void reverse_electric_to_hx_standard_kernel(
    int64_t total,
    int hx_ny,
    int hx_nz,
    int ey_nx,
    int ey_ny,
    int ey_nz,
    int ez_nx,
    int ez_ny,
    int ez_nz,
    float* adj_hx_mid,
    const float* adj_hx_post,
    const float* adj_ey_post,
    const float* adj_ez_post,
    const float* ey_curl,
    const float* ez_curl,
    float inv_dy,
    float inv_dz) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), hx_ny, hx_nz);
  const int i = static_cast<int>(coord.i);
  const int j = static_cast<int>(coord.j);
  const int k = static_cast<int>(coord.k);
  float adjoint = adj_hx_post[linear];

  if (is_ey_active_index(i, j, k, ey_nx, ey_ny, ey_nz)) {
    const long long ey_index = offset3d(coord.i, coord.j, coord.k, ey_ny, ey_nz);
    adjoint += ey_curl[ey_index] * inv_dz * adj_ey_post[ey_index];
  }
  if (is_ey_active_index(i, j, k + 1, ey_nx, ey_ny, ey_nz)) {
    const long long ey_index = offset3d(coord.i, coord.j, coord.k + 1, ey_ny, ey_nz);
    adjoint -= ey_curl[ey_index] * inv_dz * adj_ey_post[ey_index];
  }
  if (is_ez_active_index(i, j, k, ez_nx, ez_ny, ez_nz)) {
    const long long ez_index = offset3d(coord.i, coord.j, coord.k, ez_ny, ez_nz);
    adjoint -= ez_curl[ez_index] * inv_dy * adj_ez_post[ez_index];
  }
  if (is_ez_active_index(i, j + 1, k, ez_nx, ez_ny, ez_nz)) {
    const long long ez_index = offset3d(coord.i, coord.j + 1, coord.k, ez_ny, ez_nz);
    adjoint += ez_curl[ez_index] * inv_dy * adj_ez_post[ez_index];
  }
  adj_hx_mid[linear] = adjoint;
}

__global__ void reverse_electric_to_hy_standard_kernel(
    int64_t total,
    int hy_ny,
    int hy_nz,
    int ex_nx,
    int ex_ny,
    int ex_nz,
    int ez_nx,
    int ez_ny,
    int ez_nz,
    float* adj_hy_mid,
    const float* adj_hy_post,
    const float* adj_ex_post,
    const float* adj_ez_post,
    const float* ex_curl,
    const float* ez_curl,
    float inv_dx,
    float inv_dz) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), hy_ny, hy_nz);
  const int i = static_cast<int>(coord.i);
  const int j = static_cast<int>(coord.j);
  const int k = static_cast<int>(coord.k);
  float adjoint = adj_hy_post[linear];

  if (is_ex_active_index(i, j, k, ex_nx, ex_ny, ex_nz)) {
    const long long ex_index = offset3d(coord.i, coord.j, coord.k, ex_ny, ex_nz);
    adjoint -= ex_curl[ex_index] * inv_dz * adj_ex_post[ex_index];
  }
  if (is_ex_active_index(i, j, k + 1, ex_nx, ex_ny, ex_nz)) {
    const long long ex_index = offset3d(coord.i, coord.j, coord.k + 1, ex_ny, ex_nz);
    adjoint += ex_curl[ex_index] * inv_dz * adj_ex_post[ex_index];
  }
  if (is_ez_active_index(i, j, k, ez_nx, ez_ny, ez_nz)) {
    const long long ez_index = offset3d(coord.i, coord.j, coord.k, ez_ny, ez_nz);
    adjoint += ez_curl[ez_index] * inv_dx * adj_ez_post[ez_index];
  }
  if (is_ez_active_index(i + 1, j, k, ez_nx, ez_ny, ez_nz)) {
    const long long ez_index = offset3d(coord.i + 1, coord.j, coord.k, ez_ny, ez_nz);
    adjoint -= ez_curl[ez_index] * inv_dx * adj_ez_post[ez_index];
  }
  adj_hy_mid[linear] = adjoint;
}

__global__ void reverse_electric_to_hz_standard_kernel(
    int64_t total,
    int hz_ny,
    int hz_nz,
    int ex_nx,
    int ex_ny,
    int ex_nz,
    int ey_nx,
    int ey_ny,
    int ey_nz,
    float* adj_hz_mid,
    const float* adj_hz_post,
    const float* adj_ex_post,
    const float* adj_ey_post,
    const float* ex_curl,
    const float* ey_curl,
    float inv_dx,
    float inv_dy) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), hz_ny, hz_nz);
  const int i = static_cast<int>(coord.i);
  const int j = static_cast<int>(coord.j);
  const int k = static_cast<int>(coord.k);
  float adjoint = adj_hz_post[linear];

  if (is_ex_active_index(i, j, k, ex_nx, ex_ny, ex_nz)) {
    const long long ex_index = offset3d(coord.i, coord.j, coord.k, ex_ny, ex_nz);
    adjoint += ex_curl[ex_index] * inv_dy * adj_ex_post[ex_index];
  }
  if (is_ex_active_index(i, j + 1, k, ex_nx, ex_ny, ex_nz)) {
    const long long ex_index = offset3d(coord.i, coord.j + 1, coord.k, ex_ny, ex_nz);
    adjoint -= ex_curl[ex_index] * inv_dy * adj_ex_post[ex_index];
  }
  if (is_ey_active_index(i, j, k, ey_nx, ey_ny, ey_nz)) {
    const long long ey_index = offset3d(coord.i, coord.j, coord.k, ey_ny, ey_nz);
    adjoint -= ey_curl[ey_index] * inv_dx * adj_ey_post[ey_index];
  }
  if (is_ey_active_index(i + 1, j, k, ey_nx, ey_ny, ey_nz)) {
    const long long ey_index = offset3d(coord.i + 1, coord.j, coord.k, ey_ny, ey_nz);
    adjoint += ey_curl[ey_index] * inv_dx * adj_ey_post[ey_index];
  }
  adj_hz_mid[linear] = adjoint;
}

__global__ void reverse_magnetic_to_ex_standard_kernel(
    int64_t total,
    int ex_ny,
    int ex_nz,
    int hy_ny,
    int hy_nz,
    int hz_ny,
    int hz_nz,
    float* adj_ex_prev,
    float* grad_eps_ex,
    const float* adj_ex_post,
    const float* adj_hy_mid,
    const float* adj_hz_mid,
    const float* ex_decay,
    const float* ex_curl,
    const float* eps_ex,
    const float* hy_mid,
    const float* hz_mid,
    const float* hy_curl,
    const float* hz_curl,
    float inv_dy,
    float inv_dz,
    int y_low_mode,
    int y_high_mode,
    int z_low_mode,
    int z_high_mode) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ex_ny, ex_nz);
  const ElectricCellStatus status = resolve_electric_cell_status(
      static_cast<int>(coord.j), ex_ny, y_low_mode, y_high_mode,
      static_cast<int>(coord.k), ex_nz, z_low_mode, z_high_mode);
  float adjoint = 0.0f;
  float grad = 0.0f;
  if (status.inactive) {
    adjoint = adj_ex_post[linear];
  } else if (status.active) {
    const long long hy_index = offset3d(coord.i, coord.j, coord.k, hy_ny, hy_nz);
    const long long hy_prev_k = offset3d(coord.i, coord.j, coord.k - 1, hy_ny, hy_nz);
    const long long hz_index = offset3d(coord.i, coord.j, coord.k, hz_ny, hz_nz);
    const long long hz_prev_j = offset3d(coord.i, coord.j - 1, coord.k, hz_ny, hz_nz);
    const float curl_h = (hz_mid[hz_index] - hz_mid[hz_prev_j]) * inv_dy
        - (hy_mid[hy_index] - hy_mid[hy_prev_k]) * inv_dz;
    adjoint = adj_ex_post[linear] * ex_decay[linear];
    grad = -adj_ex_post[linear] * ex_curl[linear] * curl_h / eps_ex[linear];
  }
  if (static_cast<int>(coord.k) < hy_nz) {
    const long long hy_index = offset3d(coord.i, coord.j, coord.k, hy_ny, hy_nz);
    adjoint += hy_curl[hy_index] * inv_dz * adj_hy_mid[hy_index];
  }
  if (coord.k > 0) {
    const long long hy_index = offset3d(coord.i, coord.j, coord.k - 1, hy_ny, hy_nz);
    adjoint -= hy_curl[hy_index] * inv_dz * adj_hy_mid[hy_index];
  }
  if (static_cast<int>(coord.j) < hz_ny) {
    const long long hz_index = offset3d(coord.i, coord.j, coord.k, hz_ny, hz_nz);
    adjoint -= hz_curl[hz_index] * inv_dy * adj_hz_mid[hz_index];
  }
  if (coord.j > 0) {
    const long long hz_index = offset3d(coord.i, coord.j - 1, coord.k, hz_ny, hz_nz);
    adjoint += hz_curl[hz_index] * inv_dy * adj_hz_mid[hz_index];
  }
  adj_ex_prev[linear] = adjoint;
  grad_eps_ex[linear] = grad;
}

__global__ void reverse_magnetic_to_ey_standard_kernel(
    int64_t total,
    int ey_ny,
    int ey_nz,
    int hx_ny,
    int hx_nz,
    int hz_nx,
    int hz_ny,
    int hz_nz,
    float* adj_ey_prev,
    float* grad_eps_ey,
    const float* adj_ey_post,
    const float* adj_hx_mid,
    const float* adj_hz_mid,
    const float* ey_decay,
    const float* ey_curl,
    const float* eps_ey,
    const float* hx_mid,
    const float* hz_mid,
    const float* hx_curl,
    const float* hz_curl,
    float inv_dx,
    float inv_dz,
    int x_low_mode,
    int x_high_mode,
    int z_low_mode,
    int z_high_mode) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int ey_nx = static_cast<int>(total / (ey_ny * ey_nz));
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ey_ny, ey_nz);
  const ElectricCellStatus status = resolve_electric_cell_status(
      static_cast<int>(coord.i), ey_nx, x_low_mode, x_high_mode,
      static_cast<int>(coord.k), ey_nz, z_low_mode, z_high_mode);
  float adjoint = 0.0f;
  float grad = 0.0f;
  if (status.inactive) {
    adjoint = adj_ey_post[linear];
  } else if (status.active) {
    const long long hx_index = offset3d(coord.i, coord.j, coord.k, hx_ny, hx_nz);
    const long long hx_prev_k = offset3d(coord.i, coord.j, coord.k - 1, hx_ny, hx_nz);
    const long long hz_index = offset3d(coord.i, coord.j, coord.k, hz_ny, hz_nz);
    const long long hz_prev_i = offset3d(coord.i - 1, coord.j, coord.k, hz_ny, hz_nz);
    const float curl_h = (hx_mid[hx_index] - hx_mid[hx_prev_k]) * inv_dz
        - (hz_mid[hz_index] - hz_mid[hz_prev_i]) * inv_dx;
    adjoint = adj_ey_post[linear] * ey_decay[linear];
    grad = -adj_ey_post[linear] * ey_curl[linear] * curl_h / eps_ey[linear];
  }
  if (static_cast<int>(coord.k) < hx_nz) {
    const long long hx_index = offset3d(coord.i, coord.j, coord.k, hx_ny, hx_nz);
    adjoint -= hx_curl[hx_index] * inv_dz * adj_hx_mid[hx_index];
  }
  if (coord.k > 0) {
    const long long hx_index = offset3d(coord.i, coord.j, coord.k - 1, hx_ny, hx_nz);
    adjoint += hx_curl[hx_index] * inv_dz * adj_hx_mid[hx_index];
  }
  if (static_cast<int>(coord.i) < hz_nx) {
    const long long hz_index = offset3d(coord.i, coord.j, coord.k, hz_ny, hz_nz);
    adjoint += hz_curl[hz_index] * inv_dx * adj_hz_mid[hz_index];
  }
  if (coord.i > 0) {
    const long long hz_index = offset3d(coord.i - 1, coord.j, coord.k, hz_ny, hz_nz);
    adjoint -= hz_curl[hz_index] * inv_dx * adj_hz_mid[hz_index];
  }
  adj_ey_prev[linear] = adjoint;
  grad_eps_ey[linear] = grad;
}

__global__ void reverse_magnetic_to_ez_standard_kernel(
    int64_t total,
    int ez_ny,
    int ez_nz,
    int hx_ny,
    int hx_nz,
    int hy_nx,
    int hy_ny,
    int hy_nz,
    float* adj_ez_prev,
    float* grad_eps_ez,
    const float* adj_ez_post,
    const float* adj_hx_mid,
    const float* adj_hy_mid,
    const float* ez_decay,
    const float* ez_curl,
    const float* eps_ez,
    const float* hx_mid,
    const float* hy_mid,
    const float* hx_curl,
    const float* hy_curl,
    float inv_dx,
    float inv_dy,
    int x_low_mode,
    int x_high_mode,
    int y_low_mode,
    int y_high_mode) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int ez_nx = static_cast<int>(total / (ez_ny * ez_nz));
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ez_ny, ez_nz);
  const ElectricCellStatus status = resolve_electric_cell_status(
      static_cast<int>(coord.i), ez_nx, x_low_mode, x_high_mode,
      static_cast<int>(coord.j), ez_ny, y_low_mode, y_high_mode);
  float adjoint = 0.0f;
  float grad = 0.0f;
  if (status.inactive) {
    adjoint = adj_ez_post[linear];
  } else if (status.active) {
    const long long hx_index = offset3d(coord.i, coord.j, coord.k, hx_ny, hx_nz);
    const long long hx_prev_j = offset3d(coord.i, coord.j - 1, coord.k, hx_ny, hx_nz);
    const long long hy_index = offset3d(coord.i, coord.j, coord.k, hy_ny, hy_nz);
    const long long hy_prev_i = offset3d(coord.i - 1, coord.j, coord.k, hy_ny, hy_nz);
    const float curl_h = (hy_mid[hy_index] - hy_mid[hy_prev_i]) * inv_dx
        - (hx_mid[hx_index] - hx_mid[hx_prev_j]) * inv_dy;
    adjoint = adj_ez_post[linear] * ez_decay[linear];
    grad = -adj_ez_post[linear] * ez_curl[linear] * curl_h / eps_ez[linear];
  }
  if (static_cast<int>(coord.j) < hx_ny) {
    const long long hx_index = offset3d(coord.i, coord.j, coord.k, hx_ny, hx_nz);
    adjoint += hx_curl[hx_index] * inv_dy * adj_hx_mid[hx_index];
  }
  if (coord.j > 0) {
    const long long hx_index = offset3d(coord.i, coord.j - 1, coord.k, hx_ny, hx_nz);
    adjoint -= hx_curl[hx_index] * inv_dy * adj_hx_mid[hx_index];
  }
  if (static_cast<int>(coord.i) < hy_nx) {
    const long long hy_index = offset3d(coord.i, coord.j, coord.k, hy_ny, hy_nz);
    adjoint -= hy_curl[hy_index] * inv_dx * adj_hy_mid[hy_index];
  }
  if (coord.i > 0) {
    const long long hy_index = offset3d(coord.i - 1, coord.j, coord.k, hy_ny, hy_nz);
    adjoint += hy_curl[hy_index] * inv_dx * adj_hy_mid[hy_index];
  }
  adj_ez_prev[linear] = adjoint;
  grad_eps_ez[linear] = grad;
}

struct Complex2 {
  float real;
  float imag;
};

__device__ inline Complex2 complex_add(Complex2 lhs, Complex2 rhs) {
  return {lhs.real + rhs.real, lhs.imag + rhs.imag};
}

__device__ inline Complex2 complex_sub(Complex2 lhs, Complex2 rhs) {
  return {lhs.real - rhs.real, lhs.imag - rhs.imag};
}

__device__ inline Complex2 complex_scale(Complex2 value, float scale) {
  return {value.real * scale, value.imag * scale};
}

__device__ inline float complex_inner_real(Complex2 lhs, Complex2 rhs) {
  return lhs.real * rhs.real + lhs.imag * rhs.imag;
}

__device__ inline Complex2 complex_phase_positive(float phase_cos, float phase_sin, Complex2 value) {
  return {
      phase_cos * value.real - phase_sin * value.imag,
      phase_sin * value.real + phase_cos * value.imag,
  };
}

__device__ inline Complex2 complex_phase_negative(float phase_cos, float phase_sin, Complex2 value) {
  return {
      phase_cos * value.real + phase_sin * value.imag,
      phase_cos * value.imag - phase_sin * value.real,
  };
}

__device__ inline Complex2 load_complex_3d(
    const float* real_field,
    const float* imag_field,
    int i,
    int j,
    int k,
    int size_y,
    int size_z) {
  const long long index = offset3d(
      static_cast<unsigned int>(i),
      static_cast<unsigned int>(j),
      static_cast<unsigned int>(k),
      static_cast<unsigned int>(size_y),
      static_cast<unsigned int>(size_z));
  return {real_field[index], imag_field[index]};
}

__device__ inline Complex2 load_scaled_complex_adjoint(
    const float* adj_real,
    const float* adj_imag,
    const float* curl,
    int i,
    int j,
    int k,
    int size_y,
    int size_z,
    float sign) {
  const long long index = offset3d(
      static_cast<unsigned int>(i),
      static_cast<unsigned int>(j),
      static_cast<unsigned int>(k),
      static_cast<unsigned int>(size_y),
      static_cast<unsigned int>(size_z));
  const float scale = sign * curl[index];
  return {adj_real[index] * scale, adj_imag[index] * scale};
}

__device__ inline Complex2 bloch_backward_diff_axis(
    const float* real_field,
    const float* imag_field,
    int i,
    int j,
    int k,
    int size_x,
    int size_y,
    int size_z,
    int axis,
    float phase_cos,
    float phase_sin,
    float inv_delta) {
  const int coordinate = axis == 0 ? i : (axis == 1 ? j : k);
  const int field_size = axis == 0 ? size_x : (axis == 1 ? size_y : size_z);
  if (coordinate == 0 || coordinate == field_size) {
    Complex2 low;
    Complex2 high;
    if (axis == 0) {
      low = load_complex_3d(real_field, imag_field, 0, j, k, size_y, size_z);
      high = load_complex_3d(real_field, imag_field, size_x - 1, j, k, size_y, size_z);
    } else if (axis == 1) {
      low = load_complex_3d(real_field, imag_field, i, 0, k, size_y, size_z);
      high = load_complex_3d(real_field, imag_field, i, size_y - 1, k, size_y, size_z);
    } else {
      low = load_complex_3d(real_field, imag_field, i, j, 0, size_y, size_z);
      high = load_complex_3d(real_field, imag_field, i, j, size_z - 1, size_y, size_z);
    }
    if (coordinate == 0) {
      return complex_scale(complex_sub(low, complex_phase_negative(phase_cos, phase_sin, high)), inv_delta);
    }
    return complex_scale(complex_sub(complex_phase_positive(phase_cos, phase_sin, low), high), inv_delta);
  }

  Complex2 current;
  Complex2 previous;
  if (axis == 0) {
    current = load_complex_3d(real_field, imag_field, i, j, k, size_y, size_z);
    previous = load_complex_3d(real_field, imag_field, i - 1, j, k, size_y, size_z);
  } else if (axis == 1) {
    current = load_complex_3d(real_field, imag_field, i, j, k, size_y, size_z);
    previous = load_complex_3d(real_field, imag_field, i, j - 1, k, size_y, size_z);
  } else {
    current = load_complex_3d(real_field, imag_field, i, j, k, size_y, size_z);
    previous = load_complex_3d(real_field, imag_field, i, j, k - 1, size_y, size_z);
  }
  return complex_scale(complex_sub(current, previous), inv_delta);
}

__device__ inline Complex2 gather_bloch_backward_diff_adjoint_axis(
    const float* adj_real,
    const float* adj_imag,
    const float* curl,
    int i,
    int j,
    int k,
    int adj_size_x,
    int adj_size_y,
    int adj_size_z,
    int axis,
    float sign,
    float phase_cos,
    float phase_sin,
    float inv_delta) {
  const int coordinate = axis == 0 ? i : (axis == 1 ? j : k);
  const int adj_axis_size = axis == 0 ? adj_size_x : (axis == 1 ? adj_size_y : adj_size_z);
  const int field_size = adj_axis_size - 1;
  Complex2 value = {0.0f, 0.0f};

  if (coordinate > 0) {
    value = complex_add(value, complex_scale(load_scaled_complex_adjoint(
        adj_real, adj_imag, curl, i, j, k, adj_size_y, adj_size_z, sign), inv_delta));
  }
  if (coordinate + 1 < field_size) {
    int next_i = i;
    int next_j = j;
    int next_k = k;
    if (axis == 0) {
      ++next_i;
    } else if (axis == 1) {
      ++next_j;
    } else {
      ++next_k;
    }
    value = complex_sub(value, complex_scale(load_scaled_complex_adjoint(
        adj_real, adj_imag, curl, next_i, next_j, next_k, adj_size_y, adj_size_z, sign), inv_delta));
  }

  int low_i = i;
  int low_j = j;
  int low_k = k;
  int high_i = i;
  int high_j = j;
  int high_k = k;
  if (axis == 0) {
    low_i = 0;
    high_i = adj_size_x - 1;
  } else if (axis == 1) {
    low_j = 0;
    high_j = adj_size_y - 1;
  } else {
    low_k = 0;
    high_k = adj_size_z - 1;
  }
  const Complex2 low = load_scaled_complex_adjoint(
      adj_real, adj_imag, curl, low_i, low_j, low_k, adj_size_y, adj_size_z, sign);
  const Complex2 high = load_scaled_complex_adjoint(
      adj_real, adj_imag, curl, high_i, high_j, high_k, adj_size_y, adj_size_z, sign);
  if (coordinate == 0) {
    value = complex_add(value, complex_scale(low, inv_delta));
    value = complex_add(value, complex_scale(complex_phase_negative(phase_cos, phase_sin, high), inv_delta));
  }
  if (coordinate + 1 == field_size) {
    value = complex_sub(value, complex_scale(complex_phase_positive(phase_cos, phase_sin, low), inv_delta));
    value = complex_sub(value, complex_scale(high, inv_delta));
  }

  return value;
}

__global__ void reverse_electric_to_hx_bloch_kernel(
    int64_t total,
    int hx_ny,
    int hx_nz,
    int ey_nx,
    int ey_ny,
    int ey_nz,
    int ez_nx,
    int ez_ny,
    int ez_nz,
    float* adj_hx_mid_real,
    float* adj_hx_mid_imag,
    const float* adj_hx_post_real,
    const float* adj_hx_post_imag,
    const float* adj_ey_post_real,
    const float* adj_ey_post_imag,
    const float* adj_ez_post_real,
    const float* adj_ez_post_imag,
    const float* ey_curl,
    const float* ez_curl,
    float phase_cos_y,
    float phase_sin_y,
    float phase_cos_z,
    float phase_sin_z,
    float inv_dy,
    float inv_dz) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), hx_ny, hx_nz);
  const int i = static_cast<int>(coord.i);
  const int j = static_cast<int>(coord.j);
  const int k = static_cast<int>(coord.k);
  Complex2 adjoint = load_complex_3d(adj_hx_post_real, adj_hx_post_imag, i, j, k, hx_ny, hx_nz);
  adjoint = complex_add(adjoint, gather_bloch_backward_diff_adjoint_axis(
      adj_ey_post_real, adj_ey_post_imag, ey_curl, i, j, k, ey_nx, ey_ny, ey_nz,
      2, 1.0f, phase_cos_z, phase_sin_z, inv_dz));
  adjoint = complex_add(adjoint, gather_bloch_backward_diff_adjoint_axis(
      adj_ez_post_real, adj_ez_post_imag, ez_curl, i, j, k, ez_nx, ez_ny, ez_nz,
      1, -1.0f, phase_cos_y, phase_sin_y, inv_dy));
  adj_hx_mid_real[linear] = adjoint.real;
  adj_hx_mid_imag[linear] = adjoint.imag;
}

__global__ void reverse_electric_to_hy_bloch_kernel(
    int64_t total,
    int hy_ny,
    int hy_nz,
    int ex_nx,
    int ex_ny,
    int ex_nz,
    int ez_nx,
    int ez_ny,
    int ez_nz,
    float* adj_hy_mid_real,
    float* adj_hy_mid_imag,
    const float* adj_hy_post_real,
    const float* adj_hy_post_imag,
    const float* adj_ex_post_real,
    const float* adj_ex_post_imag,
    const float* adj_ez_post_real,
    const float* adj_ez_post_imag,
    const float* ex_curl,
    const float* ez_curl,
    float phase_cos_x,
    float phase_sin_x,
    float phase_cos_z,
    float phase_sin_z,
    float inv_dx,
    float inv_dz) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int hy_nx = static_cast<int>(total / (hy_ny * hy_nz));
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), hy_ny, hy_nz);
  const int i = static_cast<int>(coord.i);
  const int j = static_cast<int>(coord.j);
  const int k = static_cast<int>(coord.k);
  Complex2 adjoint = load_complex_3d(adj_hy_post_real, adj_hy_post_imag, i, j, k, hy_ny, hy_nz);
  adjoint = complex_add(adjoint, gather_bloch_backward_diff_adjoint_axis(
      adj_ex_post_real, adj_ex_post_imag, ex_curl, i, j, k, ex_nx, ex_ny, ex_nz,
      2, -1.0f, phase_cos_z, phase_sin_z, inv_dz));
  adjoint = complex_add(adjoint, gather_bloch_backward_diff_adjoint_axis(
      adj_ez_post_real, adj_ez_post_imag, ez_curl, i, j, k, ez_nx, ez_ny, ez_nz,
      0, 1.0f, phase_cos_x, phase_sin_x, inv_dx));
  adj_hy_mid_real[offset3d(coord.i, coord.j, coord.k, static_cast<unsigned int>(hy_ny), static_cast<unsigned int>(hy_nz))] = adjoint.real;
  adj_hy_mid_imag[offset3d(coord.i, coord.j, coord.k, static_cast<unsigned int>(hy_ny), static_cast<unsigned int>(hy_nz))] = adjoint.imag;
}

__global__ void reverse_electric_to_hz_bloch_kernel(
    int64_t total,
    int hz_ny,
    int hz_nz,
    int ex_nx,
    int ex_ny,
    int ex_nz,
    int ey_nx,
    int ey_ny,
    int ey_nz,
    float* adj_hz_mid_real,
    float* adj_hz_mid_imag,
    const float* adj_hz_post_real,
    const float* adj_hz_post_imag,
    const float* adj_ex_post_real,
    const float* adj_ex_post_imag,
    const float* adj_ey_post_real,
    const float* adj_ey_post_imag,
    const float* ex_curl,
    const float* ey_curl,
    float phase_cos_x,
    float phase_sin_x,
    float phase_cos_y,
    float phase_sin_y,
    float inv_dx,
    float inv_dy) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), hz_ny, hz_nz);
  const int i = static_cast<int>(coord.i);
  const int j = static_cast<int>(coord.j);
  const int k = static_cast<int>(coord.k);
  Complex2 adjoint = load_complex_3d(adj_hz_post_real, adj_hz_post_imag, i, j, k, hz_ny, hz_nz);
  adjoint = complex_add(adjoint, gather_bloch_backward_diff_adjoint_axis(
      adj_ex_post_real, adj_ex_post_imag, ex_curl, i, j, k, ex_nx, ex_ny, ex_nz,
      1, 1.0f, phase_cos_y, phase_sin_y, inv_dy));
  adjoint = complex_add(adjoint, gather_bloch_backward_diff_adjoint_axis(
      adj_ey_post_real, adj_ey_post_imag, ey_curl, i, j, k, ey_nx, ey_ny, ey_nz,
      0, -1.0f, phase_cos_x, phase_sin_x, inv_dx));
  adj_hz_mid_real[linear] = adjoint.real;
  adj_hz_mid_imag[linear] = adjoint.imag;
}

__global__ void reverse_magnetic_to_ex_bloch_kernel(
    int64_t total,
    int ex_ny,
    int ex_nz,
    int hy_nx,
    int hy_ny,
    int hy_nz,
    int hz_nx,
    int hz_ny,
    int hz_nz,
    float* adj_ex_prev_real,
    float* adj_ex_prev_imag,
    float* grad_eps_ex,
    const float* adj_ex_post_real,
    const float* adj_ex_post_imag,
    const float* adj_hy_mid_real,
    const float* adj_hy_mid_imag,
    const float* adj_hz_mid_real,
    const float* adj_hz_mid_imag,
    const float* ex_decay,
    const float* ex_curl,
    const float* eps_ex,
    const float* hy_mid_real,
    const float* hy_mid_imag,
    const float* hz_mid_real,
    const float* hz_mid_imag,
    const float* hy_curl,
    const float* hz_curl,
    float phase_cos_y,
    float phase_sin_y,
    float phase_cos_z,
    float phase_sin_z,
    float inv_dy,
    float inv_dz) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ex_ny, ex_nz);
  const int i = static_cast<int>(coord.i);
  const int j = static_cast<int>(coord.j);
  const int k = static_cast<int>(coord.k);
  const Complex2 adj_ex_post = load_complex_3d(adj_ex_post_real, adj_ex_post_imag, i, j, k, ex_ny, ex_nz);
  const Complex2 d_hz_dy = bloch_backward_diff_axis(
      hz_mid_real, hz_mid_imag, i, j, k, hz_nx, hz_ny, hz_nz, 1, phase_cos_y, phase_sin_y, inv_dy);
  const Complex2 d_hy_dz = bloch_backward_diff_axis(
      hy_mid_real, hy_mid_imag, i, j, k, hy_nx, hy_ny, hy_nz, 2, phase_cos_z, phase_sin_z, inv_dz);
  const Complex2 curl_h = complex_sub(d_hz_dy, d_hy_dz);
  float adjoint_real = adj_ex_post.real * ex_decay[linear];
  float adjoint_imag = adj_ex_post.imag * ex_decay[linear];
  float grad = -ex_curl[linear] * complex_inner_real(adj_ex_post, curl_h) / eps_ex[linear];

  if (k < hy_nz) {
    const long long hy_index = offset3d(coord.i, coord.j, coord.k, static_cast<unsigned int>(hy_ny), static_cast<unsigned int>(hy_nz));
    adjoint_real += hy_curl[hy_index] * inv_dz * adj_hy_mid_real[hy_index];
    adjoint_imag += hy_curl[hy_index] * inv_dz * adj_hy_mid_imag[hy_index];
  }
  if (k > 0) {
    const long long hy_index = offset3d(coord.i, coord.j, coord.k - 1, static_cast<unsigned int>(hy_ny), static_cast<unsigned int>(hy_nz));
    adjoint_real -= hy_curl[hy_index] * inv_dz * adj_hy_mid_real[hy_index];
    adjoint_imag -= hy_curl[hy_index] * inv_dz * adj_hy_mid_imag[hy_index];
  }
  if (j < hz_ny) {
    const long long hz_index = offset3d(coord.i, coord.j, coord.k, static_cast<unsigned int>(hz_ny), static_cast<unsigned int>(hz_nz));
    adjoint_real -= hz_curl[hz_index] * inv_dy * adj_hz_mid_real[hz_index];
    adjoint_imag -= hz_curl[hz_index] * inv_dy * adj_hz_mid_imag[hz_index];
  }
  if (j > 0) {
    const long long hz_index = offset3d(coord.i, coord.j - 1, coord.k, static_cast<unsigned int>(hz_ny), static_cast<unsigned int>(hz_nz));
    adjoint_real += hz_curl[hz_index] * inv_dy * adj_hz_mid_real[hz_index];
    adjoint_imag += hz_curl[hz_index] * inv_dy * adj_hz_mid_imag[hz_index];
  }
  adj_ex_prev_real[linear] = adjoint_real;
  adj_ex_prev_imag[linear] = adjoint_imag;
  grad_eps_ex[linear] = grad;
}

__global__ void reverse_magnetic_to_ey_bloch_kernel(
    int64_t total,
    int ey_ny,
    int ey_nz,
    int hx_nx,
    int hx_ny,
    int hx_nz,
    int hz_nx,
    int hz_ny,
    int hz_nz,
    float* adj_ey_prev_real,
    float* adj_ey_prev_imag,
    float* grad_eps_ey,
    const float* adj_ey_post_real,
    const float* adj_ey_post_imag,
    const float* adj_hx_mid_real,
    const float* adj_hx_mid_imag,
    const float* adj_hz_mid_real,
    const float* adj_hz_mid_imag,
    const float* ey_decay,
    const float* ey_curl,
    const float* eps_ey,
    const float* hx_mid_real,
    const float* hx_mid_imag,
    const float* hz_mid_real,
    const float* hz_mid_imag,
    const float* hx_curl,
    const float* hz_curl,
    float phase_cos_x,
    float phase_sin_x,
    float phase_cos_z,
    float phase_sin_z,
    float inv_dx,
    float inv_dz) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int ey_nx = static_cast<int>(total / (ey_ny * ey_nz));
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ey_ny, ey_nz);
  const int i = static_cast<int>(coord.i);
  const int j = static_cast<int>(coord.j);
  const int k = static_cast<int>(coord.k);
  const Complex2 adj_ey_post = load_complex_3d(adj_ey_post_real, adj_ey_post_imag, i, j, k, ey_ny, ey_nz);
  const Complex2 d_hx_dz = bloch_backward_diff_axis(
      hx_mid_real, hx_mid_imag, i, j, k, hx_nx, hx_ny, hx_nz, 2, phase_cos_z, phase_sin_z, inv_dz);
  const Complex2 d_hz_dx = bloch_backward_diff_axis(
      hz_mid_real, hz_mid_imag, i, j, k, hz_nx, hz_ny, hz_nz, 0, phase_cos_x, phase_sin_x, inv_dx);
  const Complex2 curl_h = complex_sub(d_hx_dz, d_hz_dx);
  float adjoint_real = adj_ey_post.real * ey_decay[linear];
  float adjoint_imag = adj_ey_post.imag * ey_decay[linear];
  float grad = -ey_curl[linear] * complex_inner_real(adj_ey_post, curl_h) / eps_ey[linear];

  if (k < hx_nz) {
    const long long hx_index = offset3d(coord.i, coord.j, coord.k, static_cast<unsigned int>(hx_ny), static_cast<unsigned int>(hx_nz));
    adjoint_real -= hx_curl[hx_index] * inv_dz * adj_hx_mid_real[hx_index];
    adjoint_imag -= hx_curl[hx_index] * inv_dz * adj_hx_mid_imag[hx_index];
  }
  if (k > 0) {
    const long long hx_index = offset3d(coord.i, coord.j, coord.k - 1, static_cast<unsigned int>(hx_ny), static_cast<unsigned int>(hx_nz));
    adjoint_real += hx_curl[hx_index] * inv_dz * adj_hx_mid_real[hx_index];
    adjoint_imag += hx_curl[hx_index] * inv_dz * adj_hx_mid_imag[hx_index];
  }
  if (i < hz_nx) {
    const long long hz_index = offset3d(coord.i, coord.j, coord.k, static_cast<unsigned int>(hz_ny), static_cast<unsigned int>(hz_nz));
    adjoint_real += hz_curl[hz_index] * inv_dx * adj_hz_mid_real[hz_index];
    adjoint_imag += hz_curl[hz_index] * inv_dx * adj_hz_mid_imag[hz_index];
  }
  if (i > 0) {
    const long long hz_index = offset3d(coord.i - 1, coord.j, coord.k, static_cast<unsigned int>(hz_ny), static_cast<unsigned int>(hz_nz));
    adjoint_real -= hz_curl[hz_index] * inv_dx * adj_hz_mid_real[hz_index];
    adjoint_imag -= hz_curl[hz_index] * inv_dx * adj_hz_mid_imag[hz_index];
  }
  adj_ey_prev_real[linear] = adjoint_real;
  adj_ey_prev_imag[linear] = adjoint_imag;
  grad_eps_ey[linear] = grad;
}

__global__ void reverse_magnetic_to_ez_bloch_kernel(
    int64_t total,
    int ez_ny,
    int ez_nz,
    int hx_nx,
    int hx_ny,
    int hx_nz,
    int hy_nx,
    int hy_ny,
    int hy_nz,
    float* adj_ez_prev_real,
    float* adj_ez_prev_imag,
    float* grad_eps_ez,
    const float* adj_ez_post_real,
    const float* adj_ez_post_imag,
    const float* adj_hx_mid_real,
    const float* adj_hx_mid_imag,
    const float* adj_hy_mid_real,
    const float* adj_hy_mid_imag,
    const float* ez_decay,
    const float* ez_curl,
    const float* eps_ez,
    const float* hx_mid_real,
    const float* hx_mid_imag,
    const float* hy_mid_real,
    const float* hy_mid_imag,
    const float* hx_curl,
    const float* hy_curl,
    float phase_cos_x,
    float phase_sin_x,
    float phase_cos_y,
    float phase_sin_y,
    float inv_dx,
    float inv_dy) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int ez_nx = static_cast<int>(total / (ez_ny * ez_nz));
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), ez_ny, ez_nz);
  const int i = static_cast<int>(coord.i);
  const int j = static_cast<int>(coord.j);
  const int k = static_cast<int>(coord.k);
  const Complex2 adj_ez_post = load_complex_3d(adj_ez_post_real, adj_ez_post_imag, i, j, k, ez_ny, ez_nz);
  const Complex2 d_hy_dx = bloch_backward_diff_axis(
      hy_mid_real, hy_mid_imag, i, j, k, hy_nx, hy_ny, hy_nz, 0, phase_cos_x, phase_sin_x, inv_dx);
  const Complex2 d_hx_dy = bloch_backward_diff_axis(
      hx_mid_real, hx_mid_imag, i, j, k, hx_nx, hx_ny, hx_nz, 1, phase_cos_y, phase_sin_y, inv_dy);
  const Complex2 curl_h = complex_sub(d_hy_dx, d_hx_dy);
  float adjoint_real = adj_ez_post.real * ez_decay[linear];
  float adjoint_imag = adj_ez_post.imag * ez_decay[linear];
  float grad = -ez_curl[linear] * complex_inner_real(adj_ez_post, curl_h) / eps_ez[linear];

  if (j < hx_ny) {
    const long long hx_index = offset3d(coord.i, coord.j, coord.k, static_cast<unsigned int>(hx_ny), static_cast<unsigned int>(hx_nz));
    adjoint_real += hx_curl[hx_index] * inv_dy * adj_hx_mid_real[hx_index];
    adjoint_imag += hx_curl[hx_index] * inv_dy * adj_hx_mid_imag[hx_index];
  }
  if (j > 0) {
    const long long hx_index = offset3d(coord.i, coord.j - 1, coord.k, static_cast<unsigned int>(hx_ny), static_cast<unsigned int>(hx_nz));
    adjoint_real -= hx_curl[hx_index] * inv_dy * adj_hx_mid_real[hx_index];
    adjoint_imag -= hx_curl[hx_index] * inv_dy * adj_hx_mid_imag[hx_index];
  }
  if (i < hy_nx) {
    const long long hy_index = offset3d(coord.i, coord.j, coord.k, static_cast<unsigned int>(hy_ny), static_cast<unsigned int>(hy_nz));
    adjoint_real -= hy_curl[hy_index] * inv_dx * adj_hy_mid_real[hy_index];
    adjoint_imag -= hy_curl[hy_index] * inv_dx * adj_hy_mid_imag[hy_index];
  }
  if (i > 0) {
    const long long hy_index = offset3d(coord.i - 1, coord.j, coord.k, static_cast<unsigned int>(hy_ny), static_cast<unsigned int>(hy_nz));
    adjoint_real += hy_curl[hy_index] * inv_dx * adj_hy_mid_real[hy_index];
    adjoint_imag += hy_curl[hy_index] * inv_dx * adj_hy_mid_imag[hy_index];
  }
  adj_ez_prev_real[linear] = adjoint_real;
  adj_ez_prev_imag[linear] = adjoint_imag;
  grad_eps_ez[linear] = grad;
}

__global__ void accumulate_diff_adjoint_kernel(
    int64_t total,
    int field_ny,
    int field_nz,
    int diff_nx,
    int diff_ny,
    int diff_nz,
    int axis,
    bool forward,
    float inv_delta,
    float* field_grad,
    const float* diff_grad) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int field_nx = static_cast<int>(total / (field_ny * field_nz));
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), field_ny, field_nz);
  const int coords[3] = {static_cast<int>(coord.i), static_cast<int>(coord.j), static_cast<int>(coord.k)};
  const int field_sizes[3] = {field_nx, field_ny, field_nz};
  const int diff_sizes[3] = {diff_nx, diff_ny, diff_nz};

  float value = 0.0f;
  if (forward) {
    if (coords[axis] < diff_sizes[axis] && diff_index_valid(coords[0], coords[1], coords[2], diff_nx, diff_ny, diff_nz)) {
      value -= inv_delta * diff_grad_value(diff_grad, coords[0], coords[1], coords[2], diff_ny, diff_nz);
    }
    if (coords[axis] > 0) {
      int prev[3] = {coords[0], coords[1], coords[2]};
      prev[axis] -= 1;
      if (diff_index_valid(prev[0], prev[1], prev[2], diff_nx, diff_ny, diff_nz)) {
        value += inv_delta * diff_grad_value(diff_grad, prev[0], prev[1], prev[2], diff_ny, diff_nz);
      }
    }
  } else {
    if (coords[axis] > 0 && diff_index_valid(coords[0], coords[1], coords[2], diff_nx, diff_ny, diff_nz)) {
      value += inv_delta * diff_grad_value(diff_grad, coords[0], coords[1], coords[2], diff_ny, diff_nz);
    }
    if (coords[axis] + 1 < field_sizes[axis]) {
      int next[3] = {coords[0], coords[1], coords[2]};
      next[axis] += 1;
      if (diff_index_valid(next[0], next[1], next[2], diff_nx, diff_ny, diff_nz)) {
        value -= inv_delta * diff_grad_value(diff_grad, next[0], next[1], next[2], diff_ny, diff_nz);
      }
    }
  }
  field_grad[linear] += value;
}

__global__ void reverse_debye_current_kernel(
    int64_t total,
    float* adj_electric_prev,
    float* adj_polarization_prev,
    const float* adj_polarization_post,
    const float* adj_current_post,
    const float* drive,
    double decay,
    double dt) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const float adj_internal = adj_polarization_post[index] + adj_current_post[index] / static_cast<float>(dt);
  adj_electric_prev[index] += drive[index] * adj_internal;
  adj_polarization_prev[index] += static_cast<float>(decay) * adj_internal - adj_current_post[index] / static_cast<float>(dt);
}

__global__ void reverse_drude_current_kernel(
    int64_t total,
    float* adj_electric_prev,
    float* adj_current_prev,
    const float* adj_current_post,
    const float* drive,
    double decay) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  adj_electric_prev[index] += drive[index] * adj_current_post[index];
  adj_current_prev[index] += static_cast<float>(decay) * adj_current_post[index];
}

__global__ void reverse_lorentz_current_kernel(
    int64_t total,
    float* adj_electric_prev,
    float* adj_polarization_prev,
    float* adj_current_prev,
    const float* adj_polarization_post,
    const float* adj_current_post,
    const float* drive,
    double decay,
    double restoring,
    double dt) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const float adj_internal = adj_current_post[index] + static_cast<float>(dt) * adj_polarization_post[index];
  adj_electric_prev[index] += drive[index] * adj_internal;
  adj_polarization_prev[index] += adj_polarization_post[index] - static_cast<float>(restoring) * adj_internal;
  adj_current_prev[index] += static_cast<float>(decay) * adj_internal;
}

__global__ void reverse_tfsf_auxiliary_electric_kernel(
    int64_t total,
    int64_t magnetic_total,
    int64_t source_index,
    float* adj_electric_prev,
    float* adj_magnetic_after,
    const float* adj_electric_post,
    const float* electric_decay,
    const float* electric_curl) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const bool overwritten = index == source_index || index == total - 1;
  const float adjoint = adj_electric_post[index];
  if (index == 0 && !overwritten) {
    adj_electric_prev[index] += adjoint;
    return;
  }
  if (index > 0 && index + 1 < total && !overwritten) {
    adj_electric_prev[index] += electric_decay[index] * adjoint;
    const float value = electric_curl[index] * adjoint;
    const int64_t lower = index - 1;
    const int64_t upper = index;
    if (lower >= 0 && lower < magnetic_total) {
      atomicAdd(adj_magnetic_after + lower, value);
    }
    if (upper >= 0 && upper < magnetic_total) {
      atomicAdd(adj_magnetic_after + upper, -value);
    }
  }
}

__global__ void reverse_electric_component_cpml_kernel(
    int64_t total,
    int component,
    int prev_ny,
    int prev_nz,
    int h_pos_ny,
    int h_pos_nz,
    int h_neg_ny,
    int h_neg_nz,
    float* adj_prev,
    float* grad_eps,
    float* adj_psi_pos_prev,
    float* adj_psi_neg_prev,
    float* adj_d_pos,
    float* adj_d_neg,
    const float* adj_post,
    const float* adj_psi_pos_post,
    const float* adj_psi_neg_post,
    const float* decay,
    const float* curl,
    const float* eps,
    const float* psi_pos,
    const float* psi_neg,
    const float* b_pos,
    const float* c_pos,
    const float* inv_kappa_pos,
    const float* b_neg,
    const float* c_neg,
    const float* inv_kappa_neg,
    const float* h_pos_mid,
    const float* h_neg_mid,
    float inv_pos,
    float inv_neg,
    int low_mode_a,
    int high_mode_a,
    int low_mode_b,
    int high_mode_b) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int prev_nx = static_cast<int>(total / (prev_ny * prev_nz));
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), prev_ny, prev_nz);
  const int i = static_cast<int>(coord.i);
  const int j = static_cast<int>(coord.j);
  const int k = static_cast<int>(coord.k);

  int coord_a = j;
  int size_a = prev_ny;
  int coord_b = k;
  int size_b = prev_nz;
  int pos_coeff_index = j;
  int neg_coeff_index = k;

  if (component == 0) {
  } else if (component == 1) {
    coord_a = i;
    size_a = prev_nx;
    coord_b = k;
    size_b = prev_nz;
    pos_coeff_index = k;
    neg_coeff_index = i;
  } else {
    coord_a = i;
    size_a = prev_nx;
    coord_b = j;
    size_b = prev_ny;
    pos_coeff_index = i;
    neg_coeff_index = j;
  }

  const ElectricCellStatus status = resolve_electric_cell_status(
      coord_a, size_a, low_mode_a, high_mode_a,
      coord_b, size_b, low_mode_b, high_mode_b);
  float adjoint = 0.0f;
  float grad = 0.0f;
  float adj_psi_pos = adj_psi_pos_post[linear];
  float adj_psi_neg = adj_psi_neg_post[linear];
  float out_adj_d_pos = 0.0f;
  float out_adj_d_neg = 0.0f;

  if (status.inactive) {
    adjoint = adj_post[linear];
  } else if (status.active) {
    float d_pos = 0.0f;
    float d_neg = 0.0f;
    if (component == 0) {
      d_pos = (h_pos_mid[offset3d(coord.i, coord.j, coord.k, h_pos_ny, h_pos_nz)]
               - h_pos_mid[offset3d(coord.i, coord.j - 1, coord.k, h_pos_ny, h_pos_nz)]) * inv_pos;
      d_neg = (h_neg_mid[offset3d(coord.i, coord.j, coord.k, h_neg_ny, h_neg_nz)]
               - h_neg_mid[offset3d(coord.i, coord.j, coord.k - 1, h_neg_ny, h_neg_nz)]) * inv_neg;
    } else if (component == 1) {
      d_pos = (h_pos_mid[offset3d(coord.i, coord.j, coord.k, h_pos_ny, h_pos_nz)]
               - h_pos_mid[offset3d(coord.i, coord.j, coord.k - 1, h_pos_ny, h_pos_nz)]) * inv_pos;
      d_neg = (h_neg_mid[offset3d(coord.i, coord.j, coord.k, h_neg_ny, h_neg_nz)]
               - h_neg_mid[offset3d(coord.i - 1, coord.j, coord.k, h_neg_ny, h_neg_nz)]) * inv_neg;
    } else {
      d_pos = (h_pos_mid[offset3d(coord.i, coord.j, coord.k, h_pos_ny, h_pos_nz)]
               - h_pos_mid[offset3d(coord.i - 1, coord.j, coord.k, h_pos_ny, h_pos_nz)]) * inv_pos;
      d_neg = (h_neg_mid[offset3d(coord.i, coord.j, coord.k, h_neg_ny, h_neg_nz)]
               - h_neg_mid[offset3d(coord.i, coord.j - 1, coord.k, h_neg_ny, h_neg_nz)]) * inv_neg;
    }
    const float psi_pos_candidate = b_pos[pos_coeff_index] * psi_pos[linear] + c_pos[pos_coeff_index] * d_pos;
    const float psi_neg_candidate = b_neg[neg_coeff_index] * psi_neg[linear] + c_neg[neg_coeff_index] * d_neg;
    const float curl_h = (d_pos * inv_kappa_pos[pos_coeff_index] + psi_pos_candidate)
        - (d_neg * inv_kappa_neg[neg_coeff_index] + psi_neg_candidate);
    const float adj_curl_h = adj_post[linear] * curl[linear];
    adjoint = adj_post[linear] * decay[linear];
    grad = -adj_post[linear] * curl[linear] * curl_h / eps[linear];
    adj_psi_pos = b_pos[pos_coeff_index] * (adj_psi_pos_post[linear] + adj_curl_h);
    adj_psi_neg = b_neg[neg_coeff_index] * (adj_psi_neg_post[linear] - adj_curl_h);
    out_adj_d_pos = inv_kappa_pos[pos_coeff_index] * adj_curl_h + c_pos[pos_coeff_index] * (adj_psi_pos_post[linear] + adj_curl_h);
    out_adj_d_neg = -inv_kappa_neg[neg_coeff_index] * adj_curl_h + c_neg[neg_coeff_index] * (adj_psi_neg_post[linear] - adj_curl_h);
  }

  adj_prev[linear] = adjoint;
  grad_eps[linear] = grad;
  adj_psi_pos_prev[linear] = adj_psi_pos;
  adj_psi_neg_prev[linear] = adj_psi_neg;
  adj_d_pos[linear] = out_adj_d_pos;
  adj_d_neg[linear] = out_adj_d_neg;
}

__global__ void reverse_magnetic_component_cpml_kernel(
    int64_t total,
    int component,
    int prev_ny,
    int prev_nz,
    float* adj_prev,
    float* adj_psi_pos_prev,
    float* adj_psi_neg_prev,
    float* adj_d_pos,
    float* adj_d_neg,
    const float* adj_post,
    const float* adj_psi_pos_post,
    const float* adj_psi_neg_post,
    const float* decay,
    const float* curl,
    const float* b_pos,
    const float* c_pos,
    const float* inv_kappa_pos,
    const float* b_neg,
    const float* c_neg,
    const float* inv_kappa_neg) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), prev_ny, prev_nz);
  int pos_coeff_index = static_cast<int>(coord.j);
  int neg_coeff_index = static_cast<int>(coord.k);
  if (component == 1) {
    pos_coeff_index = static_cast<int>(coord.k);
    neg_coeff_index = static_cast<int>(coord.i);
  } else if (component == 2) {
    pos_coeff_index = static_cast<int>(coord.i);
    neg_coeff_index = static_cast<int>(coord.j);
  }

  const float adj_curl_e = -curl[linear] * adj_post[linear];
  const float adj_psi_pos_candidate = adj_psi_pos_post[linear] + adj_curl_e;
  const float adj_psi_neg_candidate = adj_psi_neg_post[linear] - adj_curl_e;
  adj_prev[linear] = adj_post[linear] * decay[linear];
  adj_psi_pos_prev[linear] = b_pos[pos_coeff_index] * adj_psi_pos_candidate;
  adj_psi_neg_prev[linear] = b_neg[neg_coeff_index] * adj_psi_neg_candidate;
  adj_d_pos[linear] = inv_kappa_pos[pos_coeff_index] * adj_curl_e + c_pos[pos_coeff_index] * adj_psi_pos_candidate;
  adj_d_neg[linear] = -inv_kappa_neg[neg_coeff_index] * adj_curl_e + c_neg[neg_coeff_index] * adj_psi_neg_candidate;
}

__global__ void accumulate_tfsf_scalar_sample_adjoint_kernel(
    int64_t total,
    int patch_ny,
    int patch_nz,
    int64_t sample_index,
    float component_scale,
    float* adj_aux_field,
    const float* adj_field_patch,
    const float* coeff_patch) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  (void)patch_ny;
  (void)patch_nz;
  atomicAdd(adj_aux_field + sample_index, component_scale * adj_field_patch[linear] * coeff_patch[linear]);
}

__global__ void accumulate_tfsf_line_sample_adjoint_kernel(
    int64_t total,
    int patch_ny,
    int patch_nz,
    int sample_axis_code,
    float component_scale,
    float* adj_aux_field,
    const float* adj_field_patch,
    const float* coeff_patch,
    const int* sample_indices) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D coord = unflatten3d(static_cast<unsigned int>(linear), patch_ny, patch_nz);
  const int sample_linear = sample_axis_code == 0
      ? static_cast<int>(coord.i)
      : (sample_axis_code == 1 ? static_cast<int>(coord.j) : static_cast<int>(coord.k));
  const int sample_index = sample_indices[sample_linear];
  atomicAdd(adj_aux_field + sample_index, component_scale * adj_field_patch[linear] * coeff_patch[linear]);
}

__global__ void accumulate_tfsf_interpolated_sample_adjoint_kernel(
    int64_t total,
    int64_t adj_aux_count,
    float origin,
    float ds,
    float component_scale,
    float* adj_aux_field,
    const float* adj_field_patch,
    const float* coeff_patch,
    const float* sample_positions) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total || adj_aux_count <= 0) {
    return;
  }
  const int64_t last_index = adj_aux_count - 1;
  float coord = ds > 0.0f ? (sample_positions[linear] - origin) / ds : 0.0f;
  coord = fminf(fmaxf(coord, 0.0f), static_cast<float>(last_index));
  const int64_t lower = static_cast<int64_t>(floorf(coord));
  const int64_t upper = lower + 1 < last_index ? lower + 1 : last_index;
  const float frac = coord - static_cast<float>(lower);
  const float value = component_scale * adj_field_patch[linear] * coeff_patch[linear];
  atomicAdd(adj_aux_field + lower, value * (1.0f - frac));
  if (upper != lower) {
    atomicAdd(adj_aux_field + upper, value * frac);
  }
}

__global__ void reverse_tfsf_auxiliary_magnetic_kernel(
    int64_t total,
    float* adj_electric_prev,
    float* adj_magnetic_prev,
    const float* adj_magnetic_after,
    const float* magnetic_decay,
    const float* magnetic_curl) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const float adjoint = adj_magnetic_after[index];
  adj_magnetic_prev[index] = magnetic_decay[index] * adjoint;
  const float value = magnetic_curl[index] * adjoint;
  atomicAdd(adj_electric_prev + index, value);
  atomicAdd(adj_electric_prev + index + 1, -value);
}

}  // namespace

void reverse_magnetic_adjoint_decay_cuda(
    at::Tensor adj_prev,
    const at::Tensor& adj_mid,
    const at::Tensor& decay) {
  check_field(adj_prev, "adj_prev");
  check_matching_field(adj_prev, adj_mid, "adj_mid");
  check_matching_field(adj_prev, decay, "decay");
  const c10::cuda::CUDAGuard device_guard(adj_prev.device());
  const int64_t total = adj_prev.numel();
  reverse_magnetic_decay_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      adj_prev.data_ptr<float>(),
      adj_mid.data_ptr<float>(),
      decay.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void reverse_electric_adjoint_to_hx_standard_cuda(
    at::Tensor adj_hx_mid,
    const at::Tensor& adj_hx_post,
    const at::Tensor& adj_ey_post,
    const at::Tensor& adj_ez_post,
    const at::Tensor& ey_curl,
    const at::Tensor& ez_curl,
    double inv_dy,
    double inv_dz) {
  check_field(adj_hx_mid, "adj_hx_mid");
  check_matching_field(adj_hx_mid, adj_hx_post, "adj_hx_post");
  check_field(adj_ey_post, "adj_ey_post");
  check_matching_field(adj_ey_post, ey_curl, "ey_curl");
  check_field(adj_ez_post, "adj_ez_post");
  check_matching_field(adj_ez_post, ez_curl, "ez_curl");
  const c10::cuda::CUDAGuard device_guard(adj_hx_mid.device());
  const int64_t total = adj_hx_mid.numel();
  reverse_electric_to_hx_standard_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(adj_hx_mid.size(1)),
      static_cast<int>(adj_hx_mid.size(2)),
      static_cast<int>(adj_ey_post.size(0)),
      static_cast<int>(adj_ey_post.size(1)),
      static_cast<int>(adj_ey_post.size(2)),
      static_cast<int>(adj_ez_post.size(0)),
      static_cast<int>(adj_ez_post.size(1)),
      static_cast<int>(adj_ez_post.size(2)),
      adj_hx_mid.data_ptr<float>(),
      adj_hx_post.data_ptr<float>(),
      adj_ey_post.data_ptr<float>(),
      adj_ez_post.data_ptr<float>(),
      ey_curl.data_ptr<float>(),
      ez_curl.data_ptr<float>(),
      static_cast<float>(inv_dy),
      static_cast<float>(inv_dz));
  WITWIN_CUDA_CHECK();
}

void reverse_electric_adjoint_to_hy_standard_cuda(
    at::Tensor adj_hy_mid,
    const at::Tensor& adj_hy_post,
    const at::Tensor& adj_ex_post,
    const at::Tensor& adj_ez_post,
    const at::Tensor& ex_curl,
    const at::Tensor& ez_curl,
    double inv_dx,
    double inv_dz) {
  check_field(adj_hy_mid, "adj_hy_mid");
  check_matching_field(adj_hy_mid, adj_hy_post, "adj_hy_post");
  check_field(adj_ex_post, "adj_ex_post");
  check_matching_field(adj_ex_post, ex_curl, "ex_curl");
  check_field(adj_ez_post, "adj_ez_post");
  check_matching_field(adj_ez_post, ez_curl, "ez_curl");
  const c10::cuda::CUDAGuard device_guard(adj_hy_mid.device());
  const int64_t total = adj_hy_mid.numel();
  reverse_electric_to_hy_standard_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(adj_hy_mid.size(1)),
      static_cast<int>(adj_hy_mid.size(2)),
      static_cast<int>(adj_ex_post.size(0)),
      static_cast<int>(adj_ex_post.size(1)),
      static_cast<int>(adj_ex_post.size(2)),
      static_cast<int>(adj_ez_post.size(0)),
      static_cast<int>(adj_ez_post.size(1)),
      static_cast<int>(adj_ez_post.size(2)),
      adj_hy_mid.data_ptr<float>(),
      adj_hy_post.data_ptr<float>(),
      adj_ex_post.data_ptr<float>(),
      adj_ez_post.data_ptr<float>(),
      ex_curl.data_ptr<float>(),
      ez_curl.data_ptr<float>(),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dz));
  WITWIN_CUDA_CHECK();
}

void reverse_electric_adjoint_to_hz_standard_cuda(
    at::Tensor adj_hz_mid,
    const at::Tensor& adj_hz_post,
    const at::Tensor& adj_ex_post,
    const at::Tensor& adj_ey_post,
    const at::Tensor& ex_curl,
    const at::Tensor& ey_curl,
    double inv_dx,
    double inv_dy) {
  check_field(adj_hz_mid, "adj_hz_mid");
  check_matching_field(adj_hz_mid, adj_hz_post, "adj_hz_post");
  check_field(adj_ex_post, "adj_ex_post");
  check_matching_field(adj_ex_post, ex_curl, "ex_curl");
  check_field(adj_ey_post, "adj_ey_post");
  check_matching_field(adj_ey_post, ey_curl, "ey_curl");
  const c10::cuda::CUDAGuard device_guard(adj_hz_mid.device());
  const int64_t total = adj_hz_mid.numel();
  reverse_electric_to_hz_standard_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(adj_hz_mid.size(1)),
      static_cast<int>(adj_hz_mid.size(2)),
      static_cast<int>(adj_ex_post.size(0)),
      static_cast<int>(adj_ex_post.size(1)),
      static_cast<int>(adj_ex_post.size(2)),
      static_cast<int>(adj_ey_post.size(0)),
      static_cast<int>(adj_ey_post.size(1)),
      static_cast<int>(adj_ey_post.size(2)),
      adj_hz_mid.data_ptr<float>(),
      adj_hz_post.data_ptr<float>(),
      adj_ex_post.data_ptr<float>(),
      adj_ey_post.data_ptr<float>(),
      ex_curl.data_ptr<float>(),
      ey_curl.data_ptr<float>(),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dy));
  WITWIN_CUDA_CHECK();
}

void reverse_magnetic_adjoint_to_ex_standard_cuda(
    at::Tensor adj_ex_prev,
    at::Tensor grad_eps_ex,
    const at::Tensor& adj_ex_post,
    const at::Tensor& adj_hy_mid,
    const at::Tensor& adj_hz_mid,
    const at::Tensor& ex_decay,
    const at::Tensor& ex_curl,
    const at::Tensor& eps_ex,
    const at::Tensor& hy_mid,
    const at::Tensor& hz_mid,
    const at::Tensor& hy_curl,
    const at::Tensor& hz_curl,
    double inv_dy,
    double inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  check_field(adj_ex_prev, "adj_ex_prev");
  check_matching_field(adj_ex_prev, grad_eps_ex, "grad_eps_ex");
  check_matching_field(adj_ex_prev, adj_ex_post, "adj_ex_post");
  check_matching_field(adj_ex_prev, ex_decay, "ex_decay");
  check_matching_field(adj_ex_prev, ex_curl, "ex_curl");
  check_matching_field(adj_ex_prev, eps_ex, "eps_ex");
  check_field(adj_hy_mid, "adj_hy_mid");
  check_matching_field(adj_hy_mid, hy_mid, "hy_mid");
  check_matching_field(adj_hy_mid, hy_curl, "hy_curl");
  check_field(adj_hz_mid, "adj_hz_mid");
  check_matching_field(adj_hz_mid, hz_mid, "hz_mid");
  check_matching_field(adj_hz_mid, hz_curl, "hz_curl");
  const c10::cuda::CUDAGuard device_guard(adj_ex_prev.device());
  const int64_t total = adj_ex_prev.numel();
  reverse_magnetic_to_ex_standard_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(adj_ex_prev.size(1)),
      static_cast<int>(adj_ex_prev.size(2)),
      static_cast<int>(adj_hy_mid.size(1)),
      static_cast<int>(adj_hy_mid.size(2)),
      static_cast<int>(adj_hz_mid.size(1)),
      static_cast<int>(adj_hz_mid.size(2)),
      adj_ex_prev.data_ptr<float>(),
      grad_eps_ex.data_ptr<float>(),
      adj_ex_post.data_ptr<float>(),
      adj_hy_mid.data_ptr<float>(),
      adj_hz_mid.data_ptr<float>(),
      ex_decay.data_ptr<float>(),
      ex_curl.data_ptr<float>(),
      eps_ex.data_ptr<float>(),
      hy_mid.data_ptr<float>(),
      hz_mid.data_ptr<float>(),
      hy_curl.data_ptr<float>(),
      hz_curl.data_ptr<float>(),
      static_cast<float>(inv_dy),
      static_cast<float>(inv_dz),
      static_cast<int>(y_low_mode),
      static_cast<int>(y_high_mode),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode));
  WITWIN_CUDA_CHECK();
}

void reverse_magnetic_adjoint_to_ey_standard_cuda(
    at::Tensor adj_ey_prev,
    at::Tensor grad_eps_ey,
    const at::Tensor& adj_ey_post,
    const at::Tensor& adj_hx_mid,
    const at::Tensor& adj_hz_mid,
    const at::Tensor& ey_decay,
    const at::Tensor& ey_curl,
    const at::Tensor& eps_ey,
    const at::Tensor& hx_mid,
    const at::Tensor& hz_mid,
    const at::Tensor& hx_curl,
    const at::Tensor& hz_curl,
    double inv_dx,
    double inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  check_field(adj_ey_prev, "adj_ey_prev");
  check_matching_field(adj_ey_prev, grad_eps_ey, "grad_eps_ey");
  check_matching_field(adj_ey_prev, adj_ey_post, "adj_ey_post");
  check_matching_field(adj_ey_prev, ey_decay, "ey_decay");
  check_matching_field(adj_ey_prev, ey_curl, "ey_curl");
  check_matching_field(adj_ey_prev, eps_ey, "eps_ey");
  check_field(adj_hx_mid, "adj_hx_mid");
  check_matching_field(adj_hx_mid, hx_mid, "hx_mid");
  check_matching_field(adj_hx_mid, hx_curl, "hx_curl");
  check_field(adj_hz_mid, "adj_hz_mid");
  check_matching_field(adj_hz_mid, hz_mid, "hz_mid");
  check_matching_field(adj_hz_mid, hz_curl, "hz_curl");
  const c10::cuda::CUDAGuard device_guard(adj_ey_prev.device());
  const int64_t total = adj_ey_prev.numel();
  reverse_magnetic_to_ey_standard_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(adj_ey_prev.size(1)),
      static_cast<int>(adj_ey_prev.size(2)),
      static_cast<int>(adj_hx_mid.size(1)),
      static_cast<int>(adj_hx_mid.size(2)),
      static_cast<int>(adj_hz_mid.size(0)),
      static_cast<int>(adj_hz_mid.size(1)),
      static_cast<int>(adj_hz_mid.size(2)),
      adj_ey_prev.data_ptr<float>(),
      grad_eps_ey.data_ptr<float>(),
      adj_ey_post.data_ptr<float>(),
      adj_hx_mid.data_ptr<float>(),
      adj_hz_mid.data_ptr<float>(),
      ey_decay.data_ptr<float>(),
      ey_curl.data_ptr<float>(),
      eps_ey.data_ptr<float>(),
      hx_mid.data_ptr<float>(),
      hz_mid.data_ptr<float>(),
      hx_curl.data_ptr<float>(),
      hz_curl.data_ptr<float>(),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dz),
      static_cast<int>(x_low_mode),
      static_cast<int>(x_high_mode),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode));
  WITWIN_CUDA_CHECK();
}

void reverse_magnetic_adjoint_to_ez_standard_cuda(
    at::Tensor adj_ez_prev,
    at::Tensor grad_eps_ez,
    const at::Tensor& adj_ez_post,
    const at::Tensor& adj_hx_mid,
    const at::Tensor& adj_hy_mid,
    const at::Tensor& ez_decay,
    const at::Tensor& ez_curl,
    const at::Tensor& eps_ez,
    const at::Tensor& hx_mid,
    const at::Tensor& hy_mid,
    const at::Tensor& hx_curl,
    const at::Tensor& hy_curl,
    double inv_dx,
    double inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode) {
  check_field(adj_ez_prev, "adj_ez_prev");
  check_matching_field(adj_ez_prev, grad_eps_ez, "grad_eps_ez");
  check_matching_field(adj_ez_prev, adj_ez_post, "adj_ez_post");
  check_matching_field(adj_ez_prev, ez_decay, "ez_decay");
  check_matching_field(adj_ez_prev, ez_curl, "ez_curl");
  check_matching_field(adj_ez_prev, eps_ez, "eps_ez");
  check_field(adj_hx_mid, "adj_hx_mid");
  check_matching_field(adj_hx_mid, hx_mid, "hx_mid");
  check_matching_field(adj_hx_mid, hx_curl, "hx_curl");
  check_field(adj_hy_mid, "adj_hy_mid");
  check_matching_field(adj_hy_mid, hy_mid, "hy_mid");
  check_matching_field(adj_hy_mid, hy_curl, "hy_curl");
  const c10::cuda::CUDAGuard device_guard(adj_ez_prev.device());
  const int64_t total = adj_ez_prev.numel();
  reverse_magnetic_to_ez_standard_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(adj_ez_prev.size(1)),
      static_cast<int>(adj_ez_prev.size(2)),
      static_cast<int>(adj_hx_mid.size(1)),
      static_cast<int>(adj_hx_mid.size(2)),
      static_cast<int>(adj_hy_mid.size(0)),
      static_cast<int>(adj_hy_mid.size(1)),
      static_cast<int>(adj_hy_mid.size(2)),
      adj_ez_prev.data_ptr<float>(),
      grad_eps_ez.data_ptr<float>(),
      adj_ez_post.data_ptr<float>(),
      adj_hx_mid.data_ptr<float>(),
      adj_hy_mid.data_ptr<float>(),
      ez_decay.data_ptr<float>(),
      ez_curl.data_ptr<float>(),
      eps_ez.data_ptr<float>(),
      hx_mid.data_ptr<float>(),
      hy_mid.data_ptr<float>(),
      hx_curl.data_ptr<float>(),
      hy_curl.data_ptr<float>(),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dy),
      static_cast<int>(x_low_mode),
      static_cast<int>(x_high_mode),
      static_cast<int>(y_low_mode),
      static_cast<int>(y_high_mode));
  WITWIN_CUDA_CHECK();
}

void reverse_electric_adjoint_to_hx_bloch_cuda(
    at::Tensor adj_hx_mid_real,
    at::Tensor adj_hx_mid_imag,
    const at::Tensor& adj_hx_post_real,
    const at::Tensor& adj_hx_post_imag,
    const at::Tensor& adj_ey_post_real,
    const at::Tensor& adj_ey_post_imag,
    const at::Tensor& adj_ez_post_real,
    const at::Tensor& adj_ez_post_imag,
    const at::Tensor& ey_curl,
    const at::Tensor& ez_curl,
    double phase_cos_y,
    double phase_sin_y,
    double phase_cos_z,
    double phase_sin_z,
    double inv_dy,
    double inv_dz) {
  check_field(adj_hx_mid_real, "adj_hx_mid_real");
  check_matching_field(adj_hx_mid_real, adj_hx_mid_imag, "adj_hx_mid_imag");
  check_matching_field(adj_hx_mid_real, adj_hx_post_real, "adj_hx_post_real");
  check_matching_field(adj_hx_mid_real, adj_hx_post_imag, "adj_hx_post_imag");
  check_field(adj_ey_post_real, "adj_ey_post_real");
  check_matching_field(adj_ey_post_real, adj_ey_post_imag, "adj_ey_post_imag");
  check_matching_field(adj_ey_post_real, ey_curl, "ey_curl");
  check_field(adj_ez_post_real, "adj_ez_post_real");
  check_matching_field(adj_ez_post_real, adj_ez_post_imag, "adj_ez_post_imag");
  check_matching_field(adj_ez_post_real, ez_curl, "ez_curl");
  TORCH_CHECK(adj_ey_post_real.size(0) == adj_hx_mid_real.size(0)
      && adj_ey_post_real.size(1) == adj_hx_mid_real.size(1)
      && adj_ey_post_real.size(2) == adj_hx_mid_real.size(2) + 1,
      "Ey adjoint shape must match Hx Bloch stencil");
  TORCH_CHECK(adj_ez_post_real.size(0) == adj_hx_mid_real.size(0)
      && adj_ez_post_real.size(1) == adj_hx_mid_real.size(1) + 1
      && adj_ez_post_real.size(2) == adj_hx_mid_real.size(2),
      "Ez adjoint shape must match Hx Bloch stencil");
  const c10::cuda::CUDAGuard device_guard(adj_hx_mid_real.device());
  const int64_t total = adj_hx_mid_real.numel();
  reverse_electric_to_hx_bloch_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(adj_hx_mid_real.size(1)),
      static_cast<int>(adj_hx_mid_real.size(2)),
      static_cast<int>(adj_ey_post_real.size(0)),
      static_cast<int>(adj_ey_post_real.size(1)),
      static_cast<int>(adj_ey_post_real.size(2)),
      static_cast<int>(adj_ez_post_real.size(0)),
      static_cast<int>(adj_ez_post_real.size(1)),
      static_cast<int>(adj_ez_post_real.size(2)),
      adj_hx_mid_real.data_ptr<float>(),
      adj_hx_mid_imag.data_ptr<float>(),
      adj_hx_post_real.data_ptr<float>(),
      adj_hx_post_imag.data_ptr<float>(),
      adj_ey_post_real.data_ptr<float>(),
      adj_ey_post_imag.data_ptr<float>(),
      adj_ez_post_real.data_ptr<float>(),
      adj_ez_post_imag.data_ptr<float>(),
      ey_curl.data_ptr<float>(),
      ez_curl.data_ptr<float>(),
      static_cast<float>(phase_cos_y),
      static_cast<float>(phase_sin_y),
      static_cast<float>(phase_cos_z),
      static_cast<float>(phase_sin_z),
      static_cast<float>(inv_dy),
      static_cast<float>(inv_dz));
  WITWIN_CUDA_CHECK();
}

void reverse_electric_adjoint_to_hy_bloch_cuda(
    at::Tensor adj_hy_mid_real,
    at::Tensor adj_hy_mid_imag,
    const at::Tensor& adj_hy_post_real,
    const at::Tensor& adj_hy_post_imag,
    const at::Tensor& adj_ex_post_real,
    const at::Tensor& adj_ex_post_imag,
    const at::Tensor& adj_ez_post_real,
    const at::Tensor& adj_ez_post_imag,
    const at::Tensor& ex_curl,
    const at::Tensor& ez_curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_z,
    double phase_sin_z,
    double inv_dx,
    double inv_dz) {
  check_field(adj_hy_mid_real, "adj_hy_mid_real");
  check_matching_field(adj_hy_mid_real, adj_hy_mid_imag, "adj_hy_mid_imag");
  check_matching_field(adj_hy_mid_real, adj_hy_post_real, "adj_hy_post_real");
  check_matching_field(adj_hy_mid_real, adj_hy_post_imag, "adj_hy_post_imag");
  check_field(adj_ex_post_real, "adj_ex_post_real");
  check_matching_field(adj_ex_post_real, adj_ex_post_imag, "adj_ex_post_imag");
  check_matching_field(adj_ex_post_real, ex_curl, "ex_curl");
  check_field(adj_ez_post_real, "adj_ez_post_real");
  check_matching_field(adj_ez_post_real, adj_ez_post_imag, "adj_ez_post_imag");
  check_matching_field(adj_ez_post_real, ez_curl, "ez_curl");
  TORCH_CHECK(adj_ex_post_real.size(0) == adj_hy_mid_real.size(0)
      && adj_ex_post_real.size(1) == adj_hy_mid_real.size(1)
      && adj_ex_post_real.size(2) == adj_hy_mid_real.size(2) + 1,
      "Ex adjoint shape must match Hy Bloch stencil");
  TORCH_CHECK(adj_ez_post_real.size(0) == adj_hy_mid_real.size(0) + 1
      && adj_ez_post_real.size(1) == adj_hy_mid_real.size(1)
      && adj_ez_post_real.size(2) == adj_hy_mid_real.size(2),
      "Ez adjoint shape must match Hy Bloch stencil");
  const c10::cuda::CUDAGuard device_guard(adj_hy_mid_real.device());
  const int64_t total = adj_hy_mid_real.numel();
  reverse_electric_to_hy_bloch_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(adj_hy_mid_real.size(1)),
      static_cast<int>(adj_hy_mid_real.size(2)),
      static_cast<int>(adj_ex_post_real.size(0)),
      static_cast<int>(adj_ex_post_real.size(1)),
      static_cast<int>(adj_ex_post_real.size(2)),
      static_cast<int>(adj_ez_post_real.size(0)),
      static_cast<int>(adj_ez_post_real.size(1)),
      static_cast<int>(adj_ez_post_real.size(2)),
      adj_hy_mid_real.data_ptr<float>(),
      adj_hy_mid_imag.data_ptr<float>(),
      adj_hy_post_real.data_ptr<float>(),
      adj_hy_post_imag.data_ptr<float>(),
      adj_ex_post_real.data_ptr<float>(),
      adj_ex_post_imag.data_ptr<float>(),
      adj_ez_post_real.data_ptr<float>(),
      adj_ez_post_imag.data_ptr<float>(),
      ex_curl.data_ptr<float>(),
      ez_curl.data_ptr<float>(),
      static_cast<float>(phase_cos_x),
      static_cast<float>(phase_sin_x),
      static_cast<float>(phase_cos_z),
      static_cast<float>(phase_sin_z),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dz));
  WITWIN_CUDA_CHECK();
}

void reverse_electric_adjoint_to_hz_bloch_cuda(
    at::Tensor adj_hz_mid_real,
    at::Tensor adj_hz_mid_imag,
    const at::Tensor& adj_hz_post_real,
    const at::Tensor& adj_hz_post_imag,
    const at::Tensor& adj_ex_post_real,
    const at::Tensor& adj_ex_post_imag,
    const at::Tensor& adj_ey_post_real,
    const at::Tensor& adj_ey_post_imag,
    const at::Tensor& ex_curl,
    const at::Tensor& ey_curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_y,
    double phase_sin_y,
    double inv_dx,
    double inv_dy) {
  check_field(adj_hz_mid_real, "adj_hz_mid_real");
  check_matching_field(adj_hz_mid_real, adj_hz_mid_imag, "adj_hz_mid_imag");
  check_matching_field(adj_hz_mid_real, adj_hz_post_real, "adj_hz_post_real");
  check_matching_field(adj_hz_mid_real, adj_hz_post_imag, "adj_hz_post_imag");
  check_field(adj_ex_post_real, "adj_ex_post_real");
  check_matching_field(adj_ex_post_real, adj_ex_post_imag, "adj_ex_post_imag");
  check_matching_field(adj_ex_post_real, ex_curl, "ex_curl");
  check_field(adj_ey_post_real, "adj_ey_post_real");
  check_matching_field(adj_ey_post_real, adj_ey_post_imag, "adj_ey_post_imag");
  check_matching_field(adj_ey_post_real, ey_curl, "ey_curl");
  TORCH_CHECK(adj_ex_post_real.size(0) == adj_hz_mid_real.size(0)
      && adj_ex_post_real.size(1) == adj_hz_mid_real.size(1) + 1
      && adj_ex_post_real.size(2) == adj_hz_mid_real.size(2),
      "Ex adjoint shape must match Hz Bloch stencil");
  TORCH_CHECK(adj_ey_post_real.size(0) == adj_hz_mid_real.size(0) + 1
      && adj_ey_post_real.size(1) == adj_hz_mid_real.size(1)
      && adj_ey_post_real.size(2) == adj_hz_mid_real.size(2),
      "Ey adjoint shape must match Hz Bloch stencil");
  const c10::cuda::CUDAGuard device_guard(adj_hz_mid_real.device());
  const int64_t total = adj_hz_mid_real.numel();
  reverse_electric_to_hz_bloch_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(adj_hz_mid_real.size(1)),
      static_cast<int>(adj_hz_mid_real.size(2)),
      static_cast<int>(adj_ex_post_real.size(0)),
      static_cast<int>(adj_ex_post_real.size(1)),
      static_cast<int>(adj_ex_post_real.size(2)),
      static_cast<int>(adj_ey_post_real.size(0)),
      static_cast<int>(adj_ey_post_real.size(1)),
      static_cast<int>(adj_ey_post_real.size(2)),
      adj_hz_mid_real.data_ptr<float>(),
      adj_hz_mid_imag.data_ptr<float>(),
      adj_hz_post_real.data_ptr<float>(),
      adj_hz_post_imag.data_ptr<float>(),
      adj_ex_post_real.data_ptr<float>(),
      adj_ex_post_imag.data_ptr<float>(),
      adj_ey_post_real.data_ptr<float>(),
      adj_ey_post_imag.data_ptr<float>(),
      ex_curl.data_ptr<float>(),
      ey_curl.data_ptr<float>(),
      static_cast<float>(phase_cos_x),
      static_cast<float>(phase_sin_x),
      static_cast<float>(phase_cos_y),
      static_cast<float>(phase_sin_y),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dy));
  WITWIN_CUDA_CHECK();
}

void reverse_magnetic_adjoint_to_ex_bloch_cuda(
    at::Tensor adj_ex_prev_real,
    at::Tensor adj_ex_prev_imag,
    at::Tensor grad_eps_ex,
    const at::Tensor& adj_ex_post_real,
    const at::Tensor& adj_ex_post_imag,
    const at::Tensor& adj_hy_mid_real,
    const at::Tensor& adj_hy_mid_imag,
    const at::Tensor& adj_hz_mid_real,
    const at::Tensor& adj_hz_mid_imag,
    const at::Tensor& ex_decay,
    const at::Tensor& ex_curl,
    const at::Tensor& eps_ex,
    const at::Tensor& hy_mid_real,
    const at::Tensor& hy_mid_imag,
    const at::Tensor& hz_mid_real,
    const at::Tensor& hz_mid_imag,
    const at::Tensor& hy_curl,
    const at::Tensor& hz_curl,
    double phase_cos_y,
    double phase_sin_y,
    double phase_cos_z,
    double phase_sin_z,
    double inv_dy,
    double inv_dz) {
  check_field(adj_ex_prev_real, "adj_ex_prev_real");
  check_matching_field(adj_ex_prev_real, adj_ex_prev_imag, "adj_ex_prev_imag");
  check_matching_field(adj_ex_prev_real, grad_eps_ex, "grad_eps_ex");
  check_matching_field(adj_ex_prev_real, adj_ex_post_real, "adj_ex_post_real");
  check_matching_field(adj_ex_prev_real, adj_ex_post_imag, "adj_ex_post_imag");
  check_matching_field(adj_ex_prev_real, ex_decay, "ex_decay");
  check_matching_field(adj_ex_prev_real, ex_curl, "ex_curl");
  check_matching_field(adj_ex_prev_real, eps_ex, "eps_ex");
  check_field(hy_mid_real, "hy_mid_real");
  check_matching_field(hy_mid_real, hy_mid_imag, "hy_mid_imag");
  check_matching_field(hy_mid_real, adj_hy_mid_real, "adj_hy_mid_real");
  check_matching_field(hy_mid_real, adj_hy_mid_imag, "adj_hy_mid_imag");
  check_matching_field(hy_mid_real, hy_curl, "hy_curl");
  check_field(hz_mid_real, "hz_mid_real");
  check_matching_field(hz_mid_real, hz_mid_imag, "hz_mid_imag");
  check_matching_field(hz_mid_real, adj_hz_mid_real, "adj_hz_mid_real");
  check_matching_field(hz_mid_real, adj_hz_mid_imag, "adj_hz_mid_imag");
  check_matching_field(hz_mid_real, hz_curl, "hz_curl");
  const c10::cuda::CUDAGuard device_guard(adj_ex_prev_real.device());
  const int64_t total = adj_ex_prev_real.numel();
  reverse_magnetic_to_ex_bloch_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(adj_ex_prev_real.size(1)),
      static_cast<int>(adj_ex_prev_real.size(2)),
      static_cast<int>(hy_mid_real.size(0)),
      static_cast<int>(hy_mid_real.size(1)),
      static_cast<int>(hy_mid_real.size(2)),
      static_cast<int>(hz_mid_real.size(0)),
      static_cast<int>(hz_mid_real.size(1)),
      static_cast<int>(hz_mid_real.size(2)),
      adj_ex_prev_real.data_ptr<float>(),
      adj_ex_prev_imag.data_ptr<float>(),
      grad_eps_ex.data_ptr<float>(),
      adj_ex_post_real.data_ptr<float>(),
      adj_ex_post_imag.data_ptr<float>(),
      adj_hy_mid_real.data_ptr<float>(),
      adj_hy_mid_imag.data_ptr<float>(),
      adj_hz_mid_real.data_ptr<float>(),
      adj_hz_mid_imag.data_ptr<float>(),
      ex_decay.data_ptr<float>(),
      ex_curl.data_ptr<float>(),
      eps_ex.data_ptr<float>(),
      hy_mid_real.data_ptr<float>(),
      hy_mid_imag.data_ptr<float>(),
      hz_mid_real.data_ptr<float>(),
      hz_mid_imag.data_ptr<float>(),
      hy_curl.data_ptr<float>(),
      hz_curl.data_ptr<float>(),
      static_cast<float>(phase_cos_y),
      static_cast<float>(phase_sin_y),
      static_cast<float>(phase_cos_z),
      static_cast<float>(phase_sin_z),
      static_cast<float>(inv_dy),
      static_cast<float>(inv_dz));
  WITWIN_CUDA_CHECK();
}

void reverse_magnetic_adjoint_to_ey_bloch_cuda(
    at::Tensor adj_ey_prev_real,
    at::Tensor adj_ey_prev_imag,
    at::Tensor grad_eps_ey,
    const at::Tensor& adj_ey_post_real,
    const at::Tensor& adj_ey_post_imag,
    const at::Tensor& adj_hx_mid_real,
    const at::Tensor& adj_hx_mid_imag,
    const at::Tensor& adj_hz_mid_real,
    const at::Tensor& adj_hz_mid_imag,
    const at::Tensor& ey_decay,
    const at::Tensor& ey_curl,
    const at::Tensor& eps_ey,
    const at::Tensor& hx_mid_real,
    const at::Tensor& hx_mid_imag,
    const at::Tensor& hz_mid_real,
    const at::Tensor& hz_mid_imag,
    const at::Tensor& hx_curl,
    const at::Tensor& hz_curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_z,
    double phase_sin_z,
    double inv_dx,
    double inv_dz) {
  check_field(adj_ey_prev_real, "adj_ey_prev_real");
  check_matching_field(adj_ey_prev_real, adj_ey_prev_imag, "adj_ey_prev_imag");
  check_matching_field(adj_ey_prev_real, grad_eps_ey, "grad_eps_ey");
  check_matching_field(adj_ey_prev_real, adj_ey_post_real, "adj_ey_post_real");
  check_matching_field(adj_ey_prev_real, adj_ey_post_imag, "adj_ey_post_imag");
  check_matching_field(adj_ey_prev_real, ey_decay, "ey_decay");
  check_matching_field(adj_ey_prev_real, ey_curl, "ey_curl");
  check_matching_field(adj_ey_prev_real, eps_ey, "eps_ey");
  check_field(hx_mid_real, "hx_mid_real");
  check_matching_field(hx_mid_real, hx_mid_imag, "hx_mid_imag");
  check_matching_field(hx_mid_real, adj_hx_mid_real, "adj_hx_mid_real");
  check_matching_field(hx_mid_real, adj_hx_mid_imag, "adj_hx_mid_imag");
  check_matching_field(hx_mid_real, hx_curl, "hx_curl");
  check_field(hz_mid_real, "hz_mid_real");
  check_matching_field(hz_mid_real, hz_mid_imag, "hz_mid_imag");
  check_matching_field(hz_mid_real, adj_hz_mid_real, "adj_hz_mid_real");
  check_matching_field(hz_mid_real, adj_hz_mid_imag, "adj_hz_mid_imag");
  check_matching_field(hz_mid_real, hz_curl, "hz_curl");
  const c10::cuda::CUDAGuard device_guard(adj_ey_prev_real.device());
  const int64_t total = adj_ey_prev_real.numel();
  reverse_magnetic_to_ey_bloch_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(adj_ey_prev_real.size(1)),
      static_cast<int>(adj_ey_prev_real.size(2)),
      static_cast<int>(hx_mid_real.size(0)),
      static_cast<int>(hx_mid_real.size(1)),
      static_cast<int>(hx_mid_real.size(2)),
      static_cast<int>(hz_mid_real.size(0)),
      static_cast<int>(hz_mid_real.size(1)),
      static_cast<int>(hz_mid_real.size(2)),
      adj_ey_prev_real.data_ptr<float>(),
      adj_ey_prev_imag.data_ptr<float>(),
      grad_eps_ey.data_ptr<float>(),
      adj_ey_post_real.data_ptr<float>(),
      adj_ey_post_imag.data_ptr<float>(),
      adj_hx_mid_real.data_ptr<float>(),
      adj_hx_mid_imag.data_ptr<float>(),
      adj_hz_mid_real.data_ptr<float>(),
      adj_hz_mid_imag.data_ptr<float>(),
      ey_decay.data_ptr<float>(),
      ey_curl.data_ptr<float>(),
      eps_ey.data_ptr<float>(),
      hx_mid_real.data_ptr<float>(),
      hx_mid_imag.data_ptr<float>(),
      hz_mid_real.data_ptr<float>(),
      hz_mid_imag.data_ptr<float>(),
      hx_curl.data_ptr<float>(),
      hz_curl.data_ptr<float>(),
      static_cast<float>(phase_cos_x),
      static_cast<float>(phase_sin_x),
      static_cast<float>(phase_cos_z),
      static_cast<float>(phase_sin_z),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dz));
  WITWIN_CUDA_CHECK();
}

void reverse_magnetic_adjoint_to_ez_bloch_cuda(
    at::Tensor adj_ez_prev_real,
    at::Tensor adj_ez_prev_imag,
    at::Tensor grad_eps_ez,
    const at::Tensor& adj_ez_post_real,
    const at::Tensor& adj_ez_post_imag,
    const at::Tensor& adj_hx_mid_real,
    const at::Tensor& adj_hx_mid_imag,
    const at::Tensor& adj_hy_mid_real,
    const at::Tensor& adj_hy_mid_imag,
    const at::Tensor& ez_decay,
    const at::Tensor& ez_curl,
    const at::Tensor& eps_ez,
    const at::Tensor& hx_mid_real,
    const at::Tensor& hx_mid_imag,
    const at::Tensor& hy_mid_real,
    const at::Tensor& hy_mid_imag,
    const at::Tensor& hx_curl,
    const at::Tensor& hy_curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_y,
    double phase_sin_y,
    double inv_dx,
    double inv_dy) {
  check_field(adj_ez_prev_real, "adj_ez_prev_real");
  check_matching_field(adj_ez_prev_real, adj_ez_prev_imag, "adj_ez_prev_imag");
  check_matching_field(adj_ez_prev_real, grad_eps_ez, "grad_eps_ez");
  check_matching_field(adj_ez_prev_real, adj_ez_post_real, "adj_ez_post_real");
  check_matching_field(adj_ez_prev_real, adj_ez_post_imag, "adj_ez_post_imag");
  check_matching_field(adj_ez_prev_real, ez_decay, "ez_decay");
  check_matching_field(adj_ez_prev_real, ez_curl, "ez_curl");
  check_matching_field(adj_ez_prev_real, eps_ez, "eps_ez");
  check_field(hx_mid_real, "hx_mid_real");
  check_matching_field(hx_mid_real, hx_mid_imag, "hx_mid_imag");
  check_matching_field(hx_mid_real, adj_hx_mid_real, "adj_hx_mid_real");
  check_matching_field(hx_mid_real, adj_hx_mid_imag, "adj_hx_mid_imag");
  check_matching_field(hx_mid_real, hx_curl, "hx_curl");
  check_field(hy_mid_real, "hy_mid_real");
  check_matching_field(hy_mid_real, hy_mid_imag, "hy_mid_imag");
  check_matching_field(hy_mid_real, adj_hy_mid_real, "adj_hy_mid_real");
  check_matching_field(hy_mid_real, adj_hy_mid_imag, "adj_hy_mid_imag");
  check_matching_field(hy_mid_real, hy_curl, "hy_curl");
  const c10::cuda::CUDAGuard device_guard(adj_ez_prev_real.device());
  const int64_t total = adj_ez_prev_real.numel();
  reverse_magnetic_to_ez_bloch_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(adj_ez_prev_real.size(1)),
      static_cast<int>(adj_ez_prev_real.size(2)),
      static_cast<int>(hx_mid_real.size(0)),
      static_cast<int>(hx_mid_real.size(1)),
      static_cast<int>(hx_mid_real.size(2)),
      static_cast<int>(hy_mid_real.size(0)),
      static_cast<int>(hy_mid_real.size(1)),
      static_cast<int>(hy_mid_real.size(2)),
      adj_ez_prev_real.data_ptr<float>(),
      adj_ez_prev_imag.data_ptr<float>(),
      grad_eps_ez.data_ptr<float>(),
      adj_ez_post_real.data_ptr<float>(),
      adj_ez_post_imag.data_ptr<float>(),
      adj_hx_mid_real.data_ptr<float>(),
      adj_hx_mid_imag.data_ptr<float>(),
      adj_hy_mid_real.data_ptr<float>(),
      adj_hy_mid_imag.data_ptr<float>(),
      ez_decay.data_ptr<float>(),
      ez_curl.data_ptr<float>(),
      eps_ez.data_ptr<float>(),
      hx_mid_real.data_ptr<float>(),
      hx_mid_imag.data_ptr<float>(),
      hy_mid_real.data_ptr<float>(),
      hy_mid_imag.data_ptr<float>(),
      hx_curl.data_ptr<float>(),
      hy_curl.data_ptr<float>(),
      static_cast<float>(phase_cos_x),
      static_cast<float>(phase_sin_x),
      static_cast<float>(phase_cos_y),
      static_cast<float>(phase_sin_y),
      static_cast<float>(inv_dx),
      static_cast<float>(inv_dy));
  WITWIN_CUDA_CHECK();
}

void accumulate_forward_diff_adjoint_cuda(
    at::Tensor field_grad,
    const at::Tensor& diff_grad,
    int64_t axis,
    double inv_delta) {
  check_field(field_grad, "field_grad");
  check_field(diff_grad, "diff_grad");
  TORCH_CHECK(axis >= 0 && axis < 3, "axis must be in [0, 3)");
  const c10::cuda::CUDAGuard device_guard(field_grad.device());
  const int64_t total = field_grad.numel();
  accumulate_diff_adjoint_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(field_grad.size(1)),
      static_cast<int>(field_grad.size(2)),
      static_cast<int>(diff_grad.size(0)),
      static_cast<int>(diff_grad.size(1)),
      static_cast<int>(diff_grad.size(2)),
      static_cast<int>(axis),
      true,
      static_cast<float>(inv_delta),
      field_grad.data_ptr<float>(),
      diff_grad.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void accumulate_backward_diff_adjoint_cuda(
    at::Tensor field_grad,
    const at::Tensor& diff_grad,
    int64_t axis,
    double inv_delta) {
  check_field(field_grad, "field_grad");
  check_field(diff_grad, "diff_grad");
  TORCH_CHECK(axis >= 0 && axis < 3, "axis must be in [0, 3)");
  const c10::cuda::CUDAGuard device_guard(field_grad.device());
  const int64_t total = field_grad.numel();
  accumulate_diff_adjoint_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(field_grad.size(1)),
      static_cast<int>(field_grad.size(2)),
      static_cast<int>(diff_grad.size(0)),
      static_cast<int>(diff_grad.size(1)),
      static_cast<int>(diff_grad.size(2)),
      static_cast<int>(axis),
      false,
      static_cast<float>(inv_delta),
      field_grad.data_ptr<float>(),
      diff_grad.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void launch_reverse_electric_component_cpml(
    int component,
    at::Tensor adj_prev,
    at::Tensor grad_eps,
    at::Tensor adj_psi_pos_prev,
    at::Tensor adj_psi_neg_prev,
    at::Tensor adj_d_pos,
    at::Tensor adj_d_neg,
    const at::Tensor& adj_post,
    const at::Tensor& adj_psi_pos_post,
    const at::Tensor& adj_psi_neg_post,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const at::Tensor& eps,
    const at::Tensor& psi_pos,
    const at::Tensor& psi_neg,
    const at::Tensor& b_pos,
    const at::Tensor& c_pos,
    const at::Tensor& inv_kappa_pos,
    const at::Tensor& b_neg,
    const at::Tensor& c_neg,
    const at::Tensor& inv_kappa_neg,
    const at::Tensor& h_pos_mid,
    const at::Tensor& h_neg_mid,
    double inv_pos,
    double inv_neg,
    int64_t low_mode_a,
    int64_t high_mode_a,
    int64_t low_mode_b,
    int64_t high_mode_b) {
  check_field(adj_prev, "adj_prev");
  check_matching_field(adj_prev, grad_eps, "grad_eps");
  check_matching_field(adj_prev, adj_psi_pos_prev, "adj_psi_pos_prev");
  check_matching_field(adj_prev, adj_psi_neg_prev, "adj_psi_neg_prev");
  check_matching_field(adj_prev, adj_d_pos, "adj_d_pos");
  check_matching_field(adj_prev, adj_d_neg, "adj_d_neg");
  check_matching_field(adj_prev, adj_post, "adj_post");
  check_matching_field(adj_prev, adj_psi_pos_post, "adj_psi_pos_post");
  check_matching_field(adj_prev, adj_psi_neg_post, "adj_psi_neg_post");
  check_matching_field(adj_prev, decay, "decay");
  check_matching_field(adj_prev, curl, "curl");
  check_matching_field(adj_prev, eps, "eps");
  check_matching_field(adj_prev, psi_pos, "psi_pos");
  check_matching_field(adj_prev, psi_neg, "psi_neg");
  check_vector(b_pos, "b_pos");
  check_matching_vector(b_pos, c_pos, "c_pos");
  check_matching_vector(b_pos, inv_kappa_pos, "inv_kappa_pos");
  check_vector(b_neg, "b_neg");
  check_matching_vector(b_neg, c_neg, "c_neg");
  check_matching_vector(b_neg, inv_kappa_neg, "inv_kappa_neg");
  check_field(h_pos_mid, "h_pos_mid");
  check_field(h_neg_mid, "h_neg_mid");
  const c10::cuda::CUDAGuard device_guard(adj_prev.device());
  const int64_t total = adj_prev.numel();
  reverse_electric_component_cpml_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      component,
      static_cast<int>(adj_prev.size(1)),
      static_cast<int>(adj_prev.size(2)),
      static_cast<int>(h_pos_mid.size(1)),
      static_cast<int>(h_pos_mid.size(2)),
      static_cast<int>(h_neg_mid.size(1)),
      static_cast<int>(h_neg_mid.size(2)),
      adj_prev.data_ptr<float>(),
      grad_eps.data_ptr<float>(),
      adj_psi_pos_prev.data_ptr<float>(),
      adj_psi_neg_prev.data_ptr<float>(),
      adj_d_pos.data_ptr<float>(),
      adj_d_neg.data_ptr<float>(),
      adj_post.data_ptr<float>(),
      adj_psi_pos_post.data_ptr<float>(),
      adj_psi_neg_post.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>(),
      eps.data_ptr<float>(),
      psi_pos.data_ptr<float>(),
      psi_neg.data_ptr<float>(),
      b_pos.data_ptr<float>(),
      c_pos.data_ptr<float>(),
      inv_kappa_pos.data_ptr<float>(),
      b_neg.data_ptr<float>(),
      c_neg.data_ptr<float>(),
      inv_kappa_neg.data_ptr<float>(),
      h_pos_mid.data_ptr<float>(),
      h_neg_mid.data_ptr<float>(),
      static_cast<float>(inv_pos),
      static_cast<float>(inv_neg),
      static_cast<int>(low_mode_a),
      static_cast<int>(high_mode_a),
      static_cast<int>(low_mode_b),
      static_cast<int>(high_mode_b));
  WITWIN_CUDA_CHECK();
}

void reverse_electric_component_ex_cpml_cuda(
    at::Tensor adj_prev,
    at::Tensor grad_eps,
    at::Tensor adj_psi_pos_prev,
    at::Tensor adj_psi_neg_prev,
    at::Tensor adj_d_pos,
    at::Tensor adj_d_neg,
    const at::Tensor& adj_post,
    const at::Tensor& adj_psi_pos_post,
    const at::Tensor& adj_psi_neg_post,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const at::Tensor& eps,
    const at::Tensor& psi_pos,
    const at::Tensor& psi_neg,
    const at::Tensor& b_pos,
    const at::Tensor& c_pos,
    const at::Tensor& inv_kappa_pos,
    const at::Tensor& b_neg,
    const at::Tensor& c_neg,
    const at::Tensor& inv_kappa_neg,
    const at::Tensor& hy_mid,
    const at::Tensor& hz_mid,
    double inv_dy,
    double inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  launch_reverse_electric_component_cpml(
      0, adj_prev, grad_eps, adj_psi_pos_prev, adj_psi_neg_prev, adj_d_pos, adj_d_neg,
      adj_post, adj_psi_pos_post, adj_psi_neg_post, decay, curl, eps, psi_pos, psi_neg,
      b_pos, c_pos, inv_kappa_pos, b_neg, c_neg, inv_kappa_neg, hz_mid, hy_mid,
      inv_dy, inv_dz, y_low_mode, y_high_mode, z_low_mode, z_high_mode);
}

void reverse_electric_component_ey_cpml_cuda(
    at::Tensor adj_prev,
    at::Tensor grad_eps,
    at::Tensor adj_psi_pos_prev,
    at::Tensor adj_psi_neg_prev,
    at::Tensor adj_d_pos,
    at::Tensor adj_d_neg,
    const at::Tensor& adj_post,
    const at::Tensor& adj_psi_pos_post,
    const at::Tensor& adj_psi_neg_post,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const at::Tensor& eps,
    const at::Tensor& psi_pos,
    const at::Tensor& psi_neg,
    const at::Tensor& b_pos,
    const at::Tensor& c_pos,
    const at::Tensor& inv_kappa_pos,
    const at::Tensor& b_neg,
    const at::Tensor& c_neg,
    const at::Tensor& inv_kappa_neg,
    const at::Tensor& hx_mid,
    const at::Tensor& hz_mid,
    double inv_dx,
    double inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode) {
  launch_reverse_electric_component_cpml(
      1, adj_prev, grad_eps, adj_psi_pos_prev, adj_psi_neg_prev, adj_d_pos, adj_d_neg,
      adj_post, adj_psi_pos_post, adj_psi_neg_post, decay, curl, eps, psi_pos, psi_neg,
      b_pos, c_pos, inv_kappa_pos, b_neg, c_neg, inv_kappa_neg, hx_mid, hz_mid,
      inv_dz, inv_dx, x_low_mode, x_high_mode, z_low_mode, z_high_mode);
}

void reverse_electric_component_ez_cpml_cuda(
    at::Tensor adj_prev,
    at::Tensor grad_eps,
    at::Tensor adj_psi_pos_prev,
    at::Tensor adj_psi_neg_prev,
    at::Tensor adj_d_pos,
    at::Tensor adj_d_neg,
    const at::Tensor& adj_post,
    const at::Tensor& adj_psi_pos_post,
    const at::Tensor& adj_psi_neg_post,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const at::Tensor& eps,
    const at::Tensor& psi_pos,
    const at::Tensor& psi_neg,
    const at::Tensor& b_pos,
    const at::Tensor& c_pos,
    const at::Tensor& inv_kappa_pos,
    const at::Tensor& b_neg,
    const at::Tensor& c_neg,
    const at::Tensor& inv_kappa_neg,
    const at::Tensor& hx_mid,
    const at::Tensor& hy_mid,
    double inv_dx,
    double inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode) {
  launch_reverse_electric_component_cpml(
      2, adj_prev, grad_eps, adj_psi_pos_prev, adj_psi_neg_prev, adj_d_pos, adj_d_neg,
      adj_post, adj_psi_pos_post, adj_psi_neg_post, decay, curl, eps, psi_pos, psi_neg,
      b_pos, c_pos, inv_kappa_pos, b_neg, c_neg, inv_kappa_neg, hy_mid, hx_mid,
      inv_dx, inv_dy, x_low_mode, x_high_mode, y_low_mode, y_high_mode);
}

void launch_reverse_magnetic_component_cpml(
    int component,
    at::Tensor adj_prev,
    at::Tensor adj_psi_pos_prev,
    at::Tensor adj_psi_neg_prev,
    at::Tensor adj_d_pos,
    at::Tensor adj_d_neg,
    const at::Tensor& adj_post,
    const at::Tensor& adj_psi_pos_post,
    const at::Tensor& adj_psi_neg_post,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const at::Tensor& b_pos,
    const at::Tensor& c_pos,
    const at::Tensor& inv_kappa_pos,
    const at::Tensor& b_neg,
    const at::Tensor& c_neg,
    const at::Tensor& inv_kappa_neg) {
  check_field(adj_prev, "adj_prev");
  check_matching_field(adj_prev, adj_psi_pos_prev, "adj_psi_pos_prev");
  check_matching_field(adj_prev, adj_psi_neg_prev, "adj_psi_neg_prev");
  check_matching_field(adj_prev, adj_d_pos, "adj_d_pos");
  check_matching_field(adj_prev, adj_d_neg, "adj_d_neg");
  check_matching_field(adj_prev, adj_post, "adj_post");
  check_matching_field(adj_prev, adj_psi_pos_post, "adj_psi_pos_post");
  check_matching_field(adj_prev, adj_psi_neg_post, "adj_psi_neg_post");
  check_matching_field(adj_prev, decay, "decay");
  check_matching_field(adj_prev, curl, "curl");
  check_vector(b_pos, "b_pos");
  check_matching_vector(b_pos, c_pos, "c_pos");
  check_matching_vector(b_pos, inv_kappa_pos, "inv_kappa_pos");
  check_vector(b_neg, "b_neg");
  check_matching_vector(b_neg, c_neg, "c_neg");
  check_matching_vector(b_neg, inv_kappa_neg, "inv_kappa_neg");
  const c10::cuda::CUDAGuard device_guard(adj_prev.device());
  const int64_t total = adj_prev.numel();
  reverse_magnetic_component_cpml_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      component,
      static_cast<int>(adj_prev.size(1)),
      static_cast<int>(adj_prev.size(2)),
      adj_prev.data_ptr<float>(),
      adj_psi_pos_prev.data_ptr<float>(),
      adj_psi_neg_prev.data_ptr<float>(),
      adj_d_pos.data_ptr<float>(),
      adj_d_neg.data_ptr<float>(),
      adj_post.data_ptr<float>(),
      adj_psi_pos_post.data_ptr<float>(),
      adj_psi_neg_post.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>(),
      b_pos.data_ptr<float>(),
      c_pos.data_ptr<float>(),
      inv_kappa_pos.data_ptr<float>(),
      b_neg.data_ptr<float>(),
      c_neg.data_ptr<float>(),
      inv_kappa_neg.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void reverse_magnetic_component_hx_cpml_cuda(
    at::Tensor adj_prev,
    at::Tensor adj_psi_pos_prev,
    at::Tensor adj_psi_neg_prev,
    at::Tensor adj_d_pos,
    at::Tensor adj_d_neg,
    const at::Tensor& adj_post,
    const at::Tensor& adj_psi_pos_post,
    const at::Tensor& adj_psi_neg_post,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const at::Tensor& b_pos,
    const at::Tensor& c_pos,
    const at::Tensor& inv_kappa_pos,
    const at::Tensor& b_neg,
    const at::Tensor& c_neg,
    const at::Tensor& inv_kappa_neg) {
  launch_reverse_magnetic_component_cpml(
      0, adj_prev, adj_psi_pos_prev, adj_psi_neg_prev, adj_d_pos, adj_d_neg,
      adj_post, adj_psi_pos_post, adj_psi_neg_post, decay, curl,
      b_pos, c_pos, inv_kappa_pos, b_neg, c_neg, inv_kappa_neg);
}

void reverse_magnetic_component_hy_cpml_cuda(
    at::Tensor adj_prev,
    at::Tensor adj_psi_pos_prev,
    at::Tensor adj_psi_neg_prev,
    at::Tensor adj_d_pos,
    at::Tensor adj_d_neg,
    const at::Tensor& adj_post,
    const at::Tensor& adj_psi_pos_post,
    const at::Tensor& adj_psi_neg_post,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const at::Tensor& b_pos,
    const at::Tensor& c_pos,
    const at::Tensor& inv_kappa_pos,
    const at::Tensor& b_neg,
    const at::Tensor& c_neg,
    const at::Tensor& inv_kappa_neg) {
  launch_reverse_magnetic_component_cpml(
      1, adj_prev, adj_psi_pos_prev, adj_psi_neg_prev, adj_d_pos, adj_d_neg,
      adj_post, adj_psi_pos_post, adj_psi_neg_post, decay, curl,
      b_pos, c_pos, inv_kappa_pos, b_neg, c_neg, inv_kappa_neg);
}

void reverse_magnetic_component_hz_cpml_cuda(
    at::Tensor adj_prev,
    at::Tensor adj_psi_pos_prev,
    at::Tensor adj_psi_neg_prev,
    at::Tensor adj_d_pos,
    at::Tensor adj_d_neg,
    const at::Tensor& adj_post,
    const at::Tensor& adj_psi_pos_post,
    const at::Tensor& adj_psi_neg_post,
    const at::Tensor& decay,
    const at::Tensor& curl,
    const at::Tensor& b_pos,
    const at::Tensor& c_pos,
    const at::Tensor& inv_kappa_pos,
    const at::Tensor& b_neg,
    const at::Tensor& c_neg,
    const at::Tensor& inv_kappa_neg) {
  launch_reverse_magnetic_component_cpml(
      2, adj_prev, adj_psi_pos_prev, adj_psi_neg_prev, adj_d_pos, adj_d_neg,
      adj_post, adj_psi_pos_post, adj_psi_neg_post, decay, curl,
      b_pos, c_pos, inv_kappa_pos, b_neg, c_neg, inv_kappa_neg);
}

void reverse_debye_current_cuda(
    at::Tensor adj_electric_prev,
    at::Tensor adj_polarization_prev,
    const at::Tensor& adj_polarization_post,
    const at::Tensor& adj_current_post,
    const at::Tensor& drive,
    double decay,
    double dt) {
  check_field(adj_electric_prev, "adj_electric_prev");
  check_matching_field(adj_electric_prev, adj_polarization_prev, "adj_polarization_prev");
  check_matching_field(adj_electric_prev, adj_polarization_post, "adj_polarization_post");
  check_matching_field(adj_electric_prev, adj_current_post, "adj_current_post");
  check_matching_field(adj_electric_prev, drive, "drive");
  const c10::cuda::CUDAGuard device_guard(adj_electric_prev.device());
  const int64_t total = adj_electric_prev.numel();
  reverse_debye_current_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      adj_electric_prev.data_ptr<float>(),
      adj_polarization_prev.data_ptr<float>(),
      adj_polarization_post.data_ptr<float>(),
      adj_current_post.data_ptr<float>(),
      drive.data_ptr<float>(),
      decay,
      dt);
  WITWIN_CUDA_CHECK();
}

void reverse_drude_current_cuda(
    at::Tensor adj_electric_prev,
    at::Tensor adj_current_prev,
    const at::Tensor& adj_current_post,
    const at::Tensor& drive,
    double decay) {
  check_field(adj_electric_prev, "adj_electric_prev");
  check_matching_field(adj_electric_prev, adj_current_prev, "adj_current_prev");
  check_matching_field(adj_electric_prev, adj_current_post, "adj_current_post");
  check_matching_field(adj_electric_prev, drive, "drive");
  const c10::cuda::CUDAGuard device_guard(adj_electric_prev.device());
  const int64_t total = adj_electric_prev.numel();
  reverse_drude_current_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      adj_electric_prev.data_ptr<float>(),
      adj_current_prev.data_ptr<float>(),
      adj_current_post.data_ptr<float>(),
      drive.data_ptr<float>(),
      decay);
  WITWIN_CUDA_CHECK();
}

void reverse_lorentz_current_cuda(
    at::Tensor adj_electric_prev,
    at::Tensor adj_polarization_prev,
    at::Tensor adj_current_prev,
    const at::Tensor& adj_polarization_post,
    const at::Tensor& adj_current_post,
    const at::Tensor& drive,
    double decay,
    double restoring,
    double dt) {
  check_field(adj_electric_prev, "adj_electric_prev");
  check_matching_field(adj_electric_prev, adj_polarization_prev, "adj_polarization_prev");
  check_matching_field(adj_electric_prev, adj_current_prev, "adj_current_prev");
  check_matching_field(adj_electric_prev, adj_polarization_post, "adj_polarization_post");
  check_matching_field(adj_electric_prev, adj_current_post, "adj_current_post");
  check_matching_field(adj_electric_prev, drive, "drive");
  const c10::cuda::CUDAGuard device_guard(adj_electric_prev.device());
  const int64_t total = adj_electric_prev.numel();
  reverse_lorentz_current_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      adj_electric_prev.data_ptr<float>(),
      adj_polarization_prev.data_ptr<float>(),
      adj_current_prev.data_ptr<float>(),
      adj_polarization_post.data_ptr<float>(),
      adj_current_post.data_ptr<float>(),
      drive.data_ptr<float>(),
      decay,
      restoring,
      dt);
  WITWIN_CUDA_CHECK();
}

void accumulate_tfsf_scalar_sample_adjoint_cuda(
    at::Tensor adj_aux_field,
    const at::Tensor& adj_field_patch,
    const at::Tensor& coeff_patch,
    int64_t sample_index,
    double component_scale) {
  check_vector(adj_aux_field, "adj_aux_field");
  check_field(adj_field_patch, "adj_field_patch");
  check_matching_field(adj_field_patch, coeff_patch, "coeff_patch");
  TORCH_CHECK(sample_index >= 0 && sample_index < adj_aux_field.numel(), "sample_index is out of range");
  const c10::cuda::CUDAGuard device_guard(adj_aux_field.device());
  const int64_t total = adj_field_patch.numel();
  accumulate_tfsf_scalar_sample_adjoint_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(adj_field_patch.size(1)),
      static_cast<int>(adj_field_patch.size(2)),
      sample_index,
      static_cast<float>(component_scale),
      adj_aux_field.data_ptr<float>(),
      adj_field_patch.data_ptr<float>(),
      coeff_patch.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void accumulate_tfsf_line_sample_adjoint_cuda(
    at::Tensor adj_aux_field,
    const at::Tensor& adj_field_patch,
    const at::Tensor& coeff_patch,
    const at::Tensor& sample_indices,
    int64_t sample_axis_code,
    double component_scale) {
  check_vector(adj_aux_field, "adj_aux_field");
  check_field(adj_field_patch, "adj_field_patch");
  check_matching_field(adj_field_patch, coeff_patch, "coeff_patch");
  check_int32_vector(sample_indices, "sample_indices");
  TORCH_CHECK(sample_axis_code >= 0 && sample_axis_code < 3, "sample_axis_code must be in [0, 3)");
  TORCH_CHECK(sample_indices.numel() == adj_field_patch.size(sample_axis_code), "sample_indices length must match the selected patch axis");
  const c10::cuda::CUDAGuard device_guard(adj_aux_field.device());
  const int64_t total = adj_field_patch.numel();
  accumulate_tfsf_line_sample_adjoint_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(adj_field_patch.size(1)),
      static_cast<int>(adj_field_patch.size(2)),
      static_cast<int>(sample_axis_code),
      static_cast<float>(component_scale),
      adj_aux_field.data_ptr<float>(),
      adj_field_patch.data_ptr<float>(),
      coeff_patch.data_ptr<float>(),
      sample_indices.data_ptr<int>());
  WITWIN_CUDA_CHECK();
}

void accumulate_tfsf_interpolated_sample_adjoint_cuda(
    at::Tensor adj_aux_field,
    const at::Tensor& adj_field_patch,
    const at::Tensor& coeff_patch,
    const at::Tensor& sample_positions,
    double origin,
    double ds,
    double component_scale) {
  check_vector(adj_aux_field, "adj_aux_field");
  check_field(adj_field_patch, "adj_field_patch");
  check_matching_field(adj_field_patch, coeff_patch, "coeff_patch");
  check_matching_field(adj_field_patch, sample_positions, "sample_positions");
  const c10::cuda::CUDAGuard device_guard(adj_aux_field.device());
  const int64_t total = adj_field_patch.numel();
  accumulate_tfsf_interpolated_sample_adjoint_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      adj_aux_field.numel(),
      static_cast<float>(origin),
      static_cast<float>(ds),
      static_cast<float>(component_scale),
      adj_aux_field.data_ptr<float>(),
      adj_field_patch.data_ptr<float>(),
      coeff_patch.data_ptr<float>(),
      sample_positions.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void reverse_tfsf_auxiliary_electric_cuda(
    at::Tensor adj_electric_prev,
    at::Tensor adj_magnetic_after,
    const at::Tensor& adj_electric_post,
    const at::Tensor& electric_decay,
    const at::Tensor& electric_curl,
    int64_t source_index) {
  check_vector(adj_electric_prev, "adj_electric_prev");
  check_matching_vector(adj_electric_prev, adj_electric_post, "adj_electric_post");
  check_matching_vector(adj_electric_prev, electric_decay, "electric_decay");
  check_matching_vector(adj_electric_prev, electric_curl, "electric_curl");
  check_vector(adj_magnetic_after, "adj_magnetic_after");
  TORCH_CHECK(adj_magnetic_after.numel() + 1 == adj_electric_prev.numel(), "adj_magnetic_after length must be electric length - 1");
  const c10::cuda::CUDAGuard device_guard(adj_electric_prev.device());
  const int64_t total = adj_electric_prev.numel();
  reverse_tfsf_auxiliary_electric_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      adj_magnetic_after.numel(),
      source_index,
      adj_electric_prev.data_ptr<float>(),
      adj_magnetic_after.data_ptr<float>(),
      adj_electric_post.data_ptr<float>(),
      electric_decay.data_ptr<float>(),
      electric_curl.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void reverse_tfsf_auxiliary_magnetic_cuda(
    at::Tensor adj_electric_prev,
    at::Tensor adj_magnetic_prev,
    const at::Tensor& adj_magnetic_after,
    const at::Tensor& magnetic_decay,
    const at::Tensor& magnetic_curl) {
  check_vector(adj_electric_prev, "adj_electric_prev");
  check_vector(adj_magnetic_prev, "adj_magnetic_prev");
  check_matching_vector(adj_magnetic_prev, adj_magnetic_after, "adj_magnetic_after");
  check_matching_vector(adj_magnetic_prev, magnetic_decay, "magnetic_decay");
  check_matching_vector(adj_magnetic_prev, magnetic_curl, "magnetic_curl");
  TORCH_CHECK(adj_magnetic_prev.numel() + 1 == adj_electric_prev.numel(), "adj_electric_prev length must be magnetic length + 1");
  const c10::cuda::CUDAGuard device_guard(adj_electric_prev.device());
  const int64_t total = adj_magnetic_prev.numel();
  reverse_tfsf_auxiliary_magnetic_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      adj_electric_prev.data_ptr<float>(),
      adj_magnetic_prev.data_ptr<float>(),
      adj_magnetic_after.data_ptr<float>(),
      magnetic_decay.data_ptr<float>(),
      magnetic_curl.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}
