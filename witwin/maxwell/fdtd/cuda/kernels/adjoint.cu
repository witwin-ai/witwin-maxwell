#include <limits>
#include <type_traits>

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

void check_field(const torch::stable::Tensor& tensor, const char* name) {
  check_float32_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  STD_TORCH_CHECK(tensor.dim() == 3, name, " must be a contiguous 3D float32 CUDA tensor");
}

void check_matching_field(const torch::stable::Tensor& reference, const torch::stable::Tensor& tensor, const char* name) {
  check_field(tensor, name);
  check_same_cuda_device(reference, tensor, name);
  STD_TORCH_CHECK(tensor.sizes().equals(reference.sizes()), name, " must match reference shape");
}

void check_vector(const torch::stable::Tensor& tensor, const char* name) {
  check_float32_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  STD_TORCH_CHECK(tensor.dim() == 1, name, " must be a contiguous 1D float32 CUDA tensor");
}

void check_matching_vector(const torch::stable::Tensor& reference, const torch::stable::Tensor& tensor, const char* name) {
  check_vector(tensor, name);
  check_same_cuda_device(reference, tensor, name);
  STD_TORCH_CHECK(tensor.sizes().equals(reference.sizes()), name, " must match reference shape");
}

void check_spacing_vector(
    const torch::stable::Tensor& field,
    const torch::stable::Tensor& inv_delta,
    int64_t axis,
    const char* name) {
  check_vector(inv_delta, name);
  check_same_cuda_device(field, inv_delta, name);
  STD_TORCH_CHECK(inv_delta.size(0) == field.size(axis), name, " length must match the field axis");
}

void check_int32_vector(const torch::stable::Tensor& tensor, const char* name) {
  check_cuda_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::Int, name, " must be a contiguous int32 CUDA tensor");
  STD_TORCH_CHECK(tensor.dim() == 1, name, " must be a contiguous 1D int32 CUDA tensor");
}

__device__ __forceinline__ bool is_valid_index_3d(int i, int j, int k, int nx, int ny, int nz) {
  return i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz;
}

__device__ __forceinline__ bool is_interior_coordinate(int coordinate, int size) {
  return coordinate > 0 && coordinate + 1 < size;
}

__device__ __forceinline__ bool is_ex_active_index(int i, int j, int k, int nx, int ny, int nz) {
  return is_valid_index_3d(i, j, k, nx, ny, nz) && is_interior_coordinate(j, ny) && is_interior_coordinate(k, nz);
}

__device__ __forceinline__ bool is_ey_active_index(int i, int j, int k, int nx, int ny, int nz) {
  return is_valid_index_3d(i, j, k, nx, ny, nz) && is_interior_coordinate(i, nx) && is_interior_coordinate(k, nz);
}

__device__ __forceinline__ bool is_ez_active_index(int i, int j, int k, int nx, int ny, int nz) {
  return is_valid_index_3d(i, j, k, nx, ny, nz) && is_interior_coordinate(i, nx) && is_interior_coordinate(j, ny);
}

__device__ __forceinline__ bool is_boundary_index(int coordinate, int size) {
  return coordinate == 0 || coordinate + 1 == size;
}

__device__ __forceinline__ int select_boundary_mode(int low_mode, int high_mode, int coordinate, int size) {
  return coordinate == 0 ? low_mode : high_mode;
}

__device__ __forceinline__ bool is_pec_boundary_mode(int mode) {
  return mode == BOUNDARY_PEC;
}

__device__ __forceinline__ bool is_inactive_boundary_mode(int mode) {
  return mode == BOUNDARY_NONE || mode == BOUNDARY_PML;
}

struct ElectricCellStatus {
  bool active;
  bool inactive;
  bool pec;
};

__device__ __forceinline__ ElectricCellStatus resolve_electric_cell_status(
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

__device__ __forceinline__ bool diff_index_valid(int i, int j, int k, int nx, int ny, int nz) {
  return i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz;
}

__device__ __forceinline__ float diff_grad_value(const float* __restrict__ diff_grad, int i, int j, int k, int ny, int nz) {
  return diff_grad[offset3d(
      static_cast<unsigned int>(i),
      static_cast<unsigned int>(j),
      static_cast<unsigned int>(k),
      static_cast<unsigned int>(ny),
      static_cast<unsigned int>(nz))];
}

__global__ void reverse_magnetic_decay_kernel(
    int64_t total,
    float* __restrict__ adj_prev,
    const float* __restrict__ adj_mid,
    const float* __restrict__ decay) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  adj_prev[index] = adj_mid[index] * decay[index];
}

__global__ void reverse_electric_to_hx_standard_kernel(
    int hx_nx,
    int hx_ny,
    int hx_nz,
    int ey_nx,
    int ey_ny,
    int ey_nz,
    int ez_nx,
    int ez_ny,
    int ez_nz,
    float* __restrict__ adj_hx_mid,
    const float* __restrict__ adj_hx_post,
    const float* __restrict__ adj_ey_post,
    const float* __restrict__ adj_ez_post,
    const float* __restrict__ ey_curl,
    const float* __restrict__ ez_curl,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz) {
  const unsigned int k_u = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j_u = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i_u = blockIdx.z * blockDim.z + threadIdx.z;
  if (static_cast<int>(i_u) >= hx_nx || static_cast<int>(j_u) >= hx_ny || static_cast<int>(k_u) >= hx_nz) {
    return;
  }
  const int i = static_cast<int>(i_u);
  const int j = static_cast<int>(j_u);
  const int k = static_cast<int>(k_u);
  const long long linear = offset3d(i_u, j_u, k_u, hx_ny, hx_nz);
  float adjoint = adj_hx_post[linear];

  if (is_ey_active_index(i, j, k, ey_nx, ey_ny, ey_nz)) {
    const long long ey_index = offset3d(i_u, j_u, k_u, ey_ny, ey_nz);
    adjoint += ey_curl[ey_index] * inv_dz[k] * adj_ey_post[ey_index];
  }
  if (is_ey_active_index(i, j, k + 1, ey_nx, ey_ny, ey_nz)) {
    const long long ey_index = offset3d(i_u, j_u, k_u + 1, ey_ny, ey_nz);
    adjoint -= ey_curl[ey_index] * inv_dz[k + 1] * adj_ey_post[ey_index];
  }
  if (is_ez_active_index(i, j, k, ez_nx, ez_ny, ez_nz)) {
    const long long ez_index = offset3d(i_u, j_u, k_u, ez_ny, ez_nz);
    adjoint -= ez_curl[ez_index] * inv_dy[j] * adj_ez_post[ez_index];
  }
  if (is_ez_active_index(i, j + 1, k, ez_nx, ez_ny, ez_nz)) {
    const long long ez_index = offset3d(i_u, j_u + 1, k_u, ez_ny, ez_nz);
    adjoint += ez_curl[ez_index] * inv_dy[j + 1] * adj_ez_post[ez_index];
  }
  adj_hx_mid[linear] = adjoint;
}

__global__ void reverse_electric_to_hy_standard_kernel(
    int hy_nx,
    int hy_ny,
    int hy_nz,
    int ex_nx,
    int ex_ny,
    int ex_nz,
    int ez_nx,
    int ez_ny,
    int ez_nz,
    float* __restrict__ adj_hy_mid,
    const float* __restrict__ adj_hy_post,
    const float* __restrict__ adj_ex_post,
    const float* __restrict__ adj_ez_post,
    const float* __restrict__ ex_curl,
    const float* __restrict__ ez_curl,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dz) {
  const unsigned int k_u = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j_u = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i_u = blockIdx.z * blockDim.z + threadIdx.z;
  if (static_cast<int>(i_u) >= hy_nx || static_cast<int>(j_u) >= hy_ny || static_cast<int>(k_u) >= hy_nz) {
    return;
  }
  const int i = static_cast<int>(i_u);
  const int j = static_cast<int>(j_u);
  const int k = static_cast<int>(k_u);
  const long long linear = offset3d(i_u, j_u, k_u, hy_ny, hy_nz);
  float adjoint = adj_hy_post[linear];

  if (is_ex_active_index(i, j, k, ex_nx, ex_ny, ex_nz)) {
    const long long ex_index = offset3d(i_u, j_u, k_u, ex_ny, ex_nz);
    adjoint -= ex_curl[ex_index] * inv_dz[k] * adj_ex_post[ex_index];
  }
  if (is_ex_active_index(i, j, k + 1, ex_nx, ex_ny, ex_nz)) {
    const long long ex_index = offset3d(i_u, j_u, k_u + 1, ex_ny, ex_nz);
    adjoint += ex_curl[ex_index] * inv_dz[k + 1] * adj_ex_post[ex_index];
  }
  if (is_ez_active_index(i, j, k, ez_nx, ez_ny, ez_nz)) {
    const long long ez_index = offset3d(i_u, j_u, k_u, ez_ny, ez_nz);
    adjoint += ez_curl[ez_index] * inv_dx[i] * adj_ez_post[ez_index];
  }
  if (is_ez_active_index(i + 1, j, k, ez_nx, ez_ny, ez_nz)) {
    const long long ez_index = offset3d(i_u + 1, j_u, k_u, ez_ny, ez_nz);
    adjoint -= ez_curl[ez_index] * inv_dx[i + 1] * adj_ez_post[ez_index];
  }
  adj_hy_mid[linear] = adjoint;
}

__global__ void reverse_electric_to_hz_standard_kernel(
    int hz_nx,
    int hz_ny,
    int hz_nz,
    int ex_nx,
    int ex_ny,
    int ex_nz,
    int ey_nx,
    int ey_ny,
    int ey_nz,
    float* __restrict__ adj_hz_mid,
    const float* __restrict__ adj_hz_post,
    const float* __restrict__ adj_ex_post,
    const float* __restrict__ adj_ey_post,
    const float* __restrict__ ex_curl,
    const float* __restrict__ ey_curl,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy) {
  const unsigned int k_u = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j_u = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i_u = blockIdx.z * blockDim.z + threadIdx.z;
  if (static_cast<int>(i_u) >= hz_nx || static_cast<int>(j_u) >= hz_ny || static_cast<int>(k_u) >= hz_nz) {
    return;
  }
  const int i = static_cast<int>(i_u);
  const int j = static_cast<int>(j_u);
  const int k = static_cast<int>(k_u);
  const long long linear = offset3d(i_u, j_u, k_u, hz_ny, hz_nz);
  float adjoint = adj_hz_post[linear];

  if (is_ex_active_index(i, j, k, ex_nx, ex_ny, ex_nz)) {
    const long long ex_index = offset3d(i_u, j_u, k_u, ex_ny, ex_nz);
    adjoint += ex_curl[ex_index] * inv_dy[j] * adj_ex_post[ex_index];
  }
  if (is_ex_active_index(i, j + 1, k, ex_nx, ex_ny, ex_nz)) {
    const long long ex_index = offset3d(i_u, j_u + 1, k_u, ex_ny, ex_nz);
    adjoint -= ex_curl[ex_index] * inv_dy[j + 1] * adj_ex_post[ex_index];
  }
  if (is_ey_active_index(i, j, k, ey_nx, ey_ny, ey_nz)) {
    const long long ey_index = offset3d(i_u, j_u, k_u, ey_ny, ey_nz);
    adjoint -= ey_curl[ey_index] * inv_dx[i] * adj_ey_post[ey_index];
  }
  if (is_ey_active_index(i + 1, j, k, ey_nx, ey_ny, ey_nz)) {
    const long long ey_index = offset3d(i_u + 1, j_u, k_u, ey_ny, ey_nz);
    adjoint += ey_curl[ey_index] * inv_dx[i + 1] * adj_ey_post[ey_index];
  }
  adj_hz_mid[linear] = adjoint;
}

__global__ void reverse_magnetic_to_ex_standard_kernel(
    int ex_nx,
    int ex_ny,
    int ex_nz,
    int hy_ny,
    int hy_nz,
    int hz_ny,
    int hz_nz,
    float* __restrict__ adj_ex_prev,
    float* __restrict__ grad_eps_ex,
    const float* __restrict__ adj_ex_post,
    const float* __restrict__ adj_hy_mid,
    const float* __restrict__ adj_hz_mid,
    const float* __restrict__ ex_decay,
    const float* __restrict__ ex_curl,
    const float* __restrict__ eps_ex,
    const float* __restrict__ hy_mid,
    const float* __restrict__ hz_mid,
    const float* __restrict__ hy_curl,
    const float* __restrict__ hz_curl,
    const float* __restrict__ inv_dy_e,
    const float* __restrict__ inv_dz_e,
    const float* __restrict__ inv_dy_h,
    const float* __restrict__ inv_dz_h,
    int y_low_mode,
    int y_high_mode,
    int z_low_mode,
    int z_high_mode) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= static_cast<unsigned int>(ex_nx)
      || j >= static_cast<unsigned int>(ex_ny)
      || k >= static_cast<unsigned int>(ex_nz)) {
    return;
  }
  const long long linear = offset3d(i, j, k, ex_ny, ex_nz);
  const ElectricCellStatus status = resolve_electric_cell_status(
      static_cast<int>(j), ex_ny, y_low_mode, y_high_mode,
      static_cast<int>(k), ex_nz, z_low_mode, z_high_mode);
  float adjoint = 0.0f;
  float grad = 0.0f;
  if (status.inactive) {
    adjoint = adj_ex_post[linear];
  } else if (status.active) {
    const long long hy_index = offset3d(i, j, k, hy_ny, hy_nz);
    const long long hy_prev_k = offset3d(i, j, k - 1, hy_ny, hy_nz);
    const long long hz_index = offset3d(i, j, k, hz_ny, hz_nz);
    const long long hz_prev_j = offset3d(i, j - 1, k, hz_ny, hz_nz);
    const float curl_h = (hz_mid[hz_index] - hz_mid[hz_prev_j]) * inv_dy_e[j]
        - (hy_mid[hy_index] - hy_mid[hy_prev_k]) * inv_dz_e[k];
    adjoint = adj_ex_post[linear] * ex_decay[linear];
    grad = -adj_ex_post[linear] * ex_curl[linear] * curl_h / eps_ex[linear];
  }
  if (static_cast<int>(k) < hy_nz) {
    const long long hy_index = offset3d(i, j, k, hy_ny, hy_nz);
    adjoint += hy_curl[hy_index] * inv_dz_h[k] * adj_hy_mid[hy_index];
  }
  if (k > 0) {
    const long long hy_index = offset3d(i, j, k - 1, hy_ny, hy_nz);
    adjoint -= hy_curl[hy_index] * inv_dz_h[k - 1] * adj_hy_mid[hy_index];
  }
  if (static_cast<int>(j) < hz_ny) {
    const long long hz_index = offset3d(i, j, k, hz_ny, hz_nz);
    adjoint -= hz_curl[hz_index] * inv_dy_h[j] * adj_hz_mid[hz_index];
  }
  if (j > 0) {
    const long long hz_index = offset3d(i, j - 1, k, hz_ny, hz_nz);
    adjoint += hz_curl[hz_index] * inv_dy_h[j - 1] * adj_hz_mid[hz_index];
  }
  adj_ex_prev[linear] = adjoint;
  grad_eps_ex[linear] = grad;
}

__global__ void reverse_magnetic_to_ey_standard_kernel(
    int ey_nx,
    int ey_ny,
    int ey_nz,
    int hx_ny,
    int hx_nz,
    int hz_nx,
    int hz_ny,
    int hz_nz,
    float* __restrict__ adj_ey_prev,
    float* __restrict__ grad_eps_ey,
    const float* __restrict__ adj_ey_post,
    const float* __restrict__ adj_hx_mid,
    const float* __restrict__ adj_hz_mid,
    const float* __restrict__ ey_decay,
    const float* __restrict__ ey_curl,
    const float* __restrict__ eps_ey,
    const float* __restrict__ hx_mid,
    const float* __restrict__ hz_mid,
    const float* __restrict__ hx_curl,
    const float* __restrict__ hz_curl,
    const float* __restrict__ inv_dx_e,
    const float* __restrict__ inv_dz_e,
    const float* __restrict__ inv_dx_h,
    const float* __restrict__ inv_dz_h,
    int x_low_mode,
    int x_high_mode,
    int z_low_mode,
    int z_high_mode) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= static_cast<unsigned int>(ey_nx)
      || j >= static_cast<unsigned int>(ey_ny)
      || k >= static_cast<unsigned int>(ey_nz)) {
    return;
  }
  const long long linear = offset3d(i, j, k, ey_ny, ey_nz);
  const ElectricCellStatus status = resolve_electric_cell_status(
      static_cast<int>(i), ey_nx, x_low_mode, x_high_mode,
      static_cast<int>(k), ey_nz, z_low_mode, z_high_mode);
  float adjoint = 0.0f;
  float grad = 0.0f;
  if (status.inactive) {
    adjoint = adj_ey_post[linear];
  } else if (status.active) {
    const long long hx_index = offset3d(i, j, k, hx_ny, hx_nz);
    const long long hx_prev_k = offset3d(i, j, k - 1, hx_ny, hx_nz);
    const long long hz_index = offset3d(i, j, k, hz_ny, hz_nz);
    const long long hz_prev_i = offset3d(i - 1, j, k, hz_ny, hz_nz);
    const float curl_h = (hx_mid[hx_index] - hx_mid[hx_prev_k]) * inv_dz_e[k]
        - (hz_mid[hz_index] - hz_mid[hz_prev_i]) * inv_dx_e[i];
    const float adj_post = adj_ey_post[linear];
    adjoint = adj_post * ey_decay[linear];
    grad = -adj_post * ey_curl[linear] * curl_h / eps_ey[linear];
  }
  if (static_cast<int>(k) < hx_nz) {
    const long long hx_index = offset3d(i, j, k, hx_ny, hx_nz);
    adjoint -= hx_curl[hx_index] * inv_dz_h[k] * adj_hx_mid[hx_index];
  }
  if (k > 0) {
    const long long hx_index = offset3d(i, j, k - 1, hx_ny, hx_nz);
    adjoint += hx_curl[hx_index] * inv_dz_h[k - 1] * adj_hx_mid[hx_index];
  }
  if (static_cast<int>(i) < hz_nx) {
    const long long hz_index = offset3d(i, j, k, hz_ny, hz_nz);
    adjoint += hz_curl[hz_index] * inv_dx_h[i] * adj_hz_mid[hz_index];
  }
  if (i > 0) {
    const long long hz_index = offset3d(i - 1, j, k, hz_ny, hz_nz);
    adjoint -= hz_curl[hz_index] * inv_dx_h[i - 1] * adj_hz_mid[hz_index];
  }
  adj_ey_prev[linear] = adjoint;
  grad_eps_ey[linear] = grad;
}

__global__ void reverse_magnetic_to_ez_standard_kernel(
    int ez_nx,
    int ez_ny,
    int ez_nz,
    int hx_ny,
    int hx_nz,
    int hy_nx,
    int hy_ny,
    int hy_nz,
    float* __restrict__ adj_ez_prev,
    float* __restrict__ grad_eps_ez,
    const float* __restrict__ adj_ez_post,
    const float* __restrict__ adj_hx_mid,
    const float* __restrict__ adj_hy_mid,
    const float* __restrict__ ez_decay,
    const float* __restrict__ ez_curl,
    const float* __restrict__ eps_ez,
    const float* __restrict__ hx_mid,
    const float* __restrict__ hy_mid,
    const float* __restrict__ hx_curl,
    const float* __restrict__ hy_curl,
    const float* __restrict__ inv_dx_e,
    const float* __restrict__ inv_dy_e,
    const float* __restrict__ inv_dx_h,
    const float* __restrict__ inv_dy_h,
    int x_low_mode,
    int x_high_mode,
    int y_low_mode,
    int y_high_mode) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= static_cast<unsigned int>(ez_nx)
      || j >= static_cast<unsigned int>(ez_ny)
      || k >= static_cast<unsigned int>(ez_nz)) {
    return;
  }
  const long long linear = offset3d(i, j, k, ez_ny, ez_nz);
  const ElectricCellStatus status = resolve_electric_cell_status(
      static_cast<int>(i), ez_nx, x_low_mode, x_high_mode,
      static_cast<int>(j), ez_ny, y_low_mode, y_high_mode);
  float adjoint = 0.0f;
  float grad = 0.0f;
  if (status.inactive) {
    adjoint = adj_ez_post[linear];
  } else if (status.active) {
    const long long hx_index = offset3d(i, j, k, hx_ny, hx_nz);
    const long long hx_prev_j = offset3d(i, j - 1, k, hx_ny, hx_nz);
    const long long hy_index = offset3d(i, j, k, hy_ny, hy_nz);
    const long long hy_prev_i = offset3d(i - 1, j, k, hy_ny, hy_nz);
    const float curl_h = (hy_mid[hy_index] - hy_mid[hy_prev_i]) * inv_dx_e[i]
        - (hx_mid[hx_index] - hx_mid[hx_prev_j]) * inv_dy_e[j];
    const float adj_post = adj_ez_post[linear];
    adjoint = adj_post * ez_decay[linear];
    grad = -adj_post * ez_curl[linear] * curl_h / eps_ez[linear];
  }
  if (static_cast<int>(j) < hx_ny) {
    const long long hx_index = offset3d(i, j, k, hx_ny, hx_nz);
    adjoint += hx_curl[hx_index] * inv_dy_h[j] * adj_hx_mid[hx_index];
  }
  if (j > 0) {
    const long long hx_index = offset3d(i, j - 1, k, hx_ny, hx_nz);
    adjoint -= hx_curl[hx_index] * inv_dy_h[j - 1] * adj_hx_mid[hx_index];
  }
  if (static_cast<int>(i) < hy_nx) {
    const long long hy_index = offset3d(i, j, k, hy_ny, hy_nz);
    adjoint -= hy_curl[hy_index] * inv_dx_h[i] * adj_hy_mid[hy_index];
  }
  if (i > 0) {
    const long long hy_index = offset3d(i - 1, j, k, hy_ny, hy_nz);
    adjoint += hy_curl[hy_index] * inv_dx_h[i - 1] * adj_hy_mid[hy_index];
  }
  adj_ez_prev[linear] = adjoint;
  grad_eps_ez[linear] = grad;
}

struct Complex2 {
  float real;
  float imag;
};

__device__ __forceinline__ Complex2 complex_add(Complex2 lhs, Complex2 rhs) {
  return {lhs.real + rhs.real, lhs.imag + rhs.imag};
}

__device__ __forceinline__ Complex2 complex_sub(Complex2 lhs, Complex2 rhs) {
  return {lhs.real - rhs.real, lhs.imag - rhs.imag};
}

__device__ __forceinline__ Complex2 complex_scale(Complex2 value, float scale) {
  return {value.real * scale, value.imag * scale};
}

__device__ __forceinline__ float complex_inner_real(Complex2 lhs, Complex2 rhs) {
  return lhs.real * rhs.real + lhs.imag * rhs.imag;
}

__device__ __forceinline__ Complex2 complex_phase_positive(float phase_cos, float phase_sin, Complex2 value) {
  return {
      phase_cos * value.real - phase_sin * value.imag,
      phase_sin * value.real + phase_cos * value.imag,
  };
}

__device__ __forceinline__ Complex2 complex_phase_negative(float phase_cos, float phase_sin, Complex2 value) {
  return {
      phase_cos * value.real + phase_sin * value.imag,
      phase_cos * value.imag - phase_sin * value.real,
  };
}

__device__ __forceinline__ Complex2 load_complex_3d(
    const float* __restrict__ real_field,
    const float* __restrict__ imag_field,
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

__device__ __forceinline__ Complex2 load_scaled_complex_adjoint(
    const float* __restrict__ adj_real,
    const float* __restrict__ adj_imag,
    const float* __restrict__ curl,
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

template <int Axis>
__device__ __forceinline__ Complex2 bloch_backward_diff_axis(
    const float* __restrict__ real_field,
    const float* __restrict__ imag_field,
    int i,
    int j,
    int k,
    int size_x,
    int size_y,
    int size_z,
    float phase_cos,
    float phase_sin,
    const float* __restrict__ inv_delta) {
  const int coordinate = Axis == 0 ? i : (Axis == 1 ? j : k);
  const int field_size = Axis == 0 ? size_x : (Axis == 1 ? size_y : size_z);
  // Dual spacing at the (electric) target coordinate, matching the forward
  // Bloch backward difference (wrap entries live at [0] and [size]).
  const float inv = inv_delta[coordinate];
  if (coordinate == 0 || coordinate == field_size) {
    Complex2 low;
    Complex2 high;
    if constexpr (Axis == 0) {
      low = load_complex_3d(real_field, imag_field, 0, j, k, size_y, size_z);
      high = load_complex_3d(real_field, imag_field, size_x - 1, j, k, size_y, size_z);
    } else if constexpr (Axis == 1) {
      low = load_complex_3d(real_field, imag_field, i, 0, k, size_y, size_z);
      high = load_complex_3d(real_field, imag_field, i, size_y - 1, k, size_y, size_z);
    } else {
      low = load_complex_3d(real_field, imag_field, i, j, 0, size_y, size_z);
      high = load_complex_3d(real_field, imag_field, i, j, size_z - 1, size_y, size_z);
    }
    if (coordinate == 0) {
      return complex_scale(complex_sub(low, complex_phase_negative(phase_cos, phase_sin, high)), inv);
    }
    return complex_scale(complex_sub(complex_phase_positive(phase_cos, phase_sin, low), high), inv);
  }

  Complex2 current;
  Complex2 previous;
  if constexpr (Axis == 0) {
    current = load_complex_3d(real_field, imag_field, i, j, k, size_y, size_z);
    previous = load_complex_3d(real_field, imag_field, i - 1, j, k, size_y, size_z);
  } else if constexpr (Axis == 1) {
    current = load_complex_3d(real_field, imag_field, i, j, k, size_y, size_z);
    previous = load_complex_3d(real_field, imag_field, i, j - 1, k, size_y, size_z);
  } else {
    current = load_complex_3d(real_field, imag_field, i, j, k, size_y, size_z);
    previous = load_complex_3d(real_field, imag_field, i, j, k - 1, size_y, size_z);
  }
  return complex_scale(complex_sub(current, previous), inv);
}

template <int Axis>
__device__ __forceinline__ Complex2 gather_bloch_backward_diff_adjoint_axis(
    const float* __restrict__ adj_real,
    const float* __restrict__ adj_imag,
    const float* __restrict__ curl,
    int i,
    int j,
    int k,
    int adj_size_x,
    int adj_size_y,
    int adj_size_z,
    float sign,
    float phase_cos,
    float phase_sin,
    const float* __restrict__ inv_delta) {
  const int coordinate = Axis == 0 ? i : (Axis == 1 ? j : k);
  const int adj_axis_size = Axis == 0 ? adj_size_x : (Axis == 1 ? adj_size_y : adj_size_z);
  const int field_size = adj_axis_size - 1;
  // Exact transpose: each term is scaled by the dual spacing of the electric
  // element it pulls back from (thread = magnetic coordinate).
  Complex2 value = {0.0f, 0.0f};

  if (coordinate > 0) {
    value = complex_add(value, complex_scale(load_scaled_complex_adjoint(
        adj_real, adj_imag, curl, i, j, k, adj_size_y, adj_size_z, sign), inv_delta[coordinate]));
  }
  if (coordinate + 1 < field_size) {
    int next_i = i;
    int next_j = j;
    int next_k = k;
    if constexpr (Axis == 0) {
      ++next_i;
    } else if constexpr (Axis == 1) {
      ++next_j;
    } else {
      ++next_k;
    }
    value = complex_sub(value, complex_scale(load_scaled_complex_adjoint(
        adj_real, adj_imag, curl, next_i, next_j, next_k, adj_size_y, adj_size_z, sign), inv_delta[coordinate + 1]));
  }

  if (coordinate == 0) {
    int low_i = i;
    int low_j = j;
    int low_k = k;
    int high_i = i;
    int high_j = j;
    int high_k = k;
    if constexpr (Axis == 0) {
      low_i = 0;
      high_i = adj_size_x - 1;
    } else if constexpr (Axis == 1) {
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
    value = complex_add(value, complex_scale(low, inv_delta[0]));
    value = complex_add(value, complex_scale(complex_phase_negative(phase_cos, phase_sin, high), inv_delta[adj_axis_size - 1]));
  }
  if (coordinate + 1 == field_size) {
    int low_i = i;
    int low_j = j;
    int low_k = k;
    int high_i = i;
    int high_j = j;
    int high_k = k;
    if constexpr (Axis == 0) {
      low_i = 0;
      high_i = adj_size_x - 1;
    } else if constexpr (Axis == 1) {
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
    value = complex_sub(value, complex_scale(complex_phase_positive(phase_cos, phase_sin, low), inv_delta[0]));
    value = complex_sub(value, complex_scale(high, inv_delta[adj_axis_size - 1]));
  }

  return value;
}

__global__ void reverse_electric_to_hx_bloch_kernel(
    int hx_nx,
    int hx_ny,
    int hx_nz,
    int ey_nx,
    int ey_ny,
    int ey_nz,
    int ez_nx,
    int ez_ny,
    int ez_nz,
    float* __restrict__ adj_hx_mid_real,
    float* __restrict__ adj_hx_mid_imag,
    const float* __restrict__ adj_hx_post_real,
    const float* __restrict__ adj_hx_post_imag,
    const float* __restrict__ adj_ey_post_real,
    const float* __restrict__ adj_ey_post_imag,
    const float* __restrict__ adj_ez_post_real,
    const float* __restrict__ adj_ez_post_imag,
    const float* __restrict__ ey_curl,
    const float* __restrict__ ez_curl,
    float phase_cos_y,
    float phase_sin_y,
    float phase_cos_z,
    float phase_sin_z,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= static_cast<unsigned int>(hx_nx)
      || j >= static_cast<unsigned int>(hx_ny)
      || k >= static_cast<unsigned int>(hx_nz)) {
    return;
  }
  const long long linear = offset3d(i, j, k, hx_ny, hx_nz);
  Complex2 adjoint = load_complex_3d(adj_hx_post_real, adj_hx_post_imag, static_cast<int>(i), static_cast<int>(j), static_cast<int>(k), hx_ny, hx_nz);
  adjoint = complex_add(adjoint, gather_bloch_backward_diff_adjoint_axis<2>(
      adj_ey_post_real, adj_ey_post_imag, ey_curl, i, j, k, ey_nx, ey_ny, ey_nz,
      1.0f, phase_cos_z, phase_sin_z, inv_dz));
  adjoint = complex_add(adjoint, gather_bloch_backward_diff_adjoint_axis<1>(
      adj_ez_post_real, adj_ez_post_imag, ez_curl, i, j, k, ez_nx, ez_ny, ez_nz,
      -1.0f, phase_cos_y, phase_sin_y, inv_dy));
  adj_hx_mid_real[linear] = adjoint.real;
  adj_hx_mid_imag[linear] = adjoint.imag;
}

__global__ void reverse_electric_to_hy_bloch_kernel(
    int hy_nx,
    int hy_ny,
    int hy_nz,
    int ex_nx,
    int ex_ny,
    int ex_nz,
    int ez_nx,
    int ez_ny,
    int ez_nz,
    float* __restrict__ adj_hy_mid_real,
    float* __restrict__ adj_hy_mid_imag,
    const float* __restrict__ adj_hy_post_real,
    const float* __restrict__ adj_hy_post_imag,
    const float* __restrict__ adj_ex_post_real,
    const float* __restrict__ adj_ex_post_imag,
    const float* __restrict__ adj_ez_post_real,
    const float* __restrict__ adj_ez_post_imag,
    const float* __restrict__ ex_curl,
    const float* __restrict__ ez_curl,
    float phase_cos_x,
    float phase_sin_x,
    float phase_cos_z,
    float phase_sin_z,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dz) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= static_cast<unsigned int>(hy_nx)
      || j >= static_cast<unsigned int>(hy_ny)
      || k >= static_cast<unsigned int>(hy_nz)) {
    return;
  }
  const long long linear = offset3d(i, j, k, hy_ny, hy_nz);
  Complex2 adjoint = load_complex_3d(adj_hy_post_real, adj_hy_post_imag, static_cast<int>(i), static_cast<int>(j), static_cast<int>(k), hy_ny, hy_nz);
  adjoint = complex_add(adjoint, gather_bloch_backward_diff_adjoint_axis<2>(
      adj_ex_post_real, adj_ex_post_imag, ex_curl, i, j, k, ex_nx, ex_ny, ex_nz,
      -1.0f, phase_cos_z, phase_sin_z, inv_dz));
  adjoint = complex_add(adjoint, gather_bloch_backward_diff_adjoint_axis<0>(
      adj_ez_post_real, adj_ez_post_imag, ez_curl, i, j, k, ez_nx, ez_ny, ez_nz,
      1.0f, phase_cos_x, phase_sin_x, inv_dx));
  adj_hy_mid_real[linear] = adjoint.real;
  adj_hy_mid_imag[linear] = adjoint.imag;
}

__global__ void reverse_electric_to_hz_bloch_kernel(
    int hz_nx,
    int hz_ny,
    int hz_nz,
    int ex_nx,
    int ex_ny,
    int ex_nz,
    int ey_nx,
    int ey_ny,
    int ey_nz,
    float* __restrict__ adj_hz_mid_real,
    float* __restrict__ adj_hz_mid_imag,
    const float* __restrict__ adj_hz_post_real,
    const float* __restrict__ adj_hz_post_imag,
    const float* __restrict__ adj_ex_post_real,
    const float* __restrict__ adj_ex_post_imag,
    const float* __restrict__ adj_ey_post_real,
    const float* __restrict__ adj_ey_post_imag,
    const float* __restrict__ ex_curl,
    const float* __restrict__ ey_curl,
    float phase_cos_x,
    float phase_sin_x,
    float phase_cos_y,
    float phase_sin_y,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= static_cast<unsigned int>(hz_nx)
      || j >= static_cast<unsigned int>(hz_ny)
      || k >= static_cast<unsigned int>(hz_nz)) {
    return;
  }
  const long long linear = offset3d(i, j, k, hz_ny, hz_nz);
  Complex2 adjoint = load_complex_3d(adj_hz_post_real, adj_hz_post_imag, static_cast<int>(i), static_cast<int>(j), static_cast<int>(k), hz_ny, hz_nz);
  adjoint = complex_add(adjoint, gather_bloch_backward_diff_adjoint_axis<1>(
      adj_ex_post_real, adj_ex_post_imag, ex_curl, i, j, k, ex_nx, ex_ny, ex_nz,
      1.0f, phase_cos_y, phase_sin_y, inv_dy));
  adjoint = complex_add(adjoint, gather_bloch_backward_diff_adjoint_axis<0>(
      adj_ey_post_real, adj_ey_post_imag, ey_curl, i, j, k, ey_nx, ey_ny, ey_nz,
      -1.0f, phase_cos_x, phase_sin_x, inv_dx));
  adj_hz_mid_real[linear] = adjoint.real;
  adj_hz_mid_imag[linear] = adjoint.imag;
}

__global__ void reverse_magnetic_to_ex_bloch_kernel(
    int ex_nx,
    int ex_ny,
    int ex_nz,
    int hy_nx,
    int hy_ny,
    int hy_nz,
    int hz_nx,
    int hz_ny,
    int hz_nz,
    float* __restrict__ adj_ex_prev_real,
    float* __restrict__ adj_ex_prev_imag,
    float* __restrict__ grad_eps_ex,
    const float* __restrict__ adj_ex_post_real,
    const float* __restrict__ adj_ex_post_imag,
    const float* __restrict__ adj_hy_mid_real,
    const float* __restrict__ adj_hy_mid_imag,
    const float* __restrict__ adj_hz_mid_real,
    const float* __restrict__ adj_hz_mid_imag,
    const float* __restrict__ ex_decay,
    const float* __restrict__ ex_curl,
    const float* __restrict__ eps_ex,
    const float* __restrict__ hy_mid_real,
    const float* __restrict__ hy_mid_imag,
    const float* __restrict__ hz_mid_real,
    const float* __restrict__ hz_mid_imag,
    const float* __restrict__ hy_curl,
    const float* __restrict__ hz_curl,
    float phase_cos_y,
    float phase_sin_y,
    float phase_cos_z,
    float phase_sin_z,
    const float* __restrict__ inv_dy_e,
    const float* __restrict__ inv_dz_e,
    const float* __restrict__ inv_dy_h,
    const float* __restrict__ inv_dz_h) {
  const unsigned int k_u = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j_u = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i_u = blockIdx.z * blockDim.z + threadIdx.z;
  if (i_u >= static_cast<unsigned int>(ex_nx)
      || j_u >= static_cast<unsigned int>(ex_ny)
      || k_u >= static_cast<unsigned int>(ex_nz)) {
    return;
  }
  const long long linear = offset3d(i_u, j_u, k_u, ex_ny, ex_nz);
  const int i = static_cast<int>(i_u);
  const int j = static_cast<int>(j_u);
  const int k = static_cast<int>(k_u);
  const Complex2 adj_ex_post = load_complex_3d(adj_ex_post_real, adj_ex_post_imag, i, j, k, ex_ny, ex_nz);
  const Complex2 d_hz_dy = bloch_backward_diff_axis<1>(
      hz_mid_real, hz_mid_imag, i, j, k, hz_nx, hz_ny, hz_nz, phase_cos_y, phase_sin_y, inv_dy_e);
  const Complex2 d_hy_dz = bloch_backward_diff_axis<2>(
      hy_mid_real, hy_mid_imag, i, j, k, hy_nx, hy_ny, hy_nz, phase_cos_z, phase_sin_z, inv_dz_e);
  const Complex2 curl_h = complex_sub(d_hz_dy, d_hy_dz);
  float adjoint_real = adj_ex_post.real * ex_decay[linear];
  float adjoint_imag = adj_ex_post.imag * ex_decay[linear];
  float grad = -ex_curl[linear] * complex_inner_real(adj_ex_post, curl_h) / eps_ex[linear];

  if (k < hy_nz) {
    const long long hy_index = offset3d(i_u, j_u, k_u, static_cast<unsigned int>(hy_ny), static_cast<unsigned int>(hy_nz));
    adjoint_real += hy_curl[hy_index] * inv_dz_h[k] * adj_hy_mid_real[hy_index];
    adjoint_imag += hy_curl[hy_index] * inv_dz_h[k] * adj_hy_mid_imag[hy_index];
  }
  if (k > 0) {
    const long long hy_index = offset3d(i_u, j_u, k_u - 1, static_cast<unsigned int>(hy_ny), static_cast<unsigned int>(hy_nz));
    adjoint_real -= hy_curl[hy_index] * inv_dz_h[k - 1] * adj_hy_mid_real[hy_index];
    adjoint_imag -= hy_curl[hy_index] * inv_dz_h[k - 1] * adj_hy_mid_imag[hy_index];
  }
  if (j < hz_ny) {
    const long long hz_index = offset3d(i_u, j_u, k_u, static_cast<unsigned int>(hz_ny), static_cast<unsigned int>(hz_nz));
    adjoint_real -= hz_curl[hz_index] * inv_dy_h[j] * adj_hz_mid_real[hz_index];
    adjoint_imag -= hz_curl[hz_index] * inv_dy_h[j] * adj_hz_mid_imag[hz_index];
  }
  if (j > 0) {
    const long long hz_index = offset3d(i_u, j_u - 1, k_u, static_cast<unsigned int>(hz_ny), static_cast<unsigned int>(hz_nz));
    adjoint_real += hz_curl[hz_index] * inv_dy_h[j - 1] * adj_hz_mid_real[hz_index];
    adjoint_imag += hz_curl[hz_index] * inv_dy_h[j - 1] * adj_hz_mid_imag[hz_index];
  }
  adj_ex_prev_real[linear] = adjoint_real;
  adj_ex_prev_imag[linear] = adjoint_imag;
  grad_eps_ex[linear] = grad;
}

__global__ void reverse_magnetic_to_ey_bloch_kernel(
    int ey_nx,
    int ey_ny,
    int ey_nz,
    int hx_nx,
    int hx_ny,
    int hx_nz,
    int hz_nx,
    int hz_ny,
    int hz_nz,
    float* __restrict__ adj_ey_prev_real,
    float* __restrict__ adj_ey_prev_imag,
    float* __restrict__ grad_eps_ey,
    const float* __restrict__ adj_ey_post_real,
    const float* __restrict__ adj_ey_post_imag,
    const float* __restrict__ adj_hx_mid_real,
    const float* __restrict__ adj_hx_mid_imag,
    const float* __restrict__ adj_hz_mid_real,
    const float* __restrict__ adj_hz_mid_imag,
    const float* __restrict__ ey_decay,
    const float* __restrict__ ey_curl,
    const float* __restrict__ eps_ey,
    const float* __restrict__ hx_mid_real,
    const float* __restrict__ hx_mid_imag,
    const float* __restrict__ hz_mid_real,
    const float* __restrict__ hz_mid_imag,
    const float* __restrict__ hx_curl,
    const float* __restrict__ hz_curl,
    float phase_cos_x,
    float phase_sin_x,
    float phase_cos_z,
    float phase_sin_z,
    const float* __restrict__ inv_dx_e,
    const float* __restrict__ inv_dz_e,
    const float* __restrict__ inv_dx_h,
    const float* __restrict__ inv_dz_h) {
  const unsigned int k_u = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j_u = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i_u = blockIdx.z * blockDim.z + threadIdx.z;
  if (i_u >= static_cast<unsigned int>(ey_nx)
      || j_u >= static_cast<unsigned int>(ey_ny)
      || k_u >= static_cast<unsigned int>(ey_nz)) {
    return;
  }
  const long long linear = offset3d(i_u, j_u, k_u, ey_ny, ey_nz);
  const int i = static_cast<int>(i_u);
  const int j = static_cast<int>(j_u);
  const int k = static_cast<int>(k_u);
  const Complex2 adj_ey_post = load_complex_3d(adj_ey_post_real, adj_ey_post_imag, i, j, k, ey_ny, ey_nz);
  const Complex2 d_hx_dz = bloch_backward_diff_axis<2>(
      hx_mid_real, hx_mid_imag, i, j, k, hx_nx, hx_ny, hx_nz, phase_cos_z, phase_sin_z, inv_dz_e);
  const Complex2 d_hz_dx = bloch_backward_diff_axis<0>(
      hz_mid_real, hz_mid_imag, i, j, k, hz_nx, hz_ny, hz_nz, phase_cos_x, phase_sin_x, inv_dx_e);
  const Complex2 curl_h = complex_sub(d_hx_dz, d_hz_dx);
  float adjoint_real = adj_ey_post.real * ey_decay[linear];
  float adjoint_imag = adj_ey_post.imag * ey_decay[linear];
  float grad = -ey_curl[linear] * complex_inner_real(adj_ey_post, curl_h) / eps_ey[linear];

  if (k < hx_nz) {
    const long long hx_index = offset3d(i_u, j_u, k_u, static_cast<unsigned int>(hx_ny), static_cast<unsigned int>(hx_nz));
    adjoint_real -= hx_curl[hx_index] * inv_dz_h[k] * adj_hx_mid_real[hx_index];
    adjoint_imag -= hx_curl[hx_index] * inv_dz_h[k] * adj_hx_mid_imag[hx_index];
  }
  if (k > 0) {
    const long long hx_index = offset3d(i_u, j_u, k_u - 1, static_cast<unsigned int>(hx_ny), static_cast<unsigned int>(hx_nz));
    adjoint_real += hx_curl[hx_index] * inv_dz_h[k - 1] * adj_hx_mid_real[hx_index];
    adjoint_imag += hx_curl[hx_index] * inv_dz_h[k - 1] * adj_hx_mid_imag[hx_index];
  }
  if (i < hz_nx) {
    const long long hz_index = offset3d(i_u, j_u, k_u, static_cast<unsigned int>(hz_ny), static_cast<unsigned int>(hz_nz));
    adjoint_real += hz_curl[hz_index] * inv_dx_h[i] * adj_hz_mid_real[hz_index];
    adjoint_imag += hz_curl[hz_index] * inv_dx_h[i] * adj_hz_mid_imag[hz_index];
  }
  if (i > 0) {
    const long long hz_index = offset3d(i_u - 1, j_u, k_u, static_cast<unsigned int>(hz_ny), static_cast<unsigned int>(hz_nz));
    adjoint_real -= hz_curl[hz_index] * inv_dx_h[i - 1] * adj_hz_mid_real[hz_index];
    adjoint_imag -= hz_curl[hz_index] * inv_dx_h[i - 1] * adj_hz_mid_imag[hz_index];
  }
  adj_ey_prev_real[linear] = adjoint_real;
  adj_ey_prev_imag[linear] = adjoint_imag;
  grad_eps_ey[linear] = grad;
}

__global__ void reverse_magnetic_to_ez_bloch_kernel(
    int ez_nx,
    int ez_ny,
    int ez_nz,
    int hx_nx,
    int hx_ny,
    int hx_nz,
    int hy_nx,
    int hy_ny,
    int hy_nz,
    float* __restrict__ adj_ez_prev_real,
    float* __restrict__ adj_ez_prev_imag,
    float* __restrict__ grad_eps_ez,
    const float* __restrict__ adj_ez_post_real,
    const float* __restrict__ adj_ez_post_imag,
    const float* __restrict__ adj_hx_mid_real,
    const float* __restrict__ adj_hx_mid_imag,
    const float* __restrict__ adj_hy_mid_real,
    const float* __restrict__ adj_hy_mid_imag,
    const float* __restrict__ ez_decay,
    const float* __restrict__ ez_curl,
    const float* __restrict__ eps_ez,
    const float* __restrict__ hx_mid_real,
    const float* __restrict__ hx_mid_imag,
    const float* __restrict__ hy_mid_real,
    const float* __restrict__ hy_mid_imag,
    const float* __restrict__ hx_curl,
    const float* __restrict__ hy_curl,
    float phase_cos_x,
    float phase_sin_x,
    float phase_cos_y,
    float phase_sin_y,
    const float* __restrict__ inv_dx_e,
    const float* __restrict__ inv_dy_e,
    const float* __restrict__ inv_dx_h,
    const float* __restrict__ inv_dy_h) {
  const unsigned int k_u = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j_u = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i_u = blockIdx.z * blockDim.z + threadIdx.z;
  if (i_u >= static_cast<unsigned int>(ez_nx)
      || j_u >= static_cast<unsigned int>(ez_ny)
      || k_u >= static_cast<unsigned int>(ez_nz)) {
    return;
  }
  const long long linear = offset3d(i_u, j_u, k_u, ez_ny, ez_nz);
  const int i = static_cast<int>(i_u);
  const int j = static_cast<int>(j_u);
  const int k = static_cast<int>(k_u);
  const Complex2 adj_ez_post = load_complex_3d(adj_ez_post_real, adj_ez_post_imag, i, j, k, ez_ny, ez_nz);
  const Complex2 d_hy_dx = bloch_backward_diff_axis<0>(
      hy_mid_real, hy_mid_imag, i, j, k, hy_nx, hy_ny, hy_nz, phase_cos_x, phase_sin_x, inv_dx_e);
  const Complex2 d_hx_dy = bloch_backward_diff_axis<1>(
      hx_mid_real, hx_mid_imag, i, j, k, hx_nx, hx_ny, hx_nz, phase_cos_y, phase_sin_y, inv_dy_e);
  const Complex2 curl_h = complex_sub(d_hy_dx, d_hx_dy);
  float adjoint_real = adj_ez_post.real * ez_decay[linear];
  float adjoint_imag = adj_ez_post.imag * ez_decay[linear];
  float grad = -ez_curl[linear] * complex_inner_real(adj_ez_post, curl_h) / eps_ez[linear];

  if (j < hx_ny) {
    const long long hx_index = offset3d(i_u, j_u, k_u, static_cast<unsigned int>(hx_ny), static_cast<unsigned int>(hx_nz));
    adjoint_real += hx_curl[hx_index] * inv_dy_h[j] * adj_hx_mid_real[hx_index];
    adjoint_imag += hx_curl[hx_index] * inv_dy_h[j] * adj_hx_mid_imag[hx_index];
  }
  if (j > 0) {
    const long long hx_index = offset3d(i_u, j_u - 1, k_u, static_cast<unsigned int>(hx_ny), static_cast<unsigned int>(hx_nz));
    adjoint_real -= hx_curl[hx_index] * inv_dy_h[j - 1] * adj_hx_mid_real[hx_index];
    adjoint_imag -= hx_curl[hx_index] * inv_dy_h[j - 1] * adj_hx_mid_imag[hx_index];
  }
  if (i < hy_nx) {
    const long long hy_index = offset3d(i_u, j_u, k_u, static_cast<unsigned int>(hy_ny), static_cast<unsigned int>(hy_nz));
    adjoint_real -= hy_curl[hy_index] * inv_dx_h[i] * adj_hy_mid_real[hy_index];
    adjoint_imag -= hy_curl[hy_index] * inv_dx_h[i] * adj_hy_mid_imag[hy_index];
  }
  if (i > 0) {
    const long long hy_index = offset3d(i_u - 1, j_u, k_u, static_cast<unsigned int>(hy_ny), static_cast<unsigned int>(hy_nz));
    adjoint_real += hy_curl[hy_index] * inv_dx_h[i - 1] * adj_hy_mid_real[hy_index];
    adjoint_imag += hy_curl[hy_index] * inv_dx_h[i - 1] * adj_hy_mid_imag[hy_index];
  }
  adj_ez_prev_real[linear] = adjoint_real;
  adj_ez_prev_imag[linear] = adjoint_imag;
  grad_eps_ez[linear] = grad;
}

template <int Axis, bool Forward>
__global__ void accumulate_diff_adjoint_kernel(
    int field_nx,
    int field_ny,
    int field_nz,
    int diff_nx,
    int diff_ny,
    int diff_nz,
    const float* __restrict__ inv_delta,
    float* __restrict__ field_grad,
    const float* __restrict__ diff_grad) {
  const unsigned int k_u = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j_u = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i_u = blockIdx.z * blockDim.z + threadIdx.z;
  if (static_cast<int>(i_u) >= field_nx || static_cast<int>(j_u) >= field_ny || static_cast<int>(k_u) >= field_nz) {
    return;
  }
  const long long linear = offset3d(i_u, j_u, k_u, static_cast<unsigned int>(field_ny), static_cast<unsigned int>(field_nz));
  const int i = static_cast<int>(i_u);
  const int j = static_cast<int>(j_u);
  const int k = static_cast<int>(k_u);
  const int axis_coord = Axis == 0 ? i : (Axis == 1 ? j : k);
  const int axis_field_size = Axis == 0 ? field_nx : (Axis == 1 ? field_ny : field_nz);
  const int axis_diff_size = Axis == 0 ? diff_nx : (Axis == 1 ? diff_ny : diff_nz);

  // Exact transpose: each term is scaled by the spacing element of the diff
  // output it pulls back from (inv_delta has length axis_diff_size).
  float value = 0.0f;
  if constexpr (Forward) {
    if (axis_coord < axis_diff_size && diff_index_valid(i, j, k, diff_nx, diff_ny, diff_nz)) {
      value -= inv_delta[axis_coord] * diff_grad_value(diff_grad, i, j, k, diff_ny, diff_nz);
    }
    if (axis_coord > 0) {
      const int prev_i = i - (Axis == 0 ? 1 : 0);
      const int prev_j = j - (Axis == 1 ? 1 : 0);
      const int prev_k = k - (Axis == 2 ? 1 : 0);
      if (diff_index_valid(prev_i, prev_j, prev_k, diff_nx, diff_ny, diff_nz)) {
        value += inv_delta[axis_coord - 1] * diff_grad_value(diff_grad, prev_i, prev_j, prev_k, diff_ny, diff_nz);
      }
    }
  } else {
    if (axis_coord > 0 && diff_index_valid(i, j, k, diff_nx, diff_ny, diff_nz)) {
      value += inv_delta[axis_coord] * diff_grad_value(diff_grad, i, j, k, diff_ny, diff_nz);
    }
    if (axis_coord + 1 < axis_field_size) {
      const int next_i = i + (Axis == 0 ? 1 : 0);
      const int next_j = j + (Axis == 1 ? 1 : 0);
      const int next_k = k + (Axis == 2 ? 1 : 0);
      if (diff_index_valid(next_i, next_j, next_k, diff_nx, diff_ny, diff_nz)) {
        value -= inv_delta[axis_coord + 1] * diff_grad_value(diff_grad, next_i, next_j, next_k, diff_ny, diff_nz);
      }
    }
  }
  field_grad[linear] += value;
}

template <int Axis, bool Forward>
void launch_accumulate_diff_adjoint(
    const torch::stable::Tensor& field_grad,
    const torch::stable::Tensor& diff_grad,
    const torch::stable::Tensor& inv_delta) {
  const dim3 block = field_block3d();
  accumulate_diff_adjoint_kernel<Axis, Forward><<<field_grid3d(field_grad.size(0), field_grad.size(1), field_grad.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<int>(field_grad.size(0)),
      static_cast<int>(field_grad.size(1)),
      static_cast<int>(field_grad.size(2)),
      static_cast<int>(diff_grad.size(0)),
      static_cast<int>(diff_grad.size(1)),
      static_cast<int>(diff_grad.size(2)),
      inv_delta.mutable_data_ptr<float>(),
      field_grad.mutable_data_ptr<float>(),
      diff_grad.mutable_data_ptr<float>());
}

__global__ void reverse_debye_current_kernel(
    int64_t total,
    float* __restrict__ adj_electric_prev,
    float* __restrict__ adj_polarization_prev,
    const float* __restrict__ adj_polarization_post,
    const float* __restrict__ adj_current_post,
    const float* __restrict__ drive,
    float decay,
    float inv_dt) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const float adj_current_value = adj_current_post[index];
  const float adj_polarization_value = adj_polarization_post[index];
  const float drive_value = drive[index];
  const float adj_current_scaled = adj_current_value * inv_dt;
  const float adj_internal = adj_polarization_value + adj_current_scaled;
  adj_electric_prev[index] += drive_value * adj_internal;
  adj_polarization_prev[index] += decay * adj_internal - adj_current_scaled;
}

__global__ void reverse_drude_current_kernel(
    int64_t total,
    float* __restrict__ adj_electric_prev,
    float* __restrict__ adj_current_prev,
    const float* __restrict__ adj_current_post,
    const float* __restrict__ drive,
    float decay) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  adj_electric_prev[index] += drive[index] * adj_current_post[index];
  adj_current_prev[index] += decay * adj_current_post[index];
}

__global__ void reverse_lorentz_current_kernel(
    int64_t total,
    float* __restrict__ adj_electric_prev,
    float* __restrict__ adj_polarization_prev,
    float* __restrict__ adj_current_prev,
    const float* __restrict__ adj_polarization_post,
    const float* __restrict__ adj_current_post,
    const float* __restrict__ drive,
    float decay,
    float restoring,
    float dt) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const float adj_internal = adj_current_post[index] + dt * adj_polarization_post[index];
  adj_electric_prev[index] += drive[index] * adj_internal;
  adj_polarization_prev[index] += adj_polarization_post[index] - restoring * adj_internal;
  adj_current_prev[index] += decay * adj_internal;
}

__global__ void reverse_dispersive_correction_kernel(
    int64_t total,
    float* __restrict__ adj_current_corrected,
    float* __restrict__ grad_eps,
    const float* __restrict__ adj_current_post,
    const float* __restrict__ adj_electric_post,
    const float* __restrict__ current,
    const float* __restrict__ eps,
    float dt) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const float adj_electric_value = adj_electric_post[index];
  const float inv_eps = 1.0f / eps[index];
  const float dt_adj_over_eps = dt * adj_electric_value * inv_eps;
  // Post-step dispersive-current adjoint carries the -dt/eps correction the
  // electric update applied when it subtracted dt * J / eps from the field.
  adj_current_corrected[index] = adj_current_post[index] - dt_adj_over_eps;
  // The 1/eps coupling contributes an eps gradient dt * J * adj_E / eps^2; it
  // accumulates on top of the base reverse step's eps gradient (multiple poles
  // on the same component add here in turn).
  grad_eps[index] += current[index] * dt_adj_over_eps * inv_eps;
}

__global__ void reverse_tfsf_auxiliary_electric_kernel(
    int64_t total,
    int64_t magnetic_total,
    int64_t source_index,
    float* __restrict__ adj_electric_prev,
    float* __restrict__ adj_magnetic_after,
    const float* __restrict__ adj_electric_post,
    const float* __restrict__ electric_decay,
    const float* __restrict__ electric_curl) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total && index >= magnetic_total) {
    return;
  }
  if (index < total) {
    const bool overwritten = index == source_index || index == total - 1;
    const float adjoint = adj_electric_post[index];
    if (index == 0 && !overwritten) {
      adj_electric_prev[index] += adjoint;
    } else if (index > 0 && index + 1 < total && !overwritten) {
      adj_electric_prev[index] += electric_decay[index] * adjoint;
    }
  }
  if (index < magnetic_total) {
    float adjoint = 0.0f;
    const int64_t lower_electric = index;
    const int64_t upper_electric = index + 1;
    if (lower_electric > 0 && lower_electric + 1 < total && lower_electric != source_index) {
      adjoint -= electric_curl[lower_electric] * adj_electric_post[lower_electric];
    }
    if (upper_electric > 0 && upper_electric + 1 < total && upper_electric != source_index) {
      adjoint += electric_curl[upper_electric] * adj_electric_post[upper_electric];
    }
    adj_magnetic_after[index] += adjoint;
  }
}

template <int Component>
__global__ void reverse_electric_component_cpml_kernel(
    int prev_nx,
    int prev_ny,
    int prev_nz,
    int h_pos_ny,
    int h_pos_nz,
    int h_neg_ny,
    int h_neg_nz,
    float* __restrict__ adj_prev,
    float* __restrict__ grad_eps,
    float* __restrict__ adj_psi_pos_prev,
    float* __restrict__ adj_psi_neg_prev,
    float* __restrict__ adj_d_pos,
    float* __restrict__ adj_d_neg,
    const float* __restrict__ adj_post,
    const float* __restrict__ adj_psi_pos_post,
    const float* __restrict__ adj_psi_neg_post,
    const float* __restrict__ decay,
    const float* __restrict__ curl,
    const float* __restrict__ eps,
    const float* __restrict__ psi_pos,
    const float* __restrict__ psi_neg,
    const float* __restrict__ b_pos,
    const float* __restrict__ c_pos,
    const float* __restrict__ inv_kappa_pos,
    const float* __restrict__ b_neg,
    const float* __restrict__ c_neg,
    const float* __restrict__ inv_kappa_neg,
    const float* __restrict__ h_pos_mid,
    const float* __restrict__ h_neg_mid,
    const float* __restrict__ inv_pos,
    const float* __restrict__ inv_neg,
    int low_mode_a,
    int high_mode_a,
    int low_mode_b,
    int high_mode_b) {
  const unsigned int k_u = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j_u = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i_u = blockIdx.z * blockDim.z + threadIdx.z;
  if (i_u >= static_cast<unsigned int>(prev_nx)
      || j_u >= static_cast<unsigned int>(prev_ny)
      || k_u >= static_cast<unsigned int>(prev_nz)) {
    return;
  }
  const long long linear = offset3d(i_u, j_u, k_u, prev_ny, prev_nz);
  const int i = static_cast<int>(i_u);
  const int j = static_cast<int>(j_u);
  const int k = static_cast<int>(k_u);

  int coord_a = j;
  int size_a = prev_ny;
  int coord_b = k;
  int size_b = prev_nz;
  int pos_coeff_index = j;
  int neg_coeff_index = k;

  if constexpr (Component == 1) {
    coord_a = i;
    size_a = prev_nx;
    coord_b = k;
    size_b = prev_nz;
    pos_coeff_index = k;
    neg_coeff_index = i;
  } else if constexpr (Component == 2) {
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
  const float adj_post_value = adj_post[linear];
  float adjoint = 0.0f;
  float grad = 0.0f;
  const float adj_psi_pos_post_value = adj_psi_pos_post[linear];
  const float adj_psi_neg_post_value = adj_psi_neg_post[linear];
  float adj_psi_pos = adj_psi_pos_post_value;
  float adj_psi_neg = adj_psi_neg_post_value;
  float out_adj_d_pos = 0.0f;
  float out_adj_d_neg = 0.0f;

  if (status.inactive) {
    adjoint = adj_post_value;
  } else if (status.active) {
    float d_pos = 0.0f;
    float d_neg = 0.0f;
    // Dual spacing at the electric target coordinate (== the CPML coefficient
    // index along each axis), matching the forward CPML difference.
    if constexpr (Component == 0) {
      d_pos = (h_pos_mid[offset3d(i_u, j_u, k_u, h_pos_ny, h_pos_nz)]
               - h_pos_mid[offset3d(i_u, j_u - 1, k_u, h_pos_ny, h_pos_nz)]) * inv_pos[pos_coeff_index];
      d_neg = (h_neg_mid[offset3d(i_u, j_u, k_u, h_neg_ny, h_neg_nz)]
               - h_neg_mid[offset3d(i_u, j_u, k_u - 1, h_neg_ny, h_neg_nz)]) * inv_neg[neg_coeff_index];
    } else if constexpr (Component == 1) {
      d_pos = (h_pos_mid[offset3d(i_u, j_u, k_u, h_pos_ny, h_pos_nz)]
               - h_pos_mid[offset3d(i_u, j_u, k_u - 1, h_pos_ny, h_pos_nz)]) * inv_pos[pos_coeff_index];
      d_neg = (h_neg_mid[offset3d(i_u, j_u, k_u, h_neg_ny, h_neg_nz)]
               - h_neg_mid[offset3d(i_u - 1, j_u, k_u, h_neg_ny, h_neg_nz)]) * inv_neg[neg_coeff_index];
    } else {
      d_pos = (h_pos_mid[offset3d(i_u, j_u, k_u, h_pos_ny, h_pos_nz)]
               - h_pos_mid[offset3d(i_u - 1, j_u, k_u, h_pos_ny, h_pos_nz)]) * inv_pos[pos_coeff_index];
      d_neg = (h_neg_mid[offset3d(i_u, j_u, k_u, h_neg_ny, h_neg_nz)]
               - h_neg_mid[offset3d(i_u, j_u - 1, k_u, h_neg_ny, h_neg_nz)]) * inv_neg[neg_coeff_index];
    }
    const float b_pos_value = b_pos[pos_coeff_index];
    const float c_pos_value = c_pos[pos_coeff_index];
    const float inv_kappa_pos_value = inv_kappa_pos[pos_coeff_index];
    const float b_neg_value = b_neg[neg_coeff_index];
    const float c_neg_value = c_neg[neg_coeff_index];
    const float inv_kappa_neg_value = inv_kappa_neg[neg_coeff_index];
    const float psi_pos_candidate = b_pos_value * psi_pos[linear] + c_pos_value * d_pos;
    const float psi_neg_candidate = b_neg_value * psi_neg[linear] + c_neg_value * d_neg;
    const float curl_h = (d_pos * inv_kappa_pos_value + psi_pos_candidate)
        - (d_neg * inv_kappa_neg_value + psi_neg_candidate);
    const float adj_curl_h = adj_post_value * curl[linear];
    const float adj_psi_pos_total = adj_psi_pos_post_value + adj_curl_h;
    const float adj_psi_neg_total = adj_psi_neg_post_value - adj_curl_h;
    adjoint = adj_post_value * decay[linear];
    grad = -adj_curl_h * curl_h / eps[linear];
    adj_psi_pos = b_pos_value * adj_psi_pos_total;
    adj_psi_neg = b_neg_value * adj_psi_neg_total;
    out_adj_d_pos = inv_kappa_pos_value * adj_curl_h + c_pos_value * adj_psi_pos_total;
    out_adj_d_neg = -inv_kappa_neg_value * adj_curl_h + c_neg_value * adj_psi_neg_total;
  }

  adj_prev[linear] = adjoint;
  grad_eps[linear] = grad;
  adj_psi_pos_prev[linear] = adj_psi_pos;
  adj_psi_neg_prev[linear] = adj_psi_neg;
  adj_d_pos[linear] = out_adj_d_pos;
  adj_d_neg[linear] = out_adj_d_neg;
}

template <int Component>
__global__ void reverse_magnetic_component_cpml_kernel(
    int prev_nx,
    int prev_ny,
    int prev_nz,
    float* __restrict__ adj_prev,
    float* __restrict__ adj_psi_pos_prev,
    float* __restrict__ adj_psi_neg_prev,
    float* __restrict__ adj_d_pos,
    float* __restrict__ adj_d_neg,
    const float* __restrict__ adj_post,
    const float* __restrict__ adj_psi_pos_post,
    const float* __restrict__ adj_psi_neg_post,
    const float* __restrict__ decay,
    const float* __restrict__ curl,
    const float* __restrict__ b_pos,
    const float* __restrict__ c_pos,
    const float* __restrict__ inv_kappa_pos,
    const float* __restrict__ b_neg,
    const float* __restrict__ c_neg,
    const float* __restrict__ inv_kappa_neg) {
  const unsigned int k_u = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j_u = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i_u = blockIdx.z * blockDim.z + threadIdx.z;
  if (i_u >= static_cast<unsigned int>(prev_nx)
      || j_u >= static_cast<unsigned int>(prev_ny)
      || k_u >= static_cast<unsigned int>(prev_nz)) {
    return;
  }
  const long long linear = offset3d(i_u, j_u, k_u, prev_ny, prev_nz);
  int pos_coeff_index = static_cast<int>(j_u);
  int neg_coeff_index = static_cast<int>(k_u);
  if constexpr (Component == 1) {
    pos_coeff_index = static_cast<int>(k_u);
    neg_coeff_index = static_cast<int>(i_u);
  } else if constexpr (Component == 2) {
    pos_coeff_index = static_cast<int>(i_u);
    neg_coeff_index = static_cast<int>(j_u);
  }

  const float adj_post_value = adj_post[linear];
  const float adj_curl_e = -curl[linear] * adj_post_value;
  const float adj_psi_pos_candidate = adj_psi_pos_post[linear] + adj_curl_e;
  const float adj_psi_neg_candidate = adj_psi_neg_post[linear] - adj_curl_e;
  const float b_pos_value = b_pos[pos_coeff_index];
  const float c_pos_value = c_pos[pos_coeff_index];
  const float inv_kappa_pos_value = inv_kappa_pos[pos_coeff_index];
  const float b_neg_value = b_neg[neg_coeff_index];
  const float c_neg_value = c_neg[neg_coeff_index];
  const float inv_kappa_neg_value = inv_kappa_neg[neg_coeff_index];
  adj_prev[linear] = adj_post_value * decay[linear];
  adj_psi_pos_prev[linear] = b_pos_value * adj_psi_pos_candidate;
  adj_psi_neg_prev[linear] = b_neg_value * adj_psi_neg_candidate;
  adj_d_pos[linear] = inv_kappa_pos_value * adj_curl_e + c_pos_value * adj_psi_pos_candidate;
  adj_d_neg[linear] = -inv_kappa_neg_value * adj_curl_e + c_neg_value * adj_psi_neg_candidate;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  value += __shfl_down_sync(0xffffffff, value, 16);
  value += __shfl_down_sync(0xffffffff, value, 8);
  value += __shfl_down_sync(0xffffffff, value, 4);
  value += __shfl_down_sync(0xffffffff, value, 2);
  value += __shfl_down_sync(0xffffffff, value, 1);
  return value;
}

__device__ __forceinline__ float warp_reduce_sum_active(float value, unsigned int mask) {
  const unsigned int lane = threadIdx.x & 31u;
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    const float other = __shfl_down_sync(mask, value, delta);
    if ((lane + static_cast<unsigned int>(delta)) < 32u && ((mask >> (lane + delta)) & 1u) != 0u) {
      value += other;
    }
  }
  return value;
}

__global__ void accumulate_tfsf_scalar_sample_adjoint_kernel(
    int64_t total,
    int patch_ny,
    int patch_nz,
    int64_t sample_index,
    float component_scale,
    float* __restrict__ adj_aux_field,
    const float* __restrict__ adj_field_patch,
    const float* __restrict__ coeff_patch) {
  __shared__ float warp_sums[32];
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  (void)patch_ny;
  (void)patch_nz;
  float value = linear < total
      ? component_scale * adj_field_patch[linear] * coeff_patch[linear]
      : 0.0f;
  value = warp_reduce_sum(value);
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < ((blockDim.x + 31u) >> 5) ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
  }
  if (threadIdx.x == 0) {
    atomicAdd(adj_aux_field + sample_index, value);
  }
}

template <int SampleAxis>
__global__ void accumulate_tfsf_line_sample_adjoint_kernel(
    int patch_nx,
    int patch_ny,
    int patch_nz,
    float component_scale,
    float* __restrict__ adj_aux_field,
    const float* __restrict__ adj_field_patch,
    const float* __restrict__ coeff_patch,
    const int* __restrict__ sample_indices) {
  __shared__ float warp_sums[32];
  const int sample_linear = static_cast<int>(blockIdx.x);
  const int sample_count = SampleAxis == 0 ? patch_nx : (SampleAxis == 1 ? patch_ny : patch_nz);
  if (sample_linear >= sample_count) {
    return;
  }
  float value = 0.0f;
  if constexpr (SampleAxis == 0) {
    const long long sample_base = static_cast<long long>(sample_linear) * patch_ny * patch_nz;
    for (int j = static_cast<int>(threadIdx.y); j < patch_ny; j += static_cast<int>(blockDim.y)) {
      long long linear = sample_base + static_cast<long long>(j) * patch_nz + threadIdx.x;
      for (int k = static_cast<int>(threadIdx.x); k < patch_nz; k += static_cast<int>(blockDim.x)) {
        value += adj_field_patch[linear] * coeff_patch[linear];
        linear += blockDim.x;
      }
    }
  } else if constexpr (SampleAxis == 1) {
    const long long sample_offset = static_cast<long long>(sample_linear) * patch_nz;
    for (int i = static_cast<int>(threadIdx.y); i < patch_nx; i += static_cast<int>(blockDim.y)) {
      long long linear = static_cast<long long>(i) * patch_ny * patch_nz + sample_offset + threadIdx.x;
      for (int k = static_cast<int>(threadIdx.x); k < patch_nz; k += static_cast<int>(blockDim.x)) {
        value += adj_field_patch[linear] * coeff_patch[linear];
        linear += blockDim.x;
      }
    }
  } else {
    for (int i = static_cast<int>(threadIdx.y); i < patch_nx; i += static_cast<int>(blockDim.y)) {
      for (int j = static_cast<int>(threadIdx.x); j < patch_ny; j += static_cast<int>(blockDim.x)) {
        const long long linear = offset3d(
            static_cast<unsigned int>(i),
            static_cast<unsigned int>(j),
            static_cast<unsigned int>(sample_linear),
            static_cast<unsigned int>(patch_ny),
            static_cast<unsigned int>(patch_nz));
        value += adj_field_patch[linear] * coeff_patch[linear];
      }
    }
  }
  value *= component_scale;
  value = warp_reduce_sum(value);
  const unsigned int thread_linear = threadIdx.y * blockDim.x + threadIdx.x;
  const unsigned int lane = thread_linear & 31u;
  const unsigned int warp = thread_linear >> 5;
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  value = thread_linear < ((blockDim.x * blockDim.y + 31u) >> 5) ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
  }
  if (thread_linear == 0) {
    const int sample_index = sample_indices[sample_linear];
    atomicAdd(adj_aux_field + sample_index, value);
  }
}

template <int SampleAxis>
void launch_tfsf_line_sample_adjoint(
    int patch_nx,
    int patch_ny,
    int patch_nz,
    float component_scale,
    float* __restrict__ adj_aux_field,
    const float* __restrict__ adj_field_patch,
    const float* __restrict__ coeff_patch,
    const int* __restrict__ sample_indices) {
  const dim3 block(8, 32, 1);
  const int sample_count = SampleAxis == 0 ? patch_nx : (SampleAxis == 1 ? patch_ny : patch_nz);
  accumulate_tfsf_line_sample_adjoint_kernel<SampleAxis><<<sample_count, block, 0, current_cuda_stream()>>>(
      patch_nx,
      patch_ny,
      patch_nz,
      component_scale,
      adj_aux_field,
      adj_field_patch,
      coeff_patch,
      sample_indices);
}

__global__ void accumulate_tfsf_interpolated_sample_adjoint_kernel(
    int64_t total,
    int64_t adj_aux_count,
    float origin,
    float inv_ds,
    float component_scale,
    float* __restrict__ adj_aux_field,
    const float* __restrict__ adj_field_patch,
    const float* __restrict__ coeff_patch,
    const float* __restrict__ sample_positions) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total || adj_aux_count <= 0) {
    return;
  }
  const int64_t last_index = adj_aux_count - 1;
  float coord = inv_ds > 0.0f ? (sample_positions[linear] - origin) * inv_ds : 0.0f;
  coord = fminf(fmaxf(coord, 0.0f), static_cast<float>(last_index));
  const int64_t lower = static_cast<int64_t>(coord);
  const int64_t upper = lower + 1 < last_index ? lower + 1 : last_index;
  const float frac = coord - static_cast<float>(lower);
  const float value = component_scale * adj_field_patch[linear] * coeff_patch[linear];
  atomicAdd(adj_aux_field + lower, value * (1.0f - frac));
  if (upper != lower) {
    atomicAdd(adj_aux_field + upper, value * frac);
  }
}

__device__ __forceinline__ void warp_atomic_add_adjacent_repeats(
    float* __restrict__ values,
    int index,
    float value) {
  const unsigned int mask = __activemask();
  const unsigned int lane = threadIdx.x & 31u;
  const int prev_index = __shfl_up_sync(mask, index, 1);
  const int next_index = __shfl_down_sync(mask, index, 1);
  const bool adjacent_repeat =
      ((lane > 0u) && (prev_index == index))
      || ((lane < 31u) && (next_index == index));
  if (!__any_sync(mask, adjacent_repeat)) {
    atomicAdd(values + index, value);
    return;
  }

  const unsigned int peers = __match_any_sync(mask, index);
  if ((peers & (peers - 1u)) == 0u) {
    atomicAdd(values + index, value);
    return;
  }
  if (peers == mask) {
    const float sum = warp_reduce_sum_active(value, mask);
    if (lane == static_cast<unsigned int>(__ffs(peers) - 1)) {
      atomicAdd(values + index, sum);
    }
    return;
  }

  float sum = 0.0f;
  unsigned int remaining = peers;
  while (remaining != 0u) {
    const int source_lane = __ffs(remaining) - 1;
    sum += __shfl_sync(peers, value, source_lane);
    remaining &= remaining - 1u;
  }
  if (static_cast<int>(lane) == (__ffs(peers) - 1)) {
    atomicAdd(values + index, sum);
  }
}

__global__ void accumulate_tfsf_interpolated_sample_adjoint_warp_kernel(
    int64_t total,
    int adj_aux_count,
    float origin,
    float inv_ds,
    float component_scale,
    float* __restrict__ adj_aux_field,
    const float* __restrict__ adj_field_patch,
    const float* __restrict__ coeff_patch,
    const float* __restrict__ sample_positions) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total || adj_aux_count <= 0) {
    return;
  }
  const int last_index = adj_aux_count - 1;
  float coord = inv_ds > 0.0f ? (sample_positions[linear] - origin) * inv_ds : 0.0f;
  coord = fminf(fmaxf(coord, 0.0f), static_cast<float>(last_index));
  const int lower = static_cast<int>(coord);
  const int upper = lower + 1 < last_index ? lower + 1 : last_index;
  const float frac = coord - static_cast<float>(lower);
  const float value = component_scale * adj_field_patch[linear] * coeff_patch[linear];
  warp_atomic_add_adjacent_repeats(adj_aux_field, lower, value * (1.0f - frac));
  if (upper != lower) {
    warp_atomic_add_adjacent_repeats(adj_aux_field, upper, value * frac);
  }
}

__global__ void reverse_tfsf_auxiliary_magnetic_kernel(
    int64_t electric_total,
    int64_t magnetic_total,
    float* __restrict__ adj_electric_prev,
    float* __restrict__ adj_magnetic_prev,
    const float* __restrict__ adj_magnetic_after,
    const float* __restrict__ magnetic_decay,
    const float* __restrict__ magnetic_curl) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= electric_total && index >= magnetic_total) {
    return;
  }
  if (index < magnetic_total) {
    const float adjoint = adj_magnetic_after[index];
    adj_magnetic_prev[index] = magnetic_decay[index] * adjoint;
  }
  if (index < electric_total) {
    float adjoint = 0.0f;
    if (index < magnetic_total) {
      adjoint += magnetic_curl[index] * adj_magnetic_after[index];
    }
    if (index > 0) {
      const int64_t lower = index - 1;
      adjoint -= magnetic_curl[lower] * adj_magnetic_after[lower];
    }
    adj_electric_prev[index] += adjoint;
  }
}

// ---------------------------------------------------------------------------
// Seed injection (transpose of the forward DFT / observer accumulation).
//
// The forward accumulation at step ``t`` writes
//   real_accum[e, cell] += field[cell] * cos_pack[e, t]
//   imag_accum[e, cell] += field[cell] * sin_pack[e, t]
// so its vector-Jacobian product back onto the field is the weighted frequency
// reduction
//   adj_field[cell] += sum_e grad_real[e, cell] * cos_pack[e, t]
//                          + grad_imag[e, cell] * sin_pack[e, t].
// ``cos_pack`` / ``sin_pack`` are the per-entry schedule rows (shape ``(E, T)``)
// already gathered once at seed-build time, so the kernel only reads column
// ``step`` for the active reverse step. The seed contribution *accumulates* into
// the (already cloned) post-step adjoint field, matching the Torch reference.
// ---------------------------------------------------------------------------

__global__ void seed_inject_dense_kernel(
    int nx,
    int ny,
    int nz,
    int entries,
    int steps,
    int step,
    float* __restrict__ adj_field,
    const float* __restrict__ grad_real,
    const float* __restrict__ grad_imag,
    const float* __restrict__ cos_pack,
    const float* __restrict__ sin_pack) {
  const unsigned int k_u = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j_u = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i_u = blockIdx.z * blockDim.z + threadIdx.z;
  if (static_cast<int>(i_u) >= nx || static_cast<int>(j_u) >= ny || static_cast<int>(k_u) >= nz) {
    return;
  }
  const long long cell = offset3d(i_u, j_u, k_u, static_cast<unsigned int>(ny), static_cast<unsigned int>(nz));
  const long long plane = static_cast<long long>(nx) * ny * nz;
  float value = 0.0f;
  for (int e = 0; e < entries; ++e) {
    const float weight_cos = cos_pack[e * steps + step];
    const float weight_sin = sin_pack[e * steps + step];
    const long long field_index = static_cast<long long>(e) * plane + cell;
    value += grad_real[field_index] * weight_cos + grad_imag[field_index] * weight_sin;
  }
  adj_field[cell] += value;
}

template <typename IndexT>
__global__ void seed_inject_point_kernel(
    int point_count,
    int entries,
    int steps,
    int step,
    unsigned int ny,
    unsigned int nz,
    float* __restrict__ adj_field,
    const float* __restrict__ grad_real,
    const float* __restrict__ grad_imag,
    const IndexT* __restrict__ point_i,
    const IndexT* __restrict__ point_j,
    const IndexT* __restrict__ point_k,
    const float* __restrict__ cos_pack,
    const float* __restrict__ sin_pack) {
  const int p = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= point_count) {
    return;
  }
  float value = 0.0f;
  for (int e = 0; e < entries; ++e) {
    const float weight_cos = cos_pack[e * steps + step];
    const float weight_sin = sin_pack[e * steps + step];
    const long long grad_index = static_cast<long long>(e) * point_count + p;
    value += grad_real[grad_index] * weight_cos + grad_imag[grad_index] * weight_sin;
  }
  const long long cell = offset3d(
      static_cast<unsigned int>(point_i[p]),
      static_cast<unsigned int>(point_j[p]),
      static_cast<unsigned int>(point_k[p]),
      ny,
      nz);
  atomicAdd(&adj_field[cell], value);
}

template <int Axis>
__global__ void seed_inject_plane_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    int entries,
    int steps,
    int step,
    int plane_index,
    float* __restrict__ adj_field,
    const float* __restrict__ grad_real,
    const float* __restrict__ grad_imag,
    const float* __restrict__ cos_pack,
    const float* __restrict__ sin_pack) {
  unsigned int i = 0;
  unsigned int j = 0;
  unsigned int k = 0;
  long long plane_linear = 0;
  long long plane_size = 0;
  if constexpr (Axis == 0) {
    k = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= ny || k >= nz) {
      return;
    }
    i = static_cast<unsigned int>(plane_index);
    plane_linear = static_cast<long long>(j) * nz + k;
    plane_size = static_cast<long long>(ny) * nz;
  } else if constexpr (Axis == 1) {
    k = blockIdx.x * blockDim.x + threadIdx.x;
    i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || k >= nz) {
      return;
    }
    j = static_cast<unsigned int>(plane_index);
    plane_linear = static_cast<long long>(i) * nz + k;
    plane_size = static_cast<long long>(nx) * nz;
  } else {
    j = blockIdx.x * blockDim.x + threadIdx.x;
    i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) {
      return;
    }
    k = static_cast<unsigned int>(plane_index);
    plane_linear = static_cast<long long>(i) * ny + j;
    plane_size = static_cast<long long>(nx) * ny;
  }
  float value = 0.0f;
  for (int e = 0; e < entries; ++e) {
    const float weight_cos = cos_pack[e * steps + step];
    const float weight_sin = sin_pack[e * steps + step];
    const long long grad_index = static_cast<long long>(e) * plane_size + plane_linear;
    value += grad_real[grad_index] * weight_cos + grad_imag[grad_index] * weight_sin;
  }
  adj_field[offset3d(i, j, k, ny, nz)] += value;
}

// Fused element-wise accumulate: ``dst[i] += src[i]``. Replaces the per-step
// full-grid ``aten::add`` the adjoint bridge used to fold each step's material
// gradient into the running accumulator.
__global__ void accumulate_in_place_kernel(
    int64_t total,
    float* __restrict__ dst,
    const float* __restrict__ src) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  dst[index] += src[index];
}

}  // namespace

void reverse_magnetic_adjoint_decay_cuda(
    torch::stable::Tensor adj_prev,
    const torch::stable::Tensor& adj_mid,
    const torch::stable::Tensor& decay) {
  check_field(adj_prev, "adj_prev");
  check_matching_field(adj_prev, adj_mid, "adj_mid");
  check_matching_field(adj_prev, decay, "decay");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_prev.get_device_index());
  const int64_t total = adj_prev.numel();
  reverse_magnetic_decay_kernel<<<linear_grid(total, 512), 512, 0, current_cuda_stream()>>>(
      total,
      adj_prev.mutable_data_ptr<float>(),
      adj_mid.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void reverse_electric_adjoint_to_hx_standard_cuda(
    torch::stable::Tensor adj_hx_mid,
    const torch::stable::Tensor& adj_hx_post,
    const torch::stable::Tensor& adj_ey_post,
    const torch::stable::Tensor& adj_ez_post,
    const torch::stable::Tensor& ey_curl,
    const torch::stable::Tensor& ez_curl,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz) {
  check_field(adj_hx_mid, "adj_hx_mid");
  check_matching_field(adj_hx_mid, adj_hx_post, "adj_hx_post");
  check_field(adj_ey_post, "adj_ey_post");
  check_matching_field(adj_ey_post, ey_curl, "ey_curl");
  check_field(adj_ez_post, "adj_ez_post");
  check_matching_field(adj_ez_post, ez_curl, "ez_curl");
  check_spacing_vector(adj_ez_post, inv_dy, 1, "inv_dy");
  check_spacing_vector(adj_ey_post, inv_dz, 2, "inv_dz");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_hx_mid.get_device_index());
  const dim3 block = field_block3d();
  reverse_electric_to_hx_standard_kernel<<<field_grid3d(adj_hx_mid.size(0), adj_hx_mid.size(1), adj_hx_mid.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<int>(adj_hx_mid.size(0)),
      static_cast<int>(adj_hx_mid.size(1)),
      static_cast<int>(adj_hx_mid.size(2)),
      static_cast<int>(adj_ey_post.size(0)),
      static_cast<int>(adj_ey_post.size(1)),
      static_cast<int>(adj_ey_post.size(2)),
      static_cast<int>(adj_ez_post.size(0)),
      static_cast<int>(adj_ez_post.size(1)),
      static_cast<int>(adj_ez_post.size(2)),
      adj_hx_mid.mutable_data_ptr<float>(),
      adj_hx_post.mutable_data_ptr<float>(),
      adj_ey_post.mutable_data_ptr<float>(),
      adj_ez_post.mutable_data_ptr<float>(),
      ey_curl.mutable_data_ptr<float>(),
      ez_curl.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void reverse_electric_adjoint_to_hy_standard_cuda(
    torch::stable::Tensor adj_hy_mid,
    const torch::stable::Tensor& adj_hy_post,
    const torch::stable::Tensor& adj_ex_post,
    const torch::stable::Tensor& adj_ez_post,
    const torch::stable::Tensor& ex_curl,
    const torch::stable::Tensor& ez_curl,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz) {
  check_field(adj_hy_mid, "adj_hy_mid");
  check_matching_field(adj_hy_mid, adj_hy_post, "adj_hy_post");
  check_field(adj_ex_post, "adj_ex_post");
  check_matching_field(adj_ex_post, ex_curl, "ex_curl");
  check_field(adj_ez_post, "adj_ez_post");
  check_matching_field(adj_ez_post, ez_curl, "ez_curl");
  check_spacing_vector(adj_ez_post, inv_dx, 0, "inv_dx");
  check_spacing_vector(adj_ex_post, inv_dz, 2, "inv_dz");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_hy_mid.get_device_index());
  const dim3 block = field_block3d();
  reverse_electric_to_hy_standard_kernel<<<field_grid3d(adj_hy_mid.size(0), adj_hy_mid.size(1), adj_hy_mid.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<int>(adj_hy_mid.size(0)),
      static_cast<int>(adj_hy_mid.size(1)),
      static_cast<int>(adj_hy_mid.size(2)),
      static_cast<int>(adj_ex_post.size(0)),
      static_cast<int>(adj_ex_post.size(1)),
      static_cast<int>(adj_ex_post.size(2)),
      static_cast<int>(adj_ez_post.size(0)),
      static_cast<int>(adj_ez_post.size(1)),
      static_cast<int>(adj_ez_post.size(2)),
      adj_hy_mid.mutable_data_ptr<float>(),
      adj_hy_post.mutable_data_ptr<float>(),
      adj_ex_post.mutable_data_ptr<float>(),
      adj_ez_post.mutable_data_ptr<float>(),
      ex_curl.mutable_data_ptr<float>(),
      ez_curl.mutable_data_ptr<float>(),
      inv_dx.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void reverse_electric_adjoint_to_hz_standard_cuda(
    torch::stable::Tensor adj_hz_mid,
    const torch::stable::Tensor& adj_hz_post,
    const torch::stable::Tensor& adj_ex_post,
    const torch::stable::Tensor& adj_ey_post,
    const torch::stable::Tensor& ex_curl,
    const torch::stable::Tensor& ey_curl,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy) {
  check_field(adj_hz_mid, "adj_hz_mid");
  check_matching_field(adj_hz_mid, adj_hz_post, "adj_hz_post");
  check_field(adj_ex_post, "adj_ex_post");
  check_matching_field(adj_ex_post, ex_curl, "ex_curl");
  check_field(adj_ey_post, "adj_ey_post");
  check_matching_field(adj_ey_post, ey_curl, "ey_curl");
  check_spacing_vector(adj_ey_post, inv_dx, 0, "inv_dx");
  check_spacing_vector(adj_ex_post, inv_dy, 1, "inv_dy");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_hz_mid.get_device_index());
  const dim3 block = field_block3d();
  reverse_electric_to_hz_standard_kernel<<<field_grid3d(adj_hz_mid.size(0), adj_hz_mid.size(1), adj_hz_mid.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<int>(adj_hz_mid.size(0)),
      static_cast<int>(adj_hz_mid.size(1)),
      static_cast<int>(adj_hz_mid.size(2)),
      static_cast<int>(adj_ex_post.size(0)),
      static_cast<int>(adj_ex_post.size(1)),
      static_cast<int>(adj_ex_post.size(2)),
      static_cast<int>(adj_ey_post.size(0)),
      static_cast<int>(adj_ey_post.size(1)),
      static_cast<int>(adj_ey_post.size(2)),
      adj_hz_mid.mutable_data_ptr<float>(),
      adj_hz_post.mutable_data_ptr<float>(),
      adj_ex_post.mutable_data_ptr<float>(),
      adj_ey_post.mutable_data_ptr<float>(),
      ex_curl.mutable_data_ptr<float>(),
      ey_curl.mutable_data_ptr<float>(),
      inv_dx.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void reverse_magnetic_adjoint_to_ex_standard_cuda(
    torch::stable::Tensor adj_ex_prev,
    torch::stable::Tensor grad_eps_ex,
    const torch::stable::Tensor& adj_ex_post,
    const torch::stable::Tensor& adj_hy_mid,
    const torch::stable::Tensor& adj_hz_mid,
    const torch::stable::Tensor& ex_decay,
    const torch::stable::Tensor& ex_curl,
    const torch::stable::Tensor& eps_ex,
    const torch::stable::Tensor& hy_mid,
    const torch::stable::Tensor& hz_mid,
    const torch::stable::Tensor& hy_curl,
    const torch::stable::Tensor& hz_curl,
    const torch::stable::Tensor& inv_dy_e,
    const torch::stable::Tensor& inv_dz_e,
    const torch::stable::Tensor& inv_dy_h,
    const torch::stable::Tensor& inv_dz_h,
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
  check_spacing_vector(adj_ex_prev, inv_dy_e, 1, "inv_dy_e");
  check_spacing_vector(adj_ex_prev, inv_dz_e, 2, "inv_dz_e");
  check_spacing_vector(adj_hz_mid, inv_dy_h, 1, "inv_dy_h");
  check_spacing_vector(adj_hy_mid, inv_dz_h, 2, "inv_dz_h");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_ex_prev.get_device_index());
  const dim3 block = field_block3d();
  reverse_magnetic_to_ex_standard_kernel<<<field_grid3d(adj_ex_prev.size(0), adj_ex_prev.size(1), adj_ex_prev.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<int>(adj_ex_prev.size(0)),
      static_cast<int>(adj_ex_prev.size(1)),
      static_cast<int>(adj_ex_prev.size(2)),
      static_cast<int>(adj_hy_mid.size(1)),
      static_cast<int>(adj_hy_mid.size(2)),
      static_cast<int>(adj_hz_mid.size(1)),
      static_cast<int>(adj_hz_mid.size(2)),
      adj_ex_prev.mutable_data_ptr<float>(),
      grad_eps_ex.mutable_data_ptr<float>(),
      adj_ex_post.mutable_data_ptr<float>(),
      adj_hy_mid.mutable_data_ptr<float>(),
      adj_hz_mid.mutable_data_ptr<float>(),
      ex_decay.mutable_data_ptr<float>(),
      ex_curl.mutable_data_ptr<float>(),
      eps_ex.mutable_data_ptr<float>(),
      hy_mid.mutable_data_ptr<float>(),
      hz_mid.mutable_data_ptr<float>(),
      hy_curl.mutable_data_ptr<float>(),
      hz_curl.mutable_data_ptr<float>(),
      inv_dy_e.mutable_data_ptr<float>(),
      inv_dz_e.mutable_data_ptr<float>(),
      inv_dy_h.mutable_data_ptr<float>(),
      inv_dz_h.mutable_data_ptr<float>(),
      static_cast<int>(y_low_mode),
      static_cast<int>(y_high_mode),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode));
  WITWIN_CUDA_CHECK();
}

void reverse_magnetic_adjoint_to_ey_standard_cuda(
    torch::stable::Tensor adj_ey_prev,
    torch::stable::Tensor grad_eps_ey,
    const torch::stable::Tensor& adj_ey_post,
    const torch::stable::Tensor& adj_hx_mid,
    const torch::stable::Tensor& adj_hz_mid,
    const torch::stable::Tensor& ey_decay,
    const torch::stable::Tensor& ey_curl,
    const torch::stable::Tensor& eps_ey,
    const torch::stable::Tensor& hx_mid,
    const torch::stable::Tensor& hz_mid,
    const torch::stable::Tensor& hx_curl,
    const torch::stable::Tensor& hz_curl,
    const torch::stable::Tensor& inv_dx_e,
    const torch::stable::Tensor& inv_dz_e,
    const torch::stable::Tensor& inv_dx_h,
    const torch::stable::Tensor& inv_dz_h,
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
  check_spacing_vector(adj_ey_prev, inv_dx_e, 0, "inv_dx_e");
  check_spacing_vector(adj_ey_prev, inv_dz_e, 2, "inv_dz_e");
  check_spacing_vector(adj_hz_mid, inv_dx_h, 0, "inv_dx_h");
  check_spacing_vector(adj_hx_mid, inv_dz_h, 2, "inv_dz_h");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_ey_prev.get_device_index());
  const dim3 block = field_block3d();
  reverse_magnetic_to_ey_standard_kernel<<<field_grid3d(adj_ey_prev.size(0), adj_ey_prev.size(1), adj_ey_prev.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<int>(adj_ey_prev.size(0)),
      static_cast<int>(adj_ey_prev.size(1)),
      static_cast<int>(adj_ey_prev.size(2)),
      static_cast<int>(adj_hx_mid.size(1)),
      static_cast<int>(adj_hx_mid.size(2)),
      static_cast<int>(adj_hz_mid.size(0)),
      static_cast<int>(adj_hz_mid.size(1)),
      static_cast<int>(adj_hz_mid.size(2)),
      adj_ey_prev.mutable_data_ptr<float>(),
      grad_eps_ey.mutable_data_ptr<float>(),
      adj_ey_post.mutable_data_ptr<float>(),
      adj_hx_mid.mutable_data_ptr<float>(),
      adj_hz_mid.mutable_data_ptr<float>(),
      ey_decay.mutable_data_ptr<float>(),
      ey_curl.mutable_data_ptr<float>(),
      eps_ey.mutable_data_ptr<float>(),
      hx_mid.mutable_data_ptr<float>(),
      hz_mid.mutable_data_ptr<float>(),
      hx_curl.mutable_data_ptr<float>(),
      hz_curl.mutable_data_ptr<float>(),
      inv_dx_e.mutable_data_ptr<float>(),
      inv_dz_e.mutable_data_ptr<float>(),
      inv_dx_h.mutable_data_ptr<float>(),
      inv_dz_h.mutable_data_ptr<float>(),
      static_cast<int>(x_low_mode),
      static_cast<int>(x_high_mode),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode));
  WITWIN_CUDA_CHECK();
}

void reverse_magnetic_adjoint_to_ez_standard_cuda(
    torch::stable::Tensor adj_ez_prev,
    torch::stable::Tensor grad_eps_ez,
    const torch::stable::Tensor& adj_ez_post,
    const torch::stable::Tensor& adj_hx_mid,
    const torch::stable::Tensor& adj_hy_mid,
    const torch::stable::Tensor& ez_decay,
    const torch::stable::Tensor& ez_curl,
    const torch::stable::Tensor& eps_ez,
    const torch::stable::Tensor& hx_mid,
    const torch::stable::Tensor& hy_mid,
    const torch::stable::Tensor& hx_curl,
    const torch::stable::Tensor& hy_curl,
    const torch::stable::Tensor& inv_dx_e,
    const torch::stable::Tensor& inv_dy_e,
    const torch::stable::Tensor& inv_dx_h,
    const torch::stable::Tensor& inv_dy_h,
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
  check_spacing_vector(adj_ez_prev, inv_dx_e, 0, "inv_dx_e");
  check_spacing_vector(adj_ez_prev, inv_dy_e, 1, "inv_dy_e");
  check_spacing_vector(adj_hy_mid, inv_dx_h, 0, "inv_dx_h");
  check_spacing_vector(adj_hx_mid, inv_dy_h, 1, "inv_dy_h");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_ez_prev.get_device_index());
  const dim3 block = field_block3d();
  reverse_magnetic_to_ez_standard_kernel<<<field_grid3d(adj_ez_prev.size(0), adj_ez_prev.size(1), adj_ez_prev.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<int>(adj_ez_prev.size(0)),
      static_cast<int>(adj_ez_prev.size(1)),
      static_cast<int>(adj_ez_prev.size(2)),
      static_cast<int>(adj_hx_mid.size(1)),
      static_cast<int>(adj_hx_mid.size(2)),
      static_cast<int>(adj_hy_mid.size(0)),
      static_cast<int>(adj_hy_mid.size(1)),
      static_cast<int>(adj_hy_mid.size(2)),
      adj_ez_prev.mutable_data_ptr<float>(),
      grad_eps_ez.mutable_data_ptr<float>(),
      adj_ez_post.mutable_data_ptr<float>(),
      adj_hx_mid.mutable_data_ptr<float>(),
      adj_hy_mid.mutable_data_ptr<float>(),
      ez_decay.mutable_data_ptr<float>(),
      ez_curl.mutable_data_ptr<float>(),
      eps_ez.mutable_data_ptr<float>(),
      hx_mid.mutable_data_ptr<float>(),
      hy_mid.mutable_data_ptr<float>(),
      hx_curl.mutable_data_ptr<float>(),
      hy_curl.mutable_data_ptr<float>(),
      inv_dx_e.mutable_data_ptr<float>(),
      inv_dy_e.mutable_data_ptr<float>(),
      inv_dx_h.mutable_data_ptr<float>(),
      inv_dy_h.mutable_data_ptr<float>(),
      static_cast<int>(x_low_mode),
      static_cast<int>(x_high_mode),
      static_cast<int>(y_low_mode),
      static_cast<int>(y_high_mode));
  WITWIN_CUDA_CHECK();
}

void reverse_electric_adjoint_to_hx_bloch_cuda(
    torch::stable::Tensor adj_hx_mid_real,
    torch::stable::Tensor adj_hx_mid_imag,
    const torch::stable::Tensor& adj_hx_post_real,
    const torch::stable::Tensor& adj_hx_post_imag,
    const torch::stable::Tensor& adj_ey_post_real,
    const torch::stable::Tensor& adj_ey_post_imag,
    const torch::stable::Tensor& adj_ez_post_real,
    const torch::stable::Tensor& adj_ez_post_imag,
    const torch::stable::Tensor& ey_curl,
    const torch::stable::Tensor& ez_curl,
    double phase_cos_y,
    double phase_sin_y,
    double phase_cos_z,
    double phase_sin_z,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz) {
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
  STD_TORCH_CHECK(adj_ey_post_real.size(0) == adj_hx_mid_real.size(0)
      && adj_ey_post_real.size(1) == adj_hx_mid_real.size(1)
      && adj_ey_post_real.size(2) == adj_hx_mid_real.size(2) + 1,
      "Ey adjoint shape must match Hx Bloch stencil");
  STD_TORCH_CHECK(adj_ez_post_real.size(0) == adj_hx_mid_real.size(0)
      && adj_ez_post_real.size(1) == adj_hx_mid_real.size(1) + 1
      && adj_ez_post_real.size(2) == adj_hx_mid_real.size(2),
      "Ez adjoint shape must match Hx Bloch stencil");
  check_spacing_vector(adj_ez_post_real, inv_dy, 1, "inv_dy");
  check_spacing_vector(adj_ey_post_real, inv_dz, 2, "inv_dz");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_hx_mid_real.get_device_index());
  const dim3 block = field_block3d();
  reverse_electric_to_hx_bloch_kernel<<<field_grid3d(adj_hx_mid_real.size(0), adj_hx_mid_real.size(1), adj_hx_mid_real.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<int>(adj_hx_mid_real.size(0)),
      static_cast<int>(adj_hx_mid_real.size(1)),
      static_cast<int>(adj_hx_mid_real.size(2)),
      static_cast<int>(adj_ey_post_real.size(0)),
      static_cast<int>(adj_ey_post_real.size(1)),
      static_cast<int>(adj_ey_post_real.size(2)),
      static_cast<int>(adj_ez_post_real.size(0)),
      static_cast<int>(adj_ez_post_real.size(1)),
      static_cast<int>(adj_ez_post_real.size(2)),
      adj_hx_mid_real.mutable_data_ptr<float>(),
      adj_hx_mid_imag.mutable_data_ptr<float>(),
      adj_hx_post_real.mutable_data_ptr<float>(),
      adj_hx_post_imag.mutable_data_ptr<float>(),
      adj_ey_post_real.mutable_data_ptr<float>(),
      adj_ey_post_imag.mutable_data_ptr<float>(),
      adj_ez_post_real.mutable_data_ptr<float>(),
      adj_ez_post_imag.mutable_data_ptr<float>(),
      ey_curl.mutable_data_ptr<float>(),
      ez_curl.mutable_data_ptr<float>(),
      static_cast<float>(phase_cos_y),
      static_cast<float>(phase_sin_y),
      static_cast<float>(phase_cos_z),
      static_cast<float>(phase_sin_z),
      inv_dy.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void reverse_electric_adjoint_to_hy_bloch_cuda(
    torch::stable::Tensor adj_hy_mid_real,
    torch::stable::Tensor adj_hy_mid_imag,
    const torch::stable::Tensor& adj_hy_post_real,
    const torch::stable::Tensor& adj_hy_post_imag,
    const torch::stable::Tensor& adj_ex_post_real,
    const torch::stable::Tensor& adj_ex_post_imag,
    const torch::stable::Tensor& adj_ez_post_real,
    const torch::stable::Tensor& adj_ez_post_imag,
    const torch::stable::Tensor& ex_curl,
    const torch::stable::Tensor& ez_curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_z,
    double phase_sin_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz) {
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
  STD_TORCH_CHECK(adj_ex_post_real.size(0) == adj_hy_mid_real.size(0)
      && adj_ex_post_real.size(1) == adj_hy_mid_real.size(1)
      && adj_ex_post_real.size(2) == adj_hy_mid_real.size(2) + 1,
      "Ex adjoint shape must match Hy Bloch stencil");
  STD_TORCH_CHECK(adj_ez_post_real.size(0) == adj_hy_mid_real.size(0) + 1
      && adj_ez_post_real.size(1) == adj_hy_mid_real.size(1)
      && adj_ez_post_real.size(2) == adj_hy_mid_real.size(2),
      "Ez adjoint shape must match Hy Bloch stencil");
  check_spacing_vector(adj_ez_post_real, inv_dx, 0, "inv_dx");
  check_spacing_vector(adj_ex_post_real, inv_dz, 2, "inv_dz");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_hy_mid_real.get_device_index());
  const dim3 block = field_block3d();
  reverse_electric_to_hy_bloch_kernel<<<field_grid3d(adj_hy_mid_real.size(0), adj_hy_mid_real.size(1), adj_hy_mid_real.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<int>(adj_hy_mid_real.size(0)),
      static_cast<int>(adj_hy_mid_real.size(1)),
      static_cast<int>(adj_hy_mid_real.size(2)),
      static_cast<int>(adj_ex_post_real.size(0)),
      static_cast<int>(adj_ex_post_real.size(1)),
      static_cast<int>(adj_ex_post_real.size(2)),
      static_cast<int>(adj_ez_post_real.size(0)),
      static_cast<int>(adj_ez_post_real.size(1)),
      static_cast<int>(adj_ez_post_real.size(2)),
      adj_hy_mid_real.mutable_data_ptr<float>(),
      adj_hy_mid_imag.mutable_data_ptr<float>(),
      adj_hy_post_real.mutable_data_ptr<float>(),
      adj_hy_post_imag.mutable_data_ptr<float>(),
      adj_ex_post_real.mutable_data_ptr<float>(),
      adj_ex_post_imag.mutable_data_ptr<float>(),
      adj_ez_post_real.mutable_data_ptr<float>(),
      adj_ez_post_imag.mutable_data_ptr<float>(),
      ex_curl.mutable_data_ptr<float>(),
      ez_curl.mutable_data_ptr<float>(),
      static_cast<float>(phase_cos_x),
      static_cast<float>(phase_sin_x),
      static_cast<float>(phase_cos_z),
      static_cast<float>(phase_sin_z),
      inv_dx.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void reverse_electric_adjoint_to_hz_bloch_cuda(
    torch::stable::Tensor adj_hz_mid_real,
    torch::stable::Tensor adj_hz_mid_imag,
    const torch::stable::Tensor& adj_hz_post_real,
    const torch::stable::Tensor& adj_hz_post_imag,
    const torch::stable::Tensor& adj_ex_post_real,
    const torch::stable::Tensor& adj_ex_post_imag,
    const torch::stable::Tensor& adj_ey_post_real,
    const torch::stable::Tensor& adj_ey_post_imag,
    const torch::stable::Tensor& ex_curl,
    const torch::stable::Tensor& ey_curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_y,
    double phase_sin_y,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy) {
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
  STD_TORCH_CHECK(adj_ex_post_real.size(0) == adj_hz_mid_real.size(0)
      && adj_ex_post_real.size(1) == adj_hz_mid_real.size(1) + 1
      && adj_ex_post_real.size(2) == adj_hz_mid_real.size(2),
      "Ex adjoint shape must match Hz Bloch stencil");
  STD_TORCH_CHECK(adj_ey_post_real.size(0) == adj_hz_mid_real.size(0) + 1
      && adj_ey_post_real.size(1) == adj_hz_mid_real.size(1)
      && adj_ey_post_real.size(2) == adj_hz_mid_real.size(2),
      "Ey adjoint shape must match Hz Bloch stencil");
  check_spacing_vector(adj_ey_post_real, inv_dx, 0, "inv_dx");
  check_spacing_vector(adj_ex_post_real, inv_dy, 1, "inv_dy");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_hz_mid_real.get_device_index());
  const dim3 block = field_block3d();
  reverse_electric_to_hz_bloch_kernel<<<field_grid3d(adj_hz_mid_real.size(0), adj_hz_mid_real.size(1), adj_hz_mid_real.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<int>(adj_hz_mid_real.size(0)),
      static_cast<int>(adj_hz_mid_real.size(1)),
      static_cast<int>(adj_hz_mid_real.size(2)),
      static_cast<int>(adj_ex_post_real.size(0)),
      static_cast<int>(adj_ex_post_real.size(1)),
      static_cast<int>(adj_ex_post_real.size(2)),
      static_cast<int>(adj_ey_post_real.size(0)),
      static_cast<int>(adj_ey_post_real.size(1)),
      static_cast<int>(adj_ey_post_real.size(2)),
      adj_hz_mid_real.mutable_data_ptr<float>(),
      adj_hz_mid_imag.mutable_data_ptr<float>(),
      adj_hz_post_real.mutable_data_ptr<float>(),
      adj_hz_post_imag.mutable_data_ptr<float>(),
      adj_ex_post_real.mutable_data_ptr<float>(),
      adj_ex_post_imag.mutable_data_ptr<float>(),
      adj_ey_post_real.mutable_data_ptr<float>(),
      adj_ey_post_imag.mutable_data_ptr<float>(),
      ex_curl.mutable_data_ptr<float>(),
      ey_curl.mutable_data_ptr<float>(),
      static_cast<float>(phase_cos_x),
      static_cast<float>(phase_sin_x),
      static_cast<float>(phase_cos_y),
      static_cast<float>(phase_sin_y),
      inv_dx.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void reverse_magnetic_adjoint_to_ex_bloch_cuda(
    torch::stable::Tensor adj_ex_prev_real,
    torch::stable::Tensor adj_ex_prev_imag,
    torch::stable::Tensor grad_eps_ex,
    const torch::stable::Tensor& adj_ex_post_real,
    const torch::stable::Tensor& adj_ex_post_imag,
    const torch::stable::Tensor& adj_hy_mid_real,
    const torch::stable::Tensor& adj_hy_mid_imag,
    const torch::stable::Tensor& adj_hz_mid_real,
    const torch::stable::Tensor& adj_hz_mid_imag,
    const torch::stable::Tensor& ex_decay,
    const torch::stable::Tensor& ex_curl,
    const torch::stable::Tensor& eps_ex,
    const torch::stable::Tensor& hy_mid_real,
    const torch::stable::Tensor& hy_mid_imag,
    const torch::stable::Tensor& hz_mid_real,
    const torch::stable::Tensor& hz_mid_imag,
    const torch::stable::Tensor& hy_curl,
    const torch::stable::Tensor& hz_curl,
    double phase_cos_y,
    double phase_sin_y,
    double phase_cos_z,
    double phase_sin_z,
    const torch::stable::Tensor& inv_dy_e,
    const torch::stable::Tensor& inv_dz_e,
    const torch::stable::Tensor& inv_dy_h,
    const torch::stable::Tensor& inv_dz_h) {
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
  check_spacing_vector(adj_ex_prev_real, inv_dy_e, 1, "inv_dy_e");
  check_spacing_vector(adj_ex_prev_real, inv_dz_e, 2, "inv_dz_e");
  check_spacing_vector(hz_mid_real, inv_dy_h, 1, "inv_dy_h");
  check_spacing_vector(hy_mid_real, inv_dz_h, 2, "inv_dz_h");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_ex_prev_real.get_device_index());
  const dim3 block = field_block3d();
  reverse_magnetic_to_ex_bloch_kernel<<<field_grid3d(adj_ex_prev_real.size(0), adj_ex_prev_real.size(1), adj_ex_prev_real.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<int>(adj_ex_prev_real.size(0)),
      static_cast<int>(adj_ex_prev_real.size(1)),
      static_cast<int>(adj_ex_prev_real.size(2)),
      static_cast<int>(hy_mid_real.size(0)),
      static_cast<int>(hy_mid_real.size(1)),
      static_cast<int>(hy_mid_real.size(2)),
      static_cast<int>(hz_mid_real.size(0)),
      static_cast<int>(hz_mid_real.size(1)),
      static_cast<int>(hz_mid_real.size(2)),
      adj_ex_prev_real.mutable_data_ptr<float>(),
      adj_ex_prev_imag.mutable_data_ptr<float>(),
      grad_eps_ex.mutable_data_ptr<float>(),
      adj_ex_post_real.mutable_data_ptr<float>(),
      adj_ex_post_imag.mutable_data_ptr<float>(),
      adj_hy_mid_real.mutable_data_ptr<float>(),
      adj_hy_mid_imag.mutable_data_ptr<float>(),
      adj_hz_mid_real.mutable_data_ptr<float>(),
      adj_hz_mid_imag.mutable_data_ptr<float>(),
      ex_decay.mutable_data_ptr<float>(),
      ex_curl.mutable_data_ptr<float>(),
      eps_ex.mutable_data_ptr<float>(),
      hy_mid_real.mutable_data_ptr<float>(),
      hy_mid_imag.mutable_data_ptr<float>(),
      hz_mid_real.mutable_data_ptr<float>(),
      hz_mid_imag.mutable_data_ptr<float>(),
      hy_curl.mutable_data_ptr<float>(),
      hz_curl.mutable_data_ptr<float>(),
      static_cast<float>(phase_cos_y),
      static_cast<float>(phase_sin_y),
      static_cast<float>(phase_cos_z),
      static_cast<float>(phase_sin_z),
      inv_dy_e.mutable_data_ptr<float>(),
      inv_dz_e.mutable_data_ptr<float>(),
      inv_dy_h.mutable_data_ptr<float>(),
      inv_dz_h.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void reverse_magnetic_adjoint_to_ey_bloch_cuda(
    torch::stable::Tensor adj_ey_prev_real,
    torch::stable::Tensor adj_ey_prev_imag,
    torch::stable::Tensor grad_eps_ey,
    const torch::stable::Tensor& adj_ey_post_real,
    const torch::stable::Tensor& adj_ey_post_imag,
    const torch::stable::Tensor& adj_hx_mid_real,
    const torch::stable::Tensor& adj_hx_mid_imag,
    const torch::stable::Tensor& adj_hz_mid_real,
    const torch::stable::Tensor& adj_hz_mid_imag,
    const torch::stable::Tensor& ey_decay,
    const torch::stable::Tensor& ey_curl,
    const torch::stable::Tensor& eps_ey,
    const torch::stable::Tensor& hx_mid_real,
    const torch::stable::Tensor& hx_mid_imag,
    const torch::stable::Tensor& hz_mid_real,
    const torch::stable::Tensor& hz_mid_imag,
    const torch::stable::Tensor& hx_curl,
    const torch::stable::Tensor& hz_curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_z,
    double phase_sin_z,
    const torch::stable::Tensor& inv_dx_e,
    const torch::stable::Tensor& inv_dz_e,
    const torch::stable::Tensor& inv_dx_h,
    const torch::stable::Tensor& inv_dz_h) {
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
  check_spacing_vector(adj_ey_prev_real, inv_dx_e, 0, "inv_dx_e");
  check_spacing_vector(adj_ey_prev_real, inv_dz_e, 2, "inv_dz_e");
  check_spacing_vector(hz_mid_real, inv_dx_h, 0, "inv_dx_h");
  check_spacing_vector(hx_mid_real, inv_dz_h, 2, "inv_dz_h");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_ey_prev_real.get_device_index());
  const dim3 block = field_block3d();
  reverse_magnetic_to_ey_bloch_kernel<<<field_grid3d(adj_ey_prev_real.size(0), adj_ey_prev_real.size(1), adj_ey_prev_real.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<int>(adj_ey_prev_real.size(0)),
      static_cast<int>(adj_ey_prev_real.size(1)),
      static_cast<int>(adj_ey_prev_real.size(2)),
      static_cast<int>(hx_mid_real.size(0)),
      static_cast<int>(hx_mid_real.size(1)),
      static_cast<int>(hx_mid_real.size(2)),
      static_cast<int>(hz_mid_real.size(0)),
      static_cast<int>(hz_mid_real.size(1)),
      static_cast<int>(hz_mid_real.size(2)),
      adj_ey_prev_real.mutable_data_ptr<float>(),
      adj_ey_prev_imag.mutable_data_ptr<float>(),
      grad_eps_ey.mutable_data_ptr<float>(),
      adj_ey_post_real.mutable_data_ptr<float>(),
      adj_ey_post_imag.mutable_data_ptr<float>(),
      adj_hx_mid_real.mutable_data_ptr<float>(),
      adj_hx_mid_imag.mutable_data_ptr<float>(),
      adj_hz_mid_real.mutable_data_ptr<float>(),
      adj_hz_mid_imag.mutable_data_ptr<float>(),
      ey_decay.mutable_data_ptr<float>(),
      ey_curl.mutable_data_ptr<float>(),
      eps_ey.mutable_data_ptr<float>(),
      hx_mid_real.mutable_data_ptr<float>(),
      hx_mid_imag.mutable_data_ptr<float>(),
      hz_mid_real.mutable_data_ptr<float>(),
      hz_mid_imag.mutable_data_ptr<float>(),
      hx_curl.mutable_data_ptr<float>(),
      hz_curl.mutable_data_ptr<float>(),
      static_cast<float>(phase_cos_x),
      static_cast<float>(phase_sin_x),
      static_cast<float>(phase_cos_z),
      static_cast<float>(phase_sin_z),
      inv_dx_e.mutable_data_ptr<float>(),
      inv_dz_e.mutable_data_ptr<float>(),
      inv_dx_h.mutable_data_ptr<float>(),
      inv_dz_h.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void reverse_magnetic_adjoint_to_ez_bloch_cuda(
    torch::stable::Tensor adj_ez_prev_real,
    torch::stable::Tensor adj_ez_prev_imag,
    torch::stable::Tensor grad_eps_ez,
    const torch::stable::Tensor& adj_ez_post_real,
    const torch::stable::Tensor& adj_ez_post_imag,
    const torch::stable::Tensor& adj_hx_mid_real,
    const torch::stable::Tensor& adj_hx_mid_imag,
    const torch::stable::Tensor& adj_hy_mid_real,
    const torch::stable::Tensor& adj_hy_mid_imag,
    const torch::stable::Tensor& ez_decay,
    const torch::stable::Tensor& ez_curl,
    const torch::stable::Tensor& eps_ez,
    const torch::stable::Tensor& hx_mid_real,
    const torch::stable::Tensor& hx_mid_imag,
    const torch::stable::Tensor& hy_mid_real,
    const torch::stable::Tensor& hy_mid_imag,
    const torch::stable::Tensor& hx_curl,
    const torch::stable::Tensor& hy_curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_y,
    double phase_sin_y,
    const torch::stable::Tensor& inv_dx_e,
    const torch::stable::Tensor& inv_dy_e,
    const torch::stable::Tensor& inv_dx_h,
    const torch::stable::Tensor& inv_dy_h) {
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
  check_spacing_vector(adj_ez_prev_real, inv_dx_e, 0, "inv_dx_e");
  check_spacing_vector(adj_ez_prev_real, inv_dy_e, 1, "inv_dy_e");
  check_spacing_vector(hy_mid_real, inv_dx_h, 0, "inv_dx_h");
  check_spacing_vector(hx_mid_real, inv_dy_h, 1, "inv_dy_h");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_ez_prev_real.get_device_index());
  const dim3 block = field_block3d();
  reverse_magnetic_to_ez_bloch_kernel<<<field_grid3d(adj_ez_prev_real.size(0), adj_ez_prev_real.size(1), adj_ez_prev_real.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<int>(adj_ez_prev_real.size(0)),
      static_cast<int>(adj_ez_prev_real.size(1)),
      static_cast<int>(adj_ez_prev_real.size(2)),
      static_cast<int>(hx_mid_real.size(0)),
      static_cast<int>(hx_mid_real.size(1)),
      static_cast<int>(hx_mid_real.size(2)),
      static_cast<int>(hy_mid_real.size(0)),
      static_cast<int>(hy_mid_real.size(1)),
      static_cast<int>(hy_mid_real.size(2)),
      adj_ez_prev_real.mutable_data_ptr<float>(),
      adj_ez_prev_imag.mutable_data_ptr<float>(),
      grad_eps_ez.mutable_data_ptr<float>(),
      adj_ez_post_real.mutable_data_ptr<float>(),
      adj_ez_post_imag.mutable_data_ptr<float>(),
      adj_hx_mid_real.mutable_data_ptr<float>(),
      adj_hx_mid_imag.mutable_data_ptr<float>(),
      adj_hy_mid_real.mutable_data_ptr<float>(),
      adj_hy_mid_imag.mutable_data_ptr<float>(),
      ez_decay.mutable_data_ptr<float>(),
      ez_curl.mutable_data_ptr<float>(),
      eps_ez.mutable_data_ptr<float>(),
      hx_mid_real.mutable_data_ptr<float>(),
      hx_mid_imag.mutable_data_ptr<float>(),
      hy_mid_real.mutable_data_ptr<float>(),
      hy_mid_imag.mutable_data_ptr<float>(),
      hx_curl.mutable_data_ptr<float>(),
      hy_curl.mutable_data_ptr<float>(),
      static_cast<float>(phase_cos_x),
      static_cast<float>(phase_sin_x),
      static_cast<float>(phase_cos_y),
      static_cast<float>(phase_sin_y),
      inv_dx_e.mutable_data_ptr<float>(),
      inv_dy_e.mutable_data_ptr<float>(),
      inv_dx_h.mutable_data_ptr<float>(),
      inv_dy_h.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void accumulate_forward_diff_adjoint_cuda(
    torch::stable::Tensor field_grad,
    const torch::stable::Tensor& diff_grad,
    int64_t axis,
    const torch::stable::Tensor& inv_delta) {
  check_field(field_grad, "field_grad");
  check_field(diff_grad, "diff_grad");
  STD_TORCH_CHECK(axis >= 0 && axis < 3, "axis must be in [0, 3)");
  check_spacing_vector(diff_grad, inv_delta, axis, "inv_delta");
  const torch::stable::accelerator::DeviceGuard device_guard(field_grad.get_device_index());
  if (axis == 0) {
    launch_accumulate_diff_adjoint<0, true>(field_grad, diff_grad, inv_delta);
  } else if (axis == 1) {
    launch_accumulate_diff_adjoint<1, true>(field_grad, diff_grad, inv_delta);
  } else {
    launch_accumulate_diff_adjoint<2, true>(field_grad, diff_grad, inv_delta);
  }
  WITWIN_CUDA_CHECK();
}

void accumulate_backward_diff_adjoint_cuda(
    torch::stable::Tensor field_grad,
    const torch::stable::Tensor& diff_grad,
    int64_t axis,
    const torch::stable::Tensor& inv_delta) {
  check_field(field_grad, "field_grad");
  check_field(diff_grad, "diff_grad");
  STD_TORCH_CHECK(axis >= 0 && axis < 3, "axis must be in [0, 3)");
  check_spacing_vector(diff_grad, inv_delta, axis, "inv_delta");
  const torch::stable::accelerator::DeviceGuard device_guard(field_grad.get_device_index());
  if (axis == 0) {
    launch_accumulate_diff_adjoint<0, false>(field_grad, diff_grad, inv_delta);
  } else if (axis == 1) {
    launch_accumulate_diff_adjoint<1, false>(field_grad, diff_grad, inv_delta);
  } else {
    launch_accumulate_diff_adjoint<2, false>(field_grad, diff_grad, inv_delta);
  }
  WITWIN_CUDA_CHECK();
}

void launch_reverse_electric_component_cpml(
    int component,
    torch::stable::Tensor adj_prev,
    torch::stable::Tensor grad_eps,
    torch::stable::Tensor adj_psi_pos_prev,
    torch::stable::Tensor adj_psi_neg_prev,
    torch::stable::Tensor adj_d_pos,
    torch::stable::Tensor adj_d_neg,
    const torch::stable::Tensor& adj_post,
    const torch::stable::Tensor& adj_psi_pos_post,
    const torch::stable::Tensor& adj_psi_neg_post,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& eps,
    const torch::stable::Tensor& psi_pos,
    const torch::stable::Tensor& psi_neg,
    const torch::stable::Tensor& b_pos,
    const torch::stable::Tensor& c_pos,
    const torch::stable::Tensor& inv_kappa_pos,
    const torch::stable::Tensor& b_neg,
    const torch::stable::Tensor& c_neg,
    const torch::stable::Tensor& inv_kappa_neg,
    const torch::stable::Tensor& h_pos_mid,
    const torch::stable::Tensor& h_neg_mid,
    const torch::stable::Tensor& inv_pos,
    const torch::stable::Tensor& inv_neg,
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
  check_matching_vector(b_pos, inv_pos, "inv_pos");
  check_matching_vector(b_neg, inv_neg, "inv_neg");
  check_field(h_pos_mid, "h_pos_mid");
  check_field(h_neg_mid, "h_neg_mid");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_prev.get_device_index());
  const dim3 block = field_block3d();
  const auto launch = [&](auto component_tag) {
    constexpr int component_value = decltype(component_tag)::value;
    reverse_electric_component_cpml_kernel<component_value><<<field_grid3d(adj_prev.size(0), adj_prev.size(1), adj_prev.size(2), block), block, 0, current_cuda_stream()>>>(
        static_cast<int>(adj_prev.size(0)),
        static_cast<int>(adj_prev.size(1)),
        static_cast<int>(adj_prev.size(2)),
        static_cast<int>(h_pos_mid.size(1)),
        static_cast<int>(h_pos_mid.size(2)),
        static_cast<int>(h_neg_mid.size(1)),
        static_cast<int>(h_neg_mid.size(2)),
        adj_prev.mutable_data_ptr<float>(),
        grad_eps.mutable_data_ptr<float>(),
        adj_psi_pos_prev.mutable_data_ptr<float>(),
        adj_psi_neg_prev.mutable_data_ptr<float>(),
        adj_d_pos.mutable_data_ptr<float>(),
        adj_d_neg.mutable_data_ptr<float>(),
        adj_post.mutable_data_ptr<float>(),
        adj_psi_pos_post.mutable_data_ptr<float>(),
        adj_psi_neg_post.mutable_data_ptr<float>(),
        decay.mutable_data_ptr<float>(),
        curl.mutable_data_ptr<float>(),
        eps.mutable_data_ptr<float>(),
        psi_pos.mutable_data_ptr<float>(),
        psi_neg.mutable_data_ptr<float>(),
        b_pos.mutable_data_ptr<float>(),
        c_pos.mutable_data_ptr<float>(),
        inv_kappa_pos.mutable_data_ptr<float>(),
        b_neg.mutable_data_ptr<float>(),
        c_neg.mutable_data_ptr<float>(),
        inv_kappa_neg.mutable_data_ptr<float>(),
        h_pos_mid.mutable_data_ptr<float>(),
        h_neg_mid.mutable_data_ptr<float>(),
        inv_pos.mutable_data_ptr<float>(),
        inv_neg.mutable_data_ptr<float>(),
        static_cast<int>(low_mode_a),
        static_cast<int>(high_mode_a),
        static_cast<int>(low_mode_b),
        static_cast<int>(high_mode_b));
  };
  if (component == 0) {
    launch(std::integral_constant<int, 0>{});
  } else if (component == 1) {
    launch(std::integral_constant<int, 1>{});
  } else {
    launch(std::integral_constant<int, 2>{});
  }
  WITWIN_CUDA_CHECK();
}

void reverse_electric_component_ex_cpml_cuda(
    torch::stable::Tensor adj_prev,
    torch::stable::Tensor grad_eps,
    torch::stable::Tensor adj_psi_pos_prev,
    torch::stable::Tensor adj_psi_neg_prev,
    torch::stable::Tensor adj_d_pos,
    torch::stable::Tensor adj_d_neg,
    const torch::stable::Tensor& adj_post,
    const torch::stable::Tensor& adj_psi_pos_post,
    const torch::stable::Tensor& adj_psi_neg_post,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& eps,
    const torch::stable::Tensor& psi_pos,
    const torch::stable::Tensor& psi_neg,
    const torch::stable::Tensor& b_pos,
    const torch::stable::Tensor& c_pos,
    const torch::stable::Tensor& inv_kappa_pos,
    const torch::stable::Tensor& b_neg,
    const torch::stable::Tensor& c_neg,
    const torch::stable::Tensor& inv_kappa_neg,
    const torch::stable::Tensor& hy_mid,
    const torch::stable::Tensor& hz_mid,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
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
    torch::stable::Tensor adj_prev,
    torch::stable::Tensor grad_eps,
    torch::stable::Tensor adj_psi_pos_prev,
    torch::stable::Tensor adj_psi_neg_prev,
    torch::stable::Tensor adj_d_pos,
    torch::stable::Tensor adj_d_neg,
    const torch::stable::Tensor& adj_post,
    const torch::stable::Tensor& adj_psi_pos_post,
    const torch::stable::Tensor& adj_psi_neg_post,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& eps,
    const torch::stable::Tensor& psi_pos,
    const torch::stable::Tensor& psi_neg,
    const torch::stable::Tensor& b_pos,
    const torch::stable::Tensor& c_pos,
    const torch::stable::Tensor& inv_kappa_pos,
    const torch::stable::Tensor& b_neg,
    const torch::stable::Tensor& c_neg,
    const torch::stable::Tensor& inv_kappa_neg,
    const torch::stable::Tensor& hx_mid,
    const torch::stable::Tensor& hz_mid,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
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
    torch::stable::Tensor adj_prev,
    torch::stable::Tensor grad_eps,
    torch::stable::Tensor adj_psi_pos_prev,
    torch::stable::Tensor adj_psi_neg_prev,
    torch::stable::Tensor adj_d_pos,
    torch::stable::Tensor adj_d_neg,
    const torch::stable::Tensor& adj_post,
    const torch::stable::Tensor& adj_psi_pos_post,
    const torch::stable::Tensor& adj_psi_neg_post,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& eps,
    const torch::stable::Tensor& psi_pos,
    const torch::stable::Tensor& psi_neg,
    const torch::stable::Tensor& b_pos,
    const torch::stable::Tensor& c_pos,
    const torch::stable::Tensor& inv_kappa_pos,
    const torch::stable::Tensor& b_neg,
    const torch::stable::Tensor& c_neg,
    const torch::stable::Tensor& inv_kappa_neg,
    const torch::stable::Tensor& hx_mid,
    const torch::stable::Tensor& hy_mid,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
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
    torch::stable::Tensor adj_prev,
    torch::stable::Tensor adj_psi_pos_prev,
    torch::stable::Tensor adj_psi_neg_prev,
    torch::stable::Tensor adj_d_pos,
    torch::stable::Tensor adj_d_neg,
    const torch::stable::Tensor& adj_post,
    const torch::stable::Tensor& adj_psi_pos_post,
    const torch::stable::Tensor& adj_psi_neg_post,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& b_pos,
    const torch::stable::Tensor& c_pos,
    const torch::stable::Tensor& inv_kappa_pos,
    const torch::stable::Tensor& b_neg,
    const torch::stable::Tensor& c_neg,
    const torch::stable::Tensor& inv_kappa_neg) {
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
  const torch::stable::accelerator::DeviceGuard device_guard(adj_prev.get_device_index());
  const dim3 block = field_block3d();
  const auto launch = [&](auto component_tag) {
    constexpr int component_value = decltype(component_tag)::value;
    reverse_magnetic_component_cpml_kernel<component_value><<<field_grid3d(adj_prev.size(0), adj_prev.size(1), adj_prev.size(2), block), block, 0, current_cuda_stream()>>>(
        static_cast<int>(adj_prev.size(0)),
        static_cast<int>(adj_prev.size(1)),
        static_cast<int>(adj_prev.size(2)),
        adj_prev.mutable_data_ptr<float>(),
        adj_psi_pos_prev.mutable_data_ptr<float>(),
        adj_psi_neg_prev.mutable_data_ptr<float>(),
        adj_d_pos.mutable_data_ptr<float>(),
        adj_d_neg.mutable_data_ptr<float>(),
        adj_post.mutable_data_ptr<float>(),
        adj_psi_pos_post.mutable_data_ptr<float>(),
        adj_psi_neg_post.mutable_data_ptr<float>(),
        decay.mutable_data_ptr<float>(),
        curl.mutable_data_ptr<float>(),
        b_pos.mutable_data_ptr<float>(),
        c_pos.mutable_data_ptr<float>(),
        inv_kappa_pos.mutable_data_ptr<float>(),
        b_neg.mutable_data_ptr<float>(),
        c_neg.mutable_data_ptr<float>(),
        inv_kappa_neg.mutable_data_ptr<float>());
  };
  if (component == 0) {
    launch(std::integral_constant<int, 0>{});
  } else if (component == 1) {
    launch(std::integral_constant<int, 1>{});
  } else {
    launch(std::integral_constant<int, 2>{});
  }
  WITWIN_CUDA_CHECK();
}

void reverse_magnetic_component_hx_cpml_cuda(
    torch::stable::Tensor adj_prev,
    torch::stable::Tensor adj_psi_pos_prev,
    torch::stable::Tensor adj_psi_neg_prev,
    torch::stable::Tensor adj_d_pos,
    torch::stable::Tensor adj_d_neg,
    const torch::stable::Tensor& adj_post,
    const torch::stable::Tensor& adj_psi_pos_post,
    const torch::stable::Tensor& adj_psi_neg_post,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& b_pos,
    const torch::stable::Tensor& c_pos,
    const torch::stable::Tensor& inv_kappa_pos,
    const torch::stable::Tensor& b_neg,
    const torch::stable::Tensor& c_neg,
    const torch::stable::Tensor& inv_kappa_neg) {
  launch_reverse_magnetic_component_cpml(
      0, adj_prev, adj_psi_pos_prev, adj_psi_neg_prev, adj_d_pos, adj_d_neg,
      adj_post, adj_psi_pos_post, adj_psi_neg_post, decay, curl,
      b_pos, c_pos, inv_kappa_pos, b_neg, c_neg, inv_kappa_neg);
}

void reverse_magnetic_component_hy_cpml_cuda(
    torch::stable::Tensor adj_prev,
    torch::stable::Tensor adj_psi_pos_prev,
    torch::stable::Tensor adj_psi_neg_prev,
    torch::stable::Tensor adj_d_pos,
    torch::stable::Tensor adj_d_neg,
    const torch::stable::Tensor& adj_post,
    const torch::stable::Tensor& adj_psi_pos_post,
    const torch::stable::Tensor& adj_psi_neg_post,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& b_pos,
    const torch::stable::Tensor& c_pos,
    const torch::stable::Tensor& inv_kappa_pos,
    const torch::stable::Tensor& b_neg,
    const torch::stable::Tensor& c_neg,
    const torch::stable::Tensor& inv_kappa_neg) {
  launch_reverse_magnetic_component_cpml(
      1, adj_prev, adj_psi_pos_prev, adj_psi_neg_prev, adj_d_pos, adj_d_neg,
      adj_post, adj_psi_pos_post, adj_psi_neg_post, decay, curl,
      b_pos, c_pos, inv_kappa_pos, b_neg, c_neg, inv_kappa_neg);
}

void reverse_magnetic_component_hz_cpml_cuda(
    torch::stable::Tensor adj_prev,
    torch::stable::Tensor adj_psi_pos_prev,
    torch::stable::Tensor adj_psi_neg_prev,
    torch::stable::Tensor adj_d_pos,
    torch::stable::Tensor adj_d_neg,
    const torch::stable::Tensor& adj_post,
    const torch::stable::Tensor& adj_psi_pos_post,
    const torch::stable::Tensor& adj_psi_neg_post,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& b_pos,
    const torch::stable::Tensor& c_pos,
    const torch::stable::Tensor& inv_kappa_pos,
    const torch::stable::Tensor& b_neg,
    const torch::stable::Tensor& c_neg,
    const torch::stable::Tensor& inv_kappa_neg) {
  launch_reverse_magnetic_component_cpml(
      2, adj_prev, adj_psi_pos_prev, adj_psi_neg_prev, adj_d_pos, adj_d_neg,
      adj_post, adj_psi_pos_post, adj_psi_neg_post, decay, curl,
      b_pos, c_pos, inv_kappa_pos, b_neg, c_neg, inv_kappa_neg);
}

void reverse_debye_current_cuda(
    torch::stable::Tensor adj_electric_prev,
    torch::stable::Tensor adj_polarization_prev,
    const torch::stable::Tensor& adj_polarization_post,
    const torch::stable::Tensor& adj_current_post,
    const torch::stable::Tensor& drive,
    double decay,
    double dt) {
  check_field(adj_electric_prev, "adj_electric_prev");
  check_matching_field(adj_electric_prev, adj_polarization_prev, "adj_polarization_prev");
  check_matching_field(adj_electric_prev, adj_polarization_post, "adj_polarization_post");
  check_matching_field(adj_electric_prev, adj_current_post, "adj_current_post");
  check_matching_field(adj_electric_prev, drive, "drive");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_electric_prev.get_device_index());
  const int64_t total = adj_electric_prev.numel();
  reverse_debye_current_kernel<<<linear_grid(total, 256), 256, 0, current_cuda_stream()>>>(
      total,
      adj_electric_prev.mutable_data_ptr<float>(),
      adj_polarization_prev.mutable_data_ptr<float>(),
      adj_polarization_post.mutable_data_ptr<float>(),
      adj_current_post.mutable_data_ptr<float>(),
      drive.mutable_data_ptr<float>(),
      static_cast<float>(decay),
      static_cast<float>(1.0 / dt));
  WITWIN_CUDA_CHECK();
}

void reverse_drude_current_cuda(
    torch::stable::Tensor adj_electric_prev,
    torch::stable::Tensor adj_current_prev,
    const torch::stable::Tensor& adj_current_post,
    const torch::stable::Tensor& drive,
    double decay) {
  check_field(adj_electric_prev, "adj_electric_prev");
  check_matching_field(adj_electric_prev, adj_current_prev, "adj_current_prev");
  check_matching_field(adj_electric_prev, adj_current_post, "adj_current_post");
  check_matching_field(adj_electric_prev, drive, "drive");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_electric_prev.get_device_index());
  const int64_t total = adj_electric_prev.numel();
  reverse_drude_current_kernel<<<linear_grid(total, 256), 256, 0, current_cuda_stream()>>>(
      total,
      adj_electric_prev.mutable_data_ptr<float>(),
      adj_current_prev.mutable_data_ptr<float>(),
      adj_current_post.mutable_data_ptr<float>(),
      drive.mutable_data_ptr<float>(),
      static_cast<float>(decay));
  WITWIN_CUDA_CHECK();
}

void reverse_lorentz_current_cuda(
    torch::stable::Tensor adj_electric_prev,
    torch::stable::Tensor adj_polarization_prev,
    torch::stable::Tensor adj_current_prev,
    const torch::stable::Tensor& adj_polarization_post,
    const torch::stable::Tensor& adj_current_post,
    const torch::stable::Tensor& drive,
    double decay,
    double restoring,
    double dt) {
  check_field(adj_electric_prev, "adj_electric_prev");
  check_matching_field(adj_electric_prev, adj_polarization_prev, "adj_polarization_prev");
  check_matching_field(adj_electric_prev, adj_current_prev, "adj_current_prev");
  check_matching_field(adj_electric_prev, adj_polarization_post, "adj_polarization_post");
  check_matching_field(adj_electric_prev, adj_current_post, "adj_current_post");
  check_matching_field(adj_electric_prev, drive, "drive");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_electric_prev.get_device_index());
  const int64_t total = adj_electric_prev.numel();
  reverse_lorentz_current_kernel<<<linear_grid(total, 256), 256, 0, current_cuda_stream()>>>(
      total,
      adj_electric_prev.mutable_data_ptr<float>(),
      adj_polarization_prev.mutable_data_ptr<float>(),
      adj_current_prev.mutable_data_ptr<float>(),
      adj_polarization_post.mutable_data_ptr<float>(),
      adj_current_post.mutable_data_ptr<float>(),
      drive.mutable_data_ptr<float>(),
      static_cast<float>(decay),
      static_cast<float>(restoring),
      static_cast<float>(dt));
  WITWIN_CUDA_CHECK();
}

void reverse_dispersive_correction_cuda(
    torch::stable::Tensor adj_current_corrected,
    torch::stable::Tensor grad_eps,
    const torch::stable::Tensor& adj_current_post,
    const torch::stable::Tensor& adj_electric_post,
    const torch::stable::Tensor& current,
    const torch::stable::Tensor& eps,
    double dt) {
  check_field(adj_current_corrected, "adj_current_corrected");
  check_matching_field(adj_current_corrected, grad_eps, "grad_eps");
  check_matching_field(adj_current_corrected, adj_current_post, "adj_current_post");
  check_matching_field(adj_current_corrected, adj_electric_post, "adj_electric_post");
  check_matching_field(adj_current_corrected, current, "current");
  check_matching_field(adj_current_corrected, eps, "eps");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_current_corrected.get_device_index());
  const int64_t total = adj_current_corrected.numel();
  reverse_dispersive_correction_kernel<<<linear_grid(total, 256), 256, 0, current_cuda_stream()>>>(
      total,
      adj_current_corrected.mutable_data_ptr<float>(),
      grad_eps.mutable_data_ptr<float>(),
      adj_current_post.mutable_data_ptr<float>(),
      adj_electric_post.mutable_data_ptr<float>(),
      current.mutable_data_ptr<float>(),
      eps.mutable_data_ptr<float>(),
      static_cast<float>(dt));
  WITWIN_CUDA_CHECK();
}

void accumulate_tfsf_scalar_sample_adjoint_cuda(
    torch::stable::Tensor adj_aux_field,
    const torch::stable::Tensor& adj_field_patch,
    const torch::stable::Tensor& coeff_patch,
    int64_t sample_index,
    double component_scale) {
  check_vector(adj_aux_field, "adj_aux_field");
  check_field(adj_field_patch, "adj_field_patch");
  check_same_cuda_device(adj_aux_field, adj_field_patch, "adj_field_patch");
  check_matching_field(adj_field_patch, coeff_patch, "coeff_patch");
  STD_TORCH_CHECK(sample_index >= 0 && sample_index < adj_aux_field.numel(), "sample_index is out of range");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_aux_field.get_device_index());
  const int64_t total = adj_field_patch.numel();
  constexpr int block_size = 512;
  accumulate_tfsf_scalar_sample_adjoint_kernel<<<
      linear_grid(total, block_size),
      block_size,
      0,
      current_cuda_stream()>>>(
      total,
      static_cast<int>(adj_field_patch.size(1)),
      static_cast<int>(adj_field_patch.size(2)),
      sample_index,
      static_cast<float>(component_scale),
      adj_aux_field.mutable_data_ptr<float>(),
      adj_field_patch.mutable_data_ptr<float>(),
      coeff_patch.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void accumulate_tfsf_line_sample_adjoint_cuda(
    torch::stable::Tensor adj_aux_field,
    const torch::stable::Tensor& adj_field_patch,
    const torch::stable::Tensor& coeff_patch,
    const torch::stable::Tensor& sample_indices,
    int64_t sample_axis_code,
    double component_scale) {
  check_vector(adj_aux_field, "adj_aux_field");
  check_field(adj_field_patch, "adj_field_patch");
  check_same_cuda_device(adj_aux_field, adj_field_patch, "adj_field_patch");
  check_matching_field(adj_field_patch, coeff_patch, "coeff_patch");
  check_int32_vector(sample_indices, "sample_indices");
  check_same_cuda_device(adj_aux_field, sample_indices, "sample_indices");
  STD_TORCH_CHECK(sample_axis_code >= 0 && sample_axis_code < 3, "sample_axis_code must be in [0, 3)");
  STD_TORCH_CHECK(sample_indices.numel() == adj_field_patch.size(sample_axis_code), "sample_indices length must match the selected patch axis");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_aux_field.get_device_index());
  if (sample_axis_code == 0) {
    launch_tfsf_line_sample_adjoint<0>(
        static_cast<int>(adj_field_patch.size(0)),
        static_cast<int>(adj_field_patch.size(1)),
        static_cast<int>(adj_field_patch.size(2)),
        static_cast<float>(component_scale),
        adj_aux_field.mutable_data_ptr<float>(),
        adj_field_patch.mutable_data_ptr<float>(),
        coeff_patch.mutable_data_ptr<float>(),
        sample_indices.mutable_data_ptr<int>());
  } else if (sample_axis_code == 1) {
    launch_tfsf_line_sample_adjoint<1>(
        static_cast<int>(adj_field_patch.size(0)),
        static_cast<int>(adj_field_patch.size(1)),
        static_cast<int>(adj_field_patch.size(2)),
        static_cast<float>(component_scale),
        adj_aux_field.mutable_data_ptr<float>(),
        adj_field_patch.mutable_data_ptr<float>(),
        coeff_patch.mutable_data_ptr<float>(),
        sample_indices.mutable_data_ptr<int>());
  } else {
    launch_tfsf_line_sample_adjoint<2>(
        static_cast<int>(adj_field_patch.size(0)),
        static_cast<int>(adj_field_patch.size(1)),
        static_cast<int>(adj_field_patch.size(2)),
        static_cast<float>(component_scale),
        adj_aux_field.mutable_data_ptr<float>(),
        adj_field_patch.mutable_data_ptr<float>(),
        coeff_patch.mutable_data_ptr<float>(),
        sample_indices.mutable_data_ptr<int>());
  }
  WITWIN_CUDA_CHECK();
}

void accumulate_tfsf_interpolated_sample_adjoint_cuda(
    torch::stable::Tensor adj_aux_field,
    const torch::stable::Tensor& adj_field_patch,
    const torch::stable::Tensor& coeff_patch,
    const torch::stable::Tensor& sample_positions,
    double origin,
    double ds,
    double component_scale) {
  check_vector(adj_aux_field, "adj_aux_field");
  check_field(adj_field_patch, "adj_field_patch");
  check_same_cuda_device(adj_aux_field, adj_field_patch, "adj_field_patch");
  check_matching_field(adj_field_patch, coeff_patch, "coeff_patch");
  check_matching_field(adj_field_patch, sample_positions, "sample_positions");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_aux_field.get_device_index());
  const int64_t total = adj_field_patch.numel();
  const int64_t adj_aux_count = adj_aux_field.numel();
  const float inv_ds = ds > 0.0 ? static_cast<float>(1.0 / ds) : 0.0f;
  if (adj_aux_count <= static_cast<int64_t>(std::numeric_limits<int>::max())) {
    accumulate_tfsf_interpolated_sample_adjoint_warp_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
        total,
        static_cast<int>(adj_aux_count),
        static_cast<float>(origin),
        inv_ds,
        static_cast<float>(component_scale),
        adj_aux_field.mutable_data_ptr<float>(),
        adj_field_patch.mutable_data_ptr<float>(),
        coeff_patch.mutable_data_ptr<float>(),
        sample_positions.mutable_data_ptr<float>());
    WITWIN_CUDA_CHECK();
    return;
  }
  accumulate_tfsf_interpolated_sample_adjoint_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      adj_aux_count,
      static_cast<float>(origin),
      inv_ds,
      static_cast<float>(component_scale),
      adj_aux_field.mutable_data_ptr<float>(),
      adj_field_patch.mutable_data_ptr<float>(),
      coeff_patch.mutable_data_ptr<float>(),
      sample_positions.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void reverse_tfsf_auxiliary_electric_cuda(
    torch::stable::Tensor adj_electric_prev,
    torch::stable::Tensor adj_magnetic_after,
    const torch::stable::Tensor& adj_electric_post,
    const torch::stable::Tensor& electric_decay,
    const torch::stable::Tensor& electric_curl,
    int64_t source_index) {
  check_vector(adj_electric_prev, "adj_electric_prev");
  check_matching_vector(adj_electric_prev, adj_electric_post, "adj_electric_post");
  check_matching_vector(adj_electric_prev, electric_decay, "electric_decay");
  check_matching_vector(adj_electric_prev, electric_curl, "electric_curl");
  check_vector(adj_magnetic_after, "adj_magnetic_after");
  check_same_cuda_device(adj_electric_prev, adj_magnetic_after, "adj_magnetic_after");
  STD_TORCH_CHECK(adj_magnetic_after.numel() + 1 == adj_electric_prev.numel(), "adj_magnetic_after length must be electric length - 1");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_electric_prev.get_device_index());
  const int64_t total = adj_electric_prev.numel();
  reverse_tfsf_auxiliary_electric_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      adj_magnetic_after.numel(),
      source_index,
      adj_electric_prev.mutable_data_ptr<float>(),
      adj_magnetic_after.mutable_data_ptr<float>(),
      adj_electric_post.mutable_data_ptr<float>(),
      electric_decay.mutable_data_ptr<float>(),
      electric_curl.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void reverse_tfsf_auxiliary_magnetic_cuda(
    torch::stable::Tensor adj_electric_prev,
    torch::stable::Tensor adj_magnetic_prev,
    const torch::stable::Tensor& adj_magnetic_after,
    const torch::stable::Tensor& magnetic_decay,
    const torch::stable::Tensor& magnetic_curl) {
  check_vector(adj_electric_prev, "adj_electric_prev");
  check_vector(adj_magnetic_prev, "adj_magnetic_prev");
  check_same_cuda_device(adj_electric_prev, adj_magnetic_prev, "adj_magnetic_prev");
  check_matching_vector(adj_magnetic_prev, adj_magnetic_after, "adj_magnetic_after");
  check_matching_vector(adj_magnetic_prev, magnetic_decay, "magnetic_decay");
  check_matching_vector(adj_magnetic_prev, magnetic_curl, "magnetic_curl");
  STD_TORCH_CHECK(adj_magnetic_prev.numel() + 1 == adj_electric_prev.numel(), "adj_electric_prev length must be magnetic length + 1");
  const torch::stable::accelerator::DeviceGuard device_guard(adj_electric_prev.get_device_index());
  const int64_t total = adj_electric_prev.numel();
  reverse_tfsf_auxiliary_magnetic_kernel<<<linear_grid(total, 512), 512, 0, current_cuda_stream()>>>(
      total,
      adj_magnetic_prev.numel(),
      adj_electric_prev.mutable_data_ptr<float>(),
      adj_magnetic_prev.mutable_data_ptr<float>(),
      adj_magnetic_after.mutable_data_ptr<float>(),
      magnetic_decay.mutable_data_ptr<float>(),
      magnetic_curl.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

namespace {

void check_seed_schedule_pack(
    const torch::stable::Tensor& pack,
    int64_t entries,
    int64_t step,
    const char* name) {
  check_float32_tensor(pack, name);
  check_contiguous_tensor(pack, name);
  STD_TORCH_CHECK(pack.dim() == 2, name, " must be a contiguous 2D (entry, step) float32 CUDA tensor");
  STD_TORCH_CHECK(pack.size(0) == entries, name, " entry dimension must match the seed gradient batch");
  STD_TORCH_CHECK(step >= 0 && step < pack.size(1), name, " step index is out of range");
}

}  // namespace

void seed_inject_dense_cuda(
    torch::stable::Tensor adj_field,
    const torch::stable::Tensor& grad_real,
    const torch::stable::Tensor& grad_imag,
    const torch::stable::Tensor& cos_pack,
    const torch::stable::Tensor& sin_pack,
    int64_t step) {
  check_field(adj_field, "adj_field");
  check_float32_tensor(grad_real, "grad_real");
  check_contiguous_tensor(grad_real, "grad_real");
  check_float32_tensor(grad_imag, "grad_imag");
  check_contiguous_tensor(grad_imag, "grad_imag");
  STD_TORCH_CHECK(grad_real.dim() == 4, "grad_real must be a contiguous 4D (entry, nx, ny, nz) tensor");
  STD_TORCH_CHECK(grad_real.sizes().equals(grad_imag.sizes()), "grad_imag must match grad_real shape");
  STD_TORCH_CHECK(
      grad_real.size(1) == adj_field.size(0) && grad_real.size(2) == adj_field.size(1) &&
          grad_real.size(3) == adj_field.size(2),
      "grad_real spatial dims must match adj_field");
  check_same_cuda_device(adj_field, grad_real, "grad_real");
  check_same_cuda_device(adj_field, grad_imag, "grad_imag");
  const int64_t entries = grad_real.size(0);
  check_seed_schedule_pack(cos_pack, entries, step, "cos_pack");
  check_seed_schedule_pack(sin_pack, entries, step, "sin_pack");
  check_same_cuda_device(adj_field, cos_pack, "cos_pack");
  check_same_cuda_device(adj_field, sin_pack, "sin_pack");
  if (entries == 0) {
    return;
  }
  const torch::stable::accelerator::DeviceGuard device_guard(adj_field.get_device_index());
  const dim3 block = field_block3d();
  seed_inject_dense_kernel<<<field_grid3d(adj_field.size(0), adj_field.size(1), adj_field.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<int>(adj_field.size(0)),
      static_cast<int>(adj_field.size(1)),
      static_cast<int>(adj_field.size(2)),
      static_cast<int>(entries),
      static_cast<int>(cos_pack.size(1)),
      static_cast<int>(step),
      adj_field.mutable_data_ptr<float>(),
      grad_real.mutable_data_ptr<float>(),
      grad_imag.mutable_data_ptr<float>(),
      cos_pack.mutable_data_ptr<float>(),
      sin_pack.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void seed_inject_point_cuda(
    torch::stable::Tensor adj_field,
    const torch::stable::Tensor& grad_real,
    const torch::stable::Tensor& grad_imag,
    const torch::stable::Tensor& point_i,
    const torch::stable::Tensor& point_j,
    const torch::stable::Tensor& point_k,
    const torch::stable::Tensor& cos_pack,
    const torch::stable::Tensor& sin_pack,
    int64_t step) {
  check_field(adj_field, "adj_field");
  check_float32_tensor(grad_real, "grad_real");
  check_contiguous_tensor(grad_real, "grad_real");
  check_float32_tensor(grad_imag, "grad_imag");
  check_contiguous_tensor(grad_imag, "grad_imag");
  STD_TORCH_CHECK(grad_real.dim() == 2, "grad_real must be a contiguous 2D (entry, point) tensor");
  STD_TORCH_CHECK(grad_real.sizes().equals(grad_imag.sizes()), "grad_imag must match grad_real shape");
  const int64_t entries = grad_real.size(0);
  const int64_t point_count = grad_real.size(1);
  check_int32_vector(point_i, "point_i");
  check_int32_vector(point_j, "point_j");
  check_int32_vector(point_k, "point_k");
  STD_TORCH_CHECK(point_i.size(0) == point_count, "point_i length must match the point gradient batch");
  STD_TORCH_CHECK(point_j.size(0) == point_count, "point_j length must match the point gradient batch");
  STD_TORCH_CHECK(point_k.size(0) == point_count, "point_k length must match the point gradient batch");
  check_same_cuda_device(adj_field, grad_real, "grad_real");
  check_same_cuda_device(adj_field, point_i, "point_i");
  check_seed_schedule_pack(cos_pack, entries, step, "cos_pack");
  check_seed_schedule_pack(sin_pack, entries, step, "sin_pack");
  check_same_cuda_device(adj_field, cos_pack, "cos_pack");
  if (entries == 0 || point_count == 0) {
    return;
  }
  const torch::stable::accelerator::DeviceGuard device_guard(adj_field.get_device_index());
  const int block_size = 256;
  seed_inject_point_kernel<int><<<linear_grid(point_count, block_size), block_size, 0, current_cuda_stream()>>>(
      static_cast<int>(point_count),
      static_cast<int>(entries),
      static_cast<int>(cos_pack.size(1)),
      static_cast<int>(step),
      static_cast<unsigned int>(adj_field.size(1)),
      static_cast<unsigned int>(adj_field.size(2)),
      adj_field.mutable_data_ptr<float>(),
      grad_real.mutable_data_ptr<float>(),
      grad_imag.mutable_data_ptr<float>(),
      point_i.mutable_data_ptr<int>(),
      point_j.mutable_data_ptr<int>(),
      point_k.mutable_data_ptr<int>(),
      cos_pack.mutable_data_ptr<float>(),
      sin_pack.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void seed_inject_plane_cuda(
    torch::stable::Tensor adj_field,
    const torch::stable::Tensor& grad_real,
    const torch::stable::Tensor& grad_imag,
    const torch::stable::Tensor& cos_pack,
    const torch::stable::Tensor& sin_pack,
    int64_t axis,
    int64_t plane_index,
    int64_t step) {
  check_field(adj_field, "adj_field");
  check_float32_tensor(grad_real, "grad_real");
  check_contiguous_tensor(grad_real, "grad_real");
  check_float32_tensor(grad_imag, "grad_imag");
  check_contiguous_tensor(grad_imag, "grad_imag");
  STD_TORCH_CHECK(grad_real.dim() == 3, "grad_real must be a contiguous 3D (entry, plane_h, plane_w) tensor");
  STD_TORCH_CHECK(grad_real.sizes().equals(grad_imag.sizes()), "grad_imag must match grad_real shape");
  STD_TORCH_CHECK(axis >= 0 && axis <= 2, "axis must be 0, 1, or 2");
  STD_TORCH_CHECK(plane_index >= 0 && plane_index < adj_field.size(axis), "plane_index is out of range");
  const int64_t entries = grad_real.size(0);
  const int64_t plane_h = axis == 0 ? adj_field.size(1) : adj_field.size(0);
  const int64_t plane_w = axis == 2 ? adj_field.size(1) : adj_field.size(2);
  STD_TORCH_CHECK(grad_real.size(1) == plane_h && grad_real.size(2) == plane_w, "grad_real plane dims must match the selected plane");
  check_same_cuda_device(adj_field, grad_real, "grad_real");
  check_seed_schedule_pack(cos_pack, entries, step, "cos_pack");
  check_seed_schedule_pack(sin_pack, entries, step, "sin_pack");
  check_same_cuda_device(adj_field, cos_pack, "cos_pack");
  if (entries == 0) {
    return;
  }
  const torch::stable::accelerator::DeviceGuard device_guard(adj_field.get_device_index());
  const auto nx = static_cast<unsigned int>(adj_field.size(0));
  const auto ny = static_cast<unsigned int>(adj_field.size(1));
  const auto nz = static_cast<unsigned int>(adj_field.size(2));
  const dim3 block(64, 4, 1);
  const unsigned int dim_x = axis == 2 ? ny : nz;
  const unsigned int dim_y = axis == 0 ? ny : nx;
  const dim3 grid((dim_x + block.x - 1) / block.x, (dim_y + block.y - 1) / block.y, 1);
  const int steps = static_cast<int>(cos_pack.size(1));
  const int e = static_cast<int>(entries);
  const int plane = static_cast<int>(plane_index);
  const int st = static_cast<int>(step);
  float* adj_ptr = adj_field.mutable_data_ptr<float>();
  const float* gr = grad_real.mutable_data_ptr<float>();
  const float* gi = grad_imag.mutable_data_ptr<float>();
  const float* cp = cos_pack.mutable_data_ptr<float>();
  const float* sp = sin_pack.mutable_data_ptr<float>();
  if (axis == 0) {
    seed_inject_plane_kernel<0><<<grid, block, 0, current_cuda_stream()>>>(nx, ny, nz, e, steps, st, plane, adj_ptr, gr, gi, cp, sp);
  } else if (axis == 1) {
    seed_inject_plane_kernel<1><<<grid, block, 0, current_cuda_stream()>>>(nx, ny, nz, e, steps, st, plane, adj_ptr, gr, gi, cp, sp);
  } else {
    seed_inject_plane_kernel<2><<<grid, block, 0, current_cuda_stream()>>>(nx, ny, nz, e, steps, st, plane, adj_ptr, gr, gi, cp, sp);
  }
  WITWIN_CUDA_CHECK();
}

void accumulate_in_place_cuda(
    torch::stable::Tensor dst,
    const torch::stable::Tensor& src) {
  check_float32_tensor(dst, "dst");
  check_contiguous_tensor(dst, "dst");
  check_float32_tensor(src, "src");
  check_contiguous_tensor(src, "src");
  check_same_cuda_device(dst, src, "src");
  STD_TORCH_CHECK(dst.numel() == src.numel(), "src must match dst element count");
  const torch::stable::accelerator::DeviceGuard device_guard(dst.get_device_index());
  const int64_t total = dst.numel();
  if (total == 0) {
    return;
  }
  accumulate_in_place_kernel<<<linear_grid(total, 512), 512, 0, current_cuda_stream()>>>(
      total,
      dst.mutable_data_ptr<float>(),
      src.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}
