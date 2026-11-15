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

template <typename ScalarT>
__device__ __forceinline__ ScalarT boundary_diff_low(ScalarT low_value, ScalarT high_value, int low_mode, ScalarT inv_delta) {
  if (low_mode == BOUNDARY_PERIODIC) {
    return (low_value - high_value) * inv_delta;
  }
  if (low_mode == BOUNDARY_PMC) {
    return 2.0f * low_value * inv_delta;
  }
  return 0.0f;
}

template <typename ScalarT>
__device__ __forceinline__ ScalarT boundary_diff_high(ScalarT low_value, ScalarT high_value, int high_mode, ScalarT inv_delta) {
  if (high_mode == BOUNDARY_PERIODIC) {
    return (low_value - high_value) * inv_delta;
  }
  if (high_mode == BOUNDARY_PMC) {
    return -2.0f * high_value * inv_delta;
  }
  return 0.0f;
}

template <int Axis, typename ScalarT>
__device__ __forceinline__ ScalarT backward_diff_axis(
    const ScalarT* __restrict__ source,
    unsigned int target_size,
    unsigned int source_y,
    unsigned int source_z,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    int low_mode,
  int high_mode,
  const ScalarT* __restrict__ inv_delta) {
  const unsigned int coord = Axis == 0 ? i : (Axis == 1 ? j : k);
  const unsigned int source_size = target_size - 1;
  const ScalarT inv = inv_delta[coord];
  const long long low = Axis == 0 ? offset3d(0, j, k, source_y, source_z)
      : (Axis == 1 ? offset3d(i, 0, k, source_y, source_z) : offset3d(i, j, 0, source_y, source_z));
  const long long high = Axis == 0 ? offset3d(source_size - 1, j, k, source_y, source_z)
      : (Axis == 1 ? offset3d(i, source_size - 1, k, source_y, source_z)
                   : offset3d(i, j, source_size - 1, source_y, source_z));
  if (coord == 0) {
    const ScalarT low_value = source[low];
    const ScalarT high_value = source[high];
    return boundary_diff_low(low_value, high_value, low_mode, inv);
  }
  if (coord + 1 == target_size) {
    const ScalarT low_value = source[low];
    const ScalarT high_value = source[high];
    return boundary_diff_high(low_value, high_value, high_mode, inv);
  }
  const long long current = offset3d(i, j, k, source_y, source_z);
  const long long previous = Axis == 0 ? offset3d(i - 1, j, k, source_y, source_z)
      : (Axis == 1 ? offset3d(i, j - 1, k, source_y, source_z) : offset3d(i, j, k - 1, source_y, source_z));
  return (source[current] - source[previous]) * inv;
}

template <typename ScalarT>
__device__ __forceinline__ ScalarT backward_diff_axis2(
    const ScalarT* __restrict__ source,
    unsigned int source_y,
    unsigned int source_z,
    unsigned int i,
    unsigned int j,
    unsigned int k,
  int low_mode,
  int high_mode,
  const ScalarT* __restrict__ inv_delta) {
  return backward_diff_axis<2>(source, source_z + 1, source_y, source_z, i, j, k, low_mode, high_mode, inv_delta);
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

template <int Axis>
__device__ __forceinline__ ComplexValue bloch_backward_diff_axis(
    const float* __restrict__ real,
    const float* __restrict__ imag,
    unsigned int target_size,
    unsigned int source_x,
    unsigned int source_y,
    unsigned int source_z,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    float phase_cos,
    float phase_sin,
    const float* __restrict__ inv_delta) {
  const unsigned int coord = Axis == 0 ? i : (Axis == 1 ? j : k);
  const unsigned int source_size = Axis == 0 ? source_x : (Axis == 1 ? source_y : source_z);
  const float inv = inv_delta[coord];
  const long long low = Axis == 0 ? offset3d(0, j, k, source_y, source_z)
      : (Axis == 1 ? offset3d(i, 0, k, source_y, source_z) : offset3d(i, j, 0, source_y, source_z));
  const long long high = Axis == 0 ? offset3d(source_size - 1, j, k, source_y, source_z)
      : (Axis == 1 ? offset3d(i, source_size - 1, k, source_y, source_z)
                   : offset3d(i, j, source_size - 1, source_y, source_z));
  if (coord == 0) {
    const ComplexValue shifted = phase_negative(real[high], imag[high], phase_cos, phase_sin);
    return {(real[low] - shifted.real) * inv, (imag[low] - shifted.imag) * inv};
  }
  if (coord + 1 == target_size) {
    const ComplexValue shifted = phase_positive(real[low], imag[low], phase_cos, phase_sin);
    return {(shifted.real - real[high]) * inv, (shifted.imag - imag[high]) * inv};
  }
  const long long current = offset3d(i, j, k, source_y, source_z);
  const long long previous = Axis == 0 ? offset3d(i - 1, j, k, source_y, source_z)
      : (Axis == 1 ? offset3d(i, j - 1, k, source_y, source_z) : offset3d(i, j, k - 1, source_y, source_z));
  return {(real[current] - real[previous]) * inv, (imag[current] - imag[previous]) * inv};
}

__device__ __forceinline__ ComplexValue bloch_backward_diff_axis0(
    const float* real, const float* imag, unsigned int target_x, unsigned int source_x,
    unsigned int source_y, unsigned int source_z, unsigned int i, unsigned int j, unsigned int k,
    float phase_cos, float phase_sin, const float* inv_delta) {
  return bloch_backward_diff_axis<0>(real, imag, target_x, source_x, source_y, source_z, i, j, k, phase_cos, phase_sin, inv_delta);
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
  return bloch_backward_diff_axis<1>(real, imag, target_y, 0, source_y, source_z, i, j, k, phase_cos, phase_sin, inv_delta);
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
  return bloch_backward_diff_axis<2>(real, imag, target_z, 0, source_y, source_z, i, j, k, phase_cos, phase_sin, inv_delta);
}

__global__ void advance_modulation_time_kernel(
    double dt, double* __restrict__ modulation_time, int64_t* __restrict__ modulation_step) {
  const int64_t step = ++modulation_step[0];
  modulation_time[0] = static_cast<double>(step - 1) * dt;
  modulation_time[1] = static_cast<double>(step) * dt;
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

void check_electric_inputs(
    const torch::stable::Tensor& field,
    const torch::stable::Tensor& first,
    const torch::stable::Tensor& second,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const char* name) {
  check_cuda_tensor(field, name);
  STD_TORCH_CHECK(field.scalar_type() == torch::headeronly::ScalarType::Float || field.scalar_type() == torch::headeronly::ScalarType::Double, "field must be float32 or float64");
  STD_TORCH_CHECK(field.scalar_type() == field.scalar_type(), "field update dtypes must match");
  check_cuda_tensor(first, "first");
  STD_TORCH_CHECK(first.scalar_type() == field.scalar_type(), "field update dtypes must match");
  check_cuda_tensor(second, "second");
  STD_TORCH_CHECK(second.scalar_type() == field.scalar_type(), "field update dtypes must match");
  check_cuda_tensor(decay, "decay");
  STD_TORCH_CHECK(decay.scalar_type() == field.scalar_type(), "field update dtypes must match");
  check_cuda_tensor(curl, "curl");
  STD_TORCH_CHECK(curl.scalar_type() == field.scalar_type(), "field update dtypes must match");
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

void check_vector_input(const torch::stable::Tensor& tensor, int64_t length, const char* name,
    torch::headeronly::ScalarType dtype = torch::headeronly::ScalarType::Float) {
  check_cuda_tensor(tensor, name);
  STD_TORCH_CHECK(tensor.scalar_type() == dtype, "profile dtype must match field");
  check_contiguous_tensor(tensor, name);
  STD_TORCH_CHECK(tensor.dim() == 1, name, " must be rank 1");
  STD_TORCH_CHECK(tensor.size(0) == length, name, " length must match CPML field axis");
}

void check_spacing_vector(
    const torch::stable::Tensor& field,
    const torch::stable::Tensor& inv_delta,
    int64_t axis,
    const char* name) {
  check_vector_input(inv_delta, field.size(axis), name, field.scalar_type());
  check_same_cuda_device(field, inv_delta, name);
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

// CPML psi + kappa correction along y (the PML axis) for the general
// single-PML-axis Bloch+CPML electric update, where the two Bloch axes carry
// the transverse phase in the base full-Bloch kernel. Ex: derivative of Hz
// along y; tangent (skip) check is on the x/z Bloch axis that also touches Ex.
// The region is narrowed along y (the psi memory axis); region-local
// (x, ly, k) map to global (offset_i + x, offset_j + ly, offset_k + k).
__global__ void apply_electric_ex_cpml_y_correction_kernel(
    unsigned int sx,
    unsigned int sy,
    unsigned int sz,
    unsigned int hz_ny,
    unsigned int hz_nz,
    const float* __restrict__ hz,
    const float* __restrict__ curl,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ b_y,
    const float* __restrict__ c_y,
    const float* __restrict__ inv_dy,
    int offset_i,
    int offset_j,
    int offset_k,
    int z_low_mode,
    int z_high_mode,
    unsigned int full_size_z,
    int local_y_start,
    int local_y_stop,
    float* __restrict__ psi_y,
    float* __restrict__ ex) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int ly = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int x = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= sx || ly >= sy || k >= sz) {
    return;
  }
  if (static_cast<int>(ly) < local_y_start || static_cast<int>(ly) >= local_y_stop) {
    return;
  }
  const unsigned int gz = static_cast<unsigned int>(offset_k) + k;
  if (!cpml_correction_active(gz, full_size_z, z_low_mode, z_high_mode)) {
    return;
  }
  const unsigned int gx = static_cast<unsigned int>(offset_i) + x;
  const unsigned int gy = static_cast<unsigned int>(offset_j) + ly;
  const long long region_lin = offset3d(x, ly, k, sy, sz);
  const long long hz_cur = offset3d(gx, gy, gz, hz_ny, hz_nz);
  const long long hz_prev = offset3d(gx, gy - 1, gz, hz_ny, hz_nz);
  const float derivative = (hz[hz_cur] - hz[hz_prev]) * inv_dy[gy];
  const float psi_new = b_y[gy] * psi_y[region_lin] + c_y[gy] * derivative;
  psi_y[region_lin] = psi_new;
  const float correction = derivative * (inv_kappa_y[gy] - 1.0f) + psi_new;
  ex[region_lin] = ex[region_lin] + curl[region_lin] * correction;
}

// Ez: derivative of Hx along y (the PML axis); tangent check on x.
__global__ void apply_electric_ez_cpml_y_correction_kernel(
    unsigned int sx,
    unsigned int sy,
    unsigned int sz,
    unsigned int hx_ny,
    unsigned int hx_nz,
    const float* __restrict__ hx,
    const float* __restrict__ curl,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ b_y,
    const float* __restrict__ c_y,
    const float* __restrict__ inv_dy,
    int offset_i,
    int offset_j,
    int offset_k,
    int x_low_mode,
    int x_high_mode,
    unsigned int full_size_x,
    int local_y_start,
    int local_y_stop,
    float* __restrict__ psi_y,
    float* __restrict__ ez) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int ly = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int x = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= sx || ly >= sy || k >= sz) {
    return;
  }
  if (static_cast<int>(ly) < local_y_start || static_cast<int>(ly) >= local_y_stop) {
    return;
  }
  const unsigned int gx = static_cast<unsigned int>(offset_i) + x;
  if (!cpml_correction_active(gx, full_size_x, x_low_mode, x_high_mode)) {
    return;
  }
  const unsigned int gy = static_cast<unsigned int>(offset_j) + ly;
  const unsigned int gz = static_cast<unsigned int>(offset_k) + k;
  const long long region_lin = offset3d(x, ly, k, sy, sz);
  const long long hx_cur = offset3d(gx, gy, gz, hx_ny, hx_nz);
  const long long hx_prev = offset3d(gx, gy - 1, gz, hx_ny, hx_nz);
  const float derivative = (hx[hx_cur] - hx[hx_prev]) * inv_dy[gy];
  const float psi_new = b_y[gy] * psi_y[region_lin] + c_y[gy] * derivative;
  psi_y[region_lin] = psi_new;
  const float correction = derivative * (inv_kappa_y[gy] - 1.0f) + psi_new;
  ez[region_lin] = ez[region_lin] - curl[region_lin] * correction;
}

// CPML correction along x (the PML axis). Region narrowed along x; region-local
// (lx, y, k) map to global (offset_i + lx, offset_j + y, offset_k + k).
// Ey: derivative of Hz along x; tangent check on z.
__global__ void apply_electric_ey_cpml_x_correction_kernel(
    unsigned int sx,
    unsigned int sy,
    unsigned int sz,
    unsigned int hz_ny,
    unsigned int hz_nz,
    const float* __restrict__ hz,
    const float* __restrict__ curl,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ b_x,
    const float* __restrict__ c_x,
    const float* __restrict__ inv_dx,
    int offset_i,
    int offset_j,
    int offset_k,
    int z_low_mode,
    int z_high_mode,
    unsigned int full_size_z,
    int local_x_start,
    int local_x_stop,
    float* __restrict__ psi_x,
    float* __restrict__ ey) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int lx = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx >= sx || y >= sy || k >= sz) {
    return;
  }
  if (static_cast<int>(lx) < local_x_start || static_cast<int>(lx) >= local_x_stop) {
    return;
  }
  const unsigned int gz = static_cast<unsigned int>(offset_k) + k;
  if (!cpml_correction_active(gz, full_size_z, z_low_mode, z_high_mode)) {
    return;
  }
  const unsigned int gx = static_cast<unsigned int>(offset_i) + lx;
  const unsigned int gy = static_cast<unsigned int>(offset_j) + y;
  const long long region_lin = offset3d(lx, y, k, sy, sz);
  const long long hz_cur = offset3d(gx, gy, gz, hz_ny, hz_nz);
  const long long hz_prev = offset3d(gx - 1, gy, gz, hz_ny, hz_nz);
  const float derivative = (hz[hz_cur] - hz[hz_prev]) * inv_dx[gx];
  const float psi_new = b_x[gx] * psi_x[region_lin] + c_x[gx] * derivative;
  psi_x[region_lin] = psi_new;
  const float correction = derivative * (inv_kappa_x[gx] - 1.0f) + psi_new;
  ey[region_lin] = ey[region_lin] - curl[region_lin] * correction;
}

// Ez: derivative of Hy along x (the PML axis); tangent check on y.
__global__ void apply_electric_ez_cpml_x_correction_kernel(
    unsigned int sx,
    unsigned int sy,
    unsigned int sz,
    unsigned int hy_ny,
    unsigned int hy_nz,
    const float* __restrict__ hy,
    const float* __restrict__ curl,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ b_x,
    const float* __restrict__ c_x,
    const float* __restrict__ inv_dx,
    int offset_i,
    int offset_j,
    int offset_k,
    int y_low_mode,
    int y_high_mode,
    unsigned int full_size_y,
    int local_x_start,
    int local_x_stop,
    float* __restrict__ psi_x,
    float* __restrict__ ez) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int lx = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx >= sx || y >= sy || k >= sz) {
    return;
  }
  if (static_cast<int>(lx) < local_x_start || static_cast<int>(lx) >= local_x_stop) {
    return;
  }
  const unsigned int gy = static_cast<unsigned int>(offset_j) + y;
  if (!cpml_correction_active(gy, full_size_y, y_low_mode, y_high_mode)) {
    return;
  }
  const unsigned int gx = static_cast<unsigned int>(offset_i) + lx;
  const unsigned int gz = static_cast<unsigned int>(offset_k) + k;
  const long long region_lin = offset3d(lx, y, k, sy, sz);
  const long long hy_cur = offset3d(gx, gy, gz, hy_ny, hy_nz);
  const long long hy_prev = offset3d(gx - 1, gy, gz, hy_ny, hy_nz);
  const float derivative = (hy[hy_cur] - hy[hy_prev]) * inv_dx[gx];
  const float psi_new = b_x[gx] * psi_x[region_lin] + c_x[gx] * derivative;
  psi_x[region_lin] = psi_new;
  const float correction = derivative * (inv_kappa_x[gx] - 1.0f) + psi_new;
  ez[region_lin] = ez[region_lin] + curl[region_lin] * correction;
}

}  // namespace

void advance_modulation_time_cuda(
    torch::stable::Tensor modulation_time, torch::stable::Tensor modulation_step, double dt) {
  check_cuda_tensor(modulation_time, "modulation_time");
  STD_TORCH_CHECK(modulation_time.scalar_type() == torch::headeronly::ScalarType::Double,
                  "modulation_time must be float64");
  check_contiguous_tensor(modulation_time, "modulation_time");
  STD_TORCH_CHECK(
      modulation_time.dim() == 1 && modulation_time.numel() == 2,
      "modulation_time must be a contiguous float64 CUDA tensor with two elements");
  check_cuda_tensor(modulation_step, "modulation_step");
  check_same_cuda_device(modulation_time, modulation_step, "modulation_step");
  check_contiguous_tensor(modulation_step, "modulation_step");
  STD_TORCH_CHECK(modulation_step.scalar_type() == torch::headeronly::ScalarType::Long &&
                      modulation_step.dim() == 1 && modulation_step.numel() == 1,
                  "modulation_step must be a contiguous int64 CUDA tensor with one element");
  const torch::stable::accelerator::DeviceGuard guard(modulation_time.get_device_index());
  advance_modulation_time_kernel<<<1, 1, 0, current_cuda_stream()>>>(
      dt, modulation_time.mutable_data_ptr<double>(), modulation_step.mutable_data_ptr<int64_t>());
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

void apply_electric_ex_cpml_y_correction_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_y,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_dy,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    int64_t z_low_mode,
    int64_t z_high_mode,
    int64_t full_size_z,
    int64_t full_size_y) {
  const auto sizes = ex.sizes();
  const int sx = static_cast<int>(sizes[0]);
  const int sy = static_cast<int>(sizes[1]);
  const int sz = static_cast<int>(sizes[2]);
  const int start_candidate = 1 - static_cast<int>(offset_j);
  const int local_y_start = start_candidate > 0 ? start_candidate : 0;
  const int stop_candidate = static_cast<int>(full_size_y) - 1 - static_cast<int>(offset_j);
  const int local_y_stop = sy < stop_candidate ? sy : stop_candidate;
  if (local_y_stop <= local_y_start) {
    return;
  }
  check_vector_input(inv_dy, full_size_y, "inv_dy");
  check_same_cuda_device(ex, inv_dy, "inv_dy");
  torch::stable::accelerator::DeviceGuard guard(ex.get_device_index());
  const dim3 block = field_block3d();
  apply_electric_ex_cpml_y_correction_kernel<<<field_grid3d(sx, sy, sz, block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sx),
      static_cast<unsigned int>(sy),
      static_cast<unsigned int>(sz),
      static_cast<unsigned int>(hz.size(1)),
      static_cast<unsigned int>(hz.size(2)),
      hz.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      inv_kappa_y.mutable_data_ptr<float>(),
      b_y.mutable_data_ptr<float>(),
      c_y.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>(),
      static_cast<int>(offset_i),
      static_cast<int>(offset_j),
      static_cast<int>(offset_k),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode),
      static_cast<unsigned int>(full_size_z),
      local_y_start,
      local_y_stop,
      psi_y.mutable_data_ptr<float>(),
      ex.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void apply_electric_ez_cpml_y_correction_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_y,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_dy,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t full_size_x,
    int64_t full_size_y) {
  const auto sizes = ez.sizes();
  const int sx = static_cast<int>(sizes[0]);
  const int sy = static_cast<int>(sizes[1]);
  const int sz = static_cast<int>(sizes[2]);
  const int start_candidate = 1 - static_cast<int>(offset_j);
  const int local_y_start = start_candidate > 0 ? start_candidate : 0;
  const int stop_candidate = static_cast<int>(full_size_y) - 1 - static_cast<int>(offset_j);
  const int local_y_stop = sy < stop_candidate ? sy : stop_candidate;
  if (local_y_stop <= local_y_start) {
    return;
  }
  check_vector_input(inv_dy, full_size_y, "inv_dy");
  check_same_cuda_device(ez, inv_dy, "inv_dy");
  torch::stable::accelerator::DeviceGuard guard(ez.get_device_index());
  const dim3 block = field_block3d();
  apply_electric_ez_cpml_y_correction_kernel<<<field_grid3d(sx, sy, sz, block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sx),
      static_cast<unsigned int>(sy),
      static_cast<unsigned int>(sz),
      static_cast<unsigned int>(hx.size(1)),
      static_cast<unsigned int>(hx.size(2)),
      hx.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      inv_kappa_y.mutable_data_ptr<float>(),
      b_y.mutable_data_ptr<float>(),
      c_y.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>(),
      static_cast<int>(offset_i),
      static_cast<int>(offset_j),
      static_cast<int>(offset_k),
      static_cast<int>(x_low_mode),
      static_cast<int>(x_high_mode),
      static_cast<unsigned int>(full_size_x),
      local_y_start,
      local_y_stop,
      psi_y.mutable_data_ptr<float>(),
      ez.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void apply_electric_ey_cpml_x_correction_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_x,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_dx,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    int64_t z_low_mode,
    int64_t z_high_mode,
    int64_t full_size_z,
    int64_t full_size_x) {
  const auto sizes = ey.sizes();
  const int sx = static_cast<int>(sizes[0]);
  const int sy = static_cast<int>(sizes[1]);
  const int sz = static_cast<int>(sizes[2]);
  const int start_candidate = 1 - static_cast<int>(offset_i);
  const int local_x_start = start_candidate > 0 ? start_candidate : 0;
  const int stop_candidate = static_cast<int>(full_size_x) - 1 - static_cast<int>(offset_i);
  const int local_x_stop = sx < stop_candidate ? sx : stop_candidate;
  if (local_x_stop <= local_x_start) {
    return;
  }
  check_vector_input(inv_dx, full_size_x, "inv_dx");
  check_same_cuda_device(ey, inv_dx, "inv_dx");
  torch::stable::accelerator::DeviceGuard guard(ey.get_device_index());
  const dim3 block = field_block3d();
  apply_electric_ey_cpml_x_correction_kernel<<<field_grid3d(sx, sy, sz, block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sx),
      static_cast<unsigned int>(sy),
      static_cast<unsigned int>(sz),
      static_cast<unsigned int>(hz.size(1)),
      static_cast<unsigned int>(hz.size(2)),
      hz.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      inv_kappa_x.mutable_data_ptr<float>(),
      b_x.mutable_data_ptr<float>(),
      c_x.mutable_data_ptr<float>(),
      inv_dx.mutable_data_ptr<float>(),
      static_cast<int>(offset_i),
      static_cast<int>(offset_j),
      static_cast<int>(offset_k),
      static_cast<int>(z_low_mode),
      static_cast<int>(z_high_mode),
      static_cast<unsigned int>(full_size_z),
      local_x_start,
      local_x_stop,
      psi_x.mutable_data_ptr<float>(),
      ey.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void apply_electric_ez_cpml_x_correction_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_x,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_dx,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t full_size_y,
    int64_t full_size_x) {
  const auto sizes = ez.sizes();
  const int sx = static_cast<int>(sizes[0]);
  const int sy = static_cast<int>(sizes[1]);
  const int sz = static_cast<int>(sizes[2]);
  const int start_candidate = 1 - static_cast<int>(offset_i);
  const int local_x_start = start_candidate > 0 ? start_candidate : 0;
  const int stop_candidate = static_cast<int>(full_size_x) - 1 - static_cast<int>(offset_i);
  const int local_x_stop = sx < stop_candidate ? sx : stop_candidate;
  if (local_x_stop <= local_x_start) {
    return;
  }
  check_vector_input(inv_dx, full_size_x, "inv_dx");
  check_same_cuda_device(ez, inv_dx, "inv_dx");
  torch::stable::accelerator::DeviceGuard guard(ez.get_device_index());
  const dim3 block = field_block3d();
  apply_electric_ez_cpml_x_correction_kernel<<<field_grid3d(sx, sy, sz, block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sx),
      static_cast<unsigned int>(sy),
      static_cast<unsigned int>(sz),
      static_cast<unsigned int>(hy.size(1)),
      static_cast<unsigned int>(hy.size(2)),
      hy.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      inv_kappa_x.mutable_data_ptr<float>(),
      b_x.mutable_data_ptr<float>(),
      c_x.mutable_data_ptr<float>(),
      inv_dx.mutable_data_ptr<float>(),
      static_cast<int>(offset_i),
      static_cast<int>(offset_j),
      static_cast<int>(offset_k),
      static_cast<int>(y_low_mode),
      static_cast<int>(y_high_mode),
      static_cast<unsigned int>(full_size_y),
      local_x_start,
      local_x_stop,
      psi_x.mutable_data_ptr<float>(),
      ez.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}
