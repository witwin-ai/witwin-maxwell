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

__global__ void update_debye_kernel(
    int64_t total,
    const float* __restrict__ electric,
    const float* __restrict__ drive,
    float decay,
    float inv_dt,
    float* __restrict__ polarization,
    float* __restrict__ current) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const float previous = polarization[index];
  const float next = decay * previous + drive[index] * electric[index];
  polarization[index] = next;
  current[index] = (next - previous) * inv_dt;
}

__global__ void update_drude_kernel(
    int64_t total,
    const float* __restrict__ electric,
    const float* __restrict__ drive,
    float decay,
    float* __restrict__ current) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  current[index] = decay * current[index] + drive[index] * electric[index];
}

__global__ void update_lorentz_kernel(
    int64_t total,
    const float* __restrict__ electric,
    const float* __restrict__ drive,
    float decay,
    float restoring,
    float dt,
    float* __restrict__ polarization,
    float* __restrict__ current) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const float previous_current = current[index];
  const float previous_polarization = polarization[index];
  const float next_current =
      decay * previous_current -
      restoring * previous_polarization +
      drive[index] * electric[index];
  current[index] = next_current;
  polarization[index] = previous_polarization + dt * next_current;
}

__global__ void apply_polarization_kernel(
    int64_t total,
    const float* __restrict__ current,
    const float* __restrict__ inv_permittivity,
    float dt,
    float* __restrict__ electric) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  electric[index] -= dt * current[index] * inv_permittivity[index];
}

// Space-time modulated variant of the ADE polarization-current subtraction.
// The modulated E update divides the curl(H) term by the instantaneous
// modulation factor m_next(x) = 1 + mod_cos*cos(W(x) t_next) - mod_sin*sin(W(x) t_next),
// with the per-cell angular frequency W(x) supplied as the mod_omega field;
// for the conservative eps_inf(x,t) = eps_inf(x) * m(x,t) update the dispersive
// polarization current must be divided by the SAME eps_inf * m_next, so the
// resonant response and the modulated background stay mutually consistent inside
// a cell that is both modulated and dispersive. Where the modulation depth is
// zero (m_next = 1) this reduces bit-exactly to apply_polarization_kernel.
__global__ void apply_polarization_modulated_kernel(
    int64_t total,
    const float* __restrict__ current,
    const float* __restrict__ inv_permittivity,
    const float* __restrict__ mod_cos,
    const float* __restrict__ mod_sin,
    const float* __restrict__ mod_omega,
    float t_next,
    float dt,
    float* __restrict__ electric) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  // Per-cell modulation frequency: evaluate the new-time phase from this cell's own
  // angular frequency so a dispersive cell divides by the SAME m_next its modulated
  // curl(H) update used, even when the scene mixes several modulation frequencies.
  float sin_next, cos_next;
  sincosf(mod_omega[index] * t_next, &sin_next, &cos_next);
  const float m_next =
      fmaxf(1.0f + mod_cos[index] * cos_next - mod_sin[index] * sin_next, 1.0e-6f);
  electric[index] -= dt * current[index] * inv_permittivity[index] / m_next;
}

__device__ __forceinline__ int clamp_index(int index, int size) {
  if (index <= 0) {
    return 0;
  }
  const int max_index = size - 1;
  if (index > max_index) {
    return max_index;
  }
  return index;
}

__device__ __forceinline__ float sample_direct(
    const float* __restrict__ field,
    int i,
    int j,
    int k,
    int size_y,
    int size_z) {
  return field[offset3d(
      static_cast<unsigned int>(i),
      static_cast<unsigned int>(j),
      static_cast<unsigned int>(k),
      static_cast<unsigned int>(size_y),
      static_cast<unsigned int>(size_z))];
}

__device__ __forceinline__ float sample_clamped(
    const float* __restrict__ field,
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

// Collocate the three E components onto the Yee edge of `Component`
// (0 = Ex, 1 = Ey, 2 = Ez). The two off-axis components are 4-point averaged
// from the surrounding edges (clamped at domain faces); the own component is
// read directly at `linear`.
template <int Component>
__device__ __forceinline__ void collocate_electric_components(
    int i,
    int j,
    int k,
    long long linear,
    int ex_x,
    int ex_y,
    int ex_z,
    int ey_x,
    int ey_y,
    int ey_z,
    int ez_x,
    int ez_y,
    int ez_z,
    const float* __restrict__ ex,
    const float* __restrict__ ey,
    const float* __restrict__ ez,
    float& ex_value,
    float& ey_value,
    float& ez_value) {
  if constexpr (Component == 0) {
    ex_value = ex[linear];
    const bool ey_interior = i + 1 < ey_x && j > 0 && j < ey_y && k < ey_z;
    ey_value = ey_interior
        ? 0.25f * (
              sample_direct(ey, i, j - 1, k, ey_y, ey_z) +
              sample_direct(ey, i, j, k, ey_y, ey_z) +
              sample_direct(ey, i + 1, j - 1, k, ey_y, ey_z) +
              sample_direct(ey, i + 1, j, k, ey_y, ey_z))
        : 0.25f * (
              sample_clamped(ey, i, j - 1, k, ey_x, ey_y, ey_z) +
              sample_clamped(ey, i, j, k, ey_x, ey_y, ey_z) +
              sample_clamped(ey, i + 1, j - 1, k, ey_x, ey_y, ey_z) +
              sample_clamped(ey, i + 1, j, k, ey_x, ey_y, ey_z));
    const bool ez_interior = i + 1 < ez_x && j < ez_y && k > 0 && k < ez_z;
    ez_value = ez_interior
        ? 0.25f * (
              sample_direct(ez, i, j, k - 1, ez_y, ez_z) +
              sample_direct(ez, i, j, k, ez_y, ez_z) +
              sample_direct(ez, i + 1, j, k - 1, ez_y, ez_z) +
              sample_direct(ez, i + 1, j, k, ez_y, ez_z))
        : 0.25f * (
              sample_clamped(ez, i, j, k - 1, ez_x, ez_y, ez_z) +
              sample_clamped(ez, i, j, k, ez_x, ez_y, ez_z) +
              sample_clamped(ez, i + 1, j, k - 1, ez_x, ez_y, ez_z) +
              sample_clamped(ez, i + 1, j, k, ez_x, ez_y, ez_z));
  } else if constexpr (Component == 1) {
    const bool ex_interior = i > 0 && i < ex_x && j + 1 < ex_y && k < ex_z;
    ex_value = ex_interior
        ? 0.25f * (
              sample_direct(ex, i - 1, j, k, ex_y, ex_z) +
              sample_direct(ex, i, j, k, ex_y, ex_z) +
              sample_direct(ex, i - 1, j + 1, k, ex_y, ex_z) +
              sample_direct(ex, i, j + 1, k, ex_y, ex_z))
        : 0.25f * (
              sample_clamped(ex, i - 1, j, k, ex_x, ex_y, ex_z) +
              sample_clamped(ex, i, j, k, ex_x, ex_y, ex_z) +
              sample_clamped(ex, i - 1, j + 1, k, ex_x, ex_y, ex_z) +
              sample_clamped(ex, i, j + 1, k, ex_x, ex_y, ex_z));
    ey_value = ey[linear];
    const bool ez_interior = i < ez_x && j + 1 < ez_y && k > 0 && k < ez_z;
    ez_value = ez_interior
        ? 0.25f * (
              sample_direct(ez, i, j, k - 1, ez_y, ez_z) +
              sample_direct(ez, i, j, k, ez_y, ez_z) +
              sample_direct(ez, i, j + 1, k - 1, ez_y, ez_z) +
              sample_direct(ez, i, j + 1, k, ez_y, ez_z))
        : 0.25f * (
              sample_clamped(ez, i, j, k - 1, ez_x, ez_y, ez_z) +
              sample_clamped(ez, i, j, k, ez_x, ez_y, ez_z) +
              sample_clamped(ez, i, j + 1, k - 1, ez_x, ez_y, ez_z) +
              sample_clamped(ez, i, j + 1, k, ez_x, ez_y, ez_z));
  } else {
    const bool ex_interior = i > 0 && i < ex_x && j < ex_y && k + 1 < ex_z;
    ex_value = ex_interior
        ? 0.25f * (
              sample_direct(ex, i - 1, j, k, ex_y, ex_z) +
              sample_direct(ex, i, j, k, ex_y, ex_z) +
              sample_direct(ex, i - 1, j, k + 1, ex_y, ex_z) +
              sample_direct(ex, i, j, k + 1, ex_y, ex_z))
        : 0.25f * (
              sample_clamped(ex, i - 1, j, k, ex_x, ex_y, ex_z) +
              sample_clamped(ex, i, j, k, ex_x, ex_y, ex_z) +
              sample_clamped(ex, i - 1, j, k + 1, ex_x, ex_y, ex_z) +
              sample_clamped(ex, i, j, k + 1, ex_x, ex_y, ex_z));
    const bool ey_interior = i < ey_x && j > 0 && j < ey_y && k + 1 < ey_z;
    ey_value = ey_interior
        ? 0.25f * (
              sample_direct(ey, i, j - 1, k, ey_y, ey_z) +
              sample_direct(ey, i, j, k, ey_y, ey_z) +
              sample_direct(ey, i, j - 1, k + 1, ey_y, ey_z) +
              sample_direct(ey, i, j, k + 1, ey_y, ey_z))
        : 0.25f * (
              sample_clamped(ey, i, j - 1, k, ey_x, ey_y, ey_z) +
              sample_clamped(ey, i, j, k, ey_x, ey_y, ey_z) +
              sample_clamped(ey, i, j - 1, k + 1, ey_x, ey_y, ey_z) +
              sample_clamped(ey, i, j, k + 1, ey_x, ey_y, ey_z));
    ez_value = ez[linear];
  }
}

template <int Component>
__global__ void update_kerr_curl_kernel(
    int dynamic_x,
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
    const float* __restrict__ ex,
    const float* __restrict__ ey,
    const float* __restrict__ ez,
    const float* __restrict__ linear_permittivity,
    const float* __restrict__ decay,
    const float* __restrict__ chi3,
    float dt,
    float eps0,
    float* __restrict__ dynamic_curl) {
  const unsigned int k_u = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j_u = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i_u = blockIdx.z * blockDim.z + threadIdx.z;
  if (i_u >= static_cast<unsigned int>(dynamic_x)
      || j_u >= static_cast<unsigned int>(dynamic_y)
      || k_u >= static_cast<unsigned int>(dynamic_z)) {
    return;
  }
  const long long linear = offset3d(i_u, j_u, k_u, dynamic_y, dynamic_z);
  const int i = static_cast<int>(i_u);
  const int j = static_cast<int>(j_u);
  const int k = static_cast<int>(k_u);

  float ex_value;
  float ey_value;
  float ez_value;
  collocate_electric_components<Component>(
      i, j, k, linear,
      ex_x, ex_y, ex_z,
      ey_x, ey_y, ey_z,
      ez_x, ez_y, ez_z,
      ex, ey, ez,
      ex_value, ey_value, ez_value);

  float effective =
      linear_permittivity[linear] +
      eps0 * chi3[linear] * (ex_value * ex_value + ey_value * ey_value + ez_value * ez_value);
  const float floor = 1.0e-12f * eps0;
  if (effective < floor) {
    effective = floor;
  }
  dynamic_curl[linear] = (dt / effective) * decay[linear];
}

// General instantaneous-nonlinearity coefficient kernel. Recomposes the
// semi-implicit lossy decay/curl pair every step from
//   eps_eff  = eps_lin + eps0 * (chi2 * E_own + chi3 * |E|^2)
//   sigma    = sigma_static + tpa_sigma * |E|^2
//   half     = 0.5 * sigma * dt / eps_eff
//   decay    = external * (1 - half) / (1 + half)
//   curl     = external * (dt / eps_eff) / (1 + half)
// where `external` carries the PML split-field decay and the PEC open
// fraction. With all nonlinear channels zero this reproduces the static
// coefficients of `_electric_update_coefficients` exactly.
template <int Component>
__global__ void update_nonlinear_coefficients_kernel(
    int dynamic_x,
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
    const float* __restrict__ ex,
    const float* __restrict__ ey,
    const float* __restrict__ ez,
    const float* __restrict__ linear_permittivity,
    const float* __restrict__ external_decay,
    const float* __restrict__ sigma_static,
    const float* __restrict__ chi2,
    const float* __restrict__ chi3,
    const float* __restrict__ tpa_sigma,
    float dt,
    float eps0,
    float* __restrict__ dynamic_decay,
    float* __restrict__ dynamic_curl) {
  const unsigned int k_u = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j_u = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i_u = blockIdx.z * blockDim.z + threadIdx.z;
  if (i_u >= static_cast<unsigned int>(dynamic_x)
      || j_u >= static_cast<unsigned int>(dynamic_y)
      || k_u >= static_cast<unsigned int>(dynamic_z)) {
    return;
  }
  const long long linear = offset3d(i_u, j_u, k_u, dynamic_y, dynamic_z);
  const int i = static_cast<int>(i_u);
  const int j = static_cast<int>(j_u);
  const int k = static_cast<int>(k_u);

  float ex_value;
  float ey_value;
  float ez_value;
  collocate_electric_components<Component>(
      i, j, k, linear,
      ex_x, ex_y, ex_z,
      ey_x, ey_y, ey_z,
      ez_x, ez_y, ez_z,
      ex, ey, ez,
      ex_value, ey_value, ez_value);

  const float field_sq = ex_value * ex_value + ey_value * ey_value + ez_value * ez_value;
  float own_value;
  if constexpr (Component == 0) {
    own_value = ex_value;
  } else if constexpr (Component == 1) {
    own_value = ey_value;
  } else {
    own_value = ez_value;
  }

  float effective =
      linear_permittivity[linear] +
      eps0 * (chi2[linear] * own_value + chi3[linear] * field_sq);
  const float floor = 1.0e-12f * eps0;
  if (effective < floor) {
    effective = floor;
  }
  float sigma = sigma_static[linear] + tpa_sigma[linear] * field_sq;
  if (sigma < 0.0f) {
    sigma = 0.0f;
  }
  const float half = 0.5f * sigma * dt / effective;
  const float inv_denom = 1.0f / (1.0f + half);
  const float external = external_decay[linear];
  dynamic_decay[linear] = external * (1.0f - half) * inv_denom;
  dynamic_curl[linear] = external * (dt / effective) * inv_denom;
}

template <int Component>
void launch_kerr_curl_kernel(
    torch::stable::Tensor dynamic_curl,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& linear_permittivity,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& chi3,
    double dt,
    double eps0) {
  const dim3 block = field_block3d();
  update_kerr_curl_kernel<Component><<<field_grid3d(dynamic_curl.size(0), dynamic_curl.size(1), dynamic_curl.size(2), block), block, 0, current_cuda_stream()>>>(
      dynamic_curl.size(0),
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
      ex.mutable_data_ptr<float>(),
      ey.mutable_data_ptr<float>(),
      ez.mutable_data_ptr<float>(),
      linear_permittivity.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      chi3.mutable_data_ptr<float>(),
      static_cast<float>(dt),
      static_cast<float>(eps0),
      dynamic_curl.mutable_data_ptr<float>());
}

void check_matching_field(const torch::stable::Tensor& reference, const torch::stable::Tensor& value, const char* name) {
  check_float32_tensor(value, name);
  check_contiguous_tensor(value, name);
  check_same_cuda_device(reference, value, name);
  STD_TORCH_CHECK(value.sizes().equals(reference.sizes()), name, " must match field shape");
}

void check_field3d(const torch::stable::Tensor& value, const char* name) {
  check_float32_tensor(value, name);
  check_contiguous_tensor(value, name);
  STD_TORCH_CHECK(value.dim() == 3, name, " must be a 3D tensor");
}

void launch_kerr_curl(
    int component,
    torch::stable::Tensor dynamic_curl,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& linear_permittivity,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& chi3,
    double dt,
    double eps0) {
  check_field3d(dynamic_curl, "dynamic_curl");
  check_field3d(ex, "ex");
  check_field3d(ey, "ey");
  check_field3d(ez, "ez");
  check_same_cuda_device(dynamic_curl, ex, "ex");
  check_same_cuda_device(dynamic_curl, ey, "ey");
  check_same_cuda_device(dynamic_curl, ez, "ez");
  check_matching_field(dynamic_curl, linear_permittivity, "linear_permittivity");
  check_matching_field(dynamic_curl, decay, "decay");
  check_matching_field(dynamic_curl, chi3, "chi3");
  STD_TORCH_CHECK(component >= 0 && component < 3, "component must be in [0, 3)");
  if (component == 0) {
    STD_TORCH_CHECK(ex.sizes().equals(dynamic_curl.sizes()), "ex must match dynamic_curl shape");
  } else if (component == 1) {
    STD_TORCH_CHECK(ey.sizes().equals(dynamic_curl.sizes()), "ey must match dynamic_curl shape");
  } else {
    STD_TORCH_CHECK(ez.sizes().equals(dynamic_curl.sizes()), "ez must match dynamic_curl shape");
  }
  torch::stable::accelerator::DeviceGuard guard(dynamic_curl.get_device_index());
  if (component == 0) {
    launch_kerr_curl_kernel<0>(dynamic_curl, ex, ey, ez, linear_permittivity, decay, chi3, dt, eps0);
  } else if (component == 1) {
    launch_kerr_curl_kernel<1>(dynamic_curl, ex, ey, ez, linear_permittivity, decay, chi3, dt, eps0);
  } else {
    launch_kerr_curl_kernel<2>(dynamic_curl, ex, ey, ez, linear_permittivity, decay, chi3, dt, eps0);
  }
  WITWIN_CUDA_CHECK();
}

template <int Component>
void launch_nonlinear_coefficients_kernel(
    torch::stable::Tensor dynamic_decay,
    torch::stable::Tensor dynamic_curl,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& linear_permittivity,
    const torch::stable::Tensor& external_decay,
    const torch::stable::Tensor& sigma_static,
    const torch::stable::Tensor& chi2,
    const torch::stable::Tensor& chi3,
    const torch::stable::Tensor& tpa_sigma,
    double dt,
    double eps0) {
  const dim3 block = field_block3d();
  update_nonlinear_coefficients_kernel<Component><<<field_grid3d(dynamic_curl.size(0), dynamic_curl.size(1), dynamic_curl.size(2), block), block, 0, current_cuda_stream()>>>(
      dynamic_curl.size(0),
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
      ex.mutable_data_ptr<float>(),
      ey.mutable_data_ptr<float>(),
      ez.mutable_data_ptr<float>(),
      linear_permittivity.mutable_data_ptr<float>(),
      external_decay.mutable_data_ptr<float>(),
      sigma_static.mutable_data_ptr<float>(),
      chi2.mutable_data_ptr<float>(),
      chi3.mutable_data_ptr<float>(),
      tpa_sigma.mutable_data_ptr<float>(),
      static_cast<float>(dt),
      static_cast<float>(eps0),
      dynamic_decay.mutable_data_ptr<float>(),
      dynamic_curl.mutable_data_ptr<float>());
}

void launch_nonlinear_coefficients(
    int component,
    torch::stable::Tensor dynamic_decay,
    torch::stable::Tensor dynamic_curl,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& linear_permittivity,
    const torch::stable::Tensor& external_decay,
    const torch::stable::Tensor& sigma_static,
    const torch::stable::Tensor& chi2,
    const torch::stable::Tensor& chi3,
    const torch::stable::Tensor& tpa_sigma,
    double dt,
    double eps0) {
  check_field3d(dynamic_curl, "dynamic_curl");
  check_field3d(ex, "ex");
  check_field3d(ey, "ey");
  check_field3d(ez, "ez");
  check_same_cuda_device(dynamic_curl, ex, "ex");
  check_same_cuda_device(dynamic_curl, ey, "ey");
  check_same_cuda_device(dynamic_curl, ez, "ez");
  check_matching_field(dynamic_curl, dynamic_decay, "dynamic_decay");
  check_matching_field(dynamic_curl, linear_permittivity, "linear_permittivity");
  check_matching_field(dynamic_curl, external_decay, "external_decay");
  check_matching_field(dynamic_curl, sigma_static, "sigma_static");
  check_matching_field(dynamic_curl, chi2, "chi2");
  check_matching_field(dynamic_curl, chi3, "chi3");
  check_matching_field(dynamic_curl, tpa_sigma, "tpa_sigma");
  STD_TORCH_CHECK(component >= 0 && component < 3, "component must be in [0, 3)");
  if (component == 0) {
    STD_TORCH_CHECK(ex.sizes().equals(dynamic_curl.sizes()), "ex must match dynamic_curl shape");
  } else if (component == 1) {
    STD_TORCH_CHECK(ey.sizes().equals(dynamic_curl.sizes()), "ey must match dynamic_curl shape");
  } else {
    STD_TORCH_CHECK(ez.sizes().equals(dynamic_curl.sizes()), "ez must match dynamic_curl shape");
  }
  torch::stable::accelerator::DeviceGuard guard(dynamic_curl.get_device_index());
  if (component == 0) {
    launch_nonlinear_coefficients_kernel<0>(
        dynamic_decay, dynamic_curl, ex, ey, ez, linear_permittivity, external_decay,
        sigma_static, chi2, chi3, tpa_sigma, dt, eps0);
  } else if (component == 1) {
    launch_nonlinear_coefficients_kernel<1>(
        dynamic_decay, dynamic_curl, ex, ey, ez, linear_permittivity, external_decay,
        sigma_static, chi2, chi3, tpa_sigma, dt, eps0);
  } else {
    launch_nonlinear_coefficients_kernel<2>(
        dynamic_decay, dynamic_curl, ex, ey, ez, linear_permittivity, external_decay,
        sigma_static, chi2, chi3, tpa_sigma, dt, eps0);
  }
  WITWIN_CUDA_CHECK();
}

}  // namespace

void update_debye_current_cuda(
    const torch::stable::Tensor& electric,
    torch::stable::Tensor polarization,
    torch::stable::Tensor current,
    const torch::stable::Tensor& drive,
    double decay,
    double dt) {
  check_float32_tensor(electric, "electric");
  check_contiguous_tensor(electric, "electric");
  check_matching_field(electric, polarization, "polarization");
  check_matching_field(electric, current, "current");
  check_matching_field(electric, drive, "drive");
  torch::stable::accelerator::DeviceGuard guard(electric.get_device_index());
  update_debye_kernel<<<linear_grid(electric.numel()), 256, 0, current_cuda_stream()>>>(
      electric.numel(),
      electric.mutable_data_ptr<float>(),
      drive.mutable_data_ptr<float>(),
      static_cast<float>(decay),
      static_cast<float>(1.0 / dt),
      polarization.mutable_data_ptr<float>(),
      current.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_drude_current_cuda(
    const torch::stable::Tensor& electric,
    torch::stable::Tensor current,
    const torch::stable::Tensor& drive,
    double decay) {
  check_float32_tensor(electric, "electric");
  check_contiguous_tensor(electric, "electric");
  check_matching_field(electric, current, "current");
  check_matching_field(electric, drive, "drive");
  torch::stable::accelerator::DeviceGuard guard(electric.get_device_index());
  update_drude_kernel<<<linear_grid(electric.numel()), 256, 0, current_cuda_stream()>>>(
      electric.numel(),
      electric.mutable_data_ptr<float>(),
      drive.mutable_data_ptr<float>(),
      static_cast<float>(decay),
      current.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_lorentz_current_cuda(
    const torch::stable::Tensor& electric,
    torch::stable::Tensor polarization,
    torch::stable::Tensor current,
    const torch::stable::Tensor& drive,
    double decay,
    double restoring,
    double dt) {
  check_float32_tensor(electric, "electric");
  check_contiguous_tensor(electric, "electric");
  check_matching_field(electric, polarization, "polarization");
  check_matching_field(electric, current, "current");
  check_matching_field(electric, drive, "drive");
  torch::stable::accelerator::DeviceGuard guard(electric.get_device_index());
  update_lorentz_kernel<<<linear_grid(electric.numel()), 256, 0, current_cuda_stream()>>>(
      electric.numel(),
      electric.mutable_data_ptr<float>(),
      drive.mutable_data_ptr<float>(),
      static_cast<float>(decay),
      static_cast<float>(restoring),
      static_cast<float>(dt),
      polarization.mutable_data_ptr<float>(),
      current.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void apply_polarization_current_cuda(
    torch::stable::Tensor electric,
    const torch::stable::Tensor& current,
    const torch::stable::Tensor& inv_permittivity,
    double dt) {
  check_float32_tensor(electric, "electric");
  check_contiguous_tensor(electric, "electric");
  check_matching_field(electric, current, "current");
  check_matching_field(electric, inv_permittivity, "inv_permittivity");
  torch::stable::accelerator::DeviceGuard guard(electric.get_device_index());
  apply_polarization_kernel<<<linear_grid(electric.numel()), 256, 0, current_cuda_stream()>>>(
      electric.numel(),
      current.mutable_data_ptr<float>(),
      inv_permittivity.mutable_data_ptr<float>(),
      static_cast<float>(dt),
      electric.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void apply_polarization_current_modulated_cuda(
    torch::stable::Tensor electric,
    const torch::stable::Tensor& current,
    const torch::stable::Tensor& inv_permittivity,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_next,
    double dt) {
  check_float32_tensor(electric, "electric");
  check_contiguous_tensor(electric, "electric");
  check_matching_field(electric, current, "current");
  check_matching_field(electric, inv_permittivity, "inv_permittivity");
  check_matching_field(electric, mod_cos, "mod_cos");
  check_matching_field(electric, mod_sin, "mod_sin");
  check_matching_field(electric, mod_omega, "mod_omega");
  torch::stable::accelerator::DeviceGuard guard(electric.get_device_index());
  apply_polarization_modulated_kernel<<<linear_grid(electric.numel()), 256, 0, current_cuda_stream()>>>(
      electric.numel(),
      current.mutable_data_ptr<float>(),
      inv_permittivity.mutable_data_ptr<float>(),
      mod_cos.mutable_data_ptr<float>(),
      mod_sin.mutable_data_ptr<float>(),
      mod_omega.mutable_data_ptr<float>(),
      static_cast<float>(t_next),
      static_cast<float>(dt),
      electric.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_kerr_ex_curl_cuda(
    torch::stable::Tensor dynamic_curl,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& linear_permittivity,
    const torch::stable::Tensor& ex_decay,
    const torch::stable::Tensor& chi3,
    double dt,
    double eps0) {
  launch_kerr_curl(0, dynamic_curl, ex, ey, ez, linear_permittivity, ex_decay, chi3, dt, eps0);
}

void update_kerr_ey_curl_cuda(
    torch::stable::Tensor dynamic_curl,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& linear_permittivity,
    const torch::stable::Tensor& ey_decay,
    const torch::stable::Tensor& chi3,
    double dt,
    double eps0) {
  launch_kerr_curl(1, dynamic_curl, ex, ey, ez, linear_permittivity, ey_decay, chi3, dt, eps0);
}

void update_kerr_ez_curl_cuda(
    torch::stable::Tensor dynamic_curl,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& linear_permittivity,
    const torch::stable::Tensor& ez_decay,
    const torch::stable::Tensor& chi3,
    double dt,
    double eps0) {
  launch_kerr_curl(2, dynamic_curl, ex, ey, ez, linear_permittivity, ez_decay, chi3, dt, eps0);
}

void update_nonlinear_coefficients_cuda(
    torch::stable::Tensor dynamic_decay,
    torch::stable::Tensor dynamic_curl,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& linear_permittivity,
    const torch::stable::Tensor& external_decay,
    const torch::stable::Tensor& sigma_static,
    const torch::stable::Tensor& chi2,
    const torch::stable::Tensor& chi3,
    const torch::stable::Tensor& tpa_sigma,
    int64_t component,
    double dt,
    double eps0) {
  launch_nonlinear_coefficients(
      static_cast<int>(component), dynamic_decay, dynamic_curl, ex, ey, ez,
      linear_permittivity, external_decay, sigma_static, chi2, chi3, tpa_sigma, dt, eps0);
}
