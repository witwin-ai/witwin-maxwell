#include <type_traits>

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

template <int Component>
__global__ void update_magnetic_standard_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    unsigned int local_x_begin,
    unsigned int local_x_end,
    const float* __restrict__ first,
    const float* __restrict__ second,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_a,
    const float* __restrict__ inv_b,
    float* __restrict__ field) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = local_x_begin + blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= local_x_end || i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  float positive;
  float negative;
  if constexpr (Component == 0) {
    positive = (second[offset3d(i, j + 1, k, ny + 1, nz)] - second[offset3d(i, j, k, ny + 1, nz)]) * inv_a[j];
    negative = (first[offset3d(i, j, k + 1, ny, nz + 1)] - first[offset3d(i, j, k, ny, nz + 1)]) * inv_b[k];
  } else if constexpr (Component == 1) {
    positive = (first[offset3d(i, j, k + 1, ny, nz + 1)] - first[offset3d(i, j, k, ny, nz + 1)]) * inv_b[k];
    negative = (second[offset3d(i + 1, j, k, ny, nz)] - second[offset3d(i, j, k, ny, nz)]) * inv_a[i];
  } else {
    positive = (second[offset3d(i + 1, j, k, ny, nz)] - second[offset3d(i, j, k, ny, nz)]) * inv_a[i];
    negative = (first[offset3d(i, j + 1, k, ny + 1, nz)] - first[offset3d(i, j, k, ny + 1, nz)]) * inv_b[j];
  }
  field[linear] = field[linear] * decay[linear] - curl_coeff[linear] * (positive - negative);
}

template <int Component>
__global__ void update_magnetic_cpml_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ first,
    const float* __restrict__ second,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    const float* __restrict__ inv_kappa_a,
    const float* __restrict__ b_a,
    const float* __restrict__ c_a,
    const float* __restrict__ inv_kappa_b,
    const float* __restrict__ b_b,
    const float* __restrict__ c_b,
    const float* __restrict__ inv_a,
    const float* __restrict__ inv_b,
    float* __restrict__ psi_a,
    float* __restrict__ psi_b,
    float* __restrict__ field) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  float d_a;
  float d_b;
  if constexpr (Component == 0) {
    d_a = (second[offset3d(i, j + 1, k, ny + 1, nz)] - second[offset3d(i, j, k, ny + 1, nz)]) * inv_a[j];
    d_b = (first[offset3d(i, j, k + 1, ny, nz + 1)] - first[offset3d(i, j, k, ny, nz + 1)]) * inv_b[k];
  } else if constexpr (Component == 1) {
    d_a = (second[offset3d(i + 1, j, k, ny, nz)] - second[offset3d(i, j, k, ny, nz)]) * inv_a[i];
    d_b = (first[offset3d(i, j, k + 1, ny, nz + 1)] - first[offset3d(i, j, k, ny, nz + 1)]) * inv_b[k];
  } else {
    d_a = (second[offset3d(i + 1, j, k, ny, nz)] - second[offset3d(i, j, k, ny, nz)]) * inv_a[i];
    d_b = (first[offset3d(i, j + 1, k, ny + 1, nz)] - first[offset3d(i, j, k, ny + 1, nz)]) * inv_b[j];
  }
  const unsigned int coord_a = Component == 0 ? j : i;
  const unsigned int coord_b = Component == 2 ? j : k;
  const float psi_a_value = b_a[coord_a] * psi_a[linear] + c_a[coord_a] * d_a;
  const float psi_b_value = b_b[coord_b] * psi_b[linear] + c_b[coord_b] * d_b;
  psi_a[linear] = psi_a_value;
  psi_b[linear] = psi_b_value;
  const float corrected_a = d_a * inv_kappa_a[coord_a] + psi_a_value;
  const float corrected_b = d_b * inv_kappa_b[coord_b] + psi_b_value;
  const float curl = Component == 1 ? corrected_b - corrected_a : corrected_a - corrected_b;
  field[linear] = field[linear] * decay[linear] - curl_coeff[linear] * curl;
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

template <int Component, bool UniformDecay, bool UniformCurl>
__global__ void update_magnetic_cpml_compressed_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ first,
    const float* __restrict__ second,
    const float* __restrict__ decay,
    const float* __restrict__ curl_coeff,
    float decay_value,
    float curl_value,
    const float* __restrict__ inv_kappa_a,
    const float* __restrict__ b_a,
    const float* __restrict__ c_a,
    const float* __restrict__ inv_kappa_b,
    const float* __restrict__ b_b,
    const float* __restrict__ c_b,
    const float* __restrict__ inv_a,
    const float* __restrict__ inv_b,
    int a_low_length,
    int a_high_start,
    int a_high_length,
    int b_low_length,
    int b_high_start,
    int b_high_length,
    float* __restrict__ psi_a,
    float* __restrict__ psi_b,
    float* __restrict__ field) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  float d_a;
  float d_b;
  if constexpr (Component == 0) {
    d_a = (second[offset3d(i, j + 1, k, ny + 1, nz)] - second[offset3d(i, j, k, ny + 1, nz)]) * inv_a[j];
    d_b = (first[offset3d(i, j, k + 1, ny, nz + 1)] - first[offset3d(i, j, k, ny, nz + 1)]) * inv_b[k];
  } else if constexpr (Component == 1) {
    d_a = (second[offset3d(i + 1, j, k, ny, nz)] - second[offset3d(i, j, k, ny, nz)]) * inv_a[i];
    d_b = (first[offset3d(i, j, k + 1, ny, nz + 1)] - first[offset3d(i, j, k, ny, nz + 1)]) * inv_b[k];
  } else {
    d_a = (second[offset3d(i + 1, j, k, ny, nz)] - second[offset3d(i, j, k, ny, nz)]) * inv_a[i];
    d_b = (first[offset3d(i, j + 1, k, ny + 1, nz)] - first[offset3d(i, j, k, ny + 1, nz)]) * inv_b[j];
  }
  const unsigned int coord_a = Component == 0 ? j : i;
  const unsigned int coord_b = Component == 2 ? j : k;
  float psi_a_value;
  float psi_b_value;
  if constexpr (Component == 0) {
    psi_a_value = update_compact_magnetic_psi<1>(psi_a, b_a, c_a, i, j, k, ny, nz, coord_a, a_low_length, a_high_start, a_high_length, d_a);
    psi_b_value = update_compact_magnetic_psi<2>(psi_b, b_b, c_b, i, j, k, ny, nz, coord_b, b_low_length, b_high_start, b_high_length, d_b);
  } else if constexpr (Component == 1) {
    psi_a_value = update_compact_magnetic_psi<0>(psi_a, b_a, c_a, i, j, k, ny, nz, coord_a, a_low_length, a_high_start, a_high_length, d_a);
    psi_b_value = update_compact_magnetic_psi<2>(psi_b, b_b, c_b, i, j, k, ny, nz, coord_b, b_low_length, b_high_start, b_high_length, d_b);
  } else {
    psi_a_value = update_compact_magnetic_psi<0>(psi_a, b_a, c_a, i, j, k, ny, nz, coord_a, a_low_length, a_high_start, a_high_length, d_a);
    psi_b_value = update_compact_magnetic_psi<1>(psi_b, b_b, c_b, i, j, k, ny, nz, coord_b, b_low_length, b_high_start, b_high_length, d_b);
  }
  const float corrected_a = d_a * inv_kappa_a[coord_a] + psi_a_value;
  const float corrected_b = d_b * inv_kappa_b[coord_b] + psi_b_value;
  const float curl = Component == 1 ? corrected_b - corrected_a : corrected_a - corrected_b;
  const float decay_factor = UniformDecay ? decay_value : decay[linear];
  const float curl_factor = UniformCurl ? curl_value : curl_coeff[linear];
  field[linear] = field[linear] * decay_factor - curl_factor * curl;
}

void check_magnetic_inputs(
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

void check_magnetic_cpml_inputs(
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
  check_magnetic_inputs(field, first, second, decay, curl, name);
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

void check_magnetic_cpml_compressed_inputs(
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

void update_magnetic_hx_standard_bounded_cuda(
    torch::stable::Tensor hx,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t local_x_begin,
    int64_t local_x_end,
    int64_t global_x_offset,
    int64_t global_x_extent) {
  check_magnetic_inputs(hx, ey, ez, decay, curl, "hx");
  check_rank3_shape(ey, "ey", hx.size(0), hx.size(1), hx.size(2) + 1);
  check_rank3_shape(ez, "ez", hx.size(0), hx.size(1) + 1, hx.size(2));
  check_spacing_vector(hx, inv_dy, 1, "inv_dy");
  check_spacing_vector(hx, inv_dz, 2, "inv_dz");
  check_bounded_x_launch(
      hx, local_x_begin, local_x_end, global_x_offset, global_x_extent, "hx");
  if (local_x_begin == local_x_end) {
    return;
  }
  torch::stable::accelerator::DeviceGuard guard(hx.get_device_index());
  const auto sizes = hx.sizes();
  const dim3 block = field_block3d();
  update_magnetic_standard_kernel<0><<<field_grid3d(local_x_end - local_x_begin, sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      static_cast<unsigned int>(local_x_begin),
      static_cast<unsigned int>(local_x_end),
      ey.mutable_data_ptr<float>(),
      ez.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      hx.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_magnetic_hx_standard_cuda(
    torch::stable::Tensor hx,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz) {
  const int64_t x_extent = hx.size(0);
  update_magnetic_hx_standard_bounded_cuda(
      hx,
      ey,
      ez,
      decay,
      curl,
      inv_dy,
      inv_dz,
      0,
      x_extent,
      0,
      x_extent);
}

void update_magnetic_hy_standard_bounded_cuda(
    torch::stable::Tensor hy,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
    int64_t local_x_begin,
    int64_t local_x_end,
    int64_t global_x_offset,
    int64_t global_x_extent) {
  check_magnetic_inputs(hy, ex, ez, decay, curl, "hy");
  check_rank3_shape(ex, "ex", hy.size(0), hy.size(1), hy.size(2) + 1);
  check_rank3_shape(ez, "ez", hy.size(0) + 1, hy.size(1), hy.size(2));
  check_spacing_vector(hy, inv_dx, 0, "inv_dx");
  check_spacing_vector(hy, inv_dz, 2, "inv_dz");
  check_bounded_x_launch(
      hy, local_x_begin, local_x_end, global_x_offset, global_x_extent, "hy");
  if (local_x_begin == local_x_end) {
    return;
  }
  torch::stable::accelerator::DeviceGuard guard(hy.get_device_index());
  const auto sizes = hy.sizes();
  const dim3 block = field_block3d();
  update_magnetic_standard_kernel<1><<<field_grid3d(local_x_end - local_x_begin, sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      static_cast<unsigned int>(local_x_begin),
      static_cast<unsigned int>(local_x_end),
      ex.mutable_data_ptr<float>(),
      ez.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      inv_dx.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      hy.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_magnetic_hy_standard_cuda(
    torch::stable::Tensor hy,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz) {
  const int64_t x_extent = hy.size(0);
  update_magnetic_hy_standard_bounded_cuda(
      hy, ex, ez, decay, curl, inv_dx, inv_dz, 0, x_extent, 0, x_extent);
}

void update_magnetic_hz_standard_bounded_cuda(
    torch::stable::Tensor hz,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    int64_t local_x_begin,
    int64_t local_x_end,
    int64_t global_x_offset,
    int64_t global_x_extent) {
  check_magnetic_inputs(hz, ex, ey, decay, curl, "hz");
  check_rank3_shape(ex, "ex", hz.size(0), hz.size(1) + 1, hz.size(2));
  check_rank3_shape(ey, "ey", hz.size(0) + 1, hz.size(1), hz.size(2));
  check_spacing_vector(hz, inv_dx, 0, "inv_dx");
  check_spacing_vector(hz, inv_dy, 1, "inv_dy");
  check_bounded_x_launch(
      hz, local_x_begin, local_x_end, global_x_offset, global_x_extent, "hz");
  if (local_x_begin == local_x_end) {
    return;
  }
  torch::stable::accelerator::DeviceGuard guard(hz.get_device_index());
  const auto sizes = hz.sizes();
  const dim3 block = field_block3d();
  update_magnetic_standard_kernel<2><<<field_grid3d(local_x_end - local_x_begin, sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      static_cast<unsigned int>(local_x_begin),
      static_cast<unsigned int>(local_x_end),
      ex.mutable_data_ptr<float>(),
      ey.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>(),
      inv_dx.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>(),
      hz.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_magnetic_hz_standard_cuda(
    torch::stable::Tensor hz,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy) {
  const int64_t x_extent = hz.size(0);
  update_magnetic_hz_standard_bounded_cuda(
      hz, ex, ey, decay, curl, inv_dx, inv_dy, 0, x_extent, 0, x_extent);
}

void update_magnetic_hx_cpml_cuda(
    torch::stable::Tensor hx,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
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
    const torch::stable::Tensor& inv_dz) {
  check_magnetic_cpml_inputs(
      hx, ey, ez, decay, curl, psi_y, psi_z, inv_kappa_y, b_y, c_y, inv_kappa_z, b_z, c_z, 1, 2, "hx");
  check_rank3_shape(ey, "ey", hx.size(0), hx.size(1), hx.size(2) + 1);
  check_rank3_shape(ez, "ez", hx.size(0), hx.size(1) + 1, hx.size(2));
  check_spacing_vector(hx, inv_dy, 1, "inv_dy");
  check_spacing_vector(hx, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(hx.get_device_index());
  const auto sizes = hx.sizes();
  const dim3 block = field_block3d();
  update_magnetic_cpml_kernel<0><<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      ey.mutable_data_ptr<float>(),
      ez.mutable_data_ptr<float>(),
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
      hx.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_magnetic_hy_cpml_cuda(
    torch::stable::Tensor hy,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ez,
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
    const torch::stable::Tensor& inv_dz) {
  check_magnetic_cpml_inputs(
      hy, ex, ez, decay, curl, psi_x, psi_z, inv_kappa_x, b_x, c_x, inv_kappa_z, b_z, c_z, 0, 2, "hy");
  check_rank3_shape(ex, "ex", hy.size(0), hy.size(1), hy.size(2) + 1);
  check_rank3_shape(ez, "ez", hy.size(0) + 1, hy.size(1), hy.size(2));
  check_spacing_vector(hy, inv_dx, 0, "inv_dx");
  check_spacing_vector(hy, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(hy.get_device_index());
  const auto sizes = hy.sizes();
  const dim3 block = field_block3d();
  update_magnetic_cpml_kernel<1><<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      ex.mutable_data_ptr<float>(),
      ez.mutable_data_ptr<float>(),
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
      hy.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_magnetic_hz_cpml_cuda(
    torch::stable::Tensor hz,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
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
    const torch::stable::Tensor& inv_dy) {
  check_magnetic_cpml_inputs(
      hz, ex, ey, decay, curl, psi_x, psi_y, inv_kappa_x, b_x, c_x, inv_kappa_y, b_y, c_y, 0, 1, "hz");
  check_rank3_shape(ex, "ex", hz.size(0), hz.size(1) + 1, hz.size(2));
  check_rank3_shape(ey, "ey", hz.size(0) + 1, hz.size(1), hz.size(2));
  check_spacing_vector(hz, inv_dx, 0, "inv_dx");
  check_spacing_vector(hz, inv_dy, 1, "inv_dy");
  torch::stable::accelerator::DeviceGuard guard(hz.get_device_index());
  const auto sizes = hz.sizes();
  const dim3 block = field_block3d();
  update_magnetic_cpml_kernel<2><<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(sizes[0]),
      static_cast<unsigned int>(sizes[1]),
      static_cast<unsigned int>(sizes[2]),
      ex.mutable_data_ptr<float>(),
      ey.mutable_data_ptr<float>(),
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
      hz.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_magnetic_hx_cpml_compressed_cuda(
    torch::stable::Tensor hx,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
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
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length,
    std::optional<double> uniform_decay,
    std::optional<double> uniform_curl) {
  check_magnetic_cpml_compressed_inputs(
      hx, ey, ez, decay, curl, psi_y, psi_z, inv_kappa_y, b_y, c_y, inv_kappa_z, b_z, c_z,
      1, 2, y_low_length, y_high_length, z_low_length, z_high_length, "hx");
  check_rank3_shape(ey, "ey", hx.size(0), hx.size(1), hx.size(2) + 1);
  check_rank3_shape(ez, "ez", hx.size(0), hx.size(1) + 1, hx.size(2));
  check_spacing_vector(hx, inv_dy, 1, "inv_dy");
  check_spacing_vector(hx, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(hx.get_device_index());
  const auto sizes = hx.sizes();
  const dim3 block = field_block3d();
  dispatch_uniform_coefficients(uniform_decay.has_value(), uniform_curl.has_value(), [&](auto u_decay, auto u_curl) {
    update_magnetic_cpml_compressed_kernel<0, decltype(u_decay)::value, decltype(u_curl)::value><<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
        static_cast<unsigned int>(sizes[0]),
        static_cast<unsigned int>(sizes[1]),
        static_cast<unsigned int>(sizes[2]),
        ey.mutable_data_ptr<float>(),
        ez.mutable_data_ptr<float>(),
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
        static_cast<int>(y_low_length),
        static_cast<int>(y_high_start),
        static_cast<int>(y_high_length),
        static_cast<int>(z_low_length),
        static_cast<int>(z_high_start),
        static_cast<int>(z_high_length),
        psi_y.mutable_data_ptr<float>(),
        psi_z.mutable_data_ptr<float>(),
        hx.mutable_data_ptr<float>());
  });
  WITWIN_CUDA_CHECK();
}

void update_magnetic_hy_cpml_compressed_cuda(
    torch::stable::Tensor hy,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ez,
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
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length,
    std::optional<double> uniform_decay,
    std::optional<double> uniform_curl) {
  check_magnetic_cpml_compressed_inputs(
      hy, ex, ez, decay, curl, psi_x, psi_z, inv_kappa_x, b_x, c_x, inv_kappa_z, b_z, c_z,
      0, 2, x_low_length, x_high_length, z_low_length, z_high_length, "hy");
  check_rank3_shape(ex, "ex", hy.size(0), hy.size(1), hy.size(2) + 1);
  check_rank3_shape(ez, "ez", hy.size(0) + 1, hy.size(1), hy.size(2));
  check_spacing_vector(hy, inv_dx, 0, "inv_dx");
  check_spacing_vector(hy, inv_dz, 2, "inv_dz");
  torch::stable::accelerator::DeviceGuard guard(hy.get_device_index());
  const auto sizes = hy.sizes();
  const dim3 block = field_block3d();
  dispatch_uniform_coefficients(uniform_decay.has_value(), uniform_curl.has_value(), [&](auto u_decay, auto u_curl) {
    update_magnetic_cpml_compressed_kernel<1, decltype(u_decay)::value, decltype(u_curl)::value><<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
        static_cast<unsigned int>(sizes[0]),
        static_cast<unsigned int>(sizes[1]),
        static_cast<unsigned int>(sizes[2]),
        ex.mutable_data_ptr<float>(),
        ez.mutable_data_ptr<float>(),
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
        static_cast<int>(x_low_length),
        static_cast<int>(x_high_start),
        static_cast<int>(x_high_length),
        static_cast<int>(z_low_length),
        static_cast<int>(z_high_start),
        static_cast<int>(z_high_length),
        psi_x.mutable_data_ptr<float>(),
        psi_z.mutable_data_ptr<float>(),
        hy.mutable_data_ptr<float>());
  });
  WITWIN_CUDA_CHECK();
}

void update_magnetic_hz_cpml_compressed_cuda(
    torch::stable::Tensor hz,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
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
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length,
    std::optional<double> uniform_decay,
    std::optional<double> uniform_curl) {
  check_magnetic_cpml_compressed_inputs(
      hz, ex, ey, decay, curl, psi_x, psi_y, inv_kappa_x, b_x, c_x, inv_kappa_y, b_y, c_y,
      0, 1, x_low_length, x_high_length, y_low_length, y_high_length, "hz");
  check_rank3_shape(ex, "ex", hz.size(0), hz.size(1) + 1, hz.size(2));
  check_rank3_shape(ey, "ey", hz.size(0) + 1, hz.size(1), hz.size(2));
  check_spacing_vector(hz, inv_dx, 0, "inv_dx");
  check_spacing_vector(hz, inv_dy, 1, "inv_dy");
  torch::stable::accelerator::DeviceGuard guard(hz.get_device_index());
  const auto sizes = hz.sizes();
  const dim3 block = field_block3d();
  dispatch_uniform_coefficients(uniform_decay.has_value(), uniform_curl.has_value(), [&](auto u_decay, auto u_curl) {
    update_magnetic_cpml_compressed_kernel<2, decltype(u_decay)::value, decltype(u_curl)::value><<<field_grid3d(sizes[0], sizes[1], sizes[2], block), block, 0, current_cuda_stream()>>>(
        static_cast<unsigned int>(sizes[0]),
        static_cast<unsigned int>(sizes[1]),
        static_cast<unsigned int>(sizes[2]),
        ex.mutable_data_ptr<float>(),
        ey.mutable_data_ptr<float>(),
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
        static_cast<int>(x_low_length),
        static_cast<int>(x_high_start),
        static_cast<int>(x_high_length),
        static_cast<int>(y_low_length),
        static_cast<int>(y_high_start),
        static_cast<int>(y_high_length),
        psi_x.mutable_data_ptr<float>(),
        psi_y.mutable_data_ptr<float>(),
        hz.mutable_data_ptr<float>());
  });
  WITWIN_CUDA_CHECK();
}
