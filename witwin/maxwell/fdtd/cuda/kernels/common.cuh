#pragma once

#include <type_traits>
#include <cstdint>
#include <limits>
#include "../tensors.h"

struct FieldBatchLayout {
  int64_t count = 1;
  int64_t first_stride = 0;
  int64_t second_stride = 0;
  int64_t decay_stride = 0;
  int64_t curl_stride = 0;
  const int* active = nullptr;
};

inline FieldBatchLayout check_field_batch_inputs(
    const torch::stable::Tensor& field, const torch::stable::Tensor& first,
    const torch::stable::Tensor& second, const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl, const torch::stable::Tensor& inv_a,
    const torch::stable::Tensor& inv_b, const torch::stable::Tensor& active,
    int64_t component, bool electric) {
  STD_TORCH_CHECK(component >= 0 && component < 3, "component must be 0, 1 or 2");
  check_float32_tensor(field, "field");
  check_contiguous_tensor(field, "field");
  STD_TORCH_CHECK(field.dim() == 4 && field.size(0) > 0, "field must have shape [B, nx, ny, nz]");
  for (int axis = 1; axis < 4; ++axis) {
    STD_TORCH_CHECK(field.size(axis) >= 2 && field.size(axis) <= std::numeric_limits<unsigned int>::max(),
                    "batch spatial extents must be in [2, UINT_MAX]");
  }
  STD_TORCH_CHECK((field.numel() - 1) / 256 + 1 <= 2147483647LL, "batch launch exceeds CUDA grid.x limit");
  const auto check_read = [&](const torch::stable::Tensor& value, const char* name) {
    check_float32_tensor(value, name);
    check_contiguous_tensor(value, name);
    check_same_cuda_device(field, value, name);
    const auto begin = reinterpret_cast<uintptr_t>(field.mutable_data_ptr<float>());
    const auto other = reinterpret_cast<uintptr_t>(value.mutable_data_ptr<float>());
    STD_TORCH_CHECK(begin + field.numel() * sizeof(float) <= other ||
                    other + value.numel() * sizeof(float) <= begin,
                    name, " must not overlap the writable field");
  };
  const int direction = electric ? -1 : 1;
  const int first_axis = component == 2 ? 1 : 2;
  const int second_axis = component == 0 ? 1 : 0;
  const auto check_source = [&](const torch::stable::Tensor& value, int changed_axis, const char* name) {
    check_read(value, name);
    STD_TORCH_CHECK(value.dim() == 4 && value.size(0) == field.size(0), name, " batch size mismatch");
    for (int axis = 0; axis < 3; ++axis) {
      STD_TORCH_CHECK(value.size(axis + 1) == field.size(axis + 1) + (axis == changed_axis ? direction : 0),
                      name, " does not match its Yee component shape");
    }
  };
  check_source(first, first_axis, "first");
  check_source(second, second_axis, "second");
  const auto coefficient_stride = [&](const torch::stable::Tensor& value, const char* name) -> int64_t {
    check_read(value, name);
    STD_TORCH_CHECK(value.dim() == 3 || value.dim() == 4, name, " must be shared [nx,ny,nz] or per-lane [B,nx,ny,nz]");
    const int offset = value.dim() == 4 ? 1 : 0;
    if (offset) STD_TORCH_CHECK(value.size(0) == field.size(0), name, " batch size mismatch");
    for (int axis = 0; axis < 3; ++axis) {
      STD_TORCH_CHECK(value.size(axis + offset) == field.size(axis + 1), name, " spatial shape mismatch");
    }
    return offset ? value.numel() / field.size(0) : 0;
  };
  const int axis_a = component == 0 ? 1 : 0;
  const int axis_b = component == 2 ? 1 : 2;
  check_read(inv_a, "inv_a");
  check_read(inv_b, "inv_b");
  STD_TORCH_CHECK(inv_a.dim() == 1 && inv_a.size(0) == field.size(axis_a + 1), "inv_a shape mismatch");
  STD_TORCH_CHECK(inv_b.dim() == 1 && inv_b.size(0) == field.size(axis_b + 1), "inv_b shape mismatch");
  check_same_cuda_device(field, active, "active");
  check_contiguous_tensor(active, "active");
  STD_TORCH_CHECK(active.scalar_type() == torch::headeronly::ScalarType::Int &&
                  active.dim() == 1 && active.size(0) == field.size(0), "active must be int32 [B]");
  const auto field_address = reinterpret_cast<uintptr_t>(field.mutable_data_ptr<float>());
  const auto mask_address = reinterpret_cast<uintptr_t>(active.mutable_data_ptr<int>());
  STD_TORCH_CHECK(field_address + field.numel() * sizeof(float) <= mask_address ||
                  mask_address + active.numel() * sizeof(int) <= field_address,
                  "active must not overlap the writable field");
  return {field.size(0), first.numel() / field.size(0), second.numel() / field.size(0),
          coefficient_stride(decay, "decay"), coefficient_stride(curl, "curl"), active.mutable_data_ptr<int>()};
}

inline void check_cpml_batch_inputs(
    const torch::stable::Tensor& field, const torch::stable::Tensor& first,
    const torch::stable::Tensor& second, const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl, const torch::stable::Tensor& inv_a,
    const torch::stable::Tensor& inv_b, const torch::stable::Tensor& active,
    const torch::stable::Tensor& psi_a, const torch::stable::Tensor& psi_b,
    const torch::stable::Tensor& kappa_a, const torch::stable::Tensor& b_a,
    const torch::stable::Tensor& c_a, const torch::stable::Tensor& kappa_b,
    const torch::stable::Tensor& b_b, const torch::stable::Tensor& c_b,
    int64_t component, int64_t a_low, int64_t a_start, int64_t a_high,
    int64_t b_low, int64_t b_start, int64_t b_high) {
  const int axis_a = component == 0 ? 1 : 0;
  const int axis_b = component == 2 ? 1 : 2;
  const auto disjoint = [](const torch::stable::Tensor& x, const torch::stable::Tensor& y) {
    if (x.numel() == 0 || y.numel() == 0) return;
    const auto px = reinterpret_cast<uintptr_t>(x.data_ptr());
    const auto py = reinterpret_cast<uintptr_t>(y.data_ptr());
    STD_TORCH_CHECK(px + x.numel() * 4 <= py || py + y.numel() * 4 <= px,
                    "batch CPML writable storage must not overlap another input");
  };
  const auto profile = [&](const torch::stable::Tensor& v, int axis) {
    check_float32_tensor(v, "CPML profile");
    check_contiguous_tensor(v, "CPML profile");
    check_same_cuda_device(field, v, "CPML profile");
    STD_TORCH_CHECK(v.dim() == 1 && v.size(0) == field.size(axis + 1), "CPML profile shape mismatch");
    disjoint(field, v);
  };
  for (const auto& v : {kappa_a, b_a, c_a}) profile(v, axis_a);
  for (const auto& v : {kappa_b, b_b, c_b}) profile(v, axis_b);
  const auto memory = [&](const torch::stable::Tensor& psi, int axis,
                          int64_t low, int64_t start, int64_t high) {
    const auto extent = field.size(axis + 1);
    STD_TORCH_CHECK(low >= 0 && low <= extent && high >= 0 && high <= extent &&
                    start >= 0 && start <= extent && start + high <= extent &&
                    (high == 0 || start >= low), "invalid batch CPML slab layout");
    STD_TORCH_CHECK(extent <= std::numeric_limits<int>::max(), "CPML axis exceeds signed indexing range");
    check_float32_tensor(psi, "psi");
    check_contiguous_tensor(psi, "psi");
    check_same_cuda_device(field, psi, "psi");
    STD_TORCH_CHECK(psi.dim() == 4 && psi.size(0) == field.size(0), "psi must have shape [B, ...]");
    for (int d = 0; d < 3; ++d) {
      STD_TORCH_CHECK(psi.size(d + 1) == (d == axis ? low + high : field.size(d + 1)),
                      "batch CPML memory shape mismatch");
    }
    for (const auto& v : {field, first, second, decay, curl, inv_a, inv_b, active,
                          kappa_a, b_a, c_a, kappa_b, b_b, c_b}) disjoint(psi, v);
  };
  memory(psi_a, axis_a, a_low, a_start, a_high);
  memory(psi_b, axis_b, b_low, b_start, b_high);
  disjoint(psi_a, psi_b);
}

__device__ __forceinline__ bool batch_field_index(
    const FieldBatchLayout& batch, unsigned int nx, unsigned int ny, unsigned int nz,
    unsigned int& i, unsigned int& j, unsigned int& k, int64_t& lane) {
  const int64_t volume = static_cast<int64_t>(nx) * ny * nz;
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  lane = index / volume;
  if (lane >= batch.count || batch.active[lane] == 0) {
    return false;
  }
  const int64_t local = index - lane * volume;
  k = static_cast<unsigned int>(local % nz);
  j = static_cast<unsigned int>((local / nz) % ny);
  i = static_cast<unsigned int>(local / (static_cast<int64_t>(ny) * nz));
  return true;
}

struct Index3D {
  unsigned int i;
  unsigned int j;
  unsigned int k;
};

// Dispatch a kernel templated on a single Uniform flag. The scalar fast path is
// taken only when BOTH decay and curl are uniform, so the only instantiations
// ever launched are the all-scalar one and the all-array one; mixed
// scalar/array reads are avoided because they let the compiler re-associate the
// FMA contraction of the update and break bit-parity with the array kernel.
template <typename LaunchFn>
inline void dispatch_uniform_pair(bool uniform, LaunchFn&& launch) {
  if (uniform) {
    launch(std::integral_constant<bool, true>{});
  } else {
    launch(std::integral_constant<bool, false>{});
  }
}

// Dispatch a kernel templated on <UniformDecay, UniformCurl>. Uniform
// coefficients skip the full-volume decay/curl tensor reads, which removes up
// to two of the six global-memory streams of the field update.
template <typename LaunchFn>
inline void dispatch_uniform_coefficients(bool uniform_decay, bool uniform_curl, LaunchFn&& launch) {
  if (uniform_decay) {
    if (uniform_curl) {
      launch(std::integral_constant<bool, true>{}, std::integral_constant<bool, true>{});
    } else {
      launch(std::integral_constant<bool, true>{}, std::integral_constant<bool, false>{});
    }
  } else {
    if (uniform_curl) {
      launch(std::integral_constant<bool, false>{}, std::integral_constant<bool, true>{});
    } else {
      launch(std::integral_constant<bool, false>{}, std::integral_constant<bool, false>{});
    }
  }
}

constexpr int BOUNDARY_NONE = 0;
constexpr int BOUNDARY_PML = 1;
constexpr int BOUNDARY_PERIODIC = 2;
constexpr int BOUNDARY_BLOCH = 3;
constexpr int BOUNDARY_PEC = 4;
constexpr int BOUNDARY_PMC = 5;

__device__ __forceinline__ Index3D unflatten3d(unsigned int linear, unsigned int size_y, unsigned int size_z) {
  const unsigned int plane = size_y * size_z;
  const unsigned int i = linear / plane;
  const unsigned int remainder = linear - i * plane;
  const unsigned int j = remainder / size_z;
  const unsigned int k = remainder - j * size_z;
  return {i, j, k};
}

__device__ __forceinline__ long long offset3d(unsigned int i, unsigned int j, unsigned int k, unsigned int size_y, unsigned int size_z) {
  return (static_cast<long long>(i) * size_y + j) * size_z + k;
}

__device__ __forceinline__ int compact_local_index(
    unsigned int coord,
    int low_length,
    int high_start,
    int high_length) {
  if (coord < static_cast<unsigned int>(low_length)) {
    return static_cast<int>(coord);
  }
  if (coord >= static_cast<unsigned int>(high_start) &&
      coord < static_cast<unsigned int>(high_start + high_length)) {
    return low_length + static_cast<int>(coord) - high_start;
  }
  return -1;
}

template <int Axis>
__device__ __forceinline__ long long compact_offset3d_axis(
    unsigned int i,
    unsigned int j,
    unsigned int k,
    unsigned int size_y,
    unsigned int size_z,
    int local,
    unsigned int compact_length) {
  if constexpr (Axis == 0) {
    return offset3d(static_cast<unsigned int>(local), j, k, size_y, size_z);
  } else if constexpr (Axis == 1) {
    return offset3d(i, static_cast<unsigned int>(local), k, compact_length, size_z);
  } else {
    return offset3d(i, j, static_cast<unsigned int>(local), size_y, compact_length);
  }
}
