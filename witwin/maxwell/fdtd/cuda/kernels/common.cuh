#pragma once

#include <type_traits>

struct Index3D {
  unsigned int i;
  unsigned int j;
  unsigned int k;
};

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
