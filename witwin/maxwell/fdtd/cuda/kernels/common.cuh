#pragma once

struct Index3D {
  unsigned int i;
  unsigned int j;
  unsigned int k;
};

constexpr int BOUNDARY_NONE = 0;
constexpr int BOUNDARY_PML = 1;
constexpr int BOUNDARY_PERIODIC = 2;
constexpr int BOUNDARY_BLOCH = 3;
constexpr int BOUNDARY_PEC = 4;
constexpr int BOUNDARY_PMC = 5;

__device__ inline Index3D unflatten3d(unsigned int linear, unsigned int size_y, unsigned int size_z) {
  const unsigned int plane = size_y * size_z;
  const unsigned int i = linear / plane;
  const unsigned int remainder = linear - i * plane;
  const unsigned int j = remainder / size_z;
  const unsigned int k = remainder - j * size_z;
  return {i, j, k};
}

__device__ inline long long offset3d(unsigned int i, unsigned int j, unsigned int k, unsigned int size_y, unsigned int size_z) {
  return (static_cast<long long>(i) * size_y + j) * size_z + k;
}

__device__ inline int compact_local_index(
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

__device__ inline long long compact_offset3d(
    int axis,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    unsigned int size_y,
    unsigned int size_z,
    int local,
    unsigned int compact_length) {
  if (axis == 0) {
    return offset3d(static_cast<unsigned int>(local), j, k, size_y, size_z);
  }
  if (axis == 1) {
    return offset3d(i, static_cast<unsigned int>(local), k, compact_length, size_z);
  }
  return offset3d(i, j, static_cast<unsigned int>(local), size_y, compact_length);
}
