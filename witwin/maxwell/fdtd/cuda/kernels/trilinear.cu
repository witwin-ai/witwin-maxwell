#include "../launch.h"
#include "../tensors.h"

#include <climits>
#include <cmath>

namespace {
// Mirror ATen GridSampler.cuh and the single-channel 3D bilinear kernels.
// This unit relies on nvcc's default --fmad=true, without --use_fast_math.
// Keep the operand order and corner order when changing these expressions.
template <typename T>
__device__ __forceinline__ T source_index(T coord, int size) {
  coord = ((coord + 1.f) * size - 1) / 2;
  return ::min(static_cast<T>(size - 1), ::max(coord, static_cast<T>(0)));
}

template <typename T>
__device__ __forceinline__ T source_index_grad(T coord, int size, T* mult) {
  coord = ((coord + 1.f) * size - 1) / 2;
  *mult = static_cast<T>(size) / 2;
  if (coord <= static_cast<T>(0)) {
    *mult *= static_cast<T>(0);
    coord = static_cast<T>(0);
  } else if (coord >= static_cast<T>(size - 1)) {
    *mult *= static_cast<T>(0);
    coord = static_cast<T>(size - 1);
  }
  // Same integer-range guard as ATen (NaN queries are rejected by the caller).
  if (coord > INT_MAX - 1 || coord < INT_MIN || !::isfinite(static_cast<double>(coord)))
    coord = static_cast<T>(-100.0);
  return coord;
}

template <typename T>
struct Corners {
  int x, y, z;
  T ix, iy, iz;
  T weights[8];

  __device__ __forceinline__ Corners(T ix_, T iy_, T iz_) : ix(ix_), iy(iy_), iz(iz_) {
    x = static_cast<int>(::floor(ix));
    y = static_cast<int>(::floor(iy));
    z = static_cast<int>(::floor(iz));
    // tnw, tne, tsw, tse, bnw, bne, bsw, bse, exactly as in ATen.
    weights[0] = (x + 1 - ix) * (y + 1 - iy) * (z + 1 - iz);
    weights[1] = (ix - x)     * (y + 1 - iy) * (z + 1 - iz);
    weights[2] = (x + 1 - ix) * (iy - y)     * (z + 1 - iz);
    weights[3] = (ix - x)     * (iy - y)     * (z + 1 - iz);
    weights[4] = (x + 1 - ix) * (y + 1 - iy) * (iz - z);
    weights[5] = (ix - x)     * (y + 1 - iy) * (iz - z);
    weights[6] = (x + 1 - ix) * (iy - y)     * (iz - z);
    weights[7] = (ix - x)     * (iy - y)     * (iz - z);
  }

  __device__ __forceinline__ int64_t voxel(int corner, int d, int h, int w) const {
    const int cx = x + (corner & 1);
    const int cy = y + ((corner >> 1) & 1);
    const int cz = z + ((corner >> 2) & 1);
    if (cz >= 0 && cz < d && cy >= 0 && cy < h && cx >= 0 && cx < w)
      return (int64_t(cz) * h + cy) * w + cx;
    return int64_t(d) * h * w;
  }
};

template <typename T>
__global__ void forward_kernel(const T* texture, const T* query, T* out,
                               int64_t count, int d, int h, int w) {
  const int64_t q = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= count) return;
  const Corners<T> c(source_index(query[3*q], w), source_index(query[3*q+1], h),
                     source_index(query[3*q+2], d));
  T sum = 0;
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int64_t voxel = c.voxel(k, d, h, w);
    if (voxel < int64_t(d) * h * w) sum += texture[voxel] * c.weights[k];
  }
  out[q] = sum;
}

template <typename T>
__global__ void entries_kernel(const T* query, int64_t* voxels, T* weights,
                               int64_t count, int d, int h, int w) {
  const int64_t q = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= count) return;
  // The texture pullback uses ATen's backward source index (NaN and non-finite
  // coordinates map out of range and contribute nothing), as grid_sample's backward does.
  T mx, my, mz;
  const Corners<T> c(source_index_grad(query[3*q], w, &mx), source_index_grad(query[3*q+1], h, &my),
                     source_index_grad(query[3*q+2], d, &mz));
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    voxels[8*q+k] = c.voxel(k, d, h, w);
    weights[8*q+k] = c.weights[k];
  }
}

__device__ __forceinline__ float mul_rn(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ double mul_rn(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ float add_rn(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ double add_rn(double a, double b) { return __dadd_rn(a, b); }

template <typename T>
__global__ void backward_kernel(const int64_t* rows, const int64_t* queries,
                                const T* weights, const T* grad_out,
                                T* grad_texture, int64_t count) {
  const int64_t voxel = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (voxel >= count) return;
  T sum = 0;
  // One writer per voxel, sequential ascending query order, no float atomics.
  // Each term is rounded on its own before the add (no FMA contraction), as ATen
  // adds round(w * g) per corner: voxels with at most two contributions then equal
  // ATen bitwise, and larger rows sum ATen's own terms in ascending query order.
  for (int64_t k = rows[voxel]; k < rows[voxel+1]; ++k)
    sum = add_rn(sum, mul_rn(weights[k], grad_out[queries[k]]));
  grad_texture[voxel] = sum;
}

template <typename T>
__global__ void grid_backward_kernel(const T* texture, const T* query,
    const T* grad_out, T* grad_grid, int64_t count, int d, int h, int w) {
  const int64_t q = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= count) return;
  T mx, my, mz;
  const T ix = source_index_grad(query[3*q], w, &mx);
  const T iy = source_index_grad(query[3*q+1], h, &my);
  const T iz = source_index_grad(query[3*q+2], d, &mz);
  const Corners<T> c(ix, iy, iz);
  const int x = c.x, y = c.y, z = c.z;
  const T go = grad_out[q];
  T gx = 0, gy = 0, gz = 0;
  int64_t v = c.voxel(0, d, h, w);
  if (v < int64_t(d)*h*w) {
    const T value = texture[v];
    gx -= value * (y + 1 - iy) * (z + 1 - iz) * go;
    gy -= value * (x + 1 - ix) * (z + 1 - iz) * go;
    gz -= value * (x + 1 - ix) * (y + 1 - iy) * go;
  }
  v = c.voxel(1, d, h, w);
  if (v < int64_t(d)*h*w) {
    const T value = texture[v];
    gx += value * (y + 1 - iy) * (z + 1 - iz) * go;
    gy -= value * (ix - x) * (z + 1 - iz) * go;
    gz -= value * (ix - x) * (y + 1 - iy) * go;
  }
  v = c.voxel(2, d, h, w);
  if (v < int64_t(d)*h*w) {
    const T value = texture[v];
    gx -= value * (iy - y) * (z + 1 - iz) * go;
    gy += value * (x + 1 - ix) * (z + 1 - iz) * go;
    gz -= value * (x + 1 - ix) * (iy - y) * go;
  }
  v = c.voxel(3, d, h, w);
  if (v < int64_t(d)*h*w) {
    const T value = texture[v];
    gx += value * (iy - y) * (z + 1 - iz) * go;
    gy += value * (ix - x) * (z + 1 - iz) * go;
    gz -= value * (ix - x) * (iy - y) * go;
  }
  v = c.voxel(4, d, h, w);
  if (v < int64_t(d)*h*w) {
    const T value = texture[v];
    gx -= value * (y + 1 - iy) * (iz - z) * go;
    gy -= value * (x + 1 - ix) * (iz - z) * go;
    gz += value * (x + 1 - ix) * (y + 1 - iy) * go;
  }
  v = c.voxel(5, d, h, w);
  if (v < int64_t(d)*h*w) {
    const T value = texture[v];
    gx += value * (y + 1 - iy) * (iz - z) * go;
    gy -= value * (ix - x) * (iz - z) * go;
    gz += value * (ix - x) * (y + 1 - iy) * go;
  }
  v = c.voxel(6, d, h, w);
  if (v < int64_t(d)*h*w) {
    const T value = texture[v];
    gx -= value * (iy - y) * (iz - z) * go;
    gy += value * (x + 1 - ix) * (iz - z) * go;
    gz += value * (x + 1 - ix) * (iy - y) * go;
  }
  v = c.voxel(7, d, h, w);
  if (v < int64_t(d)*h*w) {
    const T value = texture[v];
    gx += value * (iy - y) * (iz - z) * go;
    gy += value * (ix - x) * (iz - z) * go;
    gz += value * (ix - x) * (iy - y) * go;
  }
  grad_grid[3*q] = mx * gx;
  grad_grid[3*q+1] = my * gy;
  grad_grid[3*q+2] = mz * gz;
}
}  // namespace

void trilinear_border_forward_cuda(const torch::stable::Tensor& texture,
    const torch::stable::Tensor& query, torch::stable::Tensor out) {
  const int64_t count = out.numel();
  if (!count) return;
  const torch::stable::accelerator::DeviceGuard guard(texture.get_device_index());
#define FORWARD(T) forward_kernel<T><<<linear_grid(count), 256, 0, current_cuda_stream()>>>( \
    texture.const_data_ptr<T>(), query.const_data_ptr<T>(), out.mutable_data_ptr<T>(), \
    count, texture.size(0), texture.size(1), texture.size(2))
  if (texture.scalar_type() == torch::headeronly::ScalarType::Double) { FORWARD(double); }
  else { FORWARD(float); }
#undef FORWARD
  WITWIN_CUDA_CHECK();
}

void trilinear_border_entries_cuda(const torch::stable::Tensor& query,
    int64_t d, int64_t h, int64_t w, torch::stable::Tensor voxels, torch::stable::Tensor weights) {
  const int64_t count = query.size(0);
  if (!count) return;
  const torch::stable::accelerator::DeviceGuard guard(query.get_device_index());
#define ENTRIES(T) entries_kernel<T><<<linear_grid(count), 256, 0, current_cuda_stream()>>>( \
    query.const_data_ptr<T>(), voxels.mutable_data_ptr<int64_t>(), weights.mutable_data_ptr<T>(), count, d, h, w)
  if (query.scalar_type() == torch::headeronly::ScalarType::Double) { ENTRIES(double); }
  else { ENTRIES(float); }
#undef ENTRIES
  WITWIN_CUDA_CHECK();
}

void trilinear_border_backward_cuda(const torch::stable::Tensor& rows,
    const torch::stable::Tensor& queries, const torch::stable::Tensor& weights,
    const torch::stable::Tensor& grad_out, torch::stable::Tensor grad_texture) {
  const int64_t count = grad_texture.numel();
  const torch::stable::accelerator::DeviceGuard guard(grad_texture.get_device_index());
#define BACKWARD(T) backward_kernel<T><<<linear_grid(count), 256, 0, current_cuda_stream()>>>( \
    rows.const_data_ptr<int64_t>(), queries.const_data_ptr<int64_t>(), weights.const_data_ptr<T>(), \
    grad_out.const_data_ptr<T>(), grad_texture.mutable_data_ptr<T>(), count)
  if (grad_texture.scalar_type() == torch::headeronly::ScalarType::Double) { BACKWARD(double); }
  else { BACKWARD(float); }
#undef BACKWARD
  WITWIN_CUDA_CHECK();
}

void trilinear_border_grid_backward_cuda(const torch::stable::Tensor& texture,
    const torch::stable::Tensor& query, const torch::stable::Tensor& grad_out,
    torch::stable::Tensor grad_grid) {
  const int64_t count = query.size(0);
  if (!count) return;
  const torch::stable::accelerator::DeviceGuard guard(texture.get_device_index());
#define GRID_BACKWARD(T) grid_backward_kernel<T><<<linear_grid(count), 256, 0, current_cuda_stream()>>>( \
    texture.const_data_ptr<T>(), query.const_data_ptr<T>(), grad_out.const_data_ptr<T>(), \
    grad_grid.mutable_data_ptr<T>(), count, texture.size(0), texture.size(1), texture.size(2))
  if (texture.scalar_type() == torch::headeronly::ScalarType::Double) { GRID_BACKWARD(double); }
  else { GRID_BACKWARD(float); }
#undef GRID_BACKWARD
  WITWIN_CUDA_CHECK();
}
