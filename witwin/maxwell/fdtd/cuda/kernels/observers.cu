#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

namespace {

template <typename IndexT>
__global__ void accumulate_point_observer_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ field,
    const IndexT* __restrict__ point_i,
    const IndexT* __restrict__ point_j,
    const IndexT* __restrict__ point_k,
    float weighted_cos,
    float weighted_sin,
    float* __restrict__ real_accum,
    float* __restrict__ imag_accum) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const long long field_offset = offset3d(
      static_cast<unsigned int>(point_i[index]),
      static_cast<unsigned int>(point_j[index]),
      static_cast<unsigned int>(point_k[index]),
      ny,
      nz);
  const float value = field[field_offset];
  real_accum[index] += value * weighted_cos;
  imag_accum[index] += value * weighted_sin;
}

template <int Axis>
__global__ void accumulate_plane_observer_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ field,
    int plane_index,
    float weighted_cos,
    float weighted_sin,
    float* __restrict__ real_accum,
    float* __restrict__ imag_accum) {
  unsigned int i = 0;
  unsigned int j = 0;
  unsigned int k = 0;
  int64_t linear = 0;
  if constexpr (Axis == 0) {
    k = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= ny || k >= nz) {
      return;
    }
    i = static_cast<unsigned int>(plane_index);
    linear = static_cast<int64_t>(j) * nz + k;
  } else if constexpr (Axis == 1) {
    k = blockIdx.x * blockDim.x + threadIdx.x;
    i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || k >= nz) {
      return;
    }
    j = static_cast<unsigned int>(plane_index);
    linear = static_cast<int64_t>(i) * nz + k;
  } else {
    j = blockIdx.x * blockDim.x + threadIdx.x;
    i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) {
      return;
    }
    k = static_cast<unsigned int>(plane_index);
    linear = static_cast<int64_t>(i) * ny + j;
  }
  const float value = field[offset3d(i, j, k, ny, nz)];
  real_accum[linear] += value * weighted_cos;
  imag_accum[linear] += value * weighted_sin;
}

// Single-block reduction of the instantaneous Poynting flux through a plane:
//   flux = scale * sum_pq (Ea*Hb - Eb*Ha) * weight
// The four tangential field planes are already Yee-averaged onto the common
// (P, Q) grid and share `weights`'s layout; the scalar result is written
// straight into out[out_index] (a preallocated time-series slot), so a sampled
// step needs no temporaries and no host round-trip. block_size must be a power
// of two for the tree reduction.
__global__ void plane_flux_reduce_kernel(
    int64_t plane_size,
    float scale,
    const float* __restrict__ ea,
    const float* __restrict__ eb,
    const float* __restrict__ ha,
    const float* __restrict__ hb,
    const float* __restrict__ weights,
    float* __restrict__ out,
    int64_t out_index) {
  extern __shared__ float sdata[];
  const unsigned int tid = threadIdx.x;
  float local = 0.0f;
  for (int64_t idx = tid; idx < plane_size; idx += blockDim.x) {
    local += (ea[idx] * hb[idx] - eb[idx] * ha[idx]) * weights[idx];
  }
  sdata[tid] = local;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[out_index] = scale * sdata[0];
  }
}

}  // namespace

namespace {

void check_index_vector(const torch::stable::Tensor& tensor, const char* name) {
  check_cuda_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Int || tensor.scalar_type() == torch::headeronly::ScalarType::Long,
      name,
      " must be int32 or int64");
  STD_TORCH_CHECK(tensor.dim() == 1, name, " must be rank 1");
}

template <typename IndexT>
void launch_point_observers(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ field,
    const IndexT* __restrict__ point_i,
    const IndexT* __restrict__ point_j,
    const IndexT* __restrict__ point_k,
    float weighted_cos,
    float weighted_sin,
    float* __restrict__ real_accum,
    float* __restrict__ imag_accum,
    int block_size) {
  accumulate_point_observer_kernel<IndexT><<<linear_grid(total, block_size), block_size, 0, current_cuda_stream()>>>(
      total,
      ny,
      nz,
      field,
      point_i,
      point_j,
      point_k,
      weighted_cos,
      weighted_sin,
      real_accum,
      imag_accum);
}

template <int Axis>
void launch_plane_observer(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ field,
    int plane_index,
    float weighted_cos,
    float weighted_sin,
    float* __restrict__ real_accum,
    float* __restrict__ imag_accum) {
  const dim3 block(64, 4, 1);
  const unsigned int dim_x = Axis == 2 ? ny : nz;
  const unsigned int dim_y = Axis == 0 ? ny : nx;
  const dim3 grid((dim_x + block.x - 1) / block.x, (dim_y + block.y - 1) / block.y, 1);
  accumulate_plane_observer_kernel<Axis><<<grid, block, 0, current_cuda_stream()>>>(
      nx,
      ny,
      nz,
      field,
      plane_index,
      weighted_cos,
      weighted_sin,
      real_accum,
      imag_accum);
}

}  // namespace

void accumulate_point_observers_cuda(
    const torch::stable::Tensor& field,
    const torch::stable::Tensor& point_i,
    const torch::stable::Tensor& point_j,
    const torch::stable::Tensor& point_k,
    torch::stable::Tensor real_accum,
    torch::stable::Tensor imag_accum,
    double weighted_cos,
    double weighted_sin) {
  check_float32_tensor(field, "field");
  check_float32_tensor(real_accum, "real_accum");
  check_float32_tensor(imag_accum, "imag_accum");
  check_contiguous_tensor(field, "field");
  check_contiguous_tensor(real_accum, "real_accum");
  check_contiguous_tensor(imag_accum, "imag_accum");
  STD_TORCH_CHECK(field.dim() == 3, "field must be rank 3");
  check_index_vector(point_i, "point_i");
  check_index_vector(point_j, "point_j");
  check_index_vector(point_k, "point_k");
  STD_TORCH_CHECK(point_i.scalar_type() == point_j.scalar_type(), "point_j must match point_i dtype");
  STD_TORCH_CHECK(point_i.scalar_type() == point_k.scalar_type(), "point_k must match point_i dtype");
  check_same_cuda_device(field, point_i, "point_i");
  check_same_cuda_device(field, point_j, "point_j");
  check_same_cuda_device(field, point_k, "point_k");
  check_same_cuda_device(field, real_accum, "real_accum");
  check_same_cuda_device(field, imag_accum, "imag_accum");
  STD_TORCH_CHECK(real_accum.sizes().equals(imag_accum.sizes()), "observer accumulators must match");
  STD_TORCH_CHECK(point_i.numel() == real_accum.numel(), "point_i size must match accumulators");
  STD_TORCH_CHECK(point_j.numel() == real_accum.numel(), "point_j size must match accumulators");
  STD_TORCH_CHECK(point_k.numel() == real_accum.numel(), "point_k size must match accumulators");
  torch::stable::accelerator::DeviceGuard guard(field.get_device_index());
  const auto sizes = field.sizes();
  const int64_t point_count = real_accum.numel();
  const int block_size = point_count <= 64 ? 64 : (point_count >= 262144 ? 512 : 256);
  if (point_i.scalar_type() == torch::headeronly::ScalarType::Int) {
    launch_point_observers<int>(
        point_count,
        static_cast<unsigned int>(sizes[1]),
        static_cast<unsigned int>(sizes[2]),
        field.mutable_data_ptr<float>(),
        point_i.mutable_data_ptr<int>(),
        point_j.mutable_data_ptr<int>(),
        point_k.mutable_data_ptr<int>(),
        static_cast<float>(weighted_cos),
        static_cast<float>(weighted_sin),
        real_accum.mutable_data_ptr<float>(),
        imag_accum.mutable_data_ptr<float>(),
        block_size);
  } else {
    launch_point_observers<int64_t>(
        point_count,
        static_cast<unsigned int>(sizes[1]),
        static_cast<unsigned int>(sizes[2]),
        field.mutable_data_ptr<float>(),
        point_i.mutable_data_ptr<int64_t>(),
        point_j.mutable_data_ptr<int64_t>(),
        point_k.mutable_data_ptr<int64_t>(),
        static_cast<float>(weighted_cos),
        static_cast<float>(weighted_sin),
        real_accum.mutable_data_ptr<float>(),
        imag_accum.mutable_data_ptr<float>(),
        block_size);
  }
  WITWIN_CUDA_CHECK();
}

void accumulate_plane_observer_cuda(
    const torch::stable::Tensor& field,
    torch::stable::Tensor real_accum,
    torch::stable::Tensor imag_accum,
    int64_t axis,
    int64_t plane_index,
    double weighted_cos,
    double weighted_sin) {
  check_float32_tensor(field, "field");
  check_float32_tensor(real_accum, "real_accum");
  check_float32_tensor(imag_accum, "imag_accum");
  check_contiguous_tensor(field, "field");
  check_contiguous_tensor(real_accum, "real_accum");
  check_contiguous_tensor(imag_accum, "imag_accum");
  STD_TORCH_CHECK(field.dim() == 3, "field must be rank 3");
  STD_TORCH_CHECK(axis >= 0 && axis <= 2, "axis must be 0, 1, or 2");
  STD_TORCH_CHECK(plane_index >= 0 && plane_index < field.size(axis), "plane_index is out of range");
  check_same_cuda_device(field, real_accum, "real_accum");
  check_same_cuda_device(field, imag_accum, "imag_accum");
  STD_TORCH_CHECK(real_accum.sizes().equals(imag_accum.sizes()), "observer accumulators must match");
  const int64_t expected = axis == 0
      ? field.size(1) * field.size(2)
      : (axis == 1 ? field.size(0) * field.size(2) : field.size(0) * field.size(1));
  STD_TORCH_CHECK(real_accum.numel() == expected, "plane observer accumulator size does not match selected plane");
  torch::stable::accelerator::DeviceGuard guard(field.get_device_index());
  const auto sizes = field.sizes();
  const auto nx = static_cast<unsigned int>(sizes[0]);
  const auto ny = static_cast<unsigned int>(sizes[1]);
  const auto nz = static_cast<unsigned int>(sizes[2]);
  const auto plane = static_cast<int>(plane_index);
  const auto cos_weight = static_cast<float>(weighted_cos);
  const auto sin_weight = static_cast<float>(weighted_sin);
  if (axis == 0) {
    launch_plane_observer<0>(
        nx, ny, nz, field.mutable_data_ptr<float>(), plane, cos_weight, sin_weight,
        real_accum.mutable_data_ptr<float>(), imag_accum.mutable_data_ptr<float>());
  } else if (axis == 1) {
    launch_plane_observer<1>(
        nx, ny, nz, field.mutable_data_ptr<float>(), plane, cos_weight, sin_weight,
        real_accum.mutable_data_ptr<float>(), imag_accum.mutable_data_ptr<float>());
  } else {
    launch_plane_observer<2>(
        nx, ny, nz, field.mutable_data_ptr<float>(), plane, cos_weight, sin_weight,
        real_accum.mutable_data_ptr<float>(), imag_accum.mutable_data_ptr<float>());
  }
  WITWIN_CUDA_CHECK();
}

void plane_flux_reduce_cuda(
    const torch::stable::Tensor& ea,
    const torch::stable::Tensor& eb,
    const torch::stable::Tensor& ha,
    const torch::stable::Tensor& hb,
    const torch::stable::Tensor& weights,
    torch::stable::Tensor out,
    int64_t out_index,
    double scale) {
  check_float32_tensor(ea, "ea");
  check_float32_tensor(eb, "eb");
  check_float32_tensor(ha, "ha");
  check_float32_tensor(hb, "hb");
  check_float32_tensor(weights, "weights");
  check_float32_tensor(out, "out");
  check_contiguous_tensor(ea, "ea");
  check_contiguous_tensor(eb, "eb");
  check_contiguous_tensor(ha, "ha");
  check_contiguous_tensor(hb, "hb");
  check_contiguous_tensor(weights, "weights");
  check_contiguous_tensor(out, "out");
  const int64_t plane_size = weights.numel();
  STD_TORCH_CHECK(ea.numel() == plane_size, "ea size must match weights");
  STD_TORCH_CHECK(eb.numel() == plane_size, "eb size must match weights");
  STD_TORCH_CHECK(ha.numel() == plane_size, "ha size must match weights");
  STD_TORCH_CHECK(hb.numel() == plane_size, "hb size must match weights");
  STD_TORCH_CHECK(out.dim() == 1, "out must be rank 1");
  STD_TORCH_CHECK(out_index >= 0 && out_index < out.numel(), "out_index is out of range");
  check_same_cuda_device(ea, eb, "eb");
  check_same_cuda_device(ea, ha, "ha");
  check_same_cuda_device(ea, hb, "hb");
  check_same_cuda_device(ea, weights, "weights");
  check_same_cuda_device(ea, out, "out");
  torch::stable::accelerator::DeviceGuard guard(ea.get_device_index());
  const int block_size = 256;
  plane_flux_reduce_kernel<<<1, block_size, block_size * sizeof(float), current_cuda_stream()>>>(
      plane_size,
      static_cast<float>(scale),
      ea.mutable_data_ptr<float>(),
      eb.mutable_data_ptr<float>(),
      ha.mutable_data_ptr<float>(),
      hb.mutable_data_ptr<float>(),
      weights.mutable_data_ptr<float>(),
      out.mutable_data_ptr<float>(),
      out_index);
  WITWIN_CUDA_CHECK();
}
