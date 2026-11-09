#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

namespace {
// One thread owns one lane/frequency recurrence. Double precision preserves the
// ordinary host recurrence and normalization; field weights are rounded once.
template <typename AccumT>
__global__ void batch_observer_weights_kernel(int64_t count, int64_t frequencies,
    int64_t lanes, int64_t signal_steps, double* state, const int64_t* schedule,
    const double* signals, const int* active, const int64_t* step, AccumT* weights) {
  const int64_t q = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= count) return;
  const int64_t lane = q / frequencies;
  if (!active[lane]) return;
  const int64_t n = step[0], start = schedule[3*q], end = schedule[3*q+1];
  const int64_t window = schedule[3*q+2];
  double* s = state + 10*q;
  double w = 0.;
  if (n >= start && n < end) {
    const double pos = static_cast<double>(n-start) / static_cast<double>(end-start);
    w = window == 1 ? .5 * (1. - cos(6.2831853071795864769 * pos)) :
        (window == 2 && pos < .1 ? .5 * (1. - cos(3.14159265358979323846 * pos / .1)) : 1.);
    s[6] += w;
    s[7] += 1.;
  }
  const double c = w*s[0], si = w*s[1];
  weights[4*q] = static_cast<AccumT>(c);
  weights[4*q+1] = static_cast<AccumT>(si);
  weights[4*q+2] = static_cast<AccumT>(c*s[4] - si*s[5]);
  weights[4*q+3] = static_cast<AccumT>(si*s[4] + c*s[5]);
  const double signal = n >= 0 && n < signal_steps ? signals[n*lanes+lane] : 0.;
  s[8] += signal*c;
  s[9] += signal*si;
  const double next_c = s[0]*s[2] - s[1]*s[3];
  s[1] = s[1]*s[2] + s[0]*s[3];
  s[0] = next_c;
}

template <typename AccumT>
__global__ void batch_observer_gather_kernel(int64_t count, int64_t points,
    int64_t frequencies, int64_t field_size, int64_t weight_count, int retard,
    const float* field, const int64_t* offsets, const int64_t* frequency_map,
    const AccumT* weights, const int* active, AccumT* real, AccumT* imag) {
  const int64_t q = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= count) return;
  const int64_t row = q / points, lane = row / frequencies;
  if (!active[lane]) return;
  const int64_t offset = offsets[lane*points+q%points], f = frequency_map[row];
  if (offset < 0 || offset >= field_size || f < 0 || f >= weight_count) return;
  const float value = field[offset];
  real[q] += value * weights[4*f+retard];
  imag[q] += value * weights[4*f+retard+1];
}

void batch_check(const torch::stable::Tensor& reference, const torch::stable::Tensor& tensor,
    torch::headeronly::ScalarType dtype, const char* name) {
  check_cuda_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  check_same_cuda_device(reference, tensor, name);
  STD_TORCH_CHECK(tensor.scalar_type() == dtype, name, " has incorrect dtype");
}

void batch_disjoint(const torch::stable::Tensor& a, const torch::stable::Tensor& b) {
  const auto x = reinterpret_cast<uintptr_t>(a.data_ptr());
  const auto y = reinterpret_cast<uintptr_t>(b.data_ptr());
  STD_TORCH_CHECK(x + a.numel()*a.element_size() <= y || y + b.numel()*b.element_size() <= x,
      "batch observer writable storage must not overlap another input");
}
}  // namespace

void batch_observer_weights_cuda(torch::stable::Tensor state,
    const torch::stable::Tensor& schedule, const torch::stable::Tensor& signals,
    const torch::stable::Tensor& active, const torch::stable::Tensor& step,
    torch::stable::Tensor weights) {
  using torch::headeronly::ScalarType;
  batch_check(state, state, ScalarType::Double, "state");
  batch_check(state, schedule, ScalarType::Long, "schedule");
  batch_check(state, signals, ScalarType::Double, "signals");
  batch_check(state, active, ScalarType::Int, "active");
  batch_check(state, step, ScalarType::Long, "step");
  STD_TORCH_CHECK(weights.scalar_type() == ScalarType::Float || weights.scalar_type() == ScalarType::Double,
      "observer weights must be float32 or float64");
  batch_check(state, weights, weights.scalar_type(), "weights");
  STD_TORCH_CHECK(state.dim() == 3 && state.size(2) == 10, "state must be [B,F,10]");
  const int64_t b = state.size(0), f = state.size(1);
  STD_TORCH_CHECK(b > 0 && f > 0 && schedule.dim() == 3 && schedule.size(0) == b &&
      schedule.size(1) == f && schedule.size(2) == 3, "schedule must be [B,F,3]");
  STD_TORCH_CHECK(weights.dim() == 3 && weights.size(0) == b && weights.size(1) == f &&
      weights.size(2) == 4, "weights must be [B,F,4]");
  STD_TORCH_CHECK(signals.dim() == 2 && signals.size(1) == b && active.numel() == b &&
      step.numel() == 1, "invalid signals, active mask or step");
  for (const auto& input : {schedule, signals, active, step}) {
    batch_disjoint(state, input);
    batch_disjoint(weights, input);
  }
  batch_disjoint(state, weights);
  torch::stable::accelerator::DeviceGuard guard(state.get_device_index());
  if (weights.scalar_type() == ScalarType::Double) {
  batch_observer_weights_kernel<double><<<linear_grid(b*f, 128), 128, 0, current_cuda_stream()>>>(
      b*f, f, b, signals.size(0), state.mutable_data_ptr<double>(), schedule.mutable_data_ptr<int64_t>(),
      signals.mutable_data_ptr<double>(), active.mutable_data_ptr<int>(), step.mutable_data_ptr<int64_t>(),
      weights.mutable_data_ptr<double>());
  } else {
  batch_observer_weights_kernel<float><<<linear_grid(b*f, 128), 128, 0, current_cuda_stream()>>>(
      b*f, f, b, signals.size(0), state.mutable_data_ptr<double>(), schedule.mutable_data_ptr<int64_t>(),
      signals.mutable_data_ptr<double>(), active.mutable_data_ptr<int>(), step.mutable_data_ptr<int64_t>(),
      weights.mutable_data_ptr<float>());
  }
  WITWIN_CUDA_CHECK();
}

void batch_observer_gather_cuda(const torch::stable::Tensor& field,
    const torch::stable::Tensor& offsets, const torch::stable::Tensor& frequency_map,
    const torch::stable::Tensor& weights, const torch::stable::Tensor& active,
    torch::stable::Tensor real, torch::stable::Tensor imag, bool magnetic) {
  using torch::headeronly::ScalarType;
  batch_check(field, field, ScalarType::Float, "field");
  batch_check(field, offsets, ScalarType::Long, "offsets");
  batch_check(field, frequency_map, ScalarType::Long, "frequency_map");
  STD_TORCH_CHECK(weights.scalar_type() == ScalarType::Float || weights.scalar_type() == ScalarType::Double,
      "observer weights must be float32 or float64");
  batch_check(field, weights, weights.scalar_type(), "weights");
  batch_check(field, active, ScalarType::Int, "active");
  batch_check(field, real, weights.scalar_type(), "real");
  batch_check(field, imag, weights.scalar_type(), "imag");
  STD_TORCH_CHECK(real.dim() == 3 && real.sizes().equals(imag.sizes()), "accumulators must match [B,F,P]");
  const int64_t b = real.size(0), f = real.size(1), p = real.size(2);
  STD_TORCH_CHECK(b > 0 && f > 0 && p > 0 && field.dim() == 4 && field.size(0) == b,
      "invalid batch field or accumulator dimensions");
  STD_TORCH_CHECK(offsets.dim() == 2 && offsets.size(0) == b && offsets.size(1) == p &&
      frequency_map.dim() == 2 && frequency_map.size(0) == b && frequency_map.size(1) == f &&
      weights.dim() == 3 && weights.size(0) == b && weights.size(2) == 4 && active.numel() == b,
      "invalid offsets, frequency map, weights or active mask");
  for (const auto& input : {field, offsets, frequency_map, weights, active}) {
    batch_disjoint(real, input);
    batch_disjoint(imag, input);
  }
  batch_disjoint(real, imag);
  torch::stable::accelerator::DeviceGuard guard(field.get_device_index());
  if (weights.scalar_type() == ScalarType::Double) {
  batch_observer_gather_kernel<double><<<linear_grid(real.numel(), 256), 256, 0, current_cuda_stream()>>>(
      real.numel(), p, f, field.numel(), weights.numel()/4, magnetic ? 2 : 0,
      field.mutable_data_ptr<float>(), offsets.mutable_data_ptr<int64_t>(),
      frequency_map.mutable_data_ptr<int64_t>(), weights.mutable_data_ptr<double>(), active.mutable_data_ptr<int>(),
      real.mutable_data_ptr<double>(), imag.mutable_data_ptr<double>());
  } else {
  batch_observer_gather_kernel<float><<<linear_grid(real.numel(), 256), 256, 0, current_cuda_stream()>>>(
      real.numel(), p, f, field.numel(), weights.numel()/4, magnetic ? 2 : 0,
      field.mutable_data_ptr<float>(), offsets.mutable_data_ptr<int64_t>(),
      frequency_map.mutable_data_ptr<int64_t>(), weights.mutable_data_ptr<float>(), active.mutable_data_ptr<int>(),
      real.mutable_data_ptr<float>(), imag.mutable_data_ptr<float>());
  }
  WITWIN_CUDA_CHECK();
}

namespace {

template <typename IndexT, typename AccumT, typename FieldT = float, bool Indexed = false>
__global__ void accumulate_point_observer_kernel(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const FieldT* __restrict__ field,
    const IndexT* __restrict__ point_i,
    const IndexT* __restrict__ point_j,
    const IndexT* __restrict__ point_k,
    AccumT weighted_cos,
    AccumT weighted_sin,
    AccumT* __restrict__ real_accum,
    AccumT* __restrict__ imag_accum,
    const AccumT* __restrict__ weights,
    const int64_t* __restrict__ step) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  if constexpr (Indexed) {
    const int64_t row = 3 * step[0];
    if (weights[row] == AccumT(0)) return;
    weighted_cos = weights[row + 1];
    weighted_sin = weights[row + 2];
  }
  const long long field_offset = offset3d(
      static_cast<unsigned int>(point_i[index]),
      static_cast<unsigned int>(point_j[index]),
      static_cast<unsigned int>(point_k[index]),
      ny,
      nz);
  const FieldT value = field[field_offset];
  real_accum[index] += value * weighted_cos;
  imag_accum[index] += value * weighted_sin;
}

template <int Axis, typename AccumT, typename FieldT = float>
__global__ void accumulate_plane_observer_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const FieldT* __restrict__ field,
    int plane_index,
    AccumT weighted_cos,
    AccumT weighted_sin,
    AccumT* __restrict__ real_accum,
    AccumT* __restrict__ imag_accum) {
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
  const FieldT value = field[offset3d(i, j, k, ny, nz)];
  real_accum[linear] += value * weighted_cos;
  imag_accum[linear] += value * weighted_sin;
}

// Native paired Yee products are averaged only after multiplication. The low
// endpoint uses the positive-side E trace; the high endpoint uses the negative
// trace. They alias for continuous fields. E and H retain their sampled clocks.
__global__ void plane_flux_reduce_kernel(
    int64_t rows, int64_t cols, int average_axis, float scale,
    const float* __restrict__ ea, const float* __restrict__ eb,
    const float* __restrict__ ha, const float* __restrict__ hb,
    const float* __restrict__ ea_positive, const float* __restrict__ eb_positive,
    const float* __restrict__ weights, float* __restrict__ out, int64_t out_index) {
  extern __shared__ float sdata[];
  const unsigned int tid = threadIdx.x;
  float local = 0.0f;
  for (int64_t idx = tid; idx < rows * cols; idx += blockDim.x) {
    const int64_t i = idx / cols, j = idx % cols;
    const int64_t a = i * (cols + (average_axis == 1)) + j;
    const int64_t b = i * (cols + (average_axis == 0)) + j;
    const int64_t a_next = a + (average_axis == 0 ? cols : 1);
    const int64_t b_next = b + (average_axis == 1 ? cols : 1);
    const float first = 0.5f * (ea_positive[a] * hb[a] + ea[a_next] * hb[a_next]);
    const float second = 0.5f * (eb_positive[b] * ha[b] + eb[b_next] * ha[b_next]);
    local += (first - second) * weights[idx];
  }
  sdata[tid] = local;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) sdata[tid] += sdata[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[out_index] = scale * sdata[0];
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

template <typename IndexT, typename AccumT, typename FieldT = float>
void launch_point_observers(
    int64_t total,
    unsigned int ny,
    unsigned int nz,
    const FieldT* __restrict__ field,
    const IndexT* __restrict__ point_i,
    const IndexT* __restrict__ point_j,
    const IndexT* __restrict__ point_k,
    AccumT weighted_cos,
    AccumT weighted_sin,
    AccumT* __restrict__ real_accum,
    AccumT* __restrict__ imag_accum,
    int block_size) {
  accumulate_point_observer_kernel<IndexT, AccumT, FieldT><<<linear_grid(total, block_size), block_size, 0, current_cuda_stream()>>>(
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
      imag_accum,
      nullptr,
      nullptr);
}

template <int Axis, typename AccumT, typename FieldT = float>
void launch_plane_observer(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const FieldT* __restrict__ field,
    int plane_index,
    AccumT weighted_cos,
    AccumT weighted_sin,
    AccumT* __restrict__ real_accum,
    AccumT* __restrict__ imag_accum) {
  const dim3 block(64, 4, 1);
  const unsigned int dim_x = Axis == 2 ? ny : nz;
  const unsigned int dim_y = Axis == 0 ? ny : nx;
  const dim3 grid((dim_x + block.x - 1) / block.x, (dim_y + block.y - 1) / block.y, 1);
  accumulate_plane_observer_kernel<Axis, AccumT, FieldT><<<grid, block, 0, current_cuda_stream()>>>(
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
  check_cuda_tensor(field, "field");
  STD_TORCH_CHECK(field.scalar_type() == torch::headeronly::ScalarType::Float ||
      field.scalar_type() == torch::headeronly::ScalarType::Double, "observer field must be float32 or float64");
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(real_accum.scalar_type() == ScalarType::Float || real_accum.scalar_type() == ScalarType::Double,
      "observer accumulators must be float32 or float64");
  STD_TORCH_CHECK(imag_accum.scalar_type() == real_accum.scalar_type(), "observer accumulator dtype must match");
  check_cuda_tensor(real_accum, "real_accum");
  check_cuda_tensor(imag_accum, "imag_accum");
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
  if (field.scalar_type() == ScalarType::Double) {
    STD_TORCH_CHECK(real_accum.scalar_type() == ScalarType::Double, "FP64 field observations require FP64 accumulation");
    if (point_i.scalar_type() == ScalarType::Int) {
      launch_point_observers<int, double, double>(point_count,
          static_cast<unsigned int>(sizes[1]), static_cast<unsigned int>(sizes[2]),
          field.mutable_data_ptr<double>(), point_i.mutable_data_ptr<int>(),
          point_j.mutable_data_ptr<int>(), point_k.mutable_data_ptr<int>(),
          weighted_cos, weighted_sin, real_accum.mutable_data_ptr<double>(),
          imag_accum.mutable_data_ptr<double>(), block_size);
    } else {
      launch_point_observers<int64_t, double, double>(point_count,
          static_cast<unsigned int>(sizes[1]), static_cast<unsigned int>(sizes[2]),
          field.mutable_data_ptr<double>(), point_i.mutable_data_ptr<int64_t>(),
          point_j.mutable_data_ptr<int64_t>(), point_k.mutable_data_ptr<int64_t>(),
          weighted_cos, weighted_sin, real_accum.mutable_data_ptr<double>(),
          imag_accum.mutable_data_ptr<double>(), block_size);
    }
    WITWIN_CUDA_CHECK();
    return;
  }
  if (real_accum.scalar_type() == ScalarType::Double) {
    if (point_i.scalar_type() == ScalarType::Int) {
      launch_point_observers<int, double>(point_count,
          static_cast<unsigned int>(sizes[1]), static_cast<unsigned int>(sizes[2]),
          field.mutable_data_ptr<float>(), point_i.mutable_data_ptr<int>(),
          point_j.mutable_data_ptr<int>(), point_k.mutable_data_ptr<int>(),
          weighted_cos, weighted_sin, real_accum.mutable_data_ptr<double>(),
          imag_accum.mutable_data_ptr<double>(), block_size);
    } else {
      launch_point_observers<int64_t, double>(point_count,
          static_cast<unsigned int>(sizes[1]), static_cast<unsigned int>(sizes[2]),
          field.mutable_data_ptr<float>(), point_i.mutable_data_ptr<int64_t>(),
          point_j.mutable_data_ptr<int64_t>(), point_k.mutable_data_ptr<int64_t>(),
          weighted_cos, weighted_sin, real_accum.mutable_data_ptr<double>(),
          imag_accum.mutable_data_ptr<double>(), block_size);
    }
    WITWIN_CUDA_CHECK();
    return;
  }
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

void accumulate_point_observers_indexed_cuda(
    const torch::stable::Tensor& field,
    const torch::stable::Tensor& point_i,
    const torch::stable::Tensor& point_j,
    const torch::stable::Tensor& point_k,
    torch::stable::Tensor real_accum,
    torch::stable::Tensor imag_accum,
    const torch::stable::Tensor& weights,
    const torch::stable::Tensor& step) {
  check_float32_tensor(field, "field");
  check_contiguous_tensor(field, "field");
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(field.dim() == 3, "field must be rank 3");
  STD_TORCH_CHECK(real_accum.scalar_type() == ScalarType::Float ||
      real_accum.scalar_type() == ScalarType::Double, "observer accumulators must be float32 or float64");
  for (const auto& tensor : {point_i, point_j, point_k, real_accum, imag_accum, weights, step}) {
    check_cuda_tensor(tensor, "indexed point observer tensor");
    check_contiguous_tensor(tensor, "indexed point observer tensor");
    check_same_cuda_device(field, tensor, "indexed point observer tensor");
  }
  for (const auto& tensor : {point_i, point_j, point_k}) {
    STD_TORCH_CHECK(tensor.scalar_type() == ScalarType::Int && tensor.dim() == 1 &&
        tensor.numel() == real_accum.numel(), "point indices must be int32 vectors matching accumulators");
  }
  STD_TORCH_CHECK(real_accum.sizes().equals(imag_accum.sizes()) &&
      imag_accum.scalar_type() == real_accum.scalar_type() && weights.scalar_type() == real_accum.scalar_type(),
      "point observer weights and accumulators must match");
  STD_TORCH_CHECK(weights.dim() == 2 && weights.size(1) == 3 &&
      step.scalar_type() == ScalarType::Long && step.numel() == 1, "invalid point observer weights or step");
  torch::stable::accelerator::DeviceGuard guard(field.get_device_index());
  const auto sizes = field.sizes();
  const int64_t count = real_accum.numel();
  const int block_size = count <= 64 ? 64 : (count >= 262144 ? 512 : 256);
  if (real_accum.scalar_type() == ScalarType::Double) {
    accumulate_point_observer_kernel<int, double, float, true><<<linear_grid(count, block_size), block_size, 0, current_cuda_stream()>>>(
        count, sizes[1], sizes[2], field.mutable_data_ptr<float>(), point_i.mutable_data_ptr<int>(),
        point_j.mutable_data_ptr<int>(), point_k.mutable_data_ptr<int>(), 0.0, 0.0,
        real_accum.mutable_data_ptr<double>(), imag_accum.mutable_data_ptr<double>(),
        weights.mutable_data_ptr<double>(), step.mutable_data_ptr<int64_t>());
  } else {
    accumulate_point_observer_kernel<int, float, float, true><<<linear_grid(count, block_size), block_size, 0, current_cuda_stream()>>>(
        count, sizes[1], sizes[2], field.mutable_data_ptr<float>(), point_i.mutable_data_ptr<int>(),
        point_j.mutable_data_ptr<int>(), point_k.mutable_data_ptr<int>(), 0.0f, 0.0f,
        real_accum.mutable_data_ptr<float>(), imag_accum.mutable_data_ptr<float>(),
        weights.mutable_data_ptr<float>(), step.mutable_data_ptr<int64_t>());
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
  check_cuda_tensor(field, "field");
  STD_TORCH_CHECK(field.scalar_type() == torch::headeronly::ScalarType::Float ||
      field.scalar_type() == torch::headeronly::ScalarType::Double, "observer field must be float32 or float64");
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(real_accum.scalar_type() == ScalarType::Float || real_accum.scalar_type() == ScalarType::Double,
      "observer accumulators must be float32 or float64");
  STD_TORCH_CHECK(imag_accum.scalar_type() == real_accum.scalar_type(), "observer accumulator dtype must match");
  check_cuda_tensor(real_accum, "real_accum");
  check_cuda_tensor(imag_accum, "imag_accum");
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
  if (field.scalar_type() == ScalarType::Double) {
    STD_TORCH_CHECK(real_accum.scalar_type() == ScalarType::Double, "FP64 field observations require FP64 accumulation");
    if (axis == 0) {
      launch_plane_observer<0, double, double>(nx, ny, nz, field.mutable_data_ptr<double>(), plane,
          weighted_cos, weighted_sin, real_accum.mutable_data_ptr<double>(), imag_accum.mutable_data_ptr<double>());
    } else if (axis == 1) {
      launch_plane_observer<1, double, double>(nx, ny, nz, field.mutable_data_ptr<double>(), plane,
          weighted_cos, weighted_sin, real_accum.mutable_data_ptr<double>(), imag_accum.mutable_data_ptr<double>());
    } else {
      launch_plane_observer<2, double, double>(nx, ny, nz, field.mutable_data_ptr<double>(), plane,
          weighted_cos, weighted_sin, real_accum.mutable_data_ptr<double>(), imag_accum.mutable_data_ptr<double>());
    }
    WITWIN_CUDA_CHECK();
    return;
  }
  if (real_accum.scalar_type() == ScalarType::Double) {
    if (axis == 0) {
      launch_plane_observer<0, double>(nx, ny, nz, field.mutable_data_ptr<float>(), plane,
          weighted_cos, weighted_sin, real_accum.mutable_data_ptr<double>(), imag_accum.mutable_data_ptr<double>());
    } else if (axis == 1) {
      launch_plane_observer<1, double>(nx, ny, nz, field.mutable_data_ptr<float>(), plane,
          weighted_cos, weighted_sin, real_accum.mutable_data_ptr<double>(), imag_accum.mutable_data_ptr<double>());
    } else {
      launch_plane_observer<2, double>(nx, ny, nz, field.mutable_data_ptr<float>(), plane,
          weighted_cos, weighted_sin, real_accum.mutable_data_ptr<double>(), imag_accum.mutable_data_ptr<double>());
    }
    WITWIN_CUDA_CHECK();
    return;
  }
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
    const torch::stable::Tensor& ea, const torch::stable::Tensor& eb,
    const torch::stable::Tensor& ha, const torch::stable::Tensor& hb,
    const torch::stable::Tensor& ea_positive, const torch::stable::Tensor& eb_positive,
    const torch::stable::Tensor& weights, torch::stable::Tensor out,
    int64_t out_index, double scale, int64_t average_axis) {
  const torch::stable::Tensor* tensors[] = {&ea, &eb, &ha, &hb, &ea_positive, &eb_positive, &weights, &out};
  for (const auto* tensor : tensors) {
    check_float32_tensor(*tensor, "time flux tensor");
    check_contiguous_tensor(*tensor, "time flux tensor");
    check_same_cuda_device(ea, *tensor, "time flux tensor");
  }
  STD_TORCH_CHECK(average_axis == 0 || average_axis == 1, "average_axis must be 0 or 1");
  STD_TORCH_CHECK(weights.dim() == 2, "weights must be rank 2");
  const int64_t rows = weights.size(0), cols = weights.size(1);
  STD_TORCH_CHECK(rows > 0 && cols > 0, "time flux plane must be nonempty");
  for (const auto* tensor : {&ea, &hb, &ea_positive}) {
    STD_TORCH_CHECK(tensor->dim() == 2 && tensor->size(0) == rows + (average_axis == 0)
        && tensor->size(1) == cols + (average_axis == 1), "Ea/Hb native shape mismatch");
  }
  for (const auto* tensor : {&eb, &ha, &eb_positive}) {
    STD_TORCH_CHECK(tensor->dim() == 2 && tensor->size(0) == rows + (average_axis == 1)
        && tensor->size(1) == cols + (average_axis == 0), "Eb/Ha native shape mismatch");
  }
  STD_TORCH_CHECK(out.dim() == 1, "out must be rank 1");
  STD_TORCH_CHECK(out_index >= 0 && out_index < out.numel(), "out_index is out of range");
  torch::stable::accelerator::DeviceGuard guard(ea.get_device_index());
  const int block_size = 256;
  plane_flux_reduce_kernel<<<1, block_size, block_size * sizeof(float), current_cuda_stream()>>>(
      rows, cols, static_cast<int>(average_axis), static_cast<float>(scale),
      ea.mutable_data_ptr<float>(), eb.mutable_data_ptr<float>(),
      ha.mutable_data_ptr<float>(), hb.mutable_data_ptr<float>(),
      ea_positive.mutable_data_ptr<float>(), eb_positive.mutable_data_ptr<float>(),
      weights.mutable_data_ptr<float>(), out.mutable_data_ptr<float>(), out_index);
  WITWIN_CUDA_CHECK();
}
