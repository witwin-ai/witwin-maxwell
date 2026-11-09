#include <cmath>
#include <type_traits>

#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"
#include "../source_gather.cuh"

namespace {
SourceGather source_gather(const torch::stable::Tensor& field,
    const torch::stable::Tensor& rows, const torch::stable::Tensor& targets,
    const torch::stable::Tensor& entries) {
  for (const auto& tensor : {rows, targets, entries}) {
    check_cuda_tensor(tensor, "source gather");
    check_contiguous_tensor(tensor, "source gather");
    check_same_cuda_device(field, tensor, "source gather");
    STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::Long,
        "source gather indices must be int64");
  }
  STD_TORCH_CHECK(rows.dim() == 1 && targets.dim() == 2 && targets.size(1) == 2 &&
      entries.dim() == 2 && entries.size(1) == 2 && rows.numel() == targets.size(0)+1,
      "invalid prepared source gather shape");
  return {rows.const_data_ptr<int64_t>(), targets.const_data_ptr<int64_t>(),
      entries.const_data_ptr<int64_t>(), targets.size(0)};
}

__global__ void batch_source_patch_kernel(SourceGather g, int64_t b, int64_t steps,
    float* field, const int* lanes, const float* first, const float* second,
    const float* signals, const int* active, const int64_t* step, bool cw, int wrap) {
  const int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= g.count) return;
  const int64_t n = step[0], dst = g.targets[2*row+1];
  if (n < 0 || n >= steps) return;
  float value = field[dst];
  for (int64_t e = g.rows[row]; e < g.rows[row+1]; ++e) {
    const int64_t q = g.entries[2*e];
    const int lane = lanes[q];
    if (lane < 0 || lane >= b || !active[lane]) continue;
    const int64_t signal = (n*b+lane)*2;
    if (cw) {
      const float delta = __fmaf_rn(signals[signal], first[q], __fmul_rn(signals[signal+1], second[q]));
      value = __fadd_rn(value, delta);
    } else if (wrap) {
      const float delta = __fmul_rn(signals[signal], first[q]);
      value = (g.entries[2*e+1] & 8) ? source_atomic_sum(value, delta) : __fadd_rn(value, delta);
    } else {
      value = __fmaf_rn(signals[signal], first[q], value);
    }
  }
  field[dst] = value;
}
}  // namespace

void batch_source_patch_cuda(torch::stable::Tensor field, const torch::stable::Tensor& offsets,
    const torch::stable::Tensor& lanes, const torch::stable::Tensor& first,
    const torch::stable::Tensor& second, const torch::stable::Tensor& signals,
    const torch::stable::Tensor& active, const torch::stable::Tensor& step, bool cw, int64_t wrap,
    const torch::stable::Tensor& gather_rows, const torch::stable::Tensor& gather_targets,
    const torch::stable::Tensor& gather_entries) {
  check_float32_tensor(field, "field");
  check_contiguous_tensor(field, "field");
  STD_TORCH_CHECK(field.dim() == 4, "field must be [B,X,Y,Z]");
  for (const auto& value : {first, second, signals}) {
    check_float32_tensor(value, "source values");
    check_contiguous_tensor(value, "source values");
    check_same_cuda_device(field, value, "source values");
  }
  for (const auto& value : {offsets, lanes, active, step}) {
    check_cuda_tensor(value, "source indices");
    check_contiguous_tensor(value, "source indices");
    check_same_cuda_device(field, value, "source indices");
  }
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(offsets.scalar_type() == ScalarType::Long && step.scalar_type() == ScalarType::Long &&
      lanes.scalar_type() == ScalarType::Int && active.scalar_type() == ScalarType::Int, "invalid source index dtype");
  const int64_t count = offsets.numel(), b = field.size(0);
  STD_TORCH_CHECK(wrap >= 0 && wrap < 8, "invalid periodic source axes");
  STD_TORCH_CHECK(offsets.dim() == 1 && lanes.numel() == count && first.numel() == count &&
      second.numel() == count && active.numel() == b && step.numel() == 1 && signals.dim() == 3 &&
      signals.size(1) == b && signals.size(2) == 2, "invalid source patch or signal shape");
  const auto x = reinterpret_cast<uintptr_t>(field.data_ptr());
  for (const auto& value : {offsets, lanes, first, second, signals, active, step}) {
    const auto y = reinterpret_cast<uintptr_t>(value.data_ptr());
    STD_TORCH_CHECK(!value.numel() || x + field.numel()*4 <= y || y + value.numel()*value.element_size() <= x,
        "source inputs must not overlap writable fields");
  }
  const auto gather = source_gather(field, gather_rows, gather_targets, gather_entries);
  if (!gather.count) return;
  torch::stable::accelerator::DeviceGuard guard(field.get_device_index());
  batch_source_patch_kernel<<<linear_grid(gather.count, 256), 256, 0, current_cuda_stream()>>>(
      gather, b, signals.size(0), field.mutable_data_ptr<float>(), lanes.const_data_ptr<int>(),
      first.const_data_ptr<float>(), second.const_data_ptr<float>(), signals.const_data_ptr<float>(),
      active.const_data_ptr<int>(), step.const_data_ptr<int64_t>(), cw, static_cast<int>(wrap));
  WITWIN_CUDA_CHECK();
}

namespace {

struct Shape3D {
  int x;
  int y;
  int z;
};

dim3 patch_block3d() {
  return dim3(32, 4, 2);
}

dim3 patch_grid3d(Shape3D shape, dim3 block) {
  return dim3(
      static_cast<unsigned int>((shape.z + block.x - 1) / block.x),
      static_cast<unsigned int>((shape.y + block.y - 1) / block.y),
      static_cast<unsigned int>((shape.x + block.z - 1) / block.z));
}

__device__ __forceinline__ long long offset_shape(int i, int j, int k, Shape3D shape) {
  return (static_cast<long long>(i) * shape.y + j) * shape.z + k;
}

__device__ __forceinline__ bool in_bounds(int i, int j, int k, Shape3D shape) {
  return i >= 0 && i < shape.x && j >= 0 && j < shape.y && k >= 0 && k < shape.z;
}

// Analytic source-time kinds: 0 = CW, 1 = Gaussian pulse, 2 = Ricker wavelet,
// 4 = broadband pulse (Gaussian pulse minus its DC-removing Gaussian pedestal).
// Kind 3 (sampled table) is evaluated by evaluate_waveform_table below.
template <int TimeKind>
__device__ __forceinline__ float evaluate_source_time(
    float sample_time,
    float angular_frequency,
    float gaussian_inv_sigma,
    float ricker_pi_frequency,
    float amplitude,
    float phase,
    float delay,
    float pedestal,
    float pedestal_inv_sigma) {
  if constexpr (TimeKind == 0) {
    return amplitude * cosf(angular_frequency * sample_time + phase);
  }
  if constexpr (TimeKind == 1 || TimeKind == 4) {
    const float tau = sample_time - delay;
    const float normalized = tau * gaussian_inv_sigma;
    const float envelope = expf(-0.5f * normalized * normalized);
    if constexpr (TimeKind == 1) {
      // Keep the float32 association (amplitude * envelope) * cos of the
      // original Gaussian-pulse kernel so existing pulses stay bit-identical.
      return amplitude * envelope * cosf(angular_frequency * tau + phase);
    }
    float signal = envelope * cosf(angular_frequency * tau + phase);
    if constexpr (TimeKind == 4) {
      const float pedestal_normalized = tau * pedestal_inv_sigma;
      signal -= pedestal * expf(-0.5f * pedestal_normalized * pedestal_normalized);
    }
    return amplitude * signal;
  }
  const float tau = sample_time - delay;
  const float alpha = ricker_pi_frequency * tau;
  const float alpha_sq = alpha * alpha;
  return amplitude * (1.0f - 2.0f * alpha_sq) * expf(-alpha_sq);
}

// Sampled waveform in the normalized knot coordinate u (uniform tables have
// knots 0..count-1). Linear tables carry zero curvature, so one cubic-segment
// formula serves both interpolations. Zero outside [knots[0], knots[count-1]].
__device__ __forceinline__ float evaluate_waveform_table(
    float u,
    const float* __restrict__ knots,
    const float* __restrict__ values,
    const float* __restrict__ curvature,
    int count,
    int uniform) {
  if (!(u >= 0.0f) || u > knots[count - 1]) {
    return 0.0f;
  }
  int index;
  if (uniform != 0) {
    index = min(static_cast<int>(floorf(u)), count - 2);
  } else {
    int low = 0;
    int high = count - 1;
    while (high - low > 1) {
      const int middle = (low + high) >> 1;
      if (knots[middle] <= u) {
        low = middle;
      } else {
        high = middle;
      }
    }
    index = low;
  }
  const float u0 = knots[index];
  const float h = knots[index + 1] - u0;
  const float b = (u - u0) / h;
  const float a = 1.0f - b;
  return a * values[index] + b * values[index + 1] +
      ((a * a * a - a) * curvature[index] + (b * b * b - b) * curvature[index + 1]) * (h * h * (1.0f / 6.0f));
}

// Preserve the former scatter kernel's sm_86 contraction, including the
// second positive rotation of a corner image. Explicit intrinsics keep routing
// changes from changing which phase products are fused.
__device__ __forceinline__ float2 source_image_phase(
    float c, float s, float2 value, bool positive, bool second_corner = false) {
  const float cx = __fmul_rn(c, value.x), sy = __fmul_rn(s, value.y);
  if (!positive) return make_float2(__fadd_rn(cx, sy),
      __fsub_rn(__fmul_rn(c, value.y), __fmul_rn(s, value.x)));
  const float imag = second_corner ? __fmaf_rn(c, value.y, __fmul_rn(s, value.x)) :
      __fmaf_rn(s, value.x, __fmul_rn(c, value.y));
  return make_float2(__fsub_rn(cx, sy), imag);
}

__device__ __forceinline__ void add_real_direct(float* __restrict__ field, Shape3D shape, int i, int j, int k, float value) {
  if (in_bounds(i, j, k, shape)) {
    field[offset_shape(i, j, k, shape)] += value;
  }
}

template <bool CheckBounds>
__device__ __forceinline__ void add_real_patch_value(
    float* __restrict__ field,
    Shape3D shape,
    int i,
    int j,
    int k,
    float value) {
  if constexpr (CheckBounds) {
    add_real_direct(field, shape, i, j, k, value);
  } else {
    field[offset_shape(i, j, k, shape)] += value;
  }
}

bool patch_contained(Shape3D field_shape, Shape3D patch_shape, int64_t offset_i, int64_t offset_j, int64_t offset_k) {
  return offset_i >= 0 && offset_j >= 0 && offset_k >= 0
      && offset_i + static_cast<int64_t>(patch_shape.x) <= field_shape.x
      && offset_j + static_cast<int64_t>(patch_shape.y) <= field_shape.y
      && offset_k + static_cast<int64_t>(patch_shape.z) <= field_shape.z;
}

template <bool CheckBounds>
__global__ void add_source_patch_kernel(
    Shape3D field_shape,
    Shape3D patch_shape,
    int offset_i,
    int offset_j,
    int offset_k,
    float signal,
    float* __restrict__ field,
    const float* __restrict__ patch) {
  const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int i = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  if (!in_bounds(i, j, k, patch_shape)) {
    return;
  }
  const long long linear = offset_shape(i, j, k, patch_shape);
  const float value = signal * patch[linear];
  add_real_patch_value<CheckBounds>(field, field_shape, offset_i + i, offset_j + j, offset_k + k, value);
}

template <bool CheckBounds>
__global__ void add_cw_phased_source_patch_kernel(
    Shape3D field_shape,
    Shape3D patch_shape,
    int offset_i,
    int offset_j,
    int offset_k,
    float signal_cos,
    float signal_sin,
    float* __restrict__ field,
    const float* __restrict__ patch_cos,
    const float* __restrict__ patch_sin) {
  const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int i = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  if (!in_bounds(i, j, k, patch_shape)) {
    return;
  }
  const long long linear = offset_shape(i, j, k, patch_shape);
  const float value = signal_cos * patch_cos[linear] + signal_sin * patch_sin[linear];
  add_real_patch_value<CheckBounds>(field, field_shape, offset_i + i, offset_j + j, offset_k + k, value);
}

template <int TimeKind, bool CheckBounds>
__global__ void add_time_shifted_source_patch_kernel(
    Shape3D field_shape,
    Shape3D patch_shape,
    int offset_i,
    int offset_j,
    int offset_k,
    float time,
    float angular_frequency,
    float gaussian_inv_sigma,
    float ricker_pi_frequency,
    float amplitude,
    float phase,
    float delay,
    float pedestal,
    float pedestal_inv_sigma,
    int causal_gate,
    float* __restrict__ field,
    const float* __restrict__ patch,
    const float* __restrict__ delay_patch,
    const float* __restrict__ activation_delay_patch) {
  const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int i = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  if (!in_bounds(i, j, k, patch_shape)) {
    return;
  }
  const long long linear = offset_shape(i, j, k, patch_shape);
  if (causal_gate != 0) {
    if (time < activation_delay_patch[linear]) {
      return;
    }
  }
  const float sample_time = time - delay_patch[linear];
  const float signal = evaluate_source_time<TimeKind>(
      sample_time,
      angular_frequency,
      gaussian_inv_sigma,
      ricker_pi_frequency,
      amplitude,
      phase,
      delay,
      pedestal,
      pedestal_inv_sigma);
  add_real_patch_value<CheckBounds>(
      field, field_shape, offset_i + i, offset_j + j, offset_k + k, signal * patch[linear]);
}

template <bool CheckBounds>
__global__ void add_table_time_shifted_source_patch_kernel(
    Shape3D field_shape,
    Shape3D patch_shape,
    int offset_i,
    int offset_j,
    int offset_k,
    float time,
    float time_offset,
    float inv_step,
    float amplitude,
    int count,
    int uniform,
    int causal_gate,
    float* __restrict__ field,
    const float* __restrict__ patch,
    const float* __restrict__ delay_patch,
    const float* __restrict__ activation_delay_patch,
    const float* __restrict__ knots,
    const float* __restrict__ values,
    const float* __restrict__ curvature) {
  const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int i = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  if (!in_bounds(i, j, k, patch_shape)) {
    return;
  }
  const long long linear = offset_shape(i, j, k, patch_shape);
  if (causal_gate != 0) {
    if (time < activation_delay_patch[linear]) {
      return;
    }
  }
  const float u = (time_offset - delay_patch[linear]) * inv_step;
  const float signal = amplitude * evaluate_waveform_table(u, knots, values, curvature, count, uniform);
  add_real_patch_value<CheckBounds>(
      field, field_shape, offset_i + i, offset_j + j, offset_k + k, signal * patch[linear]);
}

template <int TimeKind, bool CheckBounds>
void launch_time_shifted_source_patch(
    Shape3D field_shape,
    Shape3D patch_shape,
    int offset_i,
    int offset_j,
    int offset_k,
    float time,
    float angular_frequency,
    float gaussian_inv_sigma,
    float ricker_pi_frequency,
    float amplitude,
    float phase,
    float delay,
    float pedestal,
    float pedestal_inv_sigma,
    int causal_gate,
    float* __restrict__ field,
    const float* __restrict__ patch,
    const float* __restrict__ delay_patch,
    const float* __restrict__ activation_delay_patch) {
  const dim3 block = patch_block3d();
  add_time_shifted_source_patch_kernel<TimeKind, CheckBounds><<<patch_grid3d(patch_shape, block), block, 0, current_cuda_stream()>>>(
      field_shape,
      patch_shape,
      offset_i,
      offset_j,
      offset_k,
      time,
      angular_frequency,
      gaussian_inv_sigma,
      ricker_pi_frequency,
      amplitude,
      phase,
      delay,
      pedestal,
      pedestal_inv_sigma,
      causal_gate,
      field,
      patch,
      delay_patch,
      activation_delay_patch);
}

__global__ void add_periodic_source_patch_kernel(SourceGather g,
    bool wrap, float signal, float* field, const float* patch) {
  const int64_t row = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (row >= g.count) return;
  const int64_t dst = g.targets[2*row+1];
  float value = field[dst];
  for (int64_t e = g.rows[row]; e < g.rows[row+1]; ++e) {
    const int64_t q = g.entries[2*e];
    // Offline sm_86 PTX confirms every wrapped specialization materialized
    // delta, including interior entries. Only the no-wrap specialization fused.
    if (wrap) {
      const float delta = __fmul_rn(signal, patch[q]);
      value = (g.entries[2*e+1] & 8) ? source_atomic_sum(value, delta) : __fadd_rn(value, delta);
    } else {
      value = __fmaf_rn(signal, patch[q], value);
    }
  }
  field[dst] = value;
}

__global__ void add_bloch_source_patch_kernel(SourceGather g, Shape3D patch_shape,
    int oi, int oj, int ok, int axis_a, int axis_b, float sr, float si,
    float ca, float sa, float cb, float sb, float* real, float* imag, const float* patch) {
  const int64_t row = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (row >= g.count) return;
  const int64_t dst = g.targets[2*row+1];
  float r = real[dst], im = imag[dst];
  for (int64_t e = g.rows[row]; e < g.rows[row+1]; ++e) {
    const int64_t q = g.entries[2*e], subset = g.entries[2*e+1];
    const int coords[3] = {int(q/(int64_t(patch_shape.y)*patch_shape.z))+oi,
        int((q/patch_shape.z)%patch_shape.y)+oj, int(q%patch_shape.z)+ok};
    float2 delta = make_float2(__fmul_rn(sr, patch[q]), __fmul_rn(si, patch[q]));
    if (subset & (1 << axis_a))
      delta = source_image_phase(ca, sa, delta, coords[axis_a] == 0);
    if (subset & (1 << axis_b))
      delta = source_image_phase(cb, sb, delta, coords[axis_b] == 0, subset & (1 << axis_a));
    r = (subset & 8) ? source_atomic_sum(r, delta.x) : __fadd_rn(r, delta.x);
    im = (subset & 8) ? source_atomic_sum(im, delta.y) : __fadd_rn(im, delta.y);
  }
  real[dst] = r;
  imag[dst] = im;
}

__global__ void add_scaled_slice_source_patch_kernel(
    Shape3D field_shape,
    Shape3D patch_shape,
    int sample_index,
    int offset_i,
    int offset_j,
    int offset_k,
    float scale,
    float* __restrict__ field,
    const float* __restrict__ patch,
    const float* __restrict__ incident) {
  const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int i = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  if (!in_bounds(i, j, k, patch_shape)) {
    return;
  }
  const long long linear = offset_shape(i, j, k, patch_shape);
  add_real_direct(
      field,
      field_shape,
      offset_i + i,
      offset_j + j,
      offset_k + k,
      scale * incident[sample_index] * patch[linear]);
}

template <int SampleAxis>
__global__ void add_scaled_line_source_patch_kernel(
    Shape3D field_shape,
    Shape3D patch_shape,
    int offset_i,
    int offset_j,
    int offset_k,
    float scale,
    float* __restrict__ field,
    const float* __restrict__ patch,
    const float* __restrict__ incident,
    const int* __restrict__ sample_indices) {
  const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int i = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  if (!in_bounds(i, j, k, patch_shape)) {
    return;
  }
  const long long linear = offset_shape(i, j, k, patch_shape);
  int sample_linear = k;
  if constexpr (SampleAxis == 0) {
    sample_linear = i;
  } else if constexpr (SampleAxis == 1) {
    sample_linear = j;
  }
  const int sample_index = sample_indices[sample_linear];
  add_real_direct(
      field,
      field_shape,
      offset_i + i,
      offset_j + j,
      offset_k + k,
      scale * incident[sample_index] * patch[linear]);
}

template <int SampleAxis>
void launch_scaled_line_source_patch(
    Shape3D field_shape,
    Shape3D patch_shape,
    int offset_i,
    int offset_j,
    int offset_k,
    float scale,
    float* __restrict__ field,
    const float* __restrict__ patch,
    const float* __restrict__ incident,
    const int* __restrict__ sample_indices) {
  const dim3 block = patch_block3d();
  add_scaled_line_source_patch_kernel<SampleAxis><<<patch_grid3d(patch_shape, block), block, 0, current_cuda_stream()>>>(
      field_shape,
      patch_shape,
      offset_i,
      offset_j,
      offset_k,
      scale,
      field,
      patch,
      incident,
      sample_indices);
}

__device__ __forceinline__ float interpolated_incident(
    const float* __restrict__ incident,
    int incident_count,
    float position,
    float origin,
    float inv_ds) {
  const float max_index = static_cast<float>(incident_count - 1);
  float coord = inv_ds > 0.0f ? (position - origin) * inv_ds : 0.0f;
  coord = fminf(fmaxf(coord, 0.0f), max_index);
  const int lower = static_cast<int>(coord);
  const int upper = min(lower + 1, incident_count - 1);
  const float frac = coord - static_cast<float>(lower);
  return incident[lower] + frac * (incident[upper] - incident[lower]);
}

__global__ void add_interpolated_source_patch_kernel(
    Shape3D field_shape,
    Shape3D patch_shape,
    int incident_count,
    float origin,
    float inv_ds,
    int offset_i,
    int offset_j,
    int offset_k,
    float scale,
    float* __restrict__ field,
    const float* __restrict__ patch,
    const float* __restrict__ incident,
    const float* __restrict__ sample_positions) {
  const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int i = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  if (!in_bounds(i, j, k, patch_shape)) {
    return;
  }
  const long long linear = offset_shape(i, j, k, patch_shape);
  const float sampled = interpolated_incident(incident, incident_count, sample_positions[linear], origin, inv_ds);
  add_real_direct(
      field,
      field_shape,
      offset_i + i,
      offset_j + j,
      offset_k + k,
      scale * sampled * patch[linear]);
}

template <bool Interpolated>
__global__ void add_batched_source_patches_kernel(SourceGather g, int incident_count,
    float origin, float inv_ds, float* field_x, float* field_y, float* field_z,
    const float* coeff, const float* incident, const float* positions, const int* indices) {
  const int64_t row = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (row >= g.count) return;
  const int64_t code = g.targets[2*row], dst = g.targets[2*row+1];
  float* field = code == 0 ? field_x : (code == 1 ? field_y : field_z);
  float value = field[dst];
  for (int64_t e = g.rows[row]; e < g.rows[row+1]; ++e) {
    const int64_t q = g.entries[2*e];
    const float sampled = Interpolated ?
        interpolated_incident(incident, incident_count, positions[q], origin, inv_ds) : incident[indices[q]];
    value = source_atomic_sum(value, __fmul_rn(coeff[q], sampled));
  }
  field[dst] = value;
}

__global__ void update_auxiliary_magnetic_kernel(
    int64_t total,
    float* __restrict__ magnetic,
    const float* __restrict__ electric,
    const float* __restrict__ decay,
    const float* __restrict__ curl) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  magnetic[index] = decay[index] * magnetic[index] - curl[index] * (electric[index + 1] - electric[index]);
}

__global__ void update_auxiliary_electric_kernel(
    int64_t total,
    int source_index,
    float source_value,
    float* __restrict__ electric,
    const float* __restrict__ magnetic,
    const float* __restrict__ decay,
    const float* __restrict__ curl) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  if (index + 1 == total) {
    electric[index] = 0.0f;
    return;
  }
  if (index == source_index) {
    electric[index] = source_value;
    return;
  }
  if (index > 0) {
    electric[index] = decay[index] * electric[index] - curl[index] * (magnetic[index] - magnetic[index - 1]);
  }
}

Shape3D shape3d(const torch::stable::Tensor& tensor) {
  STD_TORCH_CHECK(tensor.dim() == 3, "tensor must be 3D");
  return {static_cast<int>(tensor.size(0)), static_cast<int>(tensor.size(1)), static_cast<int>(tensor.size(2))};
}

void check_float_3d(const torch::stable::Tensor& tensor, const char* name) {
  check_float32_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  STD_TORCH_CHECK(tensor.dim() == 3, name, " must be 3D");
}

void check_float_1d(const torch::stable::Tensor& tensor, const char* name) {
  check_float32_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  STD_TORCH_CHECK(tensor.dim() == 1, name, " must be 1D");
}

void check_int32_tensor(const torch::stable::Tensor& tensor, const char* name) {
  check_cuda_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::Int, name, " must be int32");
}

void check_int32_or_int64_tensor(const torch::stable::Tensor& tensor, const char* name) {
  check_cuda_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Int || tensor.scalar_type() == torch::headeronly::ScalarType::Long,
      name,
      " must be int32 or int64");
}

void check_int64_tensor(const torch::stable::Tensor& tensor, const char* name) {
  check_cuda_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::Long, name, " must be int64");
}

void check_same_shape(const torch::stable::Tensor& reference, const torch::stable::Tensor& tensor, const char* name) {
  check_same_cuda_device(reference, tensor, name);
  STD_TORCH_CHECK(tensor.sizes().equals(reference.sizes()), name, " must match reference shape");
}

}  // namespace

void add_source_patch_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal) {
  check_float_3d(field, "field");
  check_float_3d(patch, "patch");
  check_same_cuda_device(field, patch, "patch");
  torch::stable::accelerator::DeviceGuard guard(field.get_device_index());
  const Shape3D field_shape = shape3d(field);
  const Shape3D patch_shape = shape3d(patch);
  const dim3 block = patch_block3d();
  const auto launch = [&](auto check_bounds_tag) {
    constexpr bool check_bounds = decltype(check_bounds_tag)::value;
    add_source_patch_kernel<check_bounds><<<patch_grid3d(patch_shape, block), block, 0, current_cuda_stream()>>>(
        field_shape,
        patch_shape,
        static_cast<int>(offset_i),
        static_cast<int>(offset_j),
        static_cast<int>(offset_k),
        static_cast<float>(signal),
        field.mutable_data_ptr<float>(),
        patch.mutable_data_ptr<float>());
  };
  if (patch_contained(field_shape, patch_shape, offset_i, offset_j, offset_k)) {
    launch(std::false_type{});
  } else {
    launch(std::true_type{});
  }
  WITWIN_CUDA_CHECK();
}

void add_cw_phased_source_patch_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch_cos,
    const torch::stable::Tensor& patch_sin,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal_cos,
    double signal_sin) {
  check_float_3d(field, "field");
  check_float_3d(patch_cos, "patch_cos");
  check_float_3d(patch_sin, "patch_sin");
  check_same_cuda_device(field, patch_cos, "patch_cos");
  check_same_shape(patch_cos, patch_sin, "patch_sin");
  torch::stable::accelerator::DeviceGuard guard(field.get_device_index());
  const Shape3D field_shape = shape3d(field);
  const Shape3D patch_shape = shape3d(patch_cos);
  const dim3 block = patch_block3d();
  const auto launch = [&](auto check_bounds_tag) {
    constexpr bool check_bounds = decltype(check_bounds_tag)::value;
    add_cw_phased_source_patch_kernel<check_bounds><<<patch_grid3d(patch_shape, block), block, 0, current_cuda_stream()>>>(
        field_shape,
        patch_shape,
        static_cast<int>(offset_i),
        static_cast<int>(offset_j),
        static_cast<int>(offset_k),
        static_cast<float>(signal_cos),
        static_cast<float>(signal_sin),
        field.mutable_data_ptr<float>(),
        patch_cos.mutable_data_ptr<float>(),
        patch_sin.mutable_data_ptr<float>());
  };
  if (patch_contained(field_shape, patch_shape, offset_i, offset_j, offset_k)) {
    launch(std::false_type{});
  } else {
    launch(std::true_type{});
  }
  WITWIN_CUDA_CHECK();
}

void add_time_shifted_source_patch_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch,
    const torch::stable::Tensor& delay_patch,
    const torch::stable::Tensor& activation_delay_patch,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    int64_t time_kind,
    double time,
    double frequency,
    double fwidth,
    double amplitude,
    double phase,
    double delay,
    double pedestal,
    double pedestal_fwidth,
    int64_t causal_gate) {
  check_float_3d(field, "field");
  check_float_3d(patch, "patch");
  check_float_3d(delay_patch, "delay_patch");
  check_float_3d(activation_delay_patch, "activation_delay_patch");
  check_same_cuda_device(field, patch, "patch");
  check_same_shape(patch, delay_patch, "delay_patch");
  check_same_shape(patch, activation_delay_patch, "activation_delay_patch");
  STD_TORCH_CHECK(time_kind >= 0 && time_kind <= 4 && time_kind != 3,
      "add_time_shifted_source_patch supports analytic time kinds 0, 1, 2 and 4");
  torch::stable::accelerator::DeviceGuard guard(field.get_device_index());
  constexpr float two_pi = 6.283185307179586f;
  constexpr float pi = 3.141592653589793f;
  const float frequency_f = static_cast<float>(frequency);
  const float angular_frequency = two_pi * frequency_f;
  const float gaussian_inv_sigma = fmaxf(two_pi * static_cast<float>(fwidth), 1.0e-30f);
  const float ricker_pi_frequency = pi * frequency_f;
  const float pedestal_inv_sigma = two_pi * static_cast<float>(pedestal_fwidth);
  const Shape3D field_shape = shape3d(field);
  const Shape3D patch_shape = shape3d(patch);
  const bool contained = patch_contained(field_shape, patch_shape, offset_i, offset_j, offset_k);
  const auto launch = [&](auto time_kind_tag, auto check_bounds_tag) {
    constexpr int time_kind_value = decltype(time_kind_tag)::value;
    constexpr bool check_bounds = decltype(check_bounds_tag)::value;
    launch_time_shifted_source_patch<time_kind_value, check_bounds>(
        field_shape,
        patch_shape,
        static_cast<int>(offset_i),
        static_cast<int>(offset_j),
        static_cast<int>(offset_k),
        static_cast<float>(time),
        angular_frequency,
        gaussian_inv_sigma,
        ricker_pi_frequency,
        static_cast<float>(amplitude),
        static_cast<float>(phase),
        static_cast<float>(delay),
        static_cast<float>(pedestal),
        pedestal_inv_sigma,
        static_cast<int>(causal_gate),
        field.mutable_data_ptr<float>(),
        patch.mutable_data_ptr<float>(),
        delay_patch.mutable_data_ptr<float>(),
        activation_delay_patch.mutable_data_ptr<float>());
  };
  if (time_kind == 0) {
    if (contained) {
      launch(std::integral_constant<int, 0>{}, std::false_type{});
    } else {
      launch(std::integral_constant<int, 0>{}, std::true_type{});
    }
  } else if (time_kind == 1) {
    if (contained) {
      launch(std::integral_constant<int, 1>{}, std::false_type{});
    } else {
      launch(std::integral_constant<int, 1>{}, std::true_type{});
    }
  } else if (time_kind == 4) {
    if (contained) {
      launch(std::integral_constant<int, 4>{}, std::false_type{});
    } else {
      launch(std::integral_constant<int, 4>{}, std::true_type{});
    }
  } else {
    if (contained) {
      launch(std::integral_constant<int, 2>{}, std::false_type{});
    } else {
      launch(std::integral_constant<int, 2>{}, std::true_type{});
    }
  }
  WITWIN_CUDA_CHECK();
}

void add_table_time_shifted_source_patch_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch,
    const torch::stable::Tensor& delay_patch,
    const torch::stable::Tensor& activation_delay_patch,
    const torch::stable::Tensor& knots,
    const torch::stable::Tensor& values,
    const torch::stable::Tensor& curvature,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double time,
    double time_offset,
    double inv_step,
    double amplitude,
    int64_t uniform,
    int64_t causal_gate) {
  check_float_3d(field, "field");
  check_float_3d(patch, "patch");
  check_float_3d(delay_patch, "delay_patch");
  check_float_3d(activation_delay_patch, "activation_delay_patch");
  check_same_cuda_device(field, patch, "patch");
  check_same_shape(patch, delay_patch, "delay_patch");
  check_same_shape(patch, activation_delay_patch, "activation_delay_patch");
  const int64_t count = knots.numel();
  for (const auto* table : {&knots, &values, &curvature}) {
    check_float32_tensor(*table, "waveform table");
    check_contiguous_tensor(*table, "waveform table");
    check_same_cuda_device(field, *table, "waveform table");
    STD_TORCH_CHECK(table->dim() == 1 && table->numel() == count, "waveform table tensors must be 1D of equal length");
  }
  STD_TORCH_CHECK(count >= 2 && count <= 2147483647LL, "waveform table needs between 2 and INT_MAX samples");
  torch::stable::accelerator::DeviceGuard guard(field.get_device_index());
  const Shape3D field_shape = shape3d(field);
  const Shape3D patch_shape = shape3d(patch);
  const auto launch = [&](auto check_bounds_tag) {
    constexpr bool check_bounds = decltype(check_bounds_tag)::value;
    const dim3 block = patch_block3d();
    add_table_time_shifted_source_patch_kernel<check_bounds><<<patch_grid3d(patch_shape, block), block, 0, current_cuda_stream()>>>(
        field_shape,
        patch_shape,
        static_cast<int>(offset_i),
        static_cast<int>(offset_j),
        static_cast<int>(offset_k),
        static_cast<float>(time),
        static_cast<float>(time_offset),
        static_cast<float>(inv_step),
        static_cast<float>(amplitude),
        static_cast<int>(count),
        static_cast<int>(uniform),
        static_cast<int>(causal_gate),
        field.mutable_data_ptr<float>(),
        patch.mutable_data_ptr<float>(),
        delay_patch.mutable_data_ptr<float>(),
        activation_delay_patch.mutable_data_ptr<float>(),
        knots.mutable_data_ptr<float>(),
        values.mutable_data_ptr<float>(),
        curvature.mutable_data_ptr<float>());
  };
  if (patch_contained(field_shape, patch_shape, offset_i, offset_j, offset_k)) {
    launch(std::false_type{});
  } else {
    launch(std::true_type{});
  }
  WITWIN_CUDA_CHECK();
}

void add_source_patch_periodic_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal,
    int64_t axis_a,
    int64_t axis_b,
    int64_t wrap_a,
    int64_t wrap_b,
    const torch::stable::Tensor& gather_rows, const torch::stable::Tensor& gather_targets,
    const torch::stable::Tensor& gather_entries) {
  check_float_3d(field, "field");
  check_float_3d(patch, "patch");
  check_same_cuda_device(field, patch, "patch");
  STD_TORCH_CHECK(axis_a >= 0 && axis_a < 3, "axis_a must be in [0, 3)");
  STD_TORCH_CHECK(axis_b >= 0 && axis_b < 3, "axis_b must be in [0, 3)");
  STD_TORCH_CHECK(axis_a != axis_b, "axis_a and axis_b must be distinct");
  torch::stable::accelerator::DeviceGuard guard(field.get_device_index());
  const auto gather = source_gather(field, gather_rows, gather_targets, gather_entries);
  if (!gather.count) return;
  add_periodic_source_patch_kernel<<<linear_grid(gather.count), 256, 0, current_cuda_stream()>>>(
      gather, wrap_a || wrap_b, static_cast<float>(signal),
      field.mutable_data_ptr<float>(), patch.const_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void add_source_patch_bloch_cuda(
    torch::stable::Tensor ex_real,
    torch::stable::Tensor ex_imag,
    torch::stable::Tensor ey_real,
    torch::stable::Tensor ey_imag,
    torch::stable::Tensor ez_real,
    torch::stable::Tensor ez_imag,
    const torch::stable::Tensor& patch,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal_real,
    double signal_imag,
    int64_t axis_code,
    double phase_cos_a,
    double phase_sin_a,
    double phase_cos_b,
    double phase_sin_b,
    int64_t wrap_axis_a,
    int64_t wrap_axis_b,
    const torch::stable::Tensor& gather_rows, const torch::stable::Tensor& gather_targets,
    const torch::stable::Tensor& gather_entries) {
  check_float_3d(ex_real, "ex_real");
  check_float_3d(ex_imag, "ex_imag");
  check_float_3d(ey_real, "ey_real");
  check_float_3d(ey_imag, "ey_imag");
  check_float_3d(ez_real, "ez_real");
  check_float_3d(ez_imag, "ez_imag");
  check_float_3d(patch, "patch");
  check_same_cuda_device(ex_real, ex_imag, "ex_imag");
  check_same_cuda_device(ex_real, ey_real, "ey_real");
  check_same_cuda_device(ex_real, ey_imag, "ey_imag");
  check_same_cuda_device(ex_real, ez_real, "ez_real");
  check_same_cuda_device(ex_real, ez_imag, "ez_imag");
  check_same_cuda_device(ex_real, patch, "patch");
  check_same_shape(ex_real, ex_imag, "ex_imag");
  check_same_shape(ey_real, ey_imag, "ey_imag");
  check_same_shape(ez_real, ez_imag, "ez_imag");
  STD_TORCH_CHECK(axis_code >= 0 && axis_code < 3, "axis_code must be in [0, 3)");
  torch::stable::accelerator::DeviceGuard guard(ex_real.get_device_index());
  auto real = axis_code == 0 ? ex_real : (axis_code == 1 ? ey_real : ez_real);
  auto imag = axis_code == 0 ? ex_imag : (axis_code == 1 ? ey_imag : ez_imag);
  const int axis_a = axis_code == 0 ? 1 : 0;
  const int axis_b = axis_code == 2 ? 1 : 2;
  const auto gather = source_gather(real, gather_rows, gather_targets, gather_entries);
  if (!gather.count) return;
  add_bloch_source_patch_kernel<<<linear_grid(gather.count), 256, 0, current_cuda_stream()>>>(
      gather, shape3d(patch), offset_i, offset_j, offset_k, axis_a, axis_b,
      signal_real, signal_imag, phase_cos_a, phase_sin_a, phase_cos_b, phase_sin_b,
      real.mutable_data_ptr<float>(), imag.mutable_data_ptr<float>(), patch.const_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void add_scaled_slice_source_patch_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch,
    const torch::stable::Tensor& incident,
    int64_t sample_index,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double scale) {
  check_float_3d(field, "field");
  check_float_3d(patch, "patch");
  check_float_1d(incident, "incident");
  check_same_cuda_device(field, patch, "patch");
  check_same_cuda_device(field, incident, "incident");
  STD_TORCH_CHECK(sample_index >= 0 && sample_index < incident.numel(), "sample_index is out of range");
  torch::stable::accelerator::DeviceGuard guard(field.get_device_index());
  const Shape3D patch_shape = shape3d(patch);
  const dim3 block = patch_block3d();
  add_scaled_slice_source_patch_kernel<<<patch_grid3d(patch_shape, block), block, 0, current_cuda_stream()>>>(
      shape3d(field),
      patch_shape,
      static_cast<int>(sample_index),
      static_cast<int>(offset_i),
      static_cast<int>(offset_j),
      static_cast<int>(offset_k),
      static_cast<float>(scale),
      field.mutable_data_ptr<float>(),
      patch.mutable_data_ptr<float>(),
      incident.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void add_scaled_line_source_patch_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch,
    const torch::stable::Tensor& incident,
    const torch::stable::Tensor& sample_indices,
    int64_t sample_axis,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double scale) {
  check_float_3d(field, "field");
  check_float_3d(patch, "patch");
  check_float_1d(incident, "incident");
  check_int32_tensor(sample_indices, "sample_indices");
  check_same_cuda_device(field, patch, "patch");
  check_same_cuda_device(field, incident, "incident");
  check_same_cuda_device(field, sample_indices, "sample_indices");
  STD_TORCH_CHECK(sample_axis >= 0 && sample_axis < 3, "sample_axis must be in [0, 3)");
  STD_TORCH_CHECK(sample_indices.numel() == patch.size(sample_axis), "sample_indices length must match selected patch axis");
  torch::stable::accelerator::DeviceGuard guard(field.get_device_index());
  if (sample_axis == 0) {
    launch_scaled_line_source_patch<0>(
        shape3d(field),
        shape3d(patch),
        static_cast<int>(offset_i),
        static_cast<int>(offset_j),
        static_cast<int>(offset_k),
        static_cast<float>(scale),
        field.mutable_data_ptr<float>(),
        patch.mutable_data_ptr<float>(),
        incident.mutable_data_ptr<float>(),
        sample_indices.mutable_data_ptr<int>());
  } else if (sample_axis == 1) {
    launch_scaled_line_source_patch<1>(
        shape3d(field),
        shape3d(patch),
        static_cast<int>(offset_i),
        static_cast<int>(offset_j),
        static_cast<int>(offset_k),
        static_cast<float>(scale),
        field.mutable_data_ptr<float>(),
        patch.mutable_data_ptr<float>(),
        incident.mutable_data_ptr<float>(),
        sample_indices.mutable_data_ptr<int>());
  } else {
    launch_scaled_line_source_patch<2>(
        shape3d(field),
        shape3d(patch),
        static_cast<int>(offset_i),
        static_cast<int>(offset_j),
        static_cast<int>(offset_k),
        static_cast<float>(scale),
        field.mutable_data_ptr<float>(),
        patch.mutable_data_ptr<float>(),
        incident.mutable_data_ptr<float>(),
        sample_indices.mutable_data_ptr<int>());
  }
  WITWIN_CUDA_CHECK();
}

void add_interpolated_source_patch_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch,
    const torch::stable::Tensor& incident,
    const torch::stable::Tensor& sample_positions,
    double origin,
    double ds,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double scale) {
  check_float_3d(field, "field");
  check_float_3d(patch, "patch");
  check_float_1d(incident, "incident");
  check_float_3d(sample_positions, "sample_positions");
  check_same_cuda_device(field, patch, "patch");
  check_same_cuda_device(field, incident, "incident");
  check_same_shape(patch, sample_positions, "sample_positions");
  torch::stable::accelerator::DeviceGuard guard(field.get_device_index());
  const float inv_ds = ds > 0.0 ? static_cast<float>(1.0 / ds) : 0.0f;
  const Shape3D patch_shape = shape3d(patch);
  const dim3 block = patch_block3d();
  add_interpolated_source_patch_kernel<<<patch_grid3d(patch_shape, block), block, 0, current_cuda_stream()>>>(
      shape3d(field),
      patch_shape,
      static_cast<int>(incident.numel()),
      static_cast<float>(origin),
      inv_ds,
      static_cast<int>(offset_i),
      static_cast<int>(offset_j),
      static_cast<int>(offset_k),
      static_cast<float>(scale),
      field.mutable_data_ptr<float>(),
      patch.mutable_data_ptr<float>(),
      incident.mutable_data_ptr<float>(),
      sample_positions.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void add_batched_reference_source_patches_cuda(
    torch::stable::Tensor field_x,
    torch::stable::Tensor field_y,
    torch::stable::Tensor field_z,
    const torch::stable::Tensor& coeff_data,
    const torch::stable::Tensor& incident,
    const torch::stable::Tensor& field_codes_per_coeff,
    const torch::stable::Tensor& field_offsets,
    const torch::stable::Tensor& sample_indices_per_coeff,
    const torch::stable::Tensor& gather_rows, const torch::stable::Tensor& gather_targets,
    const torch::stable::Tensor& gather_entries) {
  check_float_3d(field_x, "field_x");
  check_float_3d(field_y, "field_y");
  check_float_3d(field_z, "field_z");
  check_float_1d(coeff_data, "coeff_data");
  check_float_1d(incident, "incident");
  check_int32_tensor(field_codes_per_coeff, "field_codes_per_coeff");
  check_int32_or_int64_tensor(field_offsets, "field_offsets");
  check_int32_tensor(sample_indices_per_coeff, "sample_indices_per_coeff");
  check_same_cuda_device(field_x, field_y, "field_y");
  check_same_cuda_device(field_x, field_z, "field_z");
  check_same_cuda_device(field_x, coeff_data, "coeff_data");
  check_same_cuda_device(field_x, incident, "incident");
  check_same_cuda_device(field_x, field_codes_per_coeff, "field_codes_per_coeff");
  check_same_cuda_device(field_x, field_offsets, "field_offsets");
  check_same_cuda_device(field_x, sample_indices_per_coeff, "sample_indices_per_coeff");
  STD_TORCH_CHECK(field_codes_per_coeff.numel() == coeff_data.numel(), "field_codes_per_coeff length must match coeff_data");
  STD_TORCH_CHECK(field_offsets.numel() == coeff_data.numel(), "field_offsets length must match coeff_data");
  STD_TORCH_CHECK(sample_indices_per_coeff.numel() == coeff_data.numel(), "sample_indices_per_coeff length must match coeff_data");
  torch::stable::accelerator::DeviceGuard guard(field_x.get_device_index());
  const auto gather = source_gather(field_x, gather_rows, gather_targets, gather_entries);
  if (!gather.count) return;
  add_batched_source_patches_kernel<false><<<linear_grid(gather.count), 256, 0, current_cuda_stream()>>>(
      gather, incident.numel(), 0.0f, 0.0f,
      field_x.mutable_data_ptr<float>(), field_y.mutable_data_ptr<float>(), field_z.mutable_data_ptr<float>(),
      coeff_data.const_data_ptr<float>(), incident.const_data_ptr<float>(),
      nullptr,
      sample_indices_per_coeff.const_data_ptr<int>());
  WITWIN_CUDA_CHECK();
}

void add_batched_interpolated_source_patches_cuda(
    torch::stable::Tensor field_x,
    torch::stable::Tensor field_y,
    torch::stable::Tensor field_z,
    const torch::stable::Tensor& coeff_data,
    const torch::stable::Tensor& incident,
    const torch::stable::Tensor& sample_positions,
    const torch::stable::Tensor& field_codes_per_coeff,
    const torch::stable::Tensor& field_offsets,
    double origin,
    double ds,
    const torch::stable::Tensor& gather_rows, const torch::stable::Tensor& gather_targets,
    const torch::stable::Tensor& gather_entries) {
  check_float_3d(field_x, "field_x");
  check_float_3d(field_y, "field_y");
  check_float_3d(field_z, "field_z");
  check_float_1d(coeff_data, "coeff_data");
  check_float_1d(incident, "incident");
  check_float_1d(sample_positions, "sample_positions");
  check_int32_tensor(field_codes_per_coeff, "field_codes_per_coeff");
  check_int32_or_int64_tensor(field_offsets, "field_offsets");
  check_same_cuda_device(field_x, field_y, "field_y");
  check_same_cuda_device(field_x, field_z, "field_z");
  check_same_cuda_device(field_x, coeff_data, "coeff_data");
  check_same_cuda_device(field_x, incident, "incident");
  check_same_cuda_device(field_x, sample_positions, "sample_positions");
  check_same_cuda_device(field_x, field_codes_per_coeff, "field_codes_per_coeff");
  check_same_cuda_device(field_x, field_offsets, "field_offsets");
  STD_TORCH_CHECK(sample_positions.numel() == coeff_data.numel(), "sample_positions length must match coeff_data");
  STD_TORCH_CHECK(field_codes_per_coeff.numel() == coeff_data.numel(), "field_codes_per_coeff length must match coeff_data");
  STD_TORCH_CHECK(field_offsets.numel() == coeff_data.numel(), "field_offsets length must match coeff_data");
  torch::stable::accelerator::DeviceGuard guard(field_x.get_device_index());
  const float inv_ds = ds > 0.0 ? static_cast<float>(1.0 / ds) : 0.0f;
  const auto gather = source_gather(field_x, gather_rows, gather_targets, gather_entries);
  if (!gather.count) return;
  add_batched_source_patches_kernel<true><<<linear_grid(gather.count), 256, 0, current_cuda_stream()>>>(
      gather, incident.numel(), static_cast<float>(origin), inv_ds,
      field_x.mutable_data_ptr<float>(), field_y.mutable_data_ptr<float>(), field_z.mutable_data_ptr<float>(),
      coeff_data.const_data_ptr<float>(), incident.const_data_ptr<float>(),
      sample_positions.const_data_ptr<float>(),
      nullptr);
  WITWIN_CUDA_CHECK();
}

void update_auxiliary_magnetic_cuda(
    torch::stable::Tensor magnetic,
    const torch::stable::Tensor& electric,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl) {
  check_float_1d(magnetic, "magnetic");
  check_float_1d(electric, "electric");
  check_float_1d(decay, "decay");
  check_float_1d(curl, "curl");
  check_same_cuda_device(magnetic, electric, "electric");
  check_same_cuda_device(magnetic, decay, "decay");
  check_same_cuda_device(magnetic, curl, "curl");
  STD_TORCH_CHECK(decay.numel() == magnetic.numel(), "decay must match magnetic length");
  STD_TORCH_CHECK(curl.numel() == magnetic.numel(), "curl must match magnetic length");
  STD_TORCH_CHECK(electric.numel() == magnetic.numel() + 1, "electric length must be magnetic length + 1");
  torch::stable::accelerator::DeviceGuard guard(magnetic.get_device_index());
  update_auxiliary_magnetic_kernel<<<linear_grid(magnetic.numel()), 256, 0, current_cuda_stream()>>>(
      magnetic.numel(),
      magnetic.mutable_data_ptr<float>(),
      electric.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_auxiliary_electric_cuda(
    torch::stable::Tensor electric,
    const torch::stable::Tensor& magnetic,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    int64_t source_index,
    double source_value) {
  check_float_1d(electric, "electric");
  check_float_1d(magnetic, "magnetic");
  check_float_1d(decay, "decay");
  check_float_1d(curl, "curl");
  check_same_cuda_device(electric, magnetic, "magnetic");
  check_same_cuda_device(electric, decay, "decay");
  check_same_cuda_device(electric, curl, "curl");
  STD_TORCH_CHECK(decay.numel() == electric.numel(), "decay must match electric length");
  STD_TORCH_CHECK(curl.numel() == electric.numel(), "curl must match electric length");
  STD_TORCH_CHECK(magnetic.numel() + 1 == electric.numel(), "magnetic length must be electric length - 1");
  torch::stable::accelerator::DeviceGuard guard(electric.get_device_index());
  update_auxiliary_electric_kernel<<<linear_grid(electric.numel()), 256, 0, current_cuda_stream()>>>(
      electric.numel(),
      static_cast<int>(source_index),
      static_cast<float>(source_value),
      electric.mutable_data_ptr<float>(),
      magnetic.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      curl.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}
