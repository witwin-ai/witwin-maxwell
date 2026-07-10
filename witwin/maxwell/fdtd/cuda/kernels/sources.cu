#include <cmath>
#include <type_traits>

#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

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

template <int Axis>
__device__ __forceinline__ long long offset_replace_axis(int i, int j, int k, int value, Shape3D shape) {
  if constexpr (Axis == 0) {
    return offset_shape(value, j, k, shape);
  } else if constexpr (Axis == 1) {
    return offset_shape(i, value, k, shape);
  } else {
    return offset_shape(i, j, value, shape);
  }
}

template <int AxisA, int AxisB>
__device__ __forceinline__ long long offset_replace_two_axes(
    int i,
    int j,
    int k,
    int value_a,
    int value_b,
    Shape3D shape) {
  if constexpr (AxisA == 0) {
    i = value_a;
  } else if constexpr (AxisA == 1) {
    j = value_a;
  } else {
    k = value_a;
  }
  if constexpr (AxisB == 0) {
    i = value_b;
  } else if constexpr (AxisB == 1) {
    j = value_b;
  } else {
    k = value_b;
  }
  return offset_shape(i, j, k, shape);
}

__device__ __forceinline__ bool in_bounds(int i, int j, int k, Shape3D shape) {
  return i >= 0 && i < shape.x && j >= 0 && j < shape.y && k >= 0 && k < shape.z;
}

template <int TimeKind>
__device__ __forceinline__ float evaluate_source_time(
    float sample_time,
    float angular_frequency,
    float gaussian_inv_sigma,
    float ricker_pi_frequency,
    float amplitude,
    float phase,
    float delay) {
  if constexpr (TimeKind == 0) {
    return amplitude * cosf(angular_frequency * sample_time + phase);
  }
  if constexpr (TimeKind == 1) {
    const float tau = sample_time - delay;
    const float normalized = tau * gaussian_inv_sigma;
    const float envelope = expf(-0.5f * normalized * normalized);
    return amplitude * envelope * cosf(angular_frequency * tau + phase);
  }
  const float tau = sample_time - delay;
  const float alpha = ricker_pi_frequency * tau;
  const float alpha_sq = alpha * alpha;
  return amplitude * (1.0f - 2.0f * alpha_sq) * expf(-alpha_sq);
}

__device__ __forceinline__ float2 phase_positive(float phase_cos, float phase_sin, float2 value) {
  return make_float2(
      phase_cos * value.x - phase_sin * value.y,
      phase_sin * value.x + phase_cos * value.y);
}

__device__ __forceinline__ float2 phase_negative(float phase_cos, float phase_sin, float2 value) {
  return make_float2(
      phase_cos * value.x + phase_sin * value.y,
      phase_cos * value.y - phase_sin * value.x);
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
      delay);
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
      causal_gate,
      field,
      patch,
      delay_patch,
      activation_delay_patch);
}

template <int AxisA, int AxisB, bool WrapA, bool WrapB>
__global__ void add_periodic_source_patch_kernel(
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
  const int ci = offset_i + i;
  const int cj = offset_j + j;
  const int ck = offset_k + k;
  if (!in_bounds(ci, cj, ck, field_shape)) {
    return;
  }
  const int axis_a_coord = AxisA == 0 ? ci : (AxisA == 1 ? cj : ck);
  const int axis_b_coord = AxisB == 0 ? ci : (AxisB == 1 ? cj : ck);
  const int axis_a_size = AxisA == 0 ? field_shape.x : (AxisA == 1 ? field_shape.y : field_shape.z);
  const int axis_b_size = AxisB == 0 ? field_shape.x : (AxisB == 1 ? field_shape.y : field_shape.z);
  const float delta = signal * patch[linear];
  const long long field_linear = offset_shape(ci, cj, ck, field_shape);
  const bool boundary_a = WrapA && (axis_a_coord == 0 || axis_a_coord + 1 >= axis_a_size);
  const bool boundary_b = WrapB && (axis_b_coord == 0 || axis_b_coord + 1 >= axis_b_size);
  if constexpr (!WrapA && !WrapB) {
    field[field_linear] += delta;
    return;
  } else {
    if (!boundary_a && !boundary_b) {
      field[field_linear] += delta;
      return;
    }
    atomicAdd(field + field_linear, delta);
  }

  const int pair_a = axis_a_coord == 0 ? axis_a_size - 1 : 0;
  const int pair_b = axis_b_coord == 0 ? axis_b_size - 1 : 0;
  if (boundary_a) {
    atomicAdd(field + offset_replace_axis<AxisA>(ci, cj, ck, pair_a, field_shape), delta);
  }
  if (boundary_b) {
    atomicAdd(field + offset_replace_axis<AxisB>(ci, cj, ck, pair_b, field_shape), delta);
  }
  if (boundary_a && boundary_b) {
    atomicAdd(field + offset_replace_two_axes<AxisA, AxisB>(ci, cj, ck, pair_a, pair_b, field_shape), delta);
  }
}

template <int AxisA, int AxisB, bool WrapA, bool WrapB>
void launch_periodic_source_patch(
    Shape3D field_shape,
    Shape3D patch_shape,
    int offset_i,
    int offset_j,
    int offset_k,
    float signal,
    float* __restrict__ field,
    const float* __restrict__ patch) {
  const dim3 block = patch_block3d();
  add_periodic_source_patch_kernel<AxisA, AxisB, WrapA, WrapB><<<patch_grid3d(patch_shape, block), block, 0, current_cuda_stream()>>>(
      field_shape,
      patch_shape,
      offset_i,
      offset_j,
      offset_k,
      signal,
      field,
      patch);
}

template <int AxisA, int AxisB>
void dispatch_periodic_source_wraps(
    Shape3D field_shape,
    Shape3D patch_shape,
    int offset_i,
    int offset_j,
    int offset_k,
    float signal,
    int wrap_a,
    int wrap_b,
    float* __restrict__ field,
    const float* __restrict__ patch) {
  if (wrap_a != 0 && wrap_b != 0) {
    launch_periodic_source_patch<AxisA, AxisB, true, true>(
        field_shape, patch_shape, offset_i, offset_j, offset_k, signal, field, patch);
  } else if (wrap_a != 0) {
    launch_periodic_source_patch<AxisA, AxisB, true, false>(
        field_shape, patch_shape, offset_i, offset_j, offset_k, signal, field, patch);
  } else if (wrap_b != 0) {
    launch_periodic_source_patch<AxisA, AxisB, false, true>(
        field_shape, patch_shape, offset_i, offset_j, offset_k, signal, field, patch);
  } else {
    launch_periodic_source_patch<AxisA, AxisB, false, false>(
        field_shape, patch_shape, offset_i, offset_j, offset_k, signal, field, patch);
  }
}

template <int AxisA, int AxisB>
__global__ void add_bloch_source_patch_kernel(
    Shape3D field_shape,
    Shape3D patch_shape,
    int offset_i,
    int offset_j,
    int offset_k,
    float signal_real,
    float signal_imag,
    float phase_cos_a,
    float phase_sin_a,
    float phase_cos_b,
    float phase_sin_b,
    int wrap_a,
    int wrap_b,
    float* __restrict__ real,
    float* __restrict__ imag,
    const float* __restrict__ patch) {
  const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int i = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  if (!in_bounds(i, j, k, patch_shape)) {
    return;
  }
  const long long linear = offset_shape(i, j, k, patch_shape);
  const int ci = offset_i + i;
  const int cj = offset_j + j;
  const int ck = offset_k + k;
  if (!in_bounds(ci, cj, ck, field_shape)) {
    return;
  }

  const int axis_a_coord = AxisA == 0 ? ci : (AxisA == 1 ? cj : ck);
  const int axis_b_coord = AxisB == 0 ? ci : (AxisB == 1 ? cj : ck);
  const int axis_a_size = AxisA == 0 ? field_shape.x : (AxisA == 1 ? field_shape.y : field_shape.z);
  const int axis_b_size = AxisB == 0 ? field_shape.x : (AxisB == 1 ? field_shape.y : field_shape.z);
  const float amplitude = patch[linear];
  const float2 delta = make_float2(signal_real * amplitude, signal_imag * amplitude);
  const long long field_linear = offset_shape(ci, cj, ck, field_shape);
  const bool boundary_a = wrap_a != 0 && (axis_a_coord == 0 || axis_a_coord + 1 >= axis_a_size);
  const bool boundary_b = wrap_b != 0 && (axis_b_coord == 0 || axis_b_coord + 1 >= axis_b_size);
  if (!boundary_a && !boundary_b) {
    real[field_linear] += delta.x;
    imag[field_linear] += delta.y;
    return;
  }
  atomicAdd(real + field_linear, delta.x);
  atomicAdd(imag + field_linear, delta.y);

  const int pair_a = axis_a_coord == 0 ? axis_a_size - 1 : 0;
  const int pair_b = axis_b_coord == 0 ? axis_b_size - 1 : 0;
  if (boundary_a) {
    const float2 value = axis_a_coord == 0
        ? phase_positive(phase_cos_a, phase_sin_a, delta)
        : phase_negative(phase_cos_a, phase_sin_a, delta);
    const long long offset = offset_replace_axis<AxisA>(ci, cj, ck, pair_a, field_shape);
    atomicAdd(real + offset, value.x);
    atomicAdd(imag + offset, value.y);
  }
  if (boundary_b) {
    const float2 value = axis_b_coord == 0
        ? phase_positive(phase_cos_b, phase_sin_b, delta)
        : phase_negative(phase_cos_b, phase_sin_b, delta);
    const long long offset = offset_replace_axis<AxisB>(ci, cj, ck, pair_b, field_shape);
    atomicAdd(real + offset, value.x);
    atomicAdd(imag + offset, value.y);
  }
  if (boundary_a && boundary_b) {
    float2 value = axis_a_coord == 0
        ? phase_positive(phase_cos_a, phase_sin_a, delta)
        : phase_negative(phase_cos_a, phase_sin_a, delta);
    value = axis_b_coord == 0
        ? phase_positive(phase_cos_b, phase_sin_b, value)
        : phase_negative(phase_cos_b, phase_sin_b, value);
    const long long offset = offset_replace_two_axes<AxisA, AxisB>(ci, cj, ck, pair_a, pair_b, field_shape);
    atomicAdd(real + offset, value.x);
    atomicAdd(imag + offset, value.y);
  }
}

template <int AxisA, int AxisB>
void launch_bloch_source_patch(
    Shape3D field_shape,
    Shape3D patch_shape,
    int offset_i,
    int offset_j,
    int offset_k,
    float signal_real,
    float signal_imag,
    float phase_cos_a,
    float phase_sin_a,
    float phase_cos_b,
    float phase_sin_b,
    int wrap_a,
    int wrap_b,
    float* __restrict__ real,
    float* __restrict__ imag,
    const float* __restrict__ patch) {
  const dim3 block = patch_block3d();
  add_bloch_source_patch_kernel<AxisA, AxisB><<<patch_grid3d(patch_shape, block), block, 0, current_cuda_stream()>>>(
      field_shape,
      patch_shape,
      offset_i,
      offset_j,
      offset_k,
      signal_real,
      signal_imag,
      phase_cos_a,
      phase_sin_a,
      phase_cos_b,
      phase_sin_b,
      wrap_a,
      wrap_b,
      real,
      imag,
      patch);
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

template <typename OffsetT>
__device__ __forceinline__ void add_to_field_code_offset(
    int field_code,
    float* __restrict__ field_x,
    float* __restrict__ field_y,
    float* __restrict__ field_z,
    OffsetT offset,
    float value) {
  if (field_code == 0) {
    atomicAdd(field_x + offset, value);
  } else if (field_code == 1) {
    atomicAdd(field_y + offset, value);
  } else {
    atomicAdd(field_z + offset, value);
  }
}

__device__ __forceinline__ float warp_reduce_sum_active(float value, unsigned int mask) {
  const unsigned int lane = threadIdx.x & 31u;
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    const float other = __shfl_down_sync(mask, value, delta);
    if ((lane + static_cast<unsigned int>(delta)) < 32u && ((mask >> (lane + delta)) & 1u) != 0u) {
      value += other;
    }
  }
  return value;
}

__device__ __forceinline__ void add_to_field_code_offset_warp_aggregated(
    int field_code,
    float* __restrict__ field_x,
    float* __restrict__ field_y,
    float* __restrict__ field_z,
    int offset,
    float value) {
  const unsigned int mask = __activemask();
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned long long key =
      (static_cast<unsigned long long>(static_cast<unsigned int>(field_code)) << 32)
      | static_cast<unsigned int>(offset);
  const unsigned long long prev_key = __shfl_up_sync(mask, key, 1);
  const unsigned long long next_key = __shfl_down_sync(mask, key, 1);
  const bool adjacent_repeat =
      ((lane > 0u) && prev_key == key)
      || ((lane < 31u) && next_key == key);
  if (!__any_sync(mask, adjacent_repeat)) {
    add_to_field_code_offset(field_code, field_x, field_y, field_z, offset, value);
    return;
  }

  const unsigned int peers = __match_any_sync(mask, key);
  if ((peers & (peers - 1u)) == 0u) {
    add_to_field_code_offset(field_code, field_x, field_y, field_z, offset, value);
    return;
  }
  if (peers == mask) {
    const float sum = warp_reduce_sum_active(value, mask);
    if (lane == static_cast<unsigned int>(__ffs(peers) - 1)) {
      add_to_field_code_offset(field_code, field_x, field_y, field_z, offset, sum);
    }
    return;
  }

  float sum = 0.0f;
  unsigned int remaining = peers;
  while (remaining != 0u) {
    const int source_lane = __ffs(remaining) - 1;
    sum += __shfl_sync(peers, value, source_lane);
    remaining &= remaining - 1u;
  }
  if (static_cast<int>(lane) == (__ffs(peers) - 1)) {
    add_to_field_code_offset(field_code, field_x, field_y, field_z, offset, sum);
  }
}

template <typename OffsetT>
__global__ void add_batched_reference_source_patches_kernel(
    int64_t total,
    float* __restrict__ field_x,
    float* __restrict__ field_y,
    float* __restrict__ field_z,
    const float* __restrict__ coeff_data,
    const float* __restrict__ incident,
    const int* __restrict__ field_codes_per_coeff,
    const OffsetT* __restrict__ field_offsets,
    const int* __restrict__ sample_indices_per_coeff) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int sample_index = sample_indices_per_coeff[linear];
  const float value = coeff_data[linear] * incident[sample_index];
  add_to_field_code_offset(
      field_codes_per_coeff[linear],
      field_x,
      field_y,
      field_z,
      field_offsets[linear],
      value);
}

__global__ void add_batched_reference_source_patches_warp_kernel(
    int64_t total,
    float* __restrict__ field_x,
    float* __restrict__ field_y,
    float* __restrict__ field_z,
    const float* __restrict__ coeff_data,
    const float* __restrict__ incident,
    const int* __restrict__ field_codes_per_coeff,
    const int* __restrict__ field_offsets,
    const int* __restrict__ sample_indices_per_coeff) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int sample_index = sample_indices_per_coeff[linear];
  const float value = coeff_data[linear] * incident[sample_index];
  add_to_field_code_offset_warp_aggregated(
      field_codes_per_coeff[linear],
      field_x,
      field_y,
      field_z,
      field_offsets[linear],
      value);
}

template <typename OffsetT>
__global__ void add_batched_interpolated_source_patches_kernel(
    int64_t total,
    int incident_count,
    float origin,
    float inv_ds,
    float* __restrict__ field_x,
    float* __restrict__ field_y,
    float* __restrict__ field_z,
    const float* __restrict__ coeff_data,
    const float* __restrict__ incident,
    const float* __restrict__ sample_positions,
    const int* __restrict__ field_codes_per_coeff,
    const OffsetT* __restrict__ field_offsets) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const float sampled = interpolated_incident(incident, incident_count, sample_positions[linear], origin, inv_ds);
  const float value = coeff_data[linear] * sampled;
  add_to_field_code_offset(
      field_codes_per_coeff[linear],
      field_x,
      field_y,
      field_z,
      field_offsets[linear],
      value);
}

__global__ void add_batched_interpolated_source_patches_warp_kernel(
    int64_t total,
    int incident_count,
    float origin,
    float inv_ds,
    float* __restrict__ field_x,
    float* __restrict__ field_y,
    float* __restrict__ field_z,
    const float* __restrict__ coeff_data,
    const float* __restrict__ incident,
    const float* __restrict__ sample_positions,
    const int* __restrict__ field_codes_per_coeff,
    const int* __restrict__ field_offsets) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const float sampled = interpolated_incident(incident, incident_count, sample_positions[linear], origin, inv_ds);
  const float value = coeff_data[linear] * sampled;
  add_to_field_code_offset_warp_aggregated(
      field_codes_per_coeff[linear],
      field_x,
      field_y,
      field_z,
      field_offsets[linear],
      value);
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
    int64_t causal_gate) {
  check_float_3d(field, "field");
  check_float_3d(patch, "patch");
  check_float_3d(delay_patch, "delay_patch");
  check_float_3d(activation_delay_patch, "activation_delay_patch");
  check_same_cuda_device(field, patch, "patch");
  check_same_shape(patch, delay_patch, "delay_patch");
  check_same_shape(patch, activation_delay_patch, "activation_delay_patch");
  torch::stable::accelerator::DeviceGuard guard(field.get_device_index());
  constexpr float two_pi = 6.283185307179586f;
  constexpr float pi = 3.141592653589793f;
  const float frequency_f = static_cast<float>(frequency);
  const float angular_frequency = two_pi * frequency_f;
  const float gaussian_inv_sigma = fmaxf(two_pi * static_cast<float>(fwidth), 1.0e-30f);
  const float ricker_pi_frequency = pi * frequency_f;
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
  } else {
    if (contained) {
      launch(std::integral_constant<int, 2>{}, std::false_type{});
    } else {
      launch(std::integral_constant<int, 2>{}, std::true_type{});
    }
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
    int64_t wrap_b) {
  check_float_3d(field, "field");
  check_float_3d(patch, "patch");
  check_same_cuda_device(field, patch, "patch");
  STD_TORCH_CHECK(axis_a >= 0 && axis_a < 3, "axis_a must be in [0, 3)");
  STD_TORCH_CHECK(axis_b >= 0 && axis_b < 3, "axis_b must be in [0, 3)");
  STD_TORCH_CHECK(axis_a != axis_b, "axis_a and axis_b must be distinct");
  torch::stable::accelerator::DeviceGuard guard(field.get_device_index());
  const auto launch = [&](auto axis_a_tag, auto axis_b_tag) {
    constexpr int axis_a_value = decltype(axis_a_tag)::value;
    constexpr int axis_b_value = decltype(axis_b_tag)::value;
    dispatch_periodic_source_wraps<axis_a_value, axis_b_value>(
        shape3d(field),
        shape3d(patch),
        static_cast<int>(offset_i),
        static_cast<int>(offset_j),
        static_cast<int>(offset_k),
        static_cast<float>(signal),
        static_cast<int>(wrap_a),
        static_cast<int>(wrap_b),
        field.mutable_data_ptr<float>(),
        patch.mutable_data_ptr<float>());
  };
  if (axis_a == 0 && axis_b == 1) {
    launch(std::integral_constant<int, 0>{}, std::integral_constant<int, 1>{});
  } else if (axis_a == 0 && axis_b == 2) {
    launch(std::integral_constant<int, 0>{}, std::integral_constant<int, 2>{});
  } else if (axis_a == 1 && axis_b == 0) {
    launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 0>{});
  } else if (axis_a == 1 && axis_b == 2) {
    launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 2>{});
  } else if (axis_a == 2 && axis_b == 0) {
    launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 0>{});
  } else {
    launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{});
  }
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
    int64_t wrap_axis_b) {
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
  if (axis_code == 0) {
    launch_bloch_source_patch<1, 2>(
        shape3d(ex_real),
        shape3d(patch),
        static_cast<int>(offset_i),
        static_cast<int>(offset_j),
        static_cast<int>(offset_k),
        static_cast<float>(signal_real),
        static_cast<float>(signal_imag),
        static_cast<float>(phase_cos_a),
        static_cast<float>(phase_sin_a),
        static_cast<float>(phase_cos_b),
        static_cast<float>(phase_sin_b),
        static_cast<int>(wrap_axis_a),
        static_cast<int>(wrap_axis_b),
        ex_real.mutable_data_ptr<float>(),
        ex_imag.mutable_data_ptr<float>(),
        patch.mutable_data_ptr<float>());
  } else if (axis_code == 1) {
    launch_bloch_source_patch<0, 2>(
        shape3d(ey_real),
        shape3d(patch),
        static_cast<int>(offset_i),
        static_cast<int>(offset_j),
        static_cast<int>(offset_k),
        static_cast<float>(signal_real),
        static_cast<float>(signal_imag),
        static_cast<float>(phase_cos_a),
        static_cast<float>(phase_sin_a),
        static_cast<float>(phase_cos_b),
        static_cast<float>(phase_sin_b),
        static_cast<int>(wrap_axis_a),
        static_cast<int>(wrap_axis_b),
        ey_real.mutable_data_ptr<float>(),
        ey_imag.mutable_data_ptr<float>(),
        patch.mutable_data_ptr<float>());
  } else {
    launch_bloch_source_patch<0, 1>(
        shape3d(ez_real),
        shape3d(patch),
        static_cast<int>(offset_i),
        static_cast<int>(offset_j),
        static_cast<int>(offset_k),
        static_cast<float>(signal_real),
        static_cast<float>(signal_imag),
        static_cast<float>(phase_cos_a),
        static_cast<float>(phase_sin_a),
        static_cast<float>(phase_cos_b),
        static_cast<float>(phase_sin_b),
        static_cast<int>(wrap_axis_a),
        static_cast<int>(wrap_axis_b),
        ez_real.mutable_data_ptr<float>(),
        ez_imag.mutable_data_ptr<float>(),
        patch.mutable_data_ptr<float>());
  }
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
    const torch::stable::Tensor& sample_indices_per_coeff) {
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
  if (field_offsets.scalar_type() == torch::headeronly::ScalarType::Int) {
    add_batched_reference_source_patches_warp_kernel<<<linear_grid(coeff_data.numel()), 256, 0, current_cuda_stream()>>>(
        coeff_data.numel(),
        field_x.mutable_data_ptr<float>(),
        field_y.mutable_data_ptr<float>(),
        field_z.mutable_data_ptr<float>(),
        coeff_data.mutable_data_ptr<float>(),
        incident.mutable_data_ptr<float>(),
        field_codes_per_coeff.mutable_data_ptr<int>(),
        field_offsets.mutable_data_ptr<int>(),
        sample_indices_per_coeff.mutable_data_ptr<int>());
  } else {
    add_batched_reference_source_patches_kernel<int64_t><<<linear_grid(coeff_data.numel()), 256, 0, current_cuda_stream()>>>(
        coeff_data.numel(),
        field_x.mutable_data_ptr<float>(),
        field_y.mutable_data_ptr<float>(),
        field_z.mutable_data_ptr<float>(),
        coeff_data.mutable_data_ptr<float>(),
        incident.mutable_data_ptr<float>(),
        field_codes_per_coeff.mutable_data_ptr<int>(),
        field_offsets.mutable_data_ptr<int64_t>(),
        sample_indices_per_coeff.mutable_data_ptr<int>());
  }
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
    double ds) {
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
  if (field_offsets.scalar_type() == torch::headeronly::ScalarType::Int) {
    add_batched_interpolated_source_patches_warp_kernel<<<linear_grid(coeff_data.numel()), 256, 0, current_cuda_stream()>>>(
        coeff_data.numel(),
        static_cast<int>(incident.numel()),
        static_cast<float>(origin),
        inv_ds,
        field_x.mutable_data_ptr<float>(),
        field_y.mutable_data_ptr<float>(),
        field_z.mutable_data_ptr<float>(),
        coeff_data.mutable_data_ptr<float>(),
        incident.mutable_data_ptr<float>(),
        sample_positions.mutable_data_ptr<float>(),
        field_codes_per_coeff.mutable_data_ptr<int>(),
        field_offsets.mutable_data_ptr<int>());
  } else {
    add_batched_interpolated_source_patches_kernel<int64_t><<<linear_grid(coeff_data.numel()), 256, 0, current_cuda_stream()>>>(
        coeff_data.numel(),
        static_cast<int>(incident.numel()),
        static_cast<float>(origin),
        inv_ds,
        field_x.mutable_data_ptr<float>(),
        field_y.mutable_data_ptr<float>(),
        field_z.mutable_data_ptr<float>(),
        coeff_data.mutable_data_ptr<float>(),
        incident.mutable_data_ptr<float>(),
        sample_positions.mutable_data_ptr<float>(),
        field_codes_per_coeff.mutable_data_ptr<int>(),
        field_offsets.mutable_data_ptr<int64_t>());
  }
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
