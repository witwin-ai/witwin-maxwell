#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

namespace {

struct Shape3D {
  int x;
  int y;
  int z;
};

__device__ inline long long offset_shape(int i, int j, int k, Shape3D shape) {
  return (static_cast<long long>(i) * shape.y + j) * shape.z + k;
}

__device__ inline bool in_bounds(int i, int j, int k, Shape3D shape) {
  return i >= 0 && i < shape.x && j >= 0 && j < shape.y && k >= 0 && k < shape.z;
}

__device__ inline float evaluate_source_time(
    int time_kind,
    float sample_time,
    float frequency,
    float fwidth,
    float amplitude,
    float phase,
    float delay) {
  constexpr float two_pi = 6.283185307179586f;
  if (time_kind == 0) {
    return amplitude * cosf(two_pi * frequency * sample_time + phase);
  }
  if (time_kind == 1) {
    const float sigma_t = 1.0f / fmaxf(two_pi * fwidth, 1.0e-30f);
    const float tau = sample_time - delay;
    const float normalized = tau / sigma_t;
    const float envelope = expf(-0.5f * normalized * normalized);
    return amplitude * envelope * cosf(two_pi * frequency * tau + phase);
  }
  const float tau = sample_time - delay;
  const float alpha = 3.141592653589793f * frequency * tau;
  const float alpha_sq = alpha * alpha;
  return amplitude * (1.0f - 2.0f * alpha_sq) * expf(-alpha_sq);
}

__device__ inline float2 phase_positive(float phase_cos, float phase_sin, float2 value) {
  return make_float2(
      phase_cos * value.x - phase_sin * value.y,
      phase_sin * value.x + phase_cos * value.y);
}

__device__ inline float2 phase_negative(float phase_cos, float phase_sin, float2 value) {
  return make_float2(
      phase_cos * value.x + phase_sin * value.y,
      phase_cos * value.y - phase_sin * value.x);
}

__device__ inline void add_real(float* field, Shape3D shape, int i, int j, int k, float value) {
  if (in_bounds(i, j, k, shape)) {
    atomicAdd(field + offset_shape(i, j, k, shape), value);
  }
}

__device__ inline void add_complex(
    float* real,
    float* imag,
    Shape3D shape,
    int i,
    int j,
    int k,
    float2 value) {
  if (in_bounds(i, j, k, shape)) {
    const long long offset = offset_shape(i, j, k, shape);
    atomicAdd(real + offset, value.x);
    atomicAdd(imag + offset, value.y);
  }
}

__global__ void add_source_patch_kernel(
    int64_t total,
    Shape3D field_shape,
    Shape3D patch_shape,
    int offset_i,
    int offset_j,
    int offset_k,
    float signal,
    float* field,
    const float* patch) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D local = unflatten3d(linear, patch_shape.y, patch_shape.z);
  const float value = signal * patch[linear];
  add_real(field, field_shape, offset_i + local.i, offset_j + local.j, offset_k + local.k, value);
}

__global__ void add_cw_phased_source_patch_kernel(
    int64_t total,
    Shape3D field_shape,
    Shape3D patch_shape,
    int offset_i,
    int offset_j,
    int offset_k,
    float signal_cos,
    float signal_sin,
    float* field,
    const float* patch_cos,
    const float* patch_sin) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D local = unflatten3d(linear, patch_shape.y, patch_shape.z);
  const float value = signal_cos * patch_cos[linear] + signal_sin * patch_sin[linear];
  add_real(field, field_shape, offset_i + local.i, offset_j + local.j, offset_k + local.k, value);
}

__global__ void add_time_shifted_source_patch_kernel(
    int64_t total,
    Shape3D field_shape,
    Shape3D patch_shape,
    int offset_i,
    int offset_j,
    int offset_k,
    int time_kind,
    float time,
    float frequency,
    float fwidth,
    float amplitude,
    float phase,
    float delay,
    int causal_gate,
    float* field,
    const float* patch,
    const float* delay_patch,
    const float* activation_delay_patch) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  if (causal_gate != 0 && time < activation_delay_patch[linear]) {
    return;
  }
  const Index3D local = unflatten3d(linear, patch_shape.y, patch_shape.z);
  const float sample_time = time - delay_patch[linear];
  const float signal = evaluate_source_time(time_kind, sample_time, frequency, fwidth, amplitude, phase, delay);
  add_real(field, field_shape, offset_i + local.i, offset_j + local.j, offset_k + local.k, signal * patch[linear]);
}

__global__ void add_periodic_source_patch_kernel(
    int64_t total,
    Shape3D field_shape,
    Shape3D patch_shape,
    int offset_i,
    int offset_j,
    int offset_k,
    float signal,
    int axis_a,
    int axis_b,
    int wrap_a,
    int wrap_b,
    float* field,
    const float* patch) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D local = unflatten3d(linear, patch_shape.y, patch_shape.z);
  int coords[3] = {offset_i + static_cast<int>(local.i), offset_j + static_cast<int>(local.j), offset_k + static_cast<int>(local.k)};
  if (!in_bounds(coords[0], coords[1], coords[2], field_shape)) {
    return;
  }
  const int sizes[3] = {field_shape.x, field_shape.y, field_shape.z};
  const float delta = signal * patch[linear];
  add_real(field, field_shape, coords[0], coords[1], coords[2], delta);

  const bool boundary_a = wrap_a != 0 && (coords[axis_a] == 0 || coords[axis_a] + 1 >= sizes[axis_a]);
  const bool boundary_b = wrap_b != 0 && (coords[axis_b] == 0 || coords[axis_b] + 1 >= sizes[axis_b]);
  const int pair_a = coords[axis_a] == 0 ? sizes[axis_a] - 1 : 0;
  const int pair_b = coords[axis_b] == 0 ? sizes[axis_b] - 1 : 0;
  if (boundary_a) {
    int dst[3] = {coords[0], coords[1], coords[2]};
    dst[axis_a] = pair_a;
    add_real(field, field_shape, dst[0], dst[1], dst[2], delta);
  }
  if (boundary_b) {
    int dst[3] = {coords[0], coords[1], coords[2]};
    dst[axis_b] = pair_b;
    add_real(field, field_shape, dst[0], dst[1], dst[2], delta);
  }
  if (boundary_a && boundary_b) {
    int dst[3] = {coords[0], coords[1], coords[2]};
    dst[axis_a] = pair_a;
    dst[axis_b] = pair_b;
    add_real(field, field_shape, dst[0], dst[1], dst[2], delta);
  }
}

__global__ void add_bloch_source_patch_kernel(
    int64_t total,
    Shape3D field_shape,
    Shape3D patch_shape,
    int offset_i,
    int offset_j,
    int offset_k,
    float signal_real,
    float signal_imag,
    int axis_code,
    float phase_cos_a,
    float phase_sin_a,
    float phase_cos_b,
    float phase_sin_b,
    float* ex_real,
    float* ex_imag,
    float* ey_real,
    float* ey_imag,
    float* ez_real,
    float* ez_imag,
    const float* patch) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D local = unflatten3d(linear, patch_shape.y, patch_shape.z);
  int coords[3] = {offset_i + static_cast<int>(local.i), offset_j + static_cast<int>(local.j), offset_k + static_cast<int>(local.k)};
  if (!in_bounds(coords[0], coords[1], coords[2], field_shape)) {
    return;
  }

  float* real = ex_real;
  float* imag = ex_imag;
  int axis_a = 1;
  int axis_b = 2;
  if (axis_code == 1) {
    real = ey_real;
    imag = ey_imag;
    axis_a = 0;
    axis_b = 2;
  } else if (axis_code == 2) {
    real = ez_real;
    imag = ez_imag;
    axis_a = 0;
    axis_b = 1;
  }

  const int sizes[3] = {field_shape.x, field_shape.y, field_shape.z};
  const float amplitude = patch[linear];
  const float2 delta = make_float2(signal_real * amplitude, signal_imag * amplitude);
  add_complex(real, imag, field_shape, coords[0], coords[1], coords[2], delta);

  const bool boundary_a = coords[axis_a] == 0 || coords[axis_a] + 1 >= sizes[axis_a];
  const bool boundary_b = coords[axis_b] == 0 || coords[axis_b] + 1 >= sizes[axis_b];
  const int pair_a = coords[axis_a] == 0 ? sizes[axis_a] - 1 : 0;
  const int pair_b = coords[axis_b] == 0 ? sizes[axis_b] - 1 : 0;
  if (boundary_a) {
    int dst[3] = {coords[0], coords[1], coords[2]};
    dst[axis_a] = pair_a;
    const float2 value = coords[axis_a] == 0
        ? phase_positive(phase_cos_a, phase_sin_a, delta)
        : phase_negative(phase_cos_a, phase_sin_a, delta);
    add_complex(real, imag, field_shape, dst[0], dst[1], dst[2], value);
  }
  if (boundary_b) {
    int dst[3] = {coords[0], coords[1], coords[2]};
    dst[axis_b] = pair_b;
    const float2 value = coords[axis_b] == 0
        ? phase_positive(phase_cos_b, phase_sin_b, delta)
        : phase_negative(phase_cos_b, phase_sin_b, delta);
    add_complex(real, imag, field_shape, dst[0], dst[1], dst[2], value);
  }
  if (boundary_a && boundary_b) {
    int dst[3] = {coords[0], coords[1], coords[2]};
    dst[axis_a] = pair_a;
    dst[axis_b] = pair_b;
    float2 value = coords[axis_a] == 0
        ? phase_positive(phase_cos_a, phase_sin_a, delta)
        : phase_negative(phase_cos_a, phase_sin_a, delta);
    value = coords[axis_b] == 0
        ? phase_positive(phase_cos_b, phase_sin_b, value)
        : phase_negative(phase_cos_b, phase_sin_b, value);
    add_complex(real, imag, field_shape, dst[0], dst[1], dst[2], value);
  }
}

__global__ void add_scaled_slice_source_patch_kernel(
    int64_t total,
    Shape3D field_shape,
    Shape3D patch_shape,
    int sample_index,
    int offset_i,
    int offset_j,
    int offset_k,
    float scale,
    float* field,
    const float* patch,
    const float* incident) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D local = unflatten3d(linear, patch_shape.y, patch_shape.z);
  add_real(
      field,
      field_shape,
      offset_i + local.i,
      offset_j + local.j,
      offset_k + local.k,
      scale * incident[sample_index] * patch[linear]);
}

__global__ void add_scaled_line_source_patch_kernel(
    int64_t total,
    Shape3D field_shape,
    Shape3D patch_shape,
    int sample_axis,
    int offset_i,
    int offset_j,
    int offset_k,
    float scale,
    float* field,
    const float* patch,
    const float* incident,
    const int* sample_indices) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D local = unflatten3d(linear, patch_shape.y, patch_shape.z);
  const int sample_linear = sample_axis == 0 ? local.i : (sample_axis == 1 ? local.j : local.k);
  const int sample_index = sample_indices[sample_linear];
  add_real(
      field,
      field_shape,
      offset_i + local.i,
      offset_j + local.j,
      offset_k + local.k,
      scale * incident[sample_index] * patch[linear]);
}

__device__ inline float interpolated_incident(const float* incident, int incident_count, float position, float origin, float ds) {
  const float max_index = static_cast<float>(incident_count - 1);
  float coord = ds > 0.0f ? (position - origin) / ds : 0.0f;
  coord = fminf(fmaxf(coord, 0.0f), max_index);
  const int lower = static_cast<int>(floorf(coord));
  const int upper = min(lower + 1, incident_count - 1);
  const float frac = coord - static_cast<float>(lower);
  return incident[lower] + frac * (incident[upper] - incident[lower]);
}

__global__ void add_interpolated_source_patch_kernel(
    int64_t total,
    Shape3D field_shape,
    Shape3D patch_shape,
    int incident_count,
    float origin,
    float ds,
    int offset_i,
    int offset_j,
    int offset_k,
    float scale,
    float* field,
    const float* patch,
    const float* incident,
    const float* sample_positions) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D local = unflatten3d(linear, patch_shape.y, patch_shape.z);
  const float sampled = interpolated_incident(incident, incident_count, sample_positions[linear], origin, ds);
  add_real(
      field,
      field_shape,
      offset_i + local.i,
      offset_j + local.j,
      offset_k + local.k,
      scale * sampled * patch[linear]);
}

__device__ inline int resolve_term(int linear, const int* term_starts, int term_count, int total_count) {
  for (int term = 0; term < term_count; ++term) {
    const int end = term + 1 < term_count ? term_starts[term + 1] : total_count;
    if (linear < end) {
      return term;
    }
  }
  return term_count;
}

__device__ inline void add_to_field_code(
    int field_code,
    Shape3D x_shape,
    Shape3D y_shape,
    Shape3D z_shape,
    float* field_x,
    float* field_y,
    float* field_z,
    int i,
    int j,
    int k,
    float value) {
  if (field_code == 0) {
    add_real(field_x, x_shape, i, j, k, value);
  } else if (field_code == 1) {
    add_real(field_y, y_shape, i, j, k, value);
  } else {
    add_real(field_z, z_shape, i, j, k, value);
  }
}

__global__ void add_batched_reference_source_patches_kernel(
    int64_t total,
    int term_count,
    Shape3D x_shape,
    Shape3D y_shape,
    Shape3D z_shape,
    float* field_x,
    float* field_y,
    float* field_z,
    const float* coeff_data,
    const float* incident,
    const int* term_starts,
    const int* term_shapes,
    const int* term_offsets,
    const int* field_codes,
    const int* sample_axis_codes,
    const int* sample_index_starts,
    const int* sample_indices) {
  const int linear = static_cast<int>(static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (linear >= total) {
    return;
  }
  const int term = resolve_term(linear, term_starts, term_count, static_cast<int>(total));
  if (term >= term_count) {
    return;
  }
  const int start = term_starts[term];
  const int local_linear = linear - start;
  const int sx = term_shapes[term * 3 + 0];
  const int sy = term_shapes[term * 3 + 1];
  const int sz = term_shapes[term * 3 + 2];
  const Index3D local = unflatten3d(local_linear, sy, sz);
  const int axis = sample_axis_codes[term];
  const int sample_linear = axis == 0 ? local.i : (axis == 1 ? local.j : local.k);
  const int sample_index = sample_indices[sample_index_starts[term] + sample_linear];
  const float value = coeff_data[linear] * incident[sample_index];
  add_to_field_code(
      field_codes[term],
      x_shape,
      y_shape,
      z_shape,
      field_x,
      field_y,
      field_z,
      static_cast<int>(local.i) + term_offsets[term * 3 + 0],
      static_cast<int>(local.j) + term_offsets[term * 3 + 1],
      static_cast<int>(local.k) + term_offsets[term * 3 + 2],
      value);
  (void)sx;
}

__global__ void add_batched_interpolated_source_patches_kernel(
    int64_t total,
    int term_count,
    int incident_count,
    Shape3D x_shape,
    Shape3D y_shape,
    Shape3D z_shape,
    float origin,
    float ds,
    float* field_x,
    float* field_y,
    float* field_z,
    const float* coeff_data,
    const float* incident,
    const float* sample_positions,
    const int* term_starts,
    const int* term_shapes,
    const int* term_offsets,
    const int* field_codes) {
  const int linear = static_cast<int>(static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (linear >= total) {
    return;
  }
  const int term = resolve_term(linear, term_starts, term_count, static_cast<int>(total));
  if (term >= term_count) {
    return;
  }
  const int start = term_starts[term];
  const int local_linear = linear - start;
  const int sy = term_shapes[term * 3 + 1];
  const int sz = term_shapes[term * 3 + 2];
  const Index3D local = unflatten3d(local_linear, sy, sz);
  const float sampled = interpolated_incident(incident, incident_count, sample_positions[linear], origin, ds);
  const float value = coeff_data[linear] * sampled;
  add_to_field_code(
      field_codes[term],
      x_shape,
      y_shape,
      z_shape,
      field_x,
      field_y,
      field_z,
      static_cast<int>(local.i) + term_offsets[term * 3 + 0],
      static_cast<int>(local.j) + term_offsets[term * 3 + 1],
      static_cast<int>(local.k) + term_offsets[term * 3 + 2],
      value);
}

__global__ void update_auxiliary_magnetic_kernel(
    int64_t total,
    float* magnetic,
    const float* electric,
    const float* decay,
    const float* curl) {
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
    float* electric,
    const float* magnetic,
    const float* decay,
    const float* curl) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  if (index + 1 == total) {
    electric[index] = 0.0f;
    return;
  }
  if (index > 0 && index != source_index) {
    electric[index] = decay[index] * electric[index] - curl[index] * (magnetic[index] - magnetic[index - 1]);
  }
  if (index == source_index) {
    electric[index] = source_value;
  }
}

Shape3D shape3d(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.dim() == 3, "tensor must be 3D");
  return {static_cast<int>(tensor.size(0)), static_cast<int>(tensor.size(1)), static_cast<int>(tensor.size(2))};
}

void check_float_3d(const at::Tensor& tensor, const char* name) {
  check_float32_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  TORCH_CHECK(tensor.dim() == 3, name, " must be 3D");
}

void check_float_1d(const at::Tensor& tensor, const char* name) {
  check_float32_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1D");
}

void check_int32_tensor(const at::Tensor& tensor, const char* name) {
  check_cuda_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
}

void check_same_shape(const at::Tensor& reference, const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.sizes() == reference.sizes(), name, " must match reference shape");
}

}  // namespace

void add_source_patch_cuda(
    at::Tensor field,
    const at::Tensor& patch,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal) {
  check_float_3d(field, "field");
  check_float_3d(patch, "patch");
  c10::cuda::CUDAGuard guard(field.device());
  add_source_patch_kernel<<<linear_grid(patch.numel()), 256, 0, current_cuda_stream()>>>(
      patch.numel(),
      shape3d(field),
      shape3d(patch),
      static_cast<int>(offset_i),
      static_cast<int>(offset_j),
      static_cast<int>(offset_k),
      static_cast<float>(signal),
      field.data_ptr<float>(),
      patch.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void add_cw_phased_source_patch_cuda(
    at::Tensor field,
    const at::Tensor& patch_cos,
    const at::Tensor& patch_sin,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal_cos,
    double signal_sin) {
  check_float_3d(field, "field");
  check_float_3d(patch_cos, "patch_cos");
  check_float_3d(patch_sin, "patch_sin");
  check_same_shape(patch_cos, patch_sin, "patch_sin");
  c10::cuda::CUDAGuard guard(field.device());
  add_cw_phased_source_patch_kernel<<<linear_grid(patch_cos.numel()), 256, 0, current_cuda_stream()>>>(
      patch_cos.numel(),
      shape3d(field),
      shape3d(patch_cos),
      static_cast<int>(offset_i),
      static_cast<int>(offset_j),
      static_cast<int>(offset_k),
      static_cast<float>(signal_cos),
      static_cast<float>(signal_sin),
      field.data_ptr<float>(),
      patch_cos.data_ptr<float>(),
      patch_sin.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void add_time_shifted_source_patch_cuda(
    at::Tensor field,
    const at::Tensor& patch,
    const at::Tensor& delay_patch,
    const at::Tensor& activation_delay_patch,
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
  check_same_shape(patch, delay_patch, "delay_patch");
  check_same_shape(patch, activation_delay_patch, "activation_delay_patch");
  c10::cuda::CUDAGuard guard(field.device());
  add_time_shifted_source_patch_kernel<<<linear_grid(patch.numel()), 256, 0, current_cuda_stream()>>>(
      patch.numel(),
      shape3d(field),
      shape3d(patch),
      static_cast<int>(offset_i),
      static_cast<int>(offset_j),
      static_cast<int>(offset_k),
      static_cast<int>(time_kind),
      static_cast<float>(time),
      static_cast<float>(frequency),
      static_cast<float>(fwidth),
      static_cast<float>(amplitude),
      static_cast<float>(phase),
      static_cast<float>(delay),
      static_cast<int>(causal_gate),
      field.data_ptr<float>(),
      patch.data_ptr<float>(),
      delay_patch.data_ptr<float>(),
      activation_delay_patch.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void add_source_patch_periodic_cuda(
    at::Tensor field,
    const at::Tensor& patch,
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
  c10::cuda::CUDAGuard guard(field.device());
  add_periodic_source_patch_kernel<<<linear_grid(patch.numel()), 256, 0, current_cuda_stream()>>>(
      patch.numel(),
      shape3d(field),
      shape3d(patch),
      static_cast<int>(offset_i),
      static_cast<int>(offset_j),
      static_cast<int>(offset_k),
      static_cast<float>(signal),
      static_cast<int>(axis_a),
      static_cast<int>(axis_b),
      static_cast<int>(wrap_a),
      static_cast<int>(wrap_b),
      field.data_ptr<float>(),
      patch.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void add_source_patch_bloch_cuda(
    at::Tensor ex_real,
    at::Tensor ex_imag,
    at::Tensor ey_real,
    at::Tensor ey_imag,
    at::Tensor ez_real,
    at::Tensor ez_imag,
    const at::Tensor& patch,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal_real,
    double signal_imag,
    int64_t axis_code,
    double phase_cos_a,
    double phase_sin_a,
    double phase_cos_b,
    double phase_sin_b) {
  check_float_3d(ex_real, "ex_real");
  check_float_3d(ex_imag, "ex_imag");
  check_float_3d(ey_real, "ey_real");
  check_float_3d(ey_imag, "ey_imag");
  check_float_3d(ez_real, "ez_real");
  check_float_3d(ez_imag, "ez_imag");
  check_float_3d(patch, "patch");
  check_same_shape(ex_real, ex_imag, "ex_imag");
  check_same_shape(ey_real, ey_imag, "ey_imag");
  check_same_shape(ez_real, ez_imag, "ez_imag");
  Shape3D selected_shape = shape3d(ex_real);
  if (axis_code == 1) {
    selected_shape = shape3d(ey_real);
  } else if (axis_code == 2) {
    selected_shape = shape3d(ez_real);
  }
  c10::cuda::CUDAGuard guard(ex_real.device());
  add_bloch_source_patch_kernel<<<linear_grid(patch.numel()), 256, 0, current_cuda_stream()>>>(
      patch.numel(),
      selected_shape,
      shape3d(patch),
      static_cast<int>(offset_i),
      static_cast<int>(offset_j),
      static_cast<int>(offset_k),
      static_cast<float>(signal_real),
      static_cast<float>(signal_imag),
      static_cast<int>(axis_code),
      static_cast<float>(phase_cos_a),
      static_cast<float>(phase_sin_a),
      static_cast<float>(phase_cos_b),
      static_cast<float>(phase_sin_b),
      ex_real.data_ptr<float>(),
      ex_imag.data_ptr<float>(),
      ey_real.data_ptr<float>(),
      ey_imag.data_ptr<float>(),
      ez_real.data_ptr<float>(),
      ez_imag.data_ptr<float>(),
      patch.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void add_scaled_slice_source_patch_cuda(
    at::Tensor field,
    const at::Tensor& patch,
    const at::Tensor& incident,
    int64_t sample_index,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double scale) {
  check_float_3d(field, "field");
  check_float_3d(patch, "patch");
  check_float_1d(incident, "incident");
  c10::cuda::CUDAGuard guard(field.device());
  add_scaled_slice_source_patch_kernel<<<linear_grid(patch.numel()), 256, 0, current_cuda_stream()>>>(
      patch.numel(),
      shape3d(field),
      shape3d(patch),
      static_cast<int>(sample_index),
      static_cast<int>(offset_i),
      static_cast<int>(offset_j),
      static_cast<int>(offset_k),
      static_cast<float>(scale),
      field.data_ptr<float>(),
      patch.data_ptr<float>(),
      incident.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void add_scaled_line_source_patch_cuda(
    at::Tensor field,
    const at::Tensor& patch,
    const at::Tensor& incident,
    const at::Tensor& sample_indices,
    int64_t sample_axis,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double scale) {
  check_float_3d(field, "field");
  check_float_3d(patch, "patch");
  check_float_1d(incident, "incident");
  check_int32_tensor(sample_indices, "sample_indices");
  c10::cuda::CUDAGuard guard(field.device());
  add_scaled_line_source_patch_kernel<<<linear_grid(patch.numel()), 256, 0, current_cuda_stream()>>>(
      patch.numel(),
      shape3d(field),
      shape3d(patch),
      static_cast<int>(sample_axis),
      static_cast<int>(offset_i),
      static_cast<int>(offset_j),
      static_cast<int>(offset_k),
      static_cast<float>(scale),
      field.data_ptr<float>(),
      patch.data_ptr<float>(),
      incident.data_ptr<float>(),
      sample_indices.data_ptr<int>());
  WITWIN_CUDA_CHECK();
}

void add_interpolated_source_patch_cuda(
    at::Tensor field,
    const at::Tensor& patch,
    const at::Tensor& incident,
    const at::Tensor& sample_positions,
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
  check_same_shape(patch, sample_positions, "sample_positions");
  c10::cuda::CUDAGuard guard(field.device());
  add_interpolated_source_patch_kernel<<<linear_grid(patch.numel()), 256, 0, current_cuda_stream()>>>(
      patch.numel(),
      shape3d(field),
      shape3d(patch),
      static_cast<int>(incident.numel()),
      static_cast<float>(origin),
      static_cast<float>(ds),
      static_cast<int>(offset_i),
      static_cast<int>(offset_j),
      static_cast<int>(offset_k),
      static_cast<float>(scale),
      field.data_ptr<float>(),
      patch.data_ptr<float>(),
      incident.data_ptr<float>(),
      sample_positions.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void add_batched_reference_source_patches_cuda(
    at::Tensor field_x,
    at::Tensor field_y,
    at::Tensor field_z,
    const at::Tensor& coeff_data,
    const at::Tensor& incident,
    const at::Tensor& term_starts,
    const at::Tensor& term_shapes,
    const at::Tensor& term_offsets,
    const at::Tensor& field_codes,
    const at::Tensor& sample_axis_codes,
    const at::Tensor& sample_index_starts,
    const at::Tensor& sample_indices) {
  check_float_3d(field_x, "field_x");
  check_float_3d(field_y, "field_y");
  check_float_3d(field_z, "field_z");
  check_float_1d(coeff_data, "coeff_data");
  check_float_1d(incident, "incident");
  check_int32_tensor(term_starts, "term_starts");
  check_int32_tensor(term_shapes, "term_shapes");
  check_int32_tensor(term_offsets, "term_offsets");
  check_int32_tensor(field_codes, "field_codes");
  check_int32_tensor(sample_axis_codes, "sample_axis_codes");
  check_int32_tensor(sample_index_starts, "sample_index_starts");
  check_int32_tensor(sample_indices, "sample_indices");
  c10::cuda::CUDAGuard guard(field_x.device());
  add_batched_reference_source_patches_kernel<<<linear_grid(coeff_data.numel()), 256, 0, current_cuda_stream()>>>(
      coeff_data.numel(),
      static_cast<int>(term_starts.numel()),
      shape3d(field_x),
      shape3d(field_y),
      shape3d(field_z),
      field_x.data_ptr<float>(),
      field_y.data_ptr<float>(),
      field_z.data_ptr<float>(),
      coeff_data.data_ptr<float>(),
      incident.data_ptr<float>(),
      term_starts.data_ptr<int>(),
      term_shapes.data_ptr<int>(),
      term_offsets.data_ptr<int>(),
      field_codes.data_ptr<int>(),
      sample_axis_codes.data_ptr<int>(),
      sample_index_starts.data_ptr<int>(),
      sample_indices.data_ptr<int>());
  WITWIN_CUDA_CHECK();
}

void add_batched_interpolated_source_patches_cuda(
    at::Tensor field_x,
    at::Tensor field_y,
    at::Tensor field_z,
    const at::Tensor& coeff_data,
    const at::Tensor& incident,
    const at::Tensor& sample_positions,
    const at::Tensor& term_starts,
    const at::Tensor& term_shapes,
    const at::Tensor& term_offsets,
    const at::Tensor& field_codes,
    double origin,
    double ds) {
  check_float_3d(field_x, "field_x");
  check_float_3d(field_y, "field_y");
  check_float_3d(field_z, "field_z");
  check_float_1d(coeff_data, "coeff_data");
  check_float_1d(incident, "incident");
  check_float_1d(sample_positions, "sample_positions");
  check_int32_tensor(term_starts, "term_starts");
  check_int32_tensor(term_shapes, "term_shapes");
  check_int32_tensor(term_offsets, "term_offsets");
  check_int32_tensor(field_codes, "field_codes");
  c10::cuda::CUDAGuard guard(field_x.device());
  add_batched_interpolated_source_patches_kernel<<<linear_grid(coeff_data.numel()), 256, 0, current_cuda_stream()>>>(
      coeff_data.numel(),
      static_cast<int>(term_starts.numel()),
      static_cast<int>(incident.numel()),
      shape3d(field_x),
      shape3d(field_y),
      shape3d(field_z),
      static_cast<float>(origin),
      static_cast<float>(ds),
      field_x.data_ptr<float>(),
      field_y.data_ptr<float>(),
      field_z.data_ptr<float>(),
      coeff_data.data_ptr<float>(),
      incident.data_ptr<float>(),
      sample_positions.data_ptr<float>(),
      term_starts.data_ptr<int>(),
      term_shapes.data_ptr<int>(),
      term_offsets.data_ptr<int>(),
      field_codes.data_ptr<int>());
  WITWIN_CUDA_CHECK();
}

void update_auxiliary_magnetic_cuda(
    at::Tensor magnetic,
    const at::Tensor& electric,
    const at::Tensor& decay,
    const at::Tensor& curl) {
  check_float_1d(magnetic, "magnetic");
  check_float_1d(electric, "electric");
  check_float_1d(decay, "decay");
  check_float_1d(curl, "curl");
  TORCH_CHECK(decay.numel() == magnetic.numel(), "decay must match magnetic length");
  TORCH_CHECK(curl.numel() == magnetic.numel(), "curl must match magnetic length");
  TORCH_CHECK(electric.numel() == magnetic.numel() + 1, "electric length must be magnetic length + 1");
  c10::cuda::CUDAGuard guard(magnetic.device());
  update_auxiliary_magnetic_kernel<<<linear_grid(magnetic.numel()), 256, 0, current_cuda_stream()>>>(
      magnetic.numel(),
      magnetic.data_ptr<float>(),
      electric.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_auxiliary_electric_cuda(
    at::Tensor electric,
    const at::Tensor& magnetic,
    const at::Tensor& decay,
    const at::Tensor& curl,
    int64_t source_index,
    double source_value) {
  check_float_1d(electric, "electric");
  check_float_1d(magnetic, "magnetic");
  check_float_1d(decay, "decay");
  check_float_1d(curl, "curl");
  TORCH_CHECK(decay.numel() == electric.numel(), "decay must match electric length");
  TORCH_CHECK(curl.numel() == electric.numel(), "curl must match electric length");
  TORCH_CHECK(magnetic.numel() + 1 == electric.numel(), "magnetic length must be electric length - 1");
  c10::cuda::CUDAGuard guard(electric.device());
  update_auxiliary_electric_kernel<<<linear_grid(electric.numel()), 256, 0, current_cuda_stream()>>>(
      electric.numel(),
      static_cast<int>(source_index),
      static_cast<float>(source_value),
      electric.data_ptr<float>(),
      magnetic.data_ptr<float>(),
      decay.data_ptr<float>(),
      curl.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}
