#include "step.cuh"
#include <algorithm>

namespace {
// Keep the analytic expressions and fp32 association of kernels/sources.cu.
template <int Kind>
__device__ __forceinline__ float evaluate(float sample_time, CoreWaveform w) {
  if constexpr (Kind == 0) return w.amplitude * cosf(w.omega * sample_time + w.phase);
  if constexpr (Kind == 1 || Kind == 4) {
    const float tau = sample_time - w.delay;
    const float normalized = tau * w.inv_sigma;
    const float envelope = expf(-0.5f * normalized * normalized);
    if constexpr (Kind == 1) return w.amplitude * envelope * cosf(w.omega * tau + w.phase);
    float signal = envelope * cosf(w.omega * tau + w.phase);
    if constexpr (Kind == 4) {
      const float pedestal_normalized = tau * w.pedestal_inv_sigma;
      signal -= w.pedestal * expf(-0.5f * pedestal_normalized * pedestal_normalized);
    }
    return w.amplitude * signal;
  }
  const float tau = sample_time - w.delay;
  const float alpha = w.pi_frequency * tau;
  const float alpha_sq = alpha * alpha;
  return w.amplitude * (1.0f - 2.0f * alpha_sq) * expf(-alpha_sq);
}

template <int Kind>
__global__ void delayed_kernel(CoreSource s, float* output, int64_t count) {
  const int64_t q = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= count) return;
  output[q] = evaluate<Kind>(s.times[q / s.delay_count] - s.delays[q % s.delay_count], s.waveform);
}

template <int Kind>
__global__ void patch_kernel(CoreSource s) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= s.patch_shape[0] || j >= s.patch_shape[1] || k >= s.patch_shape[2]) return;
  const int64_t q = (int64_t(i) * s.patch_shape[1] + j) * s.patch_shape[2] + k;
  const int coords[3] = {i + s.offsets[0], j + s.offsets[1], k + s.offsets[2]};
  for (int a = 0; a < 3; ++a) if (coords[a] < 0 || coords[a] >= s.shape[a]) return;
  const int64_t dst = (int64_t(coords[0]) * s.shape[1] + coords[1]) * s.shape[2] + coords[2];
  const int64_t n = s.step[0];
  if (s.kind == 1) {
    const float value = s.signals[2*n] * s.patch[q] + s.signals[2*n+1] * s.second[q];
    s.field[dst] += value;
    return;
  }
  float signal;
  if (s.kind == 2) {
    if (s.causal && s.times[n] < s.activation[q]) return;
    signal = s.tabulated ? s.signals[n*s.delay_count+s.delay_indices[q]] :
        evaluate<Kind>(s.times[n] - s.delays[q], s.waveform);
  } else {
    signal = s.signals[n];
  }
  s.field[dst] += signal * s.patch[q];
}

__global__ void periodic_patch_kernel(CoreSource s) {
  const auto g = s.gather;
  const int64_t row = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (row >= g.count) return;
  const int64_t dst = g.targets[2*row+1];
  const float signal = s.signals[s.step[0]];
  float value = s.field[dst];
  for (int64_t e = g.rows[row]; e < g.rows[row+1]; ++e) {
    const int64_t q = g.entries[2*e];
    const float delta = __fmul_rn(signal, s.patch[q]);
    value = (g.entries[2*e+1] & 8) ? source_atomic_sum(value, delta) : __fadd_rn(value, delta);
  }
  s.field[dst] = value;
}

__device__ __forceinline__ void auxiliary_electric_element(int64_t index, int64_t count,
    int source_index, float* electric, const float* magnetic, const float* decay,
    const float* curl, const float* signals, const int64_t* step) {
  if (index >= count) return;
  if (index + 1 == count) { electric[index] = 0.0f; return; }
  if (index == source_index) { electric[index] = signals[step[0]]; return; }
  if (index > 0)
    electric[index] = decay[index] * electric[index] - curl[index] * (magnetic[index] - magnetic[index-1]);
}

__global__ void auxiliary_kernel(int64_t count, int source_index, float* electric,
    const float* magnetic, const float* decay, const float* curl,
    const float* signals, const int64_t* step) {
  const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  auxiliary_electric_element(index, count, source_index, electric, magnetic, decay, curl, signals, step);
}

// The interpolation and ordered gather body below are copied from sources.cu.
// Keep their arithmetic, including source_atomic_sum's subnormal behavior.
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

template <bool Interpolated>
__device__ __forceinline__ void auxiliary_correction_element(int64_t row, SourceGather g, int incident_count,
    float origin, float inv_ds, float* field_x, float* field_y, float* field_z,
    const float* coeff, const float* incident, const float* positions, const int* indices) {
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

__global__ void auxiliary_boundary_kernel(CoreAuxiliary a) {
  const int j = a.source_index;
  const float *electric = a.electric, *magnetic = a.magnetic;
  const float ce = a.electric_curl[j], ch = a.magnetic_curl[j];
  const float source_value = a.boundary_signals[a.step[0] * a.boundary_width];
  const float next_value = a.boundary_signals[a.step[0] * a.boundary_width + 1];
  // Each torch elementwise operation rounds separately; never contract these.
  const float right_h = __fsub_rn(magnetic[j], __fmul_rn(ch, __fsub_rn(electric[j+1], electric[j])));
  const float following_h = __fsub_rn(magnetic[j+1], __fmul_rn(ch, __fsub_rn(electric[j+2], electric[j+1])));
  const float right_e = __fsub_rn(electric[j+1], __fmul_rn(ce, __fsub_rn(following_h, right_h)));
  const float next_right_h = __fsub_rn(right_h, __fmul_rn(ch, __fsub_rn(right_e, source_value)));
  const float left_h = __fadd_rn(right_h, __fdiv_rn(__fsub_rn(source_value, electric[j]), ce));
  const float next_left_h = __fadd_rn(next_right_h, __fdiv_rn(__fsub_rn(next_value, source_value), ce));
  const float left_e = __fadd_rn(electric[j], __fdiv_rn(__fsub_rn(left_h, magnetic[j-1]), ch));
  const float next_left_e = __fadd_rn(source_value, __fdiv_rn(__fsub_rn(next_left_h, left_h), ch));
  const float outer_h = __fadd_rn(left_h, __fdiv_rn(__fsub_rn(next_left_e, left_e), ce));
  a.electric[j-1] = left_e;
  a.boundary_h[0] = left_h;
  a.boundary_h[1] = outer_h;
}

template <bool Magnetic, bool Interpolated>
__global__ void auxiliary_correction_kernel(CoreAuxiliary a, int64_t patch_blocks) {
  if (blockIdx.x < patch_blocks) {
    const int64_t row = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    auxiliary_correction_element<Interpolated>(row, a.gather,
        Magnetic ? a.electric_count : a.magnetic_count, a.origin, a.inv_ds,
        a.fields[0], a.fields[1], a.fields[2], a.coeff,
        Magnetic ? a.electric : a.magnetic, a.positions, a.indices);
    return;
  }
  const int64_t index = (int64_t(blockIdx.x) - patch_blocks)*blockDim.x + threadIdx.x;
  if constexpr (Magnetic) {
    if (index >= a.magnetic_count) return;
    // Identical to update_auxiliary_magnetic_kernel, followed by its existing
    // incident-boundary overwrite, with one owner per magnetic line cell.
    float* magnetic = a.magnetic;
    const float *electric = a.electric, *decay = a.magnetic_decay, *curl = a.magnetic_curl;
    magnetic[index] = decay[index] * magnetic[index] - curl[index] * (electric[index + 1] - electric[index]);
    if (a.boundary && index < a.source_index)
      magnetic[index] = a.boundary_h[a.source_index - index - 1];
  } else {
    auxiliary_electric_element(index, a.electric_count, a.source_index, a.electric,
        a.magnetic, a.electric_decay, a.electric_curl, a.signals, a.step);
  }
}

template <bool Magnetic>
void launch_auxiliary_correction(CoreAuxiliary a, cudaStream_t stream) {
  const int64_t patch_blocks = (a.gather.count + 255)/256;
  const int64_t count = Magnetic ? a.magnetic_count : a.electric_count;
  const int64_t blocks = patch_blocks + (count + 255)/256;
  if (a.reference)
    auxiliary_correction_kernel<Magnetic, false><<<blocks, 256, 0, stream>>>(a, patch_blocks);
  else
    auxiliary_correction_kernel<Magnetic, true><<<blocks, 256, 0, stream>>>(a, patch_blocks);
}

__global__ void dft_kernel(CoreDft d) {
  const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t f = blockIdx.y;
  const int64_t row = d.step[0] * d.frequencies + f;
  const float c = d.cosine[row], s = d.sine[row];
  #pragma unroll
  for (int a = 0; a < 3; ++a) {
    if (index < d.sizes[a]) {
      const int64_t dst = f*d.sizes[a] + index;
      const float value = d.field[a][index];
      d.real[a][dst] += value*c;
      d.imag[a][dst] += value*s;
    }
  }
}

__global__ void commit_kernel(int64_t* step) { ++step[0]; }

template <int Kind>
void launch_patch(CoreSource source, cudaStream_t stream) {
  const dim3 block(32, 4, 2);
  const dim3 grid((source.patch_shape[2]+31)/32, (source.patch_shape[1]+3)/4,
                  (source.patch_shape[0]+1)/2);
  patch_kernel<Kind><<<grid, block, 0, stream>>>(source);
}
}  // namespace

void compile_delayed_amplitudes(CoreSource source, float* output, int64_t steps, cudaStream_t stream) {
  const int64_t count = steps * source.delay_count;
  if (!count) return;
  const int blocks = (count+255)/256;
  switch (source.time_kind) {
    case 0: delayed_kernel<0><<<blocks, 256, 0, stream>>>(source, output, count); break;
    case 1: delayed_kernel<1><<<blocks, 256, 0, stream>>>(source, output, count); break;
    case 2: delayed_kernel<2><<<blocks, 256, 0, stream>>>(source, output, count); break;
    case 4: delayed_kernel<4><<<blocks, 256, 0, stream>>>(source, output, count); break;
  }
}

void source_patch_indexed(CoreSource source, cudaStream_t stream) {
  if (source.wrap) {
    if (source.gather.count)
      periodic_patch_kernel<<<(source.gather.count+255)/256, 256, 0, stream>>>(source);
    return;
  }
  switch (source.time_kind) {
    case 0: launch_patch<0>(source, stream); break;
    case 1: launch_patch<1>(source, stream); break;
    case 2: launch_patch<2>(source, stream); break;
    case 4: launch_patch<4>(source, stream); break;
  }
}

void auxiliary_electric_indexed(int64_t count, int source_index, float* electric,
    const float* magnetic, const float* decay, const float* curl,
    const float* signals, const int64_t* step, cudaStream_t stream) {
  auxiliary_kernel<<<(count+255)/256, 256, 0, stream>>>(count, source_index, electric,
      magnetic, decay, curl, signals, step);
}

void dft_indexed(CoreDft dft, cudaStream_t stream) {
  const auto count = std::max(dft.sizes[0], std::max(dft.sizes[1], dft.sizes[2]));
  dft_kernel<<<dim3((count+255)/256, dft.frequencies), 256, 0, stream>>>(dft);
}

void commit_step(int64_t* step, cudaStream_t stream) { commit_kernel<<<1, 1, 0, stream>>>(step); }

void auxiliary_boundary_indexed(CoreAuxiliary auxiliary, cudaStream_t stream) {
  auxiliary_boundary_kernel<<<1, 1, 0, stream>>>(auxiliary);
}

void auxiliary_correction_indexed(CoreAuxiliary auxiliary, bool magnetic, cudaStream_t stream) {
  if (magnetic) launch_auxiliary_correction<true>(auxiliary, stream);
  else launch_auxiliary_correction<false>(auxiliary, stream);
}
