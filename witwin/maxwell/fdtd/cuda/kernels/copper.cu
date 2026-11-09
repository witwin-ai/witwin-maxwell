#include "../launch.h"
#include "../tensors.h"

namespace {
constexpr int threads = 128;

struct CopperStep {
  int64_t faces, modes, inverse_row, inverse_col;
  const int64_t* offsets;
  const bool* valid;
  const double *old_e, *B, *decay, *drive, *rates, *D, *rhs, *inverse;
  const double *scale, *inv_dual, *area;
  double *state, *loss;
  volatile double* partials;
  unsigned int* counter;
  const int32_t* active;
  double activity, dt, sign, left, right;
};

__device__ double block_sum(double value, double* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = threads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  return shared[0];
}

template <typename field_t, int modes>
__global__ void copper_electric_kernel(
    field_t* electric, const field_t* magnetic, const field_t* normal,
    field_t* right_e, CopperStep p) {
  const int64_t face = int64_t(blockIdx.x) * threads + threadIdx.x;
  const int64_t n = p.faces;
  const int64_t count = modes ? modes : p.modes;
  const double activity = p.active ? double(*p.active) : p.activity;
  double heat = 0.;
  if (face < n) {
    // The common 47-mode response retains its state in registers across the solve.
    // Other declared modal orders use the same recurrence with two state reads.
    double saved[modes ? modes : 1];
    double history0 = 0., history1 = 0.;
    #pragma unroll
    for (int64_t k = 0; k < count; ++k) {
      const double state = p.state[k * n + face];
      if constexpr (modes != 0) saved[k] = state;
      const double midpoint = (1. + p.decay[k]) / 2.;
      history0 += (p.B[2 * k] * midpoint) * state;
      history1 += (p.B[2 * k + 1] * midpoint) * state;
    }
    const double low = double(normal[p.offsets[3 * n + face]]) * p.valid[face];
    const double high = double(normal[p.offsets[4 * n + face]]) * p.valid[n + face];
    const double transverse = p.sign * (high - low) * p.inv_dual[face];
    const double forcing0 = __dadd_rn(p.sign * double(magnetic[p.offsets[n + face]]), p.left * transverse);
    const double forcing1 = __dadd_rn(-p.sign * double(magnetic[p.offsets[2 * n + face]]), p.right * transverse);
    const double old0 = p.old_e[face], old1 = p.old_e[n + face];
    const double rhs0 = __dsub_rn(__dadd_rn(
        p.rhs[4 * face] * old0 + p.rhs[4 * face + 1] * old1, forcing0), history0 * p.scale[face]);
    const double rhs1 = __dsub_rn(__dadd_rn(
        p.rhs[4 * face + 2] * old0 + p.rhs[4 * face + 3] * old1, forcing1), history1 * p.scale[face]);
    const double* inverse = p.inverse + 4 * face;
    double updated0 = inverse[0] * rhs0 + inverse[p.inverse_col] * rhs1;
    double updated1 = inverse[p.inverse_row] * rhs0 + inverse[p.inverse_row + p.inverse_col] * rhs1;
    if (p.active && activity == 0.) {
      updated0 = old0;
      updated1 = old1;
    }
    electric[p.offsets[face]] = field_t(updated0);
    right_e[face] = field_t(updated1);
    const double sum0 = updated0 + old0, sum1 = updated1 + old1;
    #pragma unroll
    for (int64_t k = 0; k < count; ++k) {
      double state;
      if constexpr (modes != 0) state = saved[k];
      else state = p.state[k * n + face];
      const double driven = p.drive[2 * k] * sum0 + p.drive[2 * k + 1] * sum1;
      double next = __dadd_rn(p.decay[k] * state, driven);
      if (p.active && activity == 0.) next = state;
      const double midpoint = (next + state) / 2.;
      heat += p.rates[k] * (midpoint * midpoint);
      p.state[k * n + face] = next;
    }
    const double mid0 = sum0 / 2., mid1 = sum1 / 2.;
    const double direct0 = p.D[0] * mid0 + p.D[1] * mid1;
    const double direct1 = p.D[2] * mid0 + p.D[3] * mid1;
    heat += mid0 * direct0 + mid1 * direct1;
    heat = (heat * p.scale[face]) * p.area[face];
  }
  __shared__ double shared[threads];
  __shared__ bool last;
  const double subtotal = block_sum(heat, shared);
  if (threadIdx.x == 0) {
    p.partials[blockIdx.x] = subtotal;
    __threadfence();
    // Only the completion counter is atomic. Loss is reduced in fixed block order.
    last = atomicInc(p.counter, gridDim.x - 1) == gridDim.x - 1;
  }
  __syncthreads();
  if (last) {
    double total = 0.;
    for (unsigned int block = threadIdx.x; block < gridDim.x; block += threads)
      total += p.partials[block];
    total = block_sum(total, shared);
    if (threadIdx.x == 0) *p.loss += (p.dt * total) * activity;
  }
}
}  // namespace

namespace {
struct CopperAuxiliary {
  int64_t faces = 0, targets = 0, frequencies = 0, end = 0;
  const int64_t *offsets = nullptr, *normal_targets = nullptr, *sources = nullptr;
  const double *high_gain = nullptr, *gains = nullptr, *frequencies_hz = nullptr, *starts = nullptr;
  const double* old_values = nullptr;
  double *old_e = nullptr, *step = nullptr, *norm = nullptr, *real = nullptr, *imag = nullptr;
  unsigned int* counter = nullptr;
  const int32_t* active = nullptr;
  double activity = 1., left = 0., right = 0., dt = 0.;
  int window = 0;
};

__device__ float field_add(float left, float right) { return __fadd_rn(left, right); }
__device__ double field_add(double left, double right) { return __dadd_rn(left, right); }

__device__ double spectral_weight(CopperAuxiliary p, int64_t frequency, double step, double activity) {
  const double start = p.starts[frequency];
  const bool valid = step >= start && step < p.end;
  const double fraction = __ddiv_rn(__dsub_rn(step, start),
                                    fmax(__dsub_rn(double(p.end), start), 1.));
  double weight = 1.;
  if (p.window == 1)
    weight = __dmul_rn(.5, __dsub_rn(1., cos(__dmul_rn(2. * M_PI, fraction))));
  // Tensor/scalar division in torch multiplies by the scalar reciprocal.
  if (p.window == 2 && fraction < .1)
    weight = __dmul_rn(.5, __dsub_rn(1., cos(__dmul_rn(__dmul_rn(M_PI, fraction), 10.))));
  return __dmul_rn(__dmul_rn(weight, double(valid)), activity);
}

template <typename field_t>
__global__ void copper_auxiliary_kernel(
    const field_t* electric, const field_t* right_e, field_t* magnetic, field_t* normal,
    CopperAuxiliary p, int phase) {
  const int64_t i = int64_t(blockIdx.x) * threads + threadIdx.x;
  const int64_t n = p.faces;
  if (phase == 0) {
    if (i < n) {
      p.old_e[i] = double(electric[p.offsets[i]]);
      p.old_e[n + i] = double(right_e[i]);
    }
    return;
  }
  const double activity = p.active ? double(*p.active) : p.activity;
  if (phase == 1) {
    if (i < n) {
      const double difference = __dmul_rn(__dsub_rn(p.old_values[n + i], p.old_values[i]), activity);
      const int64_t target = p.offsets[2 * n + i];
      magnetic[target] = field_add(magnetic[target], field_t(__dmul_rn(p.high_gain[i], difference)));
    }
    if (i < p.targets) {
      // Match torch's two global index_add_ passes, including the field cast and
      // atomic-add semantics. One thread owns each target, so the passes are ordered.
      for (int side = 0; side < 2; ++side) {
        const int64_t source = p.sources[side * p.targets + i];
        if (source >= 0) {
          const double difference = __dmul_rn(
              __dsub_rn(p.old_values[n + source], p.old_values[source]), activity);
          const double weighted = __dmul_rn(__dmul_rn(difference, p.right),
                                            __ddiv_rn(1., __dadd_rn(p.left, p.right)));
          atomicAdd(normal + p.normal_targets[i], field_t(__dmul_rn(p.gains[side * p.targets + i], weighted)));
        }
      }
    }
    return;
  }
  const double step = *p.step;
  if (i < p.frequencies * 2 * n) {
    const int64_t frequency = i / (2 * n), trace = (i / n) % 2, face = i % n;
    const double weight = spectral_weight(p, frequency, step, activity);
    const double angle = __dmul_rn(__dmul_rn(__dmul_rn(2. * M_PI, p.frequencies_hz[frequency]), p.dt), step + 1);
    const double field = trace ? double(right_e[face]) : double(electric[p.offsets[face]]);
    p.real[i] = __dadd_rn(p.real[i], __dmul_rn(__dmul_rn(weight, cos(angle)), field));
    p.imag[i] = __dadd_rn(p.imag[i], __dmul_rn(__dmul_rn(weight, sin(angle)), field));
  }
  __shared__ bool last;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    last = atomicInc(p.counter, gridDim.x - 1) == gridDim.x - 1;
  }
  __syncthreads();
  if (last) {
    for (int64_t frequency = threadIdx.x; frequency < p.frequencies; frequency += threads)
      p.norm[frequency] = __dadd_rn(p.norm[frequency], spectral_weight(p, frequency, step, activity));
    __syncthreads();
    // All blocks have consumed this clock before its one deterministic advance.
    if (threadIdx.x == 0) *p.step = __dadd_rn(*p.step, activity);
  }
}

void launch_copper_auxiliary(const torch::stable::Tensor& electric,
    const torch::stable::Tensor& right_e, torch::stable::Tensor* magnetic,
    torch::stable::Tensor* normal, CopperAuxiliary p, int phase, int64_t count) {
  const torch::stable::accelerator::DeviceGuard guard(electric.get_device_index());
  const dim3 grid = linear_grid(count, threads);
  const auto stream = current_cuda_stream();
#define LAUNCH_AUXILIARY(type) \
  copper_auxiliary_kernel<type><<<grid, threads, 0, stream>>>( \
      electric.const_data_ptr<type>(), right_e.const_data_ptr<type>(), \
      magnetic ? magnetic->mutable_data_ptr<type>() : nullptr, \
      normal ? normal->mutable_data_ptr<type>() : nullptr, p, phase)
  if (electric.scalar_type() == torch::headeronly::ScalarType::Double) { LAUNCH_AUXILIARY(double); }
  else { LAUNCH_AUXILIARY(float); }
#undef LAUNCH_AUXILIARY
  WITWIN_CUDA_CHECK();
}
}  // namespace

void begin_copper_step_cuda(const torch::stable::Tensor& electric,
    const torch::stable::Tensor& right_e, torch::stable::Tensor old_e,
    const torch::stable::Tensor& offsets) {
  CopperAuxiliary p;
  p.faces = right_e.numel();
  p.offsets = offsets.const_data_ptr<int64_t>();
  p.old_e = old_e.mutable_data_ptr<double>();
  launch_copper_auxiliary(electric, right_e, nullptr, nullptr, p, 0, p.faces);
}

void correct_copper_magnetic_cuda(torch::stable::Tensor magnetic, torch::stable::Tensor normal,
    const torch::stable::Tensor& old_e, const torch::stable::Tensor& offsets,
    const torch::stable::Tensor& high_gain, const torch::stable::Tensor& targets,
    const torch::stable::Tensor& sources, const torch::stable::Tensor& gains,
    const std::optional<torch::stable::Tensor>& active, double activity, double left, double right) {
  CopperAuxiliary p;
  p.faces = old_e.size(1);
  p.targets = targets.numel();
  p.offsets = offsets.const_data_ptr<int64_t>();
  p.old_values = old_e.const_data_ptr<double>();
  p.high_gain = high_gain.const_data_ptr<double>();
  p.normal_targets = targets.const_data_ptr<int64_t>();
  p.sources = sources.const_data_ptr<int64_t>();
  p.gains = gains.const_data_ptr<double>();
  p.active = active ? active->const_data_ptr<int32_t>() : nullptr;
  p.activity = activity; p.left = left; p.right = right;
  launch_copper_auxiliary(magnetic, magnetic, &magnetic, &normal, p, 1, std::max(p.faces, p.targets));
}

void accumulate_copper_spectra_cuda(const torch::stable::Tensor& electric,
    const torch::stable::Tensor& right_e, const torch::stable::Tensor& offsets,
    torch::stable::Tensor step, torch::stable::Tensor norm, torch::stable::Tensor real,
    torch::stable::Tensor imag, const torch::stable::Tensor& frequencies,
    const torch::stable::Tensor& starts, torch::stable::Tensor counter,
    const std::optional<torch::stable::Tensor>& active, double activity, double dt,
    int64_t end, int64_t window) {
  CopperAuxiliary p;
  p.faces = right_e.numel(); p.frequencies = frequencies.numel(); p.end = end; p.window = window;
  p.offsets = offsets.const_data_ptr<int64_t>();
  p.step = step.mutable_data_ptr<double>(); p.norm = norm.mutable_data_ptr<double>();
  p.real = real.mutable_data_ptr<double>(); p.imag = imag.mutable_data_ptr<double>();
  p.frequencies_hz = frequencies.const_data_ptr<double>(); p.starts = starts.const_data_ptr<double>();
  p.counter = reinterpret_cast<unsigned int*>(counter.mutable_data_ptr<int32_t>());
  p.active = active ? active->const_data_ptr<int32_t>() : nullptr;
  p.activity = activity; p.dt = dt;
  launch_copper_auxiliary(electric, right_e, nullptr, nullptr, p, 2, p.frequencies * 2 * p.faces);
}

void advance_copper_electric_cuda(
    torch::stable::Tensor electric, const torch::stable::Tensor& magnetic,
    const torch::stable::Tensor& normal, torch::stable::Tensor right_e,
    torch::stable::Tensor state, torch::stable::Tensor loss,
    torch::stable::Tensor partials, torch::stable::Tensor counter,
    const torch::stable::Tensor& offsets, const torch::stable::Tensor& valid,
    const torch::stable::Tensor& old_e, const torch::stable::Tensor& B,
    const torch::stable::Tensor& decay, const torch::stable::Tensor& drive,
    const torch::stable::Tensor& rates, const torch::stable::Tensor& D,
    const torch::stable::Tensor& rhs, const torch::stable::Tensor& inverse,
    const torch::stable::Tensor& scale, const torch::stable::Tensor& inv_dual,
    const torch::stable::Tensor& area, const std::optional<torch::stable::Tensor>& active,
    double activity, double dt, double sign, double left, double right) {
  const torch::stable::accelerator::DeviceGuard guard(electric.get_device_index());
  const CopperStep p{state.size(1), state.size(0), inverse.stride(1), inverse.stride(2),
      offsets.const_data_ptr<int64_t>(), valid.const_data_ptr<bool>(),
      old_e.const_data_ptr<double>(), B.const_data_ptr<double>(), decay.const_data_ptr<double>(),
      drive.const_data_ptr<double>(), rates.const_data_ptr<double>(), D.const_data_ptr<double>(),
      rhs.const_data_ptr<double>(), inverse.const_data_ptr<double>(), scale.const_data_ptr<double>(),
      inv_dual.const_data_ptr<double>(), area.const_data_ptr<double>(), state.mutable_data_ptr<double>(),
      loss.mutable_data_ptr<double>(), partials.mutable_data_ptr<double>(),
      reinterpret_cast<unsigned int*>(counter.mutable_data_ptr<int32_t>()),
      active ? active->const_data_ptr<int32_t>() : nullptr, activity, dt, sign, left, right};
  const dim3 grid = linear_grid(p.faces, threads);
  const auto stream = current_cuda_stream();
#define LAUNCH_COPPER(type, modes) \
  copper_electric_kernel<type, modes><<<grid, threads, 0, stream>>>( \
      electric.mutable_data_ptr<type>(), magnetic.const_data_ptr<type>(), \
      normal.const_data_ptr<type>(), right_e.mutable_data_ptr<type>(), p)
  if (electric.scalar_type() == torch::headeronly::ScalarType::Double) {
    if (p.modes == 47) { LAUNCH_COPPER(double, 47); }
    else { LAUNCH_COPPER(double, 0); }
  } else {
    if (p.modes == 47) { LAUNCH_COPPER(float, 47); }
    else { LAUNCH_COPPER(float, 0); }
  }
#undef LAUNCH_COPPER
  WITWIN_CUDA_CHECK();
}
