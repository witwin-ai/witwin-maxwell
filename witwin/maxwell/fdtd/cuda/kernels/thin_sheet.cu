#include "../launch.h"
#include "../tensors.h"

#include <algorithm>
#include <type_traits>

namespace {
constexpr int threads = 256;

template <typename field_t>
__global__ void thin_sheet_gather_kernel(
    const field_t* x, const field_t* y, const field_t* z, const double* extra,
    const int32_t* components, const int64_t* offsets, double* values, int64_t count) {
  const int64_t i = int64_t(blockIdx.x) * threads + threadIdx.x;
  if (i >= count) return;
  const int64_t offset = offsets[i];
  const int32_t component = components[i];
  if (component == 3) values[i] = extra[offset];
  else values[i] = double((component == 0 ? x : component == 1 ? y : z)[offset]);
}

template <typename field_t>
__global__ void thin_sheet_scatter_kernel(
    field_t* x, field_t* y, field_t* z, double* extra,
    const int32_t* components, const int64_t* offsets, const double* values, int64_t count) {
  const int64_t i = int64_t(blockIdx.x) * threads + threadIdx.x;
  if (i >= count) return;
  const int64_t offset = offsets[i];
  const int32_t component = components[i];
  if (component == 3) extra[offset] = values[i];
  else (component == 0 ? x : component == 1 ? y : z)[offset] = field_t(values[i]);
}
}  // namespace

void thin_sheet_gather_cuda(const torch::stable::Tensor& x, const torch::stable::Tensor& y,
    const torch::stable::Tensor& z, const torch::stable::Tensor& extra,
    const torch::stable::Tensor& components, const torch::stable::Tensor& offsets,
    torch::stable::Tensor values) {
  const int64_t count = values.numel();
  if (!count) return;
  const torch::stable::accelerator::DeviceGuard guard(x.get_device_index());
  const auto stream = current_cuda_stream();
#define GATHER(type) \
  thin_sheet_gather_kernel<type><<<linear_grid(count, threads), threads, 0, stream>>>( \
      x.const_data_ptr<type>(), y.const_data_ptr<type>(), z.const_data_ptr<type>(), \
      extra.const_data_ptr<double>(), components.const_data_ptr<int32_t>(), \
      offsets.const_data_ptr<int64_t>(), values.mutable_data_ptr<double>(), count)
  if (x.scalar_type() == torch::headeronly::ScalarType::Double) { GATHER(double); }
  else { GATHER(float); }
#undef GATHER
  WITWIN_CUDA_CHECK();
}

void thin_sheet_scatter_cuda(torch::stable::Tensor x, torch::stable::Tensor y,
    torch::stable::Tensor z, torch::stable::Tensor extra,
    const torch::stable::Tensor& components, const torch::stable::Tensor& offsets,
    const torch::stable::Tensor& values) {
  const int64_t count = values.numel();
  if (!count) return;
  const torch::stable::accelerator::DeviceGuard guard(x.get_device_index());
  const auto stream = current_cuda_stream();
#define SCATTER(type) \
  thin_sheet_scatter_kernel<type><<<linear_grid(count, threads), threads, 0, stream>>>( \
      x.mutable_data_ptr<type>(), y.mutable_data_ptr<type>(), z.mutable_data_ptr<type>(), \
      extra.mutable_data_ptr<double>(), components.const_data_ptr<int32_t>(), \
      offsets.const_data_ptr<int64_t>(), values.const_data_ptr<double>(), count)
  if (x.scalar_type() == torch::headeronly::ScalarType::Double) { SCATTER(double); }
  else { SCATTER(float); }
#undef SCATTER
  WITWIN_CUDA_CHECK();
}

namespace {
constexpr int copper_threads = 128;
constexpr int sheet_row_threads = 256;

template <typename Launch>
void sheet_dispatch_rows(int64_t lanes, Launch launch) {
  switch (lanes) {
    case 1: launch(std::integral_constant<int, 1>{}); break;
    case 2: launch(std::integral_constant<int, 2>{}); break;
    case 4: launch(std::integral_constant<int, 4>{}); break;
    case 8: launch(std::integral_constant<int, 8>{}); break;
    default: STD_TORCH_CHECK(false, "invalid sheet CSR lane count");
  }
}

struct SheetCsr {
  const int32_t *rows, *columns;
  const double* values;
};

SheetCsr sheet_csr(const torch::stable::Tensor& rows, const torch::stable::Tensor& columns,
                   const torch::stable::Tensor& values) {
  return {rows.const_data_ptr<int32_t>(), columns.const_data_ptr<int32_t>(), values.const_data_ptr<double>()};
}

template <int L>
__device__ double sheet_row_sum(double sum) {
  const unsigned int mask = ((1u << L) - 1u) << ((threadIdx.x % 32) / L * L);
  // Fixed tree for each L; L=1 retains the sequential FMA result without a shuffle.
  #pragma unroll
  for (int offset = L / 2; offset; offset /= 2)
    sum = __dadd_rn(sum, __shfl_down_sync(mask, sum, offset, L));
  return sum;
}

// One complete sub-warp owns a row, including in the final partial block.
// The tensor reference rounds each vector operation before its sparse product.
template <int L, int GroupL>
__device__ double sheet_affine_mv(SheetCsr a, int row, const double* x,
                                const double* y, const double* z) {
  const int lane = threadIdx.x % GroupL;
  // Two matrices share a row group; only each matrix's selected subgroup participates.
  if (lane >= L) return 0.;
  double sum = 0.;
  for (int j = a.rows[row] + lane; j < a.rows[row + 1]; j += L) {
    const int column = a.columns[j];
    double value = __dmul_rn(2., y[column]);
    if (x) value = __dadd_rn(x[column], value);
    value = __dadd_rn(value, z[column]);
    sum = __fma_rn(a.values[j], value, sum);
  }
  return sheet_row_sum<L>(sum);
}

template <int L>
__device__ double sheet_cooperative_mv(SheetCsr a, int row, const double* x) {
  const int lane = threadIdx.x % L;
  double sum = 0.;
  for (int j = a.rows[row] + lane; j < a.rows[row + 1]; j += L)
    sum = __fma_rn(a.values[j], x[a.columns[j]], sum);
  return sheet_row_sum<L>(sum);
}

__global__ void thin_sheet_chord_initialize_kernel(const double* rhs, double* y,
    double* residual, int64_t count) {
  const int64_t row = int64_t(blockIdx.x) * threads + threadIdx.x;
  if (row >= count) return;
  y[row] = 0.;
  residual[row] = rhs[row];
}

// K1 reads an immutable full direction; K3 runs only after K1 and K2 finish.
template <int L>
__global__ void thin_sheet_chord_residual_kernel(SheetCsr s, const double* direction,
    double* y, double* residual, int64_t count) {
  const int row = blockIdx.x * (sheet_row_threads / L) + threadIdx.x / L;
  if (row >= count) return;
  const double product = sheet_cooperative_mv<L>(s, row, direction);
  if (threadIdx.x % L == 0) {
    y[row] = __dadd_rn(y[row], direction[row]);
    residual[row] = __dsub_rn(residual[row], product);
  }
}

// K2 also supplies the first cooperative pass of each outer setup.
template <int L>
__global__ void thin_sheet_chord_matvec_kernel(SheetCsr r, const double* residual,
    double* temporary, int64_t count) {
  const int row = blockIdx.x * (sheet_row_threads / L) + threadIdx.x / L;
  if (row >= count) return;
  const double product = sheet_cooperative_mv<L>(r, row, residual);
  if (threadIdx.x % L == 0) temporary[row] = product;
}

// K3 also supplies setup's second pass. Keep both products rounded before the add.
template <int L>
__global__ void thin_sheet_chord_direction_kernel(SheetCsr r, const double* temporary,
    double* direction, int64_t count, double alpha, double beta, bool setup) {
  const int row = blockIdx.x * (sheet_row_threads / L) + threadIdx.x / L;
  if (row >= count) return;
  const double product = sheet_cooperative_mv<L>(r, row, temporary);
  if (threadIdx.x % L == 0) {
    const double scaled = __dmul_rn(beta, product);
    // Setup must not read the uninitialized direction buffer, even with alpha=0.
    direction[row] = setup ? scaled : __dadd_rn(__dmul_rn(alpha, direction[row]), scaled);
  }
}

__device__ double sheet_mv(SheetCsr a, int64_t row, const double* x) {
  double sum = 0.;
  for (int j = a.rows[row]; j < a.rows[row + 1]; ++j)
    sum = __fma_rn(a.values[j], x[a.columns[j]], sum);
  return sum;
}

__device__ double sheet_dot2(const double* a, double x, double y) {
  return __fma_rn(a[1], y, __dmul_rn(a[0], x));
}

struct SheetRhs {
  int64_t traces, chords, modes;
  const double *state, *history, *effective, *local, *mass, *conductance, *area, *feed_area;
  const double *current, *previous, *forcing, *old, *next, *x_previous;
  SheetCsr acx, acc, feed;
  double *history_out, *rhs, *coupled;
  double inverse_dt, quarter_dt;
  bool feeds;
};

template <int AxL, int AcL>
__global__ void thin_sheet_copper_coupling_kernel(SheetRhs p) {
  constexpr int L = AxL > AcL ? AxL : AcL;
  const int row = blockIdx.x * (sheet_row_threads / L) + threadIdx.x / L;
  if (row >= 2 * p.chords) return;
  const double acx = sheet_affine_mv<AxL, L>(p.acx, row, p.next, p.old, p.x_previous);
  const double acc = sheet_affine_mv<AcL, L>(p.acc, row, nullptr, p.current, p.previous);
  if (threadIdx.x % L == 0)
    p.coupled[row] = __dmul_rn(p.quarter_dt, __dadd_rn(acx, acc));
}

__global__ void thin_sheet_copper_rhs_kernel(SheetRhs p) {
  const int64_t trace = int64_t(blockIdx.x) * copper_threads + threadIdx.x;
  if (trace >= p.traces) return;
  double history[2] = {0., 0.};
  for (int64_t mode = 0; mode < p.modes; ++mode) {
    const double state = p.state[mode * p.traces + trace];
    history[0] = __fma_rn(p.history[mode], state, history[0]);
    history[1] = __fma_rn(p.history[p.modes + mode], state, history[1]);
  }
  p.history_out[trace] = history[0];
  p.history_out[p.traces + trace] = history[1];
  if (trace >= p.chords) return;
  const double current0 = p.current[2 * trace], current1 = p.current[2 * trace + 1];
  for (int side = 0; side < 2; ++side) {
    const int64_t row = 2 * trace + side;
    const double capacity = __dsub_rn(__dmul_rn(p.mass[row], p.inverse_dt),
                                     __dmul_rn(p.conductance[row], .5));
    double rhs = __dadd_rn(__dmul_rn(capacity, p.current[row]), p.forcing[row]);
    const double* effective = p.feeds ? p.local + 4 * trace : p.effective;
    const double local = sheet_dot2(effective + 2 * side, current0, current1);
    double history_term;
    if (p.feeds) {
      history_term = __dadd_rn(__dmul_rn(__dmul_rn(p.feed_area[trace], local), .5),
                                __dmul_rn(p.area[trace], history[side]));
    } else {
      history_term = __dmul_rn(p.area[trace], __dadd_rn(__dmul_rn(local, .5), history[side]));
    }
    rhs = __dsub_rn(rhs, history_term);
    if (p.feeds) rhs = __dsub_rn(rhs, __dmul_rn(sheet_mv(p.feed, row, p.current), .5));
    const double coupled = p.coupled[row];
    // Feed source index-adds precede this subtraction in the reference.
    p.rhs[row] = p.feeds ? rhs : __dsub_rn(rhs, coupled);
  }
}

template <int AxL, int AcL>
__global__ void thin_sheet_chord_forcing_kernel(SheetCsr acx, SheetCsr acc,
    const double* old, const double* next, const double* x_previous,
    const double* current, const double* previous, const double* updated,
    double* forcing, int64_t chords, double quarter_dt) {
  constexpr int L = AxL > AcL ? AxL : AcL;
  const int row = blockIdx.x * (sheet_row_threads / L) + threadIdx.x / L;
  if (row >= 2 * chords) return;
  const double ax = sheet_affine_mv<AxL, L>(acx, row, next, old, x_previous);
  const double ac = sheet_affine_mv<AcL, L>(acc, row, updated, current, previous);
  if (threadIdx.x % L == 0)
    forcing[row] = __dsub_rn(forcing[row], __dmul_rn(quarter_dt, __dadd_rn(ax, ac)));
}

struct SheetState {
  int64_t traces, chords, modes, edges, pair_old_row, pair_old_column, pair_new_row, pair_new_column;
  const double *decay, *drive, *rates, *D, *area, *updated, *pair_old, *pair_new, *old;
  const bool* feed_mask;
  const int64_t* promoted;
  double *state, *current, *previous, *next, *x_previous, *midpoint, *loss;
  volatile double* partials;
  unsigned int* counter;
  double dt;
};

__device__ double sheet_block_sum(double value, double* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = copper_threads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  return shared[0];
}

__global__ void thin_sheet_copper_state_kernel(SheetState p) {
  const int64_t trace = int64_t(blockIdx.x) * copper_threads + threadIdx.x;
  if (trace < p.edges) p.x_previous[trace] = p.old[trace];
  double heat = 0.;
  if (trace < p.traces) {
    double old0, old1, next0, next1;
    if (trace < p.chords) {
      old0 = p.current[2 * trace]; old1 = p.current[2 * trace + 1];
      next0 = p.updated[2 * trace]; next1 = p.updated[2 * trace + 1];
    } else {
      const int64_t pair = trace - p.chords;
      old0 = p.pair_old[pair * p.pair_old_row];
      old1 = p.pair_old[pair * p.pair_old_row + p.pair_old_column];
      next0 = p.pair_new[pair * p.pair_new_row];
      next1 = p.pair_new[pair * p.pair_new_row + p.pair_new_column];
    }
    const double sum0 = __dadd_rn(next0, old0), sum1 = __dadd_rn(next1, old1);
    const double mid0 = __dmul_rn(sum0, .5), mid1 = __dmul_rn(sum1, .5);
    p.midpoint[2 * trace] = mid0; p.midpoint[2 * trace + 1] = mid1;
    for (int64_t mode = 0; mode < p.modes; ++mode) {
      const int64_t index = mode * p.traces + trace;
      const double state = p.state[index];
      const double driven = sheet_dot2(p.drive + 2 * mode, sum0, sum1);
      const double next = p.feed_mask[trace] ? 0. : __dadd_rn(__dmul_rn(p.decay[mode], state), driven);
      const double midpoint = __dmul_rn(__dadd_rn(next, state), .5);
      heat = __dadd_rn(heat, __dmul_rn(p.rates[mode], __dmul_rn(midpoint, midpoint)));
      p.state[index] = next;
    }
    const double direct0 = sheet_dot2(p.D, mid0, mid1);
    const double direct1 = sheet_dot2(p.D + 2, mid0, mid1);
    heat = __dadd_rn(heat, __dadd_rn(__dmul_rn(mid0, direct0), __dmul_rn(mid1, direct1)));
    heat = __dmul_rn(heat, p.area[trace]);
    if (trace < p.chords) {
      p.previous[2 * trace] = old0; p.previous[2 * trace + 1] = old1;
      p.current[2 * trace] = next0; p.current[2 * trace + 1] = next1;
      if (p.promoted[2 * trace] >= 0) {
        p.next[p.promoted[2 * trace]] = next0;
        p.next[p.promoted[2 * trace + 1]] = next1;
      }
    }
  }
  __shared__ double shared[copper_threads];
  __shared__ bool last;
  const double subtotal = sheet_block_sum(heat, shared);
  if (threadIdx.x == 0) {
    p.partials[blockIdx.x] = subtotal;
    __threadfence();
    last = atomicInc(p.counter, gridDim.x - 1) == gridDim.x - 1;
  }
  __syncthreads();
  if (last) {
    double total = 0.;
    for (unsigned int block = threadIdx.x; block < gridDim.x; block += copper_threads)
      total = __dadd_rn(total, p.partials[block]);
    total = sheet_block_sum(total, shared);
    if (threadIdx.x == 0) *p.loss = __dadd_rn(*p.loss, __dmul_rn(p.dt, total));
  }
}
}  // namespace

void thin_sheet_chord_initialize_cuda(const torch::stable::Tensor& rhs,
    torch::stable::Tensor y, torch::stable::Tensor residual) {
  const int64_t count = rhs.numel();
  if (!count) return;
  const torch::stable::accelerator::DeviceGuard guard(rhs.get_device_index());
  thin_sheet_chord_initialize_kernel<<<linear_grid(count, threads), threads, 0, current_cuda_stream()>>>(
      rhs.const_data_ptr<double>(), y.mutable_data_ptr<double>(), residual.mutable_data_ptr<double>(), count);
  WITWIN_CUDA_CHECK();
}

void thin_sheet_chord_residual_cuda(const torch::stable::Tensor& rows,
    const torch::stable::Tensor& columns, const torch::stable::Tensor& values, int64_t lanes,
    const torch::stable::Tensor& direction, torch::stable::Tensor y,
    torch::stable::Tensor residual) {
  const int64_t count = residual.numel();
  if (!count) return;
  const torch::stable::accelerator::DeviceGuard guard(residual.get_device_index());
  sheet_dispatch_rows(lanes, [&](auto tag) {
    constexpr int L = decltype(tag)::value;
    thin_sheet_chord_residual_kernel<L><<<linear_grid(count, sheet_row_threads / L),
                                        sheet_row_threads, 0, current_cuda_stream()>>>(
        sheet_csr(rows, columns, values), direction.const_data_ptr<double>(),
        y.mutable_data_ptr<double>(), residual.mutable_data_ptr<double>(), count);
  });
  WITWIN_CUDA_CHECK();
}

void thin_sheet_chord_matvec_cuda(const torch::stable::Tensor& rows,
    const torch::stable::Tensor& columns, const torch::stable::Tensor& values, int64_t lanes,
    const torch::stable::Tensor& residual, torch::stable::Tensor temporary) {
  const int64_t count = temporary.numel();
  if (!count) return;
  const torch::stable::accelerator::DeviceGuard guard(temporary.get_device_index());
  sheet_dispatch_rows(lanes, [&](auto tag) {
    constexpr int L = decltype(tag)::value;
    thin_sheet_chord_matvec_kernel<L><<<linear_grid(count, sheet_row_threads / L),
                                      sheet_row_threads, 0, current_cuda_stream()>>>(
        sheet_csr(rows, columns, values), residual.const_data_ptr<double>(),
        temporary.mutable_data_ptr<double>(), count);
  });
  WITWIN_CUDA_CHECK();
}

void thin_sheet_chord_direction_cuda(const torch::stable::Tensor& rows,
    const torch::stable::Tensor& columns, const torch::stable::Tensor& values, int64_t lanes,
    const torch::stable::Tensor& temporary, torch::stable::Tensor direction,
    double alpha, double beta, bool setup) {
  const int64_t count = direction.numel();
  if (!count) return;
  const torch::stable::accelerator::DeviceGuard guard(direction.get_device_index());
  sheet_dispatch_rows(lanes, [&](auto tag) {
    constexpr int L = decltype(tag)::value;
    thin_sheet_chord_direction_kernel<L><<<linear_grid(count, sheet_row_threads / L),
                                         sheet_row_threads, 0, current_cuda_stream()>>>(
        sheet_csr(rows, columns, values), temporary.const_data_ptr<double>(),
        direction.mutable_data_ptr<double>(), count, alpha, beta, setup);
  });
  WITWIN_CUDA_CHECK();
}

void thin_sheet_copper_rhs_cuda(const torch::stable::Tensor& state,
    const torch::stable::Tensor& history, const torch::stable::Tensor& effective,
    const std::optional<torch::stable::Tensor>& local, const torch::stable::Tensor& mass,
    const torch::stable::Tensor& conductance, const torch::stable::Tensor& area,
    const torch::stable::Tensor& feed_area, const torch::stable::Tensor& current,
    const torch::stable::Tensor& previous, const torch::stable::Tensor& forcing,
    const torch::stable::Tensor& old, const torch::stable::Tensor& next,
    const torch::stable::Tensor& x_previous, const torch::stable::Tensor& ax_rows,
    const torch::stable::Tensor& ax_columns, const torch::stable::Tensor& ax_values, int64_t ax_lanes,
    const torch::stable::Tensor& ac_rows, const torch::stable::Tensor& ac_columns,
    const torch::stable::Tensor& ac_values, int64_t ac_lanes,
    const std::optional<torch::stable::Tensor>& feed_rows,
    const std::optional<torch::stable::Tensor>& feed_columns,
    const std::optional<torch::stable::Tensor>& feed_values, torch::stable::Tensor history_out,
    torch::stable::Tensor rhs, torch::stable::Tensor coupled, double dt, bool feeds) {
  const int64_t traces = state.size(1);
  if (!traces) return;
  const torch::stable::accelerator::DeviceGuard guard(state.get_device_index());
  SheetRhs p{traces, current.size(0), state.size(0), state.const_data_ptr<double>(),
      history.const_data_ptr<double>(), effective.const_data_ptr<double>(),
      local ? local->const_data_ptr<double>() : nullptr, mass.const_data_ptr<double>(),
      conductance.const_data_ptr<double>(), area.const_data_ptr<double>(), feed_area.const_data_ptr<double>(),
      current.const_data_ptr<double>(), previous.const_data_ptr<double>(), forcing.const_data_ptr<double>(),
      old.const_data_ptr<double>(), next.const_data_ptr<double>(), x_previous.const_data_ptr<double>(),
      sheet_csr(ax_rows, ax_columns, ax_values), sheet_csr(ac_rows, ac_columns, ac_values),
      feed_rows ? sheet_csr(*feed_rows, *feed_columns, *feed_values) : SheetCsr{nullptr, nullptr, nullptr},
      history_out.mutable_data_ptr<double>(), rhs.mutable_data_ptr<double>(), coupled.mutable_data_ptr<double>(),
      1. / dt, dt / 4, feeds};
  const auto stream = current_cuda_stream();
  if (p.chords) {
    sheet_dispatch_rows(ax_lanes, [&](auto ax_tag) {
      sheet_dispatch_rows(ac_lanes, [&](auto ac_tag) {
        constexpr int AxL = decltype(ax_tag)::value, AcL = decltype(ac_tag)::value;
        constexpr int L = AxL > AcL ? AxL : AcL;
        thin_sheet_copper_coupling_kernel<AxL, AcL><<<linear_grid(2 * p.chords, sheet_row_threads / L),
                                                     sheet_row_threads, 0, stream>>>(p);
      });
    });
    WITWIN_CUDA_CHECK();
  }
  thin_sheet_copper_rhs_kernel<<<linear_grid(traces, copper_threads), copper_threads, 0, stream>>>(p);
  WITWIN_CUDA_CHECK();
}

void thin_sheet_chord_forcing_cuda(const torch::stable::Tensor& ax_rows,
    const torch::stable::Tensor& ax_columns, const torch::stable::Tensor& ax_values, int64_t ax_lanes,
    const torch::stable::Tensor& ac_rows, const torch::stable::Tensor& ac_columns,
    const torch::stable::Tensor& ac_values, int64_t ac_lanes, const torch::stable::Tensor& old,
    const torch::stable::Tensor& next, const torch::stable::Tensor& x_previous,
    const torch::stable::Tensor& current, const torch::stable::Tensor& previous,
    const torch::stable::Tensor& updated, torch::stable::Tensor forcing, double dt) {
  const int64_t chords = current.size(0);
  if (!chords) return;
  const torch::stable::accelerator::DeviceGuard guard(current.get_device_index());
  sheet_dispatch_rows(ax_lanes, [&](auto ax_tag) {
    sheet_dispatch_rows(ac_lanes, [&](auto ac_tag) {
      constexpr int AxL = decltype(ax_tag)::value, AcL = decltype(ac_tag)::value;
      constexpr int L = AxL > AcL ? AxL : AcL;
      thin_sheet_chord_forcing_kernel<AxL, AcL><<<linear_grid(2 * chords, sheet_row_threads / L),
                                                 sheet_row_threads, 0, current_cuda_stream()>>>(
          sheet_csr(ax_rows, ax_columns, ax_values), sheet_csr(ac_rows, ac_columns, ac_values),
          old.const_data_ptr<double>(), next.const_data_ptr<double>(), x_previous.const_data_ptr<double>(),
          current.const_data_ptr<double>(), previous.const_data_ptr<double>(), updated.const_data_ptr<double>(),
          forcing.mutable_data_ptr<double>(), chords, dt / 4);
    });
  });
  WITWIN_CUDA_CHECK();
}

void thin_sheet_copper_state_cuda(torch::stable::Tensor state, const torch::stable::Tensor& decay,
    const torch::stable::Tensor& drive, const torch::stable::Tensor& rates,
    const torch::stable::Tensor& D, const torch::stable::Tensor& area,
    const torch::stable::Tensor& feed_mask, torch::stable::Tensor current, torch::stable::Tensor previous,
    const torch::stable::Tensor& updated, const torch::stable::Tensor& pair_old,
    const torch::stable::Tensor& pair_new, const torch::stable::Tensor& old,
    torch::stable::Tensor next, torch::stable::Tensor x_previous,
    const torch::stable::Tensor& promoted, torch::stable::Tensor midpoint,
    torch::stable::Tensor loss, torch::stable::Tensor partials, torch::stable::Tensor counter, double dt) {
  const int64_t count = std::max(state.size(1), old.numel());
  if (!count) return;
  const torch::stable::accelerator::DeviceGuard guard(state.get_device_index());
  SheetState p{state.size(1), current.size(0), state.size(0), old.numel(),
      pair_old.stride(0), pair_old.stride(1), pair_new.stride(0), pair_new.stride(1),
      decay.const_data_ptr<double>(), drive.const_data_ptr<double>(), rates.const_data_ptr<double>(),
      D.const_data_ptr<double>(), area.const_data_ptr<double>(), updated.const_data_ptr<double>(),
      pair_old.const_data_ptr<double>(), pair_new.const_data_ptr<double>(), old.const_data_ptr<double>(),
      feed_mask.const_data_ptr<bool>(), promoted.const_data_ptr<int64_t>(),
      state.mutable_data_ptr<double>(), current.mutable_data_ptr<double>(), previous.mutable_data_ptr<double>(),
      next.mutable_data_ptr<double>(), x_previous.mutable_data_ptr<double>(), midpoint.mutable_data_ptr<double>(),
      loss.mutable_data_ptr<double>(), partials.mutable_data_ptr<double>(),
      reinterpret_cast<unsigned int*>(counter.mutable_data_ptr<int32_t>()), dt};
  thin_sheet_copper_state_kernel<<<linear_grid(count, copper_threads), copper_threads, 0, current_cuda_stream()>>>(p);
  WITWIN_CUDA_CHECK();
}
