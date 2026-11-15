// Gyromagnetic (Polder ferrite) magnetization step: the drive snapshot before the
// magnetic Yee update and the coupled implicit-midpoint solve with its H correction
// after it (fdtd/runtime/gyromagnetic.py has the derivation).
//
// One thread per active ferrite cell of the shared (Nx-1, Ny-1, Nz-1) overlap, so no
// cell outside the ferrite is read. Every overlap cell owns one entry of each H
// component (identity collocation), so the per-cell read-modify-write needs no
// atomics. Per-cell state and coefficients are dense overlap fields indexed by the
// cell's overlap offset; the packed coefficient planes are `plane` elements apart.
#include "../launch.h"
#include "../tensors.h"

namespace {
constexpr int threads = 256;

struct GyromagneticCells {
  int64_t count;      // active cells
  int64_t ny, nz;     // overlap extents Ny-1, Nz-1
  int64_t plane;      // overlap cell count
  const int64_t* cells;
  int u_axis, v_axis;  // fast path: the lab H component driving m_u / m_v
  double sign_u, sign_v;
};

// Offset of overlap cell (i, j, k) inside the staggered H component `axis`.
__device__ __forceinline__ int64_t h_offset(int axis, int64_t cell, int64_t ny, int64_t nz) {
  if (axis == 0) return cell;  // Hx (Nx, Ny-1, Nz-1): the overlap is its leading block
  const int64_t ij = cell / nz, k = cell - ij * nz;
  if (axis == 1) {
    const int64_t i = ij / ny, j = ij - i * ny;
    return (i * (ny + 1) + j) * nz + k;  // Hy (Nx-1, Ny, Nz-1)
  }
  return ij * (nz + 1) + k;  // Hz (Nx-1, Ny-1, Nz)
}

// Transverse RF drive (h_u, h_v) at one cell: a signed lab component per axis on the
// fast path, the projections u . H and v . H through the per-cell frame otherwise.
template <typename T, bool general>
__device__ __forceinline__ void gather_drive(
    T* const h[3], const int64_t offset[3], const T* frame, int64_t cell,
    const GyromagneticCells& p, T& hu, T& hv) {
  if constexpr (general) {
    const T hx = h[0][offset[0]], hy = h[1][offset[1]], hz = h[2][offset[2]];
    hu = fma(frame[2 * p.plane + cell], hz, fma(frame[p.plane + cell], hy, frame[cell] * hx));
    hv = fma(frame[5 * p.plane + cell], hz, fma(frame[4 * p.plane + cell], hy, frame[3 * p.plane + cell] * hx));
  } else {
    hu = T(p.sign_u) * h[p.u_axis][offset[p.u_axis]];
    hv = T(p.sign_v) * h[p.v_axis][offset[p.v_axis]];
  }
}

template <typename T, bool general>
__global__ void gyromagnetic_snapshot_kernel(
    T* hx, T* hy, T* hz, const T* frame, T* pre_u, T* pre_v, GyromagneticCells p) {
  const int64_t index = int64_t(blockIdx.x) * threads + threadIdx.x;
  if (index >= p.count) return;
  const int64_t cell = p.cells[index];
  T* const h[3] = {hx, hy, hz};
  const int64_t offset[3] = {cell, h_offset(1, cell, p.ny, p.nz), h_offset(2, cell, p.ny, p.nz)};
  T hu, hv;
  gather_drive<T, general>(h, offset, frame, cell, p, hu, hv);
  pre_u[cell] = hu;
  pre_v[cell] = hv;
}

// m_new = B m_old + C (h_pre + h_tmp), dM = m_new - m_old, H^{n+1} = H_tmp - dM / mu_inf.
// `coupled` packs B00, B01, B10, B11, C00, C01, C10, C11.
template <typename T, bool general>
__global__ void gyromagnetic_step_kernel(
    T* hx, T* hy, T* hz, const T* frame, const T* coupled, const T* inv_mu,
    const T* pre_u, const T* pre_v, T* m_u, T* m_v, T* dm_u, T* dm_v, GyromagneticCells p) {
  const int64_t index = int64_t(blockIdx.x) * threads + threadIdx.x;
  if (index >= p.count) return;
  const int64_t cell = p.cells[index];
  T* const h[3] = {hx, hy, hz};
  const int64_t offset[3] = {cell, h_offset(1, cell, p.ny, p.nz), h_offset(2, cell, p.ny, p.nz)};
  T hu, hv;
  gather_drive<T, general>(h, offset, frame, cell, p, hu, hv);
  hu += pre_u[cell];
  hv += pre_v[cell];

  const T old_u = m_u[cell], old_v = m_v[cell];
  const T* b = coupled + cell;
  T new_u = b[0] * old_u;
  new_u = fma(b[p.plane], old_v, new_u);
  new_u = fma(b[4 * p.plane], hu, new_u);
  new_u = fma(b[5 * p.plane], hv, new_u);
  T new_v = b[2 * p.plane] * old_u;
  new_v = fma(b[3 * p.plane], old_v, new_v);
  new_v = fma(b[6 * p.plane], hu, new_v);
  new_v = fma(b[7 * p.plane], hv, new_v);
  const T du = new_u - old_u, dv = new_v - old_v;
  m_u[cell] = old_u + du;
  m_v[cell] = old_v + dv;
  dm_u[cell] = du;
  dm_v[cell] = dv;

  const T scale = inv_mu[cell];
  if constexpr (general) {
    for (int axis = 0; axis < 3; ++axis) {
      const T correction = fma(dv, frame[(axis + 3) * p.plane + cell], du * frame[axis * p.plane + cell]);
      h[axis][offset[axis]] -= correction * scale;
    }
  } else {
    h[p.u_axis][offset[p.u_axis]] -= du * scale * T(p.sign_u);
    h[p.v_axis][offset[p.v_axis]] -= dv * scale * T(p.sign_v);
  }
}

GyromagneticCells cell_params(const torch::stable::Tensor& hx, const torch::stable::Tensor& cells,
                              const torch::stable::Tensor& state, int64_t u_axis, int64_t v_axis,
                              double sign_u, double sign_v) {
  check_cuda_tensor(hx, "hx");
  check_contiguous_tensor(cells, "cells");
  check_contiguous_tensor(state, "state");
  STD_TORCH_CHECK(cells.scalar_type() == torch::headeronly::ScalarType::Long, "cells must be int64");
  STD_TORCH_CHECK(state.dim() == 3, "gyromagnetic state must be a dense (Nx-1, Ny-1, Nz-1) overlap field");
  STD_TORCH_CHECK(hx.dim() == 3 && hx.size(0) == state.size(0) + 1 && hx.size(1) == state.size(1)
                  && hx.size(2) == state.size(2), "hx must be (Nx, Ny-1, Nz-1) over the state overlap");
  STD_TORCH_CHECK(u_axis >= 0 && u_axis < 3 && v_axis >= 0 && v_axis < 3, "drive axes must be 0, 1 or 2");
  GyromagneticCells p;
  p.count = cells.numel();
  p.ny = state.size(1);
  p.nz = state.size(2);
  p.plane = state.numel();
  p.cells = cells.const_data_ptr<int64_t>();
  p.u_axis = int(u_axis);
  p.v_axis = int(v_axis);
  p.sign_u = sign_u;
  p.sign_v = sign_v;
  return p;
}

template <typename T>
const T* optional_data(const std::optional<torch::stable::Tensor>& tensor) {
  return tensor ? tensor->const_data_ptr<T>() : nullptr;
}
}  // namespace

void gyromagnetic_snapshot_cuda(
    torch::stable::Tensor hx, torch::stable::Tensor hy, torch::stable::Tensor hz,
    const torch::stable::Tensor& cells, const std::optional<torch::stable::Tensor>& frame,
    torch::stable::Tensor pre_u, torch::stable::Tensor pre_v,
    int64_t u_axis, int64_t v_axis, double sign_u, double sign_v) {
  const GyromagneticCells p = cell_params(hx, cells, pre_u, u_axis, v_axis, sign_u, sign_v);
  if (p.count == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(hx.get_device_index());
  const dim3 grid = linear_grid(p.count, threads);
  const auto stream = current_cuda_stream();
#define LAUNCH_SNAPSHOT(type, general)                                                        \
  gyromagnetic_snapshot_kernel<type, general><<<grid, threads, 0, stream>>>(                  \
      hx.mutable_data_ptr<type>(), hy.mutable_data_ptr<type>(), hz.mutable_data_ptr<type>(), \
      optional_data<type>(frame), pre_u.mutable_data_ptr<type>(), pre_v.mutable_data_ptr<type>(), p)
  if (hx.scalar_type() == torch::headeronly::ScalarType::Double) {
    if (frame) { LAUNCH_SNAPSHOT(double, true); } else { LAUNCH_SNAPSHOT(double, false); }
  } else {
    if (frame) { LAUNCH_SNAPSHOT(float, true); } else { LAUNCH_SNAPSHOT(float, false); }
  }
#undef LAUNCH_SNAPSHOT
  WITWIN_CUDA_CHECK();
}

void gyromagnetic_step_cuda(
    torch::stable::Tensor hx, torch::stable::Tensor hy, torch::stable::Tensor hz,
    const torch::stable::Tensor& cells, const std::optional<torch::stable::Tensor>& frame,
    const torch::stable::Tensor& coupled, const torch::stable::Tensor& inv_mu,
    const torch::stable::Tensor& pre_u, const torch::stable::Tensor& pre_v,
    torch::stable::Tensor m_u, torch::stable::Tensor m_v,
    torch::stable::Tensor dm_u, torch::stable::Tensor dm_v,
    int64_t u_axis, int64_t v_axis, double sign_u, double sign_v) {
  const GyromagneticCells p = cell_params(hx, cells, m_u, u_axis, v_axis, sign_u, sign_v);
  STD_TORCH_CHECK(coupled.numel() == 8 * p.plane, "coupled must pack eight overlap planes");
  STD_TORCH_CHECK(!frame || frame->numel() == 6 * p.plane, "frame must pack six overlap planes");
  if (p.count == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(hx.get_device_index());
  const dim3 grid = linear_grid(p.count, threads);
  const auto stream = current_cuda_stream();
#define LAUNCH_STEP(type, general)                                                                \
  gyromagnetic_step_kernel<type, general><<<grid, threads, 0, stream>>>(                          \
      hx.mutable_data_ptr<type>(), hy.mutable_data_ptr<type>(), hz.mutable_data_ptr<type>(),     \
      optional_data<type>(frame), coupled.const_data_ptr<type>(), inv_mu.const_data_ptr<type>(), \
      pre_u.const_data_ptr<type>(), pre_v.const_data_ptr<type>(), m_u.mutable_data_ptr<type>(),  \
      m_v.mutable_data_ptr<type>(), dm_u.mutable_data_ptr<type>(), dm_v.mutable_data_ptr<type>(), p)
  if (hx.scalar_type() == torch::headeronly::ScalarType::Double) {
    if (frame) { LAUNCH_STEP(double, true); } else { LAUNCH_STEP(double, false); }
  } else {
    if (frame) { LAUNCH_STEP(float, true); } else { LAUNCH_STEP(float, false); }
  }
#undef LAUNCH_STEP
  WITWIN_CUDA_CHECK();
}
