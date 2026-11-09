#include <cstdint>
#include <c10/cuda/CUDAException.h>
#include "yee.cuh"
#include "../kernels/modulation.cuh"

namespace {

__device__ __forceinline__ long long offset(unsigned i, unsigned j, unsigned k,
                                            unsigned ny, unsigned nz) {
  return (static_cast<long long>(i) * ny + j) * nz + k;
}

template <int Axis>
__device__ __forceinline__ long long stride(unsigned ny, unsigned nz) {
  return Axis == 0 ? static_cast<long long>(ny) * nz : (Axis == 1 ? nz : 1);
}

template <int Axis>
__device__ __forceinline__ float forward(const float* source, float center,
    unsigned i, unsigned j, unsigned k, unsigned ny, unsigned nz, const float* inv) {
  const unsigned coord = Axis == 0 ? i : (Axis == 1 ? j : k);
  return (source[offset(i, j, k, ny, nz) + stride<Axis>(ny, nz)] - center) * inv[coord];
}

template <int Axis>
__device__ __forceinline__ float backward(const float* source, float center,
    unsigned i, unsigned j, unsigned k, unsigned nx, unsigned ny, unsigned nz,
    CoreAxis axis) {
  const unsigned coord = Axis == 0 ? i : (Axis == 1 ? j : k);
  const unsigned size = Axis == 0 ? nx : (Axis == 1 ? ny : nz);
  const long long current = offset(i, j, k, ny, nz);
  if (coord == 0 || coord == size) {
    const int mode = coord == 0 ? axis.low_mode : axis.high_mode;
    if (mode == 2) {
      const long long low = current - coord * stride<Axis>(ny, nz);
      const long long high = low + (size - 1) * stride<Axis>(ny, nz);
      const float low_value = coord == 0 ? center : source[low];
      return (low_value - source[high]) * axis.inv[coord];
    }
    return 0.0f;
  }
  return (center - source[current - stride<Axis>(ny, nz)]) * axis.inv[coord];
}

__device__ __forceinline__ bool pec(unsigned coord, unsigned size, CoreAxis axis) {
  return (coord == 0 && axis.low_mode == 4) ||
         (coord + 1 == size && axis.high_mode == 4);
}

__device__ __forceinline__ bool inactive(unsigned coord, unsigned size, CoreAxis axis) {
  return (coord == 0 && (axis.low_mode == 0 || axis.low_mode == 1)) ||
         (coord + 1 == size && (axis.high_mode == 0 || axis.high_mode == 1));
}

__device__ __forceinline__ bool inactive_pair(CoreAxis axis) {
  return (axis.low_mode == 0 || axis.low_mode == 1) &&
         (axis.high_mode == 0 || axis.high_mode == 1);
}

template <int Axis>
__device__ __forceinline__ float interior_difference(const float* source, float center,
    unsigned i, unsigned j, unsigned k, unsigned ny, unsigned nz) {
  return center - source[offset(i, j, k, ny, nz) - stride<Axis>(ny, nz)];
}

template <int Axis, int Cpml, bool Explicit = false>
__device__ __forceinline__ float psi_update(CorePsi psi, CoreAxis axis, float d,
    unsigned i, unsigned j, unsigned k, unsigned ny, unsigned nz) {
  const unsigned coord = Axis == 0 ? i : (Axis == 1 ? j : k);
  long long index;
  if constexpr (Cpml == 1) {
    index = offset(i, j, k, ny, nz);
  } else {
    int local;
    if (coord < unsigned(psi.low)) local = coord;
    else if (coord >= unsigned(psi.high_start) && coord < unsigned(psi.high_start + psi.high))
      local = psi.low + coord - psi.high_start;
    else return 0.0f;
    const unsigned length = psi.low + psi.high;
    if constexpr (Axis == 0) index = offset(local, j, k, ny, nz);
    else if constexpr (Axis == 1) index = offset(i, local, k, length, nz);
    else index = offset(i, j, local, ny, length);
  }
  float value;
  if constexpr (Explicit)
    value = __fmaf_rn(axis.b[coord], psi.data[index], __fmul_rn(axis.c[coord], d));
  else
    value = axis.b[coord] * psi.data[index] + axis.c[coord] * d;
  psi.data[index] = value;
  return value;
}

// Slab-CPML z psi. Lanes run along k, so the z slab is a short run at each end of
// every (i, j) row: an early-returning branch diverges in every boundary warp, and
// the psi load would sit behind the derivative loads and the preceding psi store
// (which the compiler may not reorder across without restrict). The load is issued
// before the derivative loads by every lane of the row; lanes outside the slab read
// an in-row element and discard it, and only slab lanes store. The per-cell
// arithmetic is that of psi_update.
struct ZPsi { long long index; float old; bool in_slab; bool active; };

__device__ __forceinline__ ZPsi z_psi_load(CorePsi psi, unsigned i, unsigned j, unsigned k,
    unsigned ny) {
  ZPsi z{0, 0.0f, false, false};
  const unsigned length = psi.low + psi.high;
  if (length == 0) return z;
  int local = -1;
  if (k < unsigned(psi.low)) local = k;
  else if (k >= unsigned(psi.high_start) && k < unsigned(psi.high_start + psi.high))
    local = psi.low + k - psi.high_start;
  z.active = true;
  z.in_slab = local >= 0;
  z.index = offset(i, j, z.in_slab ? local : 0, ny, length);
  z.old = psi.data[z.index];
  return z;
}

template <bool Explicit>
__device__ __forceinline__ float z_psi_update(CorePsi psi, CoreAxis axis, float d, ZPsi z, unsigned k) {
  if (!z.active) return 0.0f;
  float value;
  if constexpr (Explicit)
    value = __fmaf_rn(axis.b[k], z.old, __fmul_rn(axis.c[k], d));
  else
    value = axis.b[k] * z.old + axis.c[k] * d;
  if (z.in_slab) psi.data[z.index] = value;
  return z.in_slab ? value : 0.0f;
}

// Coefficients: 0 = uniform, 1 = indexed table, 2 = live dense, 3 = live modulated.
template <int Component, bool Electric, int Cpml, int Mode>
__device__ __forceinline__ void component(CoreHalf half, unsigned i, unsigned j, unsigned k,
    float s0, float s1, float s2, const float* coefficients) {
  constexpr bool Uniform = Mode == 0;
  constexpr bool Live = Mode >= 2;
  constexpr bool Modulated = Mode == 3;
  constexpr int A = Component == 0 ? 1 : 0;
  constexpr int B = Component == 2 ? 1 : 2;
  const unsigned nx = half.nx - (Electric ? Component == 0 : Component != 0);
  const unsigned ny = half.ny - (Electric ? Component == 1 : Component != 1);
  const unsigned nz = half.nz - (Electric ? Component == 2 : Component != 2);
  if (i >= nx || j >= ny || k >= nz) return;
  const long long linear = offset(i, j, k, ny, nz);
  float* field = half.field[Component];
  const unsigned coord_a = A == 0 ? i : j;
  const unsigned coord_b = B == 1 ? j : k;
  if constexpr (Electric) {
    const unsigned size_a = A == 0 ? nx : ny;
    const unsigned size_b = B == 1 ? ny : nz;
    if (pec(coord_a, size_a, half.axes[A]) || pec(coord_b, size_b, half.axes[B])) {
      field[linear] = 0.0f;
      return;
    }
    if (inactive(coord_a, size_a, half.axes[A]) || inactive(coord_b, size_b, half.axes[B])) return;
  }
  ZPsi z_psi{0, 0.0f, false, false};
  if constexpr (Cpml == 2 && B == 2) z_psi = z_psi_load(half.psi[2*Component+1], i, j, k, ny);
  // The center of each cross field is loaded once by the caller and shared
  // between the two components that differentiate it.
  if constexpr (Electric && Cpml == 0 && !Modulated) {
    // Match the standard launch's per-component interior-kernel selection.
    if (inactive_pair(half.axes[A]) && inactive_pair(half.axes[B])) {
      float delta_a, delta_b;
      if constexpr (Component == 0) {
        delta_a = interior_difference<1>(half.source[2], s2, i,j,k, ny-1,nz);
        delta_b = interior_difference<2>(half.source[1], s1, i,j,k, ny,nz-1);
      } else if constexpr (Component == 1) {
        delta_a = interior_difference<0>(half.source[2], s2, i,j,k, ny,nz);
        delta_b = interior_difference<2>(half.source[0], s0, i,j,k, ny,nz-1);
      } else {
        delta_a = interior_difference<0>(half.source[1], s1, i,j,k, ny,nz);
        delta_b = interior_difference<1>(half.source[0], s0, i,j,k, ny-1,nz);
      }
      // The standard interior SASS fuses the positive derivative into curl.
      float curl;
      if constexpr (Component == 1)
        curl = __fmaf_rn(delta_b, half.axes[B].inv[coord_b], -__fmul_rn(delta_a, half.axes[A].inv[coord_a]));
      else
        curl = __fmaf_rn(delta_a, half.axes[A].inv[coord_a], -__fmul_rn(delta_b, half.axes[B].inv[coord_b]));
      const float decay = Live ? half.dense[2*Component][linear] : (Uniform ? half.coefficients[2*Component] : coefficients[2*Component]);
      const float factor = Live ? half.dense[2*Component+1][linear] : (Uniform ? half.coefficients[2*Component+1] : coefficients[2*Component+1]);
      field[linear] = __fmaf_rn(field[linear], decay, __fmul_rn(factor, curl));
      return;
    }
  }
  float d_a, d_b;
  if constexpr (Electric) {
    if constexpr (Component == 0) {
      d_a = backward<1>(half.source[2], s2, i,j,k, nx,ny-1,nz, half.axes[A]);
      d_b = backward<2>(half.source[1], s1, i,j,k, nx,ny,nz-1, half.axes[B]);
    } else if constexpr (Component == 1) {
      d_a = backward<0>(half.source[2], s2, i,j,k, nx-1,ny,nz, half.axes[A]);
      d_b = backward<2>(half.source[0], s0, i,j,k, nx,ny,nz-1, half.axes[B]);
    } else {
      d_a = backward<0>(half.source[1], s1, i,j,k, nx-1,ny,nz, half.axes[A]);
      d_b = backward<1>(half.source[0], s0, i,j,k, nx,ny-1,nz, half.axes[B]);
    }
  } else {
    if constexpr (Component == 0) {
      d_a = forward<1>(half.source[2], s2, i,j,k, ny+1,nz, half.axes[A].inv);
      d_b = forward<2>(half.source[1], s1, i,j,k, ny,nz+1, half.axes[B].inv);
    } else if constexpr (Component == 1) {
      d_a = forward<0>(half.source[2], s2, i,j,k, ny,nz, half.axes[A].inv);
      d_b = forward<2>(half.source[0], s0, i,j,k, ny,nz+1, half.axes[B].inv);
    } else {
      d_a = forward<0>(half.source[1], s1, i,j,k, ny,nz, half.axes[A].inv);
      d_b = forward<1>(half.source[0], s0, i,j,k, ny+1,nz, half.axes[B].inv);
    }
  }
  float curl;
  if constexpr (Cpml != 0) {
    const float psi_a = psi_update<A, Cpml, Electric && Mode == 2>(half.psi[2*Component], half.axes[A], d_a, i,j,k,ny,nz);
    float psi_b;
    if constexpr (Cpml == 2 && B == 2)
      psi_b = z_psi_update<Electric && Mode == 2>(half.psi[2*Component+1], half.axes[B], d_b, z_psi, k);
    else
      psi_b = psi_update<B, Cpml, Electric && Mode == 2>(half.psi[2*Component+1], half.axes[B], d_b, i,j,k,ny,nz);
    if constexpr (Electric && Mode == 2) {
      // Pin the dynamic-coefficient E kernels' sm_86 SASS rounding order, as compiled by
      // nvcc 13.0.88 (kernels/electric.cu update_electric_*cpml* with non-uniform
      // decay/curl): psi = fma(b, psi, c*d), curl assembly as below, E = fma(E, D, C*curl).
      // Those kernels rely on default contraction; this path must not.
      const float corrected_a = __fmaf_rn(d_a, half.axes[A].kappa[coord_a], psi_a);
      const float corrected_b = __fmaf_rn(d_b, half.axes[B].kappa[coord_b], psi_b);
      if constexpr (Cpml == 2) {
        if constexpr (Component == 1)
          curl = __fsub_rn(__fmaf_rn(-d_a, half.axes[A].kappa[coord_a], corrected_b), psi_a);
        else
          curl = __fsub_rn(__fmaf_rn(-d_b, half.axes[B].kappa[coord_b], corrected_a), psi_b);
      } else {
        curl = Component == 1 ? __fsub_rn(corrected_b, corrected_a) : __fsub_rn(corrected_a, corrected_b);
      }
    } else if constexpr (Electric && (Cpml == 2 || Modulated)) {
      // Slab and modulated E use the historical flat expression; static dense E
      // and both H paths group corrected derivatives before subtraction.
      if constexpr (Component == 1)
        curl = d_b * half.axes[B].kappa[coord_b] + psi_b - d_a * half.axes[A].kappa[coord_a] - psi_a;
      else
        curl = d_a * half.axes[A].kappa[coord_a] + psi_a - d_b * half.axes[B].kappa[coord_b] - psi_b;
    } else {
      const float corrected_a = d_a * half.axes[A].kappa[coord_a] + psi_a;
      const float corrected_b = d_b * half.axes[B].kappa[coord_b] + psi_b;
      curl = Component == 1 ? corrected_b - corrected_a : corrected_a - corrected_b;
    }
  } else {
    if constexpr (Electric && Mode == 2)
      curl = Component == 1 ? __fsub_rn(d_b, d_a) : __fsub_rn(d_a, d_b);
    else
      curl = Component == 1 ? d_b - d_a : d_a - d_b;
  }
  const float decay = Live ? half.dense[2*Component][linear] : (Uniform ? half.coefficients[2*Component] : coefficients[2*Component]);
  const float factor = Live ? half.dense[2*Component+1][linear] : (Uniform ? half.coefficients[2*Component+1] : coefficients[2*Component+1]);
  if constexpr (Electric && Mode == 2) {
    // Dense live loads fuse the decay product, including dense/slab CPML.
    field[linear] = __fmaf_rn(field[linear], decay, __fmul_rn(factor, curl));
  } else if constexpr (Modulated) {
    const float mc = half.mod_cos[Component][linear];
    const float ms = half.mod_sin[Component][linear];
    float cos_prev, sin_prev, cos_next, sin_next;
    modulation_sincos(half.mod_omega[Component][linear], half.modulation_time[0], sin_prev, cos_prev);
    modulation_sincos(half.mod_omega[Component][linear], half.modulation_time[1], sin_next, cos_next);
    const float m_prev = 1.0f + mc * cos_prev - ms * sin_prev;
    const float inv_next = 1.0f / fmaxf(1.0f + mc * cos_next - ms * sin_next, 1.0e-6f);
    // Preserve the modulated standard/dense/slab kernels' exact combine.
    field[linear] = field[linear] * (decay * m_prev * inv_next) + (factor * inv_next) * curl;
  } else if constexpr (Electric && Cpml == 0) {
    // Standard E rounds the curl product before fusing the decay product.
    field[linear] = __fmaf_rn(field[linear], decay, __fmul_rn(factor, curl));
  }
  else if constexpr (Electric) field[linear] = field[linear] * decay + factor * curl;
  else field[linear] = field[linear] * decay - factor * curl;
}

template <bool Electric, int Cpml, int Mode, typename Index, bool Swizzled>
__global__ void yee(CoreHalf half, unsigned strip) {
  unsigned kb = blockIdx.x, jb = blockIdx.y, ib = blockIdx.z;
  if constexpr (Swizzled) {
    const unsigned gx = gridDim.x, gy = gridDim.y, gz = gridDim.z;
    // Clamp before multiplying: every intermediate fits the checked block count.
    strip = min(strip, gy);
    const unsigned linear = kb + gx * (jb + gy * ib);
    const unsigned base = gx * strip * gz;
    const unsigned s = linear / base;
    const unsigned r = linear - s * base;
    const unsigned w = min(strip, gy - s * strip);
    ib = r / (gx * w);
    const unsigned r2 = r % (gx * w);
    jb = s * strip + r2 / gx;
    kb = r2 % gx;
  }
  const unsigned k = kb * blockDim.x + threadIdx.x;
  const unsigned j = jb * blockDim.y + threadIdx.y;
  const unsigned i = ib;
  if (i >= half.nx || j >= half.ny || k >= half.nz) return;
  float centers[3];
  #pragma unroll
  for (int c = 0; c < 3; ++c) {
    const unsigned nx = half.nx - (Electric ? c != 0 : c == 0);
    const unsigned ny = half.ny - (Electric ? c != 1 : c == 1);
    const unsigned nz = half.nz - (Electric ? c != 2 : c == 2);
    centers[c] = (i < nx && j < ny && k < nz)
        ? half.source[c][offset(i,j,k,ny,nz)] : 0.0f;
  }
  const float* coefficients = nullptr;
  if constexpr (Mode == 1) {
    const auto index = static_cast<const Index*>(half.indices)[offset(i,j,k,half.ny,half.nz)];
    coefficients = half.table + static_cast<long long>(index) * 6;
  }
  component<0,Electric,Cpml,Mode>(half,i,j,k,centers[0],centers[1],centers[2],coefficients);
  component<1,Electric,Cpml,Mode>(half,i,j,k,centers[0],centers[1],centers[2],coefficients);
  component<2,Electric,Cpml,Mode>(half,i,j,k,centers[0],centers[1],centers[2],coefficients);
}

template <bool Electric, int Cpml, bool Swizzled>
void dispatch(CoreHalf half, bool uniform, bool wide, bool live, bool modulated, cudaStream_t stream) {
  const dim3 block(128,2,1);
  const dim3 grid((uint64_t(half.nz)+127)/128, (uint64_t(half.ny)+1)/2, half.nx);
  if (modulated) yee<Electric,Cpml,3,uint16_t,Swizzled><<<grid,block,0,stream>>>(half,half.block_strip);
  else if (live) yee<Electric,Cpml,2,uint16_t,Swizzled><<<grid,block,0,stream>>>(half,half.block_strip);
  else if (uniform) yee<Electric,Cpml,0,uint16_t,Swizzled><<<grid,block,0,stream>>>(half,half.block_strip);
  else if (wide) yee<Electric,Cpml,1,uint32_t,Swizzled><<<grid,block,0,stream>>>(half,half.block_strip);
  else yee<Electric,Cpml,1,uint16_t,Swizzled><<<grid,block,0,stream>>>(half,half.block_strip);
}

template <bool Electric, bool Swizzled>
void dispatch_cpml(CoreHalf half, int cpml, bool uniform, bool wide, bool live, bool modulated, cudaStream_t stream) {
  if (cpml == 0) dispatch<Electric,0,Swizzled>(half,uniform,wide,live,modulated,stream);
  else if (cpml == 1) dispatch<Electric,1,Swizzled>(half,uniform,wide,live,modulated,stream);
  else dispatch<Electric,2,Swizzled>(half,uniform,wide,live,modulated,stream);
}

}  // namespace

void launch_core(CoreHalf half, bool electric, int cpml, bool uniform,
                 bool wide_indices, bool live, bool modulated, cudaStream_t stream) {
  // Specialize the stock order so strip zero adds no device-side remap or branch.
  if (half.block_strip) {
    if (electric) dispatch_cpml<true,true>(half,cpml,uniform,wide_indices,live,modulated,stream);
    else dispatch_cpml<false,true>(half,cpml,uniform,wide_indices,live,modulated,stream);
  } else {
    if (electric) dispatch_cpml<true,false>(half,cpml,uniform,wide_indices,live,modulated,stream);
    else dispatch_cpml<false,false>(half,cpml,uniform,wide_indices,live,modulated,stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
