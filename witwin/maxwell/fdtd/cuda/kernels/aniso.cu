// Off-diagonal (full-tensor) anisotropic permittivity correction kernels.
//
// The base electric update already applies the diagonal entry of the per-edge
// inverse permittivity tensor through the standard curl coefficient. These
// kernels add the off-diagonal coupling terms
//     E_i += coeff_ij * <curlH_j>_i
// where <curlH_j>_i is the four-neighbor average of the off-axis curl(H)
// component evaluated at its own Yee edges and collocated onto the target
// edge. Periodic axes wrap both the neighbor sites and the finite-difference
// stencils; on non-periodic axes, samples whose stencil would leave the grid
// are skipped (the off-diagonal coefficients vanish near PML/absorber faces by
// construction, so the fixed 1/4 weight stays exact where it matters).
//
// The kernels are launched only when the solver's full-anisotropy feature flag
// is set, so the base field-update path keeps its memory traffic untouched.

#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

#include <algorithm>
#include <utility>

namespace {

__global__ void capture_aniso_conduction_current_kernel(
    int64_t total,
    int64_t nx,
    int64_t ny,
    int64_t nz,
    const float* __restrict__ sigma_x,
    const float* __restrict__ sigma_y,
    const float* __restrict__ sigma_z,
    const float* __restrict__ ex,
    const float* __restrict__ ey,
    const float* __restrict__ ez,
    float* __restrict__ jx,
    float* __restrict__ jy,
    float* __restrict__ jz) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  if (linear < nx) {
    jx[linear] = sigma_x[linear] * ex[linear];
  }
  if (linear < ny) {
    jy[linear] = sigma_y[linear] * ey[linear];
  }
  if (linear < nz) {
    jz[linear] = sigma_z[linear] * ez[linear];
  }
}

dim3 aniso_block3d() {
  return dim3(128, 2, 1);
}

dim3 aniso_grid3d(int64_t nx, int64_t ny, int64_t nz, dim3 block) {
  return dim3(
      static_cast<unsigned int>((nz + block.x - 1) / block.x),
      static_cast<unsigned int>((ny + block.y - 1) / block.y),
      static_cast<unsigned int>((nx + block.z - 1) / block.z));
}

// Wrap an edge-type coordinate (array length n_nodes - 1) that may be off by
// one period. Returns -1 when the sample must be skipped.
__device__ __forceinline__ int wrap_edge_coord(int coord, int n_nodes, bool periodic) {
  const int count = n_nodes - 1;
  if (coord >= 0 && coord < count) {
    return coord;
  }
  if (!periodic) {
    return -1;
  }
  return coord < 0 ? coord + count : coord - count;
}

// Map a node-type coordinate (array length n_nodes) onto its unique plane:
// on a periodic axis the last node duplicates node 0.
__device__ __forceinline__ int map_node_coord(int coord, int n_nodes, bool periodic) {
  if (periodic && coord == n_nodes - 1) {
    return 0;
  }
  return coord;
}

// Backward-difference edge pair (coord-1, coord) at a node-type coordinate.
// Returns false when the stencil leaves a non-periodic axis.
__device__ __forceinline__ bool backward_edge_pair(
    int coord,
    int n_nodes,
    bool periodic,
    int& low,
    int& high) {
  high = coord;
  low = coord - 1;
  if (high > n_nodes - 2) {
    if (!periodic) {
      return false;
    }
    high -= n_nodes - 1;
  }
  if (low < 0) {
    if (!periodic) {
      return false;
    }
    low += n_nodes - 1;
  }
  return true;
}

// curl(H)_x = dHz/dy - dHy/dz at the Ex edge (sx edge-x, sy node-y, sz node-z).
__device__ __forceinline__ bool curl_h_x_at(
    const float* __restrict__ hy,
    const float* __restrict__ hz,
    int sx,
    int sy,
    int sz,
    int nx_nodes,
    int ny_nodes,
    int nz_nodes,
    bool periodic_x,
    bool periodic_y,
    bool periodic_z,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
    float& value) {
  sx = wrap_edge_coord(sx, nx_nodes, periodic_x);
  if (sx < 0) {
    return false;
  }
  sy = map_node_coord(sy, ny_nodes, periodic_y);
  sz = map_node_coord(sz, nz_nodes, periodic_z);
  int y_low;
  int y_high;
  int z_low;
  int z_high;
  if (!backward_edge_pair(sy, ny_nodes, periodic_y, y_low, y_high) ||
      !backward_edge_pair(sz, nz_nodes, periodic_z, z_low, z_high)) {
    return false;
  }
  const float d_y = (hz[offset3d(sx, y_high, sz, ny_nodes - 1, nz_nodes)] -
                     hz[offset3d(sx, y_low, sz, ny_nodes - 1, nz_nodes)]) * inv_dy[sy];
  const float d_z = (hy[offset3d(sx, sy, z_high, ny_nodes, nz_nodes - 1)] -
                     hy[offset3d(sx, sy, z_low, ny_nodes, nz_nodes - 1)]) * inv_dz[sz];
  value = d_y - d_z;
  return true;
}

// curl(H)_y = dHx/dz - dHz/dx at the Ey edge (sx node-x, sy edge-y, sz node-z).
__device__ __forceinline__ bool curl_h_y_at(
    const float* __restrict__ hx,
    const float* __restrict__ hz,
    int sx,
    int sy,
    int sz,
    int nx_nodes,
    int ny_nodes,
    int nz_nodes,
    bool periodic_x,
    bool periodic_y,
    bool periodic_z,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dz,
    float& value) {
  sy = wrap_edge_coord(sy, ny_nodes, periodic_y);
  if (sy < 0) {
    return false;
  }
  sx = map_node_coord(sx, nx_nodes, periodic_x);
  sz = map_node_coord(sz, nz_nodes, periodic_z);
  int x_low;
  int x_high;
  int z_low;
  int z_high;
  if (!backward_edge_pair(sx, nx_nodes, periodic_x, x_low, x_high) ||
      !backward_edge_pair(sz, nz_nodes, periodic_z, z_low, z_high)) {
    return false;
  }
  const float d_z = (hx[offset3d(sx, sy, z_high, ny_nodes - 1, nz_nodes - 1)] -
                     hx[offset3d(sx, sy, z_low, ny_nodes - 1, nz_nodes - 1)]) * inv_dz[sz];
  const float d_x = (hz[offset3d(x_high, sy, sz, ny_nodes - 1, nz_nodes)] -
                     hz[offset3d(x_low, sy, sz, ny_nodes - 1, nz_nodes)]) * inv_dx[sx];
  value = d_z - d_x;
  return true;
}

// curl(H)_z = dHy/dx - dHx/dy at the Ez edge (sx node-x, sy node-y, sz edge-z).
__device__ __forceinline__ bool curl_h_z_at(
    const float* __restrict__ hx,
    const float* __restrict__ hy,
    int sx,
    int sy,
    int sz,
    int nx_nodes,
    int ny_nodes,
    int nz_nodes,
    bool periodic_x,
    bool periodic_y,
    bool periodic_z,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy,
    float& value) {
  sz = wrap_edge_coord(sz, nz_nodes, periodic_z);
  if (sz < 0) {
    return false;
  }
  sx = map_node_coord(sx, nx_nodes, periodic_x);
  sy = map_node_coord(sy, ny_nodes, periodic_y);
  int x_low;
  int x_high;
  int y_low;
  int y_high;
  if (!backward_edge_pair(sx, nx_nodes, periodic_x, x_low, x_high) ||
      !backward_edge_pair(sy, ny_nodes, periodic_y, y_low, y_high)) {
    return false;
  }
  const float d_x = (hy[offset3d(x_high, sy, sz, ny_nodes, nz_nodes - 1)] -
                     hy[offset3d(x_low, sy, sz, ny_nodes, nz_nodes - 1)]) * inv_dx[sx];
  const float d_y = (hx[offset3d(sx, y_high, sz, ny_nodes - 1, nz_nodes - 1)] -
                     hx[offset3d(sx, y_low, sz, ny_nodes - 1, nz_nodes - 1)]) * inv_dy[sy];
  value = d_x - d_y;
  return true;
}

// --- Split (per-direction) curl(H) parts for the CPML off-diagonal update ---
//
// The CPML off-diagonal correction coordinate-stretches each spatial derivative
// separately, so it needs the two directional derivatives that make up an
// off-axis curl(H) component instead of their difference. These helpers mirror
// the curl_h_*_at stencils above exactly but return each part; when the stencil
// leaves a non-periodic axis they return false so the neighbor is skipped
// identically to the raw path (its contribution to both parts is zero).

// curl(H)_x = dHz/dy - dHy/dz at the Ex edge: outputs d_dy = dHz/dy, d_dz = dHy/dz.
__device__ __forceinline__ bool curl_h_x_parts(
    const float* __restrict__ hy,
    const float* __restrict__ hz,
    int sx,
    int sy,
    int sz,
    int nx_nodes,
    int ny_nodes,
    int nz_nodes,
    bool periodic_x,
    bool periodic_y,
    bool periodic_z,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
    float& d_dy,
    float& d_dz) {
  sx = wrap_edge_coord(sx, nx_nodes, periodic_x);
  if (sx < 0) {
    return false;
  }
  sy = map_node_coord(sy, ny_nodes, periodic_y);
  sz = map_node_coord(sz, nz_nodes, periodic_z);
  int y_low;
  int y_high;
  int z_low;
  int z_high;
  if (!backward_edge_pair(sy, ny_nodes, periodic_y, y_low, y_high) ||
      !backward_edge_pair(sz, nz_nodes, periodic_z, z_low, z_high)) {
    return false;
  }
  d_dy = (hz[offset3d(sx, y_high, sz, ny_nodes - 1, nz_nodes)] -
          hz[offset3d(sx, y_low, sz, ny_nodes - 1, nz_nodes)]) * inv_dy[sy];
  d_dz = (hy[offset3d(sx, sy, z_high, ny_nodes, nz_nodes - 1)] -
          hy[offset3d(sx, sy, z_low, ny_nodes, nz_nodes - 1)]) * inv_dz[sz];
  return true;
}

// curl(H)_y = dHx/dz - dHz/dx at the Ey edge: outputs d_dz = dHx/dz, d_dx = dHz/dx.
__device__ __forceinline__ bool curl_h_y_parts(
    const float* __restrict__ hx,
    const float* __restrict__ hz,
    int sx,
    int sy,
    int sz,
    int nx_nodes,
    int ny_nodes,
    int nz_nodes,
    bool periodic_x,
    bool periodic_y,
    bool periodic_z,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dz,
    float& d_dz,
    float& d_dx) {
  sy = wrap_edge_coord(sy, ny_nodes, periodic_y);
  if (sy < 0) {
    return false;
  }
  sx = map_node_coord(sx, nx_nodes, periodic_x);
  sz = map_node_coord(sz, nz_nodes, periodic_z);
  int x_low;
  int x_high;
  int z_low;
  int z_high;
  if (!backward_edge_pair(sx, nx_nodes, periodic_x, x_low, x_high) ||
      !backward_edge_pair(sz, nz_nodes, periodic_z, z_low, z_high)) {
    return false;
  }
  d_dz = (hx[offset3d(sx, sy, z_high, ny_nodes - 1, nz_nodes - 1)] -
          hx[offset3d(sx, sy, z_low, ny_nodes - 1, nz_nodes - 1)]) * inv_dz[sz];
  d_dx = (hz[offset3d(x_high, sy, sz, ny_nodes - 1, nz_nodes)] -
          hz[offset3d(x_low, sy, sz, ny_nodes - 1, nz_nodes)]) * inv_dx[sx];
  return true;
}

// curl(H)_z = dHy/dx - dHx/dy at the Ez edge: outputs d_dx = dHy/dx, d_dy = dHx/dy.
__device__ __forceinline__ bool curl_h_z_parts(
    const float* __restrict__ hx,
    const float* __restrict__ hy,
    int sx,
    int sy,
    int sz,
    int nx_nodes,
    int ny_nodes,
    int nz_nodes,
    bool periodic_x,
    bool periodic_y,
    bool periodic_z,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy,
    float& d_dx,
    float& d_dy) {
  sz = wrap_edge_coord(sz, nz_nodes, periodic_z);
  if (sz < 0) {
    return false;
  }
  sx = map_node_coord(sx, nx_nodes, periodic_x);
  sy = map_node_coord(sy, ny_nodes, periodic_y);
  int x_low;
  int x_high;
  int y_low;
  int y_high;
  if (!backward_edge_pair(sx, nx_nodes, periodic_x, x_low, x_high) ||
      !backward_edge_pair(sy, ny_nodes, periodic_y, y_low, y_high)) {
    return false;
  }
  d_dx = (hy[offset3d(x_high, sy, sz, ny_nodes, nz_nodes - 1)] -
          hy[offset3d(x_low, sy, sz, ny_nodes, nz_nodes - 1)]) * inv_dx[sx];
  d_dy = (hx[offset3d(sx, y_high, sz, ny_nodes - 1, nz_nodes - 1)] -
          hx[offset3d(sx, y_low, sz, ny_nodes - 1, nz_nodes - 1)]) * inv_dy[sy];
  return true;
}

// One CPML coordinate-stretch of a collocated off-diagonal driver term. The
// psi accumulator lives at the target E edge (single update per step, no race);
// b/c/inv_kappa are the axis profiles sampled at that edge's index for the axis.
// Outside the absorber (c == 0, b == 1, inv_kappa == 1) this returns the driver
// unchanged and psi stays zero, so the correction reduces to the raw update.
__device__ __forceinline__ float aniso_cpml_stretch(
    float driver,
    float* __restrict__ psi,
    long long linear,
    float b,
    float c,
    float inv_kappa) {
  const float psi_new = b * psi[linear] + c * driver;
  psi[linear] = psi_new;
  return driver * inv_kappa + psi_new;
}

// Ex(i, j, k) lives at ((i+1/2), j, k) on an (nx_nodes, ny_nodes, nz_nodes)
// node grid; field shape is (nx_nodes-1, ny_nodes, nz_nodes).
__global__ void update_electric_ex_full_aniso_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hy,
    const float* __restrict__ hz,
    const float* __restrict__ coeff_y,
    const float* __restrict__ coeff_z,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
    int periodic_x,
    int periodic_y,
    int periodic_z,
    float* __restrict__ ex) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const float c_y = coeff_y[linear];
  const float c_z = coeff_z[linear];
  if (c_y == 0.0f && c_z == 0.0f) {
    return;
  }
  const int nx_nodes = static_cast<int>(nx) + 1;  // ex shape (Nx-1, Ny, Nz)
  const int ny_nodes = static_cast<int>(ny);
  const int nz_nodes = static_cast<int>(nz);
  const bool per_x = periodic_x != 0;
  const bool per_y = periodic_y != 0;
  const bool per_z = periodic_z != 0;
  const int ii = static_cast<int>(i);
  const int jj = static_cast<int>(j);
  const int kk = static_cast<int>(k);

  // <curlH_y> over the four surrounding Ey edges (sx in {i, i+1}, sy in {j-1, j}).
  float acc_y = 0.0f;
  if (c_y != 0.0f) {
    for (int sx = ii; sx <= ii + 1; ++sx) {
      for (int sy = jj - 1; sy <= jj; ++sy) {
        float sample;
        if (curl_h_y_at(hx, hz, sx, sy, kk, nx_nodes, ny_nodes, nz_nodes,
                        per_x, per_y, per_z, inv_dx, inv_dz, sample)) {
          acc_y += sample;
        }
      }
    }
  }

  // <curlH_z> over the four surrounding Ez edges (sx in {i, i+1}, sz in {k-1, k}).
  float acc_z = 0.0f;
  if (c_z != 0.0f) {
    for (int sx = ii; sx <= ii + 1; ++sx) {
      for (int sz = kk - 1; sz <= kk; ++sz) {
        float sample;
        if (curl_h_z_at(hx, hy, sx, jj, sz, nx_nodes, ny_nodes, nz_nodes,
                        per_x, per_y, per_z, inv_dx, inv_dy, sample)) {
          acc_z += sample;
        }
      }
    }
  }

  ex[linear] += 0.25f * (c_y * acc_y + c_z * acc_z);
}

// Ey(i, j, k) lives at (i, (j+1/2), k); field shape is (Nx, Ny-1, Nz).
__global__ void update_electric_ey_full_aniso_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hy,
    const float* __restrict__ hz,
    const float* __restrict__ coeff_x,
    const float* __restrict__ coeff_z,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
    int periodic_x,
    int periodic_y,
    int periodic_z,
    float* __restrict__ ey) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const float c_x = coeff_x[linear];
  const float c_z = coeff_z[linear];
  if (c_x == 0.0f && c_z == 0.0f) {
    return;
  }
  const int nx_nodes = static_cast<int>(nx);  // ey shape (Nx, Ny-1, Nz)
  const int ny_nodes = static_cast<int>(ny) + 1;
  const int nz_nodes = static_cast<int>(nz);
  const bool per_x = periodic_x != 0;
  const bool per_y = periodic_y != 0;
  const bool per_z = periodic_z != 0;
  const int ii = static_cast<int>(i);
  const int jj = static_cast<int>(j);
  const int kk = static_cast<int>(k);

  // <curlH_x> over the four surrounding Ex edges (sx in {i-1, i}, sy in {j, j+1}).
  float acc_x = 0.0f;
  if (c_x != 0.0f) {
    for (int sx = ii - 1; sx <= ii; ++sx) {
      for (int sy = jj; sy <= jj + 1; ++sy) {
        float sample;
        if (curl_h_x_at(hy, hz, sx, sy, kk, nx_nodes, ny_nodes, nz_nodes,
                        per_x, per_y, per_z, inv_dy, inv_dz, sample)) {
          acc_x += sample;
        }
      }
    }
  }

  // <curlH_z> over the four surrounding Ez edges (sy in {j, j+1}, sz in {k-1, k}).
  float acc_z = 0.0f;
  if (c_z != 0.0f) {
    for (int sy = jj; sy <= jj + 1; ++sy) {
      for (int sz = kk - 1; sz <= kk; ++sz) {
        float sample;
        if (curl_h_z_at(hx, hy, ii, sy, sz, nx_nodes, ny_nodes, nz_nodes,
                        per_x, per_y, per_z, inv_dx, inv_dy, sample)) {
          acc_z += sample;
        }
      }
    }
  }

  ey[linear] += 0.25f * (c_x * acc_x + c_z * acc_z);
}

// Ez(i, j, k) lives at (i, j, (k+1/2)); field shape is (Nx, Ny, Nz-1).
__global__ void update_electric_ez_full_aniso_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hy,
    const float* __restrict__ hz,
    const float* __restrict__ coeff_x,
    const float* __restrict__ coeff_y,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
    int periodic_x,
    int periodic_y,
    int periodic_z,
    float* __restrict__ ez) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const float c_x = coeff_x[linear];
  const float c_y = coeff_y[linear];
  if (c_x == 0.0f && c_y == 0.0f) {
    return;
  }
  const int nx_nodes = static_cast<int>(nx);  // ez shape (Nx, Ny, Nz-1)
  const int ny_nodes = static_cast<int>(ny);
  const int nz_nodes = static_cast<int>(nz) + 1;
  const bool per_x = periodic_x != 0;
  const bool per_y = periodic_y != 0;
  const bool per_z = periodic_z != 0;
  const int ii = static_cast<int>(i);
  const int jj = static_cast<int>(j);
  const int kk = static_cast<int>(k);

  // <curlH_x> over the four surrounding Ex edges (sx in {i-1, i}, sz in {k, k+1}).
  float acc_x = 0.0f;
  if (c_x != 0.0f) {
    for (int sx = ii - 1; sx <= ii; ++sx) {
      for (int sz = kk; sz <= kk + 1; ++sz) {
        float sample;
        if (curl_h_x_at(hy, hz, sx, jj, sz, nx_nodes, ny_nodes, nz_nodes,
                        per_x, per_y, per_z, inv_dy, inv_dz, sample)) {
          acc_x += sample;
        }
      }
    }
  }

  // <curlH_y> over the four surrounding Ey edges (sy in {j-1, j}, sz in {k, k+1}).
  float acc_y = 0.0f;
  if (c_y != 0.0f) {
    for (int sy = jj - 1; sy <= jj; ++sy) {
      for (int sz = kk; sz <= kk + 1; ++sz) {
        float sample;
        if (curl_h_y_at(hx, hz, ii, sy, sz, nx_nodes, ny_nodes, nz_nodes,
                        per_x, per_y, per_z, inv_dx, inv_dz, sample)) {
          acc_y += sample;
        }
      }
    }
  }

  ez[linear] += 0.25f * (c_x * acc_x + c_y * acc_y);
}

// --- CPML-consistent off-diagonal correction (structure overlapping the PML) -
//
// The raw kernels above add the collocated off-axis curl(H) with plain finite
// differences, which is only exact where the CPML coordinate stretch is
// inactive. When an anisotropic structure reaches into the absorber the
// off-diagonal coupling must be stretched exactly like the diagonal curl. These
// kernels split each off-axis curl(H) into its two directional derivatives,
// collocate each onto the target E edge, and apply the CPML stretch per
// direction with a psi accumulator owned by that edge. The transverse
// directions use the E-field (node) profiles at the edge's node index; the
// edge's own direction sits on a half point, so it uses the H-field (half)
// profile at the edge index. The three per-direction psi buffers are zero
// outside the absorber, so this reduces exactly to the raw correction there.

__global__ void update_electric_ex_full_aniso_cpml_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hy,
    const float* __restrict__ hz,
    const float* __restrict__ coeff_y,
    const float* __restrict__ coeff_z,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ cpml_b_x,
    const float* __restrict__ cpml_c_x,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ cpml_b_y,
    const float* __restrict__ cpml_c_y,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ cpml_b_z,
    const float* __restrict__ cpml_c_z,
    int periodic_x,
    int periodic_y,
    int periodic_z,
    float* __restrict__ psi_x,
    float* __restrict__ psi_y,
    float* __restrict__ psi_z,
    float* __restrict__ ex) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const float c_y = coeff_y[linear];
  const float c_z = coeff_z[linear];
  if (c_y == 0.0f && c_z == 0.0f) {
    return;
  }
  const int nx_nodes = static_cast<int>(nx) + 1;  // ex shape (Nx-1, Ny, Nz)
  const int ny_nodes = static_cast<int>(ny);
  const int nz_nodes = static_cast<int>(nz);
  const bool per_x = periodic_x != 0;
  const bool per_y = periodic_y != 0;
  const bool per_z = periodic_z != 0;
  const int ii = static_cast<int>(i);
  const int jj = static_cast<int>(j);
  const int kk = static_cast<int>(k);

  // curl(H)_y over the four surrounding Ey edges: split into dHx/dz and dHz/dx.
  float acc_yz = 0.0f;  // dHx/dz
  float acc_yx = 0.0f;  // dHz/dx
  if (c_y != 0.0f) {
    for (int sx = ii; sx <= ii + 1; ++sx) {
      for (int sy = jj - 1; sy <= jj; ++sy) {
        float d_dz;
        float d_dx;
        if (curl_h_y_parts(hx, hz, sx, sy, kk, nx_nodes, ny_nodes, nz_nodes,
                           per_x, per_y, per_z, inv_dx, inv_dz, d_dz, d_dx)) {
          acc_yz += d_dz;
          acc_yx += d_dx;
        }
      }
    }
  }

  // curl(H)_z over the four surrounding Ez edges: split into dHy/dx and dHx/dy.
  float acc_zx = 0.0f;  // dHy/dx
  float acc_zy = 0.0f;  // dHx/dy
  if (c_z != 0.0f) {
    for (int sx = ii; sx <= ii + 1; ++sx) {
      for (int sz = kk - 1; sz <= kk; ++sz) {
        float d_dx;
        float d_dy;
        if (curl_h_z_parts(hx, hy, sx, jj, sz, nx_nodes, ny_nodes, nz_nodes,
                           per_x, per_y, per_z, inv_dx, inv_dy, d_dx, d_dy)) {
          acc_zx += d_dx;
          acc_zy += d_dy;
        }
      }
    }
  }

  const float driver_x = 0.25f * (-c_y * acc_yx + c_z * acc_zx);
  const float driver_y = 0.25f * (-c_z * acc_zy);
  const float driver_z = 0.25f * (c_y * acc_yz);

  const float out_x = aniso_cpml_stretch(driver_x, psi_x, linear, cpml_b_x[i], cpml_c_x[i], inv_kappa_x[i]);
  const float out_y = aniso_cpml_stretch(driver_y, psi_y, linear, cpml_b_y[j], cpml_c_y[j], inv_kappa_y[j]);
  const float out_z = aniso_cpml_stretch(driver_z, psi_z, linear, cpml_b_z[k], cpml_c_z[k], inv_kappa_z[k]);
  ex[linear] += out_x + out_y + out_z;
}

__global__ void update_electric_ey_full_aniso_cpml_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hy,
    const float* __restrict__ hz,
    const float* __restrict__ coeff_x,
    const float* __restrict__ coeff_z,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ cpml_b_x,
    const float* __restrict__ cpml_c_x,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ cpml_b_y,
    const float* __restrict__ cpml_c_y,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ cpml_b_z,
    const float* __restrict__ cpml_c_z,
    int periodic_x,
    int periodic_y,
    int periodic_z,
    float* __restrict__ psi_x,
    float* __restrict__ psi_y,
    float* __restrict__ psi_z,
    float* __restrict__ ey) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const float c_x = coeff_x[linear];
  const float c_z = coeff_z[linear];
  if (c_x == 0.0f && c_z == 0.0f) {
    return;
  }
  const int nx_nodes = static_cast<int>(nx);  // ey shape (Nx, Ny-1, Nz)
  const int ny_nodes = static_cast<int>(ny) + 1;
  const int nz_nodes = static_cast<int>(nz);
  const bool per_x = periodic_x != 0;
  const bool per_y = periodic_y != 0;
  const bool per_z = periodic_z != 0;
  const int ii = static_cast<int>(i);
  const int jj = static_cast<int>(j);
  const int kk = static_cast<int>(k);

  // curl(H)_x over the four surrounding Ex edges: split into dHz/dy and dHy/dz.
  float acc_xy = 0.0f;  // dHz/dy
  float acc_xz = 0.0f;  // dHy/dz
  if (c_x != 0.0f) {
    for (int sx = ii - 1; sx <= ii; ++sx) {
      for (int sy = jj; sy <= jj + 1; ++sy) {
        float d_dy;
        float d_dz;
        if (curl_h_x_parts(hy, hz, sx, sy, kk, nx_nodes, ny_nodes, nz_nodes,
                           per_x, per_y, per_z, inv_dy, inv_dz, d_dy, d_dz)) {
          acc_xy += d_dy;
          acc_xz += d_dz;
        }
      }
    }
  }

  // curl(H)_z over the four surrounding Ez edges: split into dHy/dx and dHx/dy.
  float acc_zx = 0.0f;  // dHy/dx
  float acc_zy = 0.0f;  // dHx/dy
  if (c_z != 0.0f) {
    for (int sy = jj; sy <= jj + 1; ++sy) {
      for (int sz = kk - 1; sz <= kk; ++sz) {
        float d_dx;
        float d_dy;
        if (curl_h_z_parts(hx, hy, ii, sy, sz, nx_nodes, ny_nodes, nz_nodes,
                           per_x, per_y, per_z, inv_dx, inv_dy, d_dx, d_dy)) {
          acc_zx += d_dx;
          acc_zy += d_dy;
        }
      }
    }
  }

  const float driver_x = 0.25f * (c_z * acc_zx);
  const float driver_y = 0.25f * (c_x * acc_xy - c_z * acc_zy);
  const float driver_z = 0.25f * (-c_x * acc_xz);

  const float out_x = aniso_cpml_stretch(driver_x, psi_x, linear, cpml_b_x[i], cpml_c_x[i], inv_kappa_x[i]);
  const float out_y = aniso_cpml_stretch(driver_y, psi_y, linear, cpml_b_y[j], cpml_c_y[j], inv_kappa_y[j]);
  const float out_z = aniso_cpml_stretch(driver_z, psi_z, linear, cpml_b_z[k], cpml_c_z[k], inv_kappa_z[k]);
  ey[linear] += out_x + out_y + out_z;
}

__global__ void update_electric_ez_full_aniso_cpml_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ hx,
    const float* __restrict__ hy,
    const float* __restrict__ hz,
    const float* __restrict__ coeff_x,
    const float* __restrict__ coeff_y,
    const float* __restrict__ inv_dx,
    const float* __restrict__ inv_dy,
    const float* __restrict__ inv_dz,
    const float* __restrict__ inv_kappa_x,
    const float* __restrict__ cpml_b_x,
    const float* __restrict__ cpml_c_x,
    const float* __restrict__ inv_kappa_y,
    const float* __restrict__ cpml_b_y,
    const float* __restrict__ cpml_c_y,
    const float* __restrict__ inv_kappa_z,
    const float* __restrict__ cpml_b_z,
    const float* __restrict__ cpml_c_z,
    int periodic_x,
    int periodic_y,
    int periodic_z,
    float* __restrict__ psi_x,
    float* __restrict__ psi_y,
    float* __restrict__ psi_z,
    float* __restrict__ ez) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const float c_x = coeff_x[linear];
  const float c_y = coeff_y[linear];
  if (c_x == 0.0f && c_y == 0.0f) {
    return;
  }
  const int nx_nodes = static_cast<int>(nx);  // ez shape (Nx, Ny, Nz-1)
  const int ny_nodes = static_cast<int>(ny);
  const int nz_nodes = static_cast<int>(nz) + 1;
  const bool per_x = periodic_x != 0;
  const bool per_y = periodic_y != 0;
  const bool per_z = periodic_z != 0;
  const int ii = static_cast<int>(i);
  const int jj = static_cast<int>(j);
  const int kk = static_cast<int>(k);

  // curl(H)_x over the four surrounding Ex edges: split into dHz/dy and dHy/dz.
  float acc_xy = 0.0f;  // dHz/dy
  float acc_xz = 0.0f;  // dHy/dz
  if (c_x != 0.0f) {
    for (int sx = ii - 1; sx <= ii; ++sx) {
      for (int sz = kk; sz <= kk + 1; ++sz) {
        float d_dy;
        float d_dz;
        if (curl_h_x_parts(hy, hz, sx, jj, sz, nx_nodes, ny_nodes, nz_nodes,
                           per_x, per_y, per_z, inv_dy, inv_dz, d_dy, d_dz)) {
          acc_xy += d_dy;
          acc_xz += d_dz;
        }
      }
    }
  }

  // curl(H)_y over the four surrounding Ey edges: split into dHx/dz and dHz/dx.
  float acc_yz = 0.0f;  // dHx/dz
  float acc_yx = 0.0f;  // dHz/dx
  if (c_y != 0.0f) {
    for (int sy = jj - 1; sy <= jj; ++sy) {
      for (int sz = kk; sz <= kk + 1; ++sz) {
        float d_dz;
        float d_dx;
        if (curl_h_y_parts(hx, hz, ii, sy, sz, nx_nodes, ny_nodes, nz_nodes,
                           per_x, per_y, per_z, inv_dx, inv_dz, d_dz, d_dx)) {
          acc_yz += d_dz;
          acc_yx += d_dx;
        }
      }
    }
  }

  const float driver_x = 0.25f * (-c_y * acc_yx);
  const float driver_y = 0.25f * (c_x * acc_xy);
  const float driver_z = 0.25f * (-c_x * acc_xz + c_y * acc_yz);

  const float out_x = aniso_cpml_stretch(driver_x, psi_x, linear, cpml_b_x[i], cpml_c_x[i], inv_kappa_x[i]);
  const float out_y = aniso_cpml_stretch(driver_y, psi_y, linear, cpml_b_y[j], cpml_c_y[j], inv_kappa_y[j]);
  const float out_z = aniso_cpml_stretch(driver_z, psi_z, linear, cpml_b_z[k], cpml_c_z[k], inv_kappa_z[k]);
  ez[linear] += out_x + out_y + out_z;
}

// --- Off-diagonal ADE polarization-current subtraction ---------------------
//
// For a full (off-diagonal) anisotropic permittivity the Ampere update is
//     E += dt * eps_inf^-1 . (curl H - J_p),
// so the same per-edge inverse permittivity tensor that couples curl(H) also
// couples the polarization current J_p across components. These kernels apply
// the off-diagonal part of that tensor to the accumulated per-component
// polarization current, mirroring the curl(H) collocation stencil above but
// reading the stored current field directly and subtracting:
//     E_i -= 0.25 * (coeff_ij * <J_j>_i + coeff_ik * <J_k>_i).
// The coefficients coeff_ij == dt * inv_ij / eps0 are exactly the ones used for
// the curl(H) off-diagonal correction, so magnitude and sign stay consistent.

// Read an Ex-edge field value at (sx edge-x, sy node-y, sz node-z).
__device__ __forceinline__ bool read_ex_edge_value(
    const float* __restrict__ field,
    int sx,
    int sy,
    int sz,
    int nx_nodes,
    int ny_nodes,
    int nz_nodes,
    bool periodic_x,
    bool periodic_y,
    bool periodic_z,
    float& value) {
  sx = wrap_edge_coord(sx, nx_nodes, periodic_x);
  if (sx < 0) {
    return false;
  }
  sy = map_node_coord(sy, ny_nodes, periodic_y);
  sz = map_node_coord(sz, nz_nodes, periodic_z);
  value = field[offset3d(sx, sy, sz, ny_nodes, nz_nodes)];
  return true;
}

// Read an Ey-edge field value at (sx node-x, sy edge-y, sz node-z).
__device__ __forceinline__ bool read_ey_edge_value(
    const float* __restrict__ field,
    int sx,
    int sy,
    int sz,
    int nx_nodes,
    int ny_nodes,
    int nz_nodes,
    bool periodic_x,
    bool periodic_y,
    bool periodic_z,
    float& value) {
  sy = wrap_edge_coord(sy, ny_nodes, periodic_y);
  if (sy < 0) {
    return false;
  }
  sx = map_node_coord(sx, nx_nodes, periodic_x);
  sz = map_node_coord(sz, nz_nodes, periodic_z);
  value = field[offset3d(sx, sy, sz, ny_nodes - 1, nz_nodes)];
  return true;
}

// Read an Ez-edge field value at (sx node-x, sy node-y, sz edge-z).
__device__ __forceinline__ bool read_ez_edge_value(
    const float* __restrict__ field,
    int sx,
    int sy,
    int sz,
    int nx_nodes,
    int ny_nodes,
    int nz_nodes,
    bool periodic_x,
    bool periodic_y,
    bool periodic_z,
    float& value) {
  sz = wrap_edge_coord(sz, nz_nodes, periodic_z);
  if (sz < 0) {
    return false;
  }
  sx = map_node_coord(sx, nx_nodes, periodic_x);
  sy = map_node_coord(sy, ny_nodes, periodic_y);
  value = field[offset3d(sx, sy, sz, ny_nodes, nz_nodes - 1)];
  return true;
}

__global__ void apply_aniso_offdiag_current_ex_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ jy,
    const float* __restrict__ jz,
    const float* __restrict__ coeff_y,
    const float* __restrict__ coeff_z,
    int periodic_x,
    int periodic_y,
    int periodic_z,
    float* __restrict__ ex) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const float c_y = coeff_y[linear];
  const float c_z = coeff_z[linear];
  if (c_y == 0.0f && c_z == 0.0f) {
    return;
  }
  const int nx_nodes = static_cast<int>(nx) + 1;  // ex shape (Nx-1, Ny, Nz)
  const int ny_nodes = static_cast<int>(ny);
  const int nz_nodes = static_cast<int>(nz);
  const bool per_x = periodic_x != 0;
  const bool per_y = periodic_y != 0;
  const bool per_z = periodic_z != 0;
  const int ii = static_cast<int>(i);
  const int jj = static_cast<int>(j);
  const int kk = static_cast<int>(k);

  float acc_y = 0.0f;
  if (c_y != 0.0f) {
    for (int sx = ii; sx <= ii + 1; ++sx) {
      for (int sy = jj - 1; sy <= jj; ++sy) {
        float sample;
        if (read_ey_edge_value(jy, sx, sy, kk, nx_nodes, ny_nodes, nz_nodes,
                               per_x, per_y, per_z, sample)) {
          acc_y += sample;
        }
      }
    }
  }

  float acc_z = 0.0f;
  if (c_z != 0.0f) {
    for (int sx = ii; sx <= ii + 1; ++sx) {
      for (int sz = kk - 1; sz <= kk; ++sz) {
        float sample;
        if (read_ez_edge_value(jz, sx, jj, sz, nx_nodes, ny_nodes, nz_nodes,
                               per_x, per_y, per_z, sample)) {
          acc_z += sample;
        }
      }
    }
  }

  ex[linear] -= 0.25f * (c_y * acc_y + c_z * acc_z);
}

__global__ void apply_aniso_offdiag_current_ey_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ jx,
    const float* __restrict__ jz,
    const float* __restrict__ coeff_x,
    const float* __restrict__ coeff_z,
    int periodic_x,
    int periodic_y,
    int periodic_z,
    float* __restrict__ ey) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const float c_x = coeff_x[linear];
  const float c_z = coeff_z[linear];
  if (c_x == 0.0f && c_z == 0.0f) {
    return;
  }
  const int nx_nodes = static_cast<int>(nx);  // ey shape (Nx, Ny-1, Nz)
  const int ny_nodes = static_cast<int>(ny) + 1;
  const int nz_nodes = static_cast<int>(nz);
  const bool per_x = periodic_x != 0;
  const bool per_y = periodic_y != 0;
  const bool per_z = periodic_z != 0;
  const int ii = static_cast<int>(i);
  const int jj = static_cast<int>(j);
  const int kk = static_cast<int>(k);

  float acc_x = 0.0f;
  if (c_x != 0.0f) {
    for (int sx = ii - 1; sx <= ii; ++sx) {
      for (int sy = jj; sy <= jj + 1; ++sy) {
        float sample;
        if (read_ex_edge_value(jx, sx, sy, kk, nx_nodes, ny_nodes, nz_nodes,
                               per_x, per_y, per_z, sample)) {
          acc_x += sample;
        }
      }
    }
  }

  float acc_z = 0.0f;
  if (c_z != 0.0f) {
    for (int sy = jj; sy <= jj + 1; ++sy) {
      for (int sz = kk - 1; sz <= kk; ++sz) {
        float sample;
        if (read_ez_edge_value(jz, ii, sy, sz, nx_nodes, ny_nodes, nz_nodes,
                               per_x, per_y, per_z, sample)) {
          acc_z += sample;
        }
      }
    }
  }

  ey[linear] -= 0.25f * (c_x * acc_x + c_z * acc_z);
}

__global__ void apply_aniso_offdiag_current_ez_kernel(
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    const float* __restrict__ jx,
    const float* __restrict__ jy,
    const float* __restrict__ coeff_x,
    const float* __restrict__ coeff_y,
    int periodic_x,
    int periodic_y,
    int periodic_z,
    float* __restrict__ ez) {
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const long long linear = offset3d(i, j, k, ny, nz);
  const float c_x = coeff_x[linear];
  const float c_y = coeff_y[linear];
  if (c_x == 0.0f && c_y == 0.0f) {
    return;
  }
  const int nx_nodes = static_cast<int>(nx);  // ez shape (Nx, Ny, Nz-1)
  const int ny_nodes = static_cast<int>(ny);
  const int nz_nodes = static_cast<int>(nz) + 1;
  const bool per_x = periodic_x != 0;
  const bool per_y = periodic_y != 0;
  const bool per_z = periodic_z != 0;
  const int ii = static_cast<int>(i);
  const int jj = static_cast<int>(j);
  const int kk = static_cast<int>(k);

  float acc_x = 0.0f;
  if (c_x != 0.0f) {
    for (int sx = ii - 1; sx <= ii; ++sx) {
      for (int sz = kk; sz <= kk + 1; ++sz) {
        float sample;
        if (read_ex_edge_value(jx, sx, jj, sz, nx_nodes, ny_nodes, nz_nodes,
                               per_x, per_y, per_z, sample)) {
          acc_x += sample;
        }
      }
    }
  }

  float acc_y = 0.0f;
  if (c_y != 0.0f) {
    for (int sy = jj - 1; sy <= jj; ++sy) {
      for (int sz = kk; sz <= kk + 1; ++sz) {
        float sample;
        if (read_ey_edge_value(jy, ii, sy, sz, nx_nodes, ny_nodes, nz_nodes,
                               per_x, per_y, per_z, sample)) {
          acc_y += sample;
        }
      }
    }
  }

  ez[linear] -= 0.25f * (c_x * acc_x + c_y * acc_y);
}

void check_full_aniso_inputs(
    const torch::stable::Tensor& field,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& coeff_a,
    const torch::stable::Tensor& coeff_b,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t nx_nodes,
    int64_t ny_nodes,
    int64_t nz_nodes,
    const char* name) {
  for (const torch::stable::Tensor* tensor :
       {&field, &hx, &hy, &hz, &coeff_a, &coeff_b, &inv_dx, &inv_dy, &inv_dz}) {
    check_float32_tensor(*tensor, name);
    check_contiguous_tensor(*tensor, name);
    check_same_cuda_device(field, *tensor, name);
  }
  STD_TORCH_CHECK(field.dim() == 3, name, " must be rank 3");
  STD_TORCH_CHECK(coeff_a.sizes().equals(field.sizes()), name, " coeff must match field shape");
  STD_TORCH_CHECK(coeff_b.sizes().equals(field.sizes()), name, " coeff must match field shape");
  STD_TORCH_CHECK(
      hx.dim() == 3 && hx.size(0) == nx_nodes && hx.size(1) == ny_nodes - 1 && hx.size(2) == nz_nodes - 1,
      name, " hx has an incompatible Yee-grid shape");
  STD_TORCH_CHECK(
      hy.dim() == 3 && hy.size(0) == nx_nodes - 1 && hy.size(1) == ny_nodes && hy.size(2) == nz_nodes - 1,
      name, " hy has an incompatible Yee-grid shape");
  STD_TORCH_CHECK(
      hz.dim() == 3 && hz.size(0) == nx_nodes - 1 && hz.size(1) == ny_nodes - 1 && hz.size(2) == nz_nodes,
      name, " hz has an incompatible Yee-grid shape");
  STD_TORCH_CHECK(inv_dx.dim() == 1 && inv_dx.size(0) == nx_nodes, name, " inv_dx length must be the node count");
  STD_TORCH_CHECK(inv_dy.dim() == 1 && inv_dy.size(0) == ny_nodes, name, " inv_dy length must be the node count");
  STD_TORCH_CHECK(inv_dz.dim() == 1 && inv_dz.size(0) == nz_nodes, name, " inv_dz length must be the node count");
}

void check_aniso_cpml_axis(
    const torch::stable::Tensor& field,
    const torch::stable::Tensor& inv_kappa,
    const torch::stable::Tensor& cpml_b,
    const torch::stable::Tensor& cpml_c,
    int axis,
    const char* name) {
  for (const torch::stable::Tensor* tensor : {&inv_kappa, &cpml_b, &cpml_c}) {
    check_float32_tensor(*tensor, name);
    check_contiguous_tensor(*tensor, name);
    check_same_cuda_device(field, *tensor, name);
    STD_TORCH_CHECK(
        tensor->dim() == 1 && tensor->size(0) == field.size(axis),
        name, " CPML profile length must match the field extent along its axis");
  }
}

void check_aniso_cpml_psi(
    const torch::stable::Tensor& field,
    const torch::stable::Tensor& psi_x,
    const torch::stable::Tensor& psi_y,
    const torch::stable::Tensor& psi_z,
    const char* name) {
  for (const torch::stable::Tensor* tensor : {&psi_x, &psi_y, &psi_z}) {
    check_float32_tensor(*tensor, name);
    check_contiguous_tensor(*tensor, name);
    check_same_cuda_device(field, *tensor, name);
    STD_TORCH_CHECK(tensor->sizes().equals(field.sizes()), name, " psi buffer must match field shape");
  }
}

void check_current_field(
    const torch::stable::Tensor& field,
    const torch::stable::Tensor& reference,
    int64_t s0,
    int64_t s1,
    int64_t s2,
    const char* name) {
  check_float32_tensor(field, name);
  check_contiguous_tensor(field, name);
  check_same_cuda_device(reference, field, name);
  STD_TORCH_CHECK(
      field.dim() == 3 && field.size(0) == s0 && field.size(1) == s1 && field.size(2) == s2,
      name, " has an incompatible Yee-grid shape");
}

void check_aniso_offdiag_current_inputs(
    const torch::stable::Tensor& field,
    const torch::stable::Tensor& coeff_a,
    const torch::stable::Tensor& coeff_b,
    const char* name) {
  for (const torch::stable::Tensor* tensor : {&field, &coeff_a, &coeff_b}) {
    check_float32_tensor(*tensor, name);
    check_contiguous_tensor(*tensor, name);
    check_same_cuda_device(field, *tensor, name);
  }
  STD_TORCH_CHECK(field.dim() == 3, name, " must be rank 3");
  STD_TORCH_CHECK(coeff_a.sizes().equals(field.sizes()), name, " coeff must match field shape");
  STD_TORCH_CHECK(coeff_b.sizes().equals(field.sizes()), name, " coeff must match field shape");
}

}  // namespace

void capture_aniso_conduction_current_cuda(
    const torch::stable::Tensor& sigma_x,
    const torch::stable::Tensor& sigma_y,
    const torch::stable::Tensor& sigma_z,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    torch::stable::Tensor jx,
    torch::stable::Tensor jy,
    torch::stable::Tensor jz) {
  const std::pair<const torch::stable::Tensor*, const char*> inputs[] = {
      {&sigma_x, "sigma_x"}, {&sigma_y, "sigma_y"}, {&sigma_z, "sigma_z"},
      {&ex, "ex"}, {&ey, "ey"}, {&ez, "ez"},
      {&jx, "jx"}, {&jy, "jy"}, {&jz, "jz"}};
  for (const auto& entry : inputs) {
    check_float32_tensor(*entry.first, entry.second);
    check_contiguous_tensor(*entry.first, entry.second);
    check_same_cuda_device(ex, *entry.first, entry.second);
  }
  STD_TORCH_CHECK(sigma_x.sizes().equals(ex.sizes()) && jx.sizes().equals(ex.sizes()),
                  "sigma_x and jx must match ex");
  STD_TORCH_CHECK(sigma_y.sizes().equals(ey.sizes()) && jy.sizes().equals(ey.sizes()),
                  "sigma_y and jy must match ey");
  STD_TORCH_CHECK(sigma_z.sizes().equals(ez.sizes()) && jz.sizes().equals(ez.sizes()),
                  "sigma_z and jz must match ez");
  const int64_t total = std::max(ex.numel(), std::max(ey.numel(), ez.numel()));
  if (total == 0) {
    return;
  }
  const torch::stable::accelerator::DeviceGuard guard(ex.get_device_index());
  capture_aniso_conduction_current_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total, ex.numel(), ey.numel(), ez.numel(),
      sigma_x.mutable_data_ptr<float>(), sigma_y.mutable_data_ptr<float>(), sigma_z.mutable_data_ptr<float>(),
      ex.mutable_data_ptr<float>(), ey.mutable_data_ptr<float>(), ez.mutable_data_ptr<float>(),
      jx.mutable_data_ptr<float>(), jy.mutable_data_ptr<float>(), jz.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ex_full_aniso_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& coeff_y,
    const torch::stable::Tensor& coeff_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t periodic_x,
    int64_t periodic_y,
    int64_t periodic_z) {
  const int64_t nx_nodes = ex.size(0) + 1;
  const int64_t ny_nodes = ex.size(1);
  const int64_t nz_nodes = ex.size(2);
  check_full_aniso_inputs(ex, hx, hy, hz, coeff_y, coeff_z, inv_dx, inv_dy, inv_dz,
                          nx_nodes, ny_nodes, nz_nodes, "ex");
  torch::stable::accelerator::DeviceGuard guard(ex.get_device_index());
  const dim3 block = aniso_block3d();
  update_electric_ex_full_aniso_kernel<<<
      aniso_grid3d(ex.size(0), ex.size(1), ex.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(ex.size(0)),
      static_cast<unsigned int>(ex.size(1)),
      static_cast<unsigned int>(ex.size(2)),
      hx.mutable_data_ptr<float>(),
      hy.mutable_data_ptr<float>(),
      hz.mutable_data_ptr<float>(),
      coeff_y.mutable_data_ptr<float>(),
      coeff_z.mutable_data_ptr<float>(),
      inv_dx.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      static_cast<int>(periodic_x),
      static_cast<int>(periodic_y),
      static_cast<int>(periodic_z),
      ex.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ey_full_aniso_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& coeff_x,
    const torch::stable::Tensor& coeff_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t periodic_x,
    int64_t periodic_y,
    int64_t periodic_z) {
  const int64_t nx_nodes = ey.size(0);
  const int64_t ny_nodes = ey.size(1) + 1;
  const int64_t nz_nodes = ey.size(2);
  check_full_aniso_inputs(ey, hx, hy, hz, coeff_x, coeff_z, inv_dx, inv_dy, inv_dz,
                          nx_nodes, ny_nodes, nz_nodes, "ey");
  torch::stable::accelerator::DeviceGuard guard(ey.get_device_index());
  const dim3 block = aniso_block3d();
  update_electric_ey_full_aniso_kernel<<<
      aniso_grid3d(ey.size(0), ey.size(1), ey.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(ey.size(0)),
      static_cast<unsigned int>(ey.size(1)),
      static_cast<unsigned int>(ey.size(2)),
      hx.mutable_data_ptr<float>(),
      hy.mutable_data_ptr<float>(),
      hz.mutable_data_ptr<float>(),
      coeff_x.mutable_data_ptr<float>(),
      coeff_z.mutable_data_ptr<float>(),
      inv_dx.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      static_cast<int>(periodic_x),
      static_cast<int>(periodic_y),
      static_cast<int>(periodic_z),
      ey.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ez_full_aniso_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& coeff_x,
    const torch::stable::Tensor& coeff_y,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t periodic_x,
    int64_t periodic_y,
    int64_t periodic_z) {
  const int64_t nx_nodes = ez.size(0);
  const int64_t ny_nodes = ez.size(1);
  const int64_t nz_nodes = ez.size(2) + 1;
  check_full_aniso_inputs(ez, hx, hy, hz, coeff_x, coeff_y, inv_dx, inv_dy, inv_dz,
                          nx_nodes, ny_nodes, nz_nodes, "ez");
  torch::stable::accelerator::DeviceGuard guard(ez.get_device_index());
  const dim3 block = aniso_block3d();
  update_electric_ez_full_aniso_kernel<<<
      aniso_grid3d(ez.size(0), ez.size(1), ez.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(ez.size(0)),
      static_cast<unsigned int>(ez.size(1)),
      static_cast<unsigned int>(ez.size(2)),
      hx.mutable_data_ptr<float>(),
      hy.mutable_data_ptr<float>(),
      hz.mutable_data_ptr<float>(),
      coeff_x.mutable_data_ptr<float>(),
      coeff_y.mutable_data_ptr<float>(),
      inv_dx.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      static_cast<int>(periodic_x),
      static_cast<int>(periodic_y),
      static_cast<int>(periodic_z),
      ez.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ex_full_aniso_cpml_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& coeff_y,
    const torch::stable::Tensor& coeff_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    int64_t periodic_x,
    int64_t periodic_y,
    int64_t periodic_z,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_y,
    torch::stable::Tensor psi_z) {
  const int64_t nx_nodes = ex.size(0) + 1;
  const int64_t ny_nodes = ex.size(1);
  const int64_t nz_nodes = ex.size(2);
  check_full_aniso_inputs(ex, hx, hy, hz, coeff_y, coeff_z, inv_dx, inv_dy, inv_dz,
                          nx_nodes, ny_nodes, nz_nodes, "ex_cpml");
  check_aniso_cpml_axis(ex, inv_kappa_x, b_x, c_x, 0, "ex_cpml x");
  check_aniso_cpml_axis(ex, inv_kappa_y, b_y, c_y, 1, "ex_cpml y");
  check_aniso_cpml_axis(ex, inv_kappa_z, b_z, c_z, 2, "ex_cpml z");
  check_aniso_cpml_psi(ex, psi_x, psi_y, psi_z, "ex_cpml psi");
  torch::stable::accelerator::DeviceGuard guard(ex.get_device_index());
  const dim3 block = aniso_block3d();
  update_electric_ex_full_aniso_cpml_kernel<<<
      aniso_grid3d(ex.size(0), ex.size(1), ex.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(ex.size(0)),
      static_cast<unsigned int>(ex.size(1)),
      static_cast<unsigned int>(ex.size(2)),
      hx.mutable_data_ptr<float>(),
      hy.mutable_data_ptr<float>(),
      hz.mutable_data_ptr<float>(),
      coeff_y.mutable_data_ptr<float>(),
      coeff_z.mutable_data_ptr<float>(),
      inv_dx.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      inv_kappa_x.mutable_data_ptr<float>(),
      b_x.mutable_data_ptr<float>(),
      c_x.mutable_data_ptr<float>(),
      inv_kappa_y.mutable_data_ptr<float>(),
      b_y.mutable_data_ptr<float>(),
      c_y.mutable_data_ptr<float>(),
      inv_kappa_z.mutable_data_ptr<float>(),
      b_z.mutable_data_ptr<float>(),
      c_z.mutable_data_ptr<float>(),
      static_cast<int>(periodic_x),
      static_cast<int>(periodic_y),
      static_cast<int>(periodic_z),
      psi_x.mutable_data_ptr<float>(),
      psi_y.mutable_data_ptr<float>(),
      psi_z.mutable_data_ptr<float>(),
      ex.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ey_full_aniso_cpml_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& coeff_x,
    const torch::stable::Tensor& coeff_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    int64_t periodic_x,
    int64_t periodic_y,
    int64_t periodic_z,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_y,
    torch::stable::Tensor psi_z) {
  const int64_t nx_nodes = ey.size(0);
  const int64_t ny_nodes = ey.size(1) + 1;
  const int64_t nz_nodes = ey.size(2);
  check_full_aniso_inputs(ey, hx, hy, hz, coeff_x, coeff_z, inv_dx, inv_dy, inv_dz,
                          nx_nodes, ny_nodes, nz_nodes, "ey_cpml");
  check_aniso_cpml_axis(ey, inv_kappa_x, b_x, c_x, 0, "ey_cpml x");
  check_aniso_cpml_axis(ey, inv_kappa_y, b_y, c_y, 1, "ey_cpml y");
  check_aniso_cpml_axis(ey, inv_kappa_z, b_z, c_z, 2, "ey_cpml z");
  check_aniso_cpml_psi(ey, psi_x, psi_y, psi_z, "ey_cpml psi");
  torch::stable::accelerator::DeviceGuard guard(ey.get_device_index());
  const dim3 block = aniso_block3d();
  update_electric_ey_full_aniso_cpml_kernel<<<
      aniso_grid3d(ey.size(0), ey.size(1), ey.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(ey.size(0)),
      static_cast<unsigned int>(ey.size(1)),
      static_cast<unsigned int>(ey.size(2)),
      hx.mutable_data_ptr<float>(),
      hy.mutable_data_ptr<float>(),
      hz.mutable_data_ptr<float>(),
      coeff_x.mutable_data_ptr<float>(),
      coeff_z.mutable_data_ptr<float>(),
      inv_dx.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      inv_kappa_x.mutable_data_ptr<float>(),
      b_x.mutable_data_ptr<float>(),
      c_x.mutable_data_ptr<float>(),
      inv_kappa_y.mutable_data_ptr<float>(),
      b_y.mutable_data_ptr<float>(),
      c_y.mutable_data_ptr<float>(),
      inv_kappa_z.mutable_data_ptr<float>(),
      b_z.mutable_data_ptr<float>(),
      c_z.mutable_data_ptr<float>(),
      static_cast<int>(periodic_x),
      static_cast<int>(periodic_y),
      static_cast<int>(periodic_z),
      psi_x.mutable_data_ptr<float>(),
      psi_y.mutable_data_ptr<float>(),
      psi_z.mutable_data_ptr<float>(),
      ey.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void update_electric_ez_full_aniso_cpml_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& coeff_x,
    const torch::stable::Tensor& coeff_y,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    int64_t periodic_x,
    int64_t periodic_y,
    int64_t periodic_z,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_y,
    torch::stable::Tensor psi_z) {
  const int64_t nx_nodes = ez.size(0);
  const int64_t ny_nodes = ez.size(1);
  const int64_t nz_nodes = ez.size(2) + 1;
  check_full_aniso_inputs(ez, hx, hy, hz, coeff_x, coeff_y, inv_dx, inv_dy, inv_dz,
                          nx_nodes, ny_nodes, nz_nodes, "ez_cpml");
  check_aniso_cpml_axis(ez, inv_kappa_x, b_x, c_x, 0, "ez_cpml x");
  check_aniso_cpml_axis(ez, inv_kappa_y, b_y, c_y, 1, "ez_cpml y");
  check_aniso_cpml_axis(ez, inv_kappa_z, b_z, c_z, 2, "ez_cpml z");
  check_aniso_cpml_psi(ez, psi_x, psi_y, psi_z, "ez_cpml psi");
  torch::stable::accelerator::DeviceGuard guard(ez.get_device_index());
  const dim3 block = aniso_block3d();
  update_electric_ez_full_aniso_cpml_kernel<<<
      aniso_grid3d(ez.size(0), ez.size(1), ez.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(ez.size(0)),
      static_cast<unsigned int>(ez.size(1)),
      static_cast<unsigned int>(ez.size(2)),
      hx.mutable_data_ptr<float>(),
      hy.mutable_data_ptr<float>(),
      hz.mutable_data_ptr<float>(),
      coeff_x.mutable_data_ptr<float>(),
      coeff_y.mutable_data_ptr<float>(),
      inv_dx.mutable_data_ptr<float>(),
      inv_dy.mutable_data_ptr<float>(),
      inv_dz.mutable_data_ptr<float>(),
      inv_kappa_x.mutable_data_ptr<float>(),
      b_x.mutable_data_ptr<float>(),
      c_x.mutable_data_ptr<float>(),
      inv_kappa_y.mutable_data_ptr<float>(),
      b_y.mutable_data_ptr<float>(),
      c_y.mutable_data_ptr<float>(),
      inv_kappa_z.mutable_data_ptr<float>(),
      b_z.mutable_data_ptr<float>(),
      c_z.mutable_data_ptr<float>(),
      static_cast<int>(periodic_x),
      static_cast<int>(periodic_y),
      static_cast<int>(periodic_z),
      psi_x.mutable_data_ptr<float>(),
      psi_y.mutable_data_ptr<float>(),
      psi_z.mutable_data_ptr<float>(),
      ez.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void apply_aniso_offdiag_current_ex_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& jy,
    const torch::stable::Tensor& jz,
    const torch::stable::Tensor& coeff_y,
    const torch::stable::Tensor& coeff_z,
    int64_t periodic_x,
    int64_t periodic_y,
    int64_t periodic_z) {
  const int64_t nx_nodes = ex.size(0) + 1;
  const int64_t ny_nodes = ex.size(1);
  const int64_t nz_nodes = ex.size(2);
  check_aniso_offdiag_current_inputs(ex, coeff_y, coeff_z, "ex_offdiag_current");
  check_current_field(jy, ex, nx_nodes, ny_nodes - 1, nz_nodes, "ex_offdiag_current jy");
  check_current_field(jz, ex, nx_nodes, ny_nodes, nz_nodes - 1, "ex_offdiag_current jz");
  torch::stable::accelerator::DeviceGuard guard(ex.get_device_index());
  const dim3 block = aniso_block3d();
  apply_aniso_offdiag_current_ex_kernel<<<
      aniso_grid3d(ex.size(0), ex.size(1), ex.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(ex.size(0)),
      static_cast<unsigned int>(ex.size(1)),
      static_cast<unsigned int>(ex.size(2)),
      jy.mutable_data_ptr<float>(),
      jz.mutable_data_ptr<float>(),
      coeff_y.mutable_data_ptr<float>(),
      coeff_z.mutable_data_ptr<float>(),
      static_cast<int>(periodic_x),
      static_cast<int>(periodic_y),
      static_cast<int>(periodic_z),
      ex.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void apply_aniso_offdiag_current_ey_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& jx,
    const torch::stable::Tensor& jz,
    const torch::stable::Tensor& coeff_x,
    const torch::stable::Tensor& coeff_z,
    int64_t periodic_x,
    int64_t periodic_y,
    int64_t periodic_z) {
  const int64_t nx_nodes = ey.size(0);
  const int64_t ny_nodes = ey.size(1) + 1;
  const int64_t nz_nodes = ey.size(2);
  check_aniso_offdiag_current_inputs(ey, coeff_x, coeff_z, "ey_offdiag_current");
  check_current_field(jx, ey, nx_nodes - 1, ny_nodes, nz_nodes, "ey_offdiag_current jx");
  check_current_field(jz, ey, nx_nodes, ny_nodes, nz_nodes - 1, "ey_offdiag_current jz");
  torch::stable::accelerator::DeviceGuard guard(ey.get_device_index());
  const dim3 block = aniso_block3d();
  apply_aniso_offdiag_current_ey_kernel<<<
      aniso_grid3d(ey.size(0), ey.size(1), ey.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(ey.size(0)),
      static_cast<unsigned int>(ey.size(1)),
      static_cast<unsigned int>(ey.size(2)),
      jx.mutable_data_ptr<float>(),
      jz.mutable_data_ptr<float>(),
      coeff_x.mutable_data_ptr<float>(),
      coeff_z.mutable_data_ptr<float>(),
      static_cast<int>(periodic_x),
      static_cast<int>(periodic_y),
      static_cast<int>(periodic_z),
      ey.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void apply_aniso_offdiag_current_ez_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& jx,
    const torch::stable::Tensor& jy,
    const torch::stable::Tensor& coeff_x,
    const torch::stable::Tensor& coeff_y,
    int64_t periodic_x,
    int64_t periodic_y,
    int64_t periodic_z) {
  const int64_t nx_nodes = ez.size(0);
  const int64_t ny_nodes = ez.size(1);
  const int64_t nz_nodes = ez.size(2) + 1;
  check_aniso_offdiag_current_inputs(ez, coeff_x, coeff_y, "ez_offdiag_current");
  check_current_field(jx, ez, nx_nodes - 1, ny_nodes, nz_nodes, "ez_offdiag_current jx");
  check_current_field(jy, ez, nx_nodes, ny_nodes - 1, nz_nodes, "ez_offdiag_current jy");
  torch::stable::accelerator::DeviceGuard guard(ez.get_device_index());
  const dim3 block = aniso_block3d();
  apply_aniso_offdiag_current_ez_kernel<<<
      aniso_grid3d(ez.size(0), ez.size(1), ez.size(2), block), block, 0, current_cuda_stream()>>>(
      static_cast<unsigned int>(ez.size(0)),
      static_cast<unsigned int>(ez.size(1)),
      static_cast<unsigned int>(ez.size(2)),
      jx.mutable_data_ptr<float>(),
      jy.mutable_data_ptr<float>(),
      coeff_x.mutable_data_ptr<float>(),
      coeff_y.mutable_data_ptr<float>(),
      static_cast<int>(periodic_x),
      static_cast<int>(periodic_y),
      static_cast<int>(periodic_z),
      ez.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}
