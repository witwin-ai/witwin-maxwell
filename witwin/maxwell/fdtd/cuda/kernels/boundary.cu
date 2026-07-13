#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

#include <type_traits>

namespace {

void check_field3d(const torch::stable::Tensor& field, const char* name) {
  check_float32_tensor(field, name);
  check_contiguous_tensor(field, name);
  STD_TORCH_CHECK(field.dim() == 3, name, " must be a contiguous 3D float32 CUDA tensor");
}

int64_t face_area(int64_t axis, int64_t nx, int64_t ny, int64_t nz) {
  if (axis == 0) {
    return ny * nz;
  }
  if (axis == 1) {
    return nx * nz;
  }
  return nx * ny;
}

template <int Axis>
__device__ __forceinline__ long long boundary_face_offset(
    int nx,
    int ny,
    int nz,
    int side,
    int64_t face_linear,
    int& i_out,
    int& j_out,
    int& k_out) {
  if constexpr (Axis == 0) {
    i_out = side == 0 ? 0 : nx - 1;
    j_out = static_cast<int>(face_linear / nz);
    k_out = static_cast<int>(face_linear - static_cast<int64_t>(j_out) * nz);
    return offset3d(
        static_cast<unsigned int>(i_out),
        static_cast<unsigned int>(j_out),
        static_cast<unsigned int>(k_out),
        static_cast<unsigned int>(ny),
        static_cast<unsigned int>(nz));
  }
  if constexpr (Axis == 1) {
    i_out = static_cast<int>(face_linear / nz);
    j_out = side == 0 ? 0 : ny - 1;
    k_out = static_cast<int>(face_linear - static_cast<int64_t>(i_out) * nz);
    return offset3d(
        static_cast<unsigned int>(i_out),
        static_cast<unsigned int>(j_out),
        static_cast<unsigned int>(k_out),
        static_cast<unsigned int>(ny),
        static_cast<unsigned int>(nz));
  }
  i_out = static_cast<int>(face_linear / ny);
  j_out = static_cast<int>(face_linear - static_cast<int64_t>(i_out) * ny);
  k_out = side == 0 ? 0 : nz - 1;
  return offset3d(
      static_cast<unsigned int>(i_out),
      static_cast<unsigned int>(j_out),
      static_cast<unsigned int>(k_out),
      static_cast<unsigned int>(ny),
      static_cast<unsigned int>(nz));
}

template <int Axis, int Side>
__global__ void clamp_field_face_kernel(
    int nx,
    int ny,
    int nz,
    float* __restrict__ field) {
  const int u = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int v = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
  if constexpr (Axis == 0) {
    if (u >= nz || v >= ny) {
      return;
    }
    const int i = Side == 0 ? 0 : nx - 1;
    field[offset3d(
        static_cast<unsigned int>(i),
        static_cast<unsigned int>(v),
        static_cast<unsigned int>(u),
        static_cast<unsigned int>(ny),
        static_cast<unsigned int>(nz))] = 0.0f;
  } else if constexpr (Axis == 1) {
    if (u >= nz || v >= nx) {
      return;
    }
    const int j = Side == 0 ? 0 : ny - 1;
    field[offset3d(
        static_cast<unsigned int>(v),
        static_cast<unsigned int>(j),
        static_cast<unsigned int>(u),
        static_cast<unsigned int>(ny),
        static_cast<unsigned int>(nz))] = 0.0f;
  } else {
    if (u >= ny || v >= nx) {
      return;
    }
    const int k = Side == 0 ? 0 : nz - 1;
    field[offset3d(
        static_cast<unsigned int>(v),
        static_cast<unsigned int>(u),
        static_cast<unsigned int>(k),
        static_cast<unsigned int>(ny),
        static_cast<unsigned int>(nz))] = 0.0f;
  }
}

template <int Axis, int Side>
void launch_clamp_field_face(int nx, int ny, int nz, float* __restrict__ field) {
  int width = nz;
  int height = Axis == 0 ? ny : nx;
  if constexpr (Axis == 2) {
    width = ny;
  }
  const dim3 block(32, 8, 1);
  const dim3 grid(
      static_cast<unsigned int>((width + block.x - 1) / block.x),
      static_cast<unsigned int>((height + block.y - 1) / block.y),
      1);
  clamp_field_face_kernel<Axis, Side><<<grid, block, 0, current_cuda_stream()>>>(nx, ny, nz, field);
}

template <int AxisA, int AxisB>
__global__ void clamp_pec_boundary_kernel(
    int64_t total,
    int nx,
    int ny,
    int nz,
    int64_t face_area_a,
    int64_t face_area_b,
    float* __restrict__ field) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int64_t axis_a_total = 2 * face_area_a;
  const bool second_axis = linear >= axis_a_total;
  int64_t local = linear;
  int64_t area = face_area_a;
  if (second_axis) {
    local = linear - axis_a_total;
    area = face_area_b;
  }
  const int side = local >= area ? 1 : 0;
  const int64_t face_linear = local - static_cast<int64_t>(side) * area;
  int i = 0;
  int j = 0;
  int k = 0;
  long long offset = 0;
  if (second_axis) {
    offset = boundary_face_offset<AxisB>(nx, ny, nz, side, face_linear, i, j, k);
  } else {
    offset = boundary_face_offset<AxisA>(nx, ny, nz, side, face_linear, i, j, k);
  }
  if constexpr (AxisA != AxisB) {
    const int axis_a_coord = AxisA == 0 ? i : (AxisA == 1 ? j : k);
    const int axis_a_size = AxisA == 0 ? nx : (AxisA == 1 ? ny : nz);
    if (second_axis && (axis_a_coord == 0 || axis_a_coord + 1 == axis_a_size)) {
      return;
    }
  }
  field[offset] = 0.0f;
}

template <int AxisA, int AxisB>
void launch_clamp_pec_boundary(
    int64_t total,
    int nx,
    int ny,
    int nz,
    int64_t face_area_a,
    int64_t face_area_b,
    float* __restrict__ field) {
  clamp_pec_boundary_kernel<AxisA, AxisB><<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      nx,
      ny,
      nz,
      face_area_a,
      face_area_b,
      field);
}

__device__ __forceinline__ long long mur_plane_offset(
    int axis,
    int normal_index,
    int64_t plane_linear,
    int ny,
    int nz) {
  // Row-major 3D offset of plane element `plane_linear` at the given normal
  // index. The plane spans the two axes other than `axis`, iterated in
  // ascending-axis order, matching a `field.select(axis, normal_index)` view.
  if (axis == 0) {
    const unsigned int j = static_cast<unsigned int>(plane_linear / nz);
    const unsigned int k = static_cast<unsigned int>(plane_linear - static_cast<int64_t>(j) * nz);
    return offset3d(static_cast<unsigned int>(normal_index), j, k, ny, nz);
  }
  if (axis == 1) {
    const unsigned int i = static_cast<unsigned int>(plane_linear / nz);
    const unsigned int k = static_cast<unsigned int>(plane_linear - static_cast<int64_t>(i) * nz);
    return offset3d(i, static_cast<unsigned int>(normal_index), k, ny, nz);
  }
  const unsigned int i = static_cast<unsigned int>(plane_linear / ny);
  const unsigned int j = static_cast<unsigned int>(plane_linear - static_cast<int64_t>(i) * ny);
  return offset3d(i, j, static_cast<unsigned int>(normal_index), ny, nz);
}

// First-order Mur absorbing boundary on one outer face of a single E component.
// Persistent `prev_boundary` / `prev_adjacent` plane buffers carry the previous
// step's boundary and first-interior slices; the kernel reads the current
// adjacent slice, writes the extrapolated boundary slice, and updates both
// buffers in place. No allocation and fixed pointers, so it is safe to capture
// into a CUDA graph and replay.
__global__ void mur_abc_face_kernel(
    int64_t plane_size,
    int axis,
    int ny,
    int nz,
    int boundary_index,
    int adjacent_index,
    float coef,
    float* __restrict__ field,
    float* __restrict__ prev_boundary,
    float* __restrict__ prev_adjacent) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= plane_size) {
    return;
  }
  const float na = field[mur_plane_offset(axis, adjacent_index, t, ny, nz)];
  const float pb = prev_boundary[t];
  const float pa = prev_adjacent[t];
  // E_boundary(n+1) = E_adjacent(n) + coef * (E_adjacent(n+1) - E_boundary(n)).
  const float nb = pa + coef * (na - pb);
  field[mur_plane_offset(axis, boundary_index, t, ny, nz)] = nb;
  prev_boundary[t] = nb;
  prev_adjacent[t] = na;
}

__global__ void sibc_surface_kernel(
    int64_t plane_size,
    int axis,
    int e_ny,
    int e_nz,
    int h_ny,
    int h_nz,
    int e_index,
    int h_index,
    float sign,
    float surface_r,
    float surface_l_over_dt,
    float* __restrict__ electric,
    const float* __restrict__ magnetic,
    float* __restrict__ h_prev) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= plane_size) {
    return;
  }
  const float h_now = magnetic[mur_plane_offset(axis, h_index, t, h_ny, h_nz)];
  electric[mur_plane_offset(axis, e_index, t, e_ny, e_nz)] =
      sign * (surface_r * h_now + surface_l_over_dt * (h_now - h_prev[t]));
  h_prev[t] = h_now;
}

}  // namespace

void clamp_field_face_cuda(torch::stable::Tensor field, int64_t axis, int64_t side) {
  check_field3d(field, "field");
  STD_TORCH_CHECK(axis >= 0 && axis < 3, "axis must be in [0, 3)");
  STD_TORCH_CHECK(side == 0 || side == 1, "side must be 0 (low) or 1 (high)");
  const torch::stable::accelerator::DeviceGuard device_guard(field.get_device_index());
  const int nx = static_cast<int>(field.size(0));
  const int ny = static_cast<int>(field.size(1));
  const int nz = static_cast<int>(field.size(2));
  if (field.numel() == 0) {
    return;
  }
  if (axis == 0) {
    if (side == 0) {
      launch_clamp_field_face<0, 0>(nx, ny, nz, field.mutable_data_ptr<float>());
    } else {
      launch_clamp_field_face<0, 1>(nx, ny, nz, field.mutable_data_ptr<float>());
    }
  } else if (axis == 1) {
    if (side == 0) {
      launch_clamp_field_face<1, 0>(nx, ny, nz, field.mutable_data_ptr<float>());
    } else {
      launch_clamp_field_face<1, 1>(nx, ny, nz, field.mutable_data_ptr<float>());
    }
  } else {
    if (side == 0) {
      launch_clamp_field_face<2, 0>(nx, ny, nz, field.mutable_data_ptr<float>());
    } else {
      launch_clamp_field_face<2, 1>(nx, ny, nz, field.mutable_data_ptr<float>());
    }
  }
  WITWIN_CUDA_CHECK();
}

void clamp_pec_boundary_cuda(torch::stable::Tensor field, int64_t axis_a, int64_t axis_b) {
  check_field3d(field, "field");
  STD_TORCH_CHECK(axis_a >= 0 && axis_a < 3, "axis_a must be in [0, 3)");
  STD_TORCH_CHECK(axis_b >= 0 && axis_b < 3, "axis_b must be in [0, 3)");
  const torch::stable::accelerator::DeviceGuard device_guard(field.get_device_index());
  const int64_t nx = field.size(0);
  const int64_t ny = field.size(1);
  const int64_t nz = field.size(2);
  const int64_t face_area_a = face_area(axis_a, nx, ny, nz);
  const int64_t face_area_b = axis_a == axis_b ? 0 : face_area(axis_b, nx, ny, nz);
  const int64_t total = 2 * (face_area_a + face_area_b);
  if (total == 0) {
    return;
  }
  const auto launch_for_axis_b = [&](auto axis_a_tag) {
    constexpr int axis_a_value = decltype(axis_a_tag)::value;
    if (axis_b == 0) {
      launch_clamp_pec_boundary<axis_a_value, 0>(
          total, static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz), face_area_a, face_area_b, field.mutable_data_ptr<float>());
    } else if (axis_b == 1) {
      launch_clamp_pec_boundary<axis_a_value, 1>(
          total, static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz), face_area_a, face_area_b, field.mutable_data_ptr<float>());
    } else {
      launch_clamp_pec_boundary<axis_a_value, 2>(
          total, static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz), face_area_a, face_area_b, field.mutable_data_ptr<float>());
    }
  };
  if (axis_a == 0) {
    launch_for_axis_b(std::integral_constant<int, 0>{});
  } else if (axis_a == 1) {
    launch_for_axis_b(std::integral_constant<int, 1>{});
  } else {
    launch_for_axis_b(std::integral_constant<int, 2>{});
  }
  WITWIN_CUDA_CHECK();
}

void mur_abc_face_cuda(
    torch::stable::Tensor field,
    int64_t axis,
    int64_t boundary_index,
    int64_t adjacent_index,
    double coef,
    torch::stable::Tensor prev_boundary,
    torch::stable::Tensor prev_adjacent) {
  check_field3d(field, "field");
  check_float32_tensor(prev_boundary, "prev_boundary");
  check_contiguous_tensor(prev_boundary, "prev_boundary");
  check_float32_tensor(prev_adjacent, "prev_adjacent");
  check_contiguous_tensor(prev_adjacent, "prev_adjacent");
  STD_TORCH_CHECK(axis >= 0 && axis < 3, "axis must be in [0, 3)");
  const torch::stable::accelerator::DeviceGuard device_guard(field.get_device_index());
  const int ny = static_cast<int>(field.size(1));
  const int nz = static_cast<int>(field.size(2));
  const int64_t plane_size = prev_boundary.numel();
  STD_TORCH_CHECK(
      prev_adjacent.numel() == plane_size,
      "prev_adjacent and prev_boundary must have the same number of elements");
  if (plane_size == 0) {
    return;
  }
  mur_abc_face_kernel<<<linear_grid(plane_size), 256, 0, current_cuda_stream()>>>(
      plane_size,
      static_cast<int>(axis),
      ny,
      nz,
      static_cast<int>(boundary_index),
      static_cast<int>(adjacent_index),
      static_cast<float>(coef),
      field.mutable_data_ptr<float>(),
      prev_boundary.mutable_data_ptr<float>(),
      prev_adjacent.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}

void apply_sibc_surface_cuda(
    torch::stable::Tensor electric,
    const torch::stable::Tensor& magnetic,
    int64_t axis,
    int64_t electric_index,
    int64_t magnetic_index,
    double sign,
    double surface_r,
    double surface_l_over_dt,
    torch::stable::Tensor h_prev) {
  check_field3d(electric, "electric");
  check_field3d(magnetic, "magnetic");
  check_float32_tensor(h_prev, "h_prev");
  check_contiguous_tensor(h_prev, "h_prev");
  check_same_cuda_device(electric, magnetic, "magnetic");
  check_same_cuda_device(electric, h_prev, "h_prev");
  STD_TORCH_CHECK(axis >= 0 && axis < 3, "axis must be in [0, 3)");
  const int64_t e_axis_size = electric.size(axis);
  const int64_t h_axis_size = magnetic.size(axis);
  STD_TORCH_CHECK(electric_index >= 0 && electric_index < e_axis_size, "electric_index is out of bounds");
  STD_TORCH_CHECK(magnetic_index >= 0 && magnetic_index < h_axis_size, "magnetic_index is out of bounds");
  const int64_t plane_size = face_area(axis, magnetic.size(0), magnetic.size(1), magnetic.size(2));
  STD_TORCH_CHECK(h_prev.numel() == plane_size, "h_prev must match the magnetic face area");
  STD_TORCH_CHECK(
      face_area(axis, electric.size(0), electric.size(1), electric.size(2)) == plane_size,
      "electric and magnetic face shapes must match");
  if (plane_size == 0) {
    return;
  }
  const torch::stable::accelerator::DeviceGuard device_guard(electric.get_device_index());
  sibc_surface_kernel<<<linear_grid(plane_size), 256, 0, current_cuda_stream()>>>(
      plane_size,
      static_cast<int>(axis),
      static_cast<int>(electric.size(1)),
      static_cast<int>(electric.size(2)),
      static_cast<int>(magnetic.size(1)),
      static_cast<int>(magnetic.size(2)),
      static_cast<int>(electric_index),
      static_cast<int>(magnetic_index),
      static_cast<float>(sign),
      static_cast<float>(surface_r),
      static_cast<float>(surface_l_over_dt),
      electric.mutable_data_ptr<float>(),
      magnetic.mutable_data_ptr<float>(),
      h_prev.mutable_data_ptr<float>());
  WITWIN_CUDA_CHECK();
}
