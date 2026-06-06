#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "../launch.h"
#include "../tensors.h"
#include "common.cuh"

namespace {

void check_field3d(const at::Tensor& field, const char* name) {
  check_float32_tensor(field, name);
  check_contiguous_tensor(field, name);
  TORCH_CHECK(field.dim() == 3, name, " must be a contiguous 3D float32 CUDA tensor");
}

__global__ void clamp_pec_boundary_kernel(
    int64_t total,
    int nx,
    int ny,
    int nz,
    int axis_a,
    int axis_b,
    float* field) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const Index3D index = unflatten3d(
      static_cast<unsigned int>(linear),
      static_cast<unsigned int>(ny),
      static_cast<unsigned int>(nz));
  const int coords[3] = {
      static_cast<int>(index.i),
      static_cast<int>(index.j),
      static_cast<int>(index.k),
  };
  const int sizes[3] = {nx, ny, nz};
  const bool on_axis_a = coords[axis_a] == 0 || coords[axis_a] == sizes[axis_a] - 1;
  const bool on_axis_b = coords[axis_b] == 0 || coords[axis_b] == sizes[axis_b] - 1;
  if (on_axis_a || on_axis_b) {
    field[linear] = 0.0f;
  }
}

}  // namespace

void clamp_pec_boundary_cuda(at::Tensor field, int64_t axis_a, int64_t axis_b) {
  check_field3d(field, "field");
  TORCH_CHECK(axis_a >= 0 && axis_a < 3, "axis_a must be in [0, 3)");
  TORCH_CHECK(axis_b >= 0 && axis_b < 3, "axis_b must be in [0, 3)");
  const c10::cuda::CUDAGuard device_guard(field.device());
  const int64_t total = field.numel();
  clamp_pec_boundary_kernel<<<linear_grid(total), 256, 0, current_cuda_stream()>>>(
      total,
      static_cast<int>(field.size(0)),
      static_cast<int>(field.size(1)),
      static_cast<int>(field.size(2)),
      static_cast<int>(axis_a),
      static_cast<int>(axis_b),
      field.data_ptr<float>());
  WITWIN_CUDA_CHECK();
}
