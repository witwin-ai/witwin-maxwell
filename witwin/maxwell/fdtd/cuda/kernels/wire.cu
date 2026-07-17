#include "../launch.h"
#include "../tensors.h"

#include <torch/headeronly/core/ScalarType.h>

namespace {

// These hot-path operators consume topology produced by compiler/thin_wire.py.
// Content invariants (CSR monotonicity and terminals, component/index domains,
// and unique deposit targets) are checked once by initialize_wire_runtime before
// launch. Rechecking device index contents here would add a synchronization to
// every step and would make CUDA Graph capture impossible.

constexpr int kExComponent = 0;
constexpr int kEyComponent = 1;

bool is_wire_scalar_type(torch::headeronly::ScalarType dtype) {
  return dtype == torch::headeronly::ScalarType::Float ||
      dtype == torch::headeronly::ScalarType::Double;
}

void check_wire_real_tensor(const torch::stable::Tensor& tensor, const char* name) {
  check_cuda_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  STD_TORCH_CHECK(is_wire_scalar_type(tensor.scalar_type()), name, " must be float32 or float64");
}

void check_wire_field(const torch::stable::Tensor& tensor, const char* name) {
  check_wire_real_tensor(tensor, name);
  STD_TORCH_CHECK(tensor.dim() == 3, name, " must be a contiguous 3D CUDA tensor");
}

void check_wire_vector(const torch::stable::Tensor& tensor, const char* name) {
  check_wire_real_tensor(tensor, name);
  STD_TORCH_CHECK(tensor.dim() == 1, name, " must be a contiguous 1D CUDA tensor");
}

void check_index_vector(
    const torch::stable::Tensor& tensor,
    torch::headeronly::ScalarType dtype,
    const char* name,
    const char* dtype_name) {
  check_cuda_tensor(tensor, name);
  check_contiguous_tensor(tensor, name);
  STD_TORCH_CHECK(tensor.dim() == 1, name, " must be a contiguous 1D CUDA tensor");
  STD_TORCH_CHECK(tensor.scalar_type() == dtype, name, " must be ", dtype_name);
}

void check_same_wire_dtype(
    const torch::stable::Tensor& reference,
    const torch::stable::Tensor& tensor,
    const char* name) {
  STD_TORCH_CHECK(tensor.scalar_type() == reference.scalar_type(), name, " must match the wire scalar dtype");
}

template <typename scalar_t>
__global__ void sample_wire_emf_kernel(
    int64_t segment_count,
    const scalar_t* __restrict__ ex,
    const scalar_t* __restrict__ ey,
    const scalar_t* __restrict__ ez,
    const int64_t* __restrict__ segment_offsets,
    const int32_t* __restrict__ edge_components,
    const int64_t* __restrict__ edge_offsets,
    const scalar_t* __restrict__ weights,
    scalar_t* __restrict__ emf) {
  const int64_t segment = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (segment >= segment_count) {
    return;
  }

  scalar_t value = scalar_t(0);
  const int64_t begin = segment_offsets[segment];
  const int64_t end = segment_offsets[segment + 1];
  for (int64_t entry = begin; entry < end; ++entry) {
    const int64_t offset = edge_offsets[entry];
    const int32_t component = edge_components[entry];
    const scalar_t field_value = component == kExComponent
        ? ex[offset]
        : (component == kEyComponent ? ey[offset] : ez[offset]);
    value += weights[entry] * field_value;
  }
  emf[segment] = value;
}

template <typename scalar_t>
__global__ void update_wire_current_kernel(
    int64_t segment_count,
    const scalar_t* __restrict__ emf,
    const int64_t* __restrict__ tail,
    const int64_t* __restrict__ head,
    const scalar_t* __restrict__ inductance,
    const scalar_t* __restrict__ node_capacitance,
    const bool* __restrict__ grounded,
    scalar_t dt,
    scalar_t* __restrict__ current,
    const scalar_t* __restrict__ charge) {
  const int64_t segment = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (segment >= segment_count) {
    return;
  }

  const int64_t tail_node = tail[segment];
  const int64_t head_node = head[segment];
  const scalar_t tail_voltage = grounded[tail_node]
      ? scalar_t(0)
      : charge[tail_node] / node_capacitance[tail_node];
  const scalar_t head_voltage = grounded[head_node]
      ? scalar_t(0)
      : charge[head_node] / node_capacitance[head_node];
  current[segment] +=
      dt * (emf[segment] + tail_voltage - head_voltage) / inductance[segment];
}

template <typename scalar_t>
__global__ void update_wire_charge_kernel(
    int64_t node_count,
    const int64_t* __restrict__ node_offsets,
    const int64_t* __restrict__ node_segments,
    const int32_t* __restrict__ node_signs,
    const bool* __restrict__ grounded,
    scalar_t dt,
    const scalar_t* __restrict__ current,
    scalar_t* __restrict__ charge) {
  const int64_t node = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (node >= node_count) {
    return;
  }
  if (grounded[node]) {
    charge[node] = scalar_t(0);
    return;
  }

  scalar_t incidence_current = scalar_t(0);
  const int64_t begin = node_offsets[node];
  const int64_t end = node_offsets[node + 1];
  for (int64_t entry = begin; entry < end; ++entry) {
    incidence_current +=
        static_cast<scalar_t>(node_signs[entry]) * current[node_segments[entry]];
  }
  charge[node] -= dt * incidence_current;
}

template <typename scalar_t>
__global__ void deposit_wire_current_kernel(
    int64_t target_count,
    scalar_t* __restrict__ ex,
    scalar_t* __restrict__ ey,
    scalar_t* __restrict__ ez,
    const int64_t* __restrict__ edge_group_offsets,
    const int32_t* __restrict__ target_components,
    const int64_t* __restrict__ target_offsets,
    const int64_t* __restrict__ contribution_segments,
    const scalar_t* __restrict__ contribution_scales,
    const scalar_t* __restrict__ current) {
  const int64_t target = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (target >= target_count) {
    return;
  }

  scalar_t contribution = scalar_t(0);
  const int64_t begin = edge_group_offsets[target];
  const int64_t end = edge_group_offsets[target + 1];
  for (int64_t entry = begin; entry < end; ++entry) {
    contribution += contribution_scales[entry] * current[contribution_segments[entry]];
  }

  const int64_t offset = target_offsets[target];
  const int32_t component = target_components[target];
  if (component == kExComponent) {
    ex[offset] -= contribution;
  } else if (component == kEyComponent) {
    ey[offset] -= contribution;
  } else {
    ez[offset] -= contribution;
  }
}

template <typename LaunchFloat, typename LaunchDouble>
void dispatch_wire_scalar(
    torch::headeronly::ScalarType dtype,
    LaunchFloat&& launch_float,
    LaunchDouble&& launch_double) {
  if (dtype == torch::headeronly::ScalarType::Float) {
    launch_float();
  } else {
    launch_double();
  }
}

}  // namespace

void sample_wire_emf_cuda(
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& segment_offsets,
    const torch::stable::Tensor& edge_components,
    const torch::stable::Tensor& edge_offsets,
    const torch::stable::Tensor& weights,
    torch::stable::Tensor emf) {
  check_wire_field(ex, "ex");
  check_wire_field(ey, "ey");
  check_wire_field(ez, "ez");
  check_index_vector(segment_offsets, torch::headeronly::ScalarType::Long, "segment_offsets", "int64");
  check_index_vector(edge_components, torch::headeronly::ScalarType::Int, "edge_components", "int32");
  check_index_vector(edge_offsets, torch::headeronly::ScalarType::Long, "edge_offsets", "int64");
  check_wire_vector(weights, "weights");
  check_wire_vector(emf, "emf");
  check_same_cuda_device(ex, ey, "ey");
  check_same_cuda_device(ex, ez, "ez");
  check_same_cuda_device(ex, segment_offsets, "segment_offsets");
  check_same_cuda_device(ex, edge_components, "edge_components");
  check_same_cuda_device(ex, edge_offsets, "edge_offsets");
  check_same_cuda_device(ex, weights, "weights");
  check_same_cuda_device(ex, emf, "emf");
  check_same_wire_dtype(ex, ey, "ey");
  check_same_wire_dtype(ex, ez, "ez");
  check_same_wire_dtype(ex, weights, "weights");
  check_same_wire_dtype(ex, emf, "emf");
  STD_TORCH_CHECK(
      segment_offsets.numel() == emf.numel() + 1,
      "segment_offsets must have one more entry than emf");
  STD_TORCH_CHECK(
      edge_components.numel() == edge_offsets.numel() &&
          edge_offsets.numel() == weights.numel(),
      "edge_components, edge_offsets, and weights must have matching lengths");
  if (emf.numel() == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(ex.get_device_index());
  const int64_t segment_count = emf.numel();
  const dim3 grid = linear_grid(segment_count);
  const cudaStream_t stream = current_cuda_stream();
  dispatch_wire_scalar(
      ex.scalar_type(),
      [&] {
        sample_wire_emf_kernel<float><<<grid, 256, 0, stream>>>(
            segment_count,
            ex.mutable_data_ptr<float>(),
            ey.mutable_data_ptr<float>(),
            ez.mutable_data_ptr<float>(),
            segment_offsets.mutable_data_ptr<int64_t>(),
            edge_components.mutable_data_ptr<int32_t>(),
            edge_offsets.mutable_data_ptr<int64_t>(),
            weights.mutable_data_ptr<float>(),
            emf.mutable_data_ptr<float>());
      },
      [&] {
        sample_wire_emf_kernel<double><<<grid, 256, 0, stream>>>(
            segment_count,
            ex.mutable_data_ptr<double>(),
            ey.mutable_data_ptr<double>(),
            ez.mutable_data_ptr<double>(),
            segment_offsets.mutable_data_ptr<int64_t>(),
            edge_components.mutable_data_ptr<int32_t>(),
            edge_offsets.mutable_data_ptr<int64_t>(),
            weights.mutable_data_ptr<double>(),
            emf.mutable_data_ptr<double>());
      });
  WITWIN_CUDA_CHECK();
}

void update_wire_state_cuda(
    const torch::stable::Tensor& emf,
    const torch::stable::Tensor& tail,
    const torch::stable::Tensor& head,
    const torch::stable::Tensor& inductance,
    const torch::stable::Tensor& node_capacitance,
    const torch::stable::Tensor& grounded,
    const torch::stable::Tensor& node_offsets,
    const torch::stable::Tensor& node_segments,
    const torch::stable::Tensor& node_signs,
    double dt,
    torch::stable::Tensor current,
    torch::stable::Tensor charge) {
  check_wire_vector(emf, "emf");
  check_index_vector(tail, torch::headeronly::ScalarType::Long, "tail", "int64");
  check_index_vector(head, torch::headeronly::ScalarType::Long, "head", "int64");
  check_wire_vector(inductance, "inductance");
  check_wire_vector(node_capacitance, "node_capacitance");
  check_index_vector(grounded, torch::headeronly::ScalarType::Bool, "grounded", "bool");
  check_index_vector(node_offsets, torch::headeronly::ScalarType::Long, "node_offsets", "int64");
  check_index_vector(node_segments, torch::headeronly::ScalarType::Long, "node_segments", "int64");
  check_index_vector(node_signs, torch::headeronly::ScalarType::Int, "node_signs", "int32");
  check_wire_vector(current, "current");
  check_wire_vector(charge, "charge");

  const torch::stable::Tensor* tensors[] = {
      &tail,
      &head,
      &inductance,
      &node_capacitance,
      &grounded,
      &node_offsets,
      &node_segments,
      &node_signs,
      &current,
      &charge};
  const char* names[] = {
      "tail",
      "head",
      "inductance",
      "node_capacitance",
      "grounded",
      "node_offsets",
      "node_segments",
      "node_signs",
      "current",
      "charge"};
  for (int index = 0; index < 10; ++index) {
    check_same_cuda_device(emf, *tensors[index], names[index]);
  }
  check_same_wire_dtype(emf, inductance, "inductance");
  check_same_wire_dtype(emf, node_capacitance, "node_capacitance");
  check_same_wire_dtype(emf, current, "current");
  check_same_wire_dtype(emf, charge, "charge");

  const int64_t segment_count = current.numel();
  const int64_t node_count = charge.numel();
  STD_TORCH_CHECK(
      emf.numel() == segment_count && tail.numel() == segment_count &&
          head.numel() == segment_count && inductance.numel() == segment_count,
      "emf, tail, head, inductance, and current must have matching segment lengths");
  STD_TORCH_CHECK(
      node_capacitance.numel() == node_count && grounded.numel() == node_count,
      "node_capacitance, grounded, and charge must have matching node lengths");
  STD_TORCH_CHECK(
      node_offsets.numel() == node_count + 1,
      "node_offsets must have one more entry than charge");
  STD_TORCH_CHECK(
      node_segments.numel() == node_signs.numel(),
      "node_segments and node_signs must have matching lengths");

  const torch::stable::accelerator::DeviceGuard device_guard(emf.get_device_index());
  const cudaStream_t stream = current_cuda_stream();
  if (segment_count > 0) {
    const dim3 segment_grid = linear_grid(segment_count);
    dispatch_wire_scalar(
        emf.scalar_type(),
        [&] {
          update_wire_current_kernel<float><<<segment_grid, 256, 0, stream>>>(
              segment_count,
              emf.mutable_data_ptr<float>(),
              tail.mutable_data_ptr<int64_t>(),
              head.mutable_data_ptr<int64_t>(),
              inductance.mutable_data_ptr<float>(),
              node_capacitance.mutable_data_ptr<float>(),
              grounded.mutable_data_ptr<bool>(),
              static_cast<float>(dt),
              current.mutable_data_ptr<float>(),
              charge.mutable_data_ptr<float>());
        },
        [&] {
          update_wire_current_kernel<double><<<segment_grid, 256, 0, stream>>>(
              segment_count,
              emf.mutable_data_ptr<double>(),
              tail.mutable_data_ptr<int64_t>(),
              head.mutable_data_ptr<int64_t>(),
              inductance.mutable_data_ptr<double>(),
              node_capacitance.mutable_data_ptr<double>(),
              grounded.mutable_data_ptr<bool>(),
              dt,
              current.mutable_data_ptr<double>(),
              charge.mutable_data_ptr<double>());
        });
    WITWIN_CUDA_CHECK();
  }

  if (node_count > 0) {
    const dim3 node_grid = linear_grid(node_count);
    dispatch_wire_scalar(
        emf.scalar_type(),
        [&] {
          update_wire_charge_kernel<float><<<node_grid, 256, 0, stream>>>(
              node_count,
              node_offsets.mutable_data_ptr<int64_t>(),
              node_segments.mutable_data_ptr<int64_t>(),
              node_signs.mutable_data_ptr<int32_t>(),
              grounded.mutable_data_ptr<bool>(),
              static_cast<float>(dt),
              current.mutable_data_ptr<float>(),
              charge.mutable_data_ptr<float>());
        },
        [&] {
          update_wire_charge_kernel<double><<<node_grid, 256, 0, stream>>>(
              node_count,
              node_offsets.mutable_data_ptr<int64_t>(),
              node_segments.mutable_data_ptr<int64_t>(),
              node_signs.mutable_data_ptr<int32_t>(),
              grounded.mutable_data_ptr<bool>(),
              dt,
              current.mutable_data_ptr<double>(),
              charge.mutable_data_ptr<double>());
        });
    WITWIN_CUDA_CHECK();
  }
}

void deposit_wire_current_cuda(
    torch::stable::Tensor ex,
    torch::stable::Tensor ey,
    torch::stable::Tensor ez,
    const torch::stable::Tensor& edge_group_offsets,
    const torch::stable::Tensor& target_components,
    const torch::stable::Tensor& target_offsets,
    const torch::stable::Tensor& contribution_segments,
    const torch::stable::Tensor& contribution_scales,
    const torch::stable::Tensor& current) {
  check_wire_field(ex, "ex");
  check_wire_field(ey, "ey");
  check_wire_field(ez, "ez");
  check_index_vector(edge_group_offsets, torch::headeronly::ScalarType::Long, "edge_group_offsets", "int64");
  check_index_vector(target_components, torch::headeronly::ScalarType::Int, "target_components", "int32");
  check_index_vector(target_offsets, torch::headeronly::ScalarType::Long, "target_offsets", "int64");
  check_index_vector(contribution_segments, torch::headeronly::ScalarType::Long, "contribution_segments", "int64");
  check_wire_vector(contribution_scales, "contribution_scales");
  check_wire_vector(current, "current");

  const torch::stable::Tensor* tensors[] = {
      &ey,
      &ez,
      &edge_group_offsets,
      &target_components,
      &target_offsets,
      &contribution_segments,
      &contribution_scales,
      &current};
  const char* names[] = {
      "ey",
      "ez",
      "edge_group_offsets",
      "target_components",
      "target_offsets",
      "contribution_segments",
      "contribution_scales",
      "current"};
  for (int index = 0; index < 8; ++index) {
    check_same_cuda_device(ex, *tensors[index], names[index]);
  }
  check_same_wire_dtype(ex, ey, "ey");
  check_same_wire_dtype(ex, ez, "ez");
  check_same_wire_dtype(ex, contribution_scales, "contribution_scales");
  check_same_wire_dtype(ex, current, "current");

  const int64_t target_count = target_components.numel();
  STD_TORCH_CHECK(
      target_offsets.numel() == target_count,
      "target_components and target_offsets must have matching lengths");
  STD_TORCH_CHECK(
      edge_group_offsets.numel() == target_count + 1,
      "edge_group_offsets must have one more entry than target_components");
  STD_TORCH_CHECK(
      contribution_segments.numel() == contribution_scales.numel(),
      "contribution_segments and contribution_scales must have matching lengths");
  if (target_count == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(ex.get_device_index());
  const dim3 grid = linear_grid(target_count);
  const cudaStream_t stream = current_cuda_stream();
  dispatch_wire_scalar(
      ex.scalar_type(),
      [&] {
        deposit_wire_current_kernel<float><<<grid, 256, 0, stream>>>(
            target_count,
            ex.mutable_data_ptr<float>(),
            ey.mutable_data_ptr<float>(),
            ez.mutable_data_ptr<float>(),
            edge_group_offsets.mutable_data_ptr<int64_t>(),
            target_components.mutable_data_ptr<int32_t>(),
            target_offsets.mutable_data_ptr<int64_t>(),
            contribution_segments.mutable_data_ptr<int64_t>(),
            contribution_scales.mutable_data_ptr<float>(),
            current.mutable_data_ptr<float>());
      },
      [&] {
        deposit_wire_current_kernel<double><<<grid, 256, 0, stream>>>(
            target_count,
            ex.mutable_data_ptr<double>(),
            ey.mutable_data_ptr<double>(),
            ez.mutable_data_ptr<double>(),
            edge_group_offsets.mutable_data_ptr<int64_t>(),
            target_components.mutable_data_ptr<int32_t>(),
            target_offsets.mutable_data_ptr<int64_t>(),
            contribution_segments.mutable_data_ptr<int64_t>(),
            contribution_scales.mutable_data_ptr<double>(),
            current.mutable_data_ptr<double>());
      });
  WITWIN_CUDA_CHECK();
}
