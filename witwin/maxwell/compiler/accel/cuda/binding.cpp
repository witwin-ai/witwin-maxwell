#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

void sample_setup(std::vector<torch::stable::Tensor> outputs, std::vector<torch::stable::Tensor> axes,
                  torch::stable::Tensor geometry, torch::stable::Tensor values, torch::stable::Tensor fills,
                  torch::stable::Tensor offsets, torch::stable::Tensor ids,
                  std::vector<int64_t> sizes, std::vector<double> constants,
                  std::vector<torch::stable::Tensor> pole_outputs, std::vector<torch::stable::Tensor> amplitudes,
                  std::vector<int64_t> owners);
std::vector<torch::stable::Tensor> poles_setup(torch::stable::Tensor geometry, torch::stable::Tensor geometry64,
    torch::stable::Tensor boxes, std::vector<torch::stable::Tensor> planes, std::vector<torch::stable::Tensor> centres,
    std::vector<torch::stable::Tensor> queries, std::vector<torch::stable::Tensor> widths,
    torch::stable::Tensor values, torch::stable::Tensor carriers, torch::stable::Tensor seed, torch::stable::Tensor old,
    torch::stable::Tensor slots, std::vector<int64_t> sizes, std::vector<double> constants);
torch::stable::Tensor distance_setup(torch::stable::Tensor geometry, torch::stable::Tensor points);
torch::stable::Tensor polygon_distance(torch::stable::Tensor points, torch::stable::Tensor starts, torch::stable::Tensor ends);
torch::stable::Tensor coverage_setup(torch::stable::Tensor geometry, torch::stable::Tensor points,
    std::vector<torch::stable::Tensor> widths, std::vector<int64_t> shape, std::vector<double> constants);
std::vector<torch::stable::Tensor> conductor_edges(torch::stable::Tensor distance);
std::vector<torch::stable::Tensor> bulk_setup(torch::stable::Tensor distances, torch::stable::Tensor values,
                                 torch::stable::Tensor band, double background, bool smooth);
torch::stable::Tensor window_setup(torch::stable::Tensor sample, torch::stable::Tensor offsets, int64_t axis);
torch::stable::Tensor reconstruct_setup(torch::stable::Tensor arithmetic, torch::stable::Tensor inverse,
                             std::vector<torch::stable::Tensor> gradients, int64_t axis, bool node);
std::vector<torch::stable::Tensor> coefficients_setup(std::vector<torch::stable::Tensor> material,
    std::vector<torch::stable::Tensor> sigma, std::vector<torch::stable::Tensor> profiles,
    std::vector<int64_t> shape, double dt, bool electric_loss, bool magnetic_loss);
std::vector<torch::stable::Tensor> hash_setup(std::vector<torch::stable::Tensor> input, std::vector<int64_t> shape);
torch::stable::Tensor gather_setup(std::vector<torch::stable::Tensor> input, std::vector<int64_t> shape, torch::stable::Tensor owners);
torch::stable::Tensor remap_setup(torch::stable::Tensor slots, torch::stable::Tensor remap, bool wide);
torch::stable::Tensor mesh_keys(torch::stable::Tensor triangles, torch::stable::Tensor extent);
torch::stable::Tensor mesh_topology(torch::stable::Tensor keys);
torch::stable::Tensor mesh_bounds(torch::stable::Tensor triangles, torch::stable::Tensor topology, torch::stable::Tensor ids);
torch::stable::Tensor mesh_distance(torch::stable::Tensor points, torch::stable::Tensor triangles, std::vector<torch::stable::Tensor> hierarchy);

STABLE_TORCH_LIBRARY(witwin_maxwell_setup, m) {
  m.def("sample(Tensor(a!)[] outputs, Tensor[] axes, Tensor geometry, Tensor values, Tensor fills, Tensor offsets, Tensor ids, int[] sizes, float[] constants, Tensor(b!)[] pole_outputs, Tensor[] amplitudes, int[] owners) -> ()");
  m.def("poles(Tensor geometry, Tensor geometry64, Tensor boxes, Tensor[] planes, Tensor[] centres, Tensor[] queries, Tensor[] widths, Tensor values, Tensor carriers, Tensor seed, Tensor old, Tensor slots, int[] sizes, float[] constants) -> Tensor[]");
  m.def("distance(Tensor geometry, Tensor points) -> Tensor");
  m.def("polygon_distance(Tensor points, Tensor starts, Tensor ends) -> Tensor");
  m.def("coverage(Tensor geometry, Tensor points, Tensor[] widths, int[] shape, float[] constants) -> Tensor");
  m.def("conductor_edges(Tensor distance) -> Tensor[]");
  m.def("bulk(Tensor distances, Tensor values, Tensor band, float background, bool smooth) -> Tensor[]");
  m.def("window(Tensor sample, Tensor offsets, int axis) -> Tensor");
  m.def("reconstruct(Tensor arithmetic, Tensor inverse, Tensor[] gradients, int axis, bool node) -> Tensor");
  m.def("coefficients(Tensor[] material, Tensor[] sigma, Tensor[] profiles, int[] shape, float dt, bool electric_loss, bool magnetic_loss) -> Tensor[]");
  m.def("hash(Tensor[] input, int[] shape) -> Tensor[]");
  m.def("gather(Tensor[] input, int[] shape, Tensor owners) -> Tensor");
  m.def("remap(Tensor slots, Tensor remap, bool wide) -> Tensor");
  m.def("mesh_keys(Tensor triangles, Tensor extent) -> Tensor");
  m.def("mesh_topology(Tensor keys) -> Tensor");
  m.def("mesh_bounds(Tensor triangles, Tensor topology, Tensor ids) -> Tensor");
  m.def("mesh_distance(Tensor points, Tensor triangles, Tensor[] hierarchy) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(witwin_maxwell_setup, CompositeExplicitAutograd, m) {
  m.impl("sample", TORCH_BOX(&sample_setup));
  m.impl("poles", TORCH_BOX(&poles_setup));
  m.impl("distance", TORCH_BOX(&distance_setup));
  m.impl("polygon_distance", TORCH_BOX(&polygon_distance));
  m.impl("coverage", TORCH_BOX(&coverage_setup));
  m.impl("conductor_edges", TORCH_BOX(&conductor_edges));
  m.impl("bulk", TORCH_BOX(&bulk_setup));
  m.impl("window", TORCH_BOX(&window_setup));
  m.impl("reconstruct", TORCH_BOX(&reconstruct_setup));
  m.impl("coefficients", TORCH_BOX(&coefficients_setup));
  m.impl("hash", TORCH_BOX(&hash_setup));
  m.impl("gather", TORCH_BOX(&gather_setup));
  m.impl("remap", TORCH_BOX(&remap_setup));
  m.impl("mesh_keys", TORCH_BOX(&mesh_keys));
  m.impl("mesh_topology", TORCH_BOX(&mesh_topology));
  m.impl("mesh_bounds", TORCH_BOX(&mesh_bounds));
  m.impl("mesh_distance", TORCH_BOX(&mesh_distance));
}
