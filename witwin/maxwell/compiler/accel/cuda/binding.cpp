#include <torch/extension.h>

void sample_setup(std::vector<at::Tensor> outputs, std::vector<at::Tensor> axes,
                  at::Tensor geometry, at::Tensor values, at::Tensor fills,
                  at::Tensor offsets, at::Tensor ids,
                  std::vector<int64_t> sizes, std::vector<double> constants,
                  std::vector<at::Tensor> pole_outputs, std::vector<at::Tensor> amplitudes,
                  std::vector<int64_t> owners);
std::vector<at::Tensor> poles_setup(at::Tensor geometry, at::Tensor geometry64,
    at::Tensor boxes, std::vector<at::Tensor> planes, std::vector<at::Tensor> centres,
    std::vector<at::Tensor> queries, std::vector<at::Tensor> widths,
    at::Tensor values, at::Tensor carriers, at::Tensor seed, at::Tensor old,
    at::Tensor slots, std::vector<int64_t> sizes, std::vector<double> constants);
at::Tensor distance_setup(at::Tensor geometry, at::Tensor points);
at::Tensor polygon_distance(at::Tensor points, at::Tensor starts, at::Tensor ends);
at::Tensor coverage_setup(at::Tensor geometry, at::Tensor points,
    std::vector<at::Tensor> widths, std::vector<int64_t> shape, std::vector<double> constants);
std::vector<at::Tensor> conductor_edges(at::Tensor distance);
std::vector<at::Tensor> bulk_setup(at::Tensor distances, at::Tensor values,
                                 at::Tensor band, double background, bool smooth);
at::Tensor window_setup(at::Tensor sample, at::Tensor offsets, int64_t axis);
at::Tensor reconstruct_setup(at::Tensor arithmetic, at::Tensor inverse,
                             std::vector<at::Tensor> gradients, int64_t axis, bool node);
std::vector<at::Tensor> coefficients_setup(std::vector<at::Tensor> material,
    std::vector<at::Tensor> sigma, std::vector<at::Tensor> profiles,
    std::vector<int64_t> shape, double dt, bool electric_loss, bool magnetic_loss);
std::vector<at::Tensor> hash_setup(std::vector<at::Tensor> input, std::vector<int64_t> shape);
at::Tensor gather_setup(std::vector<at::Tensor> input, std::vector<int64_t> shape, at::Tensor owners);
at::Tensor remap_setup(at::Tensor slots, at::Tensor remap, bool wide);
at::Tensor mesh_keys(at::Tensor triangles, at::Tensor extent);
at::Tensor mesh_topology(at::Tensor keys);
at::Tensor mesh_bounds(at::Tensor triangles, at::Tensor topology, at::Tensor ids);
at::Tensor mesh_distance(at::Tensor points, at::Tensor triangles, std::vector<at::Tensor> hierarchy);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample", &sample_setup);
  m.def("poles", &poles_setup);
  m.def("distance", &distance_setup);
  m.def("polygon_distance", &polygon_distance);
  m.def("coverage", &coverage_setup);
  m.def("conductor_edges", &conductor_edges);
  m.def("bulk", &bulk_setup);
  m.def("window", &window_setup);
  m.def("reconstruct", &reconstruct_setup);
  m.def("coefficients", &coefficients_setup);
  m.def("hash", &hash_setup);
  m.def("gather", &gather_setup);
  m.def("remap", &remap_setup);
  m.def("mesh_keys", &mesh_keys);
  m.def("mesh_topology", &mesh_topology);
  m.def("mesh_bounds", &mesh_bounds);
  m.def("mesh_distance", &mesh_distance);
}
