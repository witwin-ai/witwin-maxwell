#include <torch/extension.h>
#include <torch/library.h>
#include <vector>

using at::Tensor;
std::vector<Tensor> layered_nodes(Tensor, Tensor, Tensor, Tensor, Tensor,
                                double, double, double, int64_t, int64_t, int64_t);
Tensor layered_images(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor);
Tensor layered_group_images(Tensor, Tensor);
void layered_tail_bound(Tensor, Tensor, double, double, Tensor, int64_t, int64_t, int64_t);
void layered_vertical_tail_bound(Tensor,Tensor,double,double,Tensor,int64_t,int64_t,int64_t);
Tensor layered_tail_max(Tensor);
void layered_accumulate(Tensor, Tensor, Tensor);
Tensor layered_peak(Tensor, int64_t, bool);
Tensor layered_special_peak(Tensor, int64_t, int64_t);
Tensor layered_spectral(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                        Tensor, Tensor, double, int64_t, int64_t, bool);
Tensor layered_bessel(Tensor, Tensor, Tensor, Tensor, int64_t, int64_t, bool);
Tensor layered_j0(Tensor);
Tensor layered_bessel_one(Tensor,Tensor,Tensor,Tensor,int64_t,int64_t,bool);
Tensor layered_grid(Tensor, int64_t, double, double);
Tensor layered_tail(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int64_t, int64_t);
void layered_pack(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                  int64_t, int64_t, bool, double, int64_t, double, Tensor);
Tensor layered_slopes(Tensor, Tensor);
Tensor layered_admission(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor);
Tensor layered_evaluate(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                        Tensor, Tensor, Tensor, Tensor, Tensor, int64_t, double, double);
std::vector<Tensor> layered_triangle_static(Tensor, Tensor, Tensor);
Tensor layered_triangle_cross_static(Tensor,Tensor,Tensor);
Tensor layered_assemble(std::vector<Tensor>, std::vector<Tensor>, Tensor, Tensor,
                        int64_t, int64_t, double, bool, int64_t, double);

Tensor layered_assemble_tiles(std::vector<Tensor>,std::vector<Tensor>,Tensor,Tensor,
                             int64_t,int64_t,double,bool,int64_t,double);
Tensor layered_radial_indices(Tensor,Tensor);
std::vector<Tensor> layered_assemble_parts(std::vector<Tensor>, std::vector<Tensor>, Tensor, Tensor,
                        int64_t, int64_t, double, bool, int64_t, double);
std::vector<Tensor> layered_forest(Tensor,Tensor,Tensor,Tensor);
Tensor layered_cycles(Tensor,Tensor,Tensor,Tensor,Tensor,int64_t);
Tensor layered_divergence(Tensor,Tensor,Tensor,Tensor);
std::vector<Tensor> layered_scaled(Tensor,Tensor,Tensor,Tensor,Tensor,int64_t,double);
Tensor layered_loss_matrix(Tensor,Tensor,Tensor,Tensor);
Tensor layered_unscale(Tensor,Tensor);
Tensor layered_port_work(Tensor,Tensor,Tensor);
std::vector<Tensor> layered_lu_factor(Tensor);
Tensor layered_lu_solve(Tensor,Tensor,Tensor);
std::vector<Tensor> layered_refinement_check(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,int64_t,double);
void layered_refinement_update(Tensor,Tensor,Tensor);
Tensor layered_port_admittance(Tensor,Tensor);
Tensor layered_frequency_mm(Tensor,Tensor);
Tensor layered_frequency_residual(Tensor,Tensor,Tensor);
Tensor layered_frequency_norm(Tensor,bool);
Tensor layered_fma_peak(Tensor,int64_t,int64_t);
std::vector<Tensor> layered_sparse_columns(Tensor);
Tensor layered_sparse_product(Tensor,std::vector<Tensor>,bool);

TORCH_LIBRARY_FRAGMENT(witwin_maxwell_mom_layered,m) {
  m.def("sparse_columns(Tensor a) -> Tensor[]");
  m.impl("sparse_columns",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_sparse_columns)));
  m.def("sparse_product(Tensor a, Tensor[] columns, bool left) -> Tensor");
  m.impl("sparse_product",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_sparse_product)));
  m.def("radial_indices(Tensor grid, Tensor queries) -> Tensor");
  m.impl("radial_indices",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_radial_indices)));
  m.def("assemble_tiles(Tensor[] green, Tensor[] geometry, Tensor far_rule, Tensor near_rule, "
        "int frequency, int source_tile, float near_ratio, bool fp64, int mode, float ground) -> Tensor");
  m.impl("assemble_tiles",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_assemble_tiles)));
  m.def("assemble_parts(Tensor[] green, Tensor[] geometry, Tensor far_rule, Tensor near_rule, "
        "int frequency, int source_tile, float near_ratio, bool fp64, int mode, float ground) -> Tensor[]");
  m.impl("assemble_parts",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_assemble_parts)));
  m.def("fma_peak(Tensor anchor, int threads, int iterations) -> Tensor");
  m.impl("fma_peak",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_fma_peak)));
  m.def("frequency_mm(Tensor a, Tensor b) -> Tensor");
  m.impl("frequency_mm",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_frequency_mm)));
  m.def("frequency_residual(Tensor a, Tensor x, Tensor rhs) -> Tensor");
  m.impl("frequency_residual",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_frequency_residual)));
  m.def("frequency_norm(Tensor a, bool matrix) -> Tensor");
  m.impl("frequency_norm",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_frequency_norm)));
  m.def("lu_factor(Tensor a) -> Tensor[]");
  m.impl("lu_factor",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_lu_factor)));
  m.def("lu_solve(Tensor lu, Tensor pivots, Tensor rhs) -> Tensor");
  m.impl("lu_solve",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_lu_solve)));
  m.def("refinement_check(Tensor residual, Tensor matrix, Tensor solution, Tensor rhs, Tensor info, Tensor(a!) steps, int step, float tolerance) -> Tensor[]");
  m.impl("refinement_check",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_refinement_check)));
  m.def("refinement_update(Tensor(a!) x, Tensor correction, Tensor active) -> ()");
  m.impl("refinement_update",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_refinement_update)));
  m.def("port_admittance(Tensor y, Tensor frequencies) -> Tensor");
  m.impl("port_admittance",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_port_admittance)));
  m.def("port_work(Tensor transform, Tensor lengths, Tensor excitation) -> Tensor");
  m.impl("port_work",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_port_work)));
  m.def("unscale(Tensor solution, Tensor recovery) -> Tensor");
  m.impl("unscale",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_unscale)));
  m.def("forest(Tensor plus, Tensor minus, Tensor edges, Tensor signs) -> Tensor[]");
  m.impl("forest",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_forest)));
  m.def("cycles(Tensor plus, Tensor minus, Tensor lengths, Tensor tree, Tensor parents, int loops) -> Tensor");
  m.impl("cycles",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_cycles)));
  m.def("divergence(Tensor q, Tensor edges, Tensor signs, Tensor lengths) -> Tensor");
  m.impl("divergence",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_divergence)));
  m.def("scaled(Tensor magnetic, Tensor charge, Tensor loss, Tensor rhs, Tensor frequencies, int loops, float omega0) -> Tensor[]");
  m.impl("scaled",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_scaled)));
  m.def("loss_matrix(Tensor blocks, Tensor columns, Tensor supports, Tensor impedance) -> Tensor");
  m.impl("loss_matrix",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_loss_matrix)));
  m.def("triangle_static(Tensor triangles, Tensor points, Tensor free) -> Tensor[]");
  m.def("triangle_cross_static(Tensor triangles, Tensor points, Tensor beta) -> Tensor");
  m.impl("triangle_cross_static",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_triangle_cross_static)));
  m.def("assemble(Tensor[] green, Tensor[] geometry, Tensor far_rule, Tensor near_rule, "
        "int frequency, int source_tile, float near_ratio, bool fp64, int mode, float ground) -> Tensor");
  m.impl("triangle_static",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_triangle_static)));
  m.impl("assemble",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_assemble)));
}

Tensor layered_vertical_spectral(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,int64_t,int64_t,bool);
Tensor layered_orient(Tensor,Tensor,Tensor,Tensor);
Tensor layered_horizontal_z_error(std::vector<Tensor>,Tensor,Tensor,Tensor,Tensor);
Tensor layered_horizontal_rho_error(std::vector<Tensor>,Tensor,Tensor,Tensor);

Tensor layered_vertical_coeff(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor);
void layered_vertical_subtract(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor);
void layered_vertical_pack(Tensor,Tensor,Tensor,int64_t);

Tensor layered_vertical_evaluate(std::vector<Tensor>,Tensor,Tensor,Tensor,double);
Tensor layered_vertical_error(Tensor,Tensor);

Tensor layered_vertical_reference(std::vector<Tensor>,Tensor,Tensor,Tensor,Tensor,double);
Tensor layered_vertical_rho_error(std::vector<Tensor>,Tensor,Tensor,Tensor,double);

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.attr("abi")="mom_layered_abi_v1";
  m.def("nodes", &layered_nodes);
  m.def("vertical_spectral", &layered_vertical_spectral, pybind11::arg("media"), pybind11::arg("sigma"),
        pybind11::arg("edges"), pybind11::arg("frequencies"), pybind11::arg("q"),
        pybind11::arg("zs"), pybind11::arg("zo"), pybind11::arg("bottom"), pybind11::arg("top"),
        pybind11::arg("horizontal")=false);
  m.def("orient", &layered_orient);
  m.def("horizontal_z_error", &layered_horizontal_z_error);
  m.def("horizontal_rho_error", &layered_horizontal_rho_error);
  m.def("vertical_coeff", &layered_vertical_coeff);
  m.def("vertical_subtract", &layered_vertical_subtract);
  m.def("vertical_pack", &layered_vertical_pack);
  m.def("vertical_evaluate", &layered_vertical_evaluate);
  m.def("vertical_error", &layered_vertical_error);
  m.def("vertical_reference", &layered_vertical_reference);
  m.def("vertical_rho_error", &layered_vertical_rho_error);
  m.def("images", &layered_images);
  m.def("group_images", &layered_group_images);
  m.def("tail_bound", &layered_tail_bound);
  m.def("vertical_tail_bound", &layered_vertical_tail_bound);
  m.def("tail_max", &layered_tail_max);
  m.def("accumulate", &layered_accumulate);
  m.def("peak", &layered_peak);
  m.def("special_peak", &layered_special_peak);
  m.def("spectral", &layered_spectral);
  m.def("bessel", &layered_bessel);
  m.def("bessel_one", &layered_bessel_one);
  m.def("j0", &layered_j0);
  m.def("grid", &layered_grid);
  m.def("tail", &layered_tail);
  m.def("pack", &layered_pack);
  m.def("slopes", &layered_slopes);
  m.def("admission", &layered_admission);
  m.def("evaluate", &layered_evaluate);
}
