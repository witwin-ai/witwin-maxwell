#include <torch/extension.h>
#include <torch/library.h>
#include <vector>

using at::Tensor;
int64_t layered_cudss_zeroed_allocator(int64_t);
std::vector<int64_t> layered_cudss_zeroed_stats();
Tensor layered_closed_centroids(Tensor);
std::vector<Tensor> layered_closed_mesh(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor);
std::vector<Tensor> layered_contact_mesh(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor);
std::vector<Tensor> layered_admittance_operators(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,
    double,double,double,double,std::vector<int64_t>,int64_t);
std::vector<Tensor> layered_admittance_operators_serial_far(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,
    double,double,double,double,std::vector<int64_t>,int64_t);
std::vector<Tensor> layered_junction_basis(Tensor,Tensor,Tensor);
Tensor layered_junction_gram(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor);
Tensor layered_closed_impedance(Tensor,Tensor,Tensor,Tensor);


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
Tensor layered_unscale(Tensor,Tensor);
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
void layered_assemble_system(std::vector<Tensor>,std::vector<Tensor>,Tensor,Tensor,int64_t,double,bool,int64_t,
                             double,Tensor,Tensor,Tensor);
void layered_loss_fold(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor);
void layered_congruence(Tensor,std::vector<Tensor>,Tensor,int64_t);
std::vector<Tensor> layered_scale_system(Tensor,Tensor,std::vector<Tensor>,Tensor,Tensor,int64_t,double);
std::vector<Tensor> layered_factor_system(Tensor,Tensor);
Tensor layered_sparse_port_work(std::vector<Tensor>,Tensor,Tensor);
std::vector<Tensor> layered_tree_charge(std::vector<Tensor>,Tensor,Tensor,Tensor,Tensor,int64_t);

TORCH_LIBRARY_FRAGMENT(witwin_maxwell_mom_layered,m) {
  // PH1 in-place dense chunk system: one complex128 matrix and one complex64 workspace.
  m.def("assemble_system(Tensor[] green, Tensor[] geometry, Tensor far_rule, Tensor near_rule, int source_tile, "
        "float near_ratio, bool fp64, int mode, float ground, Tensor(a!) system, Tensor(b!) workspace, Tensor order) -> ()");
  m.impl("assemble_system",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_assemble_system)));
  m.def("loss_fold(Tensor(a!) system, Tensor blocks, Tensor columns, Tensor supports, Tensor impedance, "
        "Tensor order, Tensor frequencies) -> ()");
  m.impl("loss_fold",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_loss_fold)));
  m.def("congruence(Tensor(a!) system, Tensor[] columns, Tensor order, int loops) -> ()");
  m.impl("congruence",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_congruence)));
  m.def("scale_system(Tensor(a!) system, Tensor workspace, Tensor[] charge, Tensor rhs, Tensor frequencies, "
        "int loops, float omega0) -> Tensor[]");
  m.impl("scale_system",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_scale_system)));
  m.def("factor_system(Tensor system, Tensor(a!) lu) -> Tensor[]");
  m.impl("factor_system",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_factor_system)));
  m.def("sparse_port_work(Tensor[] columns, Tensor lengths, Tensor excitation) -> Tensor");
  m.impl("sparse_port_work",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_sparse_port_work)));
  m.def("tree_charge(Tensor[] columns, Tensor supports, Tensor edges, Tensor signs, Tensor lengths, int loops) -> Tensor[]");
  m.impl("tree_charge",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_tree_charge)));
  m.def("closed_mesh(Tensor x, Tensor y, Tensor z, Tensor boxes, Tensor bridges, Tensor planes) -> Tensor[]");
  m.impl("closed_mesh",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_closed_mesh)));
  m.def("contact_mesh(Tensor x, Tensor y, Tensor z, Tensor boxes, Tensor bridges, Tensor planes, Tensor groups) -> Tensor[]");
  m.impl("contact_mesh",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_contact_mesh)));
  m.def("admittance_operators(Tensor vertices, Tensor faces, Tensor gauss, Tensor test_faces, Tensor test_slots, "
        "Tensor test_coefficients, Tensor trial_faces, Tensor trial_slots, Tensor trial_coefficients, "
        "float kc_real, float kc_imag, float kb_real, float kb_imag, int[] rules, int tile) -> Tensor[]");
  m.impl("admittance_operators",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_admittance_operators)));
  // Test oracle only (tests/support/layered_admittance_oracle.py): one thread per far pair.
  m.def("admittance_operators_serial_far(Tensor vertices, Tensor faces, Tensor gauss, Tensor test_faces, "
        "Tensor test_slots, Tensor test_coefficients, Tensor trial_faces, Tensor trial_slots, "
        "Tensor trial_coefficients, float kc_real, float kc_imag, float kb_real, float kb_imag, int[] rules, "
        "int tile) -> Tensor[]");
  m.impl("admittance_operators_serial_far",
         torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_admittance_operators_serial_far)));
  m.def("junction_basis(Tensor edges, Tensor face_edges, Tensor counts) -> Tensor[]");
  m.impl("junction_basis",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_junction_basis)));
  m.def("junction_gram(Tensor triangles, Tensor areas, Tensor lengths, Tensor edges, Tensor signs, Tensor local) -> Tensor");
  m.impl("junction_gram",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_junction_gram)));
  m.def("closed_impedance(Tensor frequencies, Tensor sigma, Tensor factors, Tensor owners) -> Tensor");
  m.impl("closed_impedance",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_closed_impedance)));
  m.def("closed_centroids(Tensor triangles) -> Tensor");
  m.impl("closed_centroids",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_closed_centroids)));
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
  m.def("lu_solve(Tensor lu, Tensor pivots, Tensor rhs) -> Tensor");
  m.impl("lu_solve",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_lu_solve)));
  m.def("refinement_check(Tensor residual, Tensor matrix, Tensor solution, Tensor rhs, Tensor info, Tensor(a!) steps, int step, float tolerance) -> Tensor[]");
  m.impl("refinement_check",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_refinement_check)));
  m.def("refinement_update(Tensor(a!) x, Tensor correction, Tensor active) -> ()");
  m.impl("refinement_update",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_refinement_update)));
  m.def("port_admittance(Tensor y, Tensor frequencies) -> Tensor");
  m.impl("port_admittance",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_port_admittance)));
  m.def("unscale(Tensor solution, Tensor recovery) -> Tensor");
  m.impl("unscale",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_unscale)));
  m.def("forest(Tensor plus, Tensor minus, Tensor edges, Tensor signs) -> Tensor[]");
  m.impl("forest",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_forest)));
  m.def("cycles(Tensor plus, Tensor minus, Tensor lengths, Tensor tree, Tensor parents, int loops) -> Tensor");
  m.impl("cycles",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_cycles)));
  m.def("divergence(Tensor q, Tensor edges, Tensor signs, Tensor lengths) -> Tensor");
  m.impl("divergence",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_divergence)));
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
  m.def("cudss_zeroed_allocator", &layered_cudss_zeroed_allocator);
  m.def("cudss_zeroed_stats", &layered_cudss_zeroed_stats);
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

// Matrix-free backend: near list/tiles, far action and deterministic sparse kernels.
std::vector<Tensor> layered_assembly_cache(std::vector<Tensor>,std::vector<Tensor>,Tensor,int64_t,double);
Tensor layered_near_pairs(std::vector<Tensor>,std::vector<Tensor>,Tensor,double,int64_t,double);
Tensor layered_near_tiles(std::vector<Tensor>,std::vector<Tensor>,Tensor,Tensor,double,bool,int64_t,double,Tensor,
                          int64_t,std::vector<Tensor>);
Tensor layered_far_apply(std::vector<Tensor>,std::vector<Tensor>,Tensor,Tensor,double,bool,int64_t,double,int64_t,
                         Tensor,std::vector<Tensor>);
Tensor layered_face_coefficients(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor);
Tensor layered_edge_gather(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor);
Tensor layered_near_potential(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor);
void layered_near_loss(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,double);
Tensor layered_segment_sum(Tensor,Tensor);
Tensor layered_csr_matmul(Tensor,Tensor,Tensor,Tensor);
std::vector<Tensor> layered_cycle_entries(Tensor,Tensor,Tensor,Tensor,Tensor,int64_t);

TORCH_LIBRARY_FRAGMENT(witwin_maxwell_mom_layered,m) {
  m.def("assembly_cache(Tensor[] green, Tensor[] geometry, Tensor far_rule, int mode, float ground) -> Tensor[]");
  m.impl("assembly_cache",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_assembly_cache)));
  m.def("near_pairs(Tensor[] green, Tensor[] geometry, Tensor far_rule, float near_ratio, int mode, float ground) -> Tensor");
  m.impl("near_pairs",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_near_pairs)));
  m.def("near_tiles(Tensor[] green, Tensor[] geometry, Tensor far_rule, Tensor near_rule, float near_ratio, bool fp64, "
        "int mode, float ground, Tensor pairs, int chunk, Tensor[] cache) -> Tensor");
  m.impl("near_tiles",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_near_tiles)));
  m.def("far_apply(Tensor[] green, Tensor[] geometry, Tensor far_rule, Tensor near_rule, float near_ratio, bool fp64, "
        "int mode, float ground, int source_tile, Tensor coefficients, Tensor[] cache) -> Tensor");
  m.impl("far_apply",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_far_apply)));
  m.def("face_coefficients(Tensor u, Tensor supports, Tensor slots, Tensor slopes, Tensor face_edges, "
        "Tensor support_slopes) -> Tensor");
  m.impl("face_coefficients",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_face_coefficients)));
  m.def("edge_gather(Tensor moments, Tensor supports, Tensor slots, Tensor slopes, Tensor face_edges, "
        "Tensor support_slopes) -> Tensor");
  m.impl("edge_gather",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_edge_gather)));
  m.def("near_potential(Tensor rows, Tensor cols, Tensor near_offsets, Tensor near_cols, Tensor tiles, "
        "Tensor supports, Tensor slots, Tensor slopes, Tensor face_edges, Tensor support_slopes) -> Tensor");
  m.impl("near_potential",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_near_potential)));
  m.def("near_loss(Tensor(a!) values, Tensor offsets, Tensor cols, Tensor blocks, Tensor columns, Tensor supports, "
        "Tensor impedance, float frequency) -> ()");
  m.impl("near_loss",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_near_loss)));
  m.def("segment_sum(Tensor values, Tensor offsets) -> Tensor");
  m.impl("segment_sum",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_segment_sum)));
  m.def("csr_matmul(Tensor offsets, Tensor cols, Tensor values, Tensor x) -> Tensor");
  m.impl("csr_matmul",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_csr_matmul)));
  m.def("cycle_entries(Tensor plus, Tensor minus, Tensor lengths, Tensor order, Tensor parents, int loops) -> Tensor[]");
  m.impl("cycle_entries",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_cycle_entries)));
}

// H-matrix backend (hmatrix.cu): compressed far action on face coefficients.
Tensor layered_hmatrix_leaf(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor);
void layered_hmatrix_lowrank(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,int64_t,int64_t);

TORCH_LIBRARY_FRAGMENT(witwin_maxwell_mom_layered,m) {
  m.def("hmatrix_leaf(Tensor tiles, Tensor pairs, Tensor forward_offsets, Tensor forward, Tensor reverse_offsets, "
        "Tensor reverse, Tensor x) -> Tensor");
  m.impl("hmatrix_leaf",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_hmatrix_leaf)));
  m.def("hmatrix_lowrank(Tensor factors, Tensor blocks, Tensor layout, Tensor rank_offsets, Tensor rank_block, "
        "Tensor faces, Tensor entry_offsets, Tensor entries, Tensor x, Tensor(a!) out, int width, int component) -> ()");
  m.impl("hmatrix_lowrank",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(layered_hmatrix_lowrank)));
}
