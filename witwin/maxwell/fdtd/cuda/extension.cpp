#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include <optional>
#include <tuple>
#include <vector>

void synchronize_noop_cuda();
std::tuple<
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor>
debug_linear_indices_cuda(int64_t size_x, int64_t size_y, int64_t size_z);
void update_magnetic_hx_standard_cuda(
    torch::stable::Tensor hx,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz);
void update_magnetic_hy_standard_cuda(
    torch::stable::Tensor hy,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz);
void update_magnetic_hz_standard_cuda(
    torch::stable::Tensor hz,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy);
void update_magnetic_hx_cpml_cuda(
    torch::stable::Tensor hx,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_y,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz);
void update_magnetic_hy_cpml_cuda(
    torch::stable::Tensor hy,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz);
void update_magnetic_hz_cpml_cuda(
    torch::stable::Tensor hz,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_y,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy);
void update_magnetic_hx_cpml_compressed_cuda(
    torch::stable::Tensor hx,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_y,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length,
    std::optional<double> uniform_decay,
    std::optional<double> uniform_curl);
void update_magnetic_hy_cpml_compressed_cuda(
    torch::stable::Tensor hy,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length,
    std::optional<double> uniform_decay,
    std::optional<double> uniform_curl);
void update_magnetic_hz_cpml_compressed_cuda(
    torch::stable::Tensor hz,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_y,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length,
    std::optional<double> uniform_decay,
    std::optional<double> uniform_curl);
void update_electric_ex_standard_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void update_electric_ey_standard_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void update_electric_ez_standard_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode);
void update_electric_ex_modulated_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void update_electric_ey_modulated_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void update_electric_ez_modulated_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode);
void update_electric_ex_cpml_modulated_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    torch::stable::Tensor psi_y,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void update_electric_ey_cpml_modulated_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void update_electric_ez_cpml_modulated_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_y,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode);
void update_electric_ex_cpml_modulated_compressed_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    torch::stable::Tensor psi_y,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode,
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length);
void update_electric_ey_cpml_modulated_compressed_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode,
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length);
void update_electric_ez_cpml_modulated_compressed_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_prev,
    double t_next,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_y,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length);
void update_electric_ex_bloch_cuda(
    torch::stable::Tensor ex_real,
    torch::stable::Tensor ex_imag,
    const torch::stable::Tensor& hy_real,
    const torch::stable::Tensor& hy_imag,
    const torch::stable::Tensor& hz_real,
    const torch::stable::Tensor& hz_imag,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    double phase_cos_y,
    double phase_sin_y,
    double phase_cos_z,
    double phase_sin_z,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz);
void update_electric_ey_bloch_cuda(
    torch::stable::Tensor ey_real,
    torch::stable::Tensor ey_imag,
    const torch::stable::Tensor& hx_real,
    const torch::stable::Tensor& hx_imag,
    const torch::stable::Tensor& hz_real,
    const torch::stable::Tensor& hz_imag,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_z,
    double phase_sin_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz);
void update_electric_ez_bloch_cuda(
    torch::stable::Tensor ez_real,
    torch::stable::Tensor ez_imag,
    const torch::stable::Tensor& hx_real,
    const torch::stable::Tensor& hx_imag,
    const torch::stable::Tensor& hy_real,
    const torch::stable::Tensor& hy_imag,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_y,
    double phase_sin_y,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy);
void update_electric_ex_cpml_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_y,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void update_electric_ey_cpml_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void update_electric_ez_cpml_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_y,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode);
void update_electric_ex_cpml_compressed_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_y,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode,
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length,
    std::optional<double> uniform_decay,
    std::optional<double> uniform_curl);
void update_electric_ey_cpml_compressed_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode,
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t z_low_length,
    int64_t z_high_start,
    int64_t z_high_length,
    std::optional<double> uniform_decay,
    std::optional<double> uniform_curl);
void update_electric_ez_cpml_compressed_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_x,
    torch::stable::Tensor psi_y,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t x_low_length,
    int64_t x_high_start,
    int64_t x_high_length,
    int64_t y_low_length,
    int64_t y_high_start,
    int64_t y_high_length,
    std::optional<double> uniform_decay,
    std::optional<double> uniform_curl);
void accumulate_dft_batched_cuda(
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    torch::stable::Tensor ex_real,
    torch::stable::Tensor ex_imag,
    torch::stable::Tensor ey_real,
    torch::stable::Tensor ey_imag,
    torch::stable::Tensor ez_real,
    torch::stable::Tensor ez_imag,
    const torch::stable::Tensor& weighted_cos,
    const torch::stable::Tensor& weighted_sin);
void accumulate_point_observers_cuda(
    const torch::stable::Tensor& field,
    const torch::stable::Tensor& point_i,
    const torch::stable::Tensor& point_j,
    const torch::stable::Tensor& point_k,
    torch::stable::Tensor real_accum,
    torch::stable::Tensor imag_accum,
    double weighted_cos,
    double weighted_sin);
void accumulate_plane_observer_cuda(
    const torch::stable::Tensor& field,
    torch::stable::Tensor real_accum,
    torch::stable::Tensor imag_accum,
    int64_t axis,
    int64_t plane_index,
    double weighted_cos,
    double weighted_sin);
void plane_flux_reduce_cuda(
    const torch::stable::Tensor& ea,
    const torch::stable::Tensor& eb,
    const torch::stable::Tensor& ha,
    const torch::stable::Tensor& hb,
    const torch::stable::Tensor& weights,
    torch::stable::Tensor out,
    int64_t out_index,
    double scale);
void update_debye_current_cuda(
    const torch::stable::Tensor& electric,
    torch::stable::Tensor polarization,
    torch::stable::Tensor current,
    const torch::stable::Tensor& drive,
    double decay,
    double dt);
void update_drude_current_cuda(
    const torch::stable::Tensor& electric,
    torch::stable::Tensor current,
    const torch::stable::Tensor& drive,
    double decay);
void update_lorentz_current_cuda(
    const torch::stable::Tensor& electric,
    torch::stable::Tensor polarization,
    torch::stable::Tensor current,
    const torch::stable::Tensor& drive,
    double decay,
    double restoring,
    double dt);
void apply_polarization_current_cuda(
    torch::stable::Tensor electric,
    const torch::stable::Tensor& current,
    const torch::stable::Tensor& inv_permittivity,
    double dt);
void apply_polarization_current_modulated_cuda(
    torch::stable::Tensor electric,
    const torch::stable::Tensor& current,
    const torch::stable::Tensor& inv_permittivity,
    const torch::stable::Tensor& mod_cos,
    const torch::stable::Tensor& mod_sin,
    const torch::stable::Tensor& mod_omega,
    double t_next,
    double dt);
void update_kerr_ex_curl_cuda(
    torch::stable::Tensor dynamic_curl,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& linear_permittivity,
    const torch::stable::Tensor& ex_decay,
    const torch::stable::Tensor& chi3,
    double dt,
    double eps0);
void update_kerr_ey_curl_cuda(
    torch::stable::Tensor dynamic_curl,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& linear_permittivity,
    const torch::stable::Tensor& ey_decay,
    const torch::stable::Tensor& chi3,
    double dt,
    double eps0);
void update_kerr_ez_curl_cuda(
    torch::stable::Tensor dynamic_curl,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& linear_permittivity,
    const torch::stable::Tensor& ez_decay,
    const torch::stable::Tensor& chi3,
    double dt,
    double eps0);
void update_nonlinear_coefficients_cuda(
    torch::stable::Tensor dynamic_decay,
    torch::stable::Tensor dynamic_curl,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez,
    const torch::stable::Tensor& linear_permittivity,
    const torch::stable::Tensor& external_decay,
    const torch::stable::Tensor& sigma_static,
    const torch::stable::Tensor& chi2,
    const torch::stable::Tensor& chi3,
    const torch::stable::Tensor& tpa_sigma,
    int64_t component,
    double dt,
    double eps0);
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
    int64_t periodic_z);
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
    int64_t periodic_z);
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
    int64_t periodic_z);
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
    torch::stable::Tensor psi_z);
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
    torch::stable::Tensor psi_z);
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
    torch::stable::Tensor psi_z);
void apply_aniso_offdiag_current_ex_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& jy,
    const torch::stable::Tensor& jz,
    const torch::stable::Tensor& coeff_y,
    const torch::stable::Tensor& coeff_z,
    int64_t periodic_x,
    int64_t periodic_y,
    int64_t periodic_z);
void apply_aniso_offdiag_current_ey_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& jx,
    const torch::stable::Tensor& jz,
    const torch::stable::Tensor& coeff_x,
    const torch::stable::Tensor& coeff_z,
    int64_t periodic_x,
    int64_t periodic_y,
    int64_t periodic_z);
void apply_aniso_offdiag_current_ez_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& jx,
    const torch::stable::Tensor& jy,
    const torch::stable::Tensor& coeff_x,
    const torch::stable::Tensor& coeff_y,
    int64_t periodic_x,
    int64_t periodic_y,
    int64_t periodic_z);
void add_source_patch_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal);
void add_cw_phased_source_patch_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch_cos,
    const torch::stable::Tensor& patch_sin,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal_cos,
    double signal_sin);
void add_time_shifted_source_patch_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch,
    const torch::stable::Tensor& delay_patch,
    const torch::stable::Tensor& activation_delay_patch,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    int64_t time_kind,
    double time,
    double frequency,
    double fwidth,
    double amplitude,
    double phase,
    double delay,
    int64_t causal_gate);
void add_source_patch_periodic_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal,
    int64_t axis_a,
    int64_t axis_b,
    int64_t wrap_a,
    int64_t wrap_b);
void add_source_patch_bloch_cuda(
    torch::stable::Tensor ex_real,
    torch::stable::Tensor ex_imag,
    torch::stable::Tensor ey_real,
    torch::stable::Tensor ey_imag,
    torch::stable::Tensor ez_real,
    torch::stable::Tensor ez_imag,
    const torch::stable::Tensor& patch,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal_real,
    double signal_imag,
    int64_t axis_code,
    double phase_cos_a,
    double phase_sin_a,
    double phase_cos_b,
    double phase_sin_b,
    int64_t wrap_axis_a,
    int64_t wrap_axis_b);
void add_scaled_slice_source_patch_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch,
    const torch::stable::Tensor& incident,
    int64_t sample_index,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double scale);
void add_scaled_line_source_patch_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch,
    const torch::stable::Tensor& incident,
    const torch::stable::Tensor& sample_indices,
    int64_t sample_axis,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double scale);
void add_interpolated_source_patch_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch,
    const torch::stable::Tensor& incident,
    const torch::stable::Tensor& sample_positions,
    double origin,
    double ds,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double scale);
void add_batched_reference_source_patches_cuda(
    torch::stable::Tensor field_x,
    torch::stable::Tensor field_y,
    torch::stable::Tensor field_z,
    const torch::stable::Tensor& coeff_data,
    const torch::stable::Tensor& incident,
    const torch::stable::Tensor& field_codes_per_coeff,
    const torch::stable::Tensor& field_offsets,
    const torch::stable::Tensor& sample_indices_per_coeff);
void add_batched_interpolated_source_patches_cuda(
    torch::stable::Tensor field_x,
    torch::stable::Tensor field_y,
    torch::stable::Tensor field_z,
    const torch::stable::Tensor& coeff_data,
    const torch::stable::Tensor& incident,
    const torch::stable::Tensor& sample_positions,
    const torch::stable::Tensor& field_codes_per_coeff,
    const torch::stable::Tensor& field_offsets,
    double origin,
    double ds);
void update_auxiliary_magnetic_cuda(
    torch::stable::Tensor magnetic,
    const torch::stable::Tensor& electric,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl);
void update_auxiliary_electric_cuda(
    torch::stable::Tensor electric,
    const torch::stable::Tensor& magnetic,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    int64_t source_index,
    double source_value);
void reverse_magnetic_adjoint_decay_cuda(
    torch::stable::Tensor adj_prev,
    const torch::stable::Tensor& adj_mid,
    const torch::stable::Tensor& decay);
void reverse_electric_adjoint_to_hx_standard_cuda(
    torch::stable::Tensor adj_hx_mid,
    const torch::stable::Tensor& adj_hx_post,
    const torch::stable::Tensor& adj_ey_post,
    const torch::stable::Tensor& adj_ez_post,
    const torch::stable::Tensor& ey_curl,
    const torch::stable::Tensor& ez_curl,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz);
void reverse_electric_adjoint_to_hy_standard_cuda(
    torch::stable::Tensor adj_hy_mid,
    const torch::stable::Tensor& adj_hy_post,
    const torch::stable::Tensor& adj_ex_post,
    const torch::stable::Tensor& adj_ez_post,
    const torch::stable::Tensor& ex_curl,
    const torch::stable::Tensor& ez_curl,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz);
void reverse_electric_adjoint_to_hz_standard_cuda(
    torch::stable::Tensor adj_hz_mid,
    const torch::stable::Tensor& adj_hz_post,
    const torch::stable::Tensor& adj_ex_post,
    const torch::stable::Tensor& adj_ey_post,
    const torch::stable::Tensor& ex_curl,
    const torch::stable::Tensor& ey_curl,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy);
void reverse_magnetic_adjoint_to_ex_standard_cuda(
    torch::stable::Tensor adj_ex_prev,
    torch::stable::Tensor grad_eps_ex,
    const torch::stable::Tensor& adj_ex_post,
    const torch::stable::Tensor& adj_hy_mid,
    const torch::stable::Tensor& adj_hz_mid,
    const torch::stable::Tensor& ex_decay,
    const torch::stable::Tensor& ex_curl,
    const torch::stable::Tensor& eps_ex,
    const torch::stable::Tensor& hy_mid,
    const torch::stable::Tensor& hz_mid,
    const torch::stable::Tensor& hy_curl,
    const torch::stable::Tensor& hz_curl,
    const torch::stable::Tensor& inv_dy_e,
    const torch::stable::Tensor& inv_dz_e,
    const torch::stable::Tensor& inv_dy_h,
    const torch::stable::Tensor& inv_dz_h,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void reverse_magnetic_adjoint_to_ey_standard_cuda(
    torch::stable::Tensor adj_ey_prev,
    torch::stable::Tensor grad_eps_ey,
    const torch::stable::Tensor& adj_ey_post,
    const torch::stable::Tensor& adj_hx_mid,
    const torch::stable::Tensor& adj_hz_mid,
    const torch::stable::Tensor& ey_decay,
    const torch::stable::Tensor& ey_curl,
    const torch::stable::Tensor& eps_ey,
    const torch::stable::Tensor& hx_mid,
    const torch::stable::Tensor& hz_mid,
    const torch::stable::Tensor& hx_curl,
    const torch::stable::Tensor& hz_curl,
    const torch::stable::Tensor& inv_dx_e,
    const torch::stable::Tensor& inv_dz_e,
    const torch::stable::Tensor& inv_dx_h,
    const torch::stable::Tensor& inv_dz_h,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void reverse_magnetic_adjoint_to_ez_standard_cuda(
    torch::stable::Tensor adj_ez_prev,
    torch::stable::Tensor grad_eps_ez,
    const torch::stable::Tensor& adj_ez_post,
    const torch::stable::Tensor& adj_hx_mid,
    const torch::stable::Tensor& adj_hy_mid,
    const torch::stable::Tensor& ez_decay,
    const torch::stable::Tensor& ez_curl,
    const torch::stable::Tensor& eps_ez,
    const torch::stable::Tensor& hx_mid,
    const torch::stable::Tensor& hy_mid,
    const torch::stable::Tensor& hx_curl,
    const torch::stable::Tensor& hy_curl,
    const torch::stable::Tensor& inv_dx_e,
    const torch::stable::Tensor& inv_dy_e,
    const torch::stable::Tensor& inv_dx_h,
    const torch::stable::Tensor& inv_dy_h,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode);
void reverse_electric_adjoint_to_hx_bloch_cuda(
    torch::stable::Tensor adj_hx_mid_real,
    torch::stable::Tensor adj_hx_mid_imag,
    const torch::stable::Tensor& adj_hx_post_real,
    const torch::stable::Tensor& adj_hx_post_imag,
    const torch::stable::Tensor& adj_ey_post_real,
    const torch::stable::Tensor& adj_ey_post_imag,
    const torch::stable::Tensor& adj_ez_post_real,
    const torch::stable::Tensor& adj_ez_post_imag,
    const torch::stable::Tensor& ey_curl,
    const torch::stable::Tensor& ez_curl,
    double phase_cos_y,
    double phase_sin_y,
    double phase_cos_z,
    double phase_sin_z,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz);
void reverse_electric_adjoint_to_hy_bloch_cuda(
    torch::stable::Tensor adj_hy_mid_real,
    torch::stable::Tensor adj_hy_mid_imag,
    const torch::stable::Tensor& adj_hy_post_real,
    const torch::stable::Tensor& adj_hy_post_imag,
    const torch::stable::Tensor& adj_ex_post_real,
    const torch::stable::Tensor& adj_ex_post_imag,
    const torch::stable::Tensor& adj_ez_post_real,
    const torch::stable::Tensor& adj_ez_post_imag,
    const torch::stable::Tensor& ex_curl,
    const torch::stable::Tensor& ez_curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_z,
    double phase_sin_z,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz);
void reverse_electric_adjoint_to_hz_bloch_cuda(
    torch::stable::Tensor adj_hz_mid_real,
    torch::stable::Tensor adj_hz_mid_imag,
    const torch::stable::Tensor& adj_hz_post_real,
    const torch::stable::Tensor& adj_hz_post_imag,
    const torch::stable::Tensor& adj_ex_post_real,
    const torch::stable::Tensor& adj_ex_post_imag,
    const torch::stable::Tensor& adj_ey_post_real,
    const torch::stable::Tensor& adj_ey_post_imag,
    const torch::stable::Tensor& ex_curl,
    const torch::stable::Tensor& ey_curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_y,
    double phase_sin_y,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy);
void reverse_magnetic_adjoint_to_ex_bloch_cuda(
    torch::stable::Tensor adj_ex_prev_real,
    torch::stable::Tensor adj_ex_prev_imag,
    torch::stable::Tensor grad_eps_ex,
    const torch::stable::Tensor& adj_ex_post_real,
    const torch::stable::Tensor& adj_ex_post_imag,
    const torch::stable::Tensor& adj_hy_mid_real,
    const torch::stable::Tensor& adj_hy_mid_imag,
    const torch::stable::Tensor& adj_hz_mid_real,
    const torch::stable::Tensor& adj_hz_mid_imag,
    const torch::stable::Tensor& ex_decay,
    const torch::stable::Tensor& ex_curl,
    const torch::stable::Tensor& eps_ex,
    const torch::stable::Tensor& hy_mid_real,
    const torch::stable::Tensor& hy_mid_imag,
    const torch::stable::Tensor& hz_mid_real,
    const torch::stable::Tensor& hz_mid_imag,
    const torch::stable::Tensor& hy_curl,
    const torch::stable::Tensor& hz_curl,
    double phase_cos_y,
    double phase_sin_y,
    double phase_cos_z,
    double phase_sin_z,
    const torch::stable::Tensor& inv_dy_e,
    const torch::stable::Tensor& inv_dz_e,
    const torch::stable::Tensor& inv_dy_h,
    const torch::stable::Tensor& inv_dz_h);
void reverse_magnetic_adjoint_to_ey_bloch_cuda(
    torch::stable::Tensor adj_ey_prev_real,
    torch::stable::Tensor adj_ey_prev_imag,
    torch::stable::Tensor grad_eps_ey,
    const torch::stable::Tensor& adj_ey_post_real,
    const torch::stable::Tensor& adj_ey_post_imag,
    const torch::stable::Tensor& adj_hx_mid_real,
    const torch::stable::Tensor& adj_hx_mid_imag,
    const torch::stable::Tensor& adj_hz_mid_real,
    const torch::stable::Tensor& adj_hz_mid_imag,
    const torch::stable::Tensor& ey_decay,
    const torch::stable::Tensor& ey_curl,
    const torch::stable::Tensor& eps_ey,
    const torch::stable::Tensor& hx_mid_real,
    const torch::stable::Tensor& hx_mid_imag,
    const torch::stable::Tensor& hz_mid_real,
    const torch::stable::Tensor& hz_mid_imag,
    const torch::stable::Tensor& hx_curl,
    const torch::stable::Tensor& hz_curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_z,
    double phase_sin_z,
    const torch::stable::Tensor& inv_dx_e,
    const torch::stable::Tensor& inv_dz_e,
    const torch::stable::Tensor& inv_dx_h,
    const torch::stable::Tensor& inv_dz_h);
void reverse_magnetic_adjoint_to_ez_bloch_cuda(
    torch::stable::Tensor adj_ez_prev_real,
    torch::stable::Tensor adj_ez_prev_imag,
    torch::stable::Tensor grad_eps_ez,
    const torch::stable::Tensor& adj_ez_post_real,
    const torch::stable::Tensor& adj_ez_post_imag,
    const torch::stable::Tensor& adj_hx_mid_real,
    const torch::stable::Tensor& adj_hx_mid_imag,
    const torch::stable::Tensor& adj_hy_mid_real,
    const torch::stable::Tensor& adj_hy_mid_imag,
    const torch::stable::Tensor& ez_decay,
    const torch::stable::Tensor& ez_curl,
    const torch::stable::Tensor& eps_ez,
    const torch::stable::Tensor& hx_mid_real,
    const torch::stable::Tensor& hx_mid_imag,
    const torch::stable::Tensor& hy_mid_real,
    const torch::stable::Tensor& hy_mid_imag,
    const torch::stable::Tensor& hx_curl,
    const torch::stable::Tensor& hy_curl,
    double phase_cos_x,
    double phase_sin_x,
    double phase_cos_y,
    double phase_sin_y,
    const torch::stable::Tensor& inv_dx_e,
    const torch::stable::Tensor& inv_dy_e,
    const torch::stable::Tensor& inv_dx_h,
    const torch::stable::Tensor& inv_dy_h);
void accumulate_forward_diff_adjoint_cuda(
    torch::stable::Tensor field_grad,
    const torch::stable::Tensor& diff_grad,
    int64_t axis,
    const torch::stable::Tensor& inv_delta);
void accumulate_backward_diff_adjoint_cuda(
    torch::stable::Tensor field_grad,
    const torch::stable::Tensor& diff_grad,
    int64_t axis,
    const torch::stable::Tensor& inv_delta);
void reverse_electric_component_ex_cpml_cuda(
    torch::stable::Tensor adj_prev,
    torch::stable::Tensor grad_eps,
    torch::stable::Tensor adj_psi_pos_prev,
    torch::stable::Tensor adj_psi_neg_prev,
    torch::stable::Tensor adj_d_pos,
    torch::stable::Tensor adj_d_neg,
    const torch::stable::Tensor& adj_post,
    const torch::stable::Tensor& adj_psi_pos_post,
    const torch::stable::Tensor& adj_psi_neg_post,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& eps,
    const torch::stable::Tensor& psi_pos,
    const torch::stable::Tensor& psi_neg,
    const torch::stable::Tensor& b_pos,
    const torch::stable::Tensor& c_pos,
    const torch::stable::Tensor& inv_kappa_pos,
    const torch::stable::Tensor& b_neg,
    const torch::stable::Tensor& c_neg,
    const torch::stable::Tensor& inv_kappa_neg,
    const torch::stable::Tensor& hy_mid,
    const torch::stable::Tensor& hz_mid,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void reverse_electric_component_ey_cpml_cuda(
    torch::stable::Tensor adj_prev,
    torch::stable::Tensor grad_eps,
    torch::stable::Tensor adj_psi_pos_prev,
    torch::stable::Tensor adj_psi_neg_prev,
    torch::stable::Tensor adj_d_pos,
    torch::stable::Tensor adj_d_neg,
    const torch::stable::Tensor& adj_post,
    const torch::stable::Tensor& adj_psi_pos_post,
    const torch::stable::Tensor& adj_psi_neg_post,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& eps,
    const torch::stable::Tensor& psi_pos,
    const torch::stable::Tensor& psi_neg,
    const torch::stable::Tensor& b_pos,
    const torch::stable::Tensor& c_pos,
    const torch::stable::Tensor& inv_kappa_pos,
    const torch::stable::Tensor& b_neg,
    const torch::stable::Tensor& c_neg,
    const torch::stable::Tensor& inv_kappa_neg,
    const torch::stable::Tensor& hx_mid,
    const torch::stable::Tensor& hz_mid,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t z_low_mode,
    int64_t z_high_mode);
void reverse_electric_component_ez_cpml_cuda(
    torch::stable::Tensor adj_prev,
    torch::stable::Tensor grad_eps,
    torch::stable::Tensor adj_psi_pos_prev,
    torch::stable::Tensor adj_psi_neg_prev,
    torch::stable::Tensor adj_d_pos,
    torch::stable::Tensor adj_d_neg,
    const torch::stable::Tensor& adj_post,
    const torch::stable::Tensor& adj_psi_pos_post,
    const torch::stable::Tensor& adj_psi_neg_post,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& eps,
    const torch::stable::Tensor& psi_pos,
    const torch::stable::Tensor& psi_neg,
    const torch::stable::Tensor& b_pos,
    const torch::stable::Tensor& c_pos,
    const torch::stable::Tensor& inv_kappa_pos,
    const torch::stable::Tensor& b_neg,
    const torch::stable::Tensor& c_neg,
    const torch::stable::Tensor& inv_kappa_neg,
    const torch::stable::Tensor& hx_mid,
    const torch::stable::Tensor& hy_mid,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dy,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t y_low_mode,
    int64_t y_high_mode);
void reverse_magnetic_component_hx_cpml_cuda(
    torch::stable::Tensor adj_prev,
    torch::stable::Tensor adj_psi_pos_prev,
    torch::stable::Tensor adj_psi_neg_prev,
    torch::stable::Tensor adj_d_pos,
    torch::stable::Tensor adj_d_neg,
    const torch::stable::Tensor& adj_post,
    const torch::stable::Tensor& adj_psi_pos_post,
    const torch::stable::Tensor& adj_psi_neg_post,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& b_pos,
    const torch::stable::Tensor& c_pos,
    const torch::stable::Tensor& inv_kappa_pos,
    const torch::stable::Tensor& b_neg,
    const torch::stable::Tensor& c_neg,
    const torch::stable::Tensor& inv_kappa_neg);
void reverse_magnetic_component_hy_cpml_cuda(
    torch::stable::Tensor adj_prev,
    torch::stable::Tensor adj_psi_pos_prev,
    torch::stable::Tensor adj_psi_neg_prev,
    torch::stable::Tensor adj_d_pos,
    torch::stable::Tensor adj_d_neg,
    const torch::stable::Tensor& adj_post,
    const torch::stable::Tensor& adj_psi_pos_post,
    const torch::stable::Tensor& adj_psi_neg_post,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& b_pos,
    const torch::stable::Tensor& c_pos,
    const torch::stable::Tensor& inv_kappa_pos,
    const torch::stable::Tensor& b_neg,
    const torch::stable::Tensor& c_neg,
    const torch::stable::Tensor& inv_kappa_neg);
void reverse_magnetic_component_hz_cpml_cuda(
    torch::stable::Tensor adj_prev,
    torch::stable::Tensor adj_psi_pos_prev,
    torch::stable::Tensor adj_psi_neg_prev,
    torch::stable::Tensor adj_d_pos,
    torch::stable::Tensor adj_d_neg,
    const torch::stable::Tensor& adj_post,
    const torch::stable::Tensor& adj_psi_pos_post,
    const torch::stable::Tensor& adj_psi_neg_post,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    const torch::stable::Tensor& b_pos,
    const torch::stable::Tensor& c_pos,
    const torch::stable::Tensor& inv_kappa_pos,
    const torch::stable::Tensor& b_neg,
    const torch::stable::Tensor& c_neg,
    const torch::stable::Tensor& inv_kappa_neg);
void reverse_debye_current_cuda(
    torch::stable::Tensor adj_electric_prev,
    torch::stable::Tensor adj_polarization_prev,
    const torch::stable::Tensor& adj_polarization_post,
    const torch::stable::Tensor& adj_current_post,
    const torch::stable::Tensor& drive,
    double decay,
    double dt);
void reverse_drude_current_cuda(
    torch::stable::Tensor adj_electric_prev,
    torch::stable::Tensor adj_current_prev,
    const torch::stable::Tensor& adj_current_post,
    const torch::stable::Tensor& drive,
    double decay);
void reverse_lorentz_current_cuda(
    torch::stable::Tensor adj_electric_prev,
    torch::stable::Tensor adj_polarization_prev,
    torch::stable::Tensor adj_current_prev,
    const torch::stable::Tensor& adj_polarization_post,
    const torch::stable::Tensor& adj_current_post,
    const torch::stable::Tensor& drive,
    double decay,
    double restoring,
    double dt);
void accumulate_tfsf_scalar_sample_adjoint_cuda(
    torch::stable::Tensor adj_aux_field,
    const torch::stable::Tensor& adj_field_patch,
    const torch::stable::Tensor& coeff_patch,
    int64_t sample_index,
    double component_scale);
void accumulate_tfsf_line_sample_adjoint_cuda(
    torch::stable::Tensor adj_aux_field,
    const torch::stable::Tensor& adj_field_patch,
    const torch::stable::Tensor& coeff_patch,
    const torch::stable::Tensor& sample_indices,
    int64_t sample_axis_code,
    double component_scale);
void accumulate_tfsf_interpolated_sample_adjoint_cuda(
    torch::stable::Tensor adj_aux_field,
    const torch::stable::Tensor& adj_field_patch,
    const torch::stable::Tensor& coeff_patch,
    const torch::stable::Tensor& sample_positions,
    double origin,
    double ds,
    double component_scale);
void reverse_tfsf_auxiliary_electric_cuda(
    torch::stable::Tensor adj_electric_prev,
    torch::stable::Tensor adj_magnetic_after,
    const torch::stable::Tensor& adj_electric_post,
    const torch::stable::Tensor& electric_decay,
    const torch::stable::Tensor& electric_curl,
    int64_t source_index);
void reverse_tfsf_auxiliary_magnetic_cuda(
    torch::stable::Tensor adj_electric_prev,
    torch::stable::Tensor adj_magnetic_prev,
    const torch::stable::Tensor& adj_magnetic_after,
    const torch::stable::Tensor& magnetic_decay,
    const torch::stable::Tensor& magnetic_curl);
void clamp_field_face_cuda(torch::stable::Tensor field, int64_t axis, int64_t side);
void clamp_pec_boundary_cuda(torch::stable::Tensor field, int64_t axis_a, int64_t axis_b);
void mur_abc_face_cuda(
    torch::stable::Tensor field,
    int64_t axis,
    int64_t boundary_index,
    int64_t adjacent_index,
    double coef,
    torch::stable::Tensor prev_boundary,
    torch::stable::Tensor prev_adjacent);
void project_periodic_boundary_cuda(torch::stable::Tensor field, int64_t axis);
void project_bloch_boundary_cuda(
    torch::stable::Tensor field_real,
    torch::stable::Tensor field_imag,
    int64_t axis,
    double phase_cos,
    double phase_sin);
void update_electric_ex_bloch_y_standard_z_cuda(
    torch::stable::Tensor ex_real,
    torch::stable::Tensor ex_imag,
    const torch::stable::Tensor& hy_real,
    const torch::stable::Tensor& hy_imag,
    const torch::stable::Tensor& hz_real,
    const torch::stable::Tensor& hz_imag,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    double phase_cos_y,
    double phase_sin_y,
    const torch::stable::Tensor& inv_dy,
    const torch::stable::Tensor& inv_dz,
    int64_t z_low_mode,
    int64_t z_high_mode);
void update_electric_ey_bloch_x_standard_z_cuda(
    torch::stable::Tensor ey_real,
    torch::stable::Tensor ey_imag,
    const torch::stable::Tensor& hx_real,
    const torch::stable::Tensor& hx_imag,
    const torch::stable::Tensor& hz_real,
    const torch::stable::Tensor& hz_imag,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& curl,
    double phase_cos_x,
    double phase_sin_x,
    const torch::stable::Tensor& inv_dx,
    const torch::stable::Tensor& inv_dz,
    int64_t z_low_mode,
    int64_t z_high_mode);
void apply_electric_ex_cpml_z_correction_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dz,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t full_size_y,
    int64_t full_size_z);
void apply_electric_ey_cpml_z_correction_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_z,
    const torch::stable::Tensor& inv_kappa_z,
    const torch::stable::Tensor& b_z,
    const torch::stable::Tensor& c_z,
    const torch::stable::Tensor& inv_dz,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t full_size_x,
    int64_t full_size_z);

void add_source_patch_ex_periodic_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal,
    int64_t wrap_a,
    int64_t wrap_b) {
  add_source_patch_periodic_cuda(
      field, patch, offset_i, offset_j, offset_k, signal, 1, 2, wrap_a, wrap_b);
}

void add_source_patch_ey_periodic_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal,
    int64_t wrap_a,
    int64_t wrap_b) {
  add_source_patch_periodic_cuda(
      field, patch, offset_i, offset_j, offset_k, signal, 0, 2, wrap_a, wrap_b);
}

void add_source_patch_ez_periodic_cuda(
    torch::stable::Tensor field,
    const torch::stable::Tensor& patch,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    double signal,
    int64_t wrap_a,
    int64_t wrap_b) {
  add_source_patch_periodic_cuda(
      field, patch, offset_i, offset_j, offset_k, signal, 0, 1, wrap_a, wrap_b);
}

STABLE_TORCH_LIBRARY(witwin_maxwell_fdtd_cuda, m) {
  m.def("synchronize_noop() -> ()");
  m.def(
      "debug_linear_indices(int size_x, int size_y, int size_z) -> "
      "(Tensor, Tensor, Tensor, Tensor)");
  m.def("update_magnetic_hx_standard(Tensor(a!) hx, Tensor ey, Tensor ez, Tensor decay, Tensor curl, Tensor inv_dy, Tensor inv_dz) -> ()");
  m.def("update_magnetic_hy_standard(Tensor(a!) hy, Tensor ex, Tensor ez, Tensor decay, Tensor curl, Tensor inv_dx, Tensor inv_dz) -> ()");
  m.def("update_magnetic_hz_standard(Tensor(a!) hz, Tensor ex, Tensor ey, Tensor decay, Tensor curl, Tensor inv_dx, Tensor inv_dy) -> ()");
  m.def("update_magnetic_hx_cpml(Tensor(a!) hx, Tensor ey, Tensor ez, Tensor decay, Tensor curl, Tensor(b!) psi_y, Tensor(c!) psi_z, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dy, Tensor inv_dz) -> ()");
  m.def("update_magnetic_hy_cpml(Tensor(a!) hy, Tensor ex, Tensor ez, Tensor decay, Tensor curl, Tensor(b!) psi_x, Tensor(c!) psi_z, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dx, Tensor inv_dz) -> ()");
  m.def("update_magnetic_hz_cpml(Tensor(a!) hz, Tensor ex, Tensor ey, Tensor decay, Tensor curl, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_dx, Tensor inv_dy) -> ()");
  m.def("update_magnetic_hx_cpml_compressed(Tensor(a!) hx, Tensor ey, Tensor ez, Tensor decay, Tensor curl, Tensor(b!) psi_y, Tensor(c!) psi_z, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dy, Tensor inv_dz, int y_low_length, int y_high_start, int y_high_length, int z_low_length, int z_high_start, int z_high_length, float? uniform_decay, float? uniform_curl) -> ()");
  m.def("update_magnetic_hy_cpml_compressed(Tensor(a!) hy, Tensor ex, Tensor ez, Tensor decay, Tensor curl, Tensor(b!) psi_x, Tensor(c!) psi_z, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dx, Tensor inv_dz, int x_low_length, int x_high_start, int x_high_length, int z_low_length, int z_high_start, int z_high_length, float? uniform_decay, float? uniform_curl) -> ()");
  m.def("update_magnetic_hz_cpml_compressed(Tensor(a!) hz, Tensor ex, Tensor ey, Tensor decay, Tensor curl, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_dx, Tensor inv_dy, int x_low_length, int x_high_start, int x_high_length, int y_low_length, int y_high_start, int y_high_length, float? uniform_decay, float? uniform_curl) -> ()");
  m.def("update_electric_ex_standard(Tensor(a!) ex, Tensor hy, Tensor hz, Tensor decay, Tensor curl, Tensor inv_dy, Tensor inv_dz, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode) -> ()");
  m.def("update_electric_ey_standard(Tensor(a!) ey, Tensor hx, Tensor hz, Tensor decay, Tensor curl, Tensor inv_dx, Tensor inv_dz, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode) -> ()");
  m.def("update_electric_ez_standard(Tensor(a!) ez, Tensor hx, Tensor hy, Tensor decay, Tensor curl, Tensor inv_dx, Tensor inv_dy, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode) -> ()");
  m.def("update_electric_ex_modulated(Tensor(a!) ex, Tensor hy, Tensor hz, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor inv_dy, Tensor inv_dz, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode) -> ()");
  m.def("update_electric_ey_modulated(Tensor(a!) ey, Tensor hx, Tensor hz, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor inv_dx, Tensor inv_dz, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode) -> ()");
  m.def("update_electric_ez_modulated(Tensor(a!) ez, Tensor hx, Tensor hy, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor inv_dx, Tensor inv_dy, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode) -> ()");
  m.def("update_electric_ex_cpml_modulated(Tensor(a!) ex, Tensor hy, Tensor hz, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor(b!) psi_y, Tensor(c!) psi_z, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dy, Tensor inv_dz, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode) -> ()");
  m.def("update_electric_ey_cpml_modulated(Tensor(a!) ey, Tensor hx, Tensor hz, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor(b!) psi_x, Tensor(c!) psi_z, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dx, Tensor inv_dz, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode) -> ()");
  m.def("update_electric_ez_cpml_modulated(Tensor(a!) ez, Tensor hx, Tensor hy, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_dx, Tensor inv_dy, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode) -> ()");
  m.def("update_electric_ex_bloch(Tensor(a!) ex_real, Tensor(b!) ex_imag, Tensor hy_real, Tensor hy_imag, Tensor hz_real, Tensor hz_imag, Tensor decay, Tensor curl, float phase_cos_y, float phase_sin_y, float phase_cos_z, float phase_sin_z, Tensor inv_dy, Tensor inv_dz) -> ()");
  m.def("update_electric_ey_bloch(Tensor(a!) ey_real, Tensor(b!) ey_imag, Tensor hx_real, Tensor hx_imag, Tensor hz_real, Tensor hz_imag, Tensor decay, Tensor curl, float phase_cos_x, float phase_sin_x, float phase_cos_z, float phase_sin_z, Tensor inv_dx, Tensor inv_dz) -> ()");
  m.def("update_electric_ez_bloch(Tensor(a!) ez_real, Tensor(b!) ez_imag, Tensor hx_real, Tensor hx_imag, Tensor hy_real, Tensor hy_imag, Tensor decay, Tensor curl, float phase_cos_x, float phase_sin_x, float phase_cos_y, float phase_sin_y, Tensor inv_dx, Tensor inv_dy) -> ()");
  m.def("update_electric_ex_cpml(Tensor(a!) ex, Tensor hy, Tensor hz, Tensor decay, Tensor curl, Tensor(b!) psi_y, Tensor(c!) psi_z, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dy, Tensor inv_dz, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode) -> ()");
  m.def("update_electric_ey_cpml(Tensor(a!) ey, Tensor hx, Tensor hz, Tensor decay, Tensor curl, Tensor(b!) psi_x, Tensor(c!) psi_z, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dx, Tensor inv_dz, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode) -> ()");
  m.def("update_electric_ez_cpml(Tensor(a!) ez, Tensor hx, Tensor hy, Tensor decay, Tensor curl, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_dx, Tensor inv_dy, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode) -> ()");
  m.def("update_electric_ex_cpml_compressed(Tensor(a!) ex, Tensor hy, Tensor hz, Tensor decay, Tensor curl, Tensor(b!) psi_y, Tensor(c!) psi_z, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dy, Tensor inv_dz, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode, int y_low_length, int y_high_start, int y_high_length, int z_low_length, int z_high_start, int z_high_length, float? uniform_decay, float? uniform_curl) -> ()");
  m.def("update_electric_ey_cpml_compressed(Tensor(a!) ey, Tensor hx, Tensor hz, Tensor decay, Tensor curl, Tensor(b!) psi_x, Tensor(c!) psi_z, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dx, Tensor inv_dz, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode, int x_low_length, int x_high_start, int x_high_length, int z_low_length, int z_high_start, int z_high_length, float? uniform_decay, float? uniform_curl) -> ()");
  m.def("update_electric_ez_cpml_compressed(Tensor(a!) ez, Tensor hx, Tensor hy, Tensor decay, Tensor curl, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_dx, Tensor inv_dy, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode, int x_low_length, int x_high_start, int x_high_length, int y_low_length, int y_high_start, int y_high_length, float? uniform_decay, float? uniform_curl) -> ()");
  m.def("update_electric_ex_cpml_modulated_compressed(Tensor(a!) ex, Tensor hy, Tensor hz, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor(b!) psi_y, Tensor(c!) psi_z, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dy, Tensor inv_dz, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode, int y_low_length, int y_high_start, int y_high_length, int z_low_length, int z_high_start, int z_high_length) -> ()");
  m.def("update_electric_ey_cpml_modulated_compressed(Tensor(a!) ey, Tensor hx, Tensor hz, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor(b!) psi_x, Tensor(c!) psi_z, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dx, Tensor inv_dz, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode, int x_low_length, int x_high_start, int x_high_length, int z_low_length, int z_high_start, int z_high_length) -> ()");
  m.def("update_electric_ez_cpml_modulated_compressed(Tensor(a!) ez, Tensor hx, Tensor hy, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_dx, Tensor inv_dy, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode, int x_low_length, int x_high_start, int x_high_length, int y_low_length, int y_high_start, int y_high_length) -> ()");
  m.def("accumulate_dft_batched(Tensor ex, Tensor ey, Tensor ez, Tensor(a!) ex_real, Tensor(b!) ex_imag, Tensor(c!) ey_real, Tensor(d!) ey_imag, Tensor(e!) ez_real, Tensor(f!) ez_imag, Tensor weighted_cos, Tensor weighted_sin) -> ()");
  m.def("accumulate_point_observers(Tensor field, Tensor point_i, Tensor point_j, Tensor point_k, Tensor(a!) real_accum, Tensor(b!) imag_accum, float weighted_cos, float weighted_sin) -> ()");
  m.def("accumulate_plane_observer(Tensor field, Tensor(a!) real_accum, Tensor(b!) imag_accum, int axis, int plane_index, float weighted_cos, float weighted_sin) -> ()");
  m.def("plane_flux_reduce(Tensor ea, Tensor eb, Tensor ha, Tensor hb, Tensor weights, Tensor(a!) out, int out_index, float scale) -> ()");
  m.def("update_debye_current(Tensor electric, Tensor(a!) polarization, Tensor(b!) current, Tensor drive, float decay, float dt) -> ()");
  m.def("update_drude_current(Tensor electric, Tensor(a!) current, Tensor drive, float decay) -> ()");
  m.def("update_lorentz_current(Tensor electric, Tensor(a!) polarization, Tensor(b!) current, Tensor drive, float decay, float restoring, float dt) -> ()");
  m.def("apply_polarization_current(Tensor(a!) electric, Tensor current, Tensor inv_permittivity, float dt) -> ()");
  m.def("apply_polarization_current_modulated(Tensor(a!) electric, Tensor current, Tensor inv_permittivity, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_next, float dt) -> ()");
  m.def("update_kerr_ex_curl(Tensor(a!) dynamic_curl, Tensor ex, Tensor ey, Tensor ez, Tensor linear_permittivity, Tensor ex_decay, Tensor chi3, float dt, float eps0) -> ()");
  m.def("update_kerr_ey_curl(Tensor(a!) dynamic_curl, Tensor ex, Tensor ey, Tensor ez, Tensor linear_permittivity, Tensor ey_decay, Tensor chi3, float dt, float eps0) -> ()");
  m.def("update_kerr_ez_curl(Tensor(a!) dynamic_curl, Tensor ex, Tensor ey, Tensor ez, Tensor linear_permittivity, Tensor ez_decay, Tensor chi3, float dt, float eps0) -> ()");
  m.def("update_nonlinear_coefficients(Tensor(a!) dynamic_decay, Tensor(b!) dynamic_curl, Tensor ex, Tensor ey, Tensor ez, Tensor linear_permittivity, Tensor external_decay, Tensor sigma_static, Tensor chi2, Tensor chi3, Tensor tpa_sigma, int component, float dt, float eps0) -> ()");
  m.def("update_electric_ex_full_aniso(Tensor(a!) ex, Tensor hx, Tensor hy, Tensor hz, Tensor coeff_y, Tensor coeff_z, Tensor inv_dx, Tensor inv_dy, Tensor inv_dz, int periodic_x, int periodic_y, int periodic_z) -> ()");
  m.def("update_electric_ey_full_aniso(Tensor(a!) ey, Tensor hx, Tensor hy, Tensor hz, Tensor coeff_x, Tensor coeff_z, Tensor inv_dx, Tensor inv_dy, Tensor inv_dz, int periodic_x, int periodic_y, int periodic_z) -> ()");
  m.def("update_electric_ez_full_aniso(Tensor(a!) ez, Tensor hx, Tensor hy, Tensor hz, Tensor coeff_x, Tensor coeff_y, Tensor inv_dx, Tensor inv_dy, Tensor inv_dz, int periodic_x, int periodic_y, int periodic_z) -> ()");
  m.def("update_electric_ex_full_aniso_cpml(Tensor(a!) ex, Tensor hx, Tensor hy, Tensor hz, Tensor coeff_y, Tensor coeff_z, Tensor inv_dx, Tensor inv_dy, Tensor inv_dz, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, int periodic_x, int periodic_y, int periodic_z, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor(d!) psi_z) -> ()");
  m.def("update_electric_ey_full_aniso_cpml(Tensor(a!) ey, Tensor hx, Tensor hy, Tensor hz, Tensor coeff_x, Tensor coeff_z, Tensor inv_dx, Tensor inv_dy, Tensor inv_dz, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, int periodic_x, int periodic_y, int periodic_z, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor(d!) psi_z) -> ()");
  m.def("update_electric_ez_full_aniso_cpml(Tensor(a!) ez, Tensor hx, Tensor hy, Tensor hz, Tensor coeff_x, Tensor coeff_y, Tensor inv_dx, Tensor inv_dy, Tensor inv_dz, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, int periodic_x, int periodic_y, int periodic_z, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor(d!) psi_z) -> ()");
  m.def("apply_aniso_offdiag_current_ex(Tensor(a!) ex, Tensor jy, Tensor jz, Tensor coeff_y, Tensor coeff_z, int periodic_x, int periodic_y, int periodic_z) -> ()");
  m.def("apply_aniso_offdiag_current_ey(Tensor(a!) ey, Tensor jx, Tensor jz, Tensor coeff_x, Tensor coeff_z, int periodic_x, int periodic_y, int periodic_z) -> ()");
  m.def("apply_aniso_offdiag_current_ez(Tensor(a!) ez, Tensor jx, Tensor jy, Tensor coeff_x, Tensor coeff_y, int periodic_x, int periodic_y, int periodic_z) -> ()");
  m.def("add_source_patch(Tensor(a!) field, Tensor patch, int offset_i, int offset_j, int offset_k, float signal) -> ()");
  m.def("add_cw_phased_source_patch(Tensor(a!) field, Tensor patch_cos, Tensor patch_sin, int offset_i, int offset_j, int offset_k, float signal_cos, float signal_sin) -> ()");
  m.def("add_time_shifted_source_patch(Tensor(a!) field, Tensor patch, Tensor delay_patch, Tensor activation_delay_patch, int offset_i, int offset_j, int offset_k, int time_kind, float time, float frequency, float fwidth, float amplitude, float phase, float delay, int causal_gate) -> ()");
  m.def("add_source_patch_bloch(Tensor(a!) ex_real, Tensor(b!) ex_imag, Tensor(c!) ey_real, Tensor(d!) ey_imag, Tensor(e!) ez_real, Tensor(f!) ez_imag, Tensor patch, int offset_i, int offset_j, int offset_k, float signal_real, float signal_imag, int axis_code, float phase_cos_a, float phase_sin_a, float phase_cos_b, float phase_sin_b, int wrap_axis_a, int wrap_axis_b) -> ()");
  m.def("add_scaled_slice_source_patch(Tensor(a!) field, Tensor patch, Tensor incident, int sample_index, int offset_i, int offset_j, int offset_k, float scale) -> ()");
  m.def("add_scaled_line_source_patch(Tensor(a!) field, Tensor patch, Tensor incident, Tensor sample_indices, int sample_axis, int offset_i, int offset_j, int offset_k, float scale) -> ()");
  m.def("add_interpolated_source_patch(Tensor(a!) field, Tensor patch, Tensor incident, Tensor sample_positions, float origin, float ds, int offset_i, int offset_j, int offset_k, float scale) -> ()");
  m.def("add_batched_reference_source_patches(Tensor(a!) field_x, Tensor(b!) field_y, Tensor(c!) field_z, Tensor coeff_data, Tensor incident, Tensor field_codes_per_coeff, Tensor field_offsets, Tensor sample_indices_per_coeff) -> ()");
  m.def("add_batched_interpolated_source_patches(Tensor(a!) field_x, Tensor(b!) field_y, Tensor(c!) field_z, Tensor coeff_data, Tensor incident, Tensor sample_positions, Tensor field_codes_per_coeff, Tensor field_offsets, float origin, float ds) -> ()");
  m.def("update_auxiliary_magnetic(Tensor(a!) magnetic, Tensor electric, Tensor decay, Tensor curl) -> ()");
  m.def("update_auxiliary_electric(Tensor(a!) electric, Tensor magnetic, Tensor decay, Tensor curl, int source_index, float source_value) -> ()");
  m.def("reverse_magnetic_adjoint_decay(Tensor(a!) adj_prev, Tensor adj_mid, Tensor decay) -> ()");
  m.def("reverse_electric_adjoint_to_hx_standard(Tensor(a!) adj_hx_mid, Tensor adj_hx_post, Tensor adj_ey_post, Tensor adj_ez_post, Tensor ey_curl, Tensor ez_curl, Tensor inv_dy, Tensor inv_dz) -> ()");
  m.def("reverse_electric_adjoint_to_hy_standard(Tensor(a!) adj_hy_mid, Tensor adj_hy_post, Tensor adj_ex_post, Tensor adj_ez_post, Tensor ex_curl, Tensor ez_curl, Tensor inv_dx, Tensor inv_dz) -> ()");
  m.def("reverse_electric_adjoint_to_hz_standard(Tensor(a!) adj_hz_mid, Tensor adj_hz_post, Tensor adj_ex_post, Tensor adj_ey_post, Tensor ex_curl, Tensor ey_curl, Tensor inv_dx, Tensor inv_dy) -> ()");
  m.def("reverse_magnetic_adjoint_to_ex_standard(Tensor(a!) adj_ex_prev, Tensor(b!) grad_eps_ex, Tensor adj_ex_post, Tensor adj_hy_mid, Tensor adj_hz_mid, Tensor ex_decay, Tensor ex_curl, Tensor eps_ex, Tensor hy_mid, Tensor hz_mid, Tensor hy_curl, Tensor hz_curl, Tensor inv_dy_e, Tensor inv_dz_e, Tensor inv_dy_h, Tensor inv_dz_h, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode) -> ()");
  m.def("reverse_magnetic_adjoint_to_ey_standard(Tensor(a!) adj_ey_prev, Tensor(b!) grad_eps_ey, Tensor adj_ey_post, Tensor adj_hx_mid, Tensor adj_hz_mid, Tensor ey_decay, Tensor ey_curl, Tensor eps_ey, Tensor hx_mid, Tensor hz_mid, Tensor hx_curl, Tensor hz_curl, Tensor inv_dx_e, Tensor inv_dz_e, Tensor inv_dx_h, Tensor inv_dz_h, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode) -> ()");
  m.def("reverse_magnetic_adjoint_to_ez_standard(Tensor(a!) adj_ez_prev, Tensor(b!) grad_eps_ez, Tensor adj_ez_post, Tensor adj_hx_mid, Tensor adj_hy_mid, Tensor ez_decay, Tensor ez_curl, Tensor eps_ez, Tensor hx_mid, Tensor hy_mid, Tensor hx_curl, Tensor hy_curl, Tensor inv_dx_e, Tensor inv_dy_e, Tensor inv_dx_h, Tensor inv_dy_h, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode) -> ()");
  m.def("reverse_electric_adjoint_to_hx_bloch(Tensor(a!) adj_hx_mid_real, Tensor(b!) adj_hx_mid_imag, Tensor adj_hx_post_real, Tensor adj_hx_post_imag, Tensor adj_ey_post_real, Tensor adj_ey_post_imag, Tensor adj_ez_post_real, Tensor adj_ez_post_imag, Tensor ey_curl, Tensor ez_curl, float phase_cos_y, float phase_sin_y, float phase_cos_z, float phase_sin_z, Tensor inv_dy, Tensor inv_dz) -> ()");
  m.def("reverse_electric_adjoint_to_hy_bloch(Tensor(a!) adj_hy_mid_real, Tensor(b!) adj_hy_mid_imag, Tensor adj_hy_post_real, Tensor adj_hy_post_imag, Tensor adj_ex_post_real, Tensor adj_ex_post_imag, Tensor adj_ez_post_real, Tensor adj_ez_post_imag, Tensor ex_curl, Tensor ez_curl, float phase_cos_x, float phase_sin_x, float phase_cos_z, float phase_sin_z, Tensor inv_dx, Tensor inv_dz) -> ()");
  m.def("reverse_electric_adjoint_to_hz_bloch(Tensor(a!) adj_hz_mid_real, Tensor(b!) adj_hz_mid_imag, Tensor adj_hz_post_real, Tensor adj_hz_post_imag, Tensor adj_ex_post_real, Tensor adj_ex_post_imag, Tensor adj_ey_post_real, Tensor adj_ey_post_imag, Tensor ex_curl, Tensor ey_curl, float phase_cos_x, float phase_sin_x, float phase_cos_y, float phase_sin_y, Tensor inv_dx, Tensor inv_dy) -> ()");
  m.def("reverse_magnetic_adjoint_to_ex_bloch(Tensor(a!) adj_ex_prev_real, Tensor(b!) adj_ex_prev_imag, Tensor(c!) grad_eps_ex, Tensor adj_ex_post_real, Tensor adj_ex_post_imag, Tensor adj_hy_mid_real, Tensor adj_hy_mid_imag, Tensor adj_hz_mid_real, Tensor adj_hz_mid_imag, Tensor ex_decay, Tensor ex_curl, Tensor eps_ex, Tensor hy_mid_real, Tensor hy_mid_imag, Tensor hz_mid_real, Tensor hz_mid_imag, Tensor hy_curl, Tensor hz_curl, float phase_cos_y, float phase_sin_y, float phase_cos_z, float phase_sin_z, Tensor inv_dy_e, Tensor inv_dz_e, Tensor inv_dy_h, Tensor inv_dz_h) -> ()");
  m.def("reverse_magnetic_adjoint_to_ey_bloch(Tensor(a!) adj_ey_prev_real, Tensor(b!) adj_ey_prev_imag, Tensor(c!) grad_eps_ey, Tensor adj_ey_post_real, Tensor adj_ey_post_imag, Tensor adj_hx_mid_real, Tensor adj_hx_mid_imag, Tensor adj_hz_mid_real, Tensor adj_hz_mid_imag, Tensor ey_decay, Tensor ey_curl, Tensor eps_ey, Tensor hx_mid_real, Tensor hx_mid_imag, Tensor hz_mid_real, Tensor hz_mid_imag, Tensor hx_curl, Tensor hz_curl, float phase_cos_x, float phase_sin_x, float phase_cos_z, float phase_sin_z, Tensor inv_dx_e, Tensor inv_dz_e, Tensor inv_dx_h, Tensor inv_dz_h) -> ()");
  m.def("reverse_magnetic_adjoint_to_ez_bloch(Tensor(a!) adj_ez_prev_real, Tensor(b!) adj_ez_prev_imag, Tensor(c!) grad_eps_ez, Tensor adj_ez_post_real, Tensor adj_ez_post_imag, Tensor adj_hx_mid_real, Tensor adj_hx_mid_imag, Tensor adj_hy_mid_real, Tensor adj_hy_mid_imag, Tensor ez_decay, Tensor ez_curl, Tensor eps_ez, Tensor hx_mid_real, Tensor hx_mid_imag, Tensor hy_mid_real, Tensor hy_mid_imag, Tensor hx_curl, Tensor hy_curl, float phase_cos_x, float phase_sin_x, float phase_cos_y, float phase_sin_y, Tensor inv_dx_e, Tensor inv_dy_e, Tensor inv_dx_h, Tensor inv_dy_h) -> ()");
  m.def("accumulate_forward_diff_adjoint(Tensor(a!) field_grad, Tensor diff_grad, int axis, Tensor inv_delta) -> ()");
  m.def("accumulate_backward_diff_adjoint(Tensor(a!) field_grad, Tensor diff_grad, int axis, Tensor inv_delta) -> ()");
  m.def("reverse_electric_component_ex_cpml(Tensor(a!) adj_prev, Tensor(b!) grad_eps, Tensor(c!) adj_psi_pos_prev, Tensor(d!) adj_psi_neg_prev, Tensor(e!) adj_d_pos, Tensor(f!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor curl, Tensor eps, Tensor psi_pos, Tensor psi_neg, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg, Tensor hy_mid, Tensor hz_mid, Tensor inv_dy, Tensor inv_dz, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode) -> ()");
  m.def("reverse_electric_component_ey_cpml(Tensor(a!) adj_prev, Tensor(b!) grad_eps, Tensor(c!) adj_psi_pos_prev, Tensor(d!) adj_psi_neg_prev, Tensor(e!) adj_d_pos, Tensor(f!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor curl, Tensor eps, Tensor psi_pos, Tensor psi_neg, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg, Tensor hx_mid, Tensor hz_mid, Tensor inv_dx, Tensor inv_dz, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode) -> ()");
  m.def("reverse_electric_component_ez_cpml(Tensor(a!) adj_prev, Tensor(b!) grad_eps, Tensor(c!) adj_psi_pos_prev, Tensor(d!) adj_psi_neg_prev, Tensor(e!) adj_d_pos, Tensor(f!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor curl, Tensor eps, Tensor psi_pos, Tensor psi_neg, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg, Tensor hx_mid, Tensor hy_mid, Tensor inv_dx, Tensor inv_dy, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode) -> ()");
  m.def("reverse_magnetic_component_hx_cpml(Tensor(a!) adj_prev, Tensor(b!) adj_psi_pos_prev, Tensor(c!) adj_psi_neg_prev, Tensor(d!) adj_d_pos, Tensor(e!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor curl, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg) -> ()");
  m.def("reverse_magnetic_component_hy_cpml(Tensor(a!) adj_prev, Tensor(b!) adj_psi_pos_prev, Tensor(c!) adj_psi_neg_prev, Tensor(d!) adj_d_pos, Tensor(e!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor curl, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg) -> ()");
  m.def("reverse_magnetic_component_hz_cpml(Tensor(a!) adj_prev, Tensor(b!) adj_psi_pos_prev, Tensor(c!) adj_psi_neg_prev, Tensor(d!) adj_d_pos, Tensor(e!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor curl, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg) -> ()");
  m.def("reverse_debye_current(Tensor(a!) adj_electric_prev, Tensor(b!) adj_polarization_prev, Tensor adj_polarization_post, Tensor adj_current_post, Tensor drive, float decay, float dt) -> ()");
  m.def("reverse_drude_current(Tensor(a!) adj_electric_prev, Tensor(b!) adj_current_prev, Tensor adj_current_post, Tensor drive, float decay) -> ()");
  m.def("reverse_lorentz_current(Tensor(a!) adj_electric_prev, Tensor(b!) adj_polarization_prev, Tensor(c!) adj_current_prev, Tensor adj_polarization_post, Tensor adj_current_post, Tensor drive, float decay, float restoring, float dt) -> ()");
  m.def("accumulate_tfsf_scalar_sample_adjoint(Tensor(a!) adj_aux_field, Tensor adj_field_patch, Tensor coeff_patch, int sample_index, float component_scale) -> ()");
  m.def("accumulate_tfsf_line_sample_adjoint(Tensor(a!) adj_aux_field, Tensor adj_field_patch, Tensor coeff_patch, Tensor sample_indices, int sample_axis_code, float component_scale) -> ()");
  m.def("accumulate_tfsf_interpolated_sample_adjoint(Tensor(a!) adj_aux_field, Tensor adj_field_patch, Tensor coeff_patch, Tensor sample_positions, float origin, float ds, float component_scale) -> ()");
  m.def("reverse_tfsf_auxiliary_electric(Tensor(a!) adj_electric_prev, Tensor(b!) adj_magnetic_after, Tensor adj_electric_post, Tensor electric_decay, Tensor electric_curl, int source_index) -> ()");
  m.def("reverse_tfsf_auxiliary_magnetic(Tensor(a!) adj_electric_prev, Tensor(b!) adj_magnetic_prev, Tensor adj_magnetic_after, Tensor magnetic_decay, Tensor magnetic_curl) -> ()");
  m.def("clamp_field_face(Tensor(a!) field, int axis, int side) -> ()");
  m.def("clamp_pec_boundary(Tensor(a!) field, int axis_a, int axis_b) -> ()");
  m.def("mur_abc_face(Tensor(a!) field, int axis, int boundary_index, int adjacent_index, float coef, Tensor(b!) prev_boundary, Tensor(c!) prev_adjacent) -> ()");
  m.def("project_periodic_boundary(Tensor(a!) field, int axis) -> ()");
  m.def("project_bloch_boundary(Tensor(a!) field_real, Tensor(b!) field_imag, int axis, float phase_cos, float phase_sin) -> ()");
  m.def("update_electric_ex_bloch_y_standard_z(Tensor(a!) ex_real, Tensor(b!) ex_imag, Tensor hy_real, Tensor hy_imag, Tensor hz_real, Tensor hz_imag, Tensor decay, Tensor curl, float phase_cos_y, float phase_sin_y, Tensor inv_dy, Tensor inv_dz, int z_low_mode, int z_high_mode) -> ()");
  m.def("update_electric_ey_bloch_x_standard_z(Tensor(a!) ey_real, Tensor(b!) ey_imag, Tensor hx_real, Tensor hx_imag, Tensor hz_real, Tensor hz_imag, Tensor decay, Tensor curl, float phase_cos_x, float phase_sin_x, Tensor inv_dx, Tensor inv_dz, int z_low_mode, int z_high_mode) -> ()");
  m.def("apply_electric_ex_cpml_z_correction(Tensor(a!) ex, Tensor hy, Tensor curl, Tensor(b!) psi_z, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dz, int offset_i, int offset_j, int offset_k, int y_low_mode, int y_high_mode, int full_size_y, int full_size_z) -> ()");
  m.def("apply_electric_ey_cpml_z_correction(Tensor(a!) ey, Tensor hx, Tensor curl, Tensor(b!) psi_z, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dz, int offset_i, int offset_j, int offset_k, int x_low_mode, int x_high_mode, int full_size_x, int full_size_z) -> ()");
  m.def("add_source_patch_ex_periodic(Tensor(a!) field, Tensor patch, int offset_i, int offset_j, int offset_k, float signal, int wrap_a, int wrap_b) -> ()");
  m.def("add_source_patch_ey_periodic(Tensor(a!) field, Tensor patch, int offset_i, int offset_j, int offset_k, float signal, int wrap_a, int wrap_b) -> ()");
  m.def("add_source_patch_ez_periodic(Tensor(a!) field, Tensor patch, int offset_i, int offset_j, int offset_k, float signal, int wrap_a, int wrap_b) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(witwin_maxwell_fdtd_cuda, CUDA, m) {
  m.impl("update_magnetic_hx_standard", TORCH_BOX(&update_magnetic_hx_standard_cuda));
  m.impl("update_magnetic_hy_standard", TORCH_BOX(&update_magnetic_hy_standard_cuda));
  m.impl("update_magnetic_hz_standard", TORCH_BOX(&update_magnetic_hz_standard_cuda));
  m.impl("update_magnetic_hx_cpml", TORCH_BOX(&update_magnetic_hx_cpml_cuda));
  m.impl("update_magnetic_hy_cpml", TORCH_BOX(&update_magnetic_hy_cpml_cuda));
  m.impl("update_magnetic_hz_cpml", TORCH_BOX(&update_magnetic_hz_cpml_cuda));
  m.impl("update_magnetic_hx_cpml_compressed", TORCH_BOX(&update_magnetic_hx_cpml_compressed_cuda));
  m.impl("update_magnetic_hy_cpml_compressed", TORCH_BOX(&update_magnetic_hy_cpml_compressed_cuda));
  m.impl("update_magnetic_hz_cpml_compressed", TORCH_BOX(&update_magnetic_hz_cpml_compressed_cuda));
  m.impl("update_electric_ex_standard", TORCH_BOX(&update_electric_ex_standard_cuda));
  m.impl("update_electric_ey_standard", TORCH_BOX(&update_electric_ey_standard_cuda));
  m.impl("update_electric_ez_standard", TORCH_BOX(&update_electric_ez_standard_cuda));
  m.impl("update_electric_ex_modulated", TORCH_BOX(&update_electric_ex_modulated_cuda));
  m.impl("update_electric_ey_modulated", TORCH_BOX(&update_electric_ey_modulated_cuda));
  m.impl("update_electric_ez_modulated", TORCH_BOX(&update_electric_ez_modulated_cuda));
  m.impl("update_electric_ex_cpml_modulated", TORCH_BOX(&update_electric_ex_cpml_modulated_cuda));
  m.impl("update_electric_ey_cpml_modulated", TORCH_BOX(&update_electric_ey_cpml_modulated_cuda));
  m.impl("update_electric_ez_cpml_modulated", TORCH_BOX(&update_electric_ez_cpml_modulated_cuda));
  m.impl("update_electric_ex_bloch", TORCH_BOX(&update_electric_ex_bloch_cuda));
  m.impl("update_electric_ey_bloch", TORCH_BOX(&update_electric_ey_bloch_cuda));
  m.impl("update_electric_ez_bloch", TORCH_BOX(&update_electric_ez_bloch_cuda));
  m.impl("update_electric_ex_cpml", TORCH_BOX(&update_electric_ex_cpml_cuda));
  m.impl("update_electric_ey_cpml", TORCH_BOX(&update_electric_ey_cpml_cuda));
  m.impl("update_electric_ez_cpml", TORCH_BOX(&update_electric_ez_cpml_cuda));
  m.impl("update_electric_ex_cpml_compressed", TORCH_BOX(&update_electric_ex_cpml_compressed_cuda));
  m.impl("update_electric_ey_cpml_compressed", TORCH_BOX(&update_electric_ey_cpml_compressed_cuda));
  m.impl("update_electric_ez_cpml_compressed", TORCH_BOX(&update_electric_ez_cpml_compressed_cuda));
  m.impl("update_electric_ex_cpml_modulated_compressed", TORCH_BOX(&update_electric_ex_cpml_modulated_compressed_cuda));
  m.impl("update_electric_ey_cpml_modulated_compressed", TORCH_BOX(&update_electric_ey_cpml_modulated_compressed_cuda));
  m.impl("update_electric_ez_cpml_modulated_compressed", TORCH_BOX(&update_electric_ez_cpml_modulated_compressed_cuda));
  m.impl("accumulate_dft_batched", TORCH_BOX(&accumulate_dft_batched_cuda));
  m.impl("accumulate_point_observers", TORCH_BOX(&accumulate_point_observers_cuda));
  m.impl("accumulate_plane_observer", TORCH_BOX(&accumulate_plane_observer_cuda));
  m.impl("plane_flux_reduce", TORCH_BOX(&plane_flux_reduce_cuda));
  m.impl("update_debye_current", TORCH_BOX(&update_debye_current_cuda));
  m.impl("update_drude_current", TORCH_BOX(&update_drude_current_cuda));
  m.impl("update_lorentz_current", TORCH_BOX(&update_lorentz_current_cuda));
  m.impl("apply_polarization_current", TORCH_BOX(&apply_polarization_current_cuda));
  m.impl("apply_polarization_current_modulated", TORCH_BOX(&apply_polarization_current_modulated_cuda));
  m.impl("update_kerr_ex_curl", TORCH_BOX(&update_kerr_ex_curl_cuda));
  m.impl("update_kerr_ey_curl", TORCH_BOX(&update_kerr_ey_curl_cuda));
  m.impl("update_kerr_ez_curl", TORCH_BOX(&update_kerr_ez_curl_cuda));
  m.impl("update_nonlinear_coefficients", TORCH_BOX(&update_nonlinear_coefficients_cuda));
  m.impl("update_electric_ex_full_aniso", TORCH_BOX(&update_electric_ex_full_aniso_cuda));
  m.impl("update_electric_ey_full_aniso", TORCH_BOX(&update_electric_ey_full_aniso_cuda));
  m.impl("update_electric_ez_full_aniso", TORCH_BOX(&update_electric_ez_full_aniso_cuda));
  m.impl("update_electric_ex_full_aniso_cpml", TORCH_BOX(&update_electric_ex_full_aniso_cpml_cuda));
  m.impl("update_electric_ey_full_aniso_cpml", TORCH_BOX(&update_electric_ey_full_aniso_cpml_cuda));
  m.impl("update_electric_ez_full_aniso_cpml", TORCH_BOX(&update_electric_ez_full_aniso_cpml_cuda));
  m.impl("apply_aniso_offdiag_current_ex", TORCH_BOX(&apply_aniso_offdiag_current_ex_cuda));
  m.impl("apply_aniso_offdiag_current_ey", TORCH_BOX(&apply_aniso_offdiag_current_ey_cuda));
  m.impl("apply_aniso_offdiag_current_ez", TORCH_BOX(&apply_aniso_offdiag_current_ez_cuda));
  m.impl("add_source_patch", TORCH_BOX(&add_source_patch_cuda));
  m.impl("add_cw_phased_source_patch", TORCH_BOX(&add_cw_phased_source_patch_cuda));
  m.impl("add_time_shifted_source_patch", TORCH_BOX(&add_time_shifted_source_patch_cuda));
  m.impl("add_source_patch_bloch", TORCH_BOX(&add_source_patch_bloch_cuda));
  m.impl("add_scaled_slice_source_patch", TORCH_BOX(&add_scaled_slice_source_patch_cuda));
  m.impl("add_scaled_line_source_patch", TORCH_BOX(&add_scaled_line_source_patch_cuda));
  m.impl("add_interpolated_source_patch", TORCH_BOX(&add_interpolated_source_patch_cuda));
  m.impl("add_batched_reference_source_patches", TORCH_BOX(&add_batched_reference_source_patches_cuda));
  m.impl("add_batched_interpolated_source_patches", TORCH_BOX(&add_batched_interpolated_source_patches_cuda));
  m.impl("update_auxiliary_magnetic", TORCH_BOX(&update_auxiliary_magnetic_cuda));
  m.impl("update_auxiliary_electric", TORCH_BOX(&update_auxiliary_electric_cuda));
  m.impl("reverse_magnetic_adjoint_decay", TORCH_BOX(&reverse_magnetic_adjoint_decay_cuda));
  m.impl("reverse_electric_adjoint_to_hx_standard", TORCH_BOX(&reverse_electric_adjoint_to_hx_standard_cuda));
  m.impl("reverse_electric_adjoint_to_hy_standard", TORCH_BOX(&reverse_electric_adjoint_to_hy_standard_cuda));
  m.impl("reverse_electric_adjoint_to_hz_standard", TORCH_BOX(&reverse_electric_adjoint_to_hz_standard_cuda));
  m.impl("reverse_magnetic_adjoint_to_ex_standard", TORCH_BOX(&reverse_magnetic_adjoint_to_ex_standard_cuda));
  m.impl("reverse_magnetic_adjoint_to_ey_standard", TORCH_BOX(&reverse_magnetic_adjoint_to_ey_standard_cuda));
  m.impl("reverse_magnetic_adjoint_to_ez_standard", TORCH_BOX(&reverse_magnetic_adjoint_to_ez_standard_cuda));
  m.impl("reverse_electric_adjoint_to_hx_bloch", TORCH_BOX(&reverse_electric_adjoint_to_hx_bloch_cuda));
  m.impl("reverse_electric_adjoint_to_hy_bloch", TORCH_BOX(&reverse_electric_adjoint_to_hy_bloch_cuda));
  m.impl("reverse_electric_adjoint_to_hz_bloch", TORCH_BOX(&reverse_electric_adjoint_to_hz_bloch_cuda));
  m.impl("reverse_magnetic_adjoint_to_ex_bloch", TORCH_BOX(&reverse_magnetic_adjoint_to_ex_bloch_cuda));
  m.impl("reverse_magnetic_adjoint_to_ey_bloch", TORCH_BOX(&reverse_magnetic_adjoint_to_ey_bloch_cuda));
  m.impl("reverse_magnetic_adjoint_to_ez_bloch", TORCH_BOX(&reverse_magnetic_adjoint_to_ez_bloch_cuda));
  m.impl("accumulate_forward_diff_adjoint", TORCH_BOX(&accumulate_forward_diff_adjoint_cuda));
  m.impl("accumulate_backward_diff_adjoint", TORCH_BOX(&accumulate_backward_diff_adjoint_cuda));
  m.impl("reverse_electric_component_ex_cpml", TORCH_BOX(&reverse_electric_component_ex_cpml_cuda));
  m.impl("reverse_electric_component_ey_cpml", TORCH_BOX(&reverse_electric_component_ey_cpml_cuda));
  m.impl("reverse_electric_component_ez_cpml", TORCH_BOX(&reverse_electric_component_ez_cpml_cuda));
  m.impl("reverse_magnetic_component_hx_cpml", TORCH_BOX(&reverse_magnetic_component_hx_cpml_cuda));
  m.impl("reverse_magnetic_component_hy_cpml", TORCH_BOX(&reverse_magnetic_component_hy_cpml_cuda));
  m.impl("reverse_magnetic_component_hz_cpml", TORCH_BOX(&reverse_magnetic_component_hz_cpml_cuda));
  m.impl("reverse_debye_current", TORCH_BOX(&reverse_debye_current_cuda));
  m.impl("reverse_drude_current", TORCH_BOX(&reverse_drude_current_cuda));
  m.impl("reverse_lorentz_current", TORCH_BOX(&reverse_lorentz_current_cuda));
  m.impl("accumulate_tfsf_scalar_sample_adjoint", TORCH_BOX(&accumulate_tfsf_scalar_sample_adjoint_cuda));
  m.impl("accumulate_tfsf_line_sample_adjoint", TORCH_BOX(&accumulate_tfsf_line_sample_adjoint_cuda));
  m.impl("accumulate_tfsf_interpolated_sample_adjoint", TORCH_BOX(&accumulate_tfsf_interpolated_sample_adjoint_cuda));
  m.impl("reverse_tfsf_auxiliary_electric", TORCH_BOX(&reverse_tfsf_auxiliary_electric_cuda));
  m.impl("reverse_tfsf_auxiliary_magnetic", TORCH_BOX(&reverse_tfsf_auxiliary_magnetic_cuda));
  m.impl("clamp_field_face", TORCH_BOX(&clamp_field_face_cuda));
  m.impl("clamp_pec_boundary", TORCH_BOX(&clamp_pec_boundary_cuda));
  m.impl("mur_abc_face", TORCH_BOX(&mur_abc_face_cuda));
  m.impl("project_periodic_boundary", TORCH_BOX(&project_periodic_boundary_cuda));
  m.impl("project_bloch_boundary", TORCH_BOX(&project_bloch_boundary_cuda));
  m.impl("update_electric_ex_bloch_y_standard_z", TORCH_BOX(&update_electric_ex_bloch_y_standard_z_cuda));
  m.impl("update_electric_ey_bloch_x_standard_z", TORCH_BOX(&update_electric_ey_bloch_x_standard_z_cuda));
  m.impl("apply_electric_ex_cpml_z_correction", TORCH_BOX(&apply_electric_ex_cpml_z_correction_cuda));
  m.impl("apply_electric_ey_cpml_z_correction", TORCH_BOX(&apply_electric_ey_cpml_z_correction_cuda));
  m.impl("add_source_patch_ex_periodic", TORCH_BOX(&add_source_patch_ex_periodic_cuda));
  m.impl("add_source_patch_ey_periodic", TORCH_BOX(&add_source_patch_ey_periodic_cuda));
  m.impl("add_source_patch_ez_periodic", TORCH_BOX(&add_source_patch_ez_periodic_cuda));
}

STABLE_TORCH_LIBRARY_IMPL(witwin_maxwell_fdtd_cuda, CompositeExplicitAutograd, m) {
  m.impl("synchronize_noop", TORCH_BOX(&synchronize_noop_cuda));
  m.impl("debug_linear_indices", TORCH_BOX(&debug_linear_indices_cuda));
}
