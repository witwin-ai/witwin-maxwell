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
void seed_inject_dense_cuda(
    torch::stable::Tensor adj_field,
    const torch::stable::Tensor& grad_real,
    const torch::stable::Tensor& grad_imag,
    const torch::stable::Tensor& cos_pack,
    const torch::stable::Tensor& sin_pack,
    int64_t step);
void seed_inject_point_cuda(
    torch::stable::Tensor adj_field,
    const torch::stable::Tensor& grad_real,
    const torch::stable::Tensor& grad_imag,
    const torch::stable::Tensor& point_i,
    const torch::stable::Tensor& point_j,
    const torch::stable::Tensor& point_k,
    const torch::stable::Tensor& cos_pack,
    const torch::stable::Tensor& sin_pack,
    int64_t step);
void seed_inject_plane_cuda(
    torch::stable::Tensor adj_field,
    const torch::stable::Tensor& grad_real,
    const torch::stable::Tensor& grad_imag,
    const torch::stable::Tensor& cos_pack,
    const torch::stable::Tensor& sin_pack,
    int64_t axis,
    int64_t plane_index,
    int64_t step);
void accumulate_in_place_cuda(
    torch::stable::Tensor dst,
    const torch::stable::Tensor& src);
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
void reverse_electric_component_ex_cpml_conductive_cuda(
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
    const torch::stable::Tensor& half,
    const torch::stable::Tensor& e_prev,
    const torch::stable::Tensor& eps,
    double dt,
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
void reverse_electric_component_ey_cpml_conductive_cuda(
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
    const torch::stable::Tensor& half,
    const torch::stable::Tensor& e_prev,
    const torch::stable::Tensor& eps,
    double dt,
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
void reverse_electric_component_ez_cpml_conductive_cuda(
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
    const torch::stable::Tensor& half,
    const torch::stable::Tensor& e_prev,
    const torch::stable::Tensor& eps,
    double dt,
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
void collocation_transpose_cuda(
    torch::stable::Tensor adj_ex,
    torch::stable::Tensor adj_ey,
    torch::stable::Tensor adj_ez,
    const torch::stable::Tensor& g_ex,
    const torch::stable::Tensor& g_ey,
    const torch::stable::Tensor& g_ez,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez);
void collocate_field_square_cuda(
    torch::stable::Tensor fsq_ex,
    torch::stable::Tensor fsq_ey,
    torch::stable::Tensor fsq_ez,
    const torch::stable::Tensor& ex,
    const torch::stable::Tensor& ey,
    const torch::stable::Tensor& ez);
void full_aniso_curl_adjoint_cuda(
    torch::stable::Tensor adj_curl_x,
    torch::stable::Tensor adj_curl_y,
    torch::stable::Tensor adj_curl_z,
    const torch::stable::Tensor& adj_ex,
    const torch::stable::Tensor& adj_ey,
    const torch::stable::Tensor& adj_ez,
    const torch::stable::Tensor& coeff_ex_y,
    const torch::stable::Tensor& coeff_ex_z,
    const torch::stable::Tensor& coeff_ey_x,
    const torch::stable::Tensor& coeff_ey_z,
    const torch::stable::Tensor& coeff_ez_x,
    const torch::stable::Tensor& coeff_ez_y);
void reverse_electric_component_ex_cpml_kerr_cuda(
    torch::stable::Tensor adj_prev,
    torch::stable::Tensor grad_eps,
    torch::stable::Tensor grad_chi3,
    torch::stable::Tensor g_fsq,
    torch::stable::Tensor adj_psi_pos_prev,
    torch::stable::Tensor adj_psi_neg_prev,
    torch::stable::Tensor adj_d_pos,
    torch::stable::Tensor adj_d_neg,
    const torch::stable::Tensor& adj_post,
    const torch::stable::Tensor& adj_psi_pos_post,
    const torch::stable::Tensor& adj_psi_neg_post,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& eps,
    const torch::stable::Tensor& chi3,
    const torch::stable::Tensor& fsq,
    double dt,
    double eps0,
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
void reverse_electric_component_ey_cpml_kerr_cuda(
    torch::stable::Tensor adj_prev,
    torch::stable::Tensor grad_eps,
    torch::stable::Tensor grad_chi3,
    torch::stable::Tensor g_fsq,
    torch::stable::Tensor adj_psi_pos_prev,
    torch::stable::Tensor adj_psi_neg_prev,
    torch::stable::Tensor adj_d_pos,
    torch::stable::Tensor adj_d_neg,
    const torch::stable::Tensor& adj_post,
    const torch::stable::Tensor& adj_psi_pos_post,
    const torch::stable::Tensor& adj_psi_neg_post,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& eps,
    const torch::stable::Tensor& chi3,
    const torch::stable::Tensor& fsq,
    double dt,
    double eps0,
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
void reverse_electric_component_ez_cpml_kerr_cuda(
    torch::stable::Tensor adj_prev,
    torch::stable::Tensor grad_eps,
    torch::stable::Tensor grad_chi3,
    torch::stable::Tensor g_fsq,
    torch::stable::Tensor adj_psi_pos_prev,
    torch::stable::Tensor adj_psi_neg_prev,
    torch::stable::Tensor adj_d_pos,
    torch::stable::Tensor adj_d_neg,
    const torch::stable::Tensor& adj_post,
    const torch::stable::Tensor& adj_psi_pos_post,
    const torch::stable::Tensor& adj_psi_neg_post,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& eps,
    const torch::stable::Tensor& chi3,
    const torch::stable::Tensor& fsq,
    double dt,
    double eps0,
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
void reverse_dispersive_correction_cuda(
    torch::stable::Tensor adj_current_corrected,
    torch::stable::Tensor grad_eps,
    const torch::stable::Tensor& adj_current_post,
    const torch::stable::Tensor& adj_electric_post,
    const torch::stable::Tensor& current,
    const torch::stable::Tensor& eps,
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
void apply_electric_ex_cpml_y_correction_cuda(
    torch::stable::Tensor ex,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_y,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_dy,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    int64_t z_low_mode,
    int64_t z_high_mode,
    int64_t full_size_z,
    int64_t full_size_y);
void apply_electric_ez_cpml_y_correction_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hx,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_y,
    const torch::stable::Tensor& inv_kappa_y,
    const torch::stable::Tensor& b_y,
    const torch::stable::Tensor& c_y,
    const torch::stable::Tensor& inv_dy,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    int64_t x_low_mode,
    int64_t x_high_mode,
    int64_t full_size_x,
    int64_t full_size_y);
void apply_electric_ey_cpml_x_correction_cuda(
    torch::stable::Tensor ey,
    const torch::stable::Tensor& hz,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_x,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_dx,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    int64_t z_low_mode,
    int64_t z_high_mode,
    int64_t full_size_z,
    int64_t full_size_x);
void apply_electric_ez_cpml_x_correction_cuda(
    torch::stable::Tensor ez,
    const torch::stable::Tensor& hy,
    const torch::stable::Tensor& curl,
    torch::stable::Tensor psi_x,
    const torch::stable::Tensor& inv_kappa_x,
    const torch::stable::Tensor& b_x,
    const torch::stable::Tensor& c_x,
    const torch::stable::Tensor& inv_dx,
    int64_t offset_i,
    int64_t offset_j,
    int64_t offset_k,
    int64_t y_low_mode,
    int64_t y_high_mode,
    int64_t full_size_y,
    int64_t full_size_x);

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

#define WITWIN_CUDA_OPS(_) \
  _(update_magnetic_hx_standard, "update_magnetic_hx_standard(Tensor(a!) hx, Tensor ey, Tensor ez, Tensor decay, Tensor curl, Tensor inv_dy, Tensor inv_dz) -> ()", update_magnetic_hx_standard_cuda) \
  _(update_magnetic_hy_standard, "update_magnetic_hy_standard(Tensor(a!) hy, Tensor ex, Tensor ez, Tensor decay, Tensor curl, Tensor inv_dx, Tensor inv_dz) -> ()", update_magnetic_hy_standard_cuda) \
  _(update_magnetic_hz_standard, "update_magnetic_hz_standard(Tensor(a!) hz, Tensor ex, Tensor ey, Tensor decay, Tensor curl, Tensor inv_dx, Tensor inv_dy) -> ()", update_magnetic_hz_standard_cuda) \
  _(update_magnetic_hx_cpml, "update_magnetic_hx_cpml(Tensor(a!) hx, Tensor ey, Tensor ez, Tensor decay, Tensor curl, Tensor(b!) psi_y, Tensor(c!) psi_z, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dy, Tensor inv_dz) -> ()", update_magnetic_hx_cpml_cuda) \
  _(update_magnetic_hy_cpml, "update_magnetic_hy_cpml(Tensor(a!) hy, Tensor ex, Tensor ez, Tensor decay, Tensor curl, Tensor(b!) psi_x, Tensor(c!) psi_z, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dx, Tensor inv_dz) -> ()", update_magnetic_hy_cpml_cuda) \
  _(update_magnetic_hz_cpml, "update_magnetic_hz_cpml(Tensor(a!) hz, Tensor ex, Tensor ey, Tensor decay, Tensor curl, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_dx, Tensor inv_dy) -> ()", update_magnetic_hz_cpml_cuda) \
  _(update_magnetic_hx_cpml_compressed, "update_magnetic_hx_cpml_compressed(Tensor(a!) hx, Tensor ey, Tensor ez, Tensor decay, Tensor curl, Tensor(b!) psi_y, Tensor(c!) psi_z, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dy, Tensor inv_dz, int y_low_length, int y_high_start, int y_high_length, int z_low_length, int z_high_start, int z_high_length, float? uniform_decay, float? uniform_curl) -> ()", update_magnetic_hx_cpml_compressed_cuda) \
  _(update_magnetic_hy_cpml_compressed, "update_magnetic_hy_cpml_compressed(Tensor(a!) hy, Tensor ex, Tensor ez, Tensor decay, Tensor curl, Tensor(b!) psi_x, Tensor(c!) psi_z, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dx, Tensor inv_dz, int x_low_length, int x_high_start, int x_high_length, int z_low_length, int z_high_start, int z_high_length, float? uniform_decay, float? uniform_curl) -> ()", update_magnetic_hy_cpml_compressed_cuda) \
  _(update_magnetic_hz_cpml_compressed, "update_magnetic_hz_cpml_compressed(Tensor(a!) hz, Tensor ex, Tensor ey, Tensor decay, Tensor curl, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_dx, Tensor inv_dy, int x_low_length, int x_high_start, int x_high_length, int y_low_length, int y_high_start, int y_high_length, float? uniform_decay, float? uniform_curl) -> ()", update_magnetic_hz_cpml_compressed_cuda) \
  _(update_electric_ex_standard, "update_electric_ex_standard(Tensor(a!) ex, Tensor hy, Tensor hz, Tensor decay, Tensor curl, Tensor inv_dy, Tensor inv_dz, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode) -> ()", update_electric_ex_standard_cuda) \
  _(update_electric_ey_standard, "update_electric_ey_standard(Tensor(a!) ey, Tensor hx, Tensor hz, Tensor decay, Tensor curl, Tensor inv_dx, Tensor inv_dz, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode) -> ()", update_electric_ey_standard_cuda) \
  _(update_electric_ez_standard, "update_electric_ez_standard(Tensor(a!) ez, Tensor hx, Tensor hy, Tensor decay, Tensor curl, Tensor inv_dx, Tensor inv_dy, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode) -> ()", update_electric_ez_standard_cuda) \
  _(update_electric_ex_modulated, "update_electric_ex_modulated(Tensor(a!) ex, Tensor hy, Tensor hz, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor inv_dy, Tensor inv_dz, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode) -> ()", update_electric_ex_modulated_cuda) \
  _(update_electric_ey_modulated, "update_electric_ey_modulated(Tensor(a!) ey, Tensor hx, Tensor hz, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor inv_dx, Tensor inv_dz, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode) -> ()", update_electric_ey_modulated_cuda) \
  _(update_electric_ez_modulated, "update_electric_ez_modulated(Tensor(a!) ez, Tensor hx, Tensor hy, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor inv_dx, Tensor inv_dy, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode) -> ()", update_electric_ez_modulated_cuda) \
  _(update_electric_ex_cpml_modulated, "update_electric_ex_cpml_modulated(Tensor(a!) ex, Tensor hy, Tensor hz, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor(b!) psi_y, Tensor(c!) psi_z, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dy, Tensor inv_dz, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode) -> ()", update_electric_ex_cpml_modulated_cuda) \
  _(update_electric_ey_cpml_modulated, "update_electric_ey_cpml_modulated(Tensor(a!) ey, Tensor hx, Tensor hz, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor(b!) psi_x, Tensor(c!) psi_z, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dx, Tensor inv_dz, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode) -> ()", update_electric_ey_cpml_modulated_cuda) \
  _(update_electric_ez_cpml_modulated, "update_electric_ez_cpml_modulated(Tensor(a!) ez, Tensor hx, Tensor hy, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_dx, Tensor inv_dy, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode) -> ()", update_electric_ez_cpml_modulated_cuda) \
  _(update_electric_ex_bloch, "update_electric_ex_bloch(Tensor(a!) ex_real, Tensor(b!) ex_imag, Tensor hy_real, Tensor hy_imag, Tensor hz_real, Tensor hz_imag, Tensor decay, Tensor curl, float phase_cos_y, float phase_sin_y, float phase_cos_z, float phase_sin_z, Tensor inv_dy, Tensor inv_dz) -> ()", update_electric_ex_bloch_cuda) \
  _(update_electric_ey_bloch, "update_electric_ey_bloch(Tensor(a!) ey_real, Tensor(b!) ey_imag, Tensor hx_real, Tensor hx_imag, Tensor hz_real, Tensor hz_imag, Tensor decay, Tensor curl, float phase_cos_x, float phase_sin_x, float phase_cos_z, float phase_sin_z, Tensor inv_dx, Tensor inv_dz) -> ()", update_electric_ey_bloch_cuda) \
  _(update_electric_ez_bloch, "update_electric_ez_bloch(Tensor(a!) ez_real, Tensor(b!) ez_imag, Tensor hx_real, Tensor hx_imag, Tensor hy_real, Tensor hy_imag, Tensor decay, Tensor curl, float phase_cos_x, float phase_sin_x, float phase_cos_y, float phase_sin_y, Tensor inv_dx, Tensor inv_dy) -> ()", update_electric_ez_bloch_cuda) \
  _(update_electric_ex_cpml, "update_electric_ex_cpml(Tensor(a!) ex, Tensor hy, Tensor hz, Tensor decay, Tensor curl, Tensor(b!) psi_y, Tensor(c!) psi_z, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dy, Tensor inv_dz, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode) -> ()", update_electric_ex_cpml_cuda) \
  _(update_electric_ey_cpml, "update_electric_ey_cpml(Tensor(a!) ey, Tensor hx, Tensor hz, Tensor decay, Tensor curl, Tensor(b!) psi_x, Tensor(c!) psi_z, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dx, Tensor inv_dz, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode) -> ()", update_electric_ey_cpml_cuda) \
  _(update_electric_ez_cpml, "update_electric_ez_cpml(Tensor(a!) ez, Tensor hx, Tensor hy, Tensor decay, Tensor curl, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_dx, Tensor inv_dy, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode) -> ()", update_electric_ez_cpml_cuda) \
  _(update_electric_ex_cpml_compressed, "update_electric_ex_cpml_compressed(Tensor(a!) ex, Tensor hy, Tensor hz, Tensor decay, Tensor curl, Tensor(b!) psi_y, Tensor(c!) psi_z, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dy, Tensor inv_dz, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode, int y_low_length, int y_high_start, int y_high_length, int z_low_length, int z_high_start, int z_high_length, float? uniform_decay, float? uniform_curl) -> ()", update_electric_ex_cpml_compressed_cuda) \
  _(update_electric_ey_cpml_compressed, "update_electric_ey_cpml_compressed(Tensor(a!) ey, Tensor hx, Tensor hz, Tensor decay, Tensor curl, Tensor(b!) psi_x, Tensor(c!) psi_z, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dx, Tensor inv_dz, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode, int x_low_length, int x_high_start, int x_high_length, int z_low_length, int z_high_start, int z_high_length, float? uniform_decay, float? uniform_curl) -> ()", update_electric_ey_cpml_compressed_cuda) \
  _(update_electric_ez_cpml_compressed, "update_electric_ez_cpml_compressed(Tensor(a!) ez, Tensor hx, Tensor hy, Tensor decay, Tensor curl, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_dx, Tensor inv_dy, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode, int x_low_length, int x_high_start, int x_high_length, int y_low_length, int y_high_start, int y_high_length, float? uniform_decay, float? uniform_curl) -> ()", update_electric_ez_cpml_compressed_cuda) \
  _(update_electric_ex_cpml_modulated_compressed, "update_electric_ex_cpml_modulated_compressed(Tensor(a!) ex, Tensor hy, Tensor hz, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor(b!) psi_y, Tensor(c!) psi_z, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dy, Tensor inv_dz, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode, int y_low_length, int y_high_start, int y_high_length, int z_low_length, int z_high_start, int z_high_length) -> ()", update_electric_ex_cpml_modulated_compressed_cuda) \
  _(update_electric_ey_cpml_modulated_compressed, "update_electric_ey_cpml_modulated_compressed(Tensor(a!) ey, Tensor hx, Tensor hz, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor(b!) psi_x, Tensor(c!) psi_z, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dx, Tensor inv_dz, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode, int x_low_length, int x_high_start, int x_high_length, int z_low_length, int z_high_start, int z_high_length) -> ()", update_electric_ey_cpml_modulated_compressed_cuda) \
  _(update_electric_ez_cpml_modulated_compressed, "update_electric_ez_cpml_modulated_compressed(Tensor(a!) ez, Tensor hx, Tensor hy, Tensor decay, Tensor curl, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_prev, float t_next, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_dx, Tensor inv_dy, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode, int x_low_length, int x_high_start, int x_high_length, int y_low_length, int y_high_start, int y_high_length) -> ()", update_electric_ez_cpml_modulated_compressed_cuda) \
  _(accumulate_dft_batched, "accumulate_dft_batched(Tensor ex, Tensor ey, Tensor ez, Tensor(a!) ex_real, Tensor(b!) ex_imag, Tensor(c!) ey_real, Tensor(d!) ey_imag, Tensor(e!) ez_real, Tensor(f!) ez_imag, Tensor weighted_cos, Tensor weighted_sin) -> ()", accumulate_dft_batched_cuda) \
  _(accumulate_point_observers, "accumulate_point_observers(Tensor field, Tensor point_i, Tensor point_j, Tensor point_k, Tensor(a!) real_accum, Tensor(b!) imag_accum, float weighted_cos, float weighted_sin) -> ()", accumulate_point_observers_cuda) \
  _(accumulate_plane_observer, "accumulate_plane_observer(Tensor field, Tensor(a!) real_accum, Tensor(b!) imag_accum, int axis, int plane_index, float weighted_cos, float weighted_sin) -> ()", accumulate_plane_observer_cuda) \
  _(plane_flux_reduce, "plane_flux_reduce(Tensor ea, Tensor eb, Tensor ha, Tensor hb, Tensor weights, Tensor(a!) out, int out_index, float scale) -> ()", plane_flux_reduce_cuda) \
  _(update_debye_current, "update_debye_current(Tensor electric, Tensor(a!) polarization, Tensor(b!) current, Tensor drive, float decay, float dt) -> ()", update_debye_current_cuda) \
  _(update_drude_current, "update_drude_current(Tensor electric, Tensor(a!) current, Tensor drive, float decay) -> ()", update_drude_current_cuda) \
  _(update_lorentz_current, "update_lorentz_current(Tensor electric, Tensor(a!) polarization, Tensor(b!) current, Tensor drive, float decay, float restoring, float dt) -> ()", update_lorentz_current_cuda) \
  _(apply_polarization_current, "apply_polarization_current(Tensor(a!) electric, Tensor current, Tensor inv_permittivity, float dt) -> ()", apply_polarization_current_cuda) \
  _(apply_polarization_current_modulated, "apply_polarization_current_modulated(Tensor(a!) electric, Tensor current, Tensor inv_permittivity, Tensor mod_cos, Tensor mod_sin, Tensor mod_omega, float t_next, float dt) -> ()", apply_polarization_current_modulated_cuda) \
  _(update_kerr_ex_curl, "update_kerr_ex_curl(Tensor(a!) dynamic_curl, Tensor ex, Tensor ey, Tensor ez, Tensor linear_permittivity, Tensor ex_decay, Tensor chi3, float dt, float eps0) -> ()", update_kerr_ex_curl_cuda) \
  _(update_kerr_ey_curl, "update_kerr_ey_curl(Tensor(a!) dynamic_curl, Tensor ex, Tensor ey, Tensor ez, Tensor linear_permittivity, Tensor ey_decay, Tensor chi3, float dt, float eps0) -> ()", update_kerr_ey_curl_cuda) \
  _(update_kerr_ez_curl, "update_kerr_ez_curl(Tensor(a!) dynamic_curl, Tensor ex, Tensor ey, Tensor ez, Tensor linear_permittivity, Tensor ez_decay, Tensor chi3, float dt, float eps0) -> ()", update_kerr_ez_curl_cuda) \
  _(update_nonlinear_coefficients, "update_nonlinear_coefficients(Tensor(a!) dynamic_decay, Tensor(b!) dynamic_curl, Tensor ex, Tensor ey, Tensor ez, Tensor linear_permittivity, Tensor external_decay, Tensor sigma_static, Tensor chi2, Tensor chi3, Tensor tpa_sigma, int component, float dt, float eps0) -> ()", update_nonlinear_coefficients_cuda) \
  _(update_electric_ex_full_aniso, "update_electric_ex_full_aniso(Tensor(a!) ex, Tensor hx, Tensor hy, Tensor hz, Tensor coeff_y, Tensor coeff_z, Tensor inv_dx, Tensor inv_dy, Tensor inv_dz, int periodic_x, int periodic_y, int periodic_z) -> ()", update_electric_ex_full_aniso_cuda) \
  _(update_electric_ey_full_aniso, "update_electric_ey_full_aniso(Tensor(a!) ey, Tensor hx, Tensor hy, Tensor hz, Tensor coeff_x, Tensor coeff_z, Tensor inv_dx, Tensor inv_dy, Tensor inv_dz, int periodic_x, int periodic_y, int periodic_z) -> ()", update_electric_ey_full_aniso_cuda) \
  _(update_electric_ez_full_aniso, "update_electric_ez_full_aniso(Tensor(a!) ez, Tensor hx, Tensor hy, Tensor hz, Tensor coeff_x, Tensor coeff_y, Tensor inv_dx, Tensor inv_dy, Tensor inv_dz, int periodic_x, int periodic_y, int periodic_z) -> ()", update_electric_ez_full_aniso_cuda) \
  _(update_electric_ex_full_aniso_cpml, "update_electric_ex_full_aniso_cpml(Tensor(a!) ex, Tensor hx, Tensor hy, Tensor hz, Tensor coeff_y, Tensor coeff_z, Tensor inv_dx, Tensor inv_dy, Tensor inv_dz, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, int periodic_x, int periodic_y, int periodic_z, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor(d!) psi_z) -> ()", update_electric_ex_full_aniso_cpml_cuda) \
  _(update_electric_ey_full_aniso_cpml, "update_electric_ey_full_aniso_cpml(Tensor(a!) ey, Tensor hx, Tensor hy, Tensor hz, Tensor coeff_x, Tensor coeff_z, Tensor inv_dx, Tensor inv_dy, Tensor inv_dz, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, int periodic_x, int periodic_y, int periodic_z, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor(d!) psi_z) -> ()", update_electric_ey_full_aniso_cpml_cuda) \
  _(update_electric_ez_full_aniso_cpml, "update_electric_ez_full_aniso_cpml(Tensor(a!) ez, Tensor hx, Tensor hy, Tensor hz, Tensor coeff_x, Tensor coeff_y, Tensor inv_dx, Tensor inv_dy, Tensor inv_dz, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, int periodic_x, int periodic_y, int periodic_z, Tensor(b!) psi_x, Tensor(c!) psi_y, Tensor(d!) psi_z) -> ()", update_electric_ez_full_aniso_cpml_cuda) \
  _(apply_aniso_offdiag_current_ex, "apply_aniso_offdiag_current_ex(Tensor(a!) ex, Tensor jy, Tensor jz, Tensor coeff_y, Tensor coeff_z, int periodic_x, int periodic_y, int periodic_z) -> ()", apply_aniso_offdiag_current_ex_cuda) \
  _(apply_aniso_offdiag_current_ey, "apply_aniso_offdiag_current_ey(Tensor(a!) ey, Tensor jx, Tensor jz, Tensor coeff_x, Tensor coeff_z, int periodic_x, int periodic_y, int periodic_z) -> ()", apply_aniso_offdiag_current_ey_cuda) \
  _(apply_aniso_offdiag_current_ez, "apply_aniso_offdiag_current_ez(Tensor(a!) ez, Tensor jx, Tensor jy, Tensor coeff_x, Tensor coeff_y, int periodic_x, int periodic_y, int periodic_z) -> ()", apply_aniso_offdiag_current_ez_cuda) \
  _(add_source_patch, "add_source_patch(Tensor(a!) field, Tensor patch, int offset_i, int offset_j, int offset_k, float signal) -> ()", add_source_patch_cuda) \
  _(add_cw_phased_source_patch, "add_cw_phased_source_patch(Tensor(a!) field, Tensor patch_cos, Tensor patch_sin, int offset_i, int offset_j, int offset_k, float signal_cos, float signal_sin) -> ()", add_cw_phased_source_patch_cuda) \
  _(add_time_shifted_source_patch, "add_time_shifted_source_patch(Tensor(a!) field, Tensor patch, Tensor delay_patch, Tensor activation_delay_patch, int offset_i, int offset_j, int offset_k, int time_kind, float time, float frequency, float fwidth, float amplitude, float phase, float delay, int causal_gate) -> ()", add_time_shifted_source_patch_cuda) \
  _(add_source_patch_bloch, "add_source_patch_bloch(Tensor(a!) ex_real, Tensor(b!) ex_imag, Tensor(c!) ey_real, Tensor(d!) ey_imag, Tensor(e!) ez_real, Tensor(f!) ez_imag, Tensor patch, int offset_i, int offset_j, int offset_k, float signal_real, float signal_imag, int axis_code, float phase_cos_a, float phase_sin_a, float phase_cos_b, float phase_sin_b, int wrap_axis_a, int wrap_axis_b) -> ()", add_source_patch_bloch_cuda) \
  _(add_scaled_slice_source_patch, "add_scaled_slice_source_patch(Tensor(a!) field, Tensor patch, Tensor incident, int sample_index, int offset_i, int offset_j, int offset_k, float scale) -> ()", add_scaled_slice_source_patch_cuda) \
  _(add_scaled_line_source_patch, "add_scaled_line_source_patch(Tensor(a!) field, Tensor patch, Tensor incident, Tensor sample_indices, int sample_axis, int offset_i, int offset_j, int offset_k, float scale) -> ()", add_scaled_line_source_patch_cuda) \
  _(add_interpolated_source_patch, "add_interpolated_source_patch(Tensor(a!) field, Tensor patch, Tensor incident, Tensor sample_positions, float origin, float ds, int offset_i, int offset_j, int offset_k, float scale) -> ()", add_interpolated_source_patch_cuda) \
  _(add_batched_reference_source_patches, "add_batched_reference_source_patches(Tensor(a!) field_x, Tensor(b!) field_y, Tensor(c!) field_z, Tensor coeff_data, Tensor incident, Tensor field_codes_per_coeff, Tensor field_offsets, Tensor sample_indices_per_coeff) -> ()", add_batched_reference_source_patches_cuda) \
  _(add_batched_interpolated_source_patches, "add_batched_interpolated_source_patches(Tensor(a!) field_x, Tensor(b!) field_y, Tensor(c!) field_z, Tensor coeff_data, Tensor incident, Tensor sample_positions, Tensor field_codes_per_coeff, Tensor field_offsets, float origin, float ds) -> ()", add_batched_interpolated_source_patches_cuda) \
  _(update_auxiliary_magnetic, "update_auxiliary_magnetic(Tensor(a!) magnetic, Tensor electric, Tensor decay, Tensor curl) -> ()", update_auxiliary_magnetic_cuda) \
  _(update_auxiliary_electric, "update_auxiliary_electric(Tensor(a!) electric, Tensor magnetic, Tensor decay, Tensor curl, int source_index, float source_value) -> ()", update_auxiliary_electric_cuda) \
  _(reverse_magnetic_adjoint_decay, "reverse_magnetic_adjoint_decay(Tensor(a!) adj_prev, Tensor adj_mid, Tensor decay) -> ()", reverse_magnetic_adjoint_decay_cuda) \
  _(reverse_electric_adjoint_to_hx_standard, "reverse_electric_adjoint_to_hx_standard(Tensor(a!) adj_hx_mid, Tensor adj_hx_post, Tensor adj_ey_post, Tensor adj_ez_post, Tensor ey_curl, Tensor ez_curl, Tensor inv_dy, Tensor inv_dz) -> ()", reverse_electric_adjoint_to_hx_standard_cuda) \
  _(reverse_electric_adjoint_to_hy_standard, "reverse_electric_adjoint_to_hy_standard(Tensor(a!) adj_hy_mid, Tensor adj_hy_post, Tensor adj_ex_post, Tensor adj_ez_post, Tensor ex_curl, Tensor ez_curl, Tensor inv_dx, Tensor inv_dz) -> ()", reverse_electric_adjoint_to_hy_standard_cuda) \
  _(reverse_electric_adjoint_to_hz_standard, "reverse_electric_adjoint_to_hz_standard(Tensor(a!) adj_hz_mid, Tensor adj_hz_post, Tensor adj_ex_post, Tensor adj_ey_post, Tensor ex_curl, Tensor ey_curl, Tensor inv_dx, Tensor inv_dy) -> ()", reverse_electric_adjoint_to_hz_standard_cuda) \
  _(reverse_magnetic_adjoint_to_ex_standard, "reverse_magnetic_adjoint_to_ex_standard(Tensor(a!) adj_ex_prev, Tensor(b!) grad_eps_ex, Tensor adj_ex_post, Tensor adj_hy_mid, Tensor adj_hz_mid, Tensor ex_decay, Tensor ex_curl, Tensor eps_ex, Tensor hy_mid, Tensor hz_mid, Tensor hy_curl, Tensor hz_curl, Tensor inv_dy_e, Tensor inv_dz_e, Tensor inv_dy_h, Tensor inv_dz_h, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode) -> ()", reverse_magnetic_adjoint_to_ex_standard_cuda) \
  _(reverse_magnetic_adjoint_to_ey_standard, "reverse_magnetic_adjoint_to_ey_standard(Tensor(a!) adj_ey_prev, Tensor(b!) grad_eps_ey, Tensor adj_ey_post, Tensor adj_hx_mid, Tensor adj_hz_mid, Tensor ey_decay, Tensor ey_curl, Tensor eps_ey, Tensor hx_mid, Tensor hz_mid, Tensor hx_curl, Tensor hz_curl, Tensor inv_dx_e, Tensor inv_dz_e, Tensor inv_dx_h, Tensor inv_dz_h, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode) -> ()", reverse_magnetic_adjoint_to_ey_standard_cuda) \
  _(reverse_magnetic_adjoint_to_ez_standard, "reverse_magnetic_adjoint_to_ez_standard(Tensor(a!) adj_ez_prev, Tensor(b!) grad_eps_ez, Tensor adj_ez_post, Tensor adj_hx_mid, Tensor adj_hy_mid, Tensor ez_decay, Tensor ez_curl, Tensor eps_ez, Tensor hx_mid, Tensor hy_mid, Tensor hx_curl, Tensor hy_curl, Tensor inv_dx_e, Tensor inv_dy_e, Tensor inv_dx_h, Tensor inv_dy_h, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode) -> ()", reverse_magnetic_adjoint_to_ez_standard_cuda) \
  _(reverse_electric_adjoint_to_hx_bloch, "reverse_electric_adjoint_to_hx_bloch(Tensor(a!) adj_hx_mid_real, Tensor(b!) adj_hx_mid_imag, Tensor adj_hx_post_real, Tensor adj_hx_post_imag, Tensor adj_ey_post_real, Tensor adj_ey_post_imag, Tensor adj_ez_post_real, Tensor adj_ez_post_imag, Tensor ey_curl, Tensor ez_curl, float phase_cos_y, float phase_sin_y, float phase_cos_z, float phase_sin_z, Tensor inv_dy, Tensor inv_dz) -> ()", reverse_electric_adjoint_to_hx_bloch_cuda) \
  _(reverse_electric_adjoint_to_hy_bloch, "reverse_electric_adjoint_to_hy_bloch(Tensor(a!) adj_hy_mid_real, Tensor(b!) adj_hy_mid_imag, Tensor adj_hy_post_real, Tensor adj_hy_post_imag, Tensor adj_ex_post_real, Tensor adj_ex_post_imag, Tensor adj_ez_post_real, Tensor adj_ez_post_imag, Tensor ex_curl, Tensor ez_curl, float phase_cos_x, float phase_sin_x, float phase_cos_z, float phase_sin_z, Tensor inv_dx, Tensor inv_dz) -> ()", reverse_electric_adjoint_to_hy_bloch_cuda) \
  _(reverse_electric_adjoint_to_hz_bloch, "reverse_electric_adjoint_to_hz_bloch(Tensor(a!) adj_hz_mid_real, Tensor(b!) adj_hz_mid_imag, Tensor adj_hz_post_real, Tensor adj_hz_post_imag, Tensor adj_ex_post_real, Tensor adj_ex_post_imag, Tensor adj_ey_post_real, Tensor adj_ey_post_imag, Tensor ex_curl, Tensor ey_curl, float phase_cos_x, float phase_sin_x, float phase_cos_y, float phase_sin_y, Tensor inv_dx, Tensor inv_dy) -> ()", reverse_electric_adjoint_to_hz_bloch_cuda) \
  _(reverse_magnetic_adjoint_to_ex_bloch, "reverse_magnetic_adjoint_to_ex_bloch(Tensor(a!) adj_ex_prev_real, Tensor(b!) adj_ex_prev_imag, Tensor(c!) grad_eps_ex, Tensor adj_ex_post_real, Tensor adj_ex_post_imag, Tensor adj_hy_mid_real, Tensor adj_hy_mid_imag, Tensor adj_hz_mid_real, Tensor adj_hz_mid_imag, Tensor ex_decay, Tensor ex_curl, Tensor eps_ex, Tensor hy_mid_real, Tensor hy_mid_imag, Tensor hz_mid_real, Tensor hz_mid_imag, Tensor hy_curl, Tensor hz_curl, float phase_cos_y, float phase_sin_y, float phase_cos_z, float phase_sin_z, Tensor inv_dy_e, Tensor inv_dz_e, Tensor inv_dy_h, Tensor inv_dz_h) -> ()", reverse_magnetic_adjoint_to_ex_bloch_cuda) \
  _(reverse_magnetic_adjoint_to_ey_bloch, "reverse_magnetic_adjoint_to_ey_bloch(Tensor(a!) adj_ey_prev_real, Tensor(b!) adj_ey_prev_imag, Tensor(c!) grad_eps_ey, Tensor adj_ey_post_real, Tensor adj_ey_post_imag, Tensor adj_hx_mid_real, Tensor adj_hx_mid_imag, Tensor adj_hz_mid_real, Tensor adj_hz_mid_imag, Tensor ey_decay, Tensor ey_curl, Tensor eps_ey, Tensor hx_mid_real, Tensor hx_mid_imag, Tensor hz_mid_real, Tensor hz_mid_imag, Tensor hx_curl, Tensor hz_curl, float phase_cos_x, float phase_sin_x, float phase_cos_z, float phase_sin_z, Tensor inv_dx_e, Tensor inv_dz_e, Tensor inv_dx_h, Tensor inv_dz_h) -> ()", reverse_magnetic_adjoint_to_ey_bloch_cuda) \
  _(reverse_magnetic_adjoint_to_ez_bloch, "reverse_magnetic_adjoint_to_ez_bloch(Tensor(a!) adj_ez_prev_real, Tensor(b!) adj_ez_prev_imag, Tensor(c!) grad_eps_ez, Tensor adj_ez_post_real, Tensor adj_ez_post_imag, Tensor adj_hx_mid_real, Tensor adj_hx_mid_imag, Tensor adj_hy_mid_real, Tensor adj_hy_mid_imag, Tensor ez_decay, Tensor ez_curl, Tensor eps_ez, Tensor hx_mid_real, Tensor hx_mid_imag, Tensor hy_mid_real, Tensor hy_mid_imag, Tensor hx_curl, Tensor hy_curl, float phase_cos_x, float phase_sin_x, float phase_cos_y, float phase_sin_y, Tensor inv_dx_e, Tensor inv_dy_e, Tensor inv_dx_h, Tensor inv_dy_h) -> ()", reverse_magnetic_adjoint_to_ez_bloch_cuda) \
  _(accumulate_forward_diff_adjoint, "accumulate_forward_diff_adjoint(Tensor(a!) field_grad, Tensor diff_grad, int axis, Tensor inv_delta) -> ()", accumulate_forward_diff_adjoint_cuda) \
  _(accumulate_backward_diff_adjoint, "accumulate_backward_diff_adjoint(Tensor(a!) field_grad, Tensor diff_grad, int axis, Tensor inv_delta) -> ()", accumulate_backward_diff_adjoint_cuda) \
  _(seed_inject_dense, "seed_inject_dense(Tensor(a!) adj_field, Tensor grad_real, Tensor grad_imag, Tensor cos_pack, Tensor sin_pack, int step) -> ()", seed_inject_dense_cuda) \
  _(seed_inject_point, "seed_inject_point(Tensor(a!) adj_field, Tensor grad_real, Tensor grad_imag, Tensor point_i, Tensor point_j, Tensor point_k, Tensor cos_pack, Tensor sin_pack, int step) -> ()", seed_inject_point_cuda) \
  _(seed_inject_plane, "seed_inject_plane(Tensor(a!) adj_field, Tensor grad_real, Tensor grad_imag, Tensor cos_pack, Tensor sin_pack, int axis, int plane_index, int step) -> ()", seed_inject_plane_cuda) \
  _(accumulate_in_place, "accumulate_in_place(Tensor(a!) dst, Tensor src) -> ()", accumulate_in_place_cuda) \
  _(reverse_electric_component_ex_cpml, "reverse_electric_component_ex_cpml(Tensor(a!) adj_prev, Tensor(b!) grad_eps, Tensor(c!) adj_psi_pos_prev, Tensor(d!) adj_psi_neg_prev, Tensor(e!) adj_d_pos, Tensor(f!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor curl, Tensor eps, Tensor psi_pos, Tensor psi_neg, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg, Tensor hy_mid, Tensor hz_mid, Tensor inv_dy, Tensor inv_dz, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode) -> ()", reverse_electric_component_ex_cpml_cuda) \
  _(reverse_electric_component_ey_cpml, "reverse_electric_component_ey_cpml(Tensor(a!) adj_prev, Tensor(b!) grad_eps, Tensor(c!) adj_psi_pos_prev, Tensor(d!) adj_psi_neg_prev, Tensor(e!) adj_d_pos, Tensor(f!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor curl, Tensor eps, Tensor psi_pos, Tensor psi_neg, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg, Tensor hx_mid, Tensor hz_mid, Tensor inv_dx, Tensor inv_dz, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode) -> ()", reverse_electric_component_ey_cpml_cuda) \
  _(reverse_electric_component_ez_cpml, "reverse_electric_component_ez_cpml(Tensor(a!) adj_prev, Tensor(b!) grad_eps, Tensor(c!) adj_psi_pos_prev, Tensor(d!) adj_psi_neg_prev, Tensor(e!) adj_d_pos, Tensor(f!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor curl, Tensor eps, Tensor psi_pos, Tensor psi_neg, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg, Tensor hx_mid, Tensor hy_mid, Tensor inv_dx, Tensor inv_dy, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode) -> ()", reverse_electric_component_ez_cpml_cuda) \
  _(reverse_electric_component_ex_cpml_conductive, "reverse_electric_component_ex_cpml_conductive(Tensor(a!) adj_prev, Tensor(b!) grad_eps, Tensor(c!) adj_psi_pos_prev, Tensor(d!) adj_psi_neg_prev, Tensor(e!) adj_d_pos, Tensor(f!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor curl, Tensor half, Tensor e_prev, Tensor eps, float dt, Tensor psi_pos, Tensor psi_neg, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg, Tensor hy_mid, Tensor hz_mid, Tensor inv_dy, Tensor inv_dz, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode) -> ()", reverse_electric_component_ex_cpml_conductive_cuda) \
  _(reverse_electric_component_ey_cpml_conductive, "reverse_electric_component_ey_cpml_conductive(Tensor(a!) adj_prev, Tensor(b!) grad_eps, Tensor(c!) adj_psi_pos_prev, Tensor(d!) adj_psi_neg_prev, Tensor(e!) adj_d_pos, Tensor(f!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor curl, Tensor half, Tensor e_prev, Tensor eps, float dt, Tensor psi_pos, Tensor psi_neg, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg, Tensor hx_mid, Tensor hz_mid, Tensor inv_dx, Tensor inv_dz, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode) -> ()", reverse_electric_component_ey_cpml_conductive_cuda) \
  _(reverse_electric_component_ez_cpml_conductive, "reverse_electric_component_ez_cpml_conductive(Tensor(a!) adj_prev, Tensor(b!) grad_eps, Tensor(c!) adj_psi_pos_prev, Tensor(d!) adj_psi_neg_prev, Tensor(e!) adj_d_pos, Tensor(f!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor curl, Tensor half, Tensor e_prev, Tensor eps, float dt, Tensor psi_pos, Tensor psi_neg, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg, Tensor hx_mid, Tensor hy_mid, Tensor inv_dx, Tensor inv_dy, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode) -> ()", reverse_electric_component_ez_cpml_conductive_cuda) \
  _(collocation_transpose, "collocation_transpose(Tensor(a!) adj_ex, Tensor(b!) adj_ey, Tensor(c!) adj_ez, Tensor g_ex, Tensor g_ey, Tensor g_ez, Tensor ex, Tensor ey, Tensor ez) -> ()", collocation_transpose_cuda) \
  _(collocate_field_square, "collocate_field_square(Tensor(a!) fsq_ex, Tensor(b!) fsq_ey, Tensor(c!) fsq_ez, Tensor ex, Tensor ey, Tensor ez) -> ()", collocate_field_square_cuda) \
  _(full_aniso_curl_adjoint, "full_aniso_curl_adjoint(Tensor(a!) adj_curl_x, Tensor(b!) adj_curl_y, Tensor(c!) adj_curl_z, Tensor adj_ex, Tensor adj_ey, Tensor adj_ez, Tensor coeff_ex_y, Tensor coeff_ex_z, Tensor coeff_ey_x, Tensor coeff_ey_z, Tensor coeff_ez_x, Tensor coeff_ez_y) -> ()", full_aniso_curl_adjoint_cuda) \
  _(reverse_electric_component_ex_cpml_kerr, "reverse_electric_component_ex_cpml_kerr(Tensor(a!) adj_prev, Tensor(b!) grad_eps, Tensor(c!) grad_chi3, Tensor(d!) g_fsq, Tensor(e!) adj_psi_pos_prev, Tensor(f!) adj_psi_neg_prev, Tensor(g!) adj_d_pos, Tensor(h!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor eps, Tensor chi3, Tensor fsq, float dt, float eps0, Tensor psi_pos, Tensor psi_neg, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg, Tensor hy_mid, Tensor hz_mid, Tensor inv_dy, Tensor inv_dz, int y_low_mode, int y_high_mode, int z_low_mode, int z_high_mode) -> ()", reverse_electric_component_ex_cpml_kerr_cuda) \
  _(reverse_electric_component_ey_cpml_kerr, "reverse_electric_component_ey_cpml_kerr(Tensor(a!) adj_prev, Tensor(b!) grad_eps, Tensor(c!) grad_chi3, Tensor(d!) g_fsq, Tensor(e!) adj_psi_pos_prev, Tensor(f!) adj_psi_neg_prev, Tensor(g!) adj_d_pos, Tensor(h!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor eps, Tensor chi3, Tensor fsq, float dt, float eps0, Tensor psi_pos, Tensor psi_neg, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg, Tensor hx_mid, Tensor hz_mid, Tensor inv_dx, Tensor inv_dz, int x_low_mode, int x_high_mode, int z_low_mode, int z_high_mode) -> ()", reverse_electric_component_ey_cpml_kerr_cuda) \
  _(reverse_electric_component_ez_cpml_kerr, "reverse_electric_component_ez_cpml_kerr(Tensor(a!) adj_prev, Tensor(b!) grad_eps, Tensor(c!) grad_chi3, Tensor(d!) g_fsq, Tensor(e!) adj_psi_pos_prev, Tensor(f!) adj_psi_neg_prev, Tensor(g!) adj_d_pos, Tensor(h!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor eps, Tensor chi3, Tensor fsq, float dt, float eps0, Tensor psi_pos, Tensor psi_neg, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg, Tensor hx_mid, Tensor hy_mid, Tensor inv_dx, Tensor inv_dy, int x_low_mode, int x_high_mode, int y_low_mode, int y_high_mode) -> ()", reverse_electric_component_ez_cpml_kerr_cuda) \
  _(reverse_magnetic_component_hx_cpml, "reverse_magnetic_component_hx_cpml(Tensor(a!) adj_prev, Tensor(b!) adj_psi_pos_prev, Tensor(c!) adj_psi_neg_prev, Tensor(d!) adj_d_pos, Tensor(e!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor curl, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg) -> ()", reverse_magnetic_component_hx_cpml_cuda) \
  _(reverse_magnetic_component_hy_cpml, "reverse_magnetic_component_hy_cpml(Tensor(a!) adj_prev, Tensor(b!) adj_psi_pos_prev, Tensor(c!) adj_psi_neg_prev, Tensor(d!) adj_d_pos, Tensor(e!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor curl, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg) -> ()", reverse_magnetic_component_hy_cpml_cuda) \
  _(reverse_magnetic_component_hz_cpml, "reverse_magnetic_component_hz_cpml(Tensor(a!) adj_prev, Tensor(b!) adj_psi_pos_prev, Tensor(c!) adj_psi_neg_prev, Tensor(d!) adj_d_pos, Tensor(e!) adj_d_neg, Tensor adj_post, Tensor adj_psi_pos_post, Tensor adj_psi_neg_post, Tensor decay, Tensor curl, Tensor b_pos, Tensor c_pos, Tensor inv_kappa_pos, Tensor b_neg, Tensor c_neg, Tensor inv_kappa_neg) -> ()", reverse_magnetic_component_hz_cpml_cuda) \
  _(reverse_debye_current, "reverse_debye_current(Tensor(a!) adj_electric_prev, Tensor(b!) adj_polarization_prev, Tensor adj_polarization_post, Tensor adj_current_post, Tensor drive, float decay, float dt) -> ()", reverse_debye_current_cuda) \
  _(reverse_drude_current, "reverse_drude_current(Tensor(a!) adj_electric_prev, Tensor(b!) adj_current_prev, Tensor adj_current_post, Tensor drive, float decay) -> ()", reverse_drude_current_cuda) \
  _(reverse_lorentz_current, "reverse_lorentz_current(Tensor(a!) adj_electric_prev, Tensor(b!) adj_polarization_prev, Tensor(c!) adj_current_prev, Tensor adj_polarization_post, Tensor adj_current_post, Tensor drive, float decay, float restoring, float dt) -> ()", reverse_lorentz_current_cuda) \
  _(reverse_dispersive_correction, "reverse_dispersive_correction(Tensor(a!) adj_current_corrected, Tensor(b!) grad_eps, Tensor adj_current_post, Tensor adj_electric_post, Tensor current, Tensor eps, float dt) -> ()", reverse_dispersive_correction_cuda) \
  _(accumulate_tfsf_scalar_sample_adjoint, "accumulate_tfsf_scalar_sample_adjoint(Tensor(a!) adj_aux_field, Tensor adj_field_patch, Tensor coeff_patch, int sample_index, float component_scale) -> ()", accumulate_tfsf_scalar_sample_adjoint_cuda) \
  _(accumulate_tfsf_line_sample_adjoint, "accumulate_tfsf_line_sample_adjoint(Tensor(a!) adj_aux_field, Tensor adj_field_patch, Tensor coeff_patch, Tensor sample_indices, int sample_axis_code, float component_scale) -> ()", accumulate_tfsf_line_sample_adjoint_cuda) \
  _(accumulate_tfsf_interpolated_sample_adjoint, "accumulate_tfsf_interpolated_sample_adjoint(Tensor(a!) adj_aux_field, Tensor adj_field_patch, Tensor coeff_patch, Tensor sample_positions, float origin, float ds, float component_scale) -> ()", accumulate_tfsf_interpolated_sample_adjoint_cuda) \
  _(reverse_tfsf_auxiliary_electric, "reverse_tfsf_auxiliary_electric(Tensor(a!) adj_electric_prev, Tensor(b!) adj_magnetic_after, Tensor adj_electric_post, Tensor electric_decay, Tensor electric_curl, int source_index) -> ()", reverse_tfsf_auxiliary_electric_cuda) \
  _(reverse_tfsf_auxiliary_magnetic, "reverse_tfsf_auxiliary_magnetic(Tensor(a!) adj_electric_prev, Tensor(b!) adj_magnetic_prev, Tensor adj_magnetic_after, Tensor magnetic_decay, Tensor magnetic_curl) -> ()", reverse_tfsf_auxiliary_magnetic_cuda) \
  _(clamp_field_face, "clamp_field_face(Tensor(a!) field, int axis, int side) -> ()", clamp_field_face_cuda) \
  _(clamp_pec_boundary, "clamp_pec_boundary(Tensor(a!) field, int axis_a, int axis_b) -> ()", clamp_pec_boundary_cuda) \
  _(mur_abc_face, "mur_abc_face(Tensor(a!) field, int axis, int boundary_index, int adjacent_index, float coef, Tensor(b!) prev_boundary, Tensor(c!) prev_adjacent) -> ()", mur_abc_face_cuda) \
  _(project_periodic_boundary, "project_periodic_boundary(Tensor(a!) field, int axis) -> ()", project_periodic_boundary_cuda) \
  _(project_bloch_boundary, "project_bloch_boundary(Tensor(a!) field_real, Tensor(b!) field_imag, int axis, float phase_cos, float phase_sin) -> ()", project_bloch_boundary_cuda) \
  _(update_electric_ex_bloch_y_standard_z, "update_electric_ex_bloch_y_standard_z(Tensor(a!) ex_real, Tensor(b!) ex_imag, Tensor hy_real, Tensor hy_imag, Tensor hz_real, Tensor hz_imag, Tensor decay, Tensor curl, float phase_cos_y, float phase_sin_y, Tensor inv_dy, Tensor inv_dz, int z_low_mode, int z_high_mode) -> ()", update_electric_ex_bloch_y_standard_z_cuda) \
  _(update_electric_ey_bloch_x_standard_z, "update_electric_ey_bloch_x_standard_z(Tensor(a!) ey_real, Tensor(b!) ey_imag, Tensor hx_real, Tensor hx_imag, Tensor hz_real, Tensor hz_imag, Tensor decay, Tensor curl, float phase_cos_x, float phase_sin_x, Tensor inv_dx, Tensor inv_dz, int z_low_mode, int z_high_mode) -> ()", update_electric_ey_bloch_x_standard_z_cuda) \
  _(apply_electric_ex_cpml_z_correction, "apply_electric_ex_cpml_z_correction(Tensor(a!) ex, Tensor hy, Tensor curl, Tensor(b!) psi_z, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dz, int offset_i, int offset_j, int offset_k, int y_low_mode, int y_high_mode, int full_size_y, int full_size_z) -> ()", apply_electric_ex_cpml_z_correction_cuda) \
  _(apply_electric_ey_cpml_z_correction, "apply_electric_ey_cpml_z_correction(Tensor(a!) ey, Tensor hx, Tensor curl, Tensor(b!) psi_z, Tensor inv_kappa_z, Tensor b_z, Tensor c_z, Tensor inv_dz, int offset_i, int offset_j, int offset_k, int x_low_mode, int x_high_mode, int full_size_x, int full_size_z) -> ()", apply_electric_ey_cpml_z_correction_cuda) \
  _(apply_electric_ex_cpml_y_correction, "apply_electric_ex_cpml_y_correction(Tensor(a!) ex, Tensor hz, Tensor curl, Tensor(b!) psi_y, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_dy, int offset_i, int offset_j, int offset_k, int z_low_mode, int z_high_mode, int full_size_z, int full_size_y) -> ()", apply_electric_ex_cpml_y_correction_cuda) \
  _(apply_electric_ez_cpml_y_correction, "apply_electric_ez_cpml_y_correction(Tensor(a!) ez, Tensor hx, Tensor curl, Tensor(b!) psi_y, Tensor inv_kappa_y, Tensor b_y, Tensor c_y, Tensor inv_dy, int offset_i, int offset_j, int offset_k, int x_low_mode, int x_high_mode, int full_size_x, int full_size_y) -> ()", apply_electric_ez_cpml_y_correction_cuda) \
  _(apply_electric_ey_cpml_x_correction, "apply_electric_ey_cpml_x_correction(Tensor(a!) ey, Tensor hz, Tensor curl, Tensor(b!) psi_x, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_dx, int offset_i, int offset_j, int offset_k, int z_low_mode, int z_high_mode, int full_size_z, int full_size_x) -> ()", apply_electric_ey_cpml_x_correction_cuda) \
  _(apply_electric_ez_cpml_x_correction, "apply_electric_ez_cpml_x_correction(Tensor(a!) ez, Tensor hy, Tensor curl, Tensor(b!) psi_x, Tensor inv_kappa_x, Tensor b_x, Tensor c_x, Tensor inv_dx, int offset_i, int offset_j, int offset_k, int y_low_mode, int y_high_mode, int full_size_y, int full_size_x) -> ()", apply_electric_ez_cpml_x_correction_cuda) \
  _(add_source_patch_ex_periodic, "add_source_patch_ex_periodic(Tensor(a!) field, Tensor patch, int offset_i, int offset_j, int offset_k, float signal, int wrap_a, int wrap_b) -> ()", add_source_patch_ex_periodic_cuda) \
  _(add_source_patch_ey_periodic, "add_source_patch_ey_periodic(Tensor(a!) field, Tensor patch, int offset_i, int offset_j, int offset_k, float signal, int wrap_a, int wrap_b) -> ()", add_source_patch_ey_periodic_cuda) \
  _(add_source_patch_ez_periodic, "add_source_patch_ez_periodic(Tensor(a!) field, Tensor patch, int offset_i, int offset_j, int offset_k, float signal, int wrap_a, int wrap_b) -> ()", add_source_patch_ez_periodic_cuda)

STABLE_TORCH_LIBRARY(witwin_maxwell_fdtd_cuda, m) {
  m.def("synchronize_noop() -> ()");
  m.def("debug_linear_indices(int size_x, int size_y, int size_z) -> (Tensor, Tensor, Tensor, Tensor)");
#define WITWIN_DEFINE_CUDA_OP(name, schema, impl) m.def(schema);
  WITWIN_CUDA_OPS(WITWIN_DEFINE_CUDA_OP)
#undef WITWIN_DEFINE_CUDA_OP
}
STABLE_TORCH_LIBRARY_IMPL(witwin_maxwell_fdtd_cuda, CUDA, m) {
#define WITWIN_IMPL_CUDA_OP(name, schema, fn) m.impl(#name, TORCH_BOX(&fn));
  WITWIN_CUDA_OPS(WITWIN_IMPL_CUDA_OP)
#undef WITWIN_IMPL_CUDA_OP
}
STABLE_TORCH_LIBRARY_IMPL(witwin_maxwell_fdtd_cuda, CompositeExplicitAutograd, m) {
  m.impl("synchronize_noop", TORCH_BOX(&synchronize_noop_cuda));
  m.impl("debug_linear_indices", TORCH_BOX(&debug_linear_indices_cuda));
}
