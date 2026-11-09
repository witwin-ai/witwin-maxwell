// Dump formulation-C Gxx, Gphi, Gzz, Gzx and Gxz by direct Sommerfeld integration.
// Usage: strata_ref layers.yaml f_hz z_src z_obs rho_file out_file
#include <array>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include "MGF.hpp"

int main(int argc, char** argv) {
  if (argc != 7) { std::fprintf(stderr, "usage: strata_ref layers.yaml f z_src z_obs rho_file out\n"); return 2; }
  double f = std::stod(argv[2]), zs = std::stod(argv[3]), zo = std::stod(argv[4]);
  std::vector<double> rho;
  std::ifstream in(argv[5]);
  for (double r; in >> r;) rho.push_back(r);
  LayerManager lm;
  lm.ProcessTechFile(argv[1]);
  lm.ProcessLayers(f);
  MGF_settings s;
  s.method = MGF_INTEGRATE;
  s.verbose = false;
  MGF mgf;
  mgf.Initialize(f, lm, s);
  mgf.SetLayers(lm.FindLayer(zs), lm.FindLayer(zo));
  std::ofstream out(argv[6]);
  out << std::setprecision(17);
  for (double r : rho) {
    std::array<std::complex<double>, 9> G;
    std::complex<double> phi;
    mgf.ComputeMGF(r, 0.0, zo, zs, G, phi);
    // x_diff=r, y_diff=0 gives cos(phi)=1 for both radial cross terms.
    out << r << " " << G[0].real() << " " << G[0].imag() << " " << phi.real() << " " << phi.imag()
        << " " << G[8].real() << " " << G[8].imag()
        << " " << G[6].real() << " " << G[6].imag()
        << " " << G[2].real() << " " << G[2].imag() << "\n";
  }
  return 0;
}
