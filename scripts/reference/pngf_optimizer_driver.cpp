// Standalone adapter: compile against the separately obtained upstream headers.
// No upstream implementation is vendored into this repository.
#include <array>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>
#include <chrono>
#include <stdexcept>
#include <Eigen>
#include <cblas.h>
#include <omp.h>
using precision = double;
using lp_complex = std::complex<double>;
precision operator"" _d(long double x) { return double(x); }
extern "C" {
void zgetrf_(const int*, const int*, lp_complex*, const int*, int*, int*);
void zgetri_(const int*, lp_complex*, const int*, const int*, lp_complex*, const int*, int*);
void zgetrs_(const char*, const int*, const int*, const lp_complex*, const int*, const int*, lp_complex*, const int*, int*);
}
using namespace std;
using namespace Eigen;
#define USE_ORIG_GMAT 1
#include "linalg.hpp"
#include "pngfopt.hpp"
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
void read_array(const std::string& path, lp_complex* p, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(p), n*sizeof(lp_complex))) throw std::runtime_error(path);
}
int main(int argc, char** argv) {
    if (argc != 4) throw std::runtime_error("driver DIRECTORY CELLS_PER_AXIS THREADS");
    std::string dir = argv[1]; int m = std::stoi(argv[2]);
    omp_set_num_threads(std::stoi(argv[3]));
    PNGF_Optimizer opt((dir+"/green.bin").c_str(), m,m,1,1,3,1,1.,1.,1.);
    read_array(dir+"/observation.bin",opt.Gobj.data(),3*opt.yee_sz);
    read_array(dir+"/incident.bin",opt.Einc.data(),opt.yee_sz);
    Eigen::VectorXcd background(3);
    read_array(dir+"/background.bin",background.data(),3);
    std::vector<int> empty(m*m,0); opt.set_tile_map(empty);
    auto start=Clock::now();
    opt.compute_C_matrix();
    if (!LinAlg::invertMatrix(opt.C)) throw std::runtime_error("initial inverse failed");
    opt.compute_PEinc(); opt.compute_R_matrix(1);
    double init=elapsed(start);
    std::ifstream sequence(dir+"/sequence.txt");
    int tile, accept; sequence >> tile >> accept;
    for(int i=0;i<10;i++) opt.compute_updated_Eobj(&tile,1);
    sequence.clear(); sequence.seekg(0);
    std::ofstream out(dir+"/cpu-"+argv[3]+".csv");
    out << std::setprecision(17) << "trial_s,accept_s,rank,re0,im0,re1,im1,re2,im2\n";
    while(sequence >> tile >> accept) {
        start=Clock::now(); opt.compute_updated_Eobj(&tile,1); double trial=elapsed(start);
        Eigen::VectorXcd observed = background+opt.newEobj;
        double accepted=0;
        if(accept) {
            start=Clock::now(); opt.update_C_inv(); opt.compute_PEinc(); opt.compute_R_matrix(0);
            accepted=elapsed(start);
        }
        out << trial << ',' << accepted << ',' << opt.num_yee_flipped;
        for(int j=0;j<3;j++) out << ',' << observed(j).real() << ',' << observed(j).imag();
        out << '\n';
    }
    std::cout << "initialization_seconds=" << std::setprecision(17) << init << '\n';
}
