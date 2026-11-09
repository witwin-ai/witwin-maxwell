#include "checks.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

struct Coefficients {
  const float *material[6], *sigma[6], *profile[3];
  float *decay[6], *curl[6];
  int n[3], size[6][3], count[6];
  bool electric_loss, magnetic_loss, absorber;
  float dt;
};
__global__ void coefficients_kernel(Coefficients p) {
  int i=blockIdx.x*blockDim.x+threadIdx.x, c=blockIdx.y;
  if(i>=p.count[c]) return;
  float material=p.material[c][i], curl=(1.f/material)*p.dt, decay=1.f;
  if(c<3 ? p.electric_loss : p.magnetic_loss) {
    float half=((.5f*p.sigma[c][i])*p.dt)/material;
    float denominator=1.f+half;
    decay=(1.f-half)/denominator;
    curl=curl/denominator;
  }
  if(p.absorber) {
    int z=i%p.size[c][2], y=(i/p.size[c][2])%p.size[c][1], x=i/(p.size[c][1]*p.size[c][2]);
    int base=(x*p.n[1]+y)*p.n[2]+z, component=c%3;
    float s[2]; int j=0;
    for(int a=0;a<3;++a) if(a!=component) {
      const float* q=p.profile[a];
      if(c<3) {
        int stride=component==0 ? p.n[1]*p.n[2] : component==1 ? p.n[2] : 1;
        s[j]=.5f*(q[base]+q[base+stride]);
      } else {
        int b=(component+1)%3, d=(component+2)%3;
        // Match Python's x/y/z slice order, independent of component order.
        if(b>d) {int t=b;b=d;d=t;}
        int strides[3]={p.n[1]*p.n[2],p.n[2],1};
        s[j]=.25f*(((q[base]+q[base+strides[b]])+q[base+strides[d]])+q[base+strides[b]+strides[d]]);
      }
      ++j;
    }
    float external=(1.f/(1.f+p.dt*(s[0]+s[1])));
    decay=(c<3 ? p.electric_loss : p.magnetic_loss) ? decay*external : external;
    curl=curl*external;
  }
  p.decay[c][i]=decay; p.curl[c][i]=curl;
}

std::vector<at::Tensor> coefficients_setup(std::vector<at::Tensor> material,
    std::vector<at::Tensor> sigma, std::vector<at::Tensor> profiles,
    std::vector<int64_t> shape, double dt, bool electric_loss, bool magnetic_loss) {
  check_tensors(material, material[0], at::kFloat, "coefficients", "material");
  check_tensors(sigma, material[0], at::kFloat, "coefficients", "sigma");
  check_tensors(profiles, material[0], at::kFloat, "coefficients", "profiles");
  c10::cuda::CUDAGuard guard(material[0].device());
  Coefficients p{}; std::vector<at::Tensor> out;
  p.dt=dt; p.electric_loss=electric_loss; p.magnetic_loss=magnetic_loss;
  p.absorber=!profiles.empty(); int maximum=0;
  for(int a=0;a<3;++a) {p.n[a]=shape[a]; if(p.absorber) p.profile[a]=profiles[a].data_ptr<float>();}
  for(int c=0;c<6;++c) {
    out.push_back(at::empty_like(material[c])); out.push_back(at::empty_like(material[c]));
    p.material[c]=material[c].data_ptr<float>(); p.sigma[c]=sigma[c].data_ptr<float>();
    p.decay[c]=out[2*c].data_ptr<float>(); p.curl[c]=out[2*c+1].data_ptr<float>();
    p.count[c]=material[c].numel(); maximum=std::max(maximum,p.count[c]);
    for(int a=0;a<3;++a) p.size[c][a]=material[c].size(a);
  }
  coefficients_kernel<<<dim3((maximum+255)/256,6),256,0,c10::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
