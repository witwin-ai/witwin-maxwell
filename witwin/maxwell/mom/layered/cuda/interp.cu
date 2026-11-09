#include "green_eval.cuh"
using namespace layered;

__global__ void grid_kernel(double* rho,int N,double near,double limit) {
  int r=blockIdx.x*blockDim.x+threadIdx.x;
  if(r>=N+2) return;
  int split=N/4;
  rho[r]=r<=split ? near*r/split : near*exp(log(limit/near)*(r-split)/(N-1-split));
  if(r==N-1) rho[r]=limit;
}
at::Tensor layered_grid(at::Tensor anchor,int64_t N,double near,double limit) {
  c10::cuda::CUDAGuard guard(anchor.device());
  // Two unqueried log guard nodes make the last admitted derivative centered.
  auto out=at::empty({N+2},anchor.options().dtype(at::kDouble));
  grid_kernel<<<(N+2+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(out.data_ptr<double>(),N,near,limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// Packing adds the analytic inverse of the second extracted asymptotic term.
// Model: 16 B input, 16 B output, 16 B C3 (cacheable), ~20 FLOP + exp.
__global__ void pack_kernel(const C* transformed,const double* rho,const double* planes,
    const int64_t* pairs,const C* media,const double* sigma,const double* f,const C* tails,
    C* table,int F,int P,int N,int begin,int count,bool midpoint,double a,int mode,double ground) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=F*count*P*2) return;
  int c=t%2,p=(t/2)%P,r=(t/(2*P))%count,fi=t/(2*P*count);
  double x=midpoint ? (rho[begin+r]+rho[begin+r+1])*0.5 : rho[begin+r];
  C v;
  if(mode) {
    double zs=planes[pairs[2*p]],zo=planes[pairs[2*p+1]];
    C e=eps(media,sigma,1,2*pi*f[fi]);
    C k=root((2*pi*f[fi]/c0)*(2*pi*f[fi]/c0)*e);
    // expm1 for the finite, smooth g-g_static at a coincident source.
    double d=hypot(x,zo-zs);
    C kd=k*d;
    C numerator=ex(C(0,1)*kd)-C(1);
    if(abs(kd)<1e-4) {
      C term=C(0,1)*kd; numerator=term;
      for(int j=2;j<=8;++j) {term*=C(0,1)*kd/double(j);numerator+=term;}
    }
    v=d==0 ? C(0,1)*k/(4*pi) : numerator/(4*pi*d);
    if(mode>1) {
      d=hypot(x,zs+zo-2*ground);
      v-=(ex(C(0,1)*k*d)-C(1))/(4*pi*d);
    }
    if(c) v/=e;
    if(mode>1 && (zs==ground || zo==ground)) v=C(0);
  } else v=transformed[t]+tails[(fi*P+p)*2+c]*exp(-a*x)/(2*pi*a);
  int outN=midpoint ? N-1:N;
  table[((fi*P+p)*2+c)*outN+begin+r]=v;
}
void layered_pack(at::Tensor transformed,at::Tensor rho,at::Tensor planes,at::Tensor pairs,
    at::Tensor media,at::Tensor sigma,at::Tensor frequencies,at::Tensor tails,
    int64_t begin,int64_t end,bool midpoint,double a,int64_t mode,double ground,
    at::Tensor table) {
  c10::cuda::CUDAGuard guard(rho.device());
  int count=frequencies.numel()*(end-begin)*pairs.size(0)*2;
  pack_kernel<<<(count+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      transformed.data_ptr<C>(),rho.data_ptr<double>(),planes.data_ptr<double>(),pairs.data_ptr<int64_t>(),
      media.data_ptr<C>(),sigma.data_ptr<double>(),frequencies.data_ptr<double>(),tails.data_ptr<C>(),
      table.data_ptr<C>(),frequencies.numel(),pairs.size(0),rho.numel(),begin,end-begin,midpoint,a,mode,ground);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// K3 derivatives: five-point nonuniform polynomial, fixed arithmetic order.
// Per node: ~5*16 B values + 5*8 B grid, 16 B store; ~200 real FLOP.
__global__ void slopes_kernel(const C* y,const double* x,C* out,int N,int rows) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=rows*N) return;
  int i=t%N,first=max(0,min(N-5,i-2));C v(0);
  for(int j=first;j<first+5;++j) {
    double w=0;
    if(j==i) {
      for(int k=first;k<first+5;++k) if(k!=i) w+=1/(x[i]-x[k]);
    } else {
      w=1/(x[j]-x[i]);
      for(int k=first;k<first+5;++k) if(k!=j && k!=i) w*=(x[i]-x[k])/(x[j]-x[k]);
    }
    v+=w*y[(t/N)*N+j];
  }
  out[t]=v;
}
at::Tensor layered_slopes(at::Tensor y,at::Tensor x) {
  c10::cuda::CUDAGuard guard(y.device());auto out=at::empty_like(y);
  int count=y.numel(),N=x.numel();
  slopes_kernel<<<(count+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      y.data_ptr<C>(),x.data_ptr<double>(),out.data_ptr<C>(),N,count/N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
__global__ void magnitude_kernel(const C* mid,const double* x,const int64_t* offsets,
    const double* distances,const C* images,double* magnitudes,int N,int P,int I,int rows) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=rows*(N-3)) return;
  int r=t%(N-3),row=t/(N-3),c=row%2,p=(row/2)%P,f=row/(2*P);
  double rho=(x[r]+x[r+1])*0.5;
  C value(0),correction(0);
  for(int i=offsets[p];i<offsets[p+1];++i) {
    C term=images[(f*2+c)*I+i]/(4*pi*hypot(rho,distances[i]))-correction;
    C next=value+term;correction=(next-value)-term;value=next;
  }
  magnitudes[t]=abs(value+mid[row*(N-1)+r]);
}
__global__ void admission_kernel(const C* y,const C* d,const C* mid,const double* x,
                                  const double* magnitudes,double* out,int N,int rows) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=rows) return;
  double error=0,scale=0;
  // Per physical midpoint: |interp-direct| / max(|G_total(mid)|,
  // 1e-6*max_row |G_total|). The row maximum is over physical midpoints,
  // excluding the two guard intervals and the singular self point rho=0.
  for(int i=0;i<N-3;++i) scale=fmax(scale,magnitudes[row*(N-3)+i]);
  for(int i=0;i<N-3;++i) {
    C value=hermite((x[i]+x[i+1])*0.5,x[i],x[i+1],y[row*N+i],y[row*N+i+1],
                    d[row*N+i],d[row*N+i+1]);
    double difference=abs(value-mid[row*(N-1)+i]);
    double denominator=fmax(magnitudes[row*(N-3)+i],1e-6*scale);
    double relative=denominator==0 ? (difference==0 ? 0:INFINITY) : difference/denominator;
    error=isfinite(relative) ? fmax(error,relative) : INFINITY;
  }
  out[row]=error;
}
__global__ void admission_max(const double* errors,double* result,int rows) {
  double error=0;
  for(int i=0;i<rows;++i) error=fmax(error,errors[i]);
  result[0]=error;
}
at::Tensor layered_admission(at::Tensor y,at::Tensor d,at::Tensor mid,at::Tensor x,
                            at::Tensor offsets,at::Tensor distances,at::Tensor images) {
  c10::cuda::CUDAGuard guard(y.device());int rows=y.numel()/x.numel();
  auto errors=at::empty({rows},x.options()),result=at::empty({},x.options());
  int count=rows*(x.numel()-3);
  auto magnitudes=at::empty({count},x.options());
  magnitude_kernel<<<(count+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      mid.data_ptr<C>(),x.data_ptr<double>(),offsets.data_ptr<int64_t>(),distances.data_ptr<double>(),
      images.data_ptr<C>(),magnitudes.data_ptr<double>(),x.numel(),y.size(1),distances.numel(),rows);
  admission_kernel<<<(rows+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      y.data_ptr<C>(),d.data_ptr<C>(),mid.data_ptr<C>(),x.data_ptr<double>(),
      magnitudes.data_ptr<double>(),errors.data_ptr<double>(),x.numel(),rows);
  admission_max<<<1,1,0,c10::cuda::getCurrentCUDAStream()>>>(errors.data_ptr<double>(),result.data_ptr<double>(),rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}

// K3 evaluation: four table/slopes samples (64 B), ~60 FLOP per component,
// plus image gather (~24 B/image, sqrt/divide). Each query owns its output.
__global__ void evaluate_kernel(GreenTableView table,const double* rho,const void* ai,const void* bi,
    bool indices,int A,int B,int M,int query_count,C* out,int* status) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=table.frequencies*2*query_count) return;
  int query=t%query_count,c=(t/query_count)%2,f=t/(2*query_count);
  int i=-1,j=-1;
  if(indices) { i=((const int64_t*)ai)[A==1?0:query];j=((const int64_t*)bi)[B==1?0:query]; }
  else {
    double za=((const double*)ai)[A==1?0:query],zb=((const double*)bi)[B==1?0:query];
    for(int p=0;p<M;++p) {if(table.plane_z[p]==za) i=p;if(table.plane_z[p]==zb) j=p;}
  }
  if(i<0||j<0||i>=M||j>=M) {atomicOr(status,1);return;}
  double r=rho[query];
  if(!(r>=0 && r<=table.rho_max)) {atomicOr(status,2);return;}
  if(j<i) {int swap=i;i=j;j=swap;}
  int pair=i*M-i*(i-1)/2+j-i;
  out[t]=green_value(table,f,pair,c,r);
}
at::Tensor layered_evaluate(at::Tensor y,at::Tensor slopes,at::Tensor x,at::Tensor planes,
    at::Tensor offsets,at::Tensor distances,at::Tensor images,at::Tensor media,at::Tensor sigma,
    at::Tensor f,at::Tensor ai,at::Tensor bi,at::Tensor rho,int64_t mode,double ground,double rho_max) {
  c10::cuda::CUDAGuard guard(y.device());
  auto out=at::empty({f.numel(),2,rho.numel()},y.options());
  auto status=at::zeros({},rho.options().dtype(at::kInt));
  GreenTableView view{};
  view.rho=x.data_ptr<double>();view.distances=distances.data_ptr<double>();
  view.plane_z=planes.data_ptr<double>();view.frequency_hz=f.data_ptr<double>();
  view.sigma=sigma.data_ptr<double>();view.epsilon=media.data_ptr<C>();
  view.values=y.data_ptr<C>();view.slopes=slopes.data_ptr<C>();view.images=images.data_ptr<C>();
  view.image_offsets=offsets.data_ptr<int64_t>();
  view.frequencies=f.numel();view.pairs=y.size(1);view.nodes=x.numel();
  view.image_count=distances.numel();view.metal_planes=planes.numel();
  view.exact_mode=mode;view.ground=ground;
  view.rho_max=rho_max;
  int count=out.numel();
  evaluate_kernel<<<(count+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      view,rho.data_ptr<double>(),ai.data_ptr(),bi.data_ptr(),ai.scalar_type()==at::kLong,ai.numel(),bi.numel(),
      planes.numel(),rho.numel(),out.data_ptr<C>(),status.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  int error=status.item<int>();
  TORCH_CHECK(!(error&1),"Unknown metal plane");
  TORCH_CHECK(!(error&2),"rho must be in [0,rho_max]");
  return out;
}

__global__ void accumulate_kernel(C* sum,C* correction,const C* product,int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=count) return;
  C value=product[i]-correction[i],next=sum[i]+value;
  correction[i]=(next-sum[i])-value;sum[i]=next;
}
void layered_accumulate(at::Tensor sum,at::Tensor correction,at::Tensor product) {
  c10::cuda::CUDAGuard guard(sum.device());int count=sum.numel();
  accumulate_kernel<<<(count+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      sum.data_ptr<C>(),correction.data_ptr<C>(),product.data_ptr<C>(),count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void tail_max_kernel(const double* bounds,double* out,int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=count) return;
  double value=0;for(int b=0;b<4;++b) value=fmax(value,bounds[b*count+i]);
  out[i]=value;
}
at::Tensor layered_tail_max(at::Tensor bounds) {
  c10::cuda::CUDAGuard guard(bounds.device());
  auto out=at::empty({bounds.size(1),bounds.size(2),bounds.size(3)},bounds.options());
  int count=out.numel();
  tail_max_kernel<<<(count+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      bounds.data_ptr<double>(),out.data_ptr<double>(),count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}


template<int Operation> __device__ C special(C z) {
  if(Operation==0) return ex(z*C(.01,.01));
  if(Operation==1) return root(z+C(1,1));
  return C(1)/(z+C(1,1));
}
template<int Operation> __global__ void special_peak_kernel(C* out,int iterations) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  C a(.1,.2),b(.3,.4),c(.5,.6),d(.7,.8);
  for(int i=0;i<iterations;++i) {
    a=special<Operation>(a);b=special<Operation>(b);
    c=special<Operation>(c);d=special<Operation>(d);
  }
  out[t]=a+b+c+d;
}

at::Tensor layered_special_peak(at::Tensor anchor,int64_t iterations,int64_t operation) {
  c10::cuda::CUDAGuard guard(anchor.device());
  auto out=at::empty({65536},anchor.options().dtype(at::kComplexDouble));
  if(operation==0) special_peak_kernel<0><<<256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<C>(),iterations);
  else if(operation==1) special_peak_kernel<1><<<256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<C>(),iterations);
  else special_peak_kernel<2><<<256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<C>(),iterations);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
__device__ double fused(double x) { return __fma_rn(x,0.9999,0.0001); }
__device__ float fused(float x) { return __fmaf_rn(x,0.9999f,0.0001f); }
template<typename T> __global__ void peak_kernel(T* out,int count,int iterations) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=count) return;
  T a=T(0.1),b=T(0.2),c=T(0.3),d=T(0.4),e=T(0.5),f=T(0.6),g=T(0.7),h=T(0.8);
  for(int i=0;i<iterations;++i) {
    a=fused(a);b=fused(b);c=fused(c);d=fused(d);
    e=fused(e);f=fused(f);g=fused(g);h=fused(h);
  }
  out[t]=a+b+c+d+e+f+g+h;
}
at::Tensor layered_peak(at::Tensor anchor,int64_t iterations,bool fp64) {
  c10::cuda::CUDAGuard guard(anchor.device());
  auto out=at::empty({65536},anchor.options().dtype(fp64?at::kDouble:at::kFloat));
  if(fp64) peak_kernel<<<256,256,0,c10::cuda::getCurrentCUDAStream()>>>(out.data_ptr<double>(),65536,iterations);
  else peak_kernel<<<256,256,0,c10::cuda::getCurrentCUDAStream()>>>(out.data_ptr<float>(),65536,iterations);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
