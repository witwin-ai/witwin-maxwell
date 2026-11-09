#include "green_eval.cuh"
using namespace layered;

__global__ void vertical_coeff_kernel(const C* raw,const C* media,const double* sigma,
    const double* f,const int64_t* a,const int64_t* b,const double* geometry,C* out,int F,int I) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=F*I) return;
  int fi=t/I,i=t%I;double w=2*pi*f[fi],ds=geometry[3*i],dz=geometry[3*i+1];
  C h=raw[fi*2*I+i],d=raw[(fi*2+1)*I+i];
  C ea=eps(media,sigma,a[i],w),eb=eps(media,sigma,b[i],w);
  out[(fi*4)*I+i]=ds*dz*(h-(ea+eb)*d);
  out[(fi*4+1)*I+i]=dz*(eb*d-h);
  out[(fi*4+2)*I+i]=-ds*(ea*d-h);
  out[(fi*4+3)*I+i]=d;
}
at::Tensor layered_vertical_coeff(at::Tensor raw,at::Tensor media,at::Tensor sigma,
    at::Tensor f,at::Tensor a,at::Tensor b,at::Tensor geometry) {
  c10::cuda::CUDAGuard guard(f.device());int F=f.numel(),I=a.numel();
  auto out=at::empty({F,4,I},raw.options());
  vertical_coeff_kernel<<<(F*I+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      raw.data_ptr<C>(),media.data_ptr<C>(),sigma.data_ptr<double>(),f.data_ptr<double>(),
      a.data_ptr<int64_t>(),b.data_ptr<int64_t>(),geometry.data_ptr<double>(),out.data_ptr<C>(),F,I);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

__global__ void vertical_subtract_kernel(C* spectrum,const C* q,const double* f,
    Stack stack,const double* zs,const double* zo,const int64_t* pair_segment,
    const int64_t* offsets,const double* geometry,const int64_t* direct,const C* coeff,
    int F,int Q,int P,int I,int components) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=F*Q*P*components) return;
  int c=t%components,p=(t/components)%P,j=(t/(components*P))%Q,fi=t/(components*P*Q);
  C kr=q[fi*Q+j],value(0),correction(0);
  double source=zs[p],observer=zo[p],w=2*pi*f[fi];
  int segment=pair_segment[p];
  for(int i=offsets[segment];i<offsets[segment+1];++i) {
    C term;
    if(direct[i]) {
      C epsilon=eps(stack.media,stack.sigma,region(stack,source),w);
      C k=root((w/c0)*(w/c0)*epsilon-kr*kr);
      double distance=direct[i]==1 ? fabs(observer-source):
          fmax(0.0,geometry[3*i]*source+geometry[3*i+1]*observer+geometry[3*i+2]);
      term=(c==0||c==3||c==4) ? C(0,.5)/k*ex(C(0,1)*k*distance):C(0);
      if(direct[i]==1) {if(c==3) term/=epsilon;}
      else term*=coeff[(fi*components+c)*I+i];
    } else {
      double distance=fmax(0.0,geometry[3*i]*source+geometry[3*i+1]*observer+geometry[3*i+2]);
      term=coeff[(fi*components+c)*I+i]*ex(-kr*distance)/(2.0*kr);
    }
    term-=correction;C next=value+term;correction=(next-value)-term;value=next;
  }
  spectrum[t]-=value;
}
void layered_vertical_subtract(at::Tensor spectrum,at::Tensor q,at::Tensor f,
    at::Tensor media,at::Tensor sigma,at::Tensor edges,at::Tensor zs,at::Tensor zo,
    at::Tensor pair_segment,at::Tensor offsets,at::Tensor geometry,at::Tensor direct,at::Tensor coeff) {
  c10::cuda::CUDAGuard guard(f.device());int count=spectrum.numel();
  Stack stack{media.data_ptr<C>(),sigma.data_ptr<double>(),edges.data_ptr<double>(),int(media.numel()),0,0};
  vertical_subtract_kernel<<<(count+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      spectrum.data_ptr<C>(),q.data_ptr<C>(),f.data_ptr<double>(),stack,zs.data_ptr<double>(),
      zo.data_ptr<double>(),pair_segment.data_ptr<int64_t>(),offsets.data_ptr<int64_t>(),
      geometry.data_ptr<double>(),direct.data_ptr<int64_t>(),coeff.data_ptr<C>(),
      f.numel(),q.size(1),zs.numel(),direct.numel(),spectrum.size(3));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void vertical_pack_kernel(const C* zero,const C* one,C* out,int F,int R,int P,int N,int begin,int components) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=F*R*P*components) return;
  int c=t%components,p=(t/components)%P,r=(t/(components*P))%R,f=t/(components*P*R);
  out[((f*P+p)*components+c)*N+begin+r]=(c==1||c==2 ? one:zero)[t];
}
void layered_vertical_pack(at::Tensor zero,at::Tensor one,at::Tensor out,int64_t begin) {
  c10::cuda::CUDAGuard guard(out.device());int count=zero.numel();
  vertical_pack_kernel<<<(count+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      zero.data_ptr<C>(),one.data_ptr<C>(),out.data_ptr<C>(),out.size(0),zero.size(1),out.size(1),out.size(3),begin,out.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


VerticalGreenTableView vertical_view(const std::vector<at::Tensor>& t,double limit) {
  VerticalGreenTableView v{};
  v.rho=t[0].data_ptr<double>();v.z=t[1].data_ptr<double>();v.segments=t[2].data_ptr<double>();
  v.values=t[3].data_ptr<C>();v.slopes=t[4].data_ptr<C>();v.images=t[5].data_ptr<C>();
  v.geometry=t[6].data_ptr<double>();v.direct=t[7].data_ptr<int64_t>();v.offsets=t[8].data_ptr<int64_t>();
  v.epsilon=t[9].data_ptr<C>();v.sigma=t[10].data_ptr<double>();v.frequencies=t[11].data_ptr<double>();
  v.edges=t[12].data_ptr<double>();v.N=t[0].numel();v.Z=t[1].numel();v.S=t[2].size(0);
  v.K=v.Z/v.S;v.I=t[7].numel();v.L=t[9].numel();v.F=t[11].numel();v.rho_max=limit;
  v.boundaries=t[13].data_ptr<double>();
  if(t.size()>14) {
    v.horizontal_values=t[14].data_ptr<C>();v.horizontal_slopes=t[15].data_ptr<C>();
    v.horizontal_images=t[16].data_ptr<C>();
  }
  return v;
}
__global__ void vertical_evaluate_kernel(VerticalGreenTableView v,const double* zs,
    const double* zo,const double* rho,int A,int B,int M,C* out,int* status) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=v.F*4*M) return;
  int m=t%M,c=(t/M)%4,f=t/(M*4);
  double source=zs[A==1?0:m],observer=zo[B==1?0:m],r=rho[m];
  if(vertical_segment(v,source)<0 || vertical_segment(v,observer)<0) {atomicOr(status,1);return;}
  if(!(r>=0 && r<=v.rho_max)) {atomicOr(status,2);return;}
  out[t]=vertical_green_value(v,f,c,r,source,observer);
}
at::Tensor layered_vertical_evaluate(std::vector<at::Tensor> tensors,at::Tensor zs,
    at::Tensor zo,at::Tensor rho,double limit) {
  c10::cuda::CUDAGuard guard(rho.device());auto v=vertical_view(tensors,limit);
  TORCH_CHECK((zs.numel()==1||zs.numel()==rho.numel()) && (zo.numel()==1||zo.numel()==rho.numel()),
              "vertical coordinates must be scalar or match rho");
  auto out=at::empty({v.F,4,rho.numel()},tensors[3].options());
  auto status=at::zeros({},rho.options().dtype(at::kInt));int count=out.numel();
  vertical_evaluate_kernel<<<(count+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      v,zs.data_ptr<double>(),zo.data_ptr<double>(),rho.data_ptr<double>(),zs.numel(),zo.numel(),
      rho.numel(),out.data_ptr<C>(),status.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();int error=status.item<int>();
  TORCH_CHECK(!(error&1),"z is outside declared via spans");
  TORCH_CHECK(!(error&2),"rho must be in [0,rho_max]");
  return out;
}
__global__ void vertical_error_kernel(const C* actual,const C* expected,double* out,int F,int M) {
  int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=F) return;
  double family=0,error=0;
  for(int m=0;m<M;++m) family=fmax(family,abs(expected[f*4*M+m]));
  for(int c=0;c<4;++c) {
    double scale=0;
    for(int m=0;m<M;++m) scale=fmax(scale,abs(expected[(f*4+c)*M+m]));
    scale=fmax(scale,1e-6*family);
    for(int m=0;m<M;++m) {
      int t=(f*4+c)*M+m;
      double difference=abs(actual[t]-expected[t]),denominator=scale;
      double relative=denominator==0 ? (difference==0?0:INFINITY):difference/denominator;
      error=isfinite(relative)?fmax(error,relative):INFINITY;
    }
  }
  out[f]=error;
}
at::Tensor layered_vertical_error(at::Tensor actual,at::Tensor expected) {
  c10::cuda::CUDAGuard guard(actual.device());int F=actual.size(0);
  auto out=at::empty({F},actual.options().dtype(at::kDouble));
  vertical_error_kernel<<<(F+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      actual.data_ptr<C>(),expected.data_ptr<C>(),out.data_ptr<double>(),F,actual.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out.max();
}


__global__ void vertical_reference_kernel(VerticalGreenTableView v,const C* remainder,
    const double* zs,const double* zo,const double* rho,C* out,int M) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=v.F*4*M) return;
  int m=t%M,c=(t/M)%4,f=t/(M*4);
  double source=zs[m],observer=zo[m];
  int a=vertical_segment(v,source),b=vertical_segment(v,observer);
  out[t]=remainder[t]+vertical_green_images(v,f,c,rho[m],source,observer,a,b);
}
at::Tensor layered_vertical_reference(std::vector<at::Tensor> tensors,at::Tensor remainder,
    at::Tensor zs,at::Tensor zo,at::Tensor rho,double limit) {
  c10::cuda::CUDAGuard guard(rho.device());auto v=vertical_view(tensors,limit);
  auto out=at::empty_like(remainder);int count=out.numel();
  vertical_reference_kernel<<<(count+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      v,remainder.data_ptr<C>(),zs.data_ptr<double>(),zo.data_ptr<double>(),rho.data_ptr<double>(),
      out.data_ptr<C>(),rho.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// Admission batches radial checkpoints over one warp per row. Each point
// retains its fp64 arithmetic; max reductions select existing values exactly.
__device__ inline double admission_warp_max(double value) {
  for(int offset=16;offset>0;offset/=2)
    value=fmax(value,__shfl_down_sync(0xffffffff,value,offset));
  return __shfl_sync(0xffffffff,value,0);
}
__global__ void vertical_rho_error_kernel(VerticalGreenTableView v,const C* mid,
    const double* zs,const double* zo,double* out,int P,int N) {
  int row=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
  if(row>=v.F*P*4) return;
  int c=row%4,p=(row/4)%P,f=row/(4*P);
  double scale=0,error=0,source=zs[p],observer=zo[p];
  int a=vertical_segment(v,source),b=vertical_segment(v,observer);
  for(int r=lane;r<N-3;r+=32) {
    double rho=(v.rho[r]+v.rho[r+1])*.5;
    C total=mid[row*(N-1)+r]+vertical_green_images(v,f,c,rho,source,observer,a,b);
    scale=fmax(scale,abs(total));
  }
  // Near-zero cross rows use the vector diagonal as the admission scale.
  if(c==1||c==2) {
    double family=0;
    for(int r=lane;r<N-3;r+=32) {
      double rho=(v.rho[r]+v.rho[r+1])*.5;
      family=fmax(family,abs(mid[((f*P+p)*4)*(N-1)+r]+
          vertical_green_images(v,f,0,rho,source,observer,a,b)));
    }
    scale=fmax(scale,1e-6*family);
  }
  scale=admission_warp_max(scale);
  for(int r=lane;r<N-3;r+=32) {
    int base=row*N;
    C value=hermite((v.rho[r]+v.rho[r+1])*.5,v.rho[r],v.rho[r+1],
                    v.values[base+r],v.values[base+r+1],v.slopes[base+r],v.slopes[base+r+1]);
    double difference=abs(value-mid[row*(N-1)+r]);
    double relative=scale==0 ? (difference==0?0:INFINITY):difference/scale;
    error=isfinite(relative)?fmax(error,relative):INFINITY;
  }
  error=admission_warp_max(error);
  if(lane==0) out[row]=error;
}
at::Tensor layered_vertical_rho_error(std::vector<at::Tensor> tensors,at::Tensor mid,
    at::Tensor zs,at::Tensor zo,double limit) {
  c10::cuda::CUDAGuard guard(mid.device());auto v=vertical_view(tensors,limit);
  int rows=v.F*zs.numel()*4;auto out=at::empty({rows},zs.options());
  vertical_rho_error_kernel<<<(rows*32+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      v,mid.data_ptr<C>(),zs.data_ptr<double>(),zo.data_ptr<double>(),out.data_ptr<double>(),zs.numel(),v.N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out.max();
}


// Admission is diagnostic setup, with one fixed-order owner per checked row.
// At most 17*K*K Hermite evaluations/row; no atomics or host point loops.
__global__ void horizontal_z_error_kernel(VerticalGreenTableView v,const C* ref,const double* zs,
    const double* zo,const double* rho,double* out,int P,int R) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=v.F*P) return;
  int f=row/P,p=row%P;double scale=0,error=0;
  for(int r=0;r<R;++r) {
    int q=p*R+r;
    C image=horizontal_vertical_images(v,f,rho[q],zs[q],zo[q]);
    C expected=ref[row*R+r]+image;
    C actual=horizontal_vertical_remainder(v,f,rho[q],zs[q],zo[q])+image;
    scale=fmax(scale,abs(expected));error=fmax(error,abs(actual-expected));
  }
  out[row]=scale==0 ? (error==0?0:INFINITY):error/scale;
}
at::Tensor layered_horizontal_z_error(std::vector<at::Tensor> t,at::Tensor ref,at::Tensor zs,
    at::Tensor zo,at::Tensor rho) {
  c10::cuda::CUDAGuard guard(ref.device());auto v=vertical_view(t,0);
  int P=ref.size(1),R=ref.size(2);
  auto out=at::empty({v.F*P},rho.options());
  horizontal_z_error_kernel<<<(v.F*P+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      v,ref.data_ptr<C>(),zs.data_ptr<double>(),zo.data_ptr<double>(),rho.data_ptr<double>(),
      out.data_ptr<double>(),P,R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out.max();
}
__global__ void horizontal_rho_error_kernel(VerticalGreenTableView v,const C* ref,const double* zs,
    const double* zo,double* out,int P) {
  int row=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
  if(row>=v.F*P) return;
  int f=row/P,p=row%P;double scale=0,error=0;
  for(int r=lane;r<v.N-3;r+=32) {
    double rho=(v.rho[r]+v.rho[r+1])*.5;
    C expected=ref[row*(v.N-1)+r]+horizontal_vertical_images(v,f,rho,zs[p],zo[p]);
    int64_t base=int64_t(row)*v.N;
    C sample=hermite(rho,v.rho[r],v.rho[r+1],v.horizontal_values[base+r],v.horizontal_values[base+r+1],
                     v.horizontal_slopes[base+r],v.horizontal_slopes[base+r+1]);
    scale=fmax(scale,abs(expected));error=fmax(error,abs(sample-ref[row*(v.N-1)+r]));
  }
  scale=admission_warp_max(scale);error=admission_warp_max(error);
  if(lane==0) out[row]=scale==0 ? (error==0?0:INFINITY):error/scale;
}
at::Tensor layered_horizontal_rho_error(std::vector<at::Tensor> t,at::Tensor ref,
    at::Tensor zs,at::Tensor zo) {
  c10::cuda::CUDAGuard guard(ref.device());auto v=vertical_view(t,0);
  int P=zs.numel();auto out=at::empty({v.F*P},zs.options());
  horizontal_rho_error_kernel<<<(v.F*P*32+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      v,ref.data_ptr<C>(),zs.data_ptr<double>(),zo.data_ptr<double>(),out.data_ptr<double>(),P);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out.max();
}
