#include "common.cuh"
#include <vector>
using namespace layered;

// K1 model per (f,q), P pairs, L regions, S nonzero propagation sections,
// I grouped images: ~480*(L-1)+40*L+250*P+220*S+38*I explicit real FLOP.
// Network workspace writes 160*L B; pair output writes 32*P B. Layer/pair/image
// loads are cacheable. Transcendentals: L+P complex roots, 2*(L-2)+E+I complex
// exponentials (E counts source/propagation phases, for open exterior regions).
// Complex divides: 3*L+12*(L-1)+6*P+3*S, including the hoisted image denominator.
// Each ex uses exp/sin/cos; each root uses hypot and two sqrt plus divisions.
// Arithmetic-only FMA utilization excludes these costs and serial recurrences;
// bench includes measured exp/root/divide ceilings in its empirical roofline.
// Every output and recursive sum has one owner; no floating-point atomics.
__global__ void nodes_kernel(const double* gl,const double* gw,const double* f,
                            const C* media,const double* sigma,int layers,
                            C* q,C* weights,int* kinds,int F,int Q,double limit,
                            double rho_min,double rho_max,int linear,int first) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=F*Q) return;
  int j=first+t%Q, panel=j/32, n=j%32;
  double omega=2*pi*f[t/Q], magnitude=0;
  for(int l=0;l<layers;++l) magnitude=fmax(magnitude,abs(eps(media,sigma,l,omega)));
  double km=omega/c0*sqrt(magnitude);
  double u=(gl[n]+1)*0.5, v, jac;
  C path, derivative;
  double h=fmin(0.15*km,8.0/rho_max), L=2*km;
  double end=rho_min==0 ? limit : fmax(L,32/rho_min);
  double transition=fmin(end,fmax(L,16*pi/rho_max));
  int kind=0;
  if(panel<16) {
    // Geometric resolution keeps the detour resolved even when conductive
    // media put the vacuum branch far below the largest material wave number.
    v=L*expm1(6*(panel+u)/16)/expm1(6.0);
    jac=6*(v+L/expm1(6.0))/16;
    // Passive poles approach from Im(q)>0. The lower detour therefore has
    // the same limiting absorption prescription as the real-axis oracle.
    path=C(v,-h*sin(pi*v/L));
    derivative=C(1,-h*pi/L*cos(pi*v/L));
  } else if(panel<32) {
    double a=log(transition/L);
    v=L*exp(a*(panel-16+u)/16);
    jac=v*a/16;
    path=C(v); derivative=C(1);
  } else if(panel<32+linear) {
    jac=(end-transition)/linear;
    path=C(transition+jac*(panel-32+u));derivative=C(1);
  } else {
    int p=panel-32-linear;
    kind=p<8 ? 1 : -1;
    double height=48/rho_min;
    path=C(end,kind*height*((p%8)+u)/8);
    jac=height/8;
    // J0=(H0^(1)+H0^(2))/2. Beyond every material branch point, close
    // H1 upward and H2 downward: their exp(+/-i*q*rho) factors decay.
    derivative=C(0,kind*0.5);
  }
  q[t]=path;
  weights[t]=derivative*(0.5*gw[n]*jac);
  kinds[t]=kind;
}

std::vector<at::Tensor> layered_nodes(at::Tensor gl,at::Tensor gw,at::Tensor f,
                                     at::Tensor media,at::Tensor sigma,
                                     double limit,double rho_min,double rho_max,
                                     int64_t linear,int64_t first,int64_t Q) {
  c10::cuda::CUDAGuard guard(f.device());
  auto q=at::empty({f.numel(),Q},f.options().dtype(at::kComplexDouble));
  auto w=at::empty_like(q);
  auto kinds=at::empty({f.numel(),Q},f.options().dtype(at::kInt));
  int count=q.numel();
  nodes_kernel<<<(count+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      gl.data_ptr<double>(),gw.data_ptr<double>(),f.data_ptr<double>(),media.data_ptr<C>(),
      sigma.data_ptr<double>(),media.numel(),q.data_ptr<C>(),
      w.data_ptr<C>(),kinds.data_ptr<int>(),f.numel(),Q,limit,rho_min,rho_max,linear,first);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q,w,kinds};
}

__global__ void images_kernel(const C* media,const double* sigma,const double* frequencies,
    const int64_t* sources,const int64_t* offsets,const int64_t* events,const double* multiplicities,
    C* out,int F,int I) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=F*2*I) return;
  int i=t%I, c=(t/I)%2, f=t/(2*I);
  double w=2*pi*frequencies[f];
  C coefficient=c ? C(1)/eps(media,sigma,sources[i],w) : C(1);
  for(int e=offsets[i];e<offsets[i+1];++e) {
    int a=events[3*e], b=events[3*e+1], transmission=events[3*e+2];
    C r;
    if(b<0) r=C(-1);
    else if(c) {
      C ea=eps(media,sigma,a,w),eb=eps(media,sigma,b,w);
      r=(ea-eb)/(ea+eb);
    } else r=C(0);
    coefficient*=transmission ? C(1)+r : r;
  }
  out[t]=coefficient*multiplicities[i];
}

at::Tensor layered_images(at::Tensor media,at::Tensor sigma,at::Tensor f,
    at::Tensor sources,at::Tensor offsets,at::Tensor events,at::Tensor distances,at::Tensor multiplicities) {
  c10::cuda::CUDAGuard guard(f.device());
  auto out=at::empty({f.numel(),2,distances.numel()},media.options());
  int n=out.numel();
  images_kernel<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      media.data_ptr<C>(),sigma.data_ptr<double>(),f.data_ptr<double>(),sources.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),events.data_ptr<int64_t>(),multiplicities.data_ptr<double>(),
      out.data_ptr<C>(),f.numel(),distances.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

__global__ void group_kernel(const C* paths,const int64_t* ranges,C* images,int groups,int paths_count,int F) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=F*2*groups) return;
  int g=t%groups,row=t/groups;
  C sum(0),correction(0);
  for(int i=ranges[g];i<ranges[g+1];++i) {
    C value=paths[row*paths_count+i]-correction;
    C next=sum+value;
    correction=(next-sum)-value;sum=next;
  }
  images[t]=sum;
}
at::Tensor layered_group_images(at::Tensor paths,at::Tensor ranges) {
  c10::cuda::CUDAGuard guard(paths.device());
  auto out=at::empty({paths.size(0),2,ranges.numel()-1},paths.options());
  int n=out.numel();
  group_kernel<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      paths.data_ptr<C>(),ranges.data_ptr<int64_t>(),out.data_ptr<C>(),ranges.numel()-1,
      paths.size(2),paths.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// For each (f,q), cache kz, TE/TM/scaled-difference impedances and both
// boundary reflections. Two O(L) sweeps replace repeated O(L^2) recursions
// in every plane pair; the workspace has no fixed layer-count limit.
__device__ Modes load_modes(const C* network,int layer,int offset) {
  const C* v=network+10*layer+offset;
  return {v[0],v[1],v[2]};
}
__device__ void store_modes(C* network,int layer,int offset,Modes v) {
  C* p=network+10*layer+offset;p[0]=v.h;p[1]=v.e;p[2]=v.d;
}
__global__ void network_kernel(Stack s,const C* q,const double* f,C* network,int F,int Q) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=F*Q) return;
  C kr=q[t];double w=2*pi*f[t/Q];
  C* data=network+t*s.n*10;
  for(int l=0;l<s.n;++l) {
    C k=kz(s,l,kr,w),epsilon=eps(s.media,s.sigma,l,w);
    data[10*l]=k;
    store_modes(data,l,1,{C(1)/k,k/((w/c0)*(w/c0)*epsilon),C(1)/(epsilon*k)});
  }
  for(int direction=-1;direction<=1;direction+=2) {
    int end=direction>0 ? s.n-1:0;
    int offset=direction>0 ? 4:7;
    Modes r,one(C(1));
    store_modes(data,end,offset,r);
    for(int l=end-direction;l>=0 && l<s.n;l-=direction) {
      int next=l+direction;
      if((next==0 && s.pec_bottom)||(next==s.n-1 && s.pec_top)) r=Modes(C(-1));
      else {
        Modes a=load_modes(data,l,1),b=load_modes(data,next,1);
        Modes local=(b-a)/(b+a);
        if(next>0 && next<s.n-1)
          r=r*Modes(ex(C(0,2)*data[10*next]*(s.edges[next]-s.edges[next-1])));
        r=(local+r)/(one+local*r);
      }
      store_modes(data,l,offset,r);
    }
  }
}
__device__ Modes voltage(Stack s,const C* data,double zs,double zo) {
  if((s.pec_bottom && zs==s.edges[0]) || (s.pec_top && zo==s.edges[s.n-2])) return Modes();
  int a=region(s,zs),b=region(s,zo);
  C k=data[10*a];
  Modes one(C(1));
  Modes up=a<s.n-1 ? load_modes(data,a,4)*Modes(ex(C(0,2)*k*(s.edges[a]-zs))) : Modes();
  Modes down=a>0 ? load_modes(data,a,7)*Modes(ex(C(0,2)*k*(zs-s.edges[a-1]))) : Modes();
  Modes v=Modes(C(0,.5))*load_modes(data,a,1)*(one+up)*(one+down)/(one-up*down);
  double z=zs;
  for(int l=a;l<=b;++l) {
    double target=l==b ? zo:s.edges[l];
    if(target!=z) {
      C kl=data[10*l];
      Modes r=load_modes(data,l,4);
      Modes u0=l<s.n-1 ? r*Modes(ex(C(0,2)*kl*(s.edges[l]-z))) : Modes();
      Modes u1=l<s.n-1 ? r*Modes(ex(C(0,2)*kl*(s.edges[l]-target))) : Modes();
      v=v*Modes(ex(C(0,1)*kl*(target-z)))*(one+u1)/(one+u0);
    }
    z=target;
  }
  return v;
}

__global__ void spectral_kernel(Stack s,const C* q,const double* f,const double* planes,
    const int64_t* pairs,const int64_t* offsets,const double* distances,const C* coeff,
    const C* tails,const C* network_data,C* out,int F,int Q,int P,int I,double regularizer,bool subtract) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=F*Q*P) return;
  int p=t%P,j=(t/P)%Q,fi=t/(P*Q);
  C kr=q[fi*Q+j]; double w=2*pi*f[fi];
  double zs=planes[pairs[2*p]],zo=planes[pairs[2*p+1]];
  Modes network=voltage(s,network_data+(fi*Q+j)*s.n*10,zs,zo);
  C ah=network.h,ad=network.d;
  if(subtract) {
    C st_h(0),st_d(0),ch(0),cd(0);
    C inverse=C(1)/(2.0*kr);
    for(int image=offsets[p];image<offsets[p+1];++image) {
      C decay=ex(-kr*distances[image])*inverse;
      C th=coeff[fi*2*I+image]*decay-ch,td=coeff[(fi*2+1)*I+image]*decay-cd;
      C nh=st_h+th,nd=st_d+td;
      ch=(nh-st_h)-th;cd=(nd-st_d)-td;st_h=nh;st_d=nd;
    }
    // Inverse Hankel transform of C3/(q^2+a^2)^(3/2) is
    // C3*exp(-a*rho)/(2*pi*a); the residual tail is O(T^-3).
    C u=kr*kr+regularizer*regularizer,v=root(u);
    if(v.real()<0) v=-v;
    ah-=st_h+tails[(fi*P+p)*2]/(u*v);
    ad-=st_d+tails[(fi*P+p)*2+1]/(u*v);
  }
  out[((fi*Q+j)*P+p)*2]=ah;
  out[((fi*Q+j)*P+p)*2+1]=ad;
}

at::Tensor layered_spectral(at::Tensor media,at::Tensor sigma,at::Tensor edges,at::Tensor planes,
    at::Tensor pairs,at::Tensor f,at::Tensor q,at::Tensor offsets,at::Tensor distances,
    at::Tensor coeff,at::Tensor tails,double a,
    int64_t bottom,int64_t top,bool subtract) {
  c10::cuda::CUDAGuard guard(f.device());
  int P=pairs.size(0),Q=q.size(1),F=f.numel();
  auto out=at::empty({F,Q,P*2},media.options());
  auto network=at::empty({F,Q,media.numel(),10},media.options());
  Stack s{media.data_ptr<C>(),sigma.data_ptr<double>(),edges.data_ptr<double>(),
          int(media.numel()),int(bottom),int(top)};
  network_kernel<<<(F*Q+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      s,q.data_ptr<C>(),f.data_ptr<double>(),network.data_ptr<C>(),F,Q);
  spectral_kernel<<<(F*Q*P+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      s,q.data_ptr<C>(),f.data_ptr<double>(),planes.data_ptr<double>(),pairs.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),distances.data_ptr<double>(),coeff.data_ptr<C>(),
      tails.data_ptr<C>(),network.data_ptr<C>(),out.data_ptr<C>(),F,Q,P,distances.numel(),a,subtract);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

__global__ void tail_kernel(Stack s,const double* planes,const int64_t* pairs,
                            const double* f,C* out,int P,int F) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=P*F) return;
  int p=t%P,fi=t/P;
  C a(0),b(0);
  double z=planes[pairs[2*p]], zo=planes[pairs[2*p+1]], w=2*pi*f[fi], k0=w/c0;
  if(z==zo) {
    int l=region(s,z);
    C ea=eps(s.media,s.sigma,l,w), eb=ea;
    if(l>0 && z==s.edges[l-1]) eb=eps(s.media,s.sigma,l-1,w);
    if((l==1 && s.pec_bottom && z==s.edges[0]) ||
       (l==s.n-1 && s.pec_top && z==s.edges[s.n-2])) ea=eb=C(0);
    if(ea!=C(0)) {
      a=k0*k0*(ea+eb)/8.0;
      b=k0*k0*ea*eb/((ea+eb)*(ea+eb));
    }
  }
  out[2*t]=a;out[2*t+1]=b;
}

at::Tensor layered_tail(at::Tensor media,at::Tensor sigma,at::Tensor edges,
                        at::Tensor planes,at::Tensor pairs,at::Tensor f,int64_t bottom,int64_t top) {
  c10::cuda::CUDAGuard guard(f.device());
  auto out=at::empty({f.numel(),pairs.size(0),2},media.options());
  Stack s{media.data_ptr<C>(),sigma.data_ptr<double>(),edges.data_ptr<double>(),int(media.numel()),int(bottom),int(top)};
  int count=f.numel()*pairs.size(0);
  tail_kernel<<<(count+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      s,planes.data_ptr<double>(),pairs.data_ptr<int64_t>(),f.data_ptr<double>(),out.data_ptr<C>(),
      pairs.size(0),f.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

__global__ void bound_kernel(const C* samples,const C* q,double* bound,int F,int Q,int M,
                             double limit,double rho_min,int bin,int first,int linear) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=F*M) return;
  int f=t/M,m=t%M;double coefficient=0;
  int endpoints=rho_min==0 ? 1:2;
  for(int side=0;side<endpoints;++side) {
    int start=(32+linear+(rho_min==0 ? 0:8*(side+1)))*32-32;
    for(int j=max(0,start-first);j<min(Q,start+32-first);++j) {
      double radius=abs(q[f*Q+j]);
      double value=abs(samples[(f*Q+j)*M+m]);
      if(rho_min==0) value*=pow(radius,5)/(6*pi*limit*limit*limit);
      else value*=sqrt(2/(pi*radius*rho_min))*exp(-fabs(q[f*Q+j].imag())*rho_min)*radius/(2*pi*rho_min);
      coefficient=fmax(coefficient,value);
    }
  }
  // A posteriori asymptotic envelope estimate, not a rigorous interval bound:
  // |S(q)| <= C5/q^5 gives |tail(rho)| <= C5/(6*pi*T^3) on the real axis.
  // The two Hankel tails instead have an exp(-48*rho/rho_min) envelope.
  bound[bin*F*M+t]=fmax(bound[bin*F*M+t],coefficient);
}
void layered_tail_bound(at::Tensor samples,at::Tensor q,double limit,double rho_min,
                        at::Tensor bounds,int64_t bin,int64_t first,int64_t linear) {
  c10::cuda::CUDAGuard guard(samples.device());
  int n=samples.size(0)*samples.size(2);
  bound_kernel<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      samples.data_ptr<C>(),q.data_ptr<C>(),bounds.data_ptr<double>(),samples.size(0),q.size(1),samples.size(2),
      limit,rho_min,bin,first,linear);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Batched TL cache for vertical-source diagnostics.
at::Tensor layered_vertical_network(at::Tensor media,at::Tensor sigma,at::Tensor edges,
    at::Tensor f,at::Tensor q,int64_t bottom,int64_t top) {
  c10::cuda::CUDAGuard guard(f.device());
  int F=f.numel(),Q=q.size(1);
  auto data=at::empty({F,Q,media.numel(),10},media.options());
  Stack s{media.data_ptr<C>(),sigma.data_ptr<double>(),edges.data_ptr<double>(),
          int(media.numel()),int(bottom),int(top)};
  network_kernel<<<(F*Q+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      s,q.data_ptr<C>(),f.data_ptr<double>(),data.data_ptr<C>(),F,Q);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return data;
}

__device__ Modes reflected_wave(Stack s,const C* data,int layer,double z,bool upper) {
  bool exterior=upper ? layer==s.n-1:layer==0;
  if(exterior) return Modes();
  double distance=upper ? s.edges[layer]-z:z-s.edges[layer-1];
  return load_modes(data,layer,upper?4:7)*Modes(ex(C(0,2)*data[10*layer]*distance));
}

__global__ void vertical_spectral_kernel(Stack s,const C* q,const double* f,
    const double* zs,const double* zo,const C* data,C* out,int F,int Q,int P,int components) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=F*Q*P) return;
  int p=t%P,j=(t/P)%Q,fi=t/(P*Q);
  bool reverse=zs[p]>zo[p];
  double source=fmin(zs[p],zo[p]),observer=fmax(zs[p],zo[p]),w=2*pi*f[fi],k0=w/c0;
  C kr=q[fi*Q+j];
  const C* network=data+(fi*Q+j)*s.n*10;
  int a=region(s,source),b=region(s,observer);
  if(s.pec_top && source==s.edges[s.n-2]) a=s.n-2;
  if(s.pec_top && observer==s.edges[s.n-2]) b=s.n-2;
  Modes one(C(1)),up=reflected_wave(s,network,a,source,true);
  Modes down=reflected_wave(s,network,a,source,false);
  Modes upper=reflected_wave(s,network,b,observer,true);
  // Factor out (1+r_minus(source)) and (1+r_plus(observer)) before
  // differentiation. A PEC endpoint makes these factors zero: logarithmic
  // derivatives would form 0*infinity although the potential is finite.
  Modes common=Modes(C(0,.5))*load_modes(network,a,1)/(one-up*down);
  double position=source;
  for(int l=a;l<=b;++l) {
    double target=l==b ? observer:s.edges[l];
    if(target!=position) common=common*Modes(ex(C(0,1)*network[10*l]*(target-position)));
    if(l<b) common=common*(one+load_modes(network,l,4))/
                             (one+reflected_wave(s,network,l+1,target,true));
    position=target;
  }
  C ka=network[10*a],kb=network[10*b];
  Modes v=common*(one+down)*(one+upper);
  Modes vs=common*Modes(C(0,-1)*ka)*(one-down)*(one+upper);
  Modes vz=common*Modes(C(0,1)*kb)*(one+down)*(one-upper);
  Modes mixed=common*Modes(ka*kb)*(one-down)*(one-upper);
  // Formulation C, exp(-i*w*t), order (zz,zrho,rhoz,phi). Scaled TE-TM
  // differences propagate algebraically, avoiding cancellation near q=0.
  C zz,zrho,rhoz;
  C ks2=k0*k0*eps(s.media,s.sigma,a,w),ko2=k0*k0*eps(s.media,s.sigma,b,w),q2=kr*kr;
  if(abs(q2)>4*fmax(abs(ks2),abs(ko2))) {
    // Expand q^4-kzs^2*kzo^2 algebraically. Subtracting two O(q/k0^2)
    // terms would destroy the quasi-static zz limit at large q.
    zz=mixed.h/q2+mixed.e*((ks2+ko2)*q2-ks2*ko2)/(q2*ka*ka*kb*kb);
    zrho=(vz.h-ko2*vz.e/(kb*kb))/kr;
    rhoz=-(vs.h-ks2*vs.e/(ka*ka))/kr;
  } else {
    zz=mixed.d/(k0*k0)+q2*mixed.e/(ka*ka*kb*kb);
    zrho=kr*(vz.d/(k0*k0)-vz.e/(kb*kb));
    rhoz=-kr*(vs.d/(k0*k0)-vs.e/(ka*ka));
  }
  out[components*t]=zz;
  out[components*t+1]=reverse ? -rhoz:zrho;
  out[components*t+2]=reverse ? -zrho:rhoz;
  out[components*t+3]=v.d;
  if(components==5) out[components*t+4]=v.h;
}

at::Tensor layered_vertical_spectral(at::Tensor media,at::Tensor sigma,at::Tensor edges,
    at::Tensor f,at::Tensor q,at::Tensor zs,at::Tensor zo,int64_t bottom,int64_t top,bool horizontal) {
  c10::cuda::CUDAGuard guard(f.device());
  TORCH_CHECK(zs.numel()==zo.numel(),"vertical spectral coordinates must have equal lengths");
  int F=f.numel(),Q=q.size(1),P=zs.numel();
  int components=horizontal ? 5:4;
  auto out=at::empty({F,Q,P,components},media.options());
  auto network=layered_vertical_network(media,sigma,edges,f,q,bottom,top);
  Stack s{media.data_ptr<C>(),sigma.data_ptr<double>(),edges.data_ptr<double>(),
          int(media.numel()),int(bottom),int(top)};
  vertical_spectral_kernel<<<(F*Q*P+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      s,q.data_ptr<C>(),f.data_ptr<double>(),zs.data_ptr<double>(),zo.data_ptr<double>(),
      network.data_ptr<C>(),out.data_ptr<C>(),F,Q,P,components);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

__global__ void vertical_bound_kernel(const C* samples,const C* q,double* bound,int F,int Q,int M,
                             double limit,double rho_min,int bin,int first,int linear) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=F*M) return;
  int f=t/M,m=t%M;double coefficient=0;
  int endpoints=rho_min==0 ? 1:2;
  for(int side=0;side<endpoints;++side) {
    int start=(32+linear+(rho_min==0 ? 0:8*(side+1)))*32-32;
    for(int j=max(0,start-first);j<min(Q,start+32-first);++j) {
      double radius=abs(q[f*Q+j]);
      double value=abs(samples[(f*Q+j)*M+m]);
      if(rho_min==0) value*=pow(radius,3)/(2*pi*limit);
      else value*=sqrt(2/(pi*radius*rho_min))*exp(-fabs(q[f*Q+j].imag())*rho_min)*radius/(2*pi*rho_min);
      coefficient=fmax(coefficient,value);
    }
  }
  // A posteriori envelope: |S(q)| <= C3/q^3 gives C3/(2*pi*T).
  // Static reflected images leave an exponentially damped q^-3 remainder.
  // The two Hankel tails instead have an exp(-48*rho/rho_min) envelope.
  bound[bin*F*M+t]=fmax(bound[bin*F*M+t],coefficient);
}
void layered_vertical_tail_bound(at::Tensor samples,at::Tensor q,double limit,double rho_min,
                        at::Tensor bounds,int64_t bin,int64_t first,int64_t linear) {
  c10::cuda::CUDAGuard guard(samples.device());
  int n=samples.size(0)*samples.size(2);
  vertical_bound_kernel<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      samples.data_ptr<C>(),q.data_ptr<C>(),bounds.data_ptr<double>(),samples.size(0),q.size(1),samples.size(2),
      limit,rho_min,bin,first,linear);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
