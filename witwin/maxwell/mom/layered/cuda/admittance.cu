#include "triangle_static.cuh"
#include <vector>
using namespace layered;
using at::Tensor;

// Surface-admittance interior operators of one closed body (homogeneous
// media). For every ordered face pair (t,s) and free-vertex slots (i,j):
//   V[q][i][j] = int int g_q (x-a_i).(y-b_j),  q in {c, b, F}
//   K[q][i][j] = int int (x-a_i).(grad_x g_q x (n_s x (y-b_j))),  q in {b, F}
// (rotated sources: the trial field e is expanded in RT and M = e x n).
//   P[q]       = int int g_q,  q in {c, b}
// g_c,g_b = exp(ikR)/(4 pi R) and F=(g_c-g_b)/(kc^2-kb^2), bounded at R=0.
// Touching pairs use joint Duffy (Sauter-Schwab) rules with geometric radial
// panels for |kc| h >> 1; near pairs observer Gauss nodes times an inner
// polar rule (asinh angular and radial maps); far pairs product rules.
// Each pair is reduced by exactly one thread, warp or block in a fixed
// order, and gathers own their outputs: no floating-point atomics.
namespace {
constexpr int VALUES=47;  // 27 V + 18 K + 2 P complex accumulators
struct Pair {C v[27],k[18],p[2];};
struct Rules {int touching,panels,outer,angular,radial,far;};
// taylor[n-1] = c_n = i^n (kc^n - kb^n) / (n! 4 pi dk2) and taylor[TAYLOR+n-1] = c_n (n-1), n = 1..TAYLOR:
// the r-independent factors of the divided-difference series, formed once per launch (taylor_kernel)
// with the expressions the per-node series used, so every node evaluation is bitwise unchanged.
constexpr int TAYLOR=26;
struct Media {C kc,kb,dk2;const C* taylor;};

__device__ inline void helmholtz(C k,double r,C& g,C& dg) {
  g=ex(C(0,1)*k*r)/(4*pi*r);dg=g*(C(0,1)*k-1/r);
}
// Divided difference with a Taylor series for |kc| R < 1/2 (no cancellation).
__device__ inline void divided(Media m,double r,C& g,C& dg) {
  if(abs(m.kc)*r<.5) {
    C sum(0),dsum(0),power(1);
    for(int n=1;n<=TAYLOR;++n) {
      sum+=m.taylor[n-1]*power;
      if(n>=2) dsum+=m.taylor[TAYLOR+n-1]*power/r;
      power*=r;
    }
    g=sum;dg=dsum;return;
  }
  C ec=ex(C(0,1)*m.kc*r),eb=ex(C(0,1)*m.kb*r);
  g=(ec-eb)/(4*pi*r*m.dk2);
  dg=(C(0,1)*m.kc*ec-C(0,1)*m.kb*eb)/(4*pi*r*m.dk2)-g/r;
}
__global__ void taylor_kernel(Media m,C* taylor) {
  C pc(1),pb(1);double factorial=1;
  C in(1);
  for(int n=1;n<=TAYLOR;++n) {
    pc*=m.kc;pb*=m.kb;in*=C(0,1);factorial*=n;
    C c=in*(pc-pb)/(factorial*4*pi*m.dk2);
    taylor[n-1]=c;taylor[TAYLOR+n-1]=c*double(n-1);
  }
}
// remainder=true subtracts 1/(4 pi R) from g_c and g_b (their static part is
// integrated in closed form separately); F has no static part.
__device__ inline void accumulate(Pair& a,Media m,const V3* tv,const V3* sv,V3 x,V3 y,double w,bool magnetic,
    bool remainder=false) {
  V3 ns=cross(sv[1]-sv[0],sv[2]-sv[0]);ns=ns*(1/norm(ns));
  V3 d=x-y;double r=norm(d);
  C g[3],dg[3];
  helmholtz(m.kc,r,g[0],dg[0]);helmholtz(m.kb,r,g[1],dg[1]);divided(m,r,g[2],dg[2]);
  if(remainder) for(int q=0;q<2;++q) {g[q]-=1/(4*pi*r);dg[q]+=1/(4*pi*r*r);}
  V3 f[3]={x-tv[0],x-tv[1],x-tv[2]},h[3]={y-sv[0],y-sv[1],y-sv[2]};
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
    double fh=dot(f[i],h[j]);
    for(int q=0;q<3;++q) a.v[9*q+3*i+j]+=g[q]*(w*fh);
    if(magnetic) {
      double t=dot(f[i],cross(d,cross(ns,h[j])))*w/r;
      a.k[3*i+j]+=dg[1]*t;a.k[9+3*i+j]+=dg[2]*t;
    }
  }
  a.p[0]+=g[0]*w;a.p[1]+=g[1]*w;
}
__device__ inline void clear(Pair& a) {
  for(int i=0;i<27;++i) a.v[i]=C(0);
  for(int i=0;i<18;++i) a.k[i]=C(0);
  a.p[0]=a.p[1]=C(0);
}
__device__ inline C& slot(Pair& a,int i) {return i<27 ? a.v[i] : i<45 ? a.k[i-27] : a.p[i-45];}

struct Geometry {
  const double* vertices;const int64_t* faces;int F;
  const double* gauss;int G;          // Gauss-Legendre nodes/weights on [0,1] for orders 1..G
};
__device__ inline V3 corner(Geometry g,int f,int i) {
  const double* v=g.vertices+3*g.faces[3*f+i];return {v[0],v[1],v[2]};
}
__device__ inline double gnode(Geometry g,int order,int i) {return g.gauss[2*(order*(order-1)/2+i)];}
__device__ inline double gweight(Geometry g,int order,int i) {return g.gauss[2*(order*(order-1)/2+i)+1];}
__device__ inline double area2(V3 a,V3 b,V3 c) {return norm(cross(b-a,c-a));}
__device__ inline V3 duffy(V3 a,V3 b,V3 c,double u,double v) {return a+((b-a)*(1-v)+(c-a)*v)*u;}

// Shared-vertex count and the classification used by the dispatch.
__device__ inline int category(Geometry g,int t,int s,double& ratio) {
  int shared=0;
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) shared+=g.faces[3*t+i]==g.faces[3*s+j];
  V3 ct=(corner(g,t,0)+corner(g,t,1)+corner(g,t,2))*(1./3),cs=(corner(g,s,0)+corner(g,s,1)+corner(g,s,2))*(1./3);
  double rt=0,rs=0;
  for(int i=0;i<3;++i) {rt=fmax(rt,norm(corner(g,t,i)-ct));rs=fmax(rs,norm(corner(g,s,i)-cs));}
  ratio=norm(ct-cs)/(rt+rs);
  if(shared==3) return 0;
  if(shared==2) return 1;
  if(shared==1) return 2;
  return ratio<2 ? 3 : 4;
}

// Far pairs: product Duffy rules sized by distance ratio and by |kc| times
// the larger triangle size.
__device__ inline int far_order(Geometry g,Media m,Rules rules,const V3* tv,const V3* sv,double ratio) {
  double size=fmax(fmax(norm(tv[1]-tv[0]),norm(tv[2]-tv[1])),fmax(norm(sv[1]-sv[0]),norm(sv[2]-sv[1])));
  size=fmax(size,fmax(norm(tv[0]-tv[2]),norm(sv[0]-sv[2])));
  int order=rules.far+(ratio>6 ? 0 : ratio>3 ? 1 : 2);
  // Resolve exp(i kc R) only where it is not exponentially negligible.
  double gap=ratio*size/2;
  if(m.kc.imag()*gap<40) order=max(order,min(10,int(ceil(abs(m.kc)*size))+2));
  return order;
}

// Serial far rule: one thread per pair, nodes accumulated in (i,j,k,l)
// order. Test oracle of far_kernel (admittance_operators_serial_far).
__global__ void far_serial_kernel(Geometry g,Media m,Rules rules,int t0,int64_t q0,int64_t P,Pair* out,int8_t* cls) {
  int64_t q=q0+int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(q>=P) return;
  int t=t0+q/g.F,s=q%g.F;double ratio;
  int c=category(g,t,s,ratio);cls[q]=c;
  if(c!=4) return;
  V3 tv[3]={corner(g,t,0),corner(g,t,1),corner(g,t,2)},sv[3]={corner(g,s,0),corner(g,s,1),corner(g,s,2)};
  int order=far_order(g,m,rules,tv,sv,ratio);
  Pair a;clear(a);
  double at=area2(tv[0],tv[1],tv[2]),as=area2(sv[0],sv[1],sv[2]);
  for(int i=0;i<order;++i) for(int j=0;j<order;++j) {
    double ui=gnode(g,order,i),vj=gnode(g,order,j);
    V3 x=duffy(tv[0],tv[1],tv[2],ui,vj);double wx=gweight(g,order,i)*gweight(g,order,j)*ui*at;
    for(int k=0;k<order;++k) for(int l=0;l<order;++l) {
      double uk=gnode(g,order,k),vl=gnode(g,order,l);
      V3 y=duffy(sv[0],sv[1],sv[2],uk,vl);
      accumulate(a,m,tv,sv,x,y,wx*gweight(g,order,k)*gweight(g,order,l)*uk*as,true);
    }
  }
  out[q]=a;
}

// Far pairs, one warp per pair. Lanes evaluate 32 consecutive nodes of the
// (i,j,k,l) product rule into shared memory (the 29 real factors of every
// accumulate() term), then each lane owns up to three of the 94 real output
// slots and adds that batch's terms in node order. Every term and every
// addition is the same IEEE operation as in far_serial_kernel (--fmad=false,
// c10::complex scales componentwise), so the outputs are bitwise equal.
constexpr int FAR_WARPS=4,FAR_TERMS=29;  // terms: g[3],dg[1],dg[2] (10), w*fh (9), t (9), w
__global__ void __launch_bounds__(32*FAR_WARPS) far_kernel(Geometry g,Media m,Rules rules,int t0,int64_t q0,
    int64_t P,Pair* out,int8_t* cls) {
  __shared__ double terms[FAR_WARPS][32*FAR_TERMS];
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  int64_t q=q0+int64_t(blockIdx.x)*FAR_WARPS+warp;if(q>=P) return;
  int t=t0+q/g.F,s=q%g.F;double ratio;
  int c=category(g,t,s,ratio);
  if(lane==0) cls[q]=c;
  if(c!=4) return;
  V3 tv[3]={corner(g,t,0),corner(g,t,1),corner(g,t,2)},sv[3]={corner(g,s,0),corner(g,s,1),corner(g,s,2)};
  int order=far_order(g,m,rules,tv,sv,ratio),o2=order*order,nodes=o2*o2;
  double at=area2(tv[0],tv[1],tv[2]),as=area2(sv[0],sv[1],sv[2]);
  V3 ns=cross(sv[1]-sv[0],sv[2]-sv[0]);ns=ns*(1/norm(ns));
  // Owned real slots v = lane+32r of the Pair (v/2 complex slot, v%2 part):
  // factor offset in terms[] (g or dg component) and scalar offset.
  int factor[3],scalar[3],owned=0;double sum[3]={0,0,0};
  for(int r=0;r<3;++r) {
    int v=lane+32*r;if(v>=2*VALUES) break;
    int cs=v/2,part=v%2;
    if(cs<27) {factor[r]=2*(cs/9)+part;scalar[r]=10+cs%9;}
    else if(cs<45) {factor[r]=2*(3+(cs-27)/9)+part;scalar[r]=19+(cs-27)%9;}
    else {factor[r]=2*(cs-45)+part;scalar[r]=28;}
    owned=r+1;
  }
  double* mine=terms[warp];
  for(int base=0;base<nodes;base+=32) {
    int n=base+lane;
    if(n<nodes) {
      int i=n/(o2*order),j=(n/o2)%order,k=(n/order)%order,l=n%order;
      double ui=gnode(g,order,i),vj=gnode(g,order,j);
      V3 x=duffy(tv[0],tv[1],tv[2],ui,vj);double wx=gweight(g,order,i)*gweight(g,order,j)*ui*at;
      double uk=gnode(g,order,k),vl=gnode(g,order,l);
      V3 y=duffy(sv[0],sv[1],sv[2],uk,vl);
      double w=wx*gweight(g,order,k)*gweight(g,order,l)*uk*as;
      // accumulate(..., magnetic=true, remainder=false) term factors.
      V3 d=x-y;double r=norm(d);
      C gq[3],dg[3];
      helmholtz(m.kc,r,gq[0],dg[0]);helmholtz(m.kb,r,gq[1],dg[1]);divided(m,r,gq[2],dg[2]);
      double* e=mine+FAR_TERMS*lane;
      for(int u=0;u<3;++u) {e[2*u]=gq[u].real();e[2*u+1]=gq[u].imag();}
      e[6]=dg[1].real();e[7]=dg[1].imag();e[8]=dg[2].real();e[9]=dg[2].imag();
      V3 f[3]={x-tv[0],x-tv[1],x-tv[2]},h[3]={y-sv[0],y-sv[1],y-sv[2]};
      for(int a=0;a<3;++a) for(int b=0;b<3;++b) {
        double fh=dot(f[a],h[b]);
        e[10+3*a+b]=w*fh;
        e[19+3*a+b]=dot(f[a],cross(d,cross(ns,h[b])))*w/r;
      }
      e[28]=w;
    }
    __syncwarp();
    int count=min(32,nodes-base);
    for(int b=0;b<count;++b) {
      const double* e=mine+FAR_TERMS*b;
      for(int r=0;r<owned;++r) sum[r]+=e[factor[r]]*e[scalar[r]];
    }
    __syncwarp();
  }
  double* o=reinterpret_cast<double*>(out+q);
  for(int r=0;r<owned;++r) o[lane+32*r]=sum[r];
}

// grad_x int 1/R dS = -sign(h) Omega n - sum_edges m_e int_edge 1/R dl (same
// edge logarithms and solid angle as triangle_static).
__device__ inline V3 triangle_static_gradient(const V3* v,V3 obs) {
  V3 n=cross(v[1]-v[0],v[2]-v[0]);n=n*(1/norm(n));
  double h=dot(obs-v[0],n);
  V3 a[3]={v[0]-obs,v[1]-obs,v[2]-obs};
  double r[3]={norm(a[0]),norm(a[1]),norm(a[2])};
  V3 grad{0,0,0};
  for(int i=0;i<3;++i) {
    int j=(i+1)%3;
    V3 edge=v[j]-v[i];double length=norm(edge);
    V3 tangent=edge*(1/length),outward=cross(tangent,n);
    double s0=dot(a[i],tangent),s1=s0+length,d=dot(a[i],outward),radius2=d*d+h*h,L=0;
    if(radius2!=0) {
      if(s0>=0) L=log1p(length*(1+(s0+s1)/(r[i]+r[j]))/(r[i]+s0));
      else if(s1<=0) L=log1p(length*(1-(s0+s1)/(r[i]+r[j]))/(r[j]-s1));
      else {double radius=sqrt(radius2);L=asinh(s1/radius)-asinh(s0/radius);}
    }
    grad=grad-outward*L;
  }
  if(h!=0) {
    double numerator=fabs(dot(a[0],cross(a[1],a[2])));
    double denominator=r[0]*r[1]*r[2]+dot(a[0],a[1])*r[2]+dot(a[1],a[2])*r[0]+dot(a[2],a[0])*r[1];
    grad=grad-n*((h>0?1:-1)*2*atan2(numerator,denominator));
  }
  return grad;
}
// Static (1/(4 pi R)) part of g_c and g_b at one observer node, source in
// closed form: int (y-b_j)/R = M + (p-b_j) I and, for K with rotated sources,
// int grad_x(1/R) x (n x (y-b_j)) = n I + h grad I + grad I x (n x (x-b_j)).
__device__ inline void accumulate_static(Pair& a,const V3* tv,const V3* sv,V3 x,double w,bool magnetic) {
  double triangle[9];
  for(int i=0;i<3;++i) {triangle[3*i]=sv[i].x;triangle[3*i+1]=sv[i].y;triangle[3*i+2]=sv[i].z;}
  StaticIntegral in=triangle_static(triangle,x);
  V3 ns=cross(sv[1]-sv[0],sv[2]-sv[0]);ns=ns*(1/norm(ns));
  double h=dot(x-sv[0],ns);V3 p=x-ns*h;
  double scale=w/(4*pi);
  V3 grad=magnetic ? triangle_static_gradient(sv,x) : V3{0,0,0};
  for(int i=0;i<3;++i) {
    V3 f=x-tv[i];
    for(int j=0;j<3;++j) {
      V3 inner=in.moment+(p-sv[j])*in.scalar;
      double value=dot(f,inner)*scale;
      a.v[3*i+j]+=C(value);a.v[9+3*i+j]+=C(value);
      if(magnetic) {
        V3 k=ns*in.scalar+grad*h+cross(grad,cross(ns,x-sv[j]));
        a.k[3*i+j]+=C(dot(f,k)*scale);
      }
    }
  }
  a.p[0]+=C(in.scalar*scale);a.p[1]+=C(in.scalar*scale);
}
// Graded outer node k of a touching pair: geometric t-panels toward the
// singular set (shared edge: t->0, shared vertex: t->0, self: three
// centroid sub-triangles graded toward each edge).
constexpr int OUTER_PANELS=10,OUTER_T=6,OUTER_S=8;
constexpr double OUTER_RATIO=.2;
constexpr int STATIC_LEVELS=6,STATIC_Q=4;
__device__ inline void graded_outer(Geometry g,int c,const V3* tv,const int* ti,int k,V3& x,double& w) {
  int s=k%OUTER_S,t=(k/OUTER_S)%OUTER_T,panel=(k/(OUTER_S*OUTER_T))%OUTER_PANELS,sub=k/(OUTER_S*OUTER_T*OUTER_PANELS);
  double lo=panel==0 ? 0 : pow(OUTER_RATIO,OUTER_PANELS-panel),hi=pow(OUTER_RATIO,OUTER_PANELS-1-panel);
  double u=lo+(hi-lo)*gnode(g,OUTER_T,t),wu=(hi-lo)*gweight(g,OUTER_T,t);
  double sv=gnode(g,OUTER_S,s),ws=gweight(g,OUTER_S,s);
  if(c==1) {
    V3 A=tv[ti[0]],B=tv[ti[1]],P=tv[ti[2]];
    x=(A+(B-A)*sv)*(1-u)+P*u;w=area2(A,B,P)*(1-u)*wu*ws;
  } else if(c==2) {
    V3 V=tv[ti[0]],B=tv[ti[1]],P=tv[ti[2]];
    x=V+((B-V)*(1-sv)+(P-V)*sv)*u;w=area2(V,B,P)*u*wu*ws;
  } else {
    V3 P=(tv[0]+tv[1]+tv[2])*(1./3),A=tv[sub],B=tv[(sub+1)%3];
    x=P+((A-P)*(1-sv)+(B-P)*sv)*(1-u);w=area2(P,A,B)*(1-u)*wu*ws;
  }
}

// Inner polar rule of the source triangle around x (any height): three signed
// edge sectors; angular asinh map about the perpendicular foot; radial
// rho=d sinh(tau) (d>0) or rho=rho_max s (d=0).
template<typename F> __device__ void polar(Geometry g,const V3* sv,V3 x,int qa,int qr,F&& node) {
  V3 n=cross(sv[1]-sv[0],sv[2]-sv[0]);n=n*(1/norm(n));
  double d=dot(x-sv[0],n),h0=fabs(d);
  V3 p=x-n*d;
  for(int e=0;e<3;++e) {
    V3 a=sv[e],b=sv[(e+1)%3];
    double length=norm(b-a);V3 tangent=(b-a)*(1/length);
    double ta=dot(a-p,tangent),tb=ta+length;
    V3 foot=a-tangent*ta;   // perpendicular foot of p on the edge line
    V3 m=foot-p;double he=norm(m);
    if(he<=1e-14*length) continue;
    m=m*(1/he);
    double sign=dot(cross(m,tangent),n)>0 ? 1 : -1;
    double ea=asinh(ta/he),eb=asinh(tb/he);
    for(int i=0;i<qa;++i) {
      double eta=ea+(eb-ea)*gnode(g,qa,i),c=cosh(eta);
      double wa=(eb-ea)*gweight(g,qa,i)/c;
      double rmax=he*c;V3 u=(m*he+tangent*(he*sinh(eta)))*(1/rmax);
      if(h0>0) {
        double tmax=asinh(rmax/h0);
        for(int j=0;j<qr;++j) {
          double tau=tmax*gnode(g,qr,j),rho=h0*sinh(tau),r=h0*cosh(tau);
          node(p+u*rho,sign*wa*tmax*gweight(g,qr,j)*rho*r);
        }
      } else {
        for(int j=0;j<qr;++j) {
          double sj=gnode(g,qr,j),rho=rmax*sj;
          node(p+u*rho,sign*wa*gweight(g,qr,j)*rmax*rmax*sj);
        }
      }
    }
  }
}

__device__ void reduce(Pair& a,Pair* out,double* shared) {
  for(int v=0;v<2*VALUES;++v) {
    C& value=slot(a,v/2);
    shared[threadIdx.x]=v%2 ? value.imag() : value.real();
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride/=2) {
      if(threadIdx.x<stride) shared[threadIdx.x]+=shared[threadIdx.x+stride];
      __syncthreads();
    }
    if(threadIdx.x==0) {
      C& o=slot(*out,v/2);
      o=v%2 ? C(o.real(),shared[0]) : C(shared[0],0);
    }
    __syncthreads();
  }
}


// One block per near or touching pair; threads stride over the rule nodes.
__global__ void near_kernel(Geometry g,Media m,Rules rules,const int64_t* list,int t0,Pair* out) {
  extern __shared__ double shared[];
  int64_t q=list[blockIdx.x];int t=t0+q/g.F,s=q%g.F;double ratio;
  int c=category(g,t,s,ratio);
  V3 tv[3]={corner(g,t,0),corner(g,t,1),corner(g,t,2)},sv[3]={corner(g,s,0),corner(g,s,1),corner(g,s,2)};
  V3 nt=cross(tv[1]-tv[0],tv[2]-tv[0]),ns=cross(sv[1]-sv[0],sv[2]-sv[0]);
  bool coplanar=fabs(dot(nt,ns))>=(1-1e-12)*norm(nt)*norm(ns) && fabs(dot(tv[0]-sv[0],ns))<=1e-12*norm(ns);
  Pair a;clear(a);
  double at=area2(tv[0],tv[1],tv[2]),as=area2(sv[0],sv[1],sv[2]);
  if(c==3) {
    int o=rules.outer,nodes=o*o;
    // Bounded remainder: observer Gauss nodes times the inner polar rule.
    for(int k=threadIdx.x;k<nodes;k+=blockDim.x) {
      double u=gnode(g,o,k/o),v=gnode(g,o,k%o);
      V3 x=duffy(tv[0],tv[1],tv[2],u,v);double wx=gweight(g,o,k/o)*gweight(g,o,k%o)*u*at;
      polar(g,sv,x,rules.angular,rules.radial,[&](V3 y,double w){accumulate(a,m,tv,sv,x,y,wx*w,true,true);});
    }
    // Static part: closed-form inner integral on a uniform 4^L observer
    // subdivision with sub-triangles no larger than the pair gap.
    double size=fmax(fmax(norm(tv[1]-tv[0]),norm(tv[2]-tv[1])),norm(tv[0]-tv[2]));
    double gap=1e300;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) gap=fmin(gap,norm(tv[i]-sv[j]));
    if(fabs(dot(nt,ns))>=.99*norm(nt)*norm(ns)) {
      double plane=1e300;
      for(int i=0;i<3;++i) plane=fmin(plane,fabs(dot(tv[i]-sv[0],ns))/norm(ns));
      gap=fmin(gap,plane);
    }
    int level=0;while(level<STATIC_LEVELS && size/double(1<<level)>gap) ++level;
    int64_t subs=int64_t(1)<<(2*level),points=subs*STATIC_Q*STATIC_Q;
    for(int64_t k=threadIdx.x;k<points;k+=blockDim.x) {
      int64_t sub=k/(STATIC_Q*STATIC_Q);int q=k%(STATIC_Q*STATIC_Q);
      V3 A=tv[0],B=tv[1],Cc=tv[2];
      for(int l=0;l<level;++l) {
        int child=(sub>>(2*(level-1-l)))&3;
        V3 ab=(A+B)*.5,bc=(B+Cc)*.5,ca=(Cc+A)*.5;
        if(child==0) {B=ab;Cc=ca;} else if(child==1) {A=ab;Cc=bc;}
        else if(child==2) {A=ca;B=bc;} else {A=ab;B=bc;Cc=ca;}
      }
      double u=gnode(g,STATIC_Q,q/STATIC_Q),v=gnode(g,STATIC_Q,q%STATIC_Q);
      V3 x=duffy(A,B,Cc,u,v);
      double w=gweight(g,STATIC_Q,q/STATIC_Q)*gweight(g,STATIC_Q,q%STATIC_Q)*u*area2(A,B,Cc);
      accumulate_static(a,tv,sv,x,w,!coplanar);
    }
  } else {
    int o=rules.touching,o2=o*o,o4=o2*o2;
    // Geometric radial panels: [0,1] split at 4^j/(|kc| size).
    double size=fmax(fmax(norm(tv[1]-tv[0]),norm(tv[2]-tv[1])),norm(tv[0]-tv[2]));
    double start=1/(abs(m.kc)*size+1e-300);
    int panels=1;while(panels<rules.panels && start*pow(4.,panels-1)<1) ++panels;
    int sectors=c==0 ? 6 : c==1 ? 6 : 2;
    int64_t total=int64_t(panels)*sectors*o4;
    // Shared vertices: common edge (a,b) by global id, non-shared c/d.
    int ti[3]={0,1,2},si[3]={0,1,2};
    if(c==1) {
      int sa=-1,sb=-1;
      for(int i=0;i<3;++i) for(int j=0;j<3;++j) if(g.faces[3*t+i]==g.faces[3*s+j]) {
        if(sa<0) sa=i; else sb=i;
      }
      if(g.faces[3*t+sa]>g.faces[3*t+sb]) {int w=sa;sa=sb;sb=w;}
      ti[0]=sa;ti[1]=sb;ti[2]=3-sa-sb;
      for(int j=0;j<3;++j) {bool shared=false;for(int i=0;i<3;++i) shared|=g.faces[3*t+i]==g.faces[3*s+j];if(!shared) si[2]=j;}
    }
    if(c==2) {
      for(int i=0;i<3;++i) for(int j=0;j<3;++j) if(g.faces[3*t+i]==g.faces[3*s+j]) {ti[0]=i;si[0]=j;}
      ti[1]=(ti[0]+1)%3;ti[2]=(ti[0]+2)%3;si[1]=(si[0]+1)%3;si[2]=(si[0]+2)%3;
    }
    int outer=OUTER_PANELS*OUTER_T*OUTER_S*(c==0 ? 3 : 1);
    for(int k=threadIdx.x;k<outer;k+=blockDim.x) {
      V3 x;double w;graded_outer(g,c,tv,ti,k,x,w);
      accumulate_static(a,tv,sv,x,w,!(c==0 || coplanar));
    }
    for(int64_t k=threadIdx.x;k<total;k+=blockDim.x) {
      int panel=k/(sectors*o4),sector=(k/o4)%sectors,r=k%o4;
      int i0=r/(o2*o),i1=(r/o2)%o,i2=(r/o)%o,i3=r%o;
      double lo=panel==0 ? 0 : fmin(1.,start*pow(4.,panel-1)),hi=panel==panels-1 ? 1 : fmin(1.,start*pow(4.,panel));
      if(hi<=lo) continue;
      double rho=lo+(hi-lo)*gnode(g,o,i0),wr=(hi-lo)*gweight(g,o,i0);
      double x1=gnode(g,o,i1),x2=gnode(g,o,i2),x3=gnode(g,o,i3);
      double w=wr*gweight(g,o,i1)*gweight(g,o,i2)*gweight(g,o,i3);
      V3 x,y;
      if(c==0) {
        // z on the triangle (Duffy u=x2,v=x3), t=x1; sector = vertex i and order.
        int i=sector/2;V3 vi=tv[i],vj=tv[(i+1)%3],vk=tv[(i+2)%3];
        V3 z=duffy(tv[0],tv[1],tv[2],x2,x3);
        double wz=x2*at;
        V3 p=z*(1-rho)+vi*rho,qv=z*(1-rho)+(vj*(1-x1)+vk*x1)*rho;
        if(sector%2) {x=qv;y=p;} else {x=p;y=qv;}
        w*=at*rho*(1-rho)*(1-rho)*wz;
      } else if(c==1) {
        V3 ea=tv[ti[0]],eb=tv[ti[1]],cc=tv[ti[2]],dd=sv[si[2]];
        double alpha=x1,beta=x2,tt=x3;
        int form=sector/2;
        double rr=form==0 ? rho : rho*alpha,ss=form==1 ? rho : form==0 ? rho*alpha : rho*beta;
        double sep=form==2 ? rho : rho*beta;
        if(form==2) {rr=rho*alpha;ss=rho*beta;}
        double low=tt*(1-sep),high=low+sep;
        double tx=sector%2 ? low : high,ty=sector%2 ? high : low;
        V3 ex_=ea+(eb-ea)*tx,ey=ea+(eb-ea)*ty;
        x=ex_*(1-rr)+cc*rr;y=ey*(1-ss)+dd*ss;
        w*=at*as*rho*rho*(1-sep)*(1-rr)*(1-ss);
      } else {
        V3 common=tv[ti[0]];
        V3 la=tv[ti[1]]-common,lb=tv[ti[2]]-common,ra=sv[si[1]]-common,rb=sv[si[2]]-common;
        V3 ld=la*(1-x2)+lb*x2,rd=ra*(1-x3)+rb*x3;
        double rr=sector ? rho*x1 : rho,ss=sector ? rho : rho*x1;
        x=common+ld*rr;y=common+rd*ss;
        w*=at*as*rho*rho*rho*x1;
      }
      accumulate(a,m,tv,sv,x,y,w,!(c==0 || coplanar),true);
    }
  }
  reduce(a,out+q,shared);
}

// out[m,n] += sum over faces of m inside [t0,t0+T) and faces of n.
__global__ void gather_kernel(const Pair* pairs,const int64_t* tf,const int64_t* ts,const double* tc,
    const int64_t* sf,const int64_t* ss,const double* sc,C* out0,C* out1,C* out2,int M,int N,int F,
    int t0,int T,int kind) {
  int64_t q=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(q>=int64_t(M)*N) return;
  int m=q/N,n=q%N;
  C sum[3]={C(0),C(0),C(0)};bool any=false;
  for(int a=0;a<2;++a) {
    int t=tf[2*m+a];if(t<t0 || t>=t0+T) continue;
    for(int b=0;b<2;++b) {
      int s=sf[2*n+b];if(s<0) continue;
      const Pair& p=pairs[int64_t(t-t0)*F+s];
      double w=tc[2*m+a]*sc[2*n+b];int i=ts[2*m+a],j=ss[2*n+b];
      if(kind==0) for(int k=0;k<3;++k) sum[k]+=p.v[9*k+3*i+j]*w;
      else for(int k=0;k<2;++k) sum[k]+=p.k[9*k+3*i+j]*w;
      any=true;
    }
  }
  if(!any) return;
  out0[q]+=sum[0];out1[q]+=sum[1];
  if(kind==0) out2[q]+=sum[2];
}
__global__ void scalar_kernel(const Pair* pairs,C* pc,C* pb,int F,int t0,int T) {
  int64_t q=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(q>=int64_t(T)*F) return;
  pc[int64_t(t0)*F+q]=pairs[q].p[0];pb[int64_t(t0)*F+q]=pairs[q].p[1];
}
}

// Returns {Vc,Vb,Vhat [N,N], Kb,Khat [N,Nmu], Pc,Pb [F,F]} (complex128).
namespace {
std::vector<Tensor> admittance_operators(Tensor vertices,Tensor faces,Tensor gauss,
    Tensor test_faces,Tensor test_slots,Tensor test_coefficients,
    Tensor trial_faces,Tensor trial_slots,Tensor trial_coefficients,
    double kcr,double kci,double kbr,double kbi,std::vector<int64_t> rules,int64_t tile,bool serial_far) {
  c10::cuda::CUDAGuard guard(vertices.device());auto stream=c10::cuda::getCurrentCUDAStream();
  int F=faces.size(0),N=test_faces.size(0),U=trial_faces.size(0);
  auto copt=vertices.options().dtype(at::kComplexDouble);
  Geometry g{vertices.data_ptr<double>(),faces.data_ptr<int64_t>(),F,gauss.data_ptr<double>(),int(rules[6])};
  C kc(kcr,kci),kb(kbr,kbi);
  auto taylor=at::empty({2*TAYLOR},copt);
  Media media{kc,kb,kc*kc-kb*kb,taylor.data_ptr<C>()};
  taylor_kernel<<<1,1,0,stream>>>(media,taylor.data_ptr<C>());
  Rules r{int(rules[0]),int(rules[1]),int(rules[2]),int(rules[3]),int(rules[4]),int(rules[5])};
  int64_t far_launch=rules[7],near_launch=rules[8];
  auto vc=at::zeros({N,N},copt),vb=at::zeros({N,N},copt),vh=at::zeros({N,N},copt);
  auto kbm=at::zeros({N,U},copt),kh=at::zeros({N,U},copt);
  auto pc=at::empty({F,F},copt),pb=at::empty({F,F},copt);
  for(int t0=0;t0<F;t0+=tile) {
    int T=std::min<int64_t>(tile,F-t0);
    auto buffer=at::empty({int64_t(T)*F*VALUES},copt);
    auto cls=at::empty({int64_t(T)*F},vertices.options().dtype(at::kChar));
    Pair* pairs=reinterpret_cast<Pair*>(buffer.data_ptr<C>());
    int64_t P=int64_t(T)*F;
    // Bounded launches (far_launch pairs, near_launch near/touching blocks): every pair is
    // still owned by one warp or one block, so the outputs do not depend on the split.
    // Long single launches tripped context-switch timeouts on time-sliced shared devices.
    for(int64_t q0=0;q0<P;q0+=far_launch) {
      int64_t count=std::min<int64_t>(far_launch,P-q0);
      if(serial_far)
        far_serial_kernel<<<(count+127)/128,128,0,stream>>>(g,media,r,t0,q0,P,pairs,cls.data_ptr<int8_t>());
      else
        far_kernel<<<(count+FAR_WARPS-1)/FAR_WARPS,32*FAR_WARPS,0,stream>>>(g,media,r,t0,q0,P,pairs,
            cls.data_ptr<int8_t>());
    }
    auto list=cls.ne(4).nonzero().flatten().contiguous();
    int64_t L=list.numel();
    for(int64_t b0=0;b0<L;b0+=near_launch)
      near_kernel<<<std::min<int64_t>(near_launch,L-b0),128,128*sizeof(double),stream>>>(
          g,media,r,list.data_ptr<int64_t>()+b0,t0,pairs);
    gather_kernel<<<(int64_t(N)*N+127)/128,128,0,stream>>>(pairs,test_faces.data_ptr<int64_t>(),
        test_slots.data_ptr<int64_t>(),test_coefficients.data_ptr<double>(),test_faces.data_ptr<int64_t>(),
        test_slots.data_ptr<int64_t>(),test_coefficients.data_ptr<double>(),vc.data_ptr<C>(),vb.data_ptr<C>(),
        vh.data_ptr<C>(),N,N,F,t0,T,0);
    gather_kernel<<<(int64_t(N)*U+127)/128,128,0,stream>>>(pairs,test_faces.data_ptr<int64_t>(),
        test_slots.data_ptr<int64_t>(),test_coefficients.data_ptr<double>(),trial_faces.data_ptr<int64_t>(),
        trial_slots.data_ptr<int64_t>(),trial_coefficients.data_ptr<double>(),kbm.data_ptr<C>(),kh.data_ptr<C>(),
        nullptr,N,U,F,t0,T,1);
    scalar_kernel<<<(P+127)/128,128,0,stream>>>(pairs,pc.data_ptr<C>(),pb.data_ptr<C>(),F,t0,T);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {vc,vb,vh,kbm,kh,pc,pb};
}
}

std::vector<Tensor> layered_admittance_operators(Tensor vertices,Tensor faces,Tensor gauss,
    Tensor test_faces,Tensor test_slots,Tensor test_coefficients,
    Tensor trial_faces,Tensor trial_slots,Tensor trial_coefficients,
    double kcr,double kci,double kbr,double kbi,std::vector<int64_t> rules,int64_t tile) {
  return admittance_operators(vertices,faces,gauss,test_faces,test_slots,test_coefficients,trial_faces,trial_slots,
      trial_coefficients,kcr,kci,kbr,kbi,rules,tile,false);
}
// Test oracle only (tests/support/layered_admittance_oracle.py): the same operators with far_serial_kernel.
std::vector<Tensor> layered_admittance_operators_serial_far(Tensor vertices,Tensor faces,Tensor gauss,
    Tensor test_faces,Tensor test_slots,Tensor test_coefficients,
    Tensor trial_faces,Tensor trial_slots,Tensor trial_coefficients,
    double kcr,double kci,double kbr,double kbi,std::vector<int64_t> rules,int64_t tile) {
  return admittance_operators(vertices,faces,gauss,test_faces,test_slots,test_coefficients,trial_faces,trial_slots,
      trial_coefficients,kcr,kci,kbr,kbi,rules,tile,true);
}
