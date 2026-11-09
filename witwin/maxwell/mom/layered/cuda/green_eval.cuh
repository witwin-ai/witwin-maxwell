#pragma once
#include "common.cuh"

namespace layered {
struct GreenTableView {
  const double *rho, *distances, *plane_z, *frequency_hz, *sigma;
  const C *values, *slopes, *images, *epsilon;
  const int64_t *image_offsets;
  int frequencies, pairs, nodes, image_count, metal_planes, exact_mode;
  double ground,rho_max;
};

__device__ inline C exact_value(C,double,double,double,double,int,int,double);

__device__ inline C hermite(double x,double a,double b,C ya,C yb,C da,C db) {
  double h=b-a,t=(x-a)/h,t2=t*t,t3=t2*t;
  return (2*t3-3*t2+1)*ya+(t3-2*t2+t)*h*da+
         (-2*t3+3*t2)*yb+(t3-t2)*h*db;
}

// Invert the linear/logarithmic coarse grid, then select its interleaved
// midpoint. Two bounded corrections use stored nodes, including exact-node
// and nextafter queries; interpolation arithmetic is unchanged.
__device__ inline int radial_interval(const double* grid,int nodes,double rho,
    double near,double step) {
  int coarse=(nodes-3)/2,split=coarse/4;
  int lo;
  if(rho<near) lo=int(rho*(2*split/near));
  else {
    int cell=split+int(log(rho/near)/step);
    cell=max(split,min(coarse,cell));
    lo=2*cell;
    if(rho>=grid[lo+1]) ++lo;
  }
  lo=max(0,min(nodes-2,lo));
  if(lo>0 && rho<grid[lo]) --lo;
  if(lo>0 && rho<grid[lo]) --lo;
  if(lo<nodes-2 && rho>=grid[lo+1]) ++lo;
  if(lo<nodes-2 && rho>=grid[lo+1]) ++lo;
  return lo;
}
__device__ inline int radial_interval(const double* grid,int nodes,double rho) {
  int coarse=(nodes-3)/2,split=coarse/4;
  double near=grid[2*split];
  double step=rho<near?0:log(grid[2*split+2]/near);
  return radial_interval(grid,nodes,rho,near,step);
}

// Cancellation-free (exp(i*k*R)-1)/(4*pi*R), including the coincident limit.
__device__ inline C wave_remainder(C k,double distance) {
  C z=C(0,1)*k*distance;
  if(abs(z)<1e-4) {
    C term(1),sum(1);
    for(int j=1;j<=8;++j) {term*=z/double(j+1);sum+=term;}
    return C(0,1)*k*sum/(4*pi);
  }
  return (ex(z)-C(1))/(4*pi*distance);
}

// Assembly always integrates static images separately, also in exact modes.
// The image arrays already carry the direct and signed ground-image weights.
__device__ inline C green_remainder(const GreenTableView& t,int f,int pair,int component,double rho) {
  if(t.exact_mode) {
    int i=0,j=pair;
    while(j>=t.metal_planes-i) {j-=t.metal_planes-i;++i;}
    j+=i;
    if(t.exact_mode>1 && (t.plane_z[i]==t.ground || t.plane_z[j]==t.ground)) return C(0);
    double w=2*pi*t.frequency_hz[f];
    C epsilon=eps(t.epsilon,t.sigma,1,w),k=root((w/c0)*(w/c0)*epsilon);
    C value=wave_remainder(k,hypot(rho,t.plane_z[i]-t.plane_z[j]));
    if(t.exact_mode>1) value-=wave_remainder(k,hypot(rho,t.plane_z[i]+t.plane_z[j]-2*t.ground));
    return component ? value/epsilon : value;
  }
  int lo=radial_interval(t.rho,t.nodes,rho),hi=lo+1;
  int64_t base=((int64_t(f)*t.pairs+pair)*2+component)*t.nodes;
  return hermite(rho,t.rho[lo],t.rho[hi],t.values[base+lo],t.values[base+hi],
                 t.slopes[base+lo],t.slopes[base+hi]);
}

// R2 callers supply an admitted pair and rho in [0,rho_max]. Table storage is
// [frequency,pair,component,rho]. Self singularities stay in the analytic images.
__device__ inline C green_value(const GreenTableView& t,int f,int pair,int component,double rho) {
  if(t.exact_mode) {
    int i=0,j=pair;
    while(j>=t.metal_planes-i) {j-=t.metal_planes-i;++i;}
    j+=i;
    double w=2*pi*t.frequency_hz[f];
    return exact_value(eps(t.epsilon,t.sigma,1,w),w,rho,t.plane_z[i],t.plane_z[j],
                       component,t.exact_mode,t.ground);
  }
  int lo=radial_interval(t.rho,t.nodes,rho),hi=lo+1;
  int64_t base=((int64_t(f)*t.pairs+pair)*2+component)*t.nodes;
  C v=hermite(rho,t.rho[lo],t.rho[hi],t.values[base+lo],t.values[base+hi],
              t.slopes[base+lo],t.slopes[base+hi]);
  C analytic(0),correction(0);
  for(int i=t.image_offsets[pair];i<t.image_offsets[pair+1];++i) {
    C a=t.images[(f*2+component)*t.image_count+i];
    if(a!=C(0)) {
      C term=a/(4*pi*hypot(rho,t.distances[i]))-correction,next=analytic+term;
      correction=(next-analytic)-term;analytic=next;
    }
  }
  return v+analytic;
}

__device__ inline C exact_value(C epsilon,double w,double rho,double zs,double zo,
                                int component,int mode,double ground) {
  if(mode>1 && (zs==ground || zo==ground)) return C(0);
  C k=root((w/c0)*(w/c0)*epsilon);
  double distance=hypot(rho,zo-zs);
  C v=ex(C(0,1)*k*distance)/(4*pi*distance);
  if(mode>1) {
    distance=hypot(rho,zs+zo-2*ground);
    v-=ex(C(0,1)*k*distance)/(4*pi*distance);
  }
  return component ? v/epsilon : v;
}
}


namespace layered {
// Vertical rows are [frequency, reciprocal z-node pair, (zz,zrho,rhoz,phi), rho].
// Segment boundaries are one-sided: an interior interface belongs to its upper
// segment. Analytic images retain their actual affine z,z' distances.
struct VerticalGreenTableView {
  const double *rho,*z,*segments,*geometry,*frequencies,*sigma,*edges,*boundaries;
  const C *values,*slopes,*images,*epsilon;
  const C *horizontal_values,*horizontal_slopes,*horizontal_images;
  const int64_t *direct,*offsets;
  int F,N,Z,S,K,I,L;
  double rho_max;
};
__device__ inline int vertical_segment(const VerticalGreenTableView& t,double z) {
  int selected=-1;
  for(int i=0;i<t.S;++i) if(z>=t.segments[2*i] && z<=t.segments[2*i+1]) selected=i;
  return selected;
}
__device__ inline double vertical_basis(const VerticalGreenTableView& t,int segment,int j,double z) {
  double value=1;
  int offset=segment*t.K;
  for(int k=0;k<t.K;++k) if(k!=j)
    value*=(z-t.z[offset+k])/(t.z[offset+j]-t.z[offset+k]);
  return value;
}
__device__ inline C vertical_green_images(const VerticalGreenTableView&,int,int,double,double,double,int,int);
__device__ inline C horizontal_vertical_images(const VerticalGreenTableView&,int,double,double,double);
struct BoundaryBlend {double source,observer,zs,zo;};
__device__ inline double endpoint_blend(const VerticalGreenTableView& t,int segment,double z,double ground) {
  double result=1;
  for(int j=0;j<t.K;++j) result*=(z-t.z[segment*t.K+j])/(ground-t.z[segment*t.K+j]);
  return result;
}
__device__ inline BoundaryBlend boundary_blend(const VerticalGreenTableView& t,int c,
    double zs,double zo,int a,int b) {
  BoundaryBlend out{0,0,0,0};
  for(int side=0;side<2;++side) {
    double g=t.boundaries[side];if(!isfinite(g)) continue;
    if((c==1||c==3||c==4)&&g>=t.segments[2*a]&&g<=t.segments[2*a+1])
      {out.source=endpoint_blend(t,a,zs,g);out.zs=g;}
    if((c==2||c==3||c==4)&&g>=t.segments[2*b]&&g<=t.segments[2*b+1])
      {out.observer=endpoint_blend(t,b,zo,g);out.zo=g;}
  }
  return out;
}
__device__ inline double vertical_weight(const VerticalGreenTableView& t,int c,
    double zs,double zo,int a,int b,int j,int k) {
  auto blend=boundary_blend(t,c,zs,zo,a,b);
  double ls=vertical_basis(t,a,j,zs),lo=vertical_basis(t,b,k,zo);
  if(blend.source!=0) ls-=blend.source*vertical_basis(t,a,j,blend.zs);
  if(blend.observer!=0) lo-=blend.observer*vertical_basis(t,b,k,blend.zo);
  return ls*lo;
}
__device__ inline C boundary_image(const VerticalGreenTableView& t,int f,int c,
    double rho,double zs,double zo,int a,int b) {
  return c==4 ? horizontal_vertical_images(t,f,rho,zs,zo):
      vertical_green_images(t,f,c,rho,zs,zo,a,b);
}
__device__ inline C boundary_correction(const VerticalGreenTableView& t,int f,int c,
    double rho,double zs,double zo,int a,int b) {
  auto blend=boundary_blend(t,c,zs,zo,a,b);
  C value(0);
  // Exact endpoint constraints apply to the TOTAL kernel. Cropped static
  // ray lists need not vanish separately; subtract their endpoint values.
  if(blend.source!=0) value-=blend.source*boundary_image(t,f,c,rho,blend.zs,zo,a,b);
  if(blend.observer!=0) value-=blend.observer*boundary_image(t,f,c,rho,zs,blend.zo,a,b);
  if(blend.source!=0&&blend.observer!=0)
    value+=blend.source*blend.observer*boundary_image(t,f,c,rho,blend.zs,blend.zo,a,b);
  return value;
}
__device__ inline C vertical_green_remainder(const VerticalGreenTableView& t,int f,int c,
    double rho,double zs,double zo,int a,int b) {
  int lo=0,hi=t.N-1;
  while(hi-lo>1) {int m=(lo+hi)/2;if(t.rho[m]>rho) hi=m;else lo=m;}
  C value(0),correction(0);
  for(int j=0;j<t.K;++j) for(int k=0;k<t.K;++k) {
    int source=a*t.K+j,observer=b*t.K+k,component=c;
    double sign=1;
    if(source>observer) {
      int swap=source;source=observer;observer=swap;
      if(c==1||c==2) {component=3-c;sign=-1;}
    }
    int pair=source*t.Z-source*(source-1)/2+observer-source;
    int64_t base=((int64_t(f)*t.Z*(t.Z+1)/2+pair)*4+component)*t.N;
    C sample=hermite(rho,t.rho[lo],t.rho[hi],t.values[base+lo],t.values[base+hi],
                     t.slopes[base+lo],t.slopes[base+hi]);
    C term=sample*(sign*vertical_weight(t,c,zs,zo,a,b,j,k))-correction;
    C next=value+term;correction=(next-value)-term;value=next;
  }
  return value+boundary_correction(t,f,c,rho,zs,zo,a,b);
}
__device__ inline C vertical_green_images(const VerticalGreenTableView& t,int f,int c,
    double rho,double zs,double zo,int a,int b) {
  int pair=a*t.S-a*(a-1)/2+b-a;
  C value(0),correction(0);
  for(int i=t.offsets[pair];i<t.offsets[pair+1];++i) {
    C term;
    if(t.direct[i]) {
      Stack stack{t.epsilon,t.sigma,t.edges,t.L,0,0};
      C epsilon=eps(t.epsilon,t.sigma,region(stack,(t.segments[2*a]+t.segments[2*a+1])*.5),2*pi*t.frequencies[f]);
      C k=root(pow(2*pi*t.frequencies[f]/c0,2)*epsilon);
      double distance=t.direct[i]==1 ? fabs(zo-zs):
          fmax(0.0,t.geometry[3*i]*zs+t.geometry[3*i+1]*zo+t.geometry[3*i+2]);
      double R=hypot(rho,distance);
      if(R==0 && c==3 && (zs==t.boundaries[0]||zo==t.boundaries[0]||
                         zs==t.boundaries[1]||zo==t.boundaries[1])) continue;
      term=(c==0||c==3) ? ex(C(0,1)*k*R)/(4*pi*R):C(0);
      if(t.direct[i]==1) {if(c==3) term/=epsilon;}
      else term*=t.images[(f*4+c)*t.I+i];
    } else {
      double distance=fmax(0.0,t.geometry[3*i]*zs+t.geometry[3*i+1]*zo+t.geometry[3*i+2]);
      double R=hypot(rho,distance);
      C coefficient=t.images[(f*4+c)*t.I+i];
      term=coefficient==C(0) ? C(0):coefficient*((c==1||c==2) ?
          (rho==0 ? 0:rho/(4*pi*R*(R+distance))):1/(4*pi*R));
    }
    term-=correction;C next=value+term;correction=(next-value)-term;value=next;
  }
  return value;
}

__device__ inline C horizontal_vertical_remainder(const VerticalGreenTableView& v,int f,
    double rho,double zs,double zo) {
  if(zs>zo) {double swap=zs;zs=zo;zo=swap;}
  int a=vertical_segment(v,zs),b=vertical_segment(v,zo);
  int lo=0,hi=v.N-1;
  while(hi-lo>1) {int m=(lo+hi)/2;if(v.rho[m]>rho) hi=m;else lo=m;}
  C value(0);
  for(int j=0;j<v.K;++j) for(int k=0;k<v.K;++k) {
    int source=a*v.K+j,observer=b*v.K+k;
    if(source>observer) {int swap=source;source=observer;observer=swap;}
    int pair=source*v.Z-source*(source-1)/2+observer-source;
    int64_t base=(int64_t(f)*v.Z*(v.Z+1)/2+pair)*v.N;
    double weight=vertical_weight(v,4,zs,zo,a,b,j,k);
    value+=weight*hermite(rho,v.rho[lo],v.rho[hi],v.horizontal_values[base+lo],
        v.horizontal_values[base+hi],v.horizontal_slopes[base+lo],v.horizontal_slopes[base+hi]);
  }
  return value+boundary_correction(v,f,4,rho,zs,zo,a,b);
}
__device__ inline C horizontal_vertical_images(const VerticalGreenTableView& v,int f,
    double rho,double zs,double zo) {
  if(zs>zo) {double swap=zs;zs=zo;zo=swap;}
  int a=vertical_segment(v,zs),b=vertical_segment(v,zo);
  int pair=a*v.S-a*(a-1)/2+b-a;
  C value(0);
  Stack stack{v.epsilon,v.sigma,v.edges,v.L,0,0};
  for(int i=v.offsets[pair];i<v.offsets[pair+1];++i) {
    C coefficient=v.horizontal_images[f*v.I+i];
    if(coefficient==C(0)) continue;
    double d=v.direct[i]==1 ? fabs(zo-zs):
        fmax(0.0,v.geometry[3*i]*zs+v.geometry[3*i+1]*zo+v.geometry[3*i+2]);
    double R=hypot(rho,d);
    if(R==0 && (zs==v.boundaries[0]||zo==v.boundaries[0]||
                zs==v.boundaries[1]||zo==v.boundaries[1])) continue;
    C term=coefficient/(4*pi*R);
    if(v.direct[i]) {
      C epsilon=eps(v.epsilon,v.sigma,region(stack,(v.segments[2*a]+v.segments[2*a+1])*.5),2*pi*v.frequencies[f]);
      term*=ex(C(0,1)*root(pow(2*pi*v.frequencies[f]/c0,2)*epsilon)*R);
    }
    value+=term;
  }
  return value;
}
// Assembly supplies admitted coordinates. Swapping the two heights reverses the
// radial displacement, hence both off-diagonal coefficients change sign.
__device__ inline C vertical_green_value(const VerticalGreenTableView& t,int f,int c,
    double rho,double zs,double zo) {
  double sign=1;
  if(zs>zo) {double swap=zs;zs=zo;zo=swap;if(c==1||c==2) {c=3-c;sign=-1;}}
  int a=vertical_segment(t,zs),b=vertical_segment(t,zo);
  return sign*(vertical_green_remainder(t,f,c,rho,zs,zo,a,b)+
               vertical_green_images(t,f,c,rho,zs,zo,a,b));
}
__device__ inline C vertical_green_projection(const VerticalGreenTableView& t,int f,int c,
    double dx,double dy,double zs,double zo,bool y_component) {
  double rho=hypot(dx,dy);
  C value=vertical_green_value(t,f,c,rho,zs,zo);
  return (c==1||c==2) ? (rho==0 ? C(0):value*((y_component?dy:dx)/rho)):value;
}
}
