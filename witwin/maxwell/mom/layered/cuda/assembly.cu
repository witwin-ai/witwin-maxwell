#include "green_eval.cuh"
#include "triangle_static.cuh"
#include <vector>

using namespace layered;
using at::Tensor;
VerticalGreenTableView vertical_view(const std::vector<Tensor>&,double);

struct GeometryView {
  const double *triangles,*areas2,*centres,*edge2,*slopes;
  const int64_t *planes,*supports,*slots,*face_edges;
  int faces,edges;
  const double* z_weights;int rule_points,z_weight_stride;
  const double* scalar_points;
  const double* radial;
};

// Two grid constants are invariant across all pairs and frequencies.
__global__ void radial_metadata_kernel(const double* grid,int nodes,double* out) {
  int split=((nodes-3)/2)/4;
  double near=grid[2*split];
  out[0]=near;out[1]=log(grid[2*split+2]/near);
}

struct FacePair {
  const double *test,*source;
  int t,s,pair;
  bool transpose;
  double lateral2,threshold2,size2;
};

__device__ inline FacePair face_pair(GeometryView g,GreenTableView table,int row,int col,double ratio) {
  // Reciprocity is evaluated once in a canonical face order, including the
  // outer-rule choice for analytic inner integration. No matrix symmetrization.
  FacePair p{};p.transpose=row>col;p.t=min(row,col);p.s=max(row,col);
  p.test=g.triangles+9*p.t;p.source=g.triangles+9*p.s;
  int i=g.planes[p.t],j=g.planes[p.s];if(j<i) {int swap=i;i=j;j=swap;}
  p.pair=(i<0||j<0) ? -1:i*table.metal_planes-i*(i-1)/2+j-i;
  double dx=g.centres[3*p.t]-g.centres[3*p.s],dy=g.centres[3*p.t+1]-g.centres[3*p.s+1];
  p.lateral2=dx*dx+dy*dy;
  p.size2=fmax(g.edge2[p.t],g.edge2[p.s]);
  p.threshold2=ratio*ratio*p.size2;
  return p;
}

// Every branch predicate is fp64, even in the float regular-arithmetic variant.
__device__ inline bool image_near(FacePair p,double distance) {
  return p.lateral2+distance*distance<p.threshold2;
}
__device__ inline bool pair_near(FacePair p,GreenTableView table) {
  if(image_near(p,p.test[2]-p.source[2])) return true;
  for(int i=table.image_offsets[p.pair];i<table.image_offsets[p.pair+1];++i)
    if(image_near(p,table.distances[i])) return true;
  return false;
}
__device__ inline V3 quadrature_point(const double* t,const double* rule) {
  // Affine form retains small cells far from the coordinate origin.
  return vertex(t,0)+(vertex(t,1)-vertex(t,0))*rule[1]+(vertex(t,2)-vertex(t,0))*rule[2];
}
template<typename T> __device__ inline c10::complex<T> convert(C z) {return {T(z.real()),T(z.imag())};}
template<typename T> __device__ inline C warp_sum(c10::complex<T> value) {
  T real=value.real(),imag=value.imag();
  for(int delta=16;delta>0;delta/=2) {
    real+=__shfl_down_sync(0xffffffff,real,delta);
    imag+=__shfl_down_sync(0xffffffff,imag,delta);
  }
  return C(double(real),double(imag));
}


struct PairRule {const double* data;int count;};
__device__ inline PairRule pair_rule(const double* rules,int stride,bool adaptive,
    FacePair p,double distance) {
  if(!adaptive) return {rules,stride};
  double r2=(p.lateral2+distance*distance)/p.size2;
  int level=0;
  // Rule thresholds/counts are geometry metadata, selected in fp64 even
  // when the admitted far arithmetic uses fp32.
  for(int candidate=1;candidate<8;++candidate)
    if(r2>=rules[4*(stride*candidate+stride-1)]) level=candidate;
  return {rules+4*stride*level,int(rules[4*(stride*level+stride-1)+1])};
}

template<typename T> struct RelativePoint {T x,y;};
template<typename T> __device__ inline RelativePoint<T> relative_point(
    const double* triangle,const double* rule) {
  return {T(triangle[3]-triangle[0])*T(rule[1])+T(triangle[6]-triangle[0])*T(rule[2]),
          T(triangle[4]-triangle[1])*T(rule[1])+T(triangle[7]-triangle[1])*T(rule[2])};
}
// Cache fp64 relative quadrature coordinates once for all source tiles.
// Scalar image cancellation requires fp64 even with fp32 vector moments.
// Storage is 16*faces*rule_points bytes, independent of frequency count.
__global__ void scalar_points_kernel(GeometryView g,const double* rules,double* points) {
  int64_t q=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(q>=int64_t(g.faces)*g.rule_points) return;
  auto p=relative_point<double>(g.triangles+9*(q/g.rule_points),rules+4*(q%g.rule_points));
  points[2*q]=p.x;points[2*q+1]=p.y;
}
template<typename T> __device__ inline void add_vector_moments(c10::complex<T>* sums,
    c10::complex<T> ga,RelativePoint<T> obs,RelativePoint<T> src,FacePair p,T weight) {
  ga*=weight;
  for(int a=0;a<3;++a) for(int b=0;b<3;++b) {
    T ax=obs.x-T(p.test[3*a]-p.test[0]),ay=obs.y-T(p.test[3*a+1]-p.test[1]);
    T bx=src.x-T(p.source[3*b]-p.source[0]),by=src.y-T(p.source[3*b+1]-p.source[1]);
    if constexpr(sizeof(T)==4) {
      T moment=fmaf(ax,bx,ay*by);
      sums[3*a+b]=c10::complex<T>(fmaf(ga.real(),moment,sums[3*a+b].real()),
                                 fmaf(ga.imag(),moment,sums[3*a+b].imag()));
    } else sums[3*a+b]+=ga*(ax*bx+ay*by);
  }
}
// Horizontal remainder batches up to four frequencies per warp. Vector
// moments use T; scalar interpolation and sums remain fp64 for cancellation.
// Geometry, classification, rho, interval and Hermite weights are reused.
// Logical/sample/f: 128 B Hermite data + (64 B rules + 32 B grid +
// 32 B cached coordinates)/B, approximately 300+40/B real FLOP;
// cached static moments add 320 B per face pair. FP64 near pairs use B<=2.
template<typename T,int B> __device__ void regular_pair(FacePair p,
    GreenTableView table,GeometryView g,const double* rules,int stride,bool adaptive,
    int first,int stop,int requested,int lane,int64_t pair,int64_t pairs,C* tile) {
  using CT=c10::complex<T>;
  CT sums[B][9]{};C scalar[B]{};
  PairRule rule=pair_rule(rules,stride,adaptive,p,p.test[2]-p.source[2]);
  for(int q=lane;q<rule.count*rule.count;q+=32) {
    const double *ro=rule.data+4*(q/rule.count),*rs=rule.data+4*(q%rule.count);
    auto obs=relative_point<T>(p.test,ro),src=relative_point<T>(p.source,rs);
    const double* po=g.scalar_points+2*(int64_t(p.t)*g.rule_points+(ro-rules)/4);
    const double* ps=g.scalar_points+2*(int64_t(p.s)*g.rule_points+(rs-rules)/4);
    RelativePoint<double> precise_obs{po[0],po[1]},precise_src{ps[0],ps[1]};
    double dx=p.test[0]-p.source[0]+precise_obs.x-precise_src.x;
    double dy=p.test[1]-p.source[1]+precise_obs.y-precise_src.y;
    double rho=sqrt(dx*dx+dy*dy);
    T weight=T(ro[3])*T(rs[3])*T(g.areas2[p.t])*T(g.areas2[p.s]);
    double scalar_weight=ro[3]*rs[3]*g.areas2[p.t]*g.areas2[p.s];
    int lo=0;
    T w0=0,w1=0,w2=0,w3=0;
    double p0=0,p1=0,p2=0,p3=0;
    if(!table.exact_mode) {
      lo=radial_interval(table.rho,table.nodes,rho,g.radial[0],g.radial[1]);
      double hd=table.rho[lo+1]-table.rho[lo],td=(rho-table.rho[lo])/hd;
      double td2=td*td,td3=td2*td;
      p0=2*td3-3*td2+1;p1=(td3-2*td2+td)*hd;
      p2=-2*td3+3*td2;p3=(td3-td2)*hd;
      T h=T(hd),t=T(td),t2=t*t,t3=t2*t;
      w0=T(2)*t3-T(3)*t2+T(1);w1=(t3-T(2)*t2+t)*h;
      w2=-T(2)*t3+T(3)*t2;w3=(t3-t2)*h;
    }
    #pragma unroll
    for(int b=0;b<B;++b) if(first+b<stop) {
      int f=requested<0?first+b:requested;
      CT ga;C kp;
      if(table.exact_mode) {
        ga=convert<T>(green_remainder(table,f,p.pair,0,rho));
        kp=green_remainder(table,f,p.pair,1,rho);
      } else {
        int64_t base=((int64_t(f)*table.pairs+p.pair)*2)*table.nodes+lo;
        ga=w0*convert<T>(table.values[base])+w1*convert<T>(table.slopes[base])+
           w2*convert<T>(table.values[base+1])+w3*convert<T>(table.slopes[base+1]);
        base+=table.nodes;
        kp=p0*table.values[base]+p1*table.slopes[base]+
           p2*table.values[base+1]+p3*table.slopes[base+1];
      }
      add_vector_moments(sums[b],ga,obs,src,p,weight);
      scalar[b]+=kp*scalar_weight;
    }
  }
  #pragma unroll
  for(int b=0;b<B;++b) if(first+b<stop) for(int k=0;k<10;++k) {
    C value=k==9?warp_sum(scalar[b]):warp_sum(sums[b][k]);
    int target=k==9||!p.transpose?k:(k%3)*3+k/3;
    if(lane==0) tile[(int64_t(first+b)*pairs+pair)*10+target]+=value;
  }
}

// Wall integration uses the same P1 distance/rule classes and tile layout.
// Each lane owns ten moments; warp reductions and edge gathers are fixed order.
// Static diagonal rays use the orientation-independent triangle primitive.
// Cross rays have the radial J1 inverse, with its on-axis limit taken exactly.
struct WallRay {
  int image;
  double source_scale,observer_scale,offset,distance,source_height_scale,source_height_offset;
  bool reverse;
};
__device__ inline WallRay wall_ray(VerticalGreenTableView v,FacePair p,GeometryView g,int image) {
  double zs=g.centres[3*p.s+2],zo=g.centres[3*p.t+2];
  bool reverse=zs>zo;
  double alpha=v.geometry[3*image+(reverse?1:0)];
  double beta=v.geometry[3*image+(reverse?0:1)];
  double offset=v.geometry[3*image+2];
  if(v.direct[image]==1) return {image,-1,1,0,fabs(zo-zs),1,0,reverse};
  return {image,alpha,beta,offset,fmax(0.0,alpha*zs+beta*zo+offset),
          -alpha/beta,-offset/beta,reverse};
}
__device__ inline int wall_segment_pair(VerticalGreenTableView v,double zs,double zo) {
  int a=vertical_segment(v,fmin(zs,zo)),b=vertical_segment(v,fmax(zs,zo));
  return a*v.S-a*(a-1)/2+b-a;
}
__device__ inline C wall_coefficient(VerticalGreenTableView v,int f,WallRay ray,int component) {
  if(component==4) return v.horizontal_images[f*v.I+ray.image];
  double sign=1;
  if(ray.reverse && (component==1||component==2)) {component=3-component;sign=-1;}
  return sign*v.images[(f*4+component)*v.I+ray.image];
}
__device__ inline bool wall_near(FacePair p,VerticalGreenTableView v,GeometryView g) {
  double zs=g.centres[3*p.s+2],zo=g.centres[3*p.t+2];
  int pair=wall_segment_pair(v,zs,zo);
  for(int i=v.offsets[pair];i<v.offsets[pair+1];++i)
    if(image_near(p,wall_ray(v,p,g,i).distance)) return true;
  return false;
}
// Geometry-only interpolation weights are cached once for the complete P1
// rule pack. This removes O(K) divisions from every frequency/pair/sample.
// Traffic: 8*(2K+2) bytes per face/rule point, O(F*Q*K) setup.
__global__ void wall_weights_kernel(GeometryView g,VerticalGreenTableView v,const double* rules,
    double* weights,int points,int stride) {
  int64_t t=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(t>=int64_t(g.faces)*points) return;
  int face=t/points,q=t%points;
  V3 point=quadrature_point(g.triangles+9*face,rules+4*q);
  int segment=vertical_segment(v,point.z);if(segment<0) return;
  double blend=0,ground=0;
  for(int side=0;side<2;++side) {
    double z=v.boundaries[side];
    if(isfinite(z)&&z>=v.segments[2*segment]&&z<=v.segments[2*segment+1])
      {blend=endpoint_blend(v,segment,point.z,z);ground=z;}
  }
  for(int j=0;j<v.K;++j) {
    double w=vertical_basis(v,segment,j,point.z);
    weights[t*stride+2*j]=w;
    weights[t*stride+2*j+1]=w-(blend==0?0:blend*vertical_basis(v,segment,j,ground));
  }
  weights[t*stride+2*v.K]=blend;weights[t*stride+2*v.K+1]=ground;
}
__device__ inline C wall_boundary_correction(VerticalGreenTableView v,int f,int c,double rho,
    double zs,double zo,int a,int b,const double* sw,const double* ow) {
  double bs=(c==1||c==3||c==4)?sw[2*v.K]:0;
  double bo=(c==2||c==3||c==4)?ow[2*v.K]:0;
  double gs=sw[2*v.K+1],go=ow[2*v.K+1];
  C value(0);
  if(bs!=0) value-=bs*boundary_image(v,f,c,rho,gs,zo,a,b);
  if(bo!=0) value-=bo*boundary_image(v,f,c,rho,zs,go,a,b);
  if(bs!=0&&bo!=0) value+=bs*bo*boundary_image(v,f,c,rho,gs,go,a,b);
  return value;
}
__device__ inline void wall_remainders(VerticalGreenTableView v,int f,double rho,
    double zs,double zo,const double* sw,const double* ow,C* values,int components,const double* radial) {
  bool reverse=zs>zo;
  if(reverse) {double swap=zs;zs=zo;zo=swap;const double* ptr=sw;sw=ow;ow=ptr;}
  int a=vertical_segment(v,zs),b=vertical_segment(v,zo);
  int lo=radial_interval(v.rho,v.N,rho,radial[0],radial[1]),hi=lo+1;
  double h=v.rho[hi]-v.rho[lo],t=(rho-v.rho[lo])/h,t2=t*t,t3=t2*t;
  double w0=2*t3-3*t2+1,w1=(t3-2*t2+t)*h,w2=-2*t3+3*t2,w3=(t3-t2)*h;
  for(int j=0;j<v.K;++j) for(int k=0;k<v.K;++k) {
    int source=a*v.K+j,observer=b*v.K+k;
    bool flip=source>observer;
    if(flip) {int swap=source;source=observer;observer=swap;}
    int pair=source*v.Z-source*(source-1)/2+observer-source;
    for(int c=0;c<5;++c) if(components&(1<<c)) {
      int component=c,physical=(reverse&&(c==1||c==2)) ? 3-c:c;
      double sign=1;
      if((reverse^flip)&&(c==1||c==2)) {component=3-c;sign=-1;}
      int64_t base=(int64_t(f)*v.Z*(v.Z+1)/2+pair);
      const C *data,*slopes;
      if(c==4) {base*=v.N;data=v.horizontal_values;slopes=v.horizontal_slopes;}
      else {base=(base*4+component)*v.N;data=v.values;slopes=v.slopes;}
      bool sv=physical==1||physical==3||physical==4;
      bool ov=physical==2||physical==3||physical==4;
      double weight=sign*sw[2*j+sv]*ow[2*k+ov];
      values[c]+=weight*(w0*data[base+lo]+w1*slopes[base+lo]+
                         w2*data[base+hi]+w3*slopes[base+hi]);
    }
  }
  for(int c=0;c<5;++c) if(components&(1<<c)) {
    int physical=(reverse&&(c==1||c==2)) ? 3-c:c;
    values[c]+=(reverse&&(c==1||c==2) ? -1:1)*
        wall_boundary_correction(v,f,physical,rho,zs,zo,a,b,sw,ow);
  }
}
template<typename T> __device__ void wall_moments(c10::complex<T>* sums,C* scalar,const C* values,
    V3 obs,V3 src,FacePair p,T weight,double scalar_weight) {
  using CT=c10::complex<T>;
  T dx=T(obs.x-src.x),dy=T(obs.y-src.y),rho=sqrt(dx*dx+dy*dy);
  T ux=rho==0 ? 0:dx/rho,uy=rho==0 ? 0:dy/rho;
  *scalar+=values[3]*scalar_weight;
  for(int a=0;a<3;++a) for(int b=0;b<3;++b) {
    T ax=T(obs.x-p.test[3*a]),ay=T(obs.y-p.test[3*a+1]),az=T(obs.z-p.test[3*a+2]);
    T bx=T(src.x-p.source[3*b]),by=T(src.y-p.source[3*b+1]),bz=T(src.z-p.source[3*b+2]);
    CT value=convert<T>(values[4])*(ax*bx+ay*by)+convert<T>(values[0])*(az*bz);
    value+=convert<T>(values[1])*(az*(ux*bx+uy*by));
    value+=convert<T>(values[2])*((ax*ux+ay*uy)*bz);
    sums[3*a+b]+=value*weight;
  }
}
template<typename T> __device__ void wall_regular(FacePair p,GreenTableView table,
    VerticalGreenTableView v,GeometryView g,const double* rules,int stride,bool adaptive,
    int f,int lane,C* tile) {
  using CT=c10::complex<T>;
  CT sums[9]{};C scalar(0);
  double zs=g.centres[3*p.s+2],zo=g.centres[3*p.t+2];
  PairRule rule=pair_rule(rules,stride,adaptive,p,zo-zs);
  if(!table.exact_mode) {
    for(int q=lane;q<rule.count*rule.count;q+=32) {
      const double *ro=rule.data+4*(q/rule.count),*rs=rule.data+4*(q%rule.count);
      V3 obs=quadrature_point(p.test,ro),src=quadrature_point(p.source,rs);
      C values[5]{};
      int source_q=(rs-rules)/4,observer_q=(ro-rules)/4;
      const double* sw=g.z_weights+(int64_t(p.s)*g.rule_points+source_q)*g.z_weight_stride;
      const double* ow=g.z_weights+(int64_t(p.t)*g.rule_points+observer_q)*g.z_weight_stride;
      // A horizontal RWG numerator has exactly zero z component. Omit the
      // two dyadic components it cannot contract, retaining phi and Ahh.
      int components=31;
      if(g.planes[p.t]>=0) components&=~3;
      if(g.planes[p.s]>=0) components&=~5;
      wall_remainders(v,f,hypot(obs.x-src.x,obs.y-src.y),src.z,obs.z,sw,ow,values,components,g.radial);
      T weight=T(ro[3])*T(rs[3])*T(g.areas2[p.t])*T(g.areas2[p.s]);
      wall_moments(sums,&scalar,values,obs,src,p,weight,ro[3]*rs[3]*g.areas2[p.t]*g.areas2[p.s]);
    }
  }
  int pair=wall_segment_pair(v,zs,zo);
  Stack stack{v.epsilon,v.sigma,v.edges,v.L,0,0};
  for(int i=v.offsets[pair];i<v.offsets[pair+1];++i) {
    WallRay ray=wall_ray(v,p,g,i);
    bool near=image_near(p,ray.distance);
    rule=pair_rule(rules,stride,adaptive,p,ray.distance);
    C coefficients[5];bool nonzero=false;
    for(int c=0;c<5;++c) {coefficients[c]=wall_coefficient(v,f,ray,c);nonzero|=coefficients[c]!=C(0);}
    if(!nonzero) continue;
    for(int q=lane;q<rule.count*rule.count;q+=32) {
      const double *ro=rule.data+4*(q/rule.count),*rs=rule.data+4*(q%rule.count);
      V3 obs=quadrature_point(p.test,ro),src=quadrature_point(p.source,rs);
      double rho=hypot(obs.x-src.x,obs.y-src.y);
      double d=v.direct[i]==1 ? fabs(obs.z-src.z):
          fmax(0.0,ray.source_scale*src.z+ray.observer_scale*obs.z+ray.offset);
      double R=hypot(rho,d);
      C wave(0);
      if(v.direct[i]) {
        C epsilon=eps(v.epsilon,v.sigma,region(stack,fmin(src.z,obs.z)),2*pi*v.frequencies[f]);
        wave=wave_remainder(root(pow(2*pi*v.frequencies[f]/c0,2)*epsilon),R);
      }
      C values[5]{};
      for(int c=0;c<5;++c) {
        if(coefficients[c]==C(0)) continue;
        if(c==1||c==2) {
          if(!near && rho!=0) values[c]=coefficients[c]*(rho/(4*pi*R*(R+d)));
        } else values[c]=coefficients[c]*(wave+(near ? C(0):C(1/(4*pi*R))));
      }
      T weight=T(ro[3])*T(rs[3])*T(g.areas2[p.t])*T(g.areas2[p.s]);
      wall_moments(sums,&scalar,values,obs,src,p,weight,ro[3]*rs[3]*g.areas2[p.t]*g.areas2[p.s]);
    }
  }
  for(int k=0;k<10;++k) {
    C value=k==9?warp_sum(scalar):warp_sum(sums[k]);
    int target=k==9||!p.transpose ? k:(k%3)*3+k/3;
    if(lane==0) tile[target]=value;
  }
}
__device__ void wall_static(FacePair p,VerticalGreenTableView v,GeometryView g,
    const double* rules,int stride,bool adaptive,int f,int lane,C* tile) {
  C sums[10]{};
  int pair=wall_segment_pair(v,g.centres[3*p.s+2],g.centres[3*p.t+2]);
  for(int image=v.offsets[pair];image<v.offsets[pair+1];++image) {
    WallRay ray=wall_ray(v,p,g,image);
    if(!image_near(p,ray.distance)) continue;
    C ca=wall_coefficient(v,f,ray,4),cz=wall_coefficient(v,f,ray,0),cp=wall_coefficient(v,f,ray,3);
    C zrho=wall_coefficient(v,f,ray,1),rhoz=wall_coefficient(v,f,ray,2);
    if(ca==C(0)&&cz==C(0)&&cp==C(0)&&zrho==C(0)&&rhoz==C(0)) continue;
    double source[9];
    for(int k=0;k<9;++k) source[k]=(k%3==2) ?
        ray.source_height_scale*p.source[k]+ray.source_height_offset:p.source[k];
    V3 n=cross(vertex(source,1)-vertex(source,0),vertex(source,2)-vertex(source,0));
    n=n*(1/norm(n));
    int level=0;double r2=(p.lateral2+ray.distance*ray.distance)/p.size2;
    if(adaptive&&r2>=.5) level=1;
    if(adaptive&&r2>=1) level=2;
    if(adaptive&&r2>=2) level=3;
    constexpr int counts[4]={1024,576,256,64};
    int Q=adaptive?counts[level]:stride;
    const double* rule=rules+4*stride*level;
    for(int q=lane;q<Q;q+=32) {
      const double* ro=rule+4*q;V3 obs=quadrature_point(p.test,ro);
      StaticIntegral inner=triangle_static(source,obs);
      V3 projection=obs-n*dot(obs-vertex(source,0),n);
      double weight=ro[3]*g.areas2[p.t]/(4*pi);
      sums[9]+=cp*(weight*inner.scalar);
      double horizontal[3]{};
      if(zrho!=C(0)) triangle_cross_static(source,obs,ray.observer_scale,inner,g.areas2[p.s]*.5,horizontal);
      for(int a=0;a<3;++a) for(int b=0;b<3;++b) {
        V3 av=obs-vertex(p.test,a);
        V3 moment=inner.moment+(projection-vertex(source,b))*inner.scalar;
        sums[3*a+b]+=weight*(ca*(av.x*moment.x+av.y*moment.y)+
            cz*(av.z*moment.z/ray.source_height_scale)+zrho*(av.z*horizontal[b]));
      }
    }
    if(rhoz!=C(0)) {
      // Reciprocity integrates the horizontal test numerator on the inner
      // triangle after exchanging source/observer. Its radial direction reverses,
      // so the exchanged zrho coefficient is -rhoz.
      double test[9];
      for(int k=0;k<9;++k) test[k]=(k%3==2) ?
          -ray.observer_scale/ray.source_scale*p.test[k]-ray.offset/ray.source_scale:p.test[k];
      for(int q=lane;q<Q;q+=32) {
        const double* ro=rule+4*q;V3 obs=quadrature_point(p.source,ro);
        StaticIntegral inner=triangle_static(test,obs);
        double horizontal[3];
        triangle_cross_static(test,obs,ray.source_scale,inner,g.areas2[p.t]*.5,horizontal);
        double weight=ro[3]*g.areas2[p.s]/(4*pi);
        for(int a=0;a<3;++a) for(int b=0;b<3;++b)
          sums[3*a+b]-=rhoz*(weight*horizontal[a]*(obs.z-p.source[3*b+2]));
      }
    }
  }
  for(int k=0;k<10;++k) {
    C value=warp_sum(sums[k]);int target=k==9||!p.transpose ? k:(k%3)*3+k/3;
    if(lane==0) tile[target]+=value;
  }
}


// A frequency chunk shares these real, geometry-only image moments. Each warp
// owns a face pair and uses fixed reductions; no floating-point atomics.
// The bounded cache is [batch,faces,source_tile,10]. Regular geometry uses
// 64 logical rule bytes and ~120 real FLOP/sample; analytic geometry uses
// 32 rule bytes and ~500 FLOP/node plus SFU work, paid once per chunk.
// Coefficient application adds 40 FLOP and <=320 cache bytes/image/pair/f.
template<typename T> __device__ inline T real_warp_sum(T value) {
  for(int delta=16;delta>0;delta/=2) value+=__shfl_down_sync(0xffffffff,value,delta);
  return __shfl_sync(0xffffffff,value,0);
}
template<typename T,bool Near> __device__ void horizontal_static(
    FacePair p,GreenTableView table,GeometryView g,const double* rules,int stride,
    bool adaptive,int requested,int batches,int lane,int64_t pair_index,int64_t pairs,C* cache) {
  for(int image=table.image_offsets[p.pair];image<table.image_offsets[p.pair+1];++image) {
    double d=table.distances[image];
    if(image_near(p,d)!=Near) continue;
    // Scalar-only images omit all nine vector moments and cache updates.
    bool vector_nonzero=false,scalar_nonzero=false;
    for(int batch=0;batch<batches;++batch) {
      int f=requested<0?batch:requested;
      vector_nonzero|=table.images[f*2*table.image_count+image]!=C(0);
      scalar_nonzero|=table.images[(f*2+1)*table.image_count+image]!=C(0);
    }
    if(!vector_nonzero&&!scalar_nonzero) continue;
    T sums[10]{};
    double scalar=0;
    if constexpr(Near) {
      int level=0;double r2=(p.lateral2+d*d)/p.size2;
      if(adaptive&&r2>=.5) level=1;
      if(adaptive&&r2>=1) level=2;
      if(adaptive&&r2>=2) level=3;
      constexpr int counts[4]={1024,576,256,64};
      int Q=adaptive?counts[level]:stride;
      const double* rule=rules+4*stride*level;
      for(int q=lane;q<Q;q+=32) {
        const double* ro=rule+4*q;
        V3 obs=quadrature_point(p.test,ro);
        StaticIntegral inner=triangle_static(p.source,{obs.x,obs.y,p.source[2]+d});
        double weight=ro[3]*g.areas2[p.t]/(4*pi);
        sums[9]+=weight*inner.scalar;
        if(vector_nonzero) for(int a=0;a<3;++a) for(int b=0;b<3;++b) {
          double ax=obs.x-p.test[3*a],ay=obs.y-p.test[3*a+1];
          double bx=inner.moment.x+(obs.x-p.source[3*b])*inner.scalar;
          double by=inner.moment.y+(obs.y-p.source[3*b+1])*inner.scalar;
          sums[3*a+b]+=weight*(ax*bx+ay*by);
        }
      }
    } else {
      PairRule rule=pair_rule(rules,stride,adaptive,p,d);
      for(int q=lane;q<rule.count*rule.count;q+=32) {
        const double *ro=rule.data+4*(q/rule.count),*rs=rule.data+4*(q%rule.count);
        auto obs=relative_point<T>(p.test,ro),src=relative_point<T>(p.source,rs);
        const double* po=g.scalar_points+2*(int64_t(p.t)*g.rule_points+(ro-rules)/4);
        const double* ps=g.scalar_points+2*(int64_t(p.s)*g.rule_points+(rs-rules)/4);
        RelativePoint<double> precise_obs{po[0],po[1]},precise_src{ps[0],ps[1]};
        double dx=p.test[0]-p.source[0]+precise_obs.x-precise_src.x;
        double dy=p.test[1]-p.source[1]+precise_obs.y-precise_src.y;
        double precise_inverse=rsqrt(fma(dx,dx,fma(dy,dy,d*d)))*(1/(4*pi));
        T inverse=T(precise_inverse);
        T weight=T(ro[3])*T(rs[3])*T(g.areas2[p.t])*T(g.areas2[p.s]);
        T value=inverse*weight;
        scalar=fma(precise_inverse,ro[3]*rs[3]*g.areas2[p.t]*g.areas2[p.s],scalar);
        if(vector_nonzero) for(int a=0;a<3;++a) for(int b=0;b<3;++b) {
          T ax=obs.x-T(p.test[3*a]-p.test[0]),ay=obs.y-T(p.test[3*a+1]-p.test[1]);
          T bx=src.x-T(p.source[3*b]-p.source[0]),by=src.y-T(p.source[3*b+1]-p.source[1]);
          if constexpr(sizeof(T)==4)
            sums[3*a+b]=fmaf(value,fmaf(ax,bx,ay*by),sums[3*a+b]);
          else sums[3*a+b]+=value*(ax*bx+ay*by);
        }
      }
    }
    for(int k=0;k<10;++k) {
      if(k==9 ? !scalar_nonzero:!vector_nonzero) continue;
      double moment;
      if constexpr(Near) moment=double(real_warp_sum(sums[k]));
      else moment=k==9?real_warp_sum(scalar):double(real_warp_sum(sums[k]));
      int target=k==9||!p.transpose?k:(k%3)*3+k/3;
      for(int batch=lane;batch<batches;batch+=32) {
        int f=requested<0?batch:requested;
        C coefficient=table.images[(f*2+(k==9))*table.image_count+image];
        cache[(int64_t(batch)*pairs+pair_index)*10+target]+=coefficient*moment;
      }
    }
  }
}
template<typename Far,bool Near> __global__ void horizontal_static_geometry(
    GreenTableView table,GeometryView g,const double* rules,int stride,bool adaptive,
    int requested,int batches,int begin,int count,double ratio,C* cache) {
  int64_t pair=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
  int64_t pairs=int64_t(g.faces)*count;
  if(pair>=pairs) return;
  int lane=threadIdx.x%32;
  if constexpr(!Near) for(int batch=lane;batch<batches;batch+=32)
    for(int k=0;k<10;++k) cache[(int64_t(batch)*pairs+pair)*10+k]=C(0);
  FacePair p=face_pair(g,table,pair/count,begin+pair%count,ratio);
  if(p.pair<0) return;
  if(Near||pair_near(p,table))
    horizontal_static<double,Near>(p,table,g,rules,stride,adaptive,requested,batches,lane,pair,pairs,cache);
  else horizontal_static<Far,Near>(p,table,g,rules,stride,adaptive,requested,batches,lane,pair,pairs,cache);
}

template<typename T,int B,int Class> __global__ void regular_face_pairs(
    GreenTableView table,GeometryView g,const double* rule,int stride,bool adaptive,
    int first,int stop,int requested,int begin,int count,double ratio,C* tile) {
  int64_t pair=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
  if(pair>=int64_t(g.faces)*count) return;
  int lane=threadIdx.x%32;
  FacePair p=face_pair(g,table,pair/count,begin+pair%count,ratio);
  if(p.pair<0) return;
  if constexpr(Class!=0) if(pair_near(p,table)!=(Class==1)) return;
  regular_pair<T,B>(p,table,g,rule,stride,adaptive,first,stop,requested,lane,pair,
                    int64_t(g.faces)*count,tile);
}

// Wall remainders retain their fixed K-by-K sum and cached z weights.
// Horizontal/wall pairs read three Hermite components instead of five:
// only phi, Ahh and the cross component with a nonzero RWG z numerator.
template<typename Far> __global__ void wall_regular_face_pairs(
    GreenTableView table,VerticalGreenTableView v,GeometryView g,const double* rule,
    int stride,bool adaptive,int f,int begin,int count,double ratio,C* tile) {
  int64_t pair=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
  if(pair>=int64_t(g.faces)*count) return;
  int lane=threadIdx.x%32;
  FacePair p=face_pair(g,table,pair/count,begin+pair%count,ratio);
  if(p.pair>=0) return;
  if(wall_near(p,v,g)) wall_regular<double>(p,table,v,g,rule,stride,adaptive,f,lane,tile+10*pair);
  else wall_regular<Far>(p,table,v,g,rule,stride,adaptive,f,lane,tile+10*pair);
}

// Wall near static images: one warp/pair, A admitted images, Qn outer nodes.
// Traffic/pair: ~256 B geometry + 40I B images + 32*A*Qn B rule + 320 B
// tile read/write. FLOP/node/image: ~420 analytic + ~170 block moments;
// 3 log/asinh, one solid-angle atan2 and ~12 sqrt are separate SFU work.
__global__ void near_image_pairs(GreenTableView table,VerticalGreenTableView v,GeometryView g,const double* rules,int stride,
    bool adaptive,int f,int begin,int count,double ratio,C* tile) {
  int64_t pair=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
  if(pair>=int64_t(g.faces)*count) return;
  int lane=threadIdx.x%32;
  FacePair p=face_pair(g,table,pair/count,begin+pair%count,ratio);
  if(p.pair<0) wall_static(p,v,g,rules,stride,adaptive,f,lane,tile+10*pair);
}

// Gather: one thread/(m,n), at most four face pairs in (T+,S+), (T+,S-),
// (T-,S+), (T-,S-) order. At most 4*(32 B tile + 32 B geometry) + 32 B Z
// traffic, ~128 real FLOP/entry. Enumerate source-face slots and let the lower
// incident face own edges with two supports in this tile. This visits each
// (m,n) at most twice over ALL tiles: O(N^2), not N^2 work per tile.
// No float atomics or reduction across tiles.
__global__ void gather_mp_ie(GeometryView g,const C* tile,C* z,int begin,int count,
    const double* frequencies,int f) {
  int64_t thread=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(thread>=int64_t(g.edges)*count*3) return;
  int m=thread/(count*3),face=begin+(thread/3)%count,slot=thread%3;
  if(g.slopes[3*face+slot]==0) return;
  int n=g.face_edges[3*face+slot];
  int owner=g.supports[2*n+1]<0 ? g.supports[2*n]:min(g.supports[2*n],g.supports[2*n+1]);
  if(owner>=begin && owner<begin+count && face!=owner) return;
  int64_t index=int64_t(m)*g.edges+n;
  double w=2*pi*frequencies[f];
  C value(0);
  constexpr double mu0=1/(eps0*c0*c0);
  for(int a=0;a<2;++a) for(int b=0;b<2;++b) {
    int t=g.supports[2*m+a],s=g.supports[2*n+b];
    if(t<0 || s<begin || s>=begin+count) continue;
    int i=g.slots[2*m+a],j=g.slots[2*n+b];
    const C* block=tile+10*(int64_t(t)*count+s-begin);
    double slope=g.slopes[3*t+i]*g.slopes[3*s+j];
    value+=slope*(C(0,-w*mu0)*block[3*i+j]+C(0,4/(w*eps0))*block[9]);
  }
  z[index]+=value;
}

// Separate potentials use the identical face-pair buffer; ~64 B/entry.
// One owner per V entry and P face pair, fixed support order, no atomics.
__global__ void gather_potentials(GeometryView g,const C* tile,C* v,C* p,int begin,int count) {
  int64_t thread=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(thread<int64_t(g.faces)*count) {
    int t=thread/count,s=begin+thread%count;
    // Unit-integral P0 tests: P is the face-average kernel.
    p[int64_t(t)*g.faces+s]=tile[10*thread+9]*(4/(g.areas2[t]*g.areas2[s]));
  }
  if(thread>=int64_t(g.edges)*count*3) return;
  int m=thread/(count*3),face=begin+(thread/3)%count,slot=thread%3;
  if(g.slopes[3*face+slot]==0) return;
  int n=g.face_edges[3*face+slot];
  int owner=g.supports[2*n+1]<0 ? g.supports[2*n]:min(g.supports[2*n],g.supports[2*n+1]);
  if(owner>=begin && owner<begin+count && face!=owner) return;
  C value(0);
  for(int a=0;a<2;++a) for(int b=0;b<2;++b) {
    int t=g.supports[2*m+a],s=g.supports[2*n+b];
    if(t<0 || s<begin || s>=begin+count) continue;
    int i=g.slots[2*m+a],j=g.slots[2*n+b];
    value+=g.slopes[3*t+i]*g.slopes[3*s+j]*tile[10*(int64_t(t)*count+s-begin)+3*i+j];
  }
  v[int64_t(m)*g.edges+n]+=value;
}

// Oracle-facing primitive: 120 B input + 32 B output per triangle query,
// ~420 real FLOP plus the analytic SFU operations listed above.
__global__ void static_queries(const double* triangles,const double* points,const double* free,
    double* scalar,double* vector,int count) {
  int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=count) return;
  const double* triangle=triangles+9*q;
  V3 obs=vertex(points,q),v=vertex(free,q);
  StaticIntegral result=triangle_static(triangle,obs);
  V3 n=cross(vertex(triangle,1)-vertex(triangle,0),vertex(triangle,2)-vertex(triangle,0));
  n=n*(1/norm(n));
  V3 projection=obs-n*dot(obs-vertex(triangle,0),n);
  V3 moment=result.moment+(projection-v)*result.scalar;
  scalar[q]=result.scalar;vector[3*q]=moment.x;vector[3*q+1]=moment.y;vector[3*q+2]=moment.z;
}

std::vector<Tensor> layered_triangle_static(Tensor triangles,Tensor points,Tensor free) {
  c10::cuda::CUDAGuard guard(triangles.device());
  auto scalar=at::empty({points.size(0)},points.options()),vector=at::empty_like(points);
  static_queries<<<(points.size(0)+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      triangles.data_ptr<double>(),points.data_ptr<double>(),free.data_ptr<double>(),
      scalar.data_ptr<double>(),vector.data_ptr<double>(),points.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {scalar,vector};
}

std::vector<Tensor> assemble_impl(std::vector<Tensor> green,std::vector<Tensor> geometry,Tensor far_rule,
    Tensor near_rule,int64_t requested,int64_t source_tile,double ratio,bool fp64,int64_t mode,double ground,bool parts,bool tiles=false) {
  c10::cuda::CUDAGuard guard(geometry[0].device());
  GreenTableView table{};
  table.values=green[0].data_ptr<C>();table.slopes=green[1].data_ptr<C>();
  table.rho=green[2].data_ptr<double>();table.plane_z=green[3].data_ptr<double>();
  table.image_offsets=green[4].data_ptr<int64_t>();table.distances=green[5].data_ptr<double>();
  table.images=green[6].data_ptr<C>();table.epsilon=green[7].data_ptr<C>();
  table.sigma=green[8].data_ptr<double>();table.frequency_hz=green[9].data_ptr<double>();
  table.frequencies=green[9].numel();table.pairs=green[0].size(1);table.nodes=green[2].numel();
  table.image_count=green[5].numel();table.metal_planes=green[3].numel();
  table.exact_mode=mode;table.ground=ground;
  GeometryView g{geometry[0].data_ptr<double>(),geometry[2].data_ptr<double>(),
      geometry[3].data_ptr<double>(),geometry[4].data_ptr<double>(),geometry[5].data_ptr<double>(),
      geometry[1].data_ptr<int64_t>(),geometry[6].data_ptr<int64_t>(),geometry[7].data_ptr<int64_t>(),
      geometry[8].data_ptr<int64_t>(),
      int(geometry[0].size(0)),int(geometry[6].size(0))};
  VerticalGreenTableView vertical{};
  if(green.size()>10) {
    std::vector<Tensor> tensors(green.begin()+10,green.end());
    vertical=vertical_view(tensors,0);
  }
  int batches=requested<0 ? table.frequencies : 1;
  if(tiles) source_tile=g.faces;
  int far_stride=far_rule.size(-2),near_stride=near_rule.size(-2);
  bool far_adaptive=far_rule.dim()==3,near_adaptive=near_rule.dim()==3;
  auto z=at::zeros({batches,g.edges,g.edges},green[0].options());
  auto p=at::empty({parts ? batches : 0,g.faces,g.faces},green[0].options());
  auto buffer=at::empty({batches,g.faces,std::min(int(source_tile),g.faces),10},green[0].options());
  auto stream=c10::cuda::getCurrentCUDAStream();
  auto radial=at::empty({2},green[2].options());
  g.radial=radial.data_ptr<double>();
  radial_metadata_kernel<<<1,1,0,stream>>>(table.rho,table.nodes,radial.data_ptr<double>());
  g.rule_points=far_rule.numel()/4;
  auto points=at::empty({g.faces,g.rule_points,2},far_rule.options());
  g.scalar_points=points.data_ptr<double>();
  scalar_points_kernel<<<(int64_t(g.faces)*g.rule_points+127)/128,128,0,stream>>>(
      g,far_rule.data_ptr<double>(),points.data_ptr<double>());
  Tensor weights;
  if(green.size()>10 && !mode) {
    g.rule_points=far_rule.numel()/4;g.z_weight_stride=2*vertical.K+2;
    weights=at::empty({g.faces,g.rule_points,g.z_weight_stride},far_rule.options());
    g.z_weights=weights.data_ptr<double>();
    wall_weights_kernel<<<(int64_t(g.faces)*g.rule_points+127)/128,128,0,stream>>>(
        g,vertical,far_rule.data_ptr<double>(),weights.data_ptr<double>(),g.rule_points,g.z_weight_stride);
  }
  // Reuse static geometry within the admitted frequency chunk. Gather keeps
  // the same support order and complex128 public potential layout.
  for(int begin=0;begin<g.faces;begin+=source_tile) {
    int count=std::min(int(source_tile),g.faces-begin);
    int blocks=(int64_t(g.faces)*count+3)/4;
    if(fp64) horizontal_static_geometry<double,false><<<blocks,128,0,stream>>>(
        table,g,far_rule.data_ptr<double>(),far_stride,far_adaptive,requested,batches,begin,count,ratio,buffer.data_ptr<C>());
    else horizontal_static_geometry<float,false><<<blocks,128,0,stream>>>(
        table,g,far_rule.data_ptr<double>(),far_stride,far_adaptive,requested,batches,begin,count,ratio,buffer.data_ptr<C>());
    horizontal_static_geometry<double,true><<<blocks,128,0,stream>>>(
        table,g,near_rule.data_ptr<double>(),near_stride,near_adaptive,requested,batches,begin,count,ratio,buffer.data_ptr<C>());
    if(fp64) {
      for(int batch=0;batch<batches;++batch)
        regular_face_pairs<double,1,0><<<blocks,128,0,stream>>>(
            table,g,far_rule.data_ptr<double>(),far_stride,far_adaptive,batch,batches,
            requested,begin,count,ratio,buffer.data_ptr<C>());
    } else {
      for(int batch=0;batch<batches;batch+=2) {
        if(batch+1<batches) regular_face_pairs<double,2,1><<<blocks,128,0,stream>>>(
            table,g,far_rule.data_ptr<double>(),far_stride,far_adaptive,batch,batches,
            requested,begin,count,ratio,buffer.data_ptr<C>());
        else regular_face_pairs<double,1,1><<<blocks,128,0,stream>>>(
            table,g,far_rule.data_ptr<double>(),far_stride,far_adaptive,batch,batches,
            requested,begin,count,ratio,buffer.data_ptr<C>());
      }
      for(int batch=0;batch<batches;batch+=4) {
        switch(std::min(4,batches-batch)) {
          case 1: regular_face_pairs<float,1,2><<<blocks,128,0,stream>>>(
            table,g,far_rule.data_ptr<double>(),far_stride,far_adaptive,batch,batches,
            requested,begin,count,ratio,buffer.data_ptr<C>());break;
          case 2: regular_face_pairs<float,2,2><<<blocks,128,0,stream>>>(
            table,g,far_rule.data_ptr<double>(),far_stride,far_adaptive,batch,batches,
            requested,begin,count,ratio,buffer.data_ptr<C>());break;
          case 3: regular_face_pairs<float,3,2><<<blocks,128,0,stream>>>(
            table,g,far_rule.data_ptr<double>(),far_stride,far_adaptive,batch,batches,
            requested,begin,count,ratio,buffer.data_ptr<C>());break;
          case 4: regular_face_pairs<float,4,2><<<blocks,128,0,stream>>>(
            table,g,far_rule.data_ptr<double>(),far_stride,far_adaptive,batch,batches,
            requested,begin,count,ratio,buffer.data_ptr<C>());break;
        }
      }
    }
    for(int batch=0;batch<batches;++batch) {
    int f=requested<0 ? batch : requested;
    C* tile=buffer.data_ptr<C>()+int64_t(batch)*g.faces*count*10;
    if(green.size()>10) {
      if(fp64) wall_regular_face_pairs<double><<<blocks,128,0,stream>>>(
          table,vertical,g,far_rule.data_ptr<double>(),far_stride,far_adaptive,f,begin,count,ratio,tile);
      else wall_regular_face_pairs<float><<<blocks,128,0,stream>>>(
          table,vertical,g,far_rule.data_ptr<double>(),far_stride,far_adaptive,f,begin,count,ratio,tile);
    }
    near_image_pairs<<<blocks,128,0,stream>>>(table,vertical,g,near_rule.data_ptr<double>(),near_stride,near_adaptive,
                                            f,begin,count,ratio,tile);
    if(tiles) {C10_CUDA_KERNEL_LAUNCH_CHECK();return {buffer.select(0,0)};}
    if(parts) gather_potentials<<<(std::max(int64_t(g.edges)*count*3,int64_t(g.faces)*count)+127)/128,128,0,stream>>>(
        g,tile,z.data_ptr<C>()+int64_t(batch)*g.edges*g.edges,
        p.data_ptr<C>()+int64_t(batch)*g.faces*g.faces,begin,count);
    else gather_mp_ie<<<(int64_t(g.edges)*count*3+127)/128,128,0,stream>>>(
        g,tile,z.data_ptr<C>(),begin,count,table.frequency_hz,f);
  }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if(requested>=0) return {z.select(0,0),parts ? p.select(0,0) : p};
  return {z,p};
}

Tensor layered_assemble(std::vector<Tensor> green,std::vector<Tensor> geometry,Tensor far_rule,
    Tensor near_rule,int64_t f,int64_t source_tile,double ratio,bool fp64,int64_t mode,double ground) {
  return assemble_impl(green,geometry,far_rule,near_rule,f,source_tile,ratio,fp64,mode,ground,false)[0];
}
std::vector<Tensor> layered_assemble_parts(std::vector<Tensor> green,std::vector<Tensor> geometry,Tensor far_rule,
    Tensor near_rule,int64_t f,int64_t source_tile,double ratio,bool fp64,int64_t mode,double ground) {
  return assemble_impl(green,geometry,far_rule,near_rule,f,source_tile,ratio,fp64,mode,ground,true);
}


Tensor layered_assemble_tiles(std::vector<Tensor> green,std::vector<Tensor> geometry,Tensor far_rule,
    Tensor near_rule,int64_t f,int64_t source_tile,double ratio,bool fp64,int64_t mode,double ground) {
  return assemble_impl(green,geometry,far_rule,near_rule,f,source_tile,ratio,fp64,mode,ground,false,true)[0];
}
__global__ void radial_indices_kernel(const double* grid,const double* queries,int64_t* out,int nodes,int64_t count,const double* radial) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<count) out[i]=radial_interval(grid,nodes,queries[i],radial[0],radial[1]);
}
Tensor layered_radial_indices(Tensor grid,Tensor queries) {
  c10::cuda::CUDAGuard guard(grid.device());auto out=at::empty(queries.sizes(),queries.options().dtype(at::kLong));
  auto radial=at::empty({2},grid.options());
  radial_metadata_kernel<<<1,1,0,c10::cuda::getCurrentCUDAStream()>>>(
      grid.data_ptr<double>(),grid.numel(),radial.data_ptr<double>());
  radial_indices_kernel<<<(queries.numel()+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      grid.data_ptr<double>(),queries.data_ptr<double>(),out.data_ptr<int64_t>(),grid.numel(),queries.numel(),radial.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}


__global__ void orient_kernel(const int64_t* faces,const int64_t* incidence,const int64_t* edges,
    const int64_t* signs,int64_t* out,int64_t* work,int F) {
  if(blockIdx.x || threadIdx.x) return;
  int64_t* flip=work;int64_t* queue=work+F;
  for(int f=0;f<F;++f) flip[f]=-1;
  for(int root=0;root<F;++root) {
    if(flip[root]>=0) continue;
    flip[root]=0;queue[0]=root;int head=0,tail=1;
    while(head<tail) {
      int a=queue[head++];
      for(int slot=0;slot<3;++slot) {
        int e=edges[3*a+slot],left=incidence[2*e],right=incidence[2*e+1];
        int b=left==a ? right:left;
        if(b<0 || flip[b]>=0) continue;
        int other=0;while(edges[3*b+other]!=e) ++other;
        flip[b]=flip[a]^(signs[3*a+slot]==signs[3*b+other]);
        queue[tail++]=b;
      }
    }
  }
  for(int f=0;f<F;++f) {
    out[3*f]=faces[3*f];
    out[3*f+1]=faces[3*f+(flip[f]?2:1)];
    out[3*f+2]=faces[3*f+(flip[f]?1:2)];
  }
}
Tensor layered_orient(Tensor faces,Tensor incidence,Tensor edges,Tensor signs) {
  c10::cuda::CUDAGuard guard(faces.device());auto out=at::empty_like(faces);
  auto work=at::empty({2*faces.size(0)},faces.options());
  orient_kernel<<<1,1,0,c10::cuda::getCurrentCUDAStream()>>>(faces.data_ptr<int64_t>(),
      incidence.data_ptr<int64_t>(),edges.data_ptr<int64_t>(),signs.data_ptr<int64_t>(),
      out.data_ptr<int64_t>(),work.data_ptr<int64_t>(),faces.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}


__global__ void cross_static_queries(const double* triangles,const double* points,const double* beta,
    double* out,int count) {
  int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=count) return;
  const double* triangle=triangles+9*q;V3 obs=vertex(points,q);
  double area=.5*norm(cross(vertex(triangle,1)-vertex(triangle,0),vertex(triangle,2)-vertex(triangle,0)));
  triangle_cross_static(triangle,obs,beta[q],triangle_static(triangle,obs),area,out+3*q);
}
Tensor layered_triangle_cross_static(Tensor triangles,Tensor points,Tensor beta) {
  c10::cuda::CUDAGuard guard(triangles.device());auto out=at::empty({points.size(0),3},points.options());
  cross_static_queries<<<(points.size(0)+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      triangles.data_ptr<double>(),points.data_ptr<double>(),beta.data_ptr<double>(),out.data_ptr<double>(),points.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
