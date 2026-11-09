#pragma once
#include "common.cuh"

namespace layered {
struct V3 {
  double x,y,z;
  __device__ V3 operator+(V3 b) const {return {x+b.x,y+b.y,z+b.z};}
  __device__ V3 operator-(V3 b) const {return {x-b.x,y-b.y,z-b.z};}
  __device__ V3 operator*(double a) const {return {x*a,y*a,z*a};}
};
__device__ inline double dot(V3 a,V3 b) {return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ inline V3 cross(V3 a,V3 b) {
  return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
__device__ inline double norm(V3 a) {return sqrt(dot(a,a));}
__device__ inline V3 vertex(const double* t,int i) {return {t[3*i],t[3*i+1],t[3*i+2]};}

struct StaticIntegral {double scalar;V3 moment;};

// Unnormalized integrals I=int 1/R dS and M=int (r'-projection(obs))/R dS.
// Surface divergence theorem (Wilton/Graglia):
// I=sum_edges d*L-|h|*Omega; M=sum_edges outward*int_edge R dl.
// The signed d retains outside projections; the solid angle handles off-plane
// points. Zero-height edge/vertex limits are taken algebraically, never clipped.
__device__ inline StaticIntegral triangle_static(const double* triangle,V3 obs) {
  V3 v[3]={vertex(triangle,0),vertex(triangle,1),vertex(triangle,2)};
  V3 n=cross(v[1]-v[0],v[2]-v[0]);n=n*(1/norm(n));
  double h=dot(obs-v[0],n);
  V3 a[3]={v[0]-obs,v[1]-obs,v[2]-obs};
  double r[3]={norm(a[0]),norm(a[1]),norm(a[2])};
  StaticIntegral out{0,{0,0,0}};
  for(int i=0;i<3;++i) {
    int j=(i+1)%3;
    V3 edge=v[j]-v[i];double length=norm(edge);
    V3 tangent=edge*(1/length),outward=cross(tangent,n);
    double d=dot(a[i],outward),s0=dot(a[i],tangent),s1=s0+length;
    double radius2=d*d+h*h,L=0;
    if(radius2!=0) {
      // Same-sign endpoints avoid subtracting two large asinh values.
      if(s0>=0) L=log1p(length*(1+(s0+s1)/(r[i]+r[j]))/(r[i]+s0));
      else if(s1<=0) L=log1p(length*(1-(s0+s1)/(r[i]+r[j]))/(r[j]-s1));
      else {double radius=sqrt(radius2);L=asinh(s1/radius)-asinh(s0/radius);}
    }
    out.scalar+=d*L;
    double line=0.5*(s1*r[j]-s0*r[i]+radius2*L);
    out.moment=out.moment+outward*line;
  }
  if(h!=0) {
    double numerator=fabs(dot(a[0],cross(a[1],a[2])));
    double denominator=r[0]*r[1]*r[2]+dot(a[0],a[1])*r[2]+
                       dot(a[1],a[2])*r[0]+dot(a[2],a[0])*r[1];
    out.scalar-=fabs(h)*2*atan2(numerator,denominator);
  }
  return out;
}

// Definite edge integrals of log(R+d) and 1/(R+d), d=beta*(obs.z-r'.z).
// With s along the unit edge, R=sqrt(s*s+A), d=a*s+b and
// t=(s+R)/sqrt(A), K=int ds/[R(R+d)]=(2/sqrt(A))*int dt/Q,
// Q=(1+a)t*t+2*b*t/sqrt(A)+(1-a).
// J=[Delta(log t)-a*Delta(log(R+d))-2*b*int(dt/Q)/sqrt(A)]/(1-a*a).
// Integration by parts gives int log(R+d)= [s log(R+d)]-Delta(s)+b*J+A*K.
// Definite atan2/log1p expressions avoid subtracting large primitives.
struct EdgeLogIntegral {double logarithm,inverse;};
__device__ inline double line_log_zero_radius(double s,double a) {
  if(s==0) return 0;
  return s*(log(fabs(s))+log1p(a*(s>0?1:-1))-1);
}
__device__ inline EdgeLogIntegral edge_log_integrals(V3 start,V3 end,V3 obs,double beta,bool need_inverse) {
  V3 edge=end-start;double length=norm(edge);V3 tangent=edge*(1/length);
  V3 delta=start-obs;double s0=dot(delta,tangent),s1=s0+length;
  V3 perpendicular=delta-tangent*s0;
  double A=dot(perpendicular,perpendicular),a=-beta*tangent.z,b=-beta*perpendicular.z;
  if(A==0) return {line_log_zero_radius(s1,a)-line_log_zero_radius(s0,a),0};
  double radius=sqrt(A),R0=hypot(s0,radius),R1=hypot(s1,radius);
  double difference=length*(s0+s1)/(R0+R1),d0=a*s0+b;
  double value0=R0+d0;
  double t0=s0>=0 ? (s0+R0)/radius:radius/(R0-s0);
  double t1=s1>=0 ? (s1+R1)/radius:radius/(R1-s1);
  double dt;
  if(s0+s1>=0) dt=length/radius*(1+(s0+s1)/(R0+R1));
  else dt=radius*length*(1-(s0+s1)/(R0+R1))/((R0-s0)*(R1-s1));
  double logt=log1p(dt/t0);
  double logvalue=log1p((difference+a*length)/value0);
  if(fabs(a)==1) {
    double inverse=need_inverse ? .5*(s1/(R1+a*s1)-s0/(R0+a*s0)+logt):0;
    return {length*log(value0)+s1*logvalue-a*difference,inverse};
  }
  double B=2*b/radius,D=4*fmax(0.0,(1-a*a)-b*b/A),integral;
  if(D>0) {
    double rootD=sqrt(D);
    integral=2/rootD*atan2(rootD*dt,2*(1+a)*t0*t1+B*(t0+t1)+2*(1-a));
  } else integral=4*(1+a)*dt/((2*(1+a)*t0+B)*(2*(1+a)*t1+B));
  double inverse=(logt-a*logvalue-B*integral)/(1-a*a);
  double logarithm=length*(log(value0)-1)+s1*logvalue+b*inverse+2*radius*integral;
  return {logarithm,need_inverse?inverse:0};
}

// Unnormalized int (r'_h-v_h).(obs_h-r'_h)/[R(R+d)] dS for all
// three free vertices. For horizontal or vertical triangles:
// f_h.H=(obs_h-v_h).H-1+d/R.
// Tangential integration by parts gives int H_t=-sum m_t*int_edge log(R+d).
// On a vertical face int 1/[R(R+d)]=beta*sum m_z*int_edge 1/(R+d).
// The remaining int d/R uses the existing triangle scalar and first moment.
// Thus EVERY near static cross ray has a closed-form inner integral.
__device__ inline void triangle_cross_static(const double* triangle,V3 obs,double beta,
    StaticIntegral inner,double area,double* out) {
  V3 normal=cross(vertex(triangle,1)-vertex(triangle,0),vertex(triangle,2)-vertex(triangle,0));
  normal=normal*(1/norm(normal));
  double height=dot(obs-vertex(triangle,0),normal);
  V3 projection=obs-normal*height,edge_log{0,0,0};
  double edge_inverse=0;
  bool vertical=normal.z==0,need_inverse=vertical&&height!=0;
  for(int i=0;i<3;++i) {
    V3 first=vertex(triangle,i),second=vertex(triangle,(i+1)%3);
    V3 tangent=(second-first)*(1/norm(second-first)),outward=cross(tangent,normal);
    EdgeLogIntegral line=edge_log_integrals(first,second,obs,beta,need_inverse&&outward.z!=0);
    edge_log=edge_log+outward*line.logarithm;
    if(need_inverse&&outward.z!=0) edge_inverse+=outward.z*line.inverse;
  }
  double distance_moment=beta*((obs.z-projection.z)*inner.scalar-inner.moment.z);
  double common=-area+distance_moment+(need_inverse?height*height*beta*edge_inverse:0);
  for(int b=0;b<3;++b)
    out[b]=common-(obs.x-triangle[3*b])*edge_log.x-(obs.y-triangle[3*b+1])*edge_log.y;
}
}
