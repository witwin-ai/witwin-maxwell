#pragma once

// Shared fp32 primitive distance and fill arithmetic; compiled with --fmad=false.
static __device__ float sphere(const float* g, float x, float y, float z) {
  float dx = x-g[4], dy = y-g[5], dz = z-g[6];
  return sqrtf((dx*dx + dy*dy) + dz*dz)-g[16];
}

static __device__ void local_frame(const float* g, float& x, float& y, float& z) {
  float dx=x-g[4], dy=y-g[5], dz=z-g[6];
  x=(g[7]*dx+g[8]*dy)+g[9]*dz;
  y=(g[10]*dx+g[11]*dy)+g[12]*dz;
  z=(g[13]*dx+g[14]*dy)+g[15]*dz;
}

static __device__ float box(const float* g, float x, float y, float z) {
  local_frame(g,x,y,z);
  float qx=fabsf(x)-g[16], qy=fabsf(y)-g[17], qz=fabsf(z)-g[18];
  float ax=fmaxf(qx,0.f), ay=fmaxf(qy,0.f), az=fmaxf(qz,0.f);
  // ATen's contiguous three-element norm reduces x/z before y.
  return sqrtf((ax*ax+az*az)+ay*ay)+fminf(fmaxf(fmaxf(qx,qy),qz),0.f);
}

static __device__ float cylinder(const float* g, float x, float y, float z) {
  local_frame(g,x,y,z);
  int axis=int(g[18]);
  float axial=axis==0?x:axis==1?y:z;
  float a=axis==0?y:x, b=axis==2?y:z;
  float radial=sqrtf(a*a+b*b)-g[16], cap=fabsf(axial)-g[17];
  float r=fmaxf(radial,0.f), h=fmaxf(cap,0.f);
  return sqrtf(r*r+h*h)+fminf(fmaxf(radial,cap),0.f);
}

static __device__ float smooth_min(float a, float b, const float* g) {
  if(g[2]==0.f) return fminf(a,b);
  float h=fminf(fmaxf(.5f+(.5f*(b-a))/g[3],0.f),1.f);
  return (b+h*(a-b))-((g[3]*h)*(1.f-h));
}

static __device__ __noinline__ float tree_distance(const float* g, float qx, float qy, float qz) {
  float x[16],y[16],z[16],d[16];
  int count=int(g[23]);
  for(int n=0;n<count;++n) {
    const float* node=g+n*24;
    int parent=int(node[1]);
    x[n]=parent<0?qx:x[parent]; y[n]=parent<0?qy:y[parent]; z[n]=parent<0?qz:z[parent];
    if(node[0]>=3.f && node[0]<=5.f) local_frame(node,x[n],y[n],z[n]);
    else d[n]=node[0]==1.f?sphere(node,x[n],y[n],z[n]):
              node[0]==2.f?box(node,x[n],y[n],z[n]):cylinder(node,x[n],y[n],z[n]);
  }
  for(int n=count-1;n>=0;--n) {
    const float* node=g+n*24;
    if(node[0]<3.f || node[0]>5.f) continue;
    int first=-1, tools=-1;
    float result=0.f;
    for(int c=n+1;c<count;++c) if(int(g[c*24+1])==n) {
      if(first<0) { first=c; result=d[c]; }
      else if(node[0]==3.f) result=smooth_min(result,d[c],node);
      else if(node[0]==4.f) result=-smooth_min(-result,-d[c],node);
      else if(tools<0) { tools=c; result=d[c]; }
      else result=smooth_min(result,d[c],node);
    }
    d[n]=node[0]==5.f?-smooth_min(-d[first],result,node):result;
  }
  return d[0];
}

static __device__ float distance(const float* g, float x, float y, float z) {
  if(g[0]==1.f) return sphere(g,x,y,z);
  if(g[0]==2.f) return box(g,x,y,z);
  if(g[0]==6.f) return cylinder(g,x,y,z);
  return tree_distance(g,x,y,z);
}

template <typename Parameters>
static __device__ float coverage(const Parameters& p, const float* g, float x, float y, float z,
                          float hx, float hy, float hz) {
  float d = distance(g,x,y,z), t = p.delta;
  float px=distance(g,x+t,y,z), py=distance(g,x,y+t,z), pz=distance(g,x,y,z+t);
  float mx=distance(g,x-t,y,z), my=distance(g,x,y-t,z), mz=distance(g,x,y,z-t);
  float gx=px-mx, gy=py-my, gz=pz-mz;
  float norm=sqrtf(((gx*gx+gy*gy)+gz*gz)+p.norm_eps);
  float width=fmaxf(((fabsf(gx)*hx+fabsf(gy)*hy)+fabsf(gz)*hz)/norm,
                    fminf(fminf(hx,hy),hz));
  float lap=(((0.f+(px+mx-2.f*d))+(py+my-2.f*d))+(pz+mz-2.f*d))*p.lap_scale;
  float shift=((width*width)*lap)*(1.f/24.f);
  shift=fminf(fmaxf(shift,-.25f*width),.25f*width);
  float f=fminf(fmaxf(.5f-(d+shift)/width,0.f),1.f);
  if(f>.999f) f=1.f;
  return f<.001f ? 0.f : f;
}
