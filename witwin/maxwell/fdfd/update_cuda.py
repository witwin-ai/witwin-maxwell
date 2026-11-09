"""Fixed-rank native CUDA inverse certificates, operating on owned Torch tensors."""
from functools import lru_cache
import cupy as cp
import numpy as np
import torch

_SOURCE = r'''
using R = REAL;
struct Z {
 R x,y;
 __device__ Z(R a=0,R b=0):x(a),y(b){}
 __device__ Z operator+(Z b) const {return Z(x+b.x,y+b.y);}
 __device__ Z operator-(Z b) const {return Z(x-b.x,y-b.y);}
 __device__ Z operator*(Z b) const {return Z(x*b.x-y*b.y,x*b.y+y*b.x);}
 __device__ Z operator/(Z b) const {
   if(fabs(b.x)>=fabs(b.y)){R t=b.y/b.x,d=b.x+b.y*t;return Z((x+y*t)/d,(y-x*t)/d);}
   R t=b.x/b.y,d=b.y+b.x*t;return Z((x*t+y)/d,(y*t-x)/d);
 }
};
__device__ double mag2(Z a){return double(a.x)*a.x+double(a.y)*a.y;}
extern "C" __global__ void certify(const Z* s,const Z* p,const Z* rhs,
 Z* solution,Z* inverse,double* status,int n,double eps,double tolerance){
 if(threadIdx.x) return;
 int batch=blockIdx.x;s+=batch*n*n;p+=batch*n*n;rhs+=batch*n;solution+=batch*n;inverse+=batch*n*n;status+=batch*2;
 Z a[CAP*CAP],x[CAP*CAP],u[CAP],res[CAP];
 bool valid=true;
 for(int i=0;i<n;i++) for(int j=0;j<n;j++){a[i*CAP+j]=s[i*n+j];x[i*CAP+j]=Z(i==j);}
 for(int k=0;k<n;k++){
   int pivot=k;double best=mag2(a[k*CAP+k]);
   for(int i=k+1;i<n;i++){double v=mag2(a[i*CAP+k]);if(v>best){best=v;pivot=i;}}
   if(!(best>0) || !isfinite(best)){valid=false;break;}
   for(int j=0;j<n;j++){Z t=a[k*CAP+j];a[k*CAP+j]=a[pivot*CAP+j];a[pivot*CAP+j]=t;
     t=x[k*CAP+j];x[k*CAP+j]=x[pivot*CAP+j];x[pivot*CAP+j]=t;}
   Z d=a[k*CAP+k];
   for(int j=0;j<n;j++){a[k*CAP+j]=a[k*CAP+j]/d;x[k*CAP+j]=x[k*CAP+j]/d;}
   for(int i=0;i<n;i++) if(i!=k){Z t=a[i*CAP+k];
     for(int j=0;j<n;j++){a[i*CAP+j]=a[i*CAP+j]-t*a[k*CAP+j];x[i*CAP+j]=x[i*CAP+j]-t*x[k*CAP+j];}}
 }
 double xn=0,dn=0,pn=0,rn=0,bn=0;
 for(int i=0;i<n;i++){
   u[i]=Z();for(int j=0;j<n;j++)u[i]=u[i]+x[i*CAP+j]*rhs[j];
   for(int j=0;j<n;j++){
     Z d=Z(i==j);for(int k=0;k<n;k++)d=d-s[i*n+k]*x[k*CAP+j];
     dn+=mag2(d);xn+=mag2(x[i*CAP+j]);pn+=mag2(p[i*n+j]);
     inverse[i*n+j]=x[i*CAP+j];
   }
 }
 for(int iteration=0;iteration<2;iteration++){
   for(int i=0;i<n;i++){res[i]=rhs[i];for(int j=0;j<n;j++)res[i]=res[i]-s[i*n+j]*u[j];}
   for(int i=0;i<n;i++)for(int j=0;j<n;j++)u[i]=u[i]+x[i*CAP+j]*res[j];
 }
 for(int i=0;i<n;i++){
   Z d=rhs[i];for(int j=0;j<n;j++)d=d-s[i*n+j]*u[j];
   rn+=mag2(d);bn+=mag2(rhs[i]);solution[i]=u[i];
 }
 double lower=(1-sqrt(dn))/sqrt(xn),threshold=16*eps*n*fmax(1.,sqrt(pn));
 double relative=bn>0?sqrt(rn/bn):(rn==0?0:(1.0/0.0));
 status[0]=valid && isfinite(lower) && lower>threshold && isfinite(relative) && relative<=tolerance;
 status[1]=relative;
}
'''


@lru_cache(None)
def _kernel(capacity, double):
    source=_SOURCE.replace('REAL','double' if double else 'float').replace('CAP',str(capacity))
    return cp.RawKernel(source,'certify',options=('--std=c++17',))


def solve_small(matrix, product, rhs, tolerance):
    n=matrix.shape[0]
    capacity=next(k for k in (4,8,16,32) if k>=n)
    solution=torch.empty_like(rhs)
    inverse=torch.empty_like(matrix)
    status=torch.empty(2,device=matrix.device,dtype=torch.float64)
    with cp.cuda.Device(matrix.device.index), cp.cuda.Stream.from_external(torch.cuda.current_stream(matrix.device)):
        _kernel(capacity,matrix.dtype==torch.complex128)((1,),(1,),
            (np.uint64(matrix.data_ptr()),np.uint64(product.data_ptr()),np.uint64(rhs.data_ptr()),
             np.uint64(solution.data_ptr()),np.uint64(inverse.data_ptr()),np.uint64(status.data_ptr()),
             n,torch.finfo(rhs.real.dtype).eps,tolerance))
    safe,residual=status.cpu().tolist()
    return solution,residual,inverse,bool(safe)

_SOURCE += r'''
extern "C" __global__ void rows(const Z* g,const Z* inv,const Z* u,
 const Z* old_delta,const Z* delta,const bool* old_pec,const bool* pec,
 const long long* changed,Z* v,Z* selected,Z* forcing,int n,int r,int batches){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=n*r*batches)return;
 int batch=t/(n*r);t%=n*r;delta+=batch*n;pec+=batch*n;changed+=batch*r;v+=batch*n*r;selected+=batch*n*r;forcing+=batch*r;
 int i=t/n,j=t%n,k=changed[i];
 Z old_b=old_pec[k]?Z(1):old_delta[k],b=pec[k]?Z(1):delta[k];
 Z db=b-old_b,da=Z(int(!pec[k])-int(!old_pec[k]));
 v[i*n+j]=db*g[k*n+j]+(j==k?da:Z());
 selected[j*r+i]=inv[j*n+k];
 if(j==0)forcing[i]=db*u[k];
}
extern "C" __global__ void contractions(const Z* v,const Z* selected,const Z* current,
 Z* p,Z* small,Z* base,int n,int r,int batches){
 int batch=blockIdx.x/(r*r),block=blockIdx.x%(r*r);v+=batch*n*r;selected+=batch*n*r;p+=batch*r*r;small+=batch*r*r;base+=batch*r;
 int row=block/r,col=block%r,t=threadIdx.x;
 Z a,b;for(int k=t;k<n;k+=blockDim.x){a=a+v[row*n+k]*selected[k*r+col];if(col==0)b=b+v[row*n+k]*current[k];}
 __shared__ R ax[128],ay[128],bx[128],by[128];ax[t]=a.x;ay[t]=a.y;bx[t]=b.x;by[t]=b.y;__syncthreads();
 for(int stride=64;stride;stride/=2){if(t<stride){ax[t]+=ax[t+stride];ay[t]+=ay[t+stride];bx[t]+=bx[t+stride];by[t]+=by[t+stride];}__syncthreads();}
 if(t==0){p[row*r+col]=Z(ax[0],ay[0]);small[row*r+col]=Z(ax[0],ay[0])+Z(row==col);if(col==0)base[row]=Z(bx[0],by[0]);}
}
extern "C" __global__ void finish(const Z* selected,const Z* projected_inv,const Z* current,
 const Z* observed,const Z* forcing,const Z* correction,const long long* changed,
 Z* new_current,Z* new_observed,int n,int q,int r,int batches){
 int width=n>q?n:q,i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=batches*width)return;
 int batch=i/width;i%=width;selected+=batch*n*r;forcing+=batch*r;correction+=batch*r;changed+=batch*r;new_current+=batch*n;new_observed+=batch*q;
 if(i<n){Z value=current[i];for(int k=0;k<r;k++)value=value+selected[i*r+k]*(forcing[k]-correction[k]);new_current[i]=value;}
 if(i<q){Z value=observed[i];for(int k=0;k<r;k++)value=value-projected_inv[i*n+changed[k]]*(forcing[k]-correction[k]);new_observed[i]=value;}
}
'''


@lru_cache(None)
def _update_kernel(name, double):
    source=_SOURCE.replace('REAL','double' if double else 'float').replace('CAP','4')
    return cp.RawKernel(source,name,options=('--std=c++17',))


def _launch(name, like, grid, block, args):
    # Pass device addresses as launch metadata, avoiding a DLPack stream handshake
    # for every argument. Torch owns all storage and the kernel runs on its current
    # stream, so allocator reuse remains ordered after this launch.
    with cp.cuda.Device(like.device.index), cp.cuda.Stream.from_external(torch.cuda.current_stream(like.device)):
        _update_kernel(name,like.dtype==torch.complex128)(grid,block,
            tuple(np.uint64(x.data_ptr()) if isinstance(x,torch.Tensor) else x for x in args))


def assemble_update(response, delta, pec, changed):
    n,r=response._u0.numel(),changed.numel()
    v=torch.empty((r,n),device=delta.device,dtype=delta.dtype)
    # This scratch tensor is consumed before another call can reuse it on the stream.
    selected=response._update_workspace.get(r)
    if selected is None:
        selected=torch.empty((n,r),device=delta.device,dtype=delta.dtype)
        response._update_workspace[r]=selected
    forcing=torch.empty(r,device=delta.device,dtype=delta.dtype)
    product=torch.empty((r,r),device=delta.device,dtype=delta.dtype)
    small=torch.empty_like(product);base=torch.empty_like(forcing)
    _launch('rows',delta,((n*r+127)//128,),(128,),
        (response._green,response._inverse,response._u0,response._accepted_delta,delta,
         response._accepted_pec,pec,changed,v,selected,forcing,n,r,1))
    _launch('contractions',delta,(r*r,),(128,),
        (v,selected,response._accepted_current,product,small,base,n,r,1))
    rhs=base+product@forcing
    return v,selected,product,small,forcing,rhs


def finish_update(response, changed, selected, forcing, correction):
    current=torch.empty_like(response._u0);observed=torch.empty_like(response._y0)
    n,q,r=current.numel(),observed.numel(),changed.numel()
    _launch('finish',current,((max(n,q)+127)//128,),(128,),
        (selected,response._projected_inverse,response._accepted_current,response._accepted_projection,
         forcing,correction,changed,current,observed,n,q,r,1))
    return current,observed


def batch_update(response, delta, pec, changed):
    """Evaluate equally padded-rank proposals against one accepted generation."""
    count,n=delta.shape;r=changed.shape[1];q=response._y0.numel()
    def empty(shape):return torch.empty(shape,device=delta.device,dtype=delta.dtype)
    v=empty((count,r,n));selected=empty((count,n,r));forcing=empty((count,r))
    product=empty((count,r,r));small=empty((count,r,r));base=empty((count,r))
    _launch('rows',delta,((count*n*r+127)//128,),(128,),
        (response._green,response._inverse,response._u0,response._accepted_delta,delta,
         response._accepted_pec,pec,changed,v,selected,forcing,n,r,count))
    _launch('contractions',delta,(count*r*r,),(128,),
        (v,selected,response._accepted_current,product,small,base,n,r,count))
    rhs=base+torch.bmm(product,forcing[:,:,None])[:,:,0]
    correction=empty((count,r));inverse=empty((count,r,r))
    status=torch.empty((count,2),device=delta.device,dtype=torch.float64)
    capacity=next(k for k in (4,8,16,32) if k>=r)
    with cp.cuda.Device(delta.device.index),cp.cuda.Stream.from_external(torch.cuda.current_stream(delta.device)):
        _kernel(capacity,delta.dtype==torch.complex128)((count,),(1,),
            (np.uint64(small.data_ptr()),np.uint64(product.data_ptr()),np.uint64(rhs.data_ptr()),np.uint64(correction.data_ptr()),
             np.uint64(inverse.data_ptr()),np.uint64(status.data_ptr()),r,torch.finfo(delta.real.dtype).eps,response.config.tolerance))
    statuses=status.cpu().tolist()
    for i,(safe,residual) in enumerate(statuses):
        if not safe:
            from .update_solve import solve_update
            corrected,residual,inv=solve_update(small[i],product[i],rhs[i],response.config.tolerance)
            correction[i]=corrected;inverse[i]=inv;statuses[i][1]=residual
    current=empty((count,n));observed=empty((count,q))
    _launch('finish',delta,((count*max(n,q)+127)//128,),(128,),
        (selected,response._projected_inverse,response._accepted_current,response._accepted_projection,
         forcing,correction,changed,current,observed,n,q,r,count))
    from .response import LocalResponseTrial
    return tuple(LocalResponseTrial(response,response._generation,delta[i].clone(),pec[i].clone(),changed[i].clone(),
        v[i].clone(),small[i].clone(),current[i].clone(),observed[i].clone(),statuses[i][1],inverse[i].clone()) for i in range(count))
