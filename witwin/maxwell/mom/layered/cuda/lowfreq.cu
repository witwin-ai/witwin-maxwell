#include "common.cuh"
#include <vector>
using namespace layered;
using at::Tensor;

// Setup is integer-only Kruskal in ascending edge order, then BFS with
// ascending face roots and fixed local-slot traversal. O(N+F) storage/traffic
// (union-find root walks aside), no atomics and no host graph traversal.
__device__ int root(int64_t* sets,int v) {
  while(sets[v]!=v) {sets[v]=sets[sets[v]];v=sets[v];}
  return v;
}
__global__ void forest_kernel(const int64_t* plus,const int64_t* minus,
    const int64_t* edges,const int64_t* signs,int64_t* order,int64_t* parents,
    int64_t* counts,int64_t* work,int N,int F,int W) {
  if(threadIdx.x || blockIdx.x) return;
  int64_t *sets=work,*queue=work+F;
  for(int f=0;f<F;++f) {sets[f]=f;parents[3*f]=-1;}
  int trees=0;
  for(int e=0;e<N;++e) {
    int a=root(sets,plus[e]),b=root(sets,minus[e]);
    order[e]=-1;
    if(a!=b) {sets[max(a,b)]=min(a,b);order[e]=trees++;}
  }
  int loops=N-trees,cycle=0;
  for(int e=0;e<N;++e) order[e]=order[e]<0 ? cycle++ : loops+order[e];
  int components=0;
  for(int f=0;f<F;++f) {
    if(parents[3*f]>=0) continue;
    ++components;
    int head=0,tail=1;queue[0]=f;parents[3*f]=f;parents[3*f+1]=-1;parents[3*f+2]=0;
    while(head<tail) {
      int a=queue[head++];
      for(int slot=0;slot<W;++slot) {
        if(!signs[W*a+slot]) continue;
        int e=edges[W*a+slot];
        if(order[e]<loops) continue;
        int b=plus[e]==a ? minus[e] : plus[e];
        if(parents[3*b]>=0) continue;
        parents[3*b]=a;parents[3*b+1]=e;parents[3*b+2]=parents[3*a+2]+1;
        queue[tail++]=b;
      }
      // The optional final face is the PEC reservoir, with arbitrary valence.
      // Its charge is supplied by the image, so it has no physical face slots.
      bool reservoir=true;for(int slot=0;slot<W;++slot) reservoir &= !signs[W*a+slot];
      if(reservoir) {
        for(int e=0;e<N;++e) {
          if(order[e]<loops || (plus[e]!=a && minus[e]!=a)) continue;
          int b=plus[e]==a ? minus[e]:plus[e];
          if(parents[3*b]>=0) continue;
          parents[3*b]=a;parents[3*b+1]=e;parents[3*b+2]=parents[3*a+2]+1;
          queue[tail++]=b;
        }
      }
    }
  }
  counts[0]=components;counts[1]=loops;
}

// One thread/fundamental cycle. Every chord has its own identity pivot;
// tree columns are singleton pivots, proving [Lambda T] invertible, including
// holes and disconnected components. Traffic O(N*tree_depth), zero float atomics.
__global__ void cycles_kernel(const int64_t* plus,const int64_t* minus,const double* length,
    const int64_t* order,const int64_t* parent,double* transform,int N,int loops) {
  int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=N) return;
  int col=order[e];transform[int64_t(e)*N+col]=1/length[e];
  if(col>=loops) return;
  int a=plus[e],b=minus[e];
  while(a!=b) {
    if(parent[3*a+2]>=parent[3*b+2]) {
      int t=parent[3*a+1];
      transform[int64_t(t)*N+col]=(plus[t]==a ? -1.0 : 1.0)/length[t];
      a=parent[3*a];
    } else {
      int t=parent[3*b+1];
      transform[int64_t(t)*N+col]=(plus[t]==b ? 1.0 : -1.0)/length[t];
      b=parent[3*b];
    }
  }
}

// Three-entry fixed-order sparse divergence gather: ~64 B and 6 real FLOP/entry.
__global__ void divergence_kernel(const double* q,const int64_t* edges,const int64_t* signs,
    const double* lengths,double* out,int N,int F,int K,int W) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(F)*K) return;
  int f=i/K,k=i%K;double sum=0;
  for(int slot=0;slot<W;++slot) {
    int e=edges[W*f+slot];
    sum+=double(signs[W*f+slot])*lengths[e]*q[int64_t(e)*K+k];
  }
  out[i]=sum;
}

__global__ void unscale_kernel(const C* solution,const double* recovery,C* out,int64_t size,int P) {
  int64_t t=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(t<size) out[t]=solution[t]*recovery[t/P];
}

std::vector<Tensor> layered_forest(Tensor plus,Tensor minus,Tensor edges,Tensor signs) {
  c10::cuda::CUDAGuard guard(plus.device());int F=edges.size(0),N=plus.numel();
  auto order=at::empty_like(plus),parents=at::empty({F,3},plus.options());
  auto counts=at::empty({2},plus.options()),work=at::empty({2*F},plus.options());
  forest_kernel<<<1,1,0,c10::cuda::getCurrentCUDAStream()>>>(plus.data_ptr<int64_t>(),minus.data_ptr<int64_t>(),
    edges.data_ptr<int64_t>(),signs.data_ptr<int64_t>(),order.data_ptr<int64_t>(),parents.data_ptr<int64_t>(),
    counts.data_ptr<int64_t>(),work.data_ptr<int64_t>(),N,F,edges.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {order,parents,counts};
}
Tensor layered_cycles(Tensor plus,Tensor minus,Tensor lengths,Tensor order,Tensor parents,int64_t loops) {
  c10::cuda::CUDAGuard guard(plus.device());int N=plus.numel();
  auto out=at::zeros({N,N},lengths.options());
  cycles_kernel<<<(N+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(plus.data_ptr<int64_t>(),
    minus.data_ptr<int64_t>(),lengths.data_ptr<double>(),order.data_ptr<int64_t>(),
    parents.data_ptr<int64_t>(),out.data_ptr<double>(),N,loops);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
Tensor layered_divergence(Tensor q,Tensor edges,Tensor signs,Tensor lengths) {
  c10::cuda::CUDAGuard guard(q.device());int F=edges.size(0),K=q.size(1);
  auto out=at::empty({F,K},q.options());
  divergence_kernel<<<(int64_t(F)*K+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
    q.data_ptr<double>(),edges.data_ptr<int64_t>(),signs.data_ptr<int64_t>(),lengths.data_ptr<double>(),
    out.data_ptr<double>(),q.size(0),F,K,edges.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
Tensor layered_unscale(Tensor solution,Tensor recovery) {
  c10::cuda::CUDAGuard guard(solution.device());auto out=at::empty_like(solution);
  unscale_kernel<<<(solution.numel()+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
    solution.data_ptr<C>(),recovery.data_ptr<double>(),out.data_ptr<C>(),solution.numel(),solution.size(-1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}


// One owner per port entry, O(K*P^2) bytes and one complex scale.
__global__ void admittance_kernel(const C* y,const double* frequencies,C* out,int K,int P) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=K*P*P) return;
  out[i]=y[i]/(2*pi*frequencies[i/(P*P)]);
}
Tensor layered_port_admittance(Tensor y,Tensor frequencies) {
  c10::cuda::CUDAGuard guard(y.device());auto out=at::empty_like(y);
  admittance_kernel<<<(y.numel()+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
    y.data_ptr<C>(),frequencies.data_ptr<double>(),out.data_ptr<C>(),y.size(0),y.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// Sparse column transforms: O(K*N*nnz) work, rather than dense N^3.
// One owner/output, ascending source indices, no floating point atomics.
// Logical traffic: 16*(terms+1) B/output plus cached 16 B sparse metadata;
// 4 real FLOP/term (a real transform times a complex entry).
__global__ void column_counts_kernel(const double* a,int64_t* counts,int M,int N) {
  int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=N) return;
  int64_t count=0;for(int i=0;i<M;++i) count+=a[int64_t(i)*N+j]!=0;
  counts[j+1]=count;
}
__global__ void column_pack_kernel(const double* a,const int64_t* offsets,
    int64_t* indices,double* values,int M,int N) {
  int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=N) return;
  int64_t at=offsets[j];
  for(int i=0;i<M;++i) {
    double v=a[int64_t(i)*N+j];
    if(v!=0) {indices[at]=i;values[at++]=v;}
  }
}
std::vector<Tensor> layered_sparse_columns(Tensor a) {
  c10::cuda::CUDAGuard guard(a.device());int M=a.size(0),N=a.size(1);
  auto counts=at::zeros({N+1},a.options().dtype(at::kLong));
  auto stream=c10::cuda::getCurrentCUDAStream();
  column_counts_kernel<<<(N+127)/128,128,0,stream>>>(a.data_ptr<double>(),counts.data_ptr<int64_t>(),M,N);
  auto offsets=at::cumsum(counts,0);
  // Setup-only shape transfer; no graph or numerical data leaves the device.
  int64_t nnz=offsets.select(0,N).item<int64_t>();
  auto indices=at::empty({nnz},offsets.options()),values=at::empty({nnz},a.options());
  column_pack_kernel<<<(N+127)/128,128,0,stream>>>(a.data_ptr<double>(),offsets.data_ptr<int64_t>(),
      indices.data_ptr<int64_t>(),values.data_ptr<double>(),M,N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {offsets,indices,values};
}
__global__ void sparse_product_kernel(const C* a,const int64_t* offsets,const int64_t* indices,
    const double* values,C* out,int K,int M,int N,int S,bool left) {
  int64_t t=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  int rows=left?S:M,cols=left?N:S;
  if(t>=int64_t(K)*rows*cols) return;
  int j=t%cols,i=(t/cols)%rows,f=t/(int64_t(rows)*cols),s=left?i:j;
  C sum(0);
  for(int64_t k=offsets[s];k<offsets[s+1];++k) {
    int r=indices[k];
    sum+=a[(int64_t(f)*M+(left?r:i))*N+(left?j:r)]*values[k];
  }
  out[t]=sum;
}
Tensor layered_sparse_product(Tensor a,std::vector<Tensor> columns,bool left) {
  c10::cuda::CUDAGuard guard(a.device());
  int K=a.size(0),M=a.size(1),N=a.size(2),S=columns[0].numel()-1;
  auto out=at::empty({K,left?S:M,left?N:S},a.options());
  sparse_product_kernel<<<(out.numel()+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<C>(),columns[0].data_ptr<int64_t>(),columns[1].data_ptr<int64_t>(),
      columns[2].data_ptr<double>(),out.data_ptr<C>(),K,M,N,S,left);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// Chunk-batched vendor LU solves always use the same two-dimensional vendor
// entry point per frequency, so the memory planner's chunk size cannot
// select a batch-size-dependent algorithm and change results.
Tensor layered_lu_solve(Tensor lu,Tensor pivots,Tensor rhs) {
  c10::cuda::CUDAGuard guard(lu.device());
  auto out=at::empty_like(rhs);
  for(int64_t f=0;f<lu.size(0);++f)
    out.select(0,f).copy_(at::linalg_lu_solve(lu.select(0,f),pivots.select(0,f),
                                           rhs.select(0,f),true,false));
  return out;
}

// One owner per frequency, fixed port order. Latch the FIRST passing step,
// including zero (the initial solve); converged solutions are then frozen.
// ~32*P B/frequency, O(P) FLOP; no atomics or host convergence polling.
__global__ void refinement_check_kernel(const double* residual,const double* matrix,
    const double* solution,const double* rhs,const int* info,int64_t* steps,
    double* error,bool* active,int K,int P,int step,double tolerance) {
  int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=K) return;
  double maximum=0;
  for(int p=0;p<P;++p) {
    double value=residual[f*P+p]/(matrix[f]*solution[f*P+p]+rhs[f*P+p]);
    maximum=(!isfinite(value) || !isfinite(maximum)) ? INFINITY : fmax(maximum,value);
  }
  if(info[f]) maximum=INFINITY;
  if(steps[f]<0 && maximum<=tolerance) steps[f]=step;
  active[f]=steps[f]<0;
  error[f]=maximum;
}
std::vector<Tensor> layered_refinement_check(Tensor residual,Tensor matrix,Tensor solution,
    Tensor rhs,Tensor info,Tensor steps,int64_t step,double tolerance) {
  c10::cuda::CUDAGuard guard(residual.device());int K=residual.size(0),P=residual.size(1);
  auto error=at::empty_like(matrix),active=at::empty({K},matrix.options().dtype(at::kBool));
  refinement_check_kernel<<<(K+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
    residual.data_ptr<double>(),matrix.data_ptr<double>(),solution.data_ptr<double>(),rhs.data_ptr<double>(),
    info.data_ptr<int>(),steps.data_ptr<int64_t>(),error.data_ptr<double>(),active.data_ptr<bool>(),
    K,P,step,tolerance);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {error,active};
}
// ~48 B and one complex add/entry. Each entry has one owner; already-converged
// frequencies retain the identical solution irrespective of chunk composition.
__global__ void refinement_update_kernel(C* x,const C* correction,const bool* active,
    int64_t entries,int stride) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<entries && active[i/stride]) x[i]+=correction[i];
}
void layered_refinement_update(Tensor x,Tensor correction,Tensor active) {
  c10::cuda::CUDAGuard guard(x.device());
  refinement_update_kernel<<<(x.numel()+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
    x.data_ptr<C>(),correction.data_ptr<C>(),active.data_ptr<bool>(),x.numel(),x.size(1)*x.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Batch-size-independent vendor products. Dispatch each frequency through the
// same 2D GEMM entry point; a singleton leading dimension broadcasts a shared
// geometry transform. GPU work remains in vendor kernels, with no host reads.
Tensor layered_frequency_mm(Tensor a,Tensor b) {
  c10::cuda::CUDAGuard guard(a.device());
  int64_t K=std::max(a.size(0),b.size(0));
  auto out=at::empty({K,a.size(1),b.size(2)},a.options());
  for(int64_t f=0;f<K;++f)
    out.select(0,f).copy_(at::mm(a.select(0,a.size(0)==1 ? 0 : f),
                              b.select(0,b.size(0)==1 ? 0 : f)));
  return out;
}
Tensor layered_frequency_residual(Tensor a,Tensor x,Tensor rhs) {
  c10::cuda::CUDAGuard guard(a.device());auto out=at::empty_like(rhs);
  for(int64_t f=0;f<a.size(0);++f)
    out.select(0,f).copy_(at::addmm(rhs.select(0,f),a.select(0,f),x.select(0,f),1,-1));
  return out;
}
// Norm reductions also retain identical per-frequency dimensions, including
// the first-passing-step diagnostic. O(K*N^2) read traffic for matrix norms.
Tensor layered_frequency_norm(Tensor a,bool matrix) {
  c10::cuda::CUDAGuard guard(a.device());int64_t K=a.size(0);
  auto out=matrix ? at::empty({K},a.options().dtype(at::kDouble)) :
                   at::empty({K,a.size(2)},a.options().dtype(at::kDouble));
  for(int64_t f=0;f<K;++f) {
    auto value=a.select(0,f);
    if(matrix) value=value.reshape({-1});
    out.select(0,f).copy_(at::linalg_vector_norm(value,2,at::IntArrayRef{0},false));
  }
  return out;
}


template<typename T> __device__ T peak_fma(T a,T b,T c);
template<> __device__ float peak_fma(float a,float b,float c) {return __fmaf_rn(a,b,c);}
template<> __device__ double peak_fma(double a,double b,double c) {return __fma_rn(a,b,c);}
// Timing-only device peak: eight independent FMA chains, 16 FLOP/iteration,
// eight stores/thread. Explicit intrinsics retain FMA under --fmad=false.
template<typename T> __global__ void fma_peak_kernel(T* out,int threads,int iterations) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=threads) return;
  T a=T(1.0000001),b=T(.0000001),x0=T(t+1),x1=x0+1,x2=x0+2,x3=x0+3;
  T x4=x0+4,x5=x0+5,x6=x0+6,x7=x0+7;
  #pragma unroll 1
  for(int k=0;k<iterations;++k) {
    x0=peak_fma(a,x0,b);x1=peak_fma(a,x1,b);x2=peak_fma(a,x2,b);x3=peak_fma(a,x3,b);
    x4=peak_fma(a,x4,b);x5=peak_fma(a,x5,b);x6=peak_fma(a,x6,b);x7=peak_fma(a,x7,b);
  }
  out[8*t]=x0;out[8*t+1]=x1;out[8*t+2]=x2;out[8*t+3]=x3;
  out[8*t+4]=x4;out[8*t+5]=x5;out[8*t+6]=x6;out[8*t+7]=x7;
}
Tensor layered_fma_peak(Tensor anchor,int64_t threads,int64_t iterations) {
  c10::cuda::CUDAGuard guard(anchor.device());auto out=at::empty({threads,8},anchor.options());
  auto stream=c10::cuda::getCurrentCUDAStream();
  if(anchor.scalar_type()==at::kDouble)
    fma_peak_kernel<<<(threads+255)/256,256,0,stream>>>(out.data_ptr<double>(),threads,iterations);
  else fma_peak_kernel<<<(threads+255)/256,256,0,stream>>>(out.data_ptr<float>(),threads,iterations);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}


// ---------------------------------------------------------------------------
// In-place dense chunk system (PH1). One complex128 [K,N,N] buffer holds the
// magnetic potential in loop/tree basis positions, receives the folded loss,
// is congruence-transformed in place and finally becomes the equilibrated A.
// A complex64 workspace holds packed P during assembly and the LU afterwards.

// Fold local Gram loss into the magnetic entries before the congruence:
// V'=V+R/(-i*mu0*w), hence -i*mu0*Q^T V' Q = -i*mu0*Q^T V Q + Q^T R Q/w.
// One thread per (frequency,row); each distinct column sums its candidates
// in loss_matrix's (side,slot) order. O(W^2) index work and <=2W complex
// read-modify-writes per row: O(K*N) traffic, one owner per entry.
__global__ void loss_fold_kernel(const double* blocks,const int64_t* columns,const int64_t* supports,
    const C* impedance,const int64_t* order,const double* frequencies,C* system,int N,int F,int K,int W) {
  int64_t t=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(t>=int64_t(K)*N) return;
  int m=t%N,f=t/N;
  constexpr double mu0=1/(eps0*c0*c0);
  C scale(0,1/(mu0*2*pi*frequencies[f]));
  for(int a=0;a<2*W;++a) {
    if(supports[2*m+a/W]<0) continue;
    int n=columns[2*W*m+a];
    bool first=true;
    for(int b=0;b<a;++b) first&=!(supports[2*m+b/W]>=0 && columns[2*W*m+b]==n);
    if(!first) continue;
    C value(0);
    for(int b=a;b<2*W;++b)
      if(supports[2*m+b/W]>=0 && columns[2*W*m+b]==n)
        value+=blocks[2*W*m+b]*impedance[int64_t(f)*F+supports[2*m+b/W]];
    system[(int64_t(f)*N+order[m])*N+order[n]]+=value*scale;
  }
}

// In-place congruence A <- Q^T A Q for Q=[Lambda T] (loop columns first).
// Each loop column holds its own chord row (order[chord]=column) plus tree
// rows; tree columns are unit singletons t/|t|=1 and need no pass. The row
// pass rewrites loop rows from their own entry and tree rows only, which it
// never writes; the column pass mirrors it. Sums keep sparse_product's
// ascending source order, so the result equals the out-of-place product
// bitwise. Row pass threads run along a row (coalesced); column pass threads
// run along the loop columns of one row. Traffic/loop entry 16*(nnz+1) B plus
// 16*nnz B cached metadata, 4 real FLOP/term; no atomics.
__global__ void congruence_rows_kernel(C* a,const int64_t* offsets,const int64_t* indices,
    const double* values,const int64_t* order,int K,int N,int L) {
  int64_t t=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(t>=int64_t(K)*L*N) return;
  int j=t%N,s=(t/N)%L;C* m=a+(t/(int64_t(L)*N))*N*N;
  C sum(0);
  for(int64_t k=offsets[s];k<offsets[s+1];++k) sum+=m[order[indices[k]]*int64_t(N)+j]*values[k];
  m[int64_t(s)*N+j]=sum;
}
__global__ void congruence_columns_kernel(C* a,const int64_t* offsets,const int64_t* indices,
    const double* values,const int64_t* order,int K,int N,int L) {
  int64_t t=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(t>=int64_t(K)*L*N) return;
  int s=t%L,i=(t/L)%N;C* row=a+(t/(int64_t(L)*N))*N*N+int64_t(i)*N;
  C sum(0);
  for(int64_t k=offsets[s];k<offsets[s+1];++k) sum+=row[order[indices[k]]]*values[k];
  row[s]=sum;
}
void layered_congruence(Tensor a,std::vector<Tensor> q,Tensor order,int64_t loops) {
  c10::cuda::CUDAGuard guard(a.device());int K=a.size(0),N=a.size(1);
  int64_t entries=int64_t(K)*loops*N;if(!entries) return;
  auto stream=c10::cuda::getCurrentCUDAStream();
  congruence_rows_kernel<<<(entries+127)/128,128,0,stream>>>(a.data_ptr<C>(),q[0].data_ptr<int64_t>(),
    q[1].data_ptr<int64_t>(),q[2].data_ptr<double>(),order.data_ptr<int64_t>(),K,N,loops);
  congruence_columns_kernel<<<(entries+127)/128,128,0,stream>>>(a.data_ptr<C>(),q[0].data_ptr<int64_t>(),
    q[1].data_ptr<int64_t>(),q[2].data_ptr<double>(),order.data_ptr<int64_t>(),K,N,loops);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void layered_loss_fold(Tensor system,Tensor blocks,Tensor columns,Tensor supports,Tensor impedance,
    Tensor order,Tensor frequencies) {
  c10::cuda::CUDAGuard guard(system.device());int K=system.size(0),N=system.size(1);
  loss_fold_kernel<<<(int64_t(K)*N+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
    blocks.data_ptr<double>(),columns.data_ptr<int64_t>(),supports.data_ptr<int64_t>(),
    impedance.data_ptr<C>(),order.data_ptr<int64_t>(),frequencies.data_ptr<double>(),system.data_ptr<C>(),
    N,impedance.size(1),K,blocks.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Scalar-potential block from packed P on the fly: charge(s,t) is the
// sparse_product composition (C^T P C)[s,t] with the same inner/outer order.
__device__ inline C packed_charge(const C* p,const int64_t* offsets,const int64_t* indices,
    const double* values,int s,int t) {
  C sum(0);
  for(int64_t b=offsets[t];b<offsets[t+1];++b) {
    int64_t g=indices[b];C inner(0);
    for(int64_t a=offsets[s];a<offsets[s+1];++a) {
      int64_t f=indices[a];
      inner+=p[f>=g ? f*(f+1)/2+g : g*(g+1)/2+f]*values[a];
    }
    sum+=inner*values[b];
  }
  return sum;
}
struct ChargeView {const C* p;const int64_t *offsets,*indices;const double* values;int64_t stride;};
__device__ inline C system_entry(const C* a,ChargeView c,int f,int i,int j,int N,int L,double w,double w0) {
  double si=i<L ? 1 : w/w0,sj=j<L ? 1 : w/w0;
  constexpr double mu0=1/(eps0*c0*c0);
  C value=C(0,-mu0)*a[(int64_t(f)*N+i)*N+j]*(si*sj);
  // Loop scalar blocks are structural zeros (scaled_kernel's convention).
  if(i>=L && j>=L) value+=C(0,1/(eps0*w0*w0))*packed_charge(c.p+f*c.stride,c.offsets,c.indices,c.values,i-L,j-L);
  return value;
}
// Diagonal pass: congruence scale, recovery and drive. ~(48+32P) B/row.
__global__ void system_diagonal_kernel(const C* a,ChargeView c,const C* rhs,const double* frequencies,
    double* scale,double* recovery,C* drive,int N,int L,int P,int K,double w0) {
  int64_t t=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(t>=int64_t(K)*N) return;
  int i=t%N,f=t/N;double w=2*pi*frequencies[f];
  double d=1/sqrt(abs(system_entry(a,c,f,i,i,N,L,w,w0)));
  scale[t]=d;recovery[t]=d*(i<L ? 1/w : 1/w0);
  for(int p=0;p<P;++p) drive[t*P+p]=rhs[i*P+p]*(d*(i<L ? 1 : w/w0));
}
// Entry pass, in place: one owner per entry reads only itself plus the
// precomputed scales. ~48 B/entry (+64 B and 4 P reads per tree-tree entry),
// ~40 real FLOP; no reductions or atomics.
__global__ void system_entries_kernel(C* a,ChargeView c,const double* scale,const double* frequencies,
    int N,int L,int K,double w0) {
  int64_t t=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(t>=int64_t(K)*N*N) return;
  int j=t%N,i=(t/N)%N,f=t/(int64_t(N)*N);double w=2*pi*frequencies[f];
  a[t]=system_entry(a,c,f,i,j,N,L,w,w0)*(scale[int64_t(f)*N+i]*scale[int64_t(f)*N+j]);
}
std::vector<Tensor> layered_scale_system(Tensor a,Tensor workspace,std::vector<Tensor> charge,Tensor rhs,
    Tensor frequencies,int64_t loops,double omega0) {
  c10::cuda::CUDAGuard guard(a.device());int K=a.size(0),N=a.size(1),P=rhs.size(1);
  ChargeView c{reinterpret_cast<const C*>(workspace.data_ptr<c10::complex<float>>()),
    charge[0].data_ptr<int64_t>(),charge[1].data_ptr<int64_t>(),charge[2].data_ptr<double>(),workspace.size(1)/2};
  auto scale=at::empty({K,N},frequencies.options()),recovery=at::empty({K,N},frequencies.options());
  auto drive=at::empty({K,N,P},rhs.options());
  auto stream=c10::cuda::getCurrentCUDAStream();
  system_diagonal_kernel<<<(int64_t(K)*N+127)/128,128,0,stream>>>(a.data_ptr<C>(),c,rhs.data_ptr<C>(),
    frequencies.data_ptr<double>(),scale.data_ptr<double>(),recovery.data_ptr<double>(),drive.data_ptr<C>(),
    N,loops,P,K,omega0);
  system_entries_kernel<<<(int64_t(K)*N*N+127)/128,128,0,stream>>>(a.data_ptr<C>(),c,scale.data_ptr<double>(),
    frequencies.data_ptr<double>(),N,loops,K,omega0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {drive,recovery};
}

// complex128 row-major A -> column-major LU storage (complex64 cast, or the
// complex128 escalation copy). 32x32 shared tiles keep reads and writes
// coalesced: 24 B/entry (32 B for complex128), no FLOP.
template<typename T> __global__ void factor_cast_kernel(const C* a,T* out,int N,int64_t stride) {
  using R=typename T::value_type;
  __shared__ R tile[32][33][2];
  int f=blockIdx.z,column=blockIdx.x*32+threadIdx.x;
  const C* m=a+int64_t(f)*N*N;
  for(int k=threadIdx.y;k<32;k+=8) {
    int row=blockIdx.y*32+k;
    if(row<N && column<N) {
      T value(m[int64_t(row)*N+column]);
      tile[k][threadIdx.x][0]=value.real();tile[k][threadIdx.x][1]=value.imag();
    }
  }
  __syncthreads();
  int row=blockIdx.y*32+threadIdx.x;
  for(int k=threadIdx.y;k<32;k+=8) {
    int col=blockIdx.x*32+k;
    if(row<N && col<N) out[f*stride+int64_t(col)*N+row]=T(tile[threadIdx.x][k][0],tile[threadIdx.x][k][1]);
  }
}
// Cast into the workspace, then factor each frequency in place through the
// two-dimensional vendor LU entry point (LU aliases A: no copy), one call per
// frequency so the chunk size cannot select a batched algorithm.
std::vector<Tensor> layered_factor_system(Tensor a,Tensor lu) {
  c10::cuda::CUDAGuard guard(a.device());int64_t K=a.size(0),N=a.size(1);
  dim3 blocks((N+31)/32,(N+31)/32,K),threads(32,8);
  auto stream=c10::cuda::getCurrentCUDAStream();
  if(lu.scalar_type()==at::kComplexFloat)
    factor_cast_kernel<<<blocks,threads,0,stream>>>(a.data_ptr<C>(),lu.data_ptr<c10::complex<float>>(),N,lu.stride(0));
  else factor_cast_kernel<<<blocks,threads,0,stream>>>(a.data_ptr<C>(),lu.data_ptr<C>(),N,lu.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto pivots=at::empty({K,N},a.options().dtype(at::kInt));
  auto info=at::empty({K},a.options().dtype(at::kInt));
  for(int64_t f=0;f<K;++f) {
    auto factor=lu.select(0,f),pivot=pivots.select(0,f),status=info.select(0,f);
    at::linalg_lu_factor_ex_out(factor,pivot,status,factor,true,false);
  }
  return {pivots,info};
}

// Sparse delta-gap cycle work: the port_work circulation over one sparse
// column, in its ascending edge order. O(nnz*P) setup traffic.
__global__ void sparse_port_work_kernel(const int64_t* offsets,const int64_t* indices,const double* values,
    const double* lengths,const C* excitation,C* work,int N,int P) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N*P) return;
  int col=i/P,p=i%P;double value=0;
  for(int64_t k=offsets[col];k<offsets[col+1];++k) {
    int64_t e=indices[k];
    value+=(values[k]>0 ? 1.0 : -1.0)*(excitation[e*P+p].real()/lengths[e]);
  }
  work[i]=C(value);
}
Tensor layered_sparse_port_work(std::vector<Tensor> q,Tensor lengths,Tensor excitation) {
  c10::cuda::CUDAGuard guard(lengths.device());int N=q[0].numel()-1,P=excitation.size(1);
  auto work=at::empty({N,P},excitation.options());
  sparse_port_work_kernel<<<(N*P+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
    q[0].data_ptr<int64_t>(),q[1].data_ptr<int64_t>(),q[2].data_ptr<double>(),lengths.data_ptr<double>(),
    excitation.data_ptr<C>(),work.data_ptr<C>(),N,P);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return work;
}

// Tree-column face charge D*q for singleton tree columns: the divergence
// kernel's slot sum with the only nonzero term, in ascending face order.
__global__ void tree_charge_kernel(const int64_t* offsets,const int64_t* indices,const double* values,
    const int64_t* supports,const int64_t* edges,const int64_t* signs,const double* lengths,
    int64_t* out_offsets,int64_t* out_indices,double* out_values,int L,int T,int W) {
  int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=T) return;
  int64_t source=offsets[L+k],e=indices[source];
  int a=supports[2*e],b=supports[2*e+1];
  int faces[2]={b>=0 && b<a ? b : a,b>=0 && b<a ? a : b};
  int64_t at=2*int64_t(k);int count=0;
  for(int side=0;side<2;++side) {
    int face=faces[side];if(face<0) continue;
    for(int slot=0;slot<W;++slot) if(edges[W*face+slot]==e && signs[W*face+slot]) {
      out_indices[at+count]=face;
      out_values[at+count++]=double(signs[W*face+slot])*lengths[e]*values[source];
    }
  }
  out_offsets[k+1]=count;
}
std::vector<Tensor> layered_tree_charge(std::vector<Tensor> q,Tensor supports,Tensor edges,Tensor signs,
    Tensor lengths,int64_t loops) {
  c10::cuda::CUDAGuard guard(lengths.device());int N=q[0].numel()-1,T=N-loops;
  auto counts=at::zeros({T+1},q[0].options());
  auto indices=at::empty({2*int64_t(T)},q[0].options()),values=at::empty({2*int64_t(T)},q[2].options());
  tree_charge_kernel<<<(T+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
    q[0].data_ptr<int64_t>(),q[1].data_ptr<int64_t>(),q[2].data_ptr<double>(),supports.data_ptr<int64_t>(),
    edges.data_ptr<int64_t>(),signs.data_ptr<int64_t>(),lengths.data_ptr<double>(),counts.data_ptr<int64_t>(),
    indices.data_ptr<int64_t>(),values.data_ptr<double>(),loops,T,edges.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Compact the two padded slots per column; setup-only integer work.
  auto keep=at::arange(2*int64_t(T),q[0].options()).remainder(2).lt(counts.narrow(0,1,T).repeat_interleave(2));
  return {at::cumsum(counts,0),indices.masked_select(keep),values.masked_select(keep)};
}
