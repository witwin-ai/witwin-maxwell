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
    int64_t* counts,int64_t* work,int N,int F) {
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
      for(int slot=0;slot<3;++slot) {
        if(!signs[3*a+slot]) continue;
        int e=edges[3*a+slot];
        if(order[e]<loops) continue;
        int b=plus[e]==a ? minus[e] : plus[e];
        if(parents[3*b]>=0) continue;
        parents[3*b]=a;parents[3*b+1]=e;parents[3*b+2]=parents[3*a+2]+1;
        queue[tail++]=b;
      }
      // The optional final face is the PEC reservoir, with arbitrary valence.
      // Its charge is supplied by the image, so it has no physical face slots.
      if(!signs[3*a] && !signs[3*a+1] && !signs[3*a+2]) {
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
    const double* lengths,double* out,int N,int F,int K) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(F)*K) return;
  int f=i/K,k=i%K;double sum=0;
  for(int slot=0;slot<3;++slot) {
    int e=edges[3*f+slot];
    sum+=double(signs[3*f+slot])*lengths[e]*q[int64_t(e)*K+k];
  }
  out[i]=sum;
}

// Materialize local Gram loss with one owner per matrix entry. Six candidates
// per row, fixed side/slot order; ~128 B, at most 6 complex mul/add per entry.
__global__ void loss_kernel(const double* blocks,const int64_t* columns,const int64_t* supports,
    const C* impedance,C* out,int N,int F,int K) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(K)*N*N) return;
  int n=i%N,m=(i/N)%N,f=i/(int64_t(N)*N);C value(0);
  for(int side=0;side<2;++side) for(int slot=0;slot<3;++slot) {
    int j=6*m+3*side+slot;
    if(supports[2*m+side]>=0 && columns[j]==n) value+=blocks[j]*impedance[int64_t(f)*F+supports[2*m+side]];
  }
  out[i]=value;
}

__device__ C entry(const C* magnetic,const C* charge,const C* loss,
    int f,int i,int j,int N,int L,double w,double w0) {
  double si=i<L ? 1 : w/w0,sj=j<L ? 1 : w/w0;
  int64_t at=(int64_t(f)*N+i)*N+j;
  constexpr double mu0=1/(eps0*c0*c0);
  C value=C(0,-mu0)*magnetic[at]*(si*sj);
  // The loop scalar blocks are structural zeros: never evaluate D*Lambda,
  // never subtract a large scalar contribution from a summed EFIE matrix.
  if(i>=L && j>=L)
    value+=C(0,1/(eps0*w0*w0))*charge[(int64_t(f)*(N-L)+i-L)*(N-L)+j-L];
  if(loss) value+=loss[at]*(si*sj/w);
  return value;
}

// A=S^T Z S/w, S=[Q_loop,(w/w0)Q_tree], followed by diagonal
// congruence equilibration. O(1) PEC limit; resistive LL/w is equilibrated.
// Per entry ~128 B, ~70 real FLOP, no reductions/atomics.
__global__ void scaled_kernel(const C* magnetic,const C* charge,const C* loss,const C* rhs,
    const double* frequencies,C* matrix,C* drive,double* recovery,int N,int L,int P,int K,double w0) {
  int64_t t=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(t<int64_t(K)*N*N) {
    int j=t%N,i=(t/N)%N,f=t/(int64_t(N)*N);double w=2*pi*frequencies[f];
    double di=1/sqrt(abs(entry(magnetic,charge,loss,f,i,i,N,L,w,w0)));
    double dj=1/sqrt(abs(entry(magnetic,charge,loss,f,j,j,N,L,w,w0)));
    matrix[t]=entry(magnetic,charge,loss,f,i,j,N,L,w,w0)*(di*dj);
  }
  if(t<int64_t(K)*N) {
    int i=t%N,f=t/N;double w=2*pi*frequencies[f];
    double d=1/sqrt(abs(entry(magnetic,charge,loss,f,i,i,N,L,w,w0)));
    recovery[t]=d*(i<L ? 1/w : 1/w0);
    for(int p=0;p<P;++p) drive[t*P+p]=rhs[i*P+p]*(d*(i<L ? 1 : w/w0));
  }
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
    counts.data_ptr<int64_t>(),work.data_ptr<int64_t>(),N,F);
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
    out.data_ptr<double>(),q.size(0),F,K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
Tensor layered_loss_matrix(Tensor blocks,Tensor columns,Tensor supports,Tensor impedance) {
  c10::cuda::CUDAGuard guard(blocks.device());int N=blocks.size(0),F=impedance.size(1),K=impedance.size(0);
  auto out=at::empty({K,N,N},impedance.options());
  loss_kernel<<<(int64_t(K)*N*N+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(blocks.data_ptr<double>(),
    columns.data_ptr<int64_t>(),supports.data_ptr<int64_t>(),impedance.data_ptr<C>(),out.data_ptr<C>(),N,F,K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<Tensor> layered_scaled(Tensor magnetic,Tensor charge,Tensor loss,Tensor rhs,Tensor frequencies,
    int64_t loops,double omega0) {
  c10::cuda::CUDAGuard guard(magnetic.device());int N=magnetic.size(1),K=magnetic.size(0),P=rhs.size(1);
  auto matrix=at::empty_like(magnetic),drive=at::empty({K,N,P},rhs.options());
  auto recovery=at::empty({K,N},frequencies.options());
  scaled_kernel<<<(int64_t(K)*N*N+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
    magnetic.data_ptr<C>(),charge.data_ptr<C>(),loss.numel() ? loss.data_ptr<C>() : nullptr,
    rhs.data_ptr<C>(),frequencies.data_ptr<double>(),matrix.data_ptr<C>(),drive.data_ptr<C>(),
    recovery.data_ptr<double>(),N,loops,P,K,omega0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {matrix,drive,recovery};
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

// Delta-gap work in the fundamental basis is an integer circulation: b_e/l_e
// is exactly 0 or +/-1. Evaluate it before normalization so gradient ports have EXACT
// zero loop drive, rather than a rounding-sized drive amplified by 1/omega.
// One thread/(basis column,port), fixed edge order, O(N^2*P) setup traffic.
__global__ void port_work_kernel(const double* transform,const double* lengths,
    const C* excitation,C* work,int N,int P) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N*P) return;
  int col=i/P,p=i%P;double value=0;
  for(int e=0;e<N;++e) {
    double coefficient=transform[int64_t(e)*N+col];
    if(coefficient!=0)
      value+=(coefficient>0 ? 1.0 : -1.0)*(excitation[e*P+p].real()/lengths[e]);
  }
  work[i]=C(value);
}
Tensor layered_port_work(Tensor transform,Tensor lengths,Tensor excitation) {
  c10::cuda::CUDAGuard guard(transform.device());auto work=at::empty_like(excitation);
  int N=transform.size(0),P=excitation.size(1);
  port_work_kernel<<<(N*P+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
    transform.data_ptr<double>(),lengths.data_ptr<double>(),excitation.data_ptr<C>(),work.data_ptr<C>(),N,P);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return work;
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

// Chunk-batched vendor LU calls always use the same two-dimensional vendor
// entry point per frequency. This prevents batch-size-dependent factorization
// algorithms from changing results when the memory planner changes chunk size.
std::vector<Tensor> layered_lu_factor(Tensor a) {
  c10::cuda::CUDAGuard guard(a.device());
  int64_t K=a.size(0),N=a.size(1);
  auto lu=at::empty_strided({K,N,N},{N*N,1,N},a.options());
  auto pivots=at::empty({K,N},a.options().dtype(at::kInt));
  auto info=at::empty({K},a.options().dtype(at::kInt));
  for(int64_t f=0;f<K;++f) {
    auto result=at::linalg_lu_factor_ex(a.select(0,f),true,false);
    lu.select(0,f).copy_(std::get<0>(result));
    pivots.select(0,f).copy_(std::get<1>(result));
    info.select(0,f).copy_(std::get<2>(result));
  }
  return {lu,pivots,info};
}
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
