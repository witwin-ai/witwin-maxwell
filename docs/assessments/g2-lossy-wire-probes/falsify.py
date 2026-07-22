import math, torch
import witwin.maxwell.fdtd.wire_lossy as wl
from witwin.maxwell.compiler.wire_impedance import internal_impedance
MU_0=wl.MU_0
meta={"conductor":{"wire_names":("w",),"kinds":("finite",),"conductivity":(5.8e7,),"permeability":(MU_0,)}}
def build():
    return wl.build_lossy_segment_model(inductance=torch.tensor([1e-8],dtype=torch.float64),radius=torch.tensor([5e-4],dtype=torch.float64),length=torch.tensor([0.02],dtype=torch.float64),segment_wire_ids=torch.tensor([0]),metadata=meta,band=(4e8,3e9),dt=1e-12)
m=build()
# ---- extraction helper (same as test) ----
def extract(model, f, periods=40, spp=400):
    dt=model.dt; w=2*math.pi*f; total=periods*spp; st=model.initial_state()
    comp=model.companion_conductance; co=model.ade_output
    cprev=torch.tensor([math.cos(w*dt*-0.5)],dtype=torch.float64)
    times=[];volt=[];curr=[]
    for n in range(total):
        ch=torch.tensor([math.cos(w*dt*(n+0.5))],dtype=torch.float64); cb=0.5*(ch+cprev)
        hist=(co*st).sum(-1); vl=comp*cb+hist
        st=torch.einsum("skj,sj->sk",model.ade_transition,st)+model.ade_input*cb.unsqueeze(-1)
        times.append(n*dt);volt.append(float(vl));curr.append(float(cb));cprev=ch
    t=torch.tensor(times);v=torch.tensor(volt);c=torch.tensor(curr);tl=slice(total//2,total)
    ph=torch.exp(-1j*w*t[tl]); return (torch.sum(v[tl]*ph)/torch.sum(c[tl]*ph)).real/0.02

# Baseline analytic-AC error
f=1.5e9; base=abs(float(extract(m,f))-float(internal_impedance(5e-4,5.8e7,MU_0,[f]).real))/float(internal_impedance(5e-4,5.8e7,MU_0,[f]).real)
print(f"[baseline] analytic-AC err @1.5GHz = {base:.4f}  (gate <0.08)")

# F1: break the ADE history term (drop Cs x) -> realized impedance wrong
orig_output=m.ade_output
m2=build()
object.__setattr__(m2,'ade_output',torch.zeros_like(m2.ade_output))
errF1=abs(float(extract(m2,f))-float(internal_impedance(5e-4,5.8e7,MU_0,[f]).real))/float(internal_impedance(5e-4,5.8e7,MU_0,[f]).real)
print(f"[F1 zero ADE history] analytic-AC err = {errF1:.4f}  -> {'RED' if errF1>=0.08 else 'green'}")

# F2: DC exactness -- perturb resistance_dc by 1%
import witwin.maxwell.compiler.wire_impedance as wi
r_ok=float(m.resistance_dc[0]); expected=float(wi.dc_resistance(5e-4,5.8e7))*0.02
print(f"[baseline] R0={r_ok:.6e} expected={expected:.6e} relerr={abs(r_ok-expected)/expected:.2e} (gate <1e-12)")
print(f"[F2 +1% R0] relerr={abs(r_ok*1.01-expected)/expected:.2e} -> RED")

# F3: stability certificate -- accept a knowingly-unstable high order without the check
from witwin.maxwell.compiler.wire_impedance import fit_series_impedance
bad=fit_series_impedance(5e-4,5.8e7,band=(4e8,3e9),permeability=MU_0,order=16,dt=1e-12,samples=240,relative_tolerance=0.5,dtype=torch.float64)
sr=wl._companion_spectral_radius(bad.discrete, float(wi.dc_resistance(5e-4,5.8e7)), 1e-8, 0.02, 1e-12)
print(f"[F3 order16 no-cert] combined spectral radius = {sr:.6f} -> {'RED (>1, would grow)' if sr>=1 else 'stable'}")
