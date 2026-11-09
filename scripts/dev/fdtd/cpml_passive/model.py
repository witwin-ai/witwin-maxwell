"""CPU reduced Yee blocks for the passive-closure investigation.

No solver is constructed and no native extension is imported. Torch is used only
by the repository's CPU material compiler and profile builder. All operators,
time stepping, projections, spectra and energy calculations use NumPy/SciPy.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math

import numpy as np
from scipy import linalg

DT = 1.6666666666666667e-11
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
C0 = 299792458.0
ETA = math.sqrt(MU0 / EPS0)


@dataclass(frozen=True)
class Cell:
    polarization: str = "TM"
    wall: str = "periodic"
    order: int = 1
    dt: float = DT
    dy: float = .025
    dz: float = .025
    period: float = 1.28
    zhalf: float = .64
    layers: int = 12
    slab: bool = True
    slab_epsilon: float = 3.
    preset: str = "real"
    theta: float = 0.
    f_min: float = .3e9
    alpha_scale: float | None = None
    grading: float = 3.
    kappa: float = 1.
    reflection: float = 1e-6

    def __post_init__(self):
        if self.polarization not in ("TE", "TM") or self.wall not in ("periodic", "pec", "pmc"):
            raise ValueError("Expected TE/TM and periodic/pec/pmc")

    @property
    def ny(self):
        return math.ceil(self.period / self.dy)

    @property
    def nodes(self):
        base = np.linspace(-self.zhalf, self.zhalf, math.ceil(2*self.zhalf/self.dz)+1)
        return np.r_[base[0] - (base[1]-base[0])*np.arange(self.layers, 0, -1),
                     base, base[-1]+(base[-1]-base[-2])*np.arange(1, self.layers+1)]

    @property
    def names(self):
        return (("Ex", "Hy", "Hz", "psi_ex_z", "psi_hy_z") if self.polarization == "TE"
                else ("Ey", "Ez", "Hx", "psi_ey_z", "psi_hx_z"))

    @property
    def lengths(self):
        n = len(self.nodes)-1
        return (n-1, n, n-1, n-1, n) if self.polarization == "TE" else (n-1, n, n, n-1, n)

    @property
    def scales(self):
        h = np.diff(self.nodes)[0]
        return (1., 1/ETA, 1/ETA, 1/(ETA*h), 1/h) if self.polarization == "TE" else (1., 1., 1/ETA, 1/(ETA*h), 1/h)


def component(name):
    return name if not name.startswith("psi") else name.split("_")[1].capitalize()


def y_node(name):
    c = component(name)
    return (c[1] != "y") if c[0] == "E" else (c[1] == "y")


def z_node(name):
    c = component(name)
    return (c[1] != "z") if c[0] == "E" else (c[1] == "z")


def basis(cell, name, *, rounded=False):
    """Native probe injection and projection, including duplicate node masters."""
    node = y_node(name)
    y = np.arange(cell.ny + int(node)) + (0 if node else .5)
    if cell.wall == "periodic":
        phase = 2*np.pi*cell.order*y/cell.ny
        b = np.cos(phase)
        p = np.exp(-1j*phase[:cell.ny]) * ((1 if cell.order == 0 else 2)/cell.ny)
    else:
        c = component(name)
        sine = c in ("Ex", "Ez", "Hy")
        if cell.wall == "pmc":
            sine = not sine
        phase = np.pi*cell.order*y/cell.ny
        b = np.sin(phase) if sine else np.cos(phase)
        p = b / (b @ b)
    return (b.astype(np.float32) if rounded else b), p


@lru_cache(maxsize=64)
def material_edges(cell, native=True):
    """Read-only CPU scene compilation; retain the probe's actual subpixel edges.

    In particular, do not substitute a binary slab: the finite box's corners
    affect polarized quadrature at the transverse periodic seam.
    """
    import torch
    import witwin.maxwell as mw
    from witwin.maxwell.scene import PreparedScene
    torch.set_num_threads(1)
    scene = mw.Scene(domain=mw.Domain(bounds=((-cell.period/2, cell.period/2),
                    (-cell.period/2, cell.period/2), (-cell.zhalf, cell.zhalf))),
        grid=mw.GridSpec.anisotropic(cell.period, cell.dy, cell.dz),
        boundary=mw.BoundarySpec.pml(num_layers=cell.layers).with_faces(
            x_low="periodic", x_high="periodic", y_low=cell.wall, y_high=cell.wall),
        subpixel_samples=mw.SubpixelSpec(samples=3, averaging="polarized", pec="staircase"), device="cpu")
    if cell.slab:
        scene.add_structure(mw.Structure(geometry=mw.Box(position=(0.,0.,0.),
            size=(cell.period,cell.period,.1)), material=mw.Material(eps_r=cell.slab_epsilon)))
    prepared = PreparedScene(scene, material_dtype=torch.float32 if native else torch.float64)
    edges = prepared.compile_materials()["edge_components"]["eps"]
    return {k: v[0].numpy().copy() for k, v in edges.items()}


def profiles(cell, *, native=False):
    """Import the primary profile builder; dtype=float64 except native replay."""
    import torch
    from witwin.maxwell.fdtd.boundary.cpml import _cpml_profile_1d, DEFAULT_CPML_CONFIG
    cfg = dict(DEFAULT_CPML_CONFIG, grading_order=cell.grading, kappa_max=cell.kappa,
               reflection=cell.reflection)
    rate, floor = None, 0.
    if cell.preset == "cfs":
        rate = 2*np.pi*cell.f_min
        fraction = np.sin(np.deg2rad(cell.theta))
        floor = min(1., 2*fraction)
        cosine = np.sqrt(1-min(fraction, np.sqrt(1-(4.2/8)**2))**2)
        cfg.update(alpha_max=EPS0*rate, grading_order=2., reflection=10**(-4.2/cosine))
    elif cell.preset not in ("real", "alpha0"):
        raise ValueError("Unknown preset")
    if cell.alpha_scale is not None:
        rate, floor = 2*np.pi*C0/cell.period, 1.
        cfg["alpha_max"] = cell.alpha_scale*EPS0*rate
    dtype = torch.float32 if native else torch.float64
    nodes = cell.nodes
    out = []
    for magnetic in (False, True):
        scale = torch.tensor(1., dtype=dtype, device="cpu")
        config = dict(cfg)
        if magnetic:
            scale = scale * (MU0/EPS0)
            config["alpha_max"] *= MU0/EPS0
        pos = (nodes[:-1]+nodes[1:])/2 if magnetic else nodes
        v = _cpml_profile_1d(pos, nodes, cell.layers, cell.dt, MU0 if magnetic else EPS0,
                            scale, ETA, config, shift_rate=rate, shift_floor=floor)
        out.append(tuple(x.numpy().copy() for x in v))
    return tuple(out)


def fma32(a, b, c):
    """One rounded product/add (all operands originate as binary32)."""
    return (np.asarray(a, np.float64)*np.asarray(b, np.float64)+np.asarray(c, np.float64)).astype(np.float32)


@lru_cache(maxsize=64)
def transverse_memory_resolvent(cell, name, diffusivity, boundary_phase=1.):
    """Implicit transverse diffusion with the field's PEC/PMC parity.

    Node duplicates are eliminated for periodic fields. Wall-node sine fields
    have zero endpoints; cosine fields use the half-weight Neumann SBP closure.
    This CPU research operator is not a native runtime or arithmetic contract.
    """
    if not np.isfinite(diffusivity) or diffusivity < 0:
        raise ValueError("Memory diffusivity must be finite and nonnegative")
    node = y_node(name)
    n = cell.ny + int(node and cell.wall != "periodic")
    lap = (2*np.eye(n)-np.eye(n,k=1)-np.eye(n,k=-1)).astype(
        complex if boundary_phase != 1 else float)
    if cell.wall == "periodic":
        lap[0,-1] -= np.conj(boundary_phase)
        lap[-1,0] -= boundary_phase
    else:
        sine = component(name) in ("Ex", "Ez", "Hy")
        if cell.wall == "pmc":
            sine = not sine
        if node:
            if sine:
                # Prescribed zero endpoints are excluded from the solve.
                lap[[0,-1]] = 0
                lap[:,[0,-1]] = 0
            else:
                lap[0,1] = lap[-1,-2] = -2
        else:
            lap[0,0] = lap[-1,-1] = 3 if sine else 1
    matrix = np.eye(n)+diffusivity*cell.dt*lap/(cell.period/cell.ny)**2
    inverse = linalg.solve(matrix,np.eye(n),check_finite=False)
    inverse.flags.writeable = False
    return inverse


def native_matrix(cell, *, rounded=True, candidate="current", coefficients=None,
                  tangential_coefficients=None, memory_diffusivity=0.,
                  physical_wavenumber=None):
    """Measure the CPU stencil on the native real cosine/sine basis.

    ``rounded=True`` emulates binary32 CUDA FMA grouping. False uses float64
    arithmetic with the same compiled binary32 constants and injection basis.
    This is a projection experiment, not a linear float32 time-step operator.
    """
    if rounded and candidate!="current":
        raise ValueError("Candidate arithmetic must be float64; no unimplemented CUDA grouping is implied")
    if memory_diffusivity and (rounded or candidate != "current" or tangential_coefficients is not None):
        raise ValueError("Diffusive memory research uses float64 current normal memories only")
    if physical_wavenumber is not None and (rounded or cell.wall != "periodic"):
        raise ValueError("Complex fixed-k stencil research requires periodic float64 replay")
    if tangential_coefficients is not None and candidate != "current":
        raise ValueError("Tangential profiles are implemented only for the current recurrence")
    dtype = np.float32 if rounded else np.float64
    phase = 1. if physical_wavenumber is None else np.exp(1j*physical_wavenumber*cell.period)
    if physical_wavenumber is not None:
        dtype = np.complex128
    madd = fma32 if rounded else lambda a,b,c: a*b+c
    nodes = cell.nodes
    n = len(nodes)-1
    names = cell.names
    scales = cell.scales
    if tangential_coefficients is not None:
        names += (("psi_ex_y", "psi_hz_y") if cell.polarization == "TE"
                  else ("psi_ez_y", "psi_hx_y"))
        scales += (scales[3], scales[4])
    sizes = [cell.ny+int(y_node(k)) for k in names]
    nzs = [n+int(z_node(k)) for k in names]
    indices = [np.arange(1,n) if z_node(k) else np.arange(n) for k in names]
    total = sum(map(len, indices))
    waves = [basis(cell,k,rounded=True) for k in names]
    if physical_wavenumber is not None:
        waves = []
        for name,size in zip(names,sizes):
            y = (np.arange(size)+(0 if y_node(name) else .5))*cell.period/cell.ny
            wave = np.exp(1j*physical_wavenumber*y)
            waves.append((wave,np.conj(wave[:cell.ny])/cell.ny))
    eps = material_edges(replace(cell, polarization="TM", order=1, dt=DT), True)
    # torch computes reciprocal then multiplication for a Python scalar / tensor.
    ce = {k: (np.reciprocal(v*np.float32(EPS0))*np.float32(cell.dt)).astype(np.float32).astype(dtype) for k,v in eps.items()}
    ch = dtype(np.float32(cell.dt/MU0))
    inv_h = (1/np.diff(nodes)).astype(np.float32).astype(dtype)
    dual = np.r_[np.diff(nodes)[0], .5*(np.diff(nodes)[:-1]+np.diff(nodes)[1:]), np.diff(nodes)[-1]]
    inv_e = (1/dual).astype(np.float32).astype(dtype)
    inv_y = dtype(np.float32(cell.ny/cell.period))
    pp = profiles(cell, native=True) if coefficients is None else coefficients
    ke,be,ae = [x.astype(dtype) for x in pp[0]]
    kh,bh,ah = [x.astype(dtype) for x in pp[1]]
    ke = np.reciprocal(ke).astype(dtype)
    kh = np.reciprocal(kh).astype(dtype)
    fraction={"current":1.,"c1":.5,"c3_old":0.,"c3_average":.5}[candidate]
    if candidate=="c1":
        for b,c in ((be,ae),(bh,ah)):
            active=(b>0)&(b<1)
            r,w=np.zeros_like(b),np.zeros_like(b)
            r[active]=-np.log(b[active])/cell.dt
            w[active]=-c[active]/(1-b[active])
            b[:]=(1-.5*r*cell.dt)/(1+.5*r*cell.dt)
            c[:]=-r*w*cell.dt/(1+.5*r*cell.dt)
    result = np.empty((total,total), complex)

    def memory(old,d,b,c,k,name=None):
        new=madd(c,d,b*old)
        if memory_diffusivity:
            if name is None:
                raise ValueError("Diffusive memory needs its owning field")
            active = np.flatnonzero(c)
            inverse = transverse_memory_resolvent(cell,name,memory_diffusivity,phase)
            count = len(inverse)
            new[:count,active] = inverse@new[:count,active]
            if cell.wall == "periodic" and y_node(name):
                new[-1,active] = phase*new[0,active]
        value=new if candidate=="current" else fraction*new+(1-fraction)*old
        return new,madd(k,d,value)

    def tangent(old, d, group):
        k,b,c = [np.asarray(v,dtype=dtype) for v in group]
        return memory(old,d,b,c,np.reciprocal(k).astype(dtype))

    def forward_y(a):
        return np.diff(a, axis=0)*inv_y

    def backward_y(a):
        b = np.zeros((cell.ny+1, a.shape[1]), dtype=dtype)
        b[1:-1] = np.diff(a, axis=0)*inv_y
        if cell.wall == "periodic":
            b[0] = (a[0]-np.conj(phase)*a[-1])*inv_y
            b[-1] = phase*b[0]
        elif cell.wall == "pmc":
            b[0],b[-1] = 2*a[0]*inv_y, -2*a[-1]*inv_y
        return b

    def electric(e, curl, name):
        v = madd(ce[name], curl, e)
        if name != "Ez":
            v[:,0] = e[:,0]
            v[:,-1] = e[:,-1]
        if cell.wall == "pec" and y_node(name):
            v[0] = v[-1] = 0
        return v

    col = 0
    for j in range(len(names)):
        for z in indices[j]:
            a = [np.zeros((sy,sz), dtype=dtype) for sy,sz in zip(sizes,nzs)]
            # Native multiplication is binary32, even in the float64 replay.
            injection_dtype = np.float32 if physical_wavenumber is None else np.complex64
            a[j][:,z] = (waves[j][0]*np.float32(scales[j])).astype(injection_dtype).astype(dtype)
            py_e,py_h = a[5:] if tangential_coefficients is not None else (None,None)
            if cell.polarization == "TM":
                ey,ez,h,pe,ph = a[:5]
                dh = np.diff(ey,axis=1)*inv_h
                ph,corrected=memory(ph,dh,bh,ah,kh,"Hx")
                dy = forward_y(ez)
                if py_h is not None:
                    py_h,dy = tangent(py_h,dy,tangential_coefficients[1])
                h = madd(ch, corrected-dy, h)
                de = np.zeros_like(ey)
                de[:,1:-1] = np.diff(h,axis=1)*inv_e[1:-1]
                pen,corrected=memory(pe,de,be,ae,ke,"Ey")
                pe[:,1:-1] = pen[:,1:-1]
                ey = electric(ey, corrected, "Ey")
                dy = backward_y(h)
                if py_e is not None:
                    py_e,dy = tangent(py_e,dy,tangential_coefficients[0])
                ez = electric(ez, -dy, "Ez")
                a = ey,ez,h,pe,ph
            else:
                e,hy,hz,pe,ph = a[:5]
                dh = np.diff(e,axis=1)*inv_h
                ph,corrected=memory(ph,dh,bh,ah,kh,"Hy")
                hy = madd(-ch,corrected,hy)
                dy = forward_y(e)
                if py_h is not None:
                    py_h,dy = tangent(py_h,dy,tangential_coefficients[1])
                hz = madd(ch,dy,hz)
                de = np.zeros_like(e)
                de[:,1:-1] = np.diff(hy,axis=1)*inv_e[1:-1]
                pen,corrected=memory(pe,de,be,ae,ke,"Ex")
                pe[:,1:-1] = pen[:,1:-1]
                dy = backward_y(hz)
                if py_e is not None:
                    py_e,dy = tangent(py_e,dy,tangential_coefficients[0])
                e = electric(e, dy-corrected, "Ex")
                a = e,hy,hz,pe,ph
            if py_e is not None:
                a += (py_e,py_h)
            projection_dtype = np.float64 if physical_wavenumber is None else np.complex128
            result[:,col] = np.concatenate([(waves[k][1] @ v[:len(waves[k][1])].astype(projection_dtype))[indices[k]] / scales[k] for k,v in enumerate(a)])
            col += 1
    return result


def oscillatory_spectrum(matrix, dt):
    values = linalg.eigvals(matrix, check_finite=False)
    frequencies = np.angle(values)/(2*np.pi*dt)
    mask = np.abs(frequencies)>1e6
    i = np.argmax(np.where(mask,np.abs(values),0))
    return {"rho": float(abs(values[i])), "frequency_GHz": float(abs(frequencies[i])*1e-9),
            "all_rho": float(np.max(abs(values)))}


class Block:
    """Float64 harmonic block in real phase coordinates (normal field in quadrature).

    Fields are E and eta*H. Memories are h*psi or eta*h*psi, as in the native
    probe. F maps zero-ended tangential nodes to half nodes; B=-F.T. Transverse
    symbols include the exact Yee sine, not the continuum wavenumber.
    """
    def __init__(self, cell, *, material="repository", wavenumber=None, coefficients=None,
                 tangential_coefficients=None, memory_diffusivity=0.):
        if not np.isfinite(memory_diffusivity) or memory_diffusivity < 0:
            raise ValueError("Memory diffusivity must be finite and nonnegative")
        if memory_diffusivity and tangential_coefficients is not None:
            raise ValueError("Diffusive memory research uses normal memories only")
        self.memory_diffusivity = memory_diffusivity
        self.cell = cell
        self.n = len(cell.nodes)-1
        self.h = np.diff(cell.nodes)[0]
        self.t = cell.dt / np.sqrt(EPS0*MU0)
        self.F = np.diff(np.pad(np.eye(self.n-1), ((1,1),(0,0))), axis=0)/self.h
        self.B = -self.F.T
        phase = (2 if cell.wall == "periodic" else 1)*np.pi*cell.order/cell.ny
        self.k = 2*np.sin(phase/2)/(cell.period/cell.ny) if wavenumber is None else wavenumber
        if material == "repository":
            material_cell = Cell(wall=cell.wall,dy=cell.dy,dz=cell.dz,period=cell.period,
                                 zhalf=cell.zhalf,layers=cell.layers,slab=cell.slab,
                                 slab_epsilon=cell.slab_epsilon)
            edges = material_edges(material_cell, False)
            # Complex Fourier Galerkin projection has |basis|^2=1; wall bases
            # use their actual quadrature, including half endpoint weights.
            def inverse(name):
                v = 1/edges[name]
                if cell.wall == "periodic":
                    if y_node(name):
                        v = v[:-1]
                    return v.mean(axis=0)
                b,_ = basis(cell,name)
                weights = b*b
                if y_node(name):
                    weights = weights.copy()
                    weights[[0,-1]] *= .5
                return weights @ v / weights.sum()
            self.a = inverse("Ex" if cell.polarization == "TE" else "Ey")[1:-1]
            self.av = inverse("Ez") if cell.polarization == "TM" else np.ones(self.n)
        elif material == "binary":
            self.a = 1/np.where(np.abs(cell.nodes[1:-1])<.05,3.,1.)
            self.av = 1/np.where(np.abs((cell.nodes[:-1]+cell.nodes[1:])/2)<.05,3.,1.)
        elif material == "vacuum":
            self.a, self.av = np.ones(self.n-1), np.ones(self.n)
        else:
            raise ValueError("Unknown material")
        self.profile = profiles(cell) if coefficients is None else tuple(
            tuple(np.asarray(v,dtype=np.float64).copy() for v in group) for group in coefficients)
        if len(self.profile)!=2 or any(len(g)!=3 for g in self.profile):
            raise ValueError("Expected electric and magnetic (kappa,b,c) triples")
        for group,length in zip(self.profile,(self.n+1,self.n)):
            if any(v.shape!=(length,) or not np.isfinite(v).all() for v in group):
                raise ValueError("Coefficient vectors must be finite and match the staggered z grid")
        self.ke,self.be,self.ce = [x[1:-1] for x in self.profile[0]]
        self.kh,self.bh,self.ch = self.profile[1]
        self.lengths = cell.lengths
        self.tangential = tangential_coefficients
        if self.tangential is not None:
            length = self.n-1 if cell.polarization == "TE" else self.n
            self.tangential = tuple(tuple(np.asarray(v,dtype=np.float64).copy() for v in g)
                                    for g in self.tangential)
            if len(self.tangential)!=2 or any(len(g)!=3 or any(
                    v.shape!=(length,) or not np.isfinite(v).all() for v in g)
                    for g in self.tangential):
                raise ValueError("Tangential triples must match each target field's z positions")
            self.lengths += (length,length)
        self.offsets = np.cumsum((0,)+self.lengths)
        self.slices = [slice(a,b) for a,b in zip(self.offsets[:-1],self.offsets[1:])]
        self.size = sum(self.lengths)

    def split(self,x):
        return [x[s] for s in self.slices]

    def step(self,x, ordering="fresh"):
        """Native leapfrog algebra, or old/averaged psi ordering experiments."""
        if self.tangential is not None and ordering != "fresh":
            raise ValueError("Tangential profiles require the current recurrence")
        arrays = [a.copy() for a in self.split(x)]
        u,v,w,pe,ph = arrays[:5]
        trailing = (1,)*(x.ndim-1)
        def col(a):
            return a.reshape((-1,)+trailing)
        py_e,py_h = arrays[5:] if self.tangential is not None else (None,None)
        def tangent(old,d,group):
            kt,bt,ct = group
            new = col(bt)*old+col(ct)*h*d
            return new,d/col(kt)+new/h
        t,h,k,F,B = self.t,self.h,self.k,self.F,self.B
        if isinstance(ordering,tuple):
            fraction_e,fraction_h=ordering
        else:
            fraction_e = fraction_h = {"fresh":1.,"old":0.,"average":.5,"trapezoid":.5}[ordering]
        be,bh,ce,ch = self.be,self.bh,self.ce,self.ch
        if ordering == "trapezoid":
            (_,re,we),(_,rh,wh) = self.analog_parameters()
            be = (1-.5*re*self.cell.dt)/(1+.5*re*self.cell.dt)
            bh = (1-.5*rh*self.cell.dt)/(1+.5*rh*self.cell.dt)
            ce = -re*we*self.cell.dt/(1+.5*re*self.cell.dt)
            ch = -rh*wh*self.cell.dt/(1+.5*rh*self.cell.dt)
        dh = np.diff(np.pad(u, ((1,1),)+((0,0),)*(u.ndim-1)),axis=0)/h
        ph_new = col(bh)*ph + col(ch)*h*dh
        if self.memory_diffusivity:
            ph_new = np.where(col(ch)!=0,
                ph_new/(1+self.memory_diffusivity*self.cell.dt*k*k),ph_new)
        corrected_h = dh/col(self.kh)+(fraction_h*ph_new+(1-fraction_h)*ph)/h
        if self.cell.polarization == "TM":
            dy = k*v
            if py_h is not None:
                py_h,dy = tangent(py_h,dy,self.tangential[1])
            w += t*(corrected_h+dy)
            de = np.diff(w,axis=0)/h
            pe_new = col(be)*pe+col(ce)*h*de
            if self.memory_diffusivity:
                pe_new = np.where(col(ce)!=0,
                    pe_new/(1+self.memory_diffusivity*self.cell.dt*k*k),pe_new)
            u += t*col(self.a)*(de/col(self.ke)+(fraction_e*pe_new+(1-fraction_e)*pe)/h)
            if py_e is None:
                v -= t*col(self.av)*k*w
            else:
                py_e,dy = tangent(py_e,k*w,self.tangential[0])
                v -= t*col(self.av)*dy
        else:
            v -= t*corrected_h
            if py_h is None:
                w += t*k*u
            else:
                py_h,dy = tangent(py_h,k*u,self.tangential[1])
                w += t*dy
            de = np.diff(v,axis=0)/h
            pe_new = col(be)*pe+col(ce)*h*de
            if self.memory_diffusivity:
                pe_new = np.where(col(ce)!=0,
                    pe_new/(1+self.memory_diffusivity*self.cell.dt*k*k),pe_new)
            dy = k*w
            if py_e is not None:
                py_e,dy = tangent(py_e,dy,self.tangential[0])
            u -= t*col(self.a)*(de/col(self.ke)+(fraction_e*pe_new+(1-fraction_e)*pe)/h+dy)
        out = (u,v,w,pe_new,ph_new)
        return np.concatenate(out if py_e is None else out+(py_e,py_h))

    def bare_generator(self):
        """Field-only semi-discrete vacuum curl (physical units 1/s)."""
        m = sum(self.lengths[:3])
        A = np.zeros((m,m))
        u,v,w = self.slices[:3]
        c = 1/np.sqrt(EPS0*MU0)
        if self.cell.polarization == "TM":
            A[u,w] = c*self.a[:,None]*self.B
            A[v,w] = -c*np.diag(self.av*self.k)
            A[w,u] = c*self.F
            A[w,v] = c*self.k*np.eye(self.n)
        else:
            A[u,v] = -c*self.a[:,None]*self.B
            A[u,w] = -c*np.diag(self.a*self.k)
            A[v,u] = -c*self.F
            A[w,u] = c*self.k*np.eye(self.n-1)
        return A

    def analog_parameters(self):
        """Equivalent held-input one-pole rates, from the supplied b/c vectors.

        This deliberately does not call a time-centering change 'the same
        transfer function': CFS moment-matched inputs are interpreted as the
        supplied effective pole, DC residue and high-frequency direct term.
        """
        out=[]
        for k,b,c in ((self.ke,self.be,self.ce),(self.kh,self.bh,self.ch)):
            if np.any(((b<=0)|(b>=1))&(c!=0)):
                raise ValueError("A finite passive-pole ADE requires 0 < b < 1 in driven memory slots")
            active = (b>0)&(b<1)
            r,w = np.zeros_like(b),np.zeros_like(b)
            r[active] = -np.log(b[active])/self.cell.dt
            w[active] = -c[active]/(1-b[active])
            out.append((1/k,r,w))
        return out

    def ade_generator(self):
        """c1: simultaneous Maxwell/psi ADE, suitable for full midpoint solve."""
        A = np.zeros((self.size,self.size))
        u,v,w,pe,ph = self.slices
        c = 1/np.sqrt(EPS0*MU0)
        (de,re,we),(dh,rh,wh) = self.analog_parameters()
        if self.cell.polarization == "TM":
            A[u,w] = c*(self.a*de)[:,None]*self.B
            A[u,pe] = c/self.h*np.diag(self.a)
            A[v,w] = -c*np.diag(self.av*self.k)
            A[w,u] = c*dh[:,None]*self.F
            A[w,v] = c*self.k*np.eye(self.n)
            A[w,ph] = c/self.h*np.eye(self.n)
            A[pe,w] = -self.h*(re*we)[:,None]*self.B
        else:
            A[u,v] = -c*(self.a*de)[:,None]*self.B
            A[u,w] = -c*np.diag(self.a*self.k)
            A[u,pe] = -c/self.h*np.diag(self.a)
            A[v,u] = -c*dh[:,None]*self.F
            A[v,ph] = -c/self.h*np.eye(self.n)
            A[w,u] = c*self.k*np.eye(self.n-1)
            A[pe,v] = -self.h*(re*we)[:,None]*self.B
        A[ph,u] = -self.h*(rh*wh)[:,None]*self.F
        A[pe,pe] = -np.diag(re)
        A[ph,ph] = -np.diag(rh)
        return A

    def constitutive_generator(self):
        """c2: unsplit constitutive stretched-coordinate ADE.

        Tangential fields see s=kappa+sigma/(s+alpha); normal fields see 1/s.
        One polarization variable is attached to every field (zero-strength
        variables are removed). Duru's physical ADE is the kappa=1,alpha=0
        transfer-function specialization. Boundary values use the same Yee SBP
        pair. This is a candidate, NOT a claim that a literature theorem covers
        this staggered, heterogeneous discretization.
        """
        A0 = self.bare_generator()
        m = A0.shape[0]
        (direct_e,re,we),(direct_h,rh,wh) = self.analog_parameters()
        # G(s)=direct*(s+alpha)/(s+r); 1/G is the tangential constitutive law.
        alpha_e = re*(1-we/direct_e)
        alpha_h = rh*(1-wh/direct_h)
        u,v,w = self.slices[:3]
        direct = np.ones(m)
        alpha,rate = np.zeros(m),np.zeros(m)
        normal = np.zeros(m,dtype=bool)
        direct[u],alpha[u],rate[u] = direct_e,alpha_e,re
        if self.cell.polarization == "TM":
            direct[v],alpha[v],rate[v] = direct_h,alpha_h,rh
            direct[w],alpha[w],rate[w] = direct_h,alpha_h,rh
            normal[v] = True
        else:
            direct[v],alpha[v],rate[v] = direct_h,alpha_h,rh
            direct[w],alpha[w],rate[w] = direct_e,alpha_e,re
            normal[w] = True
        # Tangential: x'=G curl; p'=curl-r*p, x'=d*curl-d*(r-alpha)*p.
        # Normal: x'=G^-1 curl; p'=curl-alpha*p, x'=curl/d+(r-alpha)*p/d.
        strength = rate-alpha
        active = np.flatnonzero(abs(strength)>1e-10)
        scale = np.where(normal,1/direct,direct)
        A = np.zeros((m+len(active),m+len(active)))
        A[:m,:m] = scale[:,None]*A0
        A[m:,:m] = A0[active]
        A[active,m+np.arange(len(active))] = np.where(normal[active],
            strength[active]/direct[active],-direct[active]*strength[active])
        A[m:,m:] = -np.diag(np.where(normal[active],alpha[active],rate[active]))
        return A

    def matrix(self, candidate="current"):
        if self.memory_diffusivity and candidate != "current":
            raise ValueError("Diffusive memory research uses the current field integrator")
        if self.tangential is not None and candidate != "current":
            raise ValueError("Tangential profiles require the current recurrence")
        if candidate in ("current","c3_old","c3_average"):
            ordering = {"current":"fresh","c3_old":"old","c3_average":"average"}[candidate]
            return self.step(np.eye(self.size),ordering)
        if candidate == "c1":
            return self.step(np.eye(self.size),"trapezoid")
        if candidate == "c2":
            return self.constitutive_step(np.eye(len(self.constitutive_generator())))
        if candidate == "global_midpoint":
            return midpoint(self.ade_generator()*self.cell.dt)
        if candidate == "passive_control":
            # Matched scalar conductivity is a diagnostic control, not a PML.
            A = self.bare_generator()
            (_,re,we),(_,rh,wh) = self.analog_parameters()
            damping = np.r_[re*we*self.ke, rh*wh*self.kh,
                re*we*self.ke if self.cell.polarization=="TE" else rh*wh*self.kh]
            return midpoint((A-np.diag(damping))*self.cell.dt)
        raise ValueError(candidate)

    def constitutive_step(self,x):
        """Local trapezoidal ADE kicks, H first, E second; unchanged Yee interior."""
        if x.ndim==1:
            return self.constitutive_step(x[:,None])[:,0]
        A = self.constitutive_generator()
        m = self.bare_generator().shape[0]
        y = x.copy()
        mem = y[m:]
        poles = -np.diag(A[m:,m:])
        owner = np.argmax(np.abs(A[:m,m:]),axis=0)
        row_coupling = A[owner,m+np.arange(len(owner))]
        factor = (1-.5*self.cell.dt*poles)/(1+.5*self.cell.dt*poles)
        drive_factor = self.cell.dt/(1+.5*self.cell.dt*poles)
        electric_end = self.offsets[2] if self.cell.polarization=="TM" else self.offsets[1]
        for lo,hi in ((electric_end,m),(0,electric_end)):
            take = (owner>=lo)&(owner<hi)
            old_mem = mem[take].copy()
            mem[take] = factor[take,None]*old_mem + drive_factor[take,None]*(A[m:,:m][take]@y[:m])
            y[lo:hi] += self.cell.dt*(A[lo:hi,:m]@y[:m])
            y[owner[take]] += .5*self.cell.dt*row_coupling[take,None]*(old_mem+mem[take])
        return y

    def active_indices(self,candidate):
        if candidate in ("c2","passive_control"):
            return np.arange(len(self.matrix(candidate)))
        # Dummy dense psi slots outside the layer are constrained to zero.
        indices = np.r_[np.arange(self.offsets[3]),self.offsets[3]+np.flatnonzero(self.ce),
                        self.offsets[4]+np.flatnonzero(self.ch)]
        if self.tangential is not None:
            indices = np.r_[indices, *(self.offsets[5+j]+np.flatnonzero(g[2])
                                       for j,g in enumerate(self.tangential))]
        return indices

    def field_metric(self):
        return np.diag(np.r_[1/self.a, 1/self.av if self.cell.polarization=="TM" else np.ones(self.n),
                            np.ones(self.lengths[2])])

    def yee_metric(self):
        """Bare leapfrog storage in the state (E^n, eta H^(n-1/2))."""
        P = self.field_metric()
        u,v,w = self.slices[:3]
        if self.cell.polarization == "TM":
            P[w,u] = .5*self.t*self.F
            P[w,v] = .5*self.t*self.k*np.eye(self.n)
            P[u,w] = P[w,u].T
            P[v,w] = P[w,v].T
        else:
            P[v,u] = -.5*self.t*self.F
            P[w,u] = .5*self.t*self.k*np.eye(self.n-1)
            P[u,v] = P[v,u].T
            P[u,w] = P[w,u].T
        return P


def midpoint(dt_generator):
    I = np.eye(len(dt_generator))
    return linalg.solve(I-.5*dt_generator,I+.5*dt_generator,check_finite=False)
