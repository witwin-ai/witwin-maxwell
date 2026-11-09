"""Independent physical current-to-soft-E lowering for reference controls.

This validation helper does not import the production port compiler or runtime.
It targets a fixed Cartesian aperture or a radial cylindrical aperture in static
scalar bulk material. An external soft-E source adds a field increment after
Ampere, so a known impressed current needs the actual constitutive response b.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np


@dataclass(frozen=True)
class CurrentEdge:
    axis: int
    index: tuple[int,int,int]
    position: tuple[float,float,float]
    edge_length: float
    dual_area: float
    voltage_weight: float
    current_density_per_amp: float
    soft_e_per_amp: float
    epsilon: float
    conductivity: float
    source_box: tuple[tuple[float,float,float],tuple[float,float,float]]


def dual_widths(nodes):
    nodes=np.asarray(nodes,dtype=float)
    if nodes.ndim!=1 or len(nodes)<2 or not np.isfinite(nodes).all() or not np.all(np.diff(nodes)>0):
        raise ValueError('Independent reference axes require finite increasing coordinates.')
    edges=np.r_[nodes[0],(nodes[:-1]+nodes[1:])/2,nodes[-1]]
    return edges,np.diff(edges)


def clipped_dual_weights(nodes,bounds):
    edges,widths=dual_widths(nodes);lower,upper=map(float,bounds)
    if not np.isfinite((lower,upper)).all() or lower>=upper or lower<edges[0] or upper>edges[-1]:
        raise ValueError('A finite aperture must lie inside the independent grid.')
    clipped=np.maximum(0,np.minimum(edges[1:],upper)-np.maximum(edges[:-1],lower))
    return clipped/(upper-lower),widths


def compile_known_current_edges(axis_nodes, *, axis,path,aperture_bounds,dt,epsilon,conductivity=0.,cylindrical_radial=False):
    """Lower known scalar branch I into signed physical native-edge currents.

    ``path`` is oriented negative-to-positive. ``epsilon`` and conductivity are
    the independent engine's physical coefficients in SI, scalar or index
    callbacks. Covariant metric tensors must not be passed as physical epsilon.
    In a cylinder the native axes are (r,phi,z) and only radial paths are valid.
    The physical impressed Maxwell current is -I times this voltage functional,
    matching a positive field-work current I in the production PortData contract.
    """
    axes=tuple(np.asarray(values,dtype=float) for values in axis_nodes)
    for values in axes:dual_widths(values)
    if len(axes)!=3 or axis not in (0,1,2) or dt<=0 or not np.isfinite(dt):
        raise ValueError('A known-current reference needs three axes and a positive finite timestep.')
    if cylindrical_radial and (axis!=0 or axes[0][0]<=0):
        raise ValueError('Cylindrical known-current controls require positive-radius radial paths.')
    start,stop=map(float,path);sign=1. if stop>start else -1.
    if start==stop or not np.isfinite((start,stop)).all():raise ValueError('The oriented path must have nonzero finite length.')
    lower,upper=sorted((start,stop));nodes=axes[axis]
    scale=max(abs(lower),abs(upper),np.max(np.diff(nodes)))
    tolerance=8*np.finfo(float).eps*scale
    normal=np.flatnonzero((nodes[:-1]>=lower-tolerance)&(nodes[1:]<=upper+tolerance))
    if not len(normal) or abs(nodes[normal[0]]-lower)>tolerance or abs(nodes[normal[-1]+1]-upper)>tolerance:
        raise ValueError('The independently declared path endpoints must resolve to grid nodes.')
    transverse=[index for index in range(3) if index!=axis]
    measures={index:clipped_dual_weights(axes[index],aperture_bounds[index]) for index in transverse}
    support={index:np.flatnonzero(measures[index][0]>0) for index in transverse}
    coordinates=list(axes);coordinates[axis]=(nodes[:-1]+nodes[1:])/2
    result=[]
    for i,j,k in itertools.product(normal,support[transverse[0]],support[transverse[1]]):
        index=[0,0,0];index[axis]=int(i);index[transverse[0]]=int(j);index[transverse[1]]=int(k)
        index=tuple(index);point=tuple(float(coordinates[d][index[d]]) for d in range(3))
        length=float(nodes[i+1]-nodes[i])
        fraction=np.prod([measures[d][0][index[d]] for d in transverse])
        area=float(np.prod([measures[d][1][index[d]] for d in transverse]))
        if cylindrical_radial:area*=point[0]
        eps=float(epsilon(index) if callable(epsilon) else epsilon)
        sigma=float(conductivity(index) if callable(conductivity) else conductivity)
        if not np.isfinite((eps,sigma)).all() or eps<=0 or sigma<0:raise ValueError('Physical static source coefficients must be passive and finite.')
        weight=sign*length*fraction;density=sign*fraction/area
        amplitude=dt/(eps+sigma*dt/2)*density
        widths=[]
        for d in range(3):
            values=coordinates[d];n=index[d]
            widths.append(min(float(values[n]-values[max(0,n-1)]) or np.inf,float(values[min(n+1,len(values)-1)]-values[n]) or np.inf)/8)
        box=(tuple(point[d]-widths[d] for d in range(3)),tuple(point[d]+widths[d] for d in range(3)))
        result.append(CurrentEdge(axis,index,point,length,area,float(weight),float(density),float(amplitude),eps,sigma,box))
    return tuple(result)


def install_known_current_sources(csx,edges,*,amplitude=1.,prefix='known_current'):
    """Create disjoint per-edge soft-E supports, without a local resistor."""
    for number,edge in enumerate(edges):
        value=np.zeros(3);value[edge.axis]=amplitude*edge.soft_e_per_amp
        source=csx.AddExcitation(f'{prefix}_{number}',exc_type=0,exc_val=value)
        source.AddBox(*edge.source_box,priority=30)


def known_current_spectrum(source_times,source_values,frequencies,*,dt,amplitude=1.,native_convention=True):
    """Exact midpoint source clock and single-sided pulse spectrum.

    Engine iteration n adds f[n] to E((n+1)dt); its impressed current belongs to
    (n+1/2)dt. Voltage probes use n*dt and magnetic probes (n+1/2)*dt. The source
    table's et timestamps n*dt therefore require +dt/2. This is a fixed clock
    convention, not a fitted reference-plane phase.
    """
    time=np.asarray(source_times,dtype=float);values=np.asarray(source_values,dtype=float)
    frequency=np.asarray(frequencies,dtype=float)
    if time.ndim!=1 or time.shape!=values.shape or len(time)<2:raise ValueError('A sampled current requires matching one-dimensional source arrays.')
    # et text files use reduced output precision; use the authoritative engine
    # dt for both the clock and quadrature rather than rounded et increments.
    if not np.allclose(time,np.arange(len(time))*dt,rtol=2e-5,atol=1e-8*dt):raise ValueError('Source table timestamps disagree with the engine timestep.')
    midpoint=(np.arange(len(time))+.5)*dt
    spectrum=2*dt*amplitude*(np.exp(-2j*np.pi*frequency[:,None]*midpoint)@values)
    return spectrum.conj() if native_convention else spectrum
