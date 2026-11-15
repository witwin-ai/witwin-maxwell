"""Four-port transformer extraction and independent-reference analysis (CPU fp64)."""
import numpy as np


DIFFERENTIAL = np.array(((1.,0.),(-1.,0.),(0.,1.),(0.,-1.)))


def winding_metrics(frequencies, z):
    """I_port=D I_w; V_w=D.T V_port, so Z_w=D.T Z_port D.

    Winding 1 is p1-p2 and winding 2 is p3-p4, common-mode currents zero.
    Positive mutual inductance corresponds to the same winding circulation.
    This open-common-mode convention differs from grounding both cold ends.
    Coupling is undefined when either effective self-inductance is nonpositive.
    """
    f=np.asarray(frequencies)
    zw=DIFFERENTIAL.T@z@DIFFERENTIAL
    inductance=-zw.imag/(2*np.pi*f[:,None,None])
    l1,l2=inductance[:,0,0],inductance[:,1,1]
    coupling=[float(m/np.sqrt(a*b)) if a>0 and b>0 else None
              for a,b,m in zip(l1,l2,inductance[:,0,1])]
    return dict(frequency_Hz=f.tolist(),L1_H=inductance[:,0,0].tolist(),
        L2_H=inductance[:,1,1].tolist(),M_H=inductance[:,0,1].tolist(),
        k=coupling,
        R1_ohm=zw[:,0,0].real.tolist(),R2_ohm=zw[:,1,1].real.tolist(),
        Q1=(-zw[:,0,0].imag/zw[:,0,0].real).tolist(),
        Q2=(-zw[:,1,1].imag/zw[:,1,1].real).tolist())


def unpack(data, name):
    return np.asarray(data[name+"_real"])+1j*np.asarray(data[name+"_imag"])


def reference_checks(data):
    s=unpack(data,"s")
    return dict(max_S_reciprocity=float(np.max(np.abs(s-s.swapaxes(-1,-2)))),
        max_singular_S=float(np.linalg.svd(s,compute_uv=False).max()))


def observed_grid_order(values):
    """Report raw three-grid order and monotonicity without extrapolation."""
    a,b,c=map(float,values)
    ratio=(a-b)/(b-c)
    return dict(observed_order=float(np.log2(abs(ratio))),monotone=ratio>0,
        caveat="Feature-aligned grids need not form an asymptotic halving sequence; temporal and model errors remain.")


def geometry_components(rectangles, vias):
    """Exact rectangle-overlap graph before adding any ideal port bridge."""
    shapes=list(rectangles)
    parent=list(range(len(shapes)))
    def find(a):
        while parent[a]!=a:
            a=parent[a]
        return a
    def join(a,b):
        parent[find(b)]=find(a)
    def overlap(a,b):
        return (min(a[1],b[1])>=max(a[0],b[0]) and min(a[3],b[3])>=max(a[2],b[2])
                and (min(a[1],b[1])>max(a[0],b[0]) or min(a[3],b[3])>max(a[2],b[2])))
    for i,a in enumerate(shapes):
        for j,b in enumerate(shapes[:i]):
            if a[4]==b[4] and overlap(a,b):
                join(i,j)
    for via in vias:
        attached=[i for i,r in enumerate(shapes) if r[4] in via[4:] and overlap(r,via)]
        assert attached
        for i in attached[1:]:
            join(attached[0],i)
    return [find(i) for i in range(len(shapes))]
