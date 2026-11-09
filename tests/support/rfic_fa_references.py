"""Independent F-A magnetostatic and circuit references.

Greenhouse, IEEE TPHP 10(2), 101-109 (1974), doi:10.1109/TPHP.1974.1134841:
add partial self terms and signed mutuals, including antiparallel segments.
Grover, Inductance Calculations (1946), straight parallel finite conductors.
Ruehli, IBM JRD 16(5), 470-481 (1972), doi:10.1147/rd.165.0470:
partial inductances sum to the inductance of the completed current loop.
Mohan et al., IEEE JSSC 34(10), 1419-1424 (1999),
https://web.stanford.edu/~boyd/papers/pdf/inductance_expressions.pdf.

This CPU fp64 oracle integrates finite straight filaments analytically; it
does not call the runtime Green tables, Galerkin assembly or loop/tree solve.
For PEC thin strips the transverse equilibrium measure is the arcsine law,
whose logarithmic capacity is w/4. Mutual strip GMDs use that same measure.
Uniform-current GMD (w*exp(-3/2)) is also available to quantify that choice.
Finite bends and landing-pad spreading are omitted, so this is an approximate
independent reference, not a substitute for mesh convergence.
"""
import math
import numpy as np

MU0 = 4e-7 * math.pi


def partial_inductance(segments, *, surface_current=True, order=64):
    """Signed Greenhouse sum over axis-aligned directed segments (SI units)."""
    terms = []
    # Gauss-Chebyshev nodes integrate the PEC equilibrium transverse measure.
    if surface_current:
        nodes = np.cos((np.arange(order)+.5)*math.pi/order)/2
        weights = np.full(order,1/order)
    else:
        nodes,weights = np.polynomial.legendre.leggauss(order)
        nodes,weights = nodes/2,weights/2
    for a,b,w,t in segments:
        a,b = np.asarray(a),np.asarray(b)
        axis = int(np.flatnonzero(a!=b)[0])
        sign = float(np.sign(b[axis]-a[axis]))
        lo,hi = sorted((a[axis],b[axis]))
        length = hi-lo
        radius = w/4 if surface_current else w*math.exp(-1.5)
        if axis == 2:
            radius = w*math.gamma(.25)**2/(4*math.pi**1.5)
        value = MU0/(2*math.pi)*(length*math.asinh(length/radius)-math.hypot(length,radius)+radius)
        terms.append((axis,sign,lo,hi,a,w,value))
    total = sum(t[-1] for t in terms)
    self_total = total
    positive,negative = 0.,0.
    for i,(axis,sign,a,b,p,w,_) in enumerate(terms):
        for ax,sg,c,d,q,v,_ in terms[:i]:
            if axis != ax:
                continue
            transverse = [j for j in range(3) if j!=axis]
            delta = p[transverse]-q[transverse]
            if axis == 2:
                separation = np.linalg.norm(delta)
            elif np.linalg.norm(delta)==0:
                separation = w/4 if surface_current else w*math.exp(-1.5)
            else:
                lateral = 1-axis
                distance = p[lateral]-q[lateral]+w*nodes[:,None]-v*nodes[None,:]
                r = np.hypot(distance,p[2]-q[2])
                separation = math.exp(float(np.sum(weights[:,None]*weights[None,:]*np.log(r))))
            def primitive(x):
                return x*math.asinh(x/separation)-math.hypot(x,separation)
            mutual = MU0/(4*math.pi)*sign*sg*(primitive(b-c)+primitive(a-d)-primitive(a-c)-primitive(b-d))
            total += 2*mutual
            if mutual>=0:
                positive += 2*mutual
            else:
                negative += 2*mutual
    return dict(total_H=total,self_H=self_total,positive_mutual_H=positive,negative_mutual_H=negative)


def greenhouse_fa(fixture, *, surface_current=True):
    assert fixture.port_segment is not None
    return partial_inductance((*fixture.segments,fixture.port_segment),surface_current=surface_current)


def dc_resistance_fa(fixture, conductivity=5.8e7):
    """Centreline sheets plus solid rho*h/A vias; ideal gap is excluded."""
    sheets,vias = 0.,0.
    for a,b,width,thickness in fixture.segments:
        length = math.dist(a,b)
        if thickness:
            sheets += length/(conductivity*width*thickness)
        else:
            vias += length/(conductivity*width**2)
    # Two 5-um terminal caps carry current between their centreline and gap.
    if fixture.port_segment is not None:
        _,_,width,thickness = fixture.port_segment
        sheets += 10e-6/(conductivity*width*thickness)
    return dict(sheet_ohm=sheets,solid_via_ohm=vias,total_ohm=sheets+vias)


def mohan_spiral():
    n,dout,din = 3.5,200e-6,105e-6
    average,rho = (dout+din)/2,(dout-din)/(dout+din)
    return dict(current_sheet_H=MU0*n*n*average*1.27/2*(math.log(2.07/rho)+.18*rho+.13*rho*rho),
                modified_wheeler_H=2.34*MU0*n*n*average/(1+2.75*rho))


def extract_fa_admittance(frequencies,y):
    """Public-network extraction with exp(-i omega t) signs.

    d/dt=-i*omega gives Z_L=-i*omega*L. A pi series admittance contributes
    Y21=-1/Zs, hence Zs=-1/Y21, L=-Im(Zs)/omega, R=Re(Zs).
    With V2=0, I1=Y11*V1, Zin=1/Y11; Q=-Im(Zin)/Re(Zin).
    """
    omega = 2*math.pi*frequencies
    series = -1/y[:,1,0] if y.shape[-1]==2 else 1/y[:,0,0]
    zin = 1/y[:,0,0]
    return dict(frequency_Hz=frequencies.tolist(),L_H=(-series.imag/omega).tolist(),
                R_ohm=series.real.tolist(),L11_H=(-zin.imag/omega).tolist(),
                Rin_ohm=zin.real.tolist(),Q=(-zin.imag/zin.real).tolist())


def extract_fa(network):
    """Extract the standard F-A quantities from a public NetworkData object."""
    return extract_fa_admittance(network.frequencies,network.to_y())


def resonance_and_peak(data):
    f,q,L = data["frequency_Hz"],data["Q"],data["L11_H"]
    inductive = [i for i,x in enumerate(L) if x>0]
    peak = max(inductive,key=lambda i:q[i])
    crossings = []
    for i in range(len(f)-1):
        if L[i]>0 and L[i+1]<=0:
            # Locate the sampled reactive-impedance zero by linear interpolation.
            x0,x1 = -2*math.pi*f[i]*L[i],-2*math.pi*f[i+1]*L[i+1]
            crossings.append(f[i]+(f[i+1]-f[i])*(-x0)/(x1-x0))
    return dict(Q_peak=q[peak],peak_frequency_Hz=f[peak],
                self_resonance_Hz=crossings[0] if crossings else None,
                self_resonance_brackets_Hz=[(f[i],f[i+1]) for i in range(len(f)-1) if L[i]>0>=L[i+1]])


def stranded_greenhouse_fa(fixture, strands=16):
    """Refine Greenhouse strips into parallel sub-strips; minimize magnetic energy.

    Ruehli's partial-element energy is W=I.T@Lp@I/2. Each straight physical
    segment carries unit total current: C.T@I=1. Stationarity gives
    I=Lp^-1 C (C.T Lp^-1 C)^-1 1. This admits transverse PEC proximity
    redistribution while retaining independent finite straight-segment kernels.
    Square vias remain single capacity-radius segments because h=3 um and
    their entire inductance is negligible on the nanohenry coil scale.
    No runtime mesh, operator, current or fitted numerical coefficient enters.
    """
    segments=(*fixture.segments,fixture.port_segment)
    assert fixture.port_segment is not None
    filaments,owners=[],[]
    for owner,(a,b,width,thickness) in enumerate(segments):
        axis=next(i for i in range(3) if a[i]!=b[i])
        count=1 if axis==2 else strands
        for j in range(count):
            start,stop=list(a),list(b)
            if axis!=2:
                shift=width*((j+.5)/count-.5)
                start[1-axis]+=shift
                stop[1-axis]+=shift
            filaments.append((tuple(start),tuple(stop),width/count,thickness))
            owners.append(owner)
    count=len(filaments)
    matrix=np.zeros((count,count))
    # Each off-diagonal entry is the signed strip-GMD mutual term.
    for i,a in enumerate(filaments):
        matrix[i,i]=partial_inductance((a,),order=16)["total_H"]
        for j,b in enumerate(filaments[:i]):
            result=partial_inductance((a,b),order=16)
            matrix[i,j]=matrix[j,i]=(result["positive_mutual_H"]+result["negative_mutual_H"])/2
    incidence=np.eye(len(segments))[owners]
    inverse=np.linalg.solve(matrix,incidence)
    multipliers=np.linalg.solve(incidence.T@inverse,np.ones(len(segments)))
    currents=inverse@multipliers
    error=float(np.max(np.abs(incidence.T@currents-1)))
    return dict(total_H=float(currents@matrix@currents),strands=strands,
                constraint_error=error,
                minimum_partial_eigenvalue_H=float(np.linalg.eigvalsh(matrix)[0]))
