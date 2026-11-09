"""CPU AHK-family transverse-memory diffusion probe; no passivity is presumed.

The sole discretization is the B0 H-before-E map with a backward-Euler
transverse Laplacian on each fresh normal memory. nu=c0*hy/2, eta=0.
No parameter search, stronger scalar CFS, or previously rejected closure runs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
from scipy import linalg
from scipy.linalg import solve_banded

from model import (Block, Cell, C0, DT, EPS0, MU0, native_matrix,
                   oscillatory_spectrum, transverse_memory_resolvent)
from energy import work_balance
from reflection import layer_profiles, reflection, transfer

ROOT = Path(__file__).resolve().parent
LIMIT = 1+1e-12


def diffusivity(cell):
    return .5*C0*cell.period/cell.ny


def block(cell, **kwargs):
    return Block(cell, memory_diffusivity=diffusivity(cell), **kwargs)


def spectrum(b):
    idx = b.active_indices("current")
    return oscillatory_spectrum(b.matrix()[np.ix_(idx,idx)],b.cell.dt)


def geometry(job):
    wall,layers,refinement = job
    rows = []
    base = Cell(wall=wall,layers=layers,theta=35.,
                dy=.0125 if refinement=="y_half" else .025,
                dz=.0125 if refinement=="z_half" else .025)
    for pol,q,factor in product(("TE","TM"),(1,2,3),(1.,.5,.25)):
        for preset in ("real","cfs","alpha0"):
            cell = replace(base,polarization=pol,order=q,dt=DT*factor,preset=preset)
            rows.append(dict(cell=asdict(cell),refinement=refinement,result=spectrum(block(cell))))
    return rows


def fixed(job):
    fmin,angle = job
    rows = []
    for pol,q,factor in product(("TE","TM"),(0,1,2,3),(1.,.5,.25)):
        cell = Cell(polarization=pol,order=q,dt=DT*factor,preset="cfs",theta=angle,f_min=fmin)
        ky = 2*np.pi*q/cell.period+2*np.pi*fmin/C0*np.sin(np.deg2rad(angle))
        hy = cell.period/cell.ny
        rows.append(dict(cell=asdict(cell),physical_ky=ky,
            result=spectrum(block(cell,wavenumber=2/hy*np.sin(ky*hy/2)))))
    return rows


def summarize_rows(rows):
    worst = max(rows,key=lambda r:r["result"]["rho"])
    return dict(cases=len(rows),failed=sum(r["result"]["rho"]>LIMIT for r in rows),
        all_eigenvalue_failed=sum(r["result"]["all_rho"]>LIMIT for r in rows),
        worst=worst,largest_all_rho=max(r["result"]["all_rho"] for r in rows),
        worst_ten=sorted(rows,key=lambda r:r["result"]["rho"],reverse=True)[:10])


def sweep(workers):
    start = time.monotonic()
    cart,floquet = [],[]
    jobs = list(product(("periodic","pec","pmc"),(6,12,24),("base","y_half","z_half")))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for job,rows in zip(jobs,pool.map(geometry,jobs)):
            cart.extend(rows)
            print(json.dumps(dict(geometry=job,cases=len(cart),seconds=time.monotonic()-start)),flush=True)
        for rows in pool.map(fixed,product((.3e9,2e9),(0.,10.,30.,45.,60.,75.))):
            floquet.extend(rows)
    assert len(cart)==1458 and len(floquet)==288
    raw = Path('/tmp/cpml-es-sweep-cases.json')
    raw.write_text(json.dumps(dict(cartesian=cart,fixed_k=floquet),separators=(',',':'))+'\n')
    return dict(cartesian=summarize_rows(cart),fixed_k=summarize_rows(floquet),threshold=LIMIT,
        raw_path=str(raw),raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        seconds=time.monotonic()-start,complete=True)


def reflection_row(f,theta,preset,evanescent):
    # Reuse the exact B0 one-sided stack and its discrete mode decomposition.
    if theta==0 and not evanescent:
        return reflection(f,theta,'current',preset=preset,evanescent=evanescent)
    h = C0/(20*3e9)
    dt = h/(2*C0*np.sqrt(2))
    nodes,profiles = layer_profiles(h,dt,preset,theta)
    omega = 2*np.pi*f
    lam = np.exp(-1j*omega*dt)
    c0 = 1/np.sqrt(EPS0*MU0)
    ky = omega/c0*(1.25 if evanescent else np.sin(np.deg2rad(theta)))
    k = 2/h*np.sin(ky*h/2)
    denominator = 1+.5*C0*h*dt*k*k
    # Actual fresh-memory transfer after the implicit diffusion solve.
    modified = [(kap,b/denominator,c/denominator) for kap,b,c in profiles]
    ge,gh = [transfer(p,lam,dt,'current') for p in modified]
    K2 = (2*np.sin(omega*dt/2)/(c0*dt))**2-k*k
    kz = 2/h*np.arcsin(np.sqrt(complex(K2))*h/2)
    lo,hi = ge[1:-1]*gh[:-1]/h**2,ge[1:-1]*gh[1:]/h**2
    band = np.zeros((3,len(lo)),complex)
    band[0,1:],band[1],band[2,:-1] = hi[:-1],-lo-hi+K2,lo[1:]
    rhs = np.zeros(len(lo),complex)
    rhs[0] = -lo[0]
    ratio = solve_banded((1,1),band,rhs,check_finite=False)[0]
    r = (np.exp(1j*kz*h)-ratio)/(ratio-np.exp(-1j*kz*h))
    return float(abs(r*np.exp(2j*kz*nodes[0])))


def reflections():
    rows = []
    angles = dict(normal=0.,**{'60deg':60.,'80deg':80.,'evanescent':0.})
    for preset,metric,f in product(('real','cfs','alpha0'),angles,(.3,.6,1.,2.,3.)):
        ev = metric=='evanescent'
        theta = angles[metric]
        current = reflection(f*1e9,theta,'current',preset=preset,evanescent=ev)
        value = reflection_row(f*1e9,theta,preset,ev)
        passed = value<current if ev else value<=current*(2 if metric=='80deg' else 1)
        rows.append(dict(preset=preset,metric=metric,frequency_GHz=f,
            polarizations=['TE','TM'],current_R=current,R=value,ratio=value/current,passes=bool(passed)))
    metrics = {m:dict(max_ratio=max(r['ratio'] for r in rows if r['metric']==m),
        failed_rows=sum(not r['passes'] for r in rows if r['metric']==m)) for m in angles}
    return dict(layers=12,metrics=metrics,rows=rows,passes=all(r['passes'] for r in rows),
        limits=dict(normal=1,**{'60deg':1,'80deg':2,'evanescent':'strictly < 1'}))


def time_run(b,steps,orders=None):
    if orders is None:
        x = np.random.default_rng(431).normal(size=b.size)
    else:
        x = np.random.default_rng(431).normal(size=(b.size,len(orders)))
    x[b.offsets[3]:] = 0
    x /= np.linalg.norm(x)
    total,checkpoints = 0.,[]
    start = time.monotonic()
    for step in range(1,steps+1):
        x = b.step(x)
        if step%100==0:
            norm = np.linalg.norm(x)
            total += np.log(norm)
            x /= norm
            if step>=steps//2:
                checkpoints.append((step,total))
        if step%20000==0:
            print(json.dumps(dict(pol=b.cell.polarization,steps=step,
                seconds=time.monotonic()-start)),flush=True)
    fit = np.polyfit(np.asarray(checkpoints)[:,0],np.asarray(checkpoints)[:,1],1)
    rho = float(np.exp(fit[0]))
    return dict(steps=steps,fitted_rho=rho,passes=rho<=LIMIT,seed=431,
        fit_window=[steps//2,steps],rescale_interval=100,
        fit_first=checkpoints[0],fit_last=checkpoints[-1],
        dominant_order=None if orders is None else int(orders[np.argmax(np.linalg.norm(x,axis=0))]),
        seconds=time.monotonic()-start)


def power_job(pol):
    cell = Cell(polarization=pol,period=4*1.28)
    b = block(cell)
    orders = np.arange(cell.ny//2+1)
    b.k = 2*np.sin(np.pi*orders/cell.ny)/(cell.period/cell.ny)
    return dict(polarization=pol,period_m=cell.period,ny=cell.ny,harmonics=len(orders),
        **time_run(b,100000,orders))


def g2b():
    # B0 geometry, explicit requested eps=4 slab and carrier k_parallel=.55 k0.
    # f0 is fixed at 2 GHz, matching the existing upper fixed-k carrier sweep.
    f0 = 2e9
    cell = Cell(polarization='TM',order=0,slab_epsilon=4.,preset='cfs',
                f_min=f0,theta=float(np.rad2deg(np.arcsin(.55))))
    ky = .55*2*np.pi*f0/C0
    hy = cell.period/cell.ny
    b = block(cell,wavenumber=2/hy*np.sin(ky*hy/2))
    return dict(cell=asdict(cell),f0_Hz=f0,physical_k_parallel=ky,k_h=b.k,
        slab_thickness_m=.1,spectrum=spectrum(b),time=time_run(b,320000),
        scope='B0 transverse-mean eps=4 slab, fixed k; source-free norm fit, no r/t qualification')


def exact_uniform():
    """Exact rational discrete storage at one homogeneous Fourier symbol.

    This is a consistency certificate, not a piecewise-layer or graded proof.
    Three fields (u,v,w), two memories (p,q); all quantities dimensionless.
    """
    import sympy as s
    u,v,w,p,q = s.symbols('u v w p q')
    x = s.Matrix([u,v,w,p,q])
    t,k,d,b,c,den = s.Rational(1,4),s.Rational(1,2),s.Rational(1,2),s.Rational(4,5),-s.Rational(1,10),s.Rational(17,16)
    qn = (b*q+c*d*u)/den
    wn = w+t*(d*u+k*v+qn)
    pn = (b*p-c*d*wn)/den
    un = u+t*(-d*wn+pn)
    vn = v-t*k*wn
    A = s.Matrix([un,vn,wn,pn,qn]).jacobian(x)
    r = (A-s.eye(5)).nullspace()[0]
    Q = s.eye(5)-r*r.T/(r.dot(r))
    variables = s.symbols('p0:15')
    P = s.zeros(5)
    it = iter(variables)
    for i in range(5):
        for j in range(i,5):
            P[i,j]=P[j,i]=next(it)
    eq = A.T*P*A-P+Q
    sol = s.solve([eq[i,j] for i in range(5) for j in range(i,5)]+[(r.T*P*r)[0]-1],variables)
    P = P.subs(sol)
    residual = A.T*P*A-P+Q
    minors = [s.factor(P[:i,:i].det()) for i in range(1,6)]
    assert residual==s.zeros(5) and all(m>0 for m in minors)
    # Independent symbolic memory work identity, including the diffusion term.
    old,drive,bb,cc,dd = s.symbols('old drive b c delta',real=True)
    new = (bb*old+cc*drive)/(1+dd)
    balance = s.factor(new**2-old**2-((bb**2-1)*old**2+2*bb*cc*old*drive+cc**2*drive**2-(2*dd+dd**2)*new**2))
    assert balance==0
    return dict(A=[[str(v) for v in row] for row in A.tolist()],
        P=[[str(v) for v in row] for row in P.tolist()],
        Q=[[str(v) for v in row] for row in Q.tolist()],
        positive_leading_minors=[str(v) for v in minors],
        identity='A.T*P*A-P=-Q exactly; Q=I-r*r.T/(r.T*r) >= 0',
        exact_residual_zero=True,memory_identity_zero=True,
        scope='single homogeneous Fourier symbol only; not an R0 exit certificate')


def exact_piecewise():
    """Exact rational countercertificate for a piecewise-constant layer.

    Time is normalized to dt=1, c=1/4; h=hy=1/2 and nu=c*hy/2=1/16.
    All physical coefficients and integer-state stepping use exact arithmetic.
    """
    import sympy as s
    cell = Cell(dz=.5,dy=.5,zhalf=.5,period=2.,layers=1,slab=False)
    profiles = []
    for length in (5,4):
        bb,cc = np.ones(length),np.zeros(length)
        ids = [1,-2] if length==5 else [0,-1]
        bb[ids],cc[ids] = .8,-.1
        profiles.append((np.ones(length),bb,cc))
    b = block(cell,material='vacuum',coefficients=profiles,wavenumber=.5)
    b.a[:],b.av[1:-1] = .25,.25
    b.cell = replace(cell,dt=s.Integer(1))
    b.t,b.h,b.k,b.memory_diffusivity = s.Rational(1,4),s.Rational(1,2),s.Rational(1,2),s.Rational(1,16)
    for name in ('ke','kh','be','bh','ce','ch','a','av'):
        setattr(b,name,np.array([s.Rational(float(v)).limit_denominator(100) for v in getattr(b,name)],dtype=object))
    active = b.active_indices('current')
    full = b.step(np.eye(b.size,dtype=int).astype(object),ordering=(1,1))
    A = s.Matrix(full[np.ix_(active,active)].tolist())
    assert not A.atoms(s.Float)
    z,ss = s.symbols('z s')
    factors = s.factor_list(A.charpoly(z).as_expr())[1]
    polynomial = s.prod(f**m for f,m in factors if f!=z-1)
    degree = s.degree(polynomial,z)
    transformed = s.Poly(s.cancel((1-ss)**degree*polynomial.subs(z,(1+ss)/(1-ss))),ss)
    coefficients = transformed.all_coeffs()
    width = (len(coefficients)+1)//2
    table = [list(coefficients[::2])+[0]*(width-len(coefficients[::2])),
             list(coefficients[1::2])+[0]*(width-len(coefficients[1::2]))]
    for _ in range(len(coefficients)-2):
        a,c = table[-2:]
        assert c[0]!=0
        row = [s.factor((c[0]*a[j+1]-a[0]*c[j+1])/c[0]) for j in range(width-1)]+[0]
        assert any(v!=0 for v in row)
        table.append(row)
    signs = [int(s.sign(row[0])) for row in table]
    changes = sum(a!=c for a,c in zip(signs,signs[1:]))
    assert changes>0
    Af = np.array(A,float)
    vals,vecs = linalg.eig(Af,check_finite=False)
    i = np.argmax(abs(vals))
    return dict(state_count=A.rows,active_indices=active.tolist(),
        matrix=[[str(v) for v in row] for row in A.tolist()],
        characteristic_factors=[dict(polynomial=str(f),multiplicity=m) for f,m in factors],
        bilinear_polynomial=str(transformed.as_expr()),routh_first_column_signs=signs,
        exact_outside_unit_disk_count=changes,rho=float(abs(vals[i])),
        eigen_residual=float(np.linalg.norm(Af@vecs[:,i]-vals[i]*vecs[:,i])),
        forbids_positive_definite_storage=True,
        geometry='four z intervals; outer half-cell memories and interface-node memories active; eps_t=4 on all three interior nodes, eps_n=(1,4,4,1)',
        parameters=dict(dt='1',c='1/4',h='1/2',hy='1/2',k_h='1/2',nu='1/16',b='4/5',c_memory='-1/10',kappa='1'))


def verify():
    report = dict(exact_uniform=exact_uniform(),exact_piecewise=exact_piecewise(),rows=[])
    rng = np.random.default_rng(431)
    for pol,wall in product(('TE','TM'),('periodic','pec','pmc')):
        cell = Cell(polarization=pol,wall=wall)
        b = block(cell)
        den = 1+diffusivity(cell)*cell.dt*b.k*b.k
        equivalent = [(kap,np.where(cc!=0,bb/den,bb),cc/den) for kap,bb,cc in b.profile]
        other = Block(cell,coefficients=equivalent)
        active = b.active_indices('current')
        A = b.matrix()[np.ix_(active,active)]
        values,vectors = linalg.eig(A,check_finite=False)
        j = np.argmax(abs(values))
        eigen_res = float(np.linalg.norm(A@vectors[:,j]-values[j]*vectors[:,j]))
        residuals = []
        for _ in range(16):
            x = np.zeros(b.size,dtype=complex)
            x[active] = rng.normal(size=len(active))+1j*rng.normal(size=len(active))
            residuals.append(abs(work_balance(b,x)['relative_residual']))
        native = native_matrix(cell,rounded=False,memory_diffusivity=diffusivity(cell))
        equivalent_native = native_matrix(cell,rounded=False,coefficients=tuple(
            tuple(v.astype(np.float32) for v in group) for group in equivalent))
        # The latter rounds altered coefficients, so it is a loose replay check.
        ni = transverse_memory_resolvent(cell,cell.names[3],diffusivity(cell))
        basis_y = np.arange(len(ni))+(0 if pol=='TE' else .5)
        phase = (2 if wall=='periodic' else 1)*np.pi*cell.order*basis_y/cell.ny
        sine = pol=='TE'
        if wall=='pmc':
            sine = not sine
        wave = np.cos(phase) if wall=='periodic' or not sine else np.sin(phase)
        diff_res = float(np.linalg.norm(ni@wave-wave/den)/np.linalg.norm(wave))
        error = float(np.max(abs(b.matrix()-other.matrix())))
        ns = oscillatory_spectrum(native,cell.dt)
        report['rows'].append(dict(polarization=pol,wall=wall,
            harmonic_transfer_entry_error=error,transverse_resolvent_residual=diff_res,
            native_equivalent_entry_error=float(np.max(abs(native-equivalent_native))),
            native_spectrum=ns,all_rho=float(abs(values[j])),eigen_residual=eigen_res,
            forbids_positive_definite_storage=bool(abs(values[j])>LIMIT),
            work_identity_max_residual=max(residuals)))
        assert error<1e-12 and diff_res<1e-12 and max(residuals)<1e-12
        assert np.max(abs(native-equivalent_native))<2e-6
    # Independent wave elimination from actual amplification eigenvectors.
    wave_rows = []
    for pol,preset in product(('TE','TM'),('real','cfs')):
        b = block(Cell(polarization=pol,preset=preset,theta=35.),material='vacuum')
        vals,vecs = linalg.eig(b.matrix(),check_finite=False)
        eligible = (abs(np.angle(vals))>2*np.pi*b.cell.dt*1e6)&(np.linalg.norm(vecs[:b.n-1],axis=0)>.01)
        i = np.argmax(np.where(eligible,abs(vals),0))
        lam,u = vals[i],vecs[:b.n-1,i]
        den = 1+b.memory_diffusivity*b.cell.dt*b.k*b.k
        ge,gh = [transfer((kap,bb/den,cc/den),lam,b.cell.dt,'current') for kap,bb,cc in b.profile]
        K2 = -(lam-1)**2/(b.t*b.t*lam)-b.k*b.k
        L = ge[1:-1,None]*(b.B@(gh[:,None]*b.F))+K2*np.eye(b.n-1)
        residual = float(np.linalg.norm(L@u)/(np.linalg.norm(L)*np.linalg.norm(u)))
        assert residual<1e-11
        wave_rows.append(dict(polarization=pol,preset=preset,residual=residual))
    report['wave_elimination'] = wave_rows
    cell = Cell(polarization='TM',order=0,slab_epsilon=4.,preset='cfs',f_min=2e9,
                theta=float(np.rad2deg(np.arcsin(.55))))
    ky = .55*2*np.pi*cell.f_min/C0
    hy = cell.period/cell.ny
    kh = 2/hy*np.sin(ky*hy/2)
    b = block(cell,wavenumber=kh)
    native = native_matrix(cell,rounded=False,memory_diffusivity=diffusivity(cell),physical_wavenumber=ky)
    den = 1+diffusivity(cell)*cell.dt*kh*kh
    from model import profiles
    equivalent = tuple((kap,np.where(cc!=0,bb/den,bb).astype(np.float32),(cc/den).astype(np.float32))
                       for kap,bb,cc in profiles(cell,native=True))
    other = native_matrix(cell,rounded=False,coefficients=equivalent,physical_wavenumber=ky)
    error = float(np.max(abs(native-other)))
    assert error<2e-6
    native_spec,harmonic_spec = oscillatory_spectrum(native,cell.dt),spectrum(b)
    assert abs(native_spec['rho']-harmonic_spec['rho'])<1e-6
    report['complex_fixed_k_native'] = dict(physical_ky=ky,k_h=kh,cell=asdict(cell),
        native_equivalent_entry_error=error,native_spectrum=native_spec,
        harmonic_spectrum=harmonic_spec,scope='complex Bloch stencil projection with binary32 input constants')
    report['passed_algebra_checks'] = True
    return report


def provenance():
    paths = [ROOT/n for n in ('energy_stable.py','model.py','sweep.py','bloch_sweep.py','reflection.py','energy.py','power.py')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def summary():
    inputs = {name:json.loads((ROOT/'results'/f'es_{name}.json').read_text())
              for name in ('verify','sweep','power','g2b','reflection','native_regression')}
    spectral = inputs['sweep']
    assert spectral['complete'] and spectral['cartesian']['cases']==1458 and spectral['fixed_k']['cases']==288
    assert inputs['verify']['passed_algebra_checks'] and inputs['native_regression']['passed']
    assert {r['polarization'] for r in inputs['power']['results']}=={'TE','TM'}
    assert all(r['steps']==100000 for r in inputs['power']['results'])
    assert inputs['g2b']['time']['steps']==320000
    assert inputs['reflection']['limits']==dict(normal=1,**{'60deg':1,'80deg':2,'evanescent':'strictly < 1'})
    gates = dict(discrete_storage=not inputs['verify']['exact_piecewise']['forbids_positive_definite_storage'],
        cartesian=spectral['cartesian']['failed']==0,fixed_k=spectral['fixed_k']['failed']==0,
        large_cell=all(r['passes'] for r in inputs['power']['results']),
        g2b=inputs['g2b']['spectrum']['all_rho']<=LIMIT and inputs['g2b']['time']['passes'],
        reflection=inputs['reflection']['passes'])
    return dict(recommendation='GO' if all(gates.values()) else 'NO-GO',gates=gates,
        r0='rejected at piecewise-constant discrete storage gate',
        r1='completed diagnostic gates for the rejected transplant; no qualified R1 candidate',
        selected_for_r2=[],scope='one AHK-family IMEX discretization; no rejection of the entire family',
        sources=dict(AHK='https://kth.diva-portal.org/smash/get/diva2:12464/FULLTEXT01',
                     Duru='https://arxiv.org/pdf/1405.0536'),
        reports_sha256={name:hashlib.sha256((ROOT/'results'/f'es_{name}.json').read_bytes()).hexdigest() for name in inputs})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify','sweep','reflection','power','g2b','summary'))
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    run_hashes = provenance()
    if args.action=='verify':
        report = verify()
    elif args.action=='sweep':
        report = sweep(args.workers)
    elif args.action=='reflection':
        report = reflections()
    elif args.action=='g2b':
        report = g2b()
    elif args.action=='summary':
        report = summary()
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers,2)) as pool:
            report = dict(results=list(pool.map(power_job,('TE','TM'))))
    report.update(candidate='AHK-family IMEX transverse-memory diffusion, nu=c0*hy/2, eta=0',
        input_sha256=run_hashes,cpu_only=True)
    path = ROOT/'results'/f'es_{args.action}.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(str(path),flush=True)


if __name__=='__main__':
    main()
