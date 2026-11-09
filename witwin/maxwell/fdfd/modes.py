"""Native CUDA transverse Yee eigenmodes with rectangular PEC walls.

Independent first-order Maxwell elimination for exp(i beta w - i omega t).
The transverse aperture may have graded Cartesian cells and diagonal epsilon.
Dense eigensolves have an explicit size cap; no CPU fallback is provided.
"""
from __future__ import annotations
import math
import torch
from ..constants import ETA_0
from ..result import Result


def _differences(nodes):
    spacing = nodes.diff()
    cells = spacing.numel()
    incidence = torch.zeros((cells-1,cells),device=nodes.device,dtype=torch.float64)
    row = torch.arange(cells-1,device=nodes.device)
    incidence[row,row] = -1
    incidence[row,row+1] = 1
    dual = (spacing[:-1]+spacing[1:])*.5
    return incidence/dual[:,None], -incidence.T/spacing[:,None]


@torch.no_grad()
def solve_modes(prepared, *, axis='z', position=0., bounds=None, num_modes=1,
                max_dofs=2048, tolerance=1e-8):
    """Return a Result containing beta and six staggered cross-section profiles.

    Bounds are two wall-to-wall intervals in cyclic transverse axis order:
    (x,y) for z, (y,z) for x, (z,x) for y. Walls impose zero tangential E.
    Profiles have leading mode dimension and native transverse Yee dimensions.
    Fixed PEC inside the aperture and trainable mode solves are unsupported.
    """
    if axis not in 'xyz' or len(axis) != 1 or not math.isfinite(position):
        raise ValueError('Mode axis and position must identify a finite transverse plane.')
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (num_modes,max_dofs)):
        raise ValueError('Mode count and matrix size limit must be positive integers.')
    if not math.isfinite(tolerance) or not 0<tolerance<1:
        raise ValueError('Mode tolerance must be between zero and one.')
    if prepared.simulation.has_trainable_parameters:
        raise ValueError('Transverse eigenmodes currently require fixed material inputs.')
    solver = prepared.solver
    scene = solver.scene
    normal = 'xyz'.index(axis)
    order = ((normal+1)%3,(normal+2)%3,normal)
    names = tuple('xyz'[i] for i in order)
    if bounds is None:
        bounds = tuple(scene.domain.bounds[i] for i in order[:2])
    if len(bounds)!=2 or any(len(pair)!=2 or not all(math.isfinite(v) for v in pair) or pair[0]>=pair[1] for pair in bounds):
        raise ValueError('Mode bounds need two finite increasing intervals.')
    if not scene.domain.bounds[normal][0] <= position <= scene.domain.bounds[normal][1]:
        raise ValueError('Mode plane must lie inside the physical domain.')
    selections,coordinates = [],[]
    for name,pair,index in zip(names[:2],bounds,order[:2]):
        if pair[0]<scene.domain.bounds[index][0] or pair[1]>scene.domain.bounds[index][1]:
            raise ValueError('Mode aperture must lie inside the physical domain.')
        nodes = torch.tensor(getattr(scene,name+'_nodes64'),device=scene.device)
        lo,hi = (int((nodes-value).abs().argmin()) for value in pair)
        if hi-lo<2:
            raise ValueError('A mode aperture needs at least two cells per transverse axis.')
        selections.append(slice(lo,hi+1))
        coordinates.append(nodes[lo:hi+1])
    plane = int((getattr(scene,axis)-position).abs().argmin())
    nu,nv = (nodes.numel()-1 for nodes in coordinates)
    a,b = nu*(nv-1),(nu-1)*nv
    count = a+b
    if count>max_dofs or num_modes>count:
        raise ValueError(f'Transverse mode matrix needs {count} unknowns; max_dofs={max_dofs}, num_modes={num_modes}.')
    # Restrict material nodes before constructing any dense eigensolver arrays.
    solver._ensure_material_components()
    components = solver.material_eps_components
    if any(bool((mu != 1).any()) for mu in solver.material_mu_components.values()):
        raise NotImplementedError('Transverse eigenmodes currently require vacuum permeability.')
    epsilon = [components[name].detach().permute(order)
               [selections[0],selections[1],plane].to(torch.complex128) for name in names]
    from ..compiler.materials import _pec_edge_fill
    fill = _pec_edge_fill(scene)
    if fill is not None and any(bool((value>0).any()) for value in fill.values()):
        raise ValueError('Transverse mode solve supports aperture PEC walls, not in-domain PEC geometry.')
    eps_u = ((epsilon[0][:-1]+epsilon[0][1:])*.5)[:,1:-1].flatten()
    eps_v = ((epsilon[1][:,:-1]+epsilon[1][:,1:])*.5)[1:-1].flatten()
    eps_w = epsilon[2][1:-1,1:-1].flatten()
    if bool((eps_w.abs()<1e-14).any()):
        raise ValueError('Longitudinal epsilon must be nonzero for transverse elimination.')
    gu,su = _differences(coordinates[0])
    gv,sv = _differences(coordinates[1])
    eye = lambda n: torch.eye(n,device=scene.device,dtype=torch.float64)
    kron = lambda x,y: torch.kron(x.contiguous(),y.contiguous()).to(torch.complex128)
    du_ez,dv_ez = kron(su,eye(nv-1)),kron(eye(nu-1),sv)
    du_hy,dv_hx = kron(gu,eye(nv-1)),kron(eye(nu-1),gv)
    du_ey,dv_ex = kron(su,eye(nv)),kron(eye(nu),sv)
    du_hz,dv_hz = kron(gu,eye(nv)),kron(eye(nu),gv)
    curl_h = torch.cat((-dv_hx,du_hy),dim=1)
    curl_e = torch.cat((-dv_ex,du_ey),dim=1)
    k0 = solver.k0
    p = torch.cat((du_ez@(curl_h/eps_w[:,None]),dv_ez@(curl_h/eps_w[:,None])),dim=0)/k0
    p[:a,b:] += k0*eye(a)
    p[a:,:b] -= k0*eye(b)
    q = -torch.cat((du_hz@curl_e,dv_hz@curl_e),dim=0)/k0
    q[:b,a:] -= k0*torch.diag(eps_v)
    q[b:,:a] += k0*torch.diag(eps_u)
    operator = p@q
    eigenvalues,eigenvectors = torch.linalg.eig(operator)
    selected = torch.argsort(eigenvalues.real,descending=True)[:num_modes]
    values,electric = eigenvalues[selected],eigenvectors[:,selected]
    beta = torch.sqrt(values)
    # The passive outgoing branch has positive attenuation.
    beta = torch.where(beta.imag < -1e-10*beta.abs(),-beta,beta)
    if bool((beta.abs()<1e-12*k0).any()):
        raise RuntimeError('Requested mode is at cutoff; longitudinal recovery is singular.')
    residual = torch.linalg.vector_norm(operator@electric-electric*values,dim=0)/(
        torch.linalg.matrix_norm(operator)*torch.linalg.vector_norm(electric,dim=0)).clamp_min(1e-30)
    if not bool(torch.isfinite(beta).all()) or bool((residual>tolerance).any()):
        raise RuntimeError('Transverse eigensolve failed its residual check.')
    peak = electric.abs().argmax(dim=0)
    electric = electric/electric[peak,torch.arange(num_modes,device=scene.device)]
    magnetic = (q@electric)/beta[None,:]  # eta0 * H
    ew = 1j*(curl_h@magnetic)/(k0*eps_w[:,None])
    hw = (curl_e@electric)/(1j*k0*ETA_0)
    shapes = ((nu,nv-1),(nu-1,nv),(nu-1,nv-1),(nu-1,nv),(nu,nv-1),(nu,nv))
    arrays = (electric[:a],electric[a:],ew,magnetic[:b]/ETA_0,magnetic[b:]/ETA_0,hw)
    profiles = {kind+name:value.T.reshape(num_modes,*shape).contiguous()
                for kind,name,value,shape in zip(('E',)*3+('H',)*3,names+names,arrays,shapes)}
    return Result(method='fdfd',scene=prepared.simulation.scene,prepared_scene=scene,frequency=solver.frequency,
                  monitors={'modes':{'beta':beta,'effective_index':beta/k0,'eigen_residual':residual}},
                  raw_output={'profiles':profiles,'coordinates':dict(zip(names[:2],coordinates)),
                              'axis':axis,'position':float(getattr(scene,axis)[plane]),
                              'normalization':'peak transverse electric component = 1 V/m'},
                  solver_stats={'converged':True,'mode_dofs':count,'device':str(scene.x.device),
                                'final_residual':float(residual.max())})
