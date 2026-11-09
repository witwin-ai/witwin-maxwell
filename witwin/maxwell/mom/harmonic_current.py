"""Eliminated isolated-body DC current equations for internal harmonic balance."""
import torch
from .static_current import FaceCurrentBalance


class HarmonicCurrentClosure:
    """Solve each isolated body's current potential at the actual AC iterate.

    All supplied bodies use current balance; a mixture with dielectric DC
    displacement blocks requires separate global static coupling. Conductivity
    is the physical DC sample, never a numerical regularization. TPA coefficients
    are supplied by the HB model so the constitutive law is included once.
    """
    formulation='isolated_face_current_balance'
    relative_tolerance_default = 1e-10
    absolute_tolerance_default = 1e-16

    @property
    def volumes(self):
        return tuple(block.volume for block in self.blocks)

    def __init__(self,volumes,conductivities,*,relative_tolerance=relative_tolerance_default,
                 absolute_tolerance=absolute_tolerance_default,tangent_rule=None,tangent_column_batch=8):
        if tangent_rule is not None:
            from .linalg import StoppingRule
            if (type(tangent_rule) is not StoppingRule or tangent_rule.method!='gmres'
                    or tangent_rule.preconditioner!='none' or tangent_rule.recycle_size or tangent_rule.fallback!='none'):
                raise ValueError('DC closure requires unpreconditioned unrecycled GMRES without fallback')
        if type(tangent_column_batch) is not int or tangent_column_batch<1:
            raise ValueError('DC closure tangent column batch must be a positive integer')
        if tangent_rule is None and tangent_column_batch!=8:
            raise ValueError('DC closure column batch requires an active tangent rule')
        self.blocks = tuple(FaceCurrentBalance(volume) for volume in volumes)
        if not self.blocks or len(self.blocks) != len(conductivities):
            raise ValueError('DC closure requires one conductivity per volume')
        reference = self.blocks[0].volume.vertices
        if any(b.volume.vertices.device != reference.device or b.volume.vertices.dtype != reference.dtype
               for b in self.blocks):
            raise ValueError('DC closure volumes must share device and precision')
        if len({b.volume.region_id for b in self.blocks}) != len(self.blocks):
            raise ValueError('DC closure volume regions must be unique')
        self.conductivities = tuple(b._conductivity(s).clone()
                                   for b,s in zip(self.blocks,conductivities))
        self.counts = tuple(len(b.volume.cells) for b in self.blocks)
        self.relative_tolerance,self.absolute_tolerance = relative_tolerance,absolute_tolerance
        self.tangent_rule,self.tangent_column_batch=tangent_rule,tangent_column_batch
        self._versions = self._state()

    def _state(self):
        return (tuple(id(b) for b in self.blocks),self.counts,self.relative_tolerance,self.absolute_tolerance,
                tuple((id(s),s._version) for s in self.conductivities),self.tangent_rule,self.tangent_column_batch)

    def numerical_policy(self):
        """Canonical active inner policy, excluding inactive default options."""
        self.assert_current()
        if self.tangent_rule is None:return {}
        from dataclasses import asdict
        return {'tangent_rule':asdict(self.tangent_rule),'tangent_column_batch':self.tangent_column_batch}

    def validate_receipt(self,record,block):
        """Check historical inner policy and bounds without repeating solves."""
        self.assert_current()
        validate_current_receipt(record,rule=self.tangent_rule,column_batch=self.tangent_column_batch,
            size=block.n_dofs-1,dtype=block.volume.vertices.dtype)

    def assert_current(self):
        if self._versions != self._state():
            raise RuntimeError('Harmonic DC current closure was mutated')
        for block in self.blocks:
            block._assert_current()

    def ohmic_current(self,electric):
        """Apply each owned scalar/tensor DC law in its volume cell order."""
        self.assert_current()
        self.blocks[0]._real(electric,(sum(self.counts),3),'DC electric field')
        return torch.cat([torch.einsum('tij,tj->ti',sigma,field) if sigma.ndim==3
                          else sigma[:,None]*field
                          for sigma,field in zip(self.conductivities,electric.split(self.counts))])

    def solve(self,ac_electric,tpa_scale):
        self.assert_current()
        if (not isinstance(ac_electric,torch.Tensor) or ac_electric.ndim != 3
                or ac_electric.shape[1:] != (sum(self.counts),3)):
            raise ValueError('DC closure AC fields must match the volume cell ordering')
        if isinstance(tpa_scale,torch.Tensor) and tpa_scale.ndim:
            if tpa_scale.shape != (sum(self.counts),):
                raise ValueError('DC closure TPA coefficients must match the volume cell ordering')
            coefficients = tpa_scale.split(self.counts)
        else:
            coefficients = (tpa_scale,)*len(self.blocks)
        return tuple(block.solve(ac,conductivity=sigma,tpa_scale=tpa,
                                 relative_tolerance=self.relative_tolerance,absolute_tolerance=self.absolute_tolerance,
                                 tangent_rule=self.tangent_rule,tangent_column_batch=self.tangent_column_batch)
            for block,ac,sigma,tpa in zip(self.blocks,ac_electric.split(self.counts,dim=1),
                                        self.conductivities,coefficients))

    def electric(self,ac_electric,tpa_scale,*,polarization=None):
        return torch.cat([solution.electric for solution in self.solve(ac_electric,tpa_scale)])


def surface_face_potentials(block,binding,potential):
    """Map volume face DOFs to the exact owning body's surface face order."""
    block._assert_current()
    binding.assert_current()
    if binding.volume is not block.volume:
        raise ValueError('Current potential must use its exact bound volume')
    block._real(potential,(block.n_dofs,),'Face potential')
    volume = block.volume
    body = binding.surface_system.bodies[volume.region_id-1]
    mapping = binding.boundary_to_surface_vertices
    surface_to_volume = torch.empty_like(mapping).index_copy(0,mapping,volume.boundary_vertex_ids)
    surface_faces = surface_to_volume[body.mesh.faces].sort(-1).values
    unique,inverse = torch.unique(torch.cat((block.faces,surface_faces)),dim=0,return_inverse=True)
    if not torch.equal(unique,block.faces):
        raise ValueError('Current boundary faces do not match their owning surface')
    return potential[inverse[len(block.faces):]]

def validate_current_receipt(record,*,rule,column_batch,size,dtype):
    """Validate recorded policy using independently bound dimension and precision."""
    import math
    from dataclasses import asdict
    if not isinstance(record,dict):raise ValueError('Invalid stored DC solver receipt')
    if rule is None:
        if any(k in record for k in ('linear_solver','tangent_rule','tangent_checks')):
            raise ValueError('Stored DC solver policy mismatch')
        return
    iterations=record.get('iterations');checks=record.get('tangent_checks')
    if (record.get('linear_solver')!='streamed_inverse_checked_gmres'
            or record.get('tangent_rule')!=asdict(rule)
            or record.get('tangent_column_batch')!=column_batch
            or record.get('global_current_jacobian_stored') is not False
            or record.get('fast_work_qualified') is not False or record.get('physical_qualification') is not False
            or record.get('converged') is not True or type(iterations) is not int or iterations<0
            or not isinstance(checks,list) or len(checks)!=iterations+2):
        raise ValueError('Stored DC solver policy mismatch')
    for index,entry in enumerate(checks):
        if not isinstance(entry,dict) or entry.get('phase')!=('newton' if index<iterations else 'implicit'):
            raise ValueError('Invalid stored DC tangent phase')
        bound=entry.get('bound');solve=entry.get('solve')
        if (not isinstance(bound,dict) or bound.get('method')!='complete_streamed_inverse_residual'
                or bound.get('complete') is not True or bound.get('admitted') is not True
                or bound.get('inconclusive') is not False or bound.get('solve_failure') is not None
                or bound.get('columns_checked')!=size or bound.get('column_batch')!=column_batch
                or bound.get('tangent_rule')!=asdict(rule)
                or any(bound.get(k) is not False for k in ('underlying_action_error_certified','interval_certified','physical_qualification','fast_work_qualified'))
                or not isinstance(solve,dict) or solve.get('converged') is not True):
            raise ValueError('Invalid stored DC tangent bound')
        lower,upper,cutoff,inverse,residual=(bound.get(k) for k in (
            'sigma_min_lower','sigma_max_upper','numerical_singularity_cutoff','inverse_norm_upper','inverse_residual_norm_upper'))
        if (any(type(v) not in (float,int) or not math.isfinite(v) for v in (lower,upper,cutoff,inverse,residual))
                or not 0<=cutoff<lower<=upper or not inverse>0 or not 0<=residual<1
                or lower!=(1-residual)/inverse
                or cutoff!=64*torch.finfo(dtype).eps*size*upper):
            raise ValueError('Invalid stored DC tangent numerical bound')


def face_potential_fields(block,potential,offset,points):
    """Evaluate retained CR face means in unique tetrahedral cell interiors."""
    volume=block.volume
    values,fields = [],[]
    for batch in points.split(32):
        displacement = batch[:,None,:]-volume.vertices[volume.cells[:,0]][None]
        local = torch.einsum('qti,tai->qta',displacement,-block.gradients[:,1:]/3)
        barycentric = torch.cat((1-local.sum(-1,keepdim=True),local),-1)
        matches = (barycentric >= -128*torch.finfo(points.dtype).eps).all(-1)
        if bool((matches.sum(-1) != 1).any()):
            raise ValueError('DC face-potential electric queries require a unique cell interior; cell-interface traces are unresolved')
        cell = matches.to(torch.int64).argmax(-1)
        bary = barycentric[torch.arange(len(batch),device=points.device),cell]
        nodal = potential[block.cell_dofs[cell]]
        values.append(((1-3*bary)*nodal).sum(-1)+offset)
        fields.append(-torch.einsum('qa,qai->qi',nodal,block.gradients[cell]))
    result = (torch.cat(values),torch.cat(fields)) if len(points) else (points.new_empty(0),points.new_empty(0,3))
    if not all(bool(torch.isfinite(value).all()) for value in result):
        raise ValueError("DC face-potential query is not finite")
    return result
