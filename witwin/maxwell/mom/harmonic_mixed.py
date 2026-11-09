"""Eliminated mixed dielectric/current DC equations in actual HB cell order."""
import torch
from .harmonic_current import HarmonicCurrentClosure,surface_face_potentials
from .static_mixed import MixedStaticOperator
from .static_tensor_floating import StaticTensorFloatingOperator


class MixedHarmonicClosure:
    formulation='mixed_dielectric_face_current_balance'

    def __init__(self,operator,bindings,current_conductivities,*,dc_policy=None,max_response_bytes=None):
        if not isinstance(operator,(MixedStaticOperator,StaticTensorFloatingOperator)):
            raise TypeError('Mixed harmonic closure requires its prepared static operator')
        operator.assert_current()
        self.operator=operator
        tensor=isinstance(operator,StaticTensorFloatingOperator)
        system=operator.system if tensor else operator.laplace.system
        if tensor:self.formulation='mixed_tensor_dielectric_face_current_balance'
        self.bindings=tuple(bindings)
        regions=tuple(binding.volume.region_id for binding in self.bindings)
        if len(set(regions))!=len(regions):
            raise ValueError('Mixed harmonic volume regions must be unique')
        if set(current_conductivities)!=set(operator.current_regions) or not set(operator.current_regions)<=set(regions):
            raise ValueError('Each current body requires its volume and physical DC conductivity')
        self.current_bindings=tuple(b for b in self.bindings if b.volume.region_id in operator.current_regions)
        self.dielectric_bindings=tuple(b for b in self.bindings if b.volume.region_id not in operator.current_regions)
        if not self.current_bindings or (not tensor and not self.dielectric_bindings):
            raise ValueError('Mixed harmonic closure requires current and dielectric volumes')
        for binding in self.bindings:
            binding.assert_current()
            if binding.surface_system is not system:
                raise ValueError('Mixed harmonic volumes must bind the exact interface system')
        self.volumes=tuple(b.volume for b in self.bindings)
        self.counts=tuple(len(v.cells) for v in self.volumes)
        self.current=HarmonicCurrentClosure([b.volume for b in self.current_bindings],
            [current_conductivities[b.volume.region_id] for b in self.current_bindings],**(dc_policy or {}))
        sample=self.volumes[0].vertices
        current_indices,dielectric_indices=[],[]
        offset=0
        for binding,count in zip(self.bindings,self.counts):
            indices=torch.arange(offset,offset+count,device=sample.device)
            (current_indices if binding.volume.region_id in operator.current_regions else dielectric_indices).append(indices)
            offset+=count
        self.current_indices=torch.cat(current_indices)
        self.dielectric_indices=torch.cat(dielectric_indices) if dielectric_indices else torch.empty(0,device=sample.device,dtype=torch.int64)
        if max_response_bytes is not None and not tensor:
            raise ValueError('Explicit mixed response byte budget requires the tensor interface')
        self.maps=operator.prepare_cell_response(self.dielectric_bindings,
            **({'max_response_bytes':max_response_bytes} if max_response_bytes is not None else {}))
        self._versions=self._state()

    def _state(self):
        values=(self.current_indices,self.dielectric_indices,
                *(self.maps[k] for k in ('electric_response','surface_response','offset_response')),
                *((self.maps['conormal_response'],) if 'conormal_response' in self.maps else ()))
        return (id(self.operator),id(self.current),self.counts,tuple(id(b) for b in self.bindings),
                tuple(id(b) for b in self.current_bindings),tuple(id(b) for b in self.dielectric_bindings),
                tuple(id(v) for v in self.volumes))+tuple((id(t),t._version) for t in values)

    def assert_current(self):
        self.operator.assert_current()
        self.current.assert_current()
        for binding in self.bindings:
            binding.assert_current()
        if self._versions!=self._state():
            raise RuntimeError('Mixed harmonic closure was mutated')

    def solve(self,ac_electric,tpa_scale):
        self.assert_current()
        if ac_electric.ndim!=3 or ac_electric.shape[1:]!=(sum(self.counts),3):
            raise ValueError('Mixed DC AC fields must match complete volume cell order')
        if isinstance(tpa_scale,torch.Tensor) and tpa_scale.ndim:
            if tpa_scale.shape!=(sum(self.counts),):
                raise ValueError('Mixed DC TPA coefficients must match complete volume cell order')
            if bool(tpa_scale[self.dielectric_indices].any()):
                raise ValueError('Dielectric DC cells cannot carry a TPA current law')
            tpa=tpa_scale[self.current_indices]
        else:
            if bool(tpa_scale!=0):
                raise ValueError('Mixed DC requires per-cell TPA coefficients or scalar zero')
            tpa=tpa_scale
        return self.current.solve(ac_electric[:,self.current_indices],tpa)

    def electric(self,ac_electric,tpa_scale,*,polarization):
        states=self.solve(ac_electric,tpa_scale)
        if isinstance(self.operator,StaticTensorFloatingOperator):
            self.operator.exterior._validate(polarization,(sum(self.counts),3))
        else:
            self.operator.laplace._real(polarization,(sum(self.counts),3),'Mixed DC polarization')
        boundary={binding.volume.region_id:surface_face_potentials(block,binding,state.potential)
                  for binding,block,state in zip(self.current_bindings,self.current.blocks,states)}
        packed=torch.cat((polarization[self.dielectric_indices].flatten(),
                          *(boundary[region] for region in self.operator.current_regions)))
        dielectric=(self.maps['electric_response']@packed).reshape(-1,3)
        electric=torch.zeros_like(polarization).index_copy(0,self.current_indices,
            torch.cat([state.electric for state in states]))
        return electric.index_copy(0,self.dielectric_indices,dielectric)
