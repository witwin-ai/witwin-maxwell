"""Harmonic residuals backed by complete compressed positive-frequency actions."""
import math
import torch
from .harmonic_balance import DenseHarmonicBalance
from .coupled_volume_action import CoupledVolumeAction


class ActionHarmonicBalance(DenseHarmonicBalance):
    """Reuse the native constitutive/DC/nonlinear solver with compressed AC maps.

    Positive harmonics apply G_J to J-i*n*omega*P once. DC maps remain explicit
    dense arrays unless an existing conductive closure owns DC. Newton and
    implicit root differentiation still assemble dense nonlinear Jacobians;
    this class does not claim a fully fast nonlinear solver or qualification.
    """
    def __init__(self,background,current_actions,cell_volumes,fundamental_frequency,*,
                 dc_polarization_response=None,dc_current_response=None,
                 chi2=0.,chi3=0.,tpa_scale=0.,dc_current_closure=None,chi2_tensor=None,chi3_tensor=None):
        self._validate_background(background,cell_volumes,fundamental_frequency)
        actions=tuple(current_actions);size=3*background.shape[1]
        if len(actions)!=len(background)-1 or any(type(a) is not CoupledVolumeAction for a in actions):
            raise ValueError('Action HB requires one complete electric response per positive harmonic')
        volumes=None
        for harmonic,action in enumerate(actions,1):
            action.assert_current()
            if (action.field!='E' or action.shape!=(size,size)
                    or action.device!=background.device or action.dtype!=background.dtype
                    or not math.isclose(action.inverse.operator.provider.exterior.frequency,
                        harmonic*fundamental_frequency,rel_tol=1e-12,abs_tol=0)):
                raise ValueError('Action HB field, dimensions, precision or harmonic frequency differ')
            current_volumes=tuple(s.provider.binding.volume for s in action.sources)
            if volumes is not None and (len(volumes)!=len(current_volumes)
                    or any(a is not b for a,b in zip(volumes,current_volumes))):
                raise ValueError('Action HB harmonics must share exact ordered volume meshes')
            volumes=current_volumes
            actual=torch.cat([v.volumes for v in volumes])
            if not torch.equal(actual,cell_volumes):
                raise ValueError('Action HB weights must match actual volume cells')
        if dc_current_closure is not None:
            if dc_polarization_response is not None or dc_current_response is not None:
                raise ValueError('Conductive DC closure must not also receive response matrices')
            if (len(dc_current_closure.volumes)!=len(volumes)
                    or any(a is not b for a,b in zip(dc_current_closure.volumes,volumes))):
                raise ValueError('Action HB DC closure must share exact ordered volume meshes')
        else:
            for value in (dc_polarization_response,dc_current_response):
                if (not isinstance(value,torch.Tensor) or value.shape!=(size,size)
                        or value.device!=background.device or value.dtype!=background.dtype
                        or not bool(torch.isfinite(value).all()) or bool(value.imag.any())):
                    raise ValueError('Action HB needs explicit finite real DC maps in matching complex dtype')
        self.current_actions=actions
        self.dc_polarization_response=None if dc_polarization_response is None else dc_polarization_response.clone()
        self.dc_current_response=None if dc_current_response is None else dc_current_response.clone()
        self._initialize(background,cell_volumes,fundamental_frequency,chi2=chi2,chi3=chi3,
            tpa_scale=tpa_scale,dc_current_closure=dc_current_closure,
            chi2_tensor=chi2_tensor,chi3_tensor=chi3_tensor,
            dc_maps_empty=dc_current_closure is not None)

    def _state(self):
        values=(self.background,self.cell_volumes,self.dc_polarization_response,self.dc_current_response,
                self.chi2,self.chi3,self.tpa_scale,self.chi2_tensor,self.chi3_tensor)
        return (self.frequency,id(self.dc_current_closure),tuple(map(id,self.current_actions)))+tuple(
            (id(v),v._version) if isinstance(v,torch.Tensor) else v for v in values)

    def _assert_current(self):
        super()._assert_current()
        for action in self.current_actions:action.assert_current()

    def _response(self,polarization,current):
        dc=(torch.zeros_like(polarization[0]) if self.dc_current_closure is not None else
            self.dc_polarization_response@polarization[0]+self.dc_current_response@current[0])
        values=[dc]
        for harmonic,action in enumerate(self.current_actions,1):
            values.append(action.apply(current[harmonic]-1j*(2*math.pi*self.frequency*harmonic)*polarization[harmonic]))
        return torch.stack(values)
