"""Native support completeness for bounded physical line power diagnostics."""
import pytest
import torch
from benchmark.scenes.rf.printed_lines import printed_line_scene
from scripts.dev.fdtd.printed_line_regional_power import regional_monitors
from witwin.maxwell.compiler.power_loss import compile_power_loss_monitor
from witwin.maxwell.compiler.copper_foil import compile_copper_faces
from witwin.maxwell.scene import prepare_scene


@pytest.mark.parametrize('metal,sigma',[('pec',0.),('pec',.01),('copper',0.),('copper',.01)])
def test_regional_observations_cover_exact_native_nonzero_sigma_and_split_trace_support(metal,sigma):
    scene=printed_line_scene('stripline',dx=.00125,longitudinal_dl=.0025,metal=metal,sigma_e=sigma,
        transverse_refinement_radius=.02,transverse_padding=.02,device='cpu')
    bounds=((-0.015,.015),(-.012,.012),(-.007,.007))
    monitors,schema=regional_monitors(scene,bounds,(.8e9,1e9,1.2e9))
    prepared=prepare_scene(scene);compiled=compile_power_loss_monitor(prepared,monitors[0])
    by_name={monitor.name:monitor for monitor in monitors}
    for component,record in schema.items():
        selected=torch.zeros_like(compiled.component_masks[component])
        index=torch.tensor(record['indices'],dtype=torch.long).reshape(-1,3)
        selected[tuple(index.T)]=True
        expected=torch.zeros_like(selected)
        mask=compiled.component_masks[component];expected[mask]=compiled.conductivity[component]!=0
        assert not (expected & ~selected).any()
        for name in record['names']:assert by_name[name].fields==(component,)
        for face in compile_copper_faces(prepared,device='cpu'):
            if component!='E'+'xyz'[face.component]:continue
            indices=face.indices;inside=mask[tuple(indices.T)]
            assert selected[tuple(indices[inside].T)].all()
    if metal=='pec' and sigma==0.:assert all(not record['names'] for record in schema.values())
