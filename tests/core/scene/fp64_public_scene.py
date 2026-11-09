"""Shared direct public Scene fixture for the bounded native FP64 path."""
import torch
import witwin.maxwell as mw


def scene(*,device='cpu',curvature=-3.,bulk_sigma=.15,boundary=None,material=None):
    def box(position,size):return mw.Box(position=torch.tensor(position,dtype=torch.float64),size=torch.tensor(size,dtype=torch.float64))
    port=mw.LumpedPort('feed',positive=(-.015,-.005,.005),negative=(-.015,-.005,-.005),voltage_path=mw.AxisPath('z'),
        current_surface=box((-.015,-.005,-.0025),(.015,.015,0.)),voltage_aperture=box((-.015,-.005,0.),(.01,.01,.01)))
    return mw.Scene(domain=mw.Domain(bounds=((- .03,.03),)*3),grid=mw.GridSpec.uniform(.005),
        boundary=boundary or mw.BoundarySpec.pml(num_layers=4),coordinates=None if curvature is None else mw.BendCoordinates(curvature=curvature,origin=(.2,-.1,.3)),
        device=device,ports=(port,),structures=(mw.Structure(box((0.,.005,0.),(.04,.02,.04)),material or mw.Material(eps_r=2.,sigma_e=bulk_sigma)),
            mw.Structure(box((0.,.005,0.),(.02,0.,.02)),mw.CopperFoil(thickness=1e-6,conductivity=1e6,frequency_range=(2e9,4e9),mode_count=16))),
        subpixel_samples=mw.SubpixelSpec(samples=(2,4,2),averaging='arithmetic'),
        monitors=(mw.ClosedSurfaceMonitor.box('radiation',size=(.05,.05,.05),sample_spacing=.007),mw.PowerLossMonitor('bulk',position=(0.,.005,0.),size=(.035,.025,.035))))


def simulation(value,*,dtype=torch.float64,**kwargs):
    options=dict(frequency=3e9,field_dtype=dtype,full_field_dft=False,cuda_graph=False,cpml_config={'memory_mode':'dense'},
        excitations=mw.PortExcitation('feed',source_time=mw.CW(3e9,phase=.37)),
        run_time=mw.TimeConfig(time_steps=96,dt=5e-12,transient_cycles=0,steady_cycles=1),
        spectral_sampler=mw.SpectralSampler(window='none',normalize_source=False,observer_dtype=torch.float64))
    options.update(kwargs)
    return mw.Simulation.fdtd(value,**options)
