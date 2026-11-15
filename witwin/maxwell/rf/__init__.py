"""Public namespace ``mw.rf``: ports, lumped elements, thin wires, path integrals, network
data and Touchstone I/O, with rational fitting in ``mw.rf.rational``."""
from ..lumped import (
    Capacitor,
    CoaxialLumpedResistor,
    Inductor,
    ParallelRLC,
    PortExcitation,
    PortSweep,
    Resistor,
    SeriesRLC,
)
from ..modal_sweep import ModalPortSweep, run_modal_port_sweep
from ..monitors import WireMonitor
from ..network import (
    EmbeddedNetworkData,
    NetworkBlock,
    NetworkData,
    NetworkPhysicalityReport,
    PortData,
    TouchstoneNetwork,
)
from ..path_integrals import (
    AxisSegment,
    CircleLoop,
    CompositeCurrentIntegral,
    CompositeVoltageIntegral,
    CustomCurrentIntegral,
    CustomVoltageIntegral,
    PolygonLoop,
    PolylinePath,
    RectangleLoop,
)
from ..ports import (
    CoaxialLumpedPort,
    PECFrame,
    TerminalPort,
    TerminalRef,
    TerminalWavePort,
    WaveModeSpec,
    WavePort,
    WireNodeRef,
    WirePortBinding,
)
from ..ports_postprocess import PortDrive, port_phasors, port_voltage, trapezoidal_impedance
from ..postprocess.low_frequency import LowFrequencySmoothingSpec, smooth_low_frequency
from ..terminal_waveport import TerminalWavePortApertureWarning
from ..thin_wire import ThinWire, WireConductor, WireData, WireEnd
from ..touchstone import TouchstoneParseError
from . import rational
from .._stability import experimental


@experimental("networks")
def embedded_networks(result) -> dict[str, EmbeddedNetworkData]:
    """The embedded-network diagnostics of ``result`` by network name."""
    result._require_absolute_spectra()
    return dict(result._embedded_networks)


@experimental("networks")
def embedded_network(result, name: str) -> EmbeddedNetworkData:
    """The ``EmbeddedNetworkData`` of embedded network ``name`` in ``result``."""
    return result._embedded_network(name)


@experimental("pcb")
def conductor_losses(result) -> dict:
    """Independent copies of the per-foil energy (J), spectral work (W) and geometry of ``result``.

    Native spectral tile work is detached diagnostic data; a separately supplied live
    graph is kept. ``energy_representation`` tells physical real-field joules from
    combined complex-bank algorithmic work, not real transient heat.
    """
    return result._conductor_losses()


@experimental("pcb")
def conductor_loss_columns(result) -> dict:
    """Port-sweep conductor losses of ``result`` keyed by excitation port, then foil name (J / W).

    Each column keeps the single-run scalar energies and [F] surface powers; independent
    excitation runs are never summed into one loss.
    """
    return result._conductor_loss_columns()


__all__ = [
    "AxisSegment",
    "Capacitor",
    "CircleLoop",
    "CoaxialLumpedPort",
    "CoaxialLumpedResistor",
    "CompositeCurrentIntegral",
    "CompositeVoltageIntegral",
    "CustomCurrentIntegral",
    "CustomVoltageIntegral",
    "EmbeddedNetworkData",
    "Inductor",
    "LowFrequencySmoothingSpec",
    "ModalPortSweep",
    "NetworkBlock",
    "NetworkData",
    "NetworkPhysicalityReport",
    "PECFrame",
    "ParallelRLC",
    "PolygonLoop",
    "PolylinePath",
    "PortData",
    "PortDrive",
    "PortExcitation",
    "PortSweep",
    "RectangleLoop",
    "Resistor",
    "SeriesRLC",
    "TerminalPort",
    "TerminalRef",
    "TerminalWavePort",
    "TerminalWavePortApertureWarning",
    "ThinWire",
    "TouchstoneNetwork",
    "TouchstoneParseError",
    "WaveModeSpec",
    "WavePort",
    "WireConductor",
    "WireData",
    "WireEnd",
    "WireMonitor",
    "WireNodeRef",
    "WirePortBinding",
    "conductor_loss_columns",
    "conductor_losses",
    "embedded_network",
    "embedded_networks",
    "port_phasors",
    "port_voltage",
    "rational",
    "run_modal_port_sweep",
    "smooth_low_frequency",
    "trapezoidal_impedance",
]
