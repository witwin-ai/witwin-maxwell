"""Streamed per-frequency layered networks for graded mm-wave Green tables (R4-2).

A graded vertical table (vertical_refine.py) stores every frequency at the
graded size (0.5-1.8 GB per frequency, 6 GiB build peak on a via stack), so a
multi-frequency 60-200 GHz table does not fit a 7.5 GiB budget. When the
full-set ungraded step-0 table fails its z gate, prepare_layered_network
returns a StreamedLayeredNetwork instead of grading every frequency in one
table. The solve then runs the exact single-frequency prepare and solve for each
frequency in request order (own table, grading level and wall mesh) and frees
it before the next, so

- each frequency is torch.equal to its own single-frequency call, and
- the device peak is the largest single-frequency peak.

Multi-frequency graded tables per grading level were rejected: horizontal and
vertical table bits depend on the band (k_max, truncation, radial grid and GEMM
chunking use fmin/fmax), so a shared table cannot equal per-frequency calls,
and a memory-sized split would make results depend on free device memory.

Frequency groups are maximal runs of consecutive frequencies whose wall mesh
(fixed by the grading level) is identical. Their currents share one layout.
An admitted ungraded step 0 never reaches this module (one group, unchanged).
"""

from dataclasses import dataclass

import torch

from ...errors import UnsupportedFeatureError
from ...network import NetworkData
from .runtime import CURRENT_KIND, LayeredSurfaceData

# Per-frequency tuples written by solve_prepared_network; concatenated over the stream.
PER_FREQUENCY_KEYS = (
    "factor_dtype", "refinement_steps", "backward_error", "block_gmres_steps", "operator_actions",
    "admittance_bound", "preconditioner_factor_nnz", "frequency_seconds", "near_setup_seconds",
    "action_seconds", "compression_bound_worst_case", "compression_estimate_quadrature",
    "compression_exact_check", "hmatrix")


class StreamedLayeredNetwork:
    """prepare_layered_network arguments; tables are built one frequency at a time at solve.

    metadata holds caller extras (runtime: sheets, breakpoint tolerance,
    hmatrix_exact_check) applied to each single-frequency prepared network,
    exactly as a single-frequency call applies them.
    """

    def __init__(self, arguments):
        self.arguments = arguments
        self.frequencies = tuple(float(f) for f in arguments["frequencies"])
        self.ports = tuple(arguments["ports"])
        self.device = torch.device(arguments["device"])
        self.metadata = {}


@dataclass
class FrequencyGroup:
    """Consecutive frequencies [start, stop) solved on one wall mesh."""

    start: int
    stop: int
    layout: object
    currents: torch.Tensor


def _same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(map(_same, a, b))
    return a == b


def _same_mesh(a, b):
    return (_same(a.mesh.vertices, b.mesh.vertices) and _same(a.mesh.faces, b.mesh.faces)
            and _same(a.basis.edges, b.basis.edges))


def merged_metadata(metadata, groups):
    """Shared keys at top level, per-frequency tuples concatenated, the rest per frequency."""
    first = metadata[0]
    merged = {key: value for key, value in first.items() if key not in PER_FREQUENCY_KEYS
              and all(key in other and _same(value, other[key]) for other in metadata[1:])}
    merged.update({key: sum((tuple(other[key]) for other in metadata), ()) for key in PER_FREQUENCY_KEYS
                   if key in first})
    merged.update(frequency_streaming="per_frequency_graded_tables",
                  frequency_groups=tuple((group.start, group.stop) for group in groups),
                  frequency_metadata=tuple(metadata))
    return merged


def solve_streamed_network(streamed, *, frequency_chunk_size=None):
    """Single-frequency prepare+solve per frequency; NetworkData S is their exact concatenation."""
    from .network import prepare_layered_network, solve_prepared_network
    networks, groups = [], []
    for index, frequency in enumerate(streamed.frequencies):
        prepared = prepare_layered_network(**{**streamed.arguments, "frequencies": (frequency,)})
        prepared.metadata.update(streamed.metadata)
        network, currents = solve_prepared_network(prepared, frequency_chunk_size=frequency_chunk_size)
        layout = prepared.layout
        # Release the graded table before the next frequency builds its own.
        del prepared
        if groups and _same_mesh(groups[-1][0].layout, layout):
            groups[-1].append(FrequencyGroup(index, index + 1, groups[-1][0].layout, currents))
        else:
            groups.append([FrequencyGroup(index, index + 1, layout, currents)])
        networks.append(network)
    groups = tuple(FrequencyGroup(group[0].start, group[-1].stop, group[0].layout,
                                  torch.cat([part.currents for part in group]))
                   for group in groups)
    network = NetworkData(frequencies=torch.cat([n.frequencies for n in networks]),
                          s=torch.cat([n.s for n in networks]), z0=torch.cat([n.z0 for n in networks]),
                          port_names=networks[0].port_names,
                          metadata=merged_metadata([dict(n.metadata) for n in networks], groups))
    return network, groups


def resolve_streamed(streamed, frequencies):
    """(network, groups) of further frequencies, each solved exactly as a single-frequency call."""
    further = StreamedLayeredNetwork({**streamed.arguments, "frequencies": tuple(frequencies)})
    further.metadata.update(streamed.metadata)
    return solve_streamed_network(further)


def merge_streamed_sweep(networks, batches, joined):
    """Adaptive-sweep merge of streamed batches (R6-3): frequency-sorted samples, regrouped by wall mesh.

    ``joined`` is the sweep's frequency-sorted NetworkData of all batches; the
    per-frequency metadata and currents follow the same stable order.
    """
    samples = [(float(network.frequencies[k]), network.metadata["frequency_metadata"][k], group.layout,
                group.currents[k - group.start])
               for network, groups in zip(networks, batches) for group in groups for k in range(group.start, group.stop)]
    samples = sorted(samples, key=lambda sample: sample[0])
    runs = []
    for index, (_, _, layout, currents) in enumerate(samples):
        if runs and _same_mesh(runs[-1][0].layout, layout):
            runs[-1].append(FrequencyGroup(index, index + 1, runs[-1][0].layout, currents[None]))
        else:
            runs.append([FrequencyGroup(index, index + 1, layout, currents[None])])
    groups = tuple(FrequencyGroup(run[0].start, run[-1].stop, run[0].layout, torch.cat([part.currents for part in run]))
                   for run in runs)
    network = NetworkData(frequencies=joined.frequencies, s=joined.s, z0=joined.z0, port_names=joined.port_names,
                          metadata=merged_metadata([sample[1] for sample in samples], groups))
    return network, groups


class GroupedLayeredSurfaceData(LayeredSurfaceData):
    """Streamed currents: groups[i] is a LayeredSurfaceData on that group's wall mesh.

    frequency_groups[i] = (start, stop) indexes the network frequencies of group i.
    """

    def __init__(self, groups, frequencies, port_names):
        self.groups = tuple(LayeredSurfaceData(group.currents, group.layout, frequencies[group.start:group.stop],
                                               port_names) for group in groups)
        self.frequency_groups = tuple((group.start, group.stop) for group in groups)
        self.frequencies = frequencies
        self.port_names = port_names
        self.current_kind = CURRENT_KIND

    @property
    def currents(self):
        raise UnsupportedFeatureError("Streamed mm-wave layered results have one wall mesh per frequency group; "
                                      "read surface_currents.groups[i].currents (frequency_groups[i] = (start, stop))",
                                      feature="guard:5e7a41c2", method="mom.layered")

    @property
    def layout(self):
        raise UnsupportedFeatureError("Streamed mm-wave layered results have one wall mesh per frequency group; "
                                      "read surface_currents.groups[i].layout (frequency_groups[i] = (start, stop))",
                                      feature="guard:5e7a41c2", method="mom.layered")
