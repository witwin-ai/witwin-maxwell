"""pytest plugin (-p tests.support.conductor_census): log every layered runtime's conductor_model selection.

Writes one JSON line per runtime to $RFIC_OUT/conductor_selection.jsonl (or
$CONDUCTOR_CENSUS_LOG): test node, requested model, max t/delta at f_max and the
selected model ("raise" when auto refuses). Census of the A2 default switch; it
does not change any result.
"""
import json
import os

_node = {"id": None}


def pytest_runtest_setup(item):
    _node["id"] = item.nodeid


def pytest_configure(config):
    from witwin.maxwell.mom.layered import runtime as cs
    path = os.environ.get("CONDUCTOR_CENSUS_LOG") or os.path.join(os.environ["RFIC_OUT"], "conductor_selection.jsonl")
    original = cs.select_conductor_model

    def logged(requested, compiled, frequencies, *, sweep, gradients, planar):
        f_max = sweep.f_max if sweep is not None else max(frequencies)
        record = dict(node=_node["id"], requested=requested, f_max_Hz=f_max,
                      max_t_over_delta=cs.max_t_over_delta(compiled[3], f_max), selected="raise")
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        model, selection = original(requested, compiled, frequencies, sweep=sweep, gradients=gradients, planar=planar)
        with open(path, "a") as handle:
            handle.write(json.dumps({**record, "selected": model}) + "\n")
        return model, selection

    cs.select_conductor_model = logged
