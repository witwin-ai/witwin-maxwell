#!/usr/bin/env bash
# Coordinator only: existing frozen-baseline gates, eager/Graph parity,
# unchanged observer checks, the revised capability guard, and two new contracts.
PY=${PY:-python}  # the maxwell environment interpreter
export CUDA_HOME=${CUDA_HOME:-$("$PY" -c 'import os,sysconfig;print(os.path.join(sysconfig.get_paths()["purelib"],"nvidia","cu13"))')}
export WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR=${WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR:-$HOME/.cache/witwin-bc-fdtd-build-v6}
env CUDA_VISIBLE_DEVICES=0 "$PY" -m pytest \
    tests/rf/lumped/test_rf_port_bit_identity.py \
    tests/rf/lumped/test_port_step_graph_bit_identity.py \
    tests/rf/lumped/test_port_step_graph.py \
    tests/monitors/in_simulation/test_existing_monitor_bit_identity.py \
    tests/monitors/in_simulation/test_in_simulation_monitors_gpu.py \
    tests/sources/tfsf/test_complex_cw_capability_boundaries.py \
    tests/rf/lumped/test_bloch_lumped.py
