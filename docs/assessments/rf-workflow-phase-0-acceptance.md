# RF Workflow Phase 0 Acceptance

Date: 2026-07-15
Status: accepted
Scope: single-device API, mathematical contracts, and discrete prototypes

## Delivered

- Public `AxisPath`, `LumpedPort`, `PortData`, and `NetworkData` contracts.
- Peak-phasor Kurokawa power-wave transforms for complex reference impedance
  with `Re(Z0) > 0`.
- Explicit `[F]` port and `[F, output, input]` network tensor shapes.
- Torch-native S/Z/Y conversion and reference-impedance renormalization.
- Sparse Yee-grid voltage-path and current-loop compilation for an axis-aligned
  `LumpedPort`.
- An independent device-resident port DFT prototype with separate electric and
  magnetic sample times.
- Scene type/name validation and internal `PreparedScene.compile_ports()`
  integration without a second public solver entrypoint.

## Exit-Gate Evidence

- Uniform-grid analytic TEM voltage/current integration error: below 1%.
- Port reversal: V and I both reverse; average power is invariant.
- Peak/RMS factor and Yee half-step phase tests: passed.
- Complex Kurokawa V/I-to-a/b round trip and power identity: passed.
- Well-conditioned S/Z/Y round trips and renormalization: passed with autograd.
- Existing public `Scene -> Simulation -> Result` API regression: passed.

Commands:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest -q -p no:cacheprovider tests/rf tests/api/public/test_public_api.py tests/core/scene/test_scene.py
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m ruff check witwin/maxwell/network.py witwin/maxwell/ports.py witwin/maxwell/compiler/ports.py witwin/maxwell/fdtd/ports.py witwin/maxwell/scene.py witwin/maxwell/__init__.py tests/rf
```

Measured result: 94 tests passed; lint passed.

## Boundary Of This Acceptance

Phase 0 does not claim FDTD port injection, an RLC load, a terminal port, an RF
wave port, or a populated `Result.port()` / `Result.network`. Those capabilities
require the later runtime phases and must not be inferred from the passing
discrete prototypes.
