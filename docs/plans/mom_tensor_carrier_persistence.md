# Exact tensor incident carrier restoration

## Scope

`TensorIncidentCarrier.from_payload(payload, device=..., medium=...)` restores
the existing `analytic_tensor_plane_modes_v1` state on CUDA. It requires the
exact field set, version and phasor convention; retains complex64/complex128;
checks finite arrays and dimensions; and reconstructs owned mode arrays and
origin. A supplied exterior must match the saved frequency, reciprocity,
precision, device and constitutive tensors exactly. Otherwise a new validated
TensorMedium is created. The carrier constructor rechecks both Maxwell curl
equations and nonzero fields. Mutating the input payload cannot mutate the
restored carrier. This restores fixed state, not an autograd graph.

This is a prerequisite for public tensor-background source and snapshot
integration. It does not introduce a second public solver entrypoint. Public
Scene background compilation, general source-mode selection, complete periodic
system/result persistence, Bench integration and physical qualification remain
open. Lattice/Bloch validation remains explicit at the consuming operator.

## Targeted evidence

All numerical checks ran on physical GPU 1 with the maxwell environment.
The original base HEAD was `bec7a76f20c140829c3788d269ddb45f8ad3679d`.
Both runs bind 601 Python source/test files before and after; all unchanged.

- `/tmp/gsp-tensor-carrier-restore-v1`: 3 passed, 4 failed in 9.650 seconds
  runner elapsed. The positive test fixture confused dimensionless `q/k0`
  with physical Bloch wavevector. The correct sector guard rejected it.
  Receipt SHA256:
  `40117d142d431c48ca307ed3a4301b539c1bb1b9983fc036e844da456dee1c0f`.
- `/tmp/gsp-tensor-carrier-restore-v2`: after deriving physical Bloch values
  from the carrier, 7 passed, zero failures/errors/skips in 9.874 seconds
  runner elapsed (pytest 8.65 seconds). Receipt SHA256:
  `12ef386deef20a899de088509bb7185da6515ea16e07be435a5da0acc52e031a`.

Coverage includes actual two-mode anisotropic carriers in both precisions and
reciprocity classes; JSON identity and exact point-field replay; newly owned and
rebound media; malformed versions/shapes/nonfinite values; forged fields and
active constitutive responses; wrong exterior and Bloch sector; and the
existing scalar-limit native amplitude-gradient test. Independent audit
identified the initial fixture issue. Its failure evidence remains preserved.
Terminal independent review found no remaining confirmed P1/P2 and verified
the runner, XML, log and all 601 hashes against the current source. Production
implementation was unchanged between the two targeted runs.

The independently frozen periodic merge regression at bec7a76 does not include
this later stage. Its results must not be presented as coverage of this change.
