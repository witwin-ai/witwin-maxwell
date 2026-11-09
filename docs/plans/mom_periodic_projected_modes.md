# Explicit projected Maxwell modal quantities

GSP-07C permits explicitly defined projected quantities at true spectral points,
while ordinary forced solutions must continue to reject unresolved poles. The
internal projected_modes method returns a caller-selected truncated-SVD map for
each requested unique3D Bloch order. It does not replace spectral_modes and no
solver, field recovery, spatial sum or public Result automatically selects it.

For the dimensionless Maxwell symbol A acting on x=[E,eta0 H], take A=U S V^H.
The requested policy retains s_j > rank_relative_tolerance*s_max. The returned
physical map is D_field V S_retained^-1 U^H D_source / cell_volume, with precisely
the same source and field scales as the ordinary current Green spectrum.
The removed-forcing projector is U_discard U_discard^H; homogeneous-field
projector is V_discard V_discard^H. Their Euclidean orthogonality belongs to
scaled-symbol coordinates, not physical current/field units. The projected
response is minimum norm in [E,eta0 H]. Truncation tolerance is a numerical
policy, not certification of an exact nullspace or eigenfrequency. Nothing
asserts a finite or path-independent limiting-absorption forced solution.

Trainable constitutive tensors reject because derivatives through rank changes
are undefined in this contract. External current-vector gradients remain native.
Lattice geometry remains fixed. Empty order sets have shaped empty outputs;
invalid tolerances, duplicate orders and mutated material/lattice samples reject.
This finite requested-mode diagnostic does not close3D spatial primary/contact
separation, global spectral coverage, limiting absorption, Wood anomalies,
public assembly/storage or physical qualification.

Targeted GPU1 runner /tmp/gsp-periodic-projected-v1.py passes5nodes, no skips,
errors or failures,6.187 runner seconds (XML5.02seconds),598source/test hashes
unchanged. Four new nodes include independent scalar axial resonance minimum-
norm response, projector identities and removed-forcing handling, current
adjoint gradient, off-pole reciprocal/nonreciprocal fulltensor resolvents and
input/material-gradient guards. Existing ordinary pole rejection also passes.
Source-plane fulltensor continuation76454 is independent and remains frozen.

Receipt SHA256 a90db5f043325e575958297c6800312b5cfe04a76313b67433d29c9cf2e5003f

Independent review found no P1/P2, but identified the missing rank-deficient
nonreciprocal control. Added a lossless axial gyromagnetic circular-channel
resonance with rank5: its resonant2x2 block uses the independent rank-one
identity B+ = B^H/||B||_F^2, its opposite-handed block uses the analytic2x2
inverse, and longitudinal channels are diagonal. This confirms both the
physical map and distinct left/right projectors for a nonnormal symbol.
Only this new node was run in /tmp/gsp-periodic-projected-gyro-v1:1pass,
zero skips/errors/failures,5.318runnerseconds. All598source hashes unchanged;
only the appended test file differs from initial v1, production identical.
Six distinct current nodes are covered across the two preserved runs.
Gyromagnetic receipt SHA256 903d04234126feb1b3d7c6f9fd8c3865d04d70ac50fab6f92f2ae2b2653607a4
