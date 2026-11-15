"""Frozen test-only oracles.

Reference implementations that used to ship inside ``witwin.maxwell`` although no
production path ran them: the Torch Yee kernel subset of the retired CPU reference
profile, the Torch thin-sheet and copper arithmetic, and the analytic ferrite,
surface-impedance and thin-wire references. Tests compare the CUDA runtime against
them; they are never imported by the package.
"""
