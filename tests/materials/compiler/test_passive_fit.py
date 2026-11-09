"""Shared fixed-pole fitting with an exact integrator and paired residues."""

import numpy as np

from witwin.maxwell.compiler.passive_fit import fit_passive_foster


def test_fixed_integrator_equalities_and_shared_paired_residues():
    frequencies = np.geomspace(.1, 10., 41)
    s = -2j * np.pi * frequencies
    even_rates, odd_rates = np.array([1., 5.]), np.array([2., 8.])
    columns = np.stack((
        np.column_stack((1 / (s[:, None] + even_rates), np.ones(len(s)), np.zeros(len(s)))),
        np.column_stack((1 / (s[:, None] + odd_rates), np.zeros(len(s)), np.ones(len(s)))),
    ), 1)
    expected = np.array([.7, 1.2, .2, .3])
    fixed = np.column_stack((np.zeros(len(s)), 3 / s))
    target = columns @ expected + fixed
    equality = np.array([[1., 1., 0., 0.],
                         [1 / even_rates[0], 1 / even_rates[1], 1., 0.],
                         [1 / odd_rates[0], 1 / odd_rates[1], 0., 1.]])
    moments = equality @ expected
    fitted = fit_passive_foster(frequencies, target, columns, 1 / np.abs(target), fixed_terms=fixed)
    assert np.all(fitted >= 0)
    np.testing.assert_allclose(fitted, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(columns @ fitted + fixed, target, rtol=1e-12, atol=1e-12)

    # A biased measurement must not move the exact DC/tail moments.
    biased = target + .1
    unconstrained = fit_passive_foster(frequencies, biased, columns, fixed_terms=fixed)
    assert np.max(np.abs(equality @ unconstrained - moments)) > .01
    constrained = fit_passive_foster(frequencies, biased, columns, fixed_terms=fixed,
                                    equalities=(equality, moments))
    assert np.all(constrained >= 0)
    np.testing.assert_allclose(equality @ constrained, moments, rtol=1e-12, atol=1e-12)
