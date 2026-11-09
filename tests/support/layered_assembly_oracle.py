"""Adaptive CPU fp64 source-triangle integration, independent of edge formulas."""

import numpy as np
from scipy.integrate import quad_vec


def static_triangle(triangle, observer, free_vertex):
    """Signed projected fans; adaptive integration over both fan coordinates.

    At zero height only the radial Jacobian cancellation is used. No logarithm,
    solid angle, or closed-form source integral from the runtime is reused.
    """
    triangle, observer, free_vertex = map(np.asarray, (triangle, observer, free_vertex))
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    height = np.dot(observer - triangle[0], normal)
    projection = observer - height * normal
    edges = triangle - projection
    result = np.zeros(4)
    for a, b in zip(edges, np.roll(edges, -1, axis=0)):
        area2 = np.dot(np.cross(a, b), normal)
        if area2 == 0:
            continue

        def angular(v):
            ray = (1 - v) * a + v * b
            radius = np.linalg.norm(ray)

            def radial(u):
                point = projection + u * ray
                jacobian_over_r = (1 / radius if height == 0 else
                                   u / np.hypot(height, u * radius))
                return np.r_[1., point - free_vertex] * jacobian_over_r

            return quad_vec(radial, 0., 1., epsabs=2e-13, epsrel=2e-13)[0] * area2

        direction = b - a
        closest = -np.dot(a, direction) / np.dot(direction, direction)
        result += quad_vec(angular, 0., 1., epsabs=2e-13, epsrel=2e-13,
                           points=[closest] if 0 < closest < 1 else None)[0]
    return result[0], result[1:]
