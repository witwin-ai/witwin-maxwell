"""Independent MP100 spatial Maxwell reference for exact matched diagonal SPD."""
import mpmath as mp
from witwin.maxwell.constants import EPSILON_0, MU_0


def mp_matched_spd_spatial_reference(point, diagonal, frequency, *, regular_part, return_mp=False):
    with mp.workdps(100):
        omega = 2*mp.pi*mp.mpf(frequency)
        epsilon, mu = mp.mpf(EPSILON_0), mp.mpf(MU_0)
        k = omega*mp.sqrt(epsilon*mu)
        values = [mp.mpf(v) for v in diagonal]
        determinant = mp.fprod(values)
        inverse = [mp.sqrt(determinant/v) for v in values]
        y = [inverse[i]*mp.mpf(point[i]) for i in range(3)]
        radius = mp.sqrt(mp.fsum(v*v for v in y))
        n = [v/radius for v in y]
        g = mp.exp(1j*k*radius)/(4*mp.pi*radius)
        first = g*(1j*k-1/radius)
        second = g*(-k*k-2j*k/radius+2/radius**2)
        hessian = mp.matrix(3)
        for i in range(3):
            for j in range(3):
                hessian[i, j] = second*n[i]*n[j]+first/radius*((i == j)-n[i]*n[j])
                if regular_part:
                    hessian[i, j] -= (3*n[i]*n[j]-(i == j))/(4*mp.pi*radius**3)
        gradient = [first*v for v in n]
        curl = mp.matrix([[0, -gradient[2], gradient[1]],
                          [gradient[2], 0, -gradient[0]],
                          [-gradient[1], gradient[0], 0]])
        result = mp.matrix(6)
        for i in range(3):
            for j in range(3):
                scale = inverse[i]*inverse[j]
                result[i, j] = scale*(1j*omega*mu*g*(i == j)+1j/(omega*epsilon)*hessian[i, j])
                result[i+3, j+3] = scale*(1j*omega*epsilon*g*(i == j)+1j/(omega*mu)*hessian[i, j])
                result[i, j+3], result[i+3, j] = -scale*curl[i, j], scale*curl[i, j]
        if return_mp:
            return result
        return [[complex(result[i, j]) for j in range(6)] for i in range(6)]
