from numpy import ndarray, sqrt, clip
from scipy.special import ellipk, ellipe


def psi_from_Jtor(R0: ndarray, z0: ndarray, R: ndarray, z: ndarray) -> ndarray:
    """
    Calculates the poloidal flux at (R, Z) due to a unit, toroidally symmetric current
    at (R0, Z0) using Greens function.
    """

    # Calculate k^2
    L = 0.25 * ((R + R0)**2 + (z - z0)**2)
    k2 = R * R0 / L
    # Clip to between 0 and 1 to avoid nans e.g. when coil is on grid point
    k2 = clip(k2, 1e-10, 1.0 - 1e-10)
    coeff = 2e-7  # mu_0 / (2 * pi)

    # Note definition of ellipk, ellipe in scipy is K(k^2), E(k^2)
    return coeff * sqrt(L) * ((2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2))


def Br_from_Jtor(R0: ndarray, z0: ndarray, R: ndarray, z: ndarray, eps=1e-4) -> ndarray:
    f1 = psi_from_Jtor(R0, z0, R, z - eps)
    f2 = psi_from_Jtor(R0, z0, R, z + eps)
    return (f1 - f2) / (2 * eps * R)


def Bz_from_Jtor(R0: ndarray, z0: ndarray, R: ndarray, z: ndarray, eps=1e-4) -> ndarray:
    f1 = psi_from_Jtor(R0, z0, R - eps, z)
    f2 = psi_from_Jtor(R0, z0, R + eps, z)
    return (f2 - f1) / (2 * eps * R)