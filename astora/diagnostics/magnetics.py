from dataclasses import dataclass
from numpy import ndarray, sqrt, clip
from scipy.special import ellipk, ellipe
from abc import ABC, abstractmethod


@dataclass
class FluxloopData:
    R: ndarray
    z: ndarray
    name: ndarray
    measurements: ndarray
    errors: ndarray
    computed: ndarray


@dataclass
class FieldSensorData(FluxloopData):
    poloidal_angle: ndarray


@dataclass
class CurrentData:
    measurements: ndarray
    errors: ndarray
    computed: ndarray


class BaseFieldCoil(ABC):
    @abstractmethod
    def psi_prediction(self, current: float, R: ndarray, z: ndarray) -> ndarray:
        pass

    def Br_prediction(self, current: float, R: ndarray, z: ndarray, eps=1e-4) -> ndarray:
        f1 = self.psi_prediction(current, R, z - eps)
        f2 = self.psi_prediction(current, R, z + eps)
        return (f1 - f2) / (2 * eps * R)

    def Bz_prediction(self, current: float, R: ndarray, z: ndarray, eps=1e-4) -> ndarray:
        f1 = self.psi_prediction(current, R - eps, z)
        f2 = self.psi_prediction(current, R + eps, z)
        return (f2 - f1) / (2 * eps * R)


class PoloidalFieldCoil(BaseFieldCoil):
    def __init__(self, R_filaments: ndarray, z_filaments: ndarray):
        assert R_filaments.size == z_filaments.size
        assert R_filaments.ndim == 1 and z_filaments.ndim == 1
        assert (R_filaments > 0.).all()
        self.R_fil = R_filaments
        self.z_fil = z_filaments
        self.n_filaments = R_filaments.size

    def psi_prediction(self, current: float, R: ndarray, z: ndarray) -> ndarray:
        return psi_from_Jtor(
            R0=self.R_fil[None, :],
            z0=self.z_fil[None, :],
            R=R[:, None],
            z=z[:, None],
        ).sum(axis=1) * (current / self.n_filaments)


class CoilCircuit(BaseFieldCoil):
    def __init__(self, coils: list[PoloidalFieldCoil], multipliers: list[float]):
        self.coils = coils
        self.multipliers = multipliers

    def psi_prediction(self, current: float, R: ndarray, z: ndarray) -> ndarray:
        return sum(
            coil.psi_prediction(current * mult, R, z)
            for coil, mult in zip(self.coils, self.multipliers)
        )


class CoilSet:
    def __init__(self, coils: list[PoloidalFieldCoil]):
        self.coils = coils
        self.n_coils = len(coils)

    def psi(self, currents: ndarray, R: ndarray, z: ndarray) -> ndarray:
        return sum(
            coil.psi_prediction(current, R, z)
            for coil, current in zip(self.coils, currents)
        )

    def Bz(self, currents: ndarray, R: ndarray, z: ndarray) -> ndarray:
        return sum(
            coil.Bz_prediction(current, R, z)
            for coil, current in zip(self.coils, currents)
        )

    def Br(self, currents: ndarray, R: ndarray, z: ndarray) -> ndarray:
        return sum(
            coil.Br_prediction(current, R, z)
            for coil, current in zip(self.coils, currents)
        )


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
