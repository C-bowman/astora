from numpy import ndarray, zeros, isclose
from abc import ABC, abstractmethod
from astora.diagnostics.magnetics.fields import psi_from_Jtor


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
    def __init__(self, R_filaments: ndarray, z_filaments: ndarray, weights: ndarray = None):
        assert R_filaments.size == z_filaments.size
        assert R_filaments.ndim == 1 and z_filaments.ndim == 1
        assert (R_filaments > 0.).all()
        if weights is not None:
            assert isclose(weights.sum(), 1.0)
            assert weights.ndim == 1
            assert weights.size == R_filaments.size
            self.weights = weights[None, :]
        else:
            self.weights = 1 / R_filaments.size

        self.R_fil = R_filaments
        self.z_fil = z_filaments
        self.n_filaments = R_filaments.size

    def psi_prediction(self, current: float, R: ndarray, z: ndarray) -> ndarray:
        return (self.weights * psi_from_Jtor(
            R0=self.R_fil[None, :],
            z0=self.z_fil[None, :],
            R=R[:, None],
            z=z[:, None],
        )).sum(axis=1) * current


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
    def __init__(self, coils: list[BaseFieldCoil]):
        self.coils = coils
        print(self.coils)
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

    def get_psi_matrix(self, R: ndarray, z: ndarray) -> ndarray:
        M = zeros([R.size, self.n_coils])
        for i, coil in enumerate(self.coils):
            M[:, i] = coil.psi_prediction(1.0, R, z)
        return M

    def get_Br_matrix(self, R: ndarray, z: ndarray) -> ndarray:
        M = zeros([R.size, self.n_coils])
        for i, coil in enumerate(self.coils):
            M[:, i] = coil.Br_prediction(1.0, R, z)
        return M

    def get_Bz_matrix(self, R: ndarray, z: ndarray) -> ndarray:
        M = zeros([R.size, self.n_coils])
        for i, coil in enumerate(self.coils):
            M[:, i] = coil.Bz_prediction(1.0, R, z)
        return M