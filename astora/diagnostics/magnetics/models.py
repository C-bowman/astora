from numpy import exp, sin, cos, ndarray

from .data import FluxloopData, FieldSensorData, CurrentData
from .coils import CoilSet
from astora.mesh.basis import BasisFunction
from midas.parameters import ParameterVector
from midas.models import DiagnosticModel


class FluxloopModel(DiagnosticModel):
    def __init__(
        self,
        fluxloop_data: FluxloopData,
        basis: BasisFunction,
        coil_set: CoilSet
    ):
        self.data = fluxloop_data
        self.basis = basis
        self.coils = coil_set

        self.coil_matrix = self.coils.get_psi_matrix(R=self.data.R, z=self.data.z)
        self.basis_matrix = self.basis.get_psi_matrix(R=self.data.R, z=self.data.z)

        self.parameters = [
            ParameterVector(name="ln_J", size=self.basis.n_basis),
            ParameterVector(name="coil_currents", size=self.coils.n_coils)
        ]

        self.field_requests = []

    def predictions(self, ln_J: ndarray, coil_currents: ndarray):
        basis_currents = exp(ln_J)
        return self.basis_matrix @ basis_currents + self.coil_matrix @ coil_currents

    def predictions_and_jacobians(self, ln_J: ndarray, coil_currents: ndarray):
        basis_currents = exp(ln_J)
        predictions = self.basis_matrix @ basis_currents + self.coil_matrix @ coil_currents
        jacobians = {
            "ln_J": self.basis_matrix * basis_currents[None, :],
            "coil_currents": self.coil_matrix
        }
        return predictions, jacobians


class FieldSensorModel(DiagnosticModel):
    def __init__(
        self,
        field_sensor_data: FieldSensorData,
        basis: BasisFunction,
        coil_set: CoilSet
    ):
        self.data = field_sensor_data
        self.basis = basis
        self.coils = coil_set

        M_IR = self.coils.get_Br_matrix(R=self.data.R, z=self.data.z)
        M_Iz = self.coils.get_Bz_matrix(R=self.data.R, z=self.data.z)
        M_JR = self.basis.get_Br_matrix(R=self.data.R, z=self.data.z)
        M_Jz = self.basis.get_Bz_matrix(R=self.data.R, z=self.data.z)

        sin_t = sin(self.data.poloidal_angle)
        cos_t = cos(self.data.poloidal_angle)
        self.basis_matrix = (cos_t[:, None] * M_JR + sin_t[:, None] * M_Jz) * self.data.calibration[:, None]
        self.coil_matrix = (cos_t[:, None] * M_IR + sin_t[:, None] * M_Iz) * self.data.calibration[:, None]

        self.parameters = [
            ParameterVector(name="ln_J", size=self.basis.n_basis),
            ParameterVector(name="coil_currents", size=self.coils.n_coils)
        ]

        self.field_requests = []

    def predictions(self, ln_J: ndarray, coil_currents: ndarray):
        basis_J = exp(ln_J)
        return self.basis_matrix @ basis_J + self.coil_matrix @ coil_currents

    def predictions_and_jacobians(self, ln_J: ndarray, coil_currents: ndarray):
        basis_J = exp(ln_J)
        predictions = self.basis_matrix @ basis_J + self.coil_matrix @ coil_currents
        jacobians = {
            "ln_J": self.basis_matrix * basis_J[None, :],
            "coil_currents": self.coil_matrix
        }
        return predictions, jacobians


class PlasmaCurrentModel(DiagnosticModel):
    def __init__(
        self,
        current_data: CurrentData,
        basis: BasisFunction
    ):
        self.data = current_data
        self.basis = basis

        self.parameters = [
            ParameterVector(name="ln_J", size=self.basis.n_basis),
        ]

        self.field_requests = []

    def predictions(self, ln_J: ndarray):
        return exp(ln_J).sum() * self.basis.total_current

    def predictions_and_jacobians(self, ln_J: ndarray, coil_currents: ndarray):
        basis_I = exp(ln_J) * self.basis.total_current
        predictions = basis_I.sum()
        jacobians = {
            "ln_J": basis_I.reshape(1, self.basis.n_basis),
        }
        return predictions, jacobians